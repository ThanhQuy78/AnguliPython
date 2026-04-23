import math

import numpy as np

from common import (
    binarize_int_like_anguli,
    normalize_0_100_float,
    normalize_int_like_anguli,
)

# ---------------------------------------------------------------------------
# Optional torch import — GPU path active only when torch + CUDA are available
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class FilteringMixin:
    def pre_filtering(self):
        self.orient_ind_2Dmat = ((self.orient * 180.0 / math.pi).astype(np.int32)) % self.distnct_o
        self.freq_ind_2Dmat = 100 - ((self.f_den_2Dmat * 100.0).astype(np.int32) % self.distnct_f)
        self.freq_ind_2Dmat = np.clip(self.freq_ind_2Dmat, 15, 75)
        return self.orient_ind_2Dmat, self.freq_ind_2Dmat

    def set_filter_area(self):
        Hm = self.H + self.margin
        Wm = self.W + self.margin
        half = int(math.floor(self.max_filter_size / 2.0))

        # faithful Anguli: modify filt_map in-place, row pass first
        for i in range(Hm):
            flag_in = 0
            j = 0
            while j < Wm:
                if self.filt_map[i, j] == 1 and flag_in == 0:
                    for k in range(j - half, j):
                        if k > 0:
                            self.filt_map[i, k] = 1
                    flag_in = 1

                if self.filt_map[i, j] == 0 and flag_in == 1:
                    for k in range(j, j + half + 1):
                        if k < Wm:
                            self.filt_map[i, k] = 1
                    j = j + half + 1
                    flag_in = 0
                    continue

                j += 1

        # column pass
        for j in range(Wm):
            flag_in = 0
            i = 0
            while i < Hm:
                if self.filt_map[i, j] == 1 and flag_in == 0:
                    for k in range(i - half, i):
                        if k > 0:
                            self.filt_map[k, j] = 1
                    flag_in = 1

                if self.filt_map[i, j] == 0 and flag_in == 1:
                    for k in range(i, i + half + 1):
                        if k < Hm:
                            self.filt_map[k, j] = 1
                    i = i + half + 1
                    flag_in = 0
                    continue

                i += 1

        return self.filt_map

    # -----------------------------------------------------------------------
    # CPU reference path (original, untouched)
    # -----------------------------------------------------------------------
    def _apply_filter_pass_cpu(self, only_map: bool = False, first_pass: bool = False):
        Hm = self.H + self.margin
        Wm = self.W + self.margin

        if first_pass:
            self.f_print1_2Dmat = binarize_int_like_anguli(self.f_print1_2Dmat, 55)
            self.f_print1_2Dmat = normalize_int_like_anguli(self.f_print1_2Dmat)

        dst = self.f_print2_2Dmat.copy()

        for i in range(Hm):
            for j in range(Wm):
                if only_map and self.filt_map[i, j] != 1:
                    continue

                fi = int(np.clip(self.freq_ind_2Dmat[i, j], 0, 100))
                oi = int(np.clip(self.orient_ind_2Dmat[i, j], 0, self.distnct_o - 1))

                filter_size = int(self.filter_sizes[fi, oi])
                filt_size_half = int(math.floor(filter_size / 2.0))

                ker_full = self.filterbank[fi, oi]
                start = (self.max_filter_size - filter_size) // 2
                ker = ker_full[start:start + filter_size, start:start + filter_size]

                tmp1 = 0
                for r in range(filter_size):
                    for c in range(filter_size):
                        ind1 = i + r - filt_size_half
                        ind2 = j + c - filt_size_half

                        if ind1 < 0:
                            ind1 = 0
                        elif ind1 >= Hm:
                            ind1 = Hm - 1

                        if ind2 < 0:
                            ind2 = 0
                        elif ind2 >= Wm:
                            ind2 = Wm - 1

                        tmp1 += int(self.f_print1_2Dmat[ind1, ind2]) * int(ker[r, c])

                dst[i, j] = tmp1

        self.f_print2_2Dmat = dst
        self.f_print1_2Dmat = self.f_print2_2Dmat.copy()

        self.f_print1_2Dmat = normalize_int_like_anguli(self.f_print1_2Dmat)
        self.f_print1_2Dmat = binarize_int_like_anguli(self.f_print1_2Dmat, 45)

        if not first_pass and not only_map:
            self.f_print1_2Dmat = normalize_int_like_anguli(self.f_print1_2Dmat)

        return self.f_print1_2Dmat

    # -----------------------------------------------------------------------
    # GPU path — PyTorch unfold + gather + einsum
    # -----------------------------------------------------------------------
    def _apply_filter_pass_gpu(self, only_map: bool = False, first_pass: bool = False):
        """GPU-accelerated spatially-varying convolution via PyTorch.

        Strategy
        --------
        Each pixel (i, j) needs a different kernel chosen by
        (freq_ind[i,j], orient_ind[i,j]).  We handle this with:

        1. Unfold the padded input into overlapping patches → (Hm, Wm, k*k)
        2. Flatten the filterbank to (n_freq * n_orient, k*k)
        3. Build a per-pixel linear index into the flat filterbank
        4. Gather the correct kernel per pixel → (Hm, Wm, k*k)
        5. Dot product (einsum) → (Hm, Wm)  [one multiply-accumulate per patch]

        The filterbank is already 0-padded to max_filter_size × max_filter_size,
        so inactive kernel entries contribute zero — numerically identical to the
        cropped sub-kernel used in the CPU path.
        """
        device = self._gpu_device
        k = self.max_filter_size
        Hm = self.H + self.margin
        Wm = self.W + self.margin

        # -- Pre-filtering binarize / normalize (on CPU then upload) ----------
        fp1 = self.f_print1_2Dmat
        if first_pass:
            fp1 = binarize_int_like_anguli(fp1, 55)
            fp1 = normalize_int_like_anguli(fp1)

        # -- Upload source image to GPU ----------------------------------------
        src = torch.from_numpy(fp1.astype(np.float32)).to(device)   # (Hm, Wm)

        # -- Replicate-pad so border pixels get clamped neighbours (matches CPU) -
        half = k // 2
        src_pad = F.pad(
            src.unsqueeze(0).unsqueeze(0),          # (1, 1, Hm, Wm)
            (half, half, half, half),
            mode='replicate',
        )  # (1, 1, Hm+2*half, Wm+2*half)

        # -- Unfold into patches -----------------------------------------------
        patches = src_pad.unfold(2, k, 1).unfold(3, k, 1)
        # shape: (1, 1, Hm, Wm, k, k)
        patches = patches.squeeze(0).squeeze(0)           # (Hm, Wm, k, k)
        patches_flat = patches.reshape(Hm, Wm, k * k)    # (Hm, Wm, k²)

        # -- Build per-pixel kernel index (linear over freq × orient axis) -----
        fi = torch.from_numpy(
            np.clip(self.freq_ind_2Dmat, 0, 100).astype(np.int64)
        ).to(device)   # (Hm, Wm)
        oi = torch.from_numpy(
            np.clip(self.orient_ind_2Dmat, 0, self.distnct_o - 1).astype(np.int64)
        ).to(device)   # (Hm, Wm)

        lin_idx = (fi * self.distnct_o + oi)  # (Hm, Wm)  linear index into flat FB

        # -- Gather the per-pixel kernel from the pre-built flat filterbank ----
        # _fb_flat shape: (n_freq * n_orient, k²)  — built once in pipeline __init__
        fb_flat = self._fb_flat   # already on device, int32

        # Expand lin_idx to (Hm, Wm, k²) so we can gather along dim 0
        idx_exp = lin_idx.unsqueeze(-1).expand(Hm, Wm, k * k)  # (Hm, Wm, k²)
        # fb_flat needs to be navigated as (n, k²) → index per (i,j)
        kernels = fb_flat[lin_idx.reshape(-1)].reshape(Hm, Wm, k * k).float()
        # shape: (Hm, Wm, k²)

        # -- Spatially-varying dot product ------------------------------------
        dst = (patches_flat * kernels).sum(dim=-1)  # (Hm, Wm)

        # -- Apply filt_map mask (only_map=True means skip pixels outside map) -
        if only_map:
            mask = torch.from_numpy(self.filt_map.astype(np.bool_)).to(device)
            # Pixels outside the map keep their current f_print2 value
            orig_dst = torch.from_numpy(self.f_print2_2Dmat.astype(np.float32)).to(device)
            dst = torch.where(mask, dst, orig_dst)

        # -- Download result and update state ----------------------------------
        dst_np = dst.cpu().numpy().astype(np.int32)
        self.f_print2_2Dmat = dst_np
        self.f_print1_2Dmat = dst_np.copy()

        # -- Post-pass normalize + binarize (mirrors CPU path) ----------------
        self.f_print1_2Dmat = normalize_int_like_anguli(self.f_print1_2Dmat)
        self.f_print1_2Dmat = binarize_int_like_anguli(self.f_print1_2Dmat, 45)

        if not first_pass and not only_map:
            self.f_print1_2Dmat = normalize_int_like_anguli(self.f_print1_2Dmat)

        return self.f_print1_2Dmat

    # -----------------------------------------------------------------------
    # Dispatch wrapper — chooses GPU or CPU at runtime
    # -----------------------------------------------------------------------
    def _apply_filter_pass(self, only_map: bool = False, first_pass: bool = False):
        use_gpu = (
            _TORCH_AVAILABLE
            and getattr(self, '_gpu_device', None) is not None
            and getattr(self, '_fb_flat', None) is not None
        )
        if use_gpu:
            return self._apply_filter_pass_gpu(only_map=only_map, first_pass=first_pass)
        return self._apply_filter_pass_cpu(only_map=only_map, first_pass=first_pass)

    # -----------------------------------------------------------------------
    # Public pass methods (unchanged interface)
    # -----------------------------------------------------------------------
    def filter_image_firstpass(self):
        return self._apply_filter_pass(only_map=True, first_pass=True)

    def filter_image_withmap(self):
        return self._apply_filter_pass(only_map=True, first_pass=False)

    def filter_image(self):
        return self._apply_filter_pass(only_map=False, first_pass=False)

    def picturize(self) -> np.ndarray:
        self.temp_fp_2Dmat = 1.0 - self.f_print1_2Dmat.astype(np.float32) / 100.0
        return np.clip(self.temp_fp_2Dmat * 255.0, 0, 255).astype(np.uint8)

    def picturize_preview(self) -> np.ndarray:
        return self.picturize()

    def normalized_conv_preview(self) -> np.ndarray:
        return np.clip(normalize_0_100_float(self.f_print2_2Dmat) * 255.0 / 100.0, 0, 255).astype(np.uint8)
