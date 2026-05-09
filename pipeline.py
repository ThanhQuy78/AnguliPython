import math
import random
from pathlib import Path
from re import sub
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp

from assets import FILTER_ZERO_POINT, index_densitymaps, load_filterbank_assets, load_noise_blobs, select_and_merge_densitymap
from common import ComplexPoint, NoiseBlob, normalize_int_like_anguli, binarize_int_like_anguli, get_device
from filtering import FilteringMixin
from orientation import OrientationMixin

# ---------------------------------------------------------------------------
# Optional torch import
# ---------------------------------------------------------------------------
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class AnguliFaithfulGenerator(OrientationMixin, FilteringMixin):
    """Best-effort faithful Python port of the Anguli master/impression path.

    Faithful points:
    - init_para / set_param / seed_pos / orientmap logic
    - density-map selection and 2-or-3 merge policy
    - set_filter_area row/column expansion (not dilation)
    - firstpass / withmap / global filter pass ordering
    - integer-like normalize/binarize behavior of filter passes
    - picturize, scratches, fixed noise, random impression masking, and distortion

    GPU acceleration:
    - orientmap() is fully vectorized with NumPy (no Python loop)
    - _apply_filter_pass() dispatches to PyTorch CUDA when use_gpu=True and
      CUDA is available, otherwise falls back to the original CPU loop
    - distortion() is vectorized with NumPy (no Python loop)

    Notes:
    - filterbank decoding uses a configurable zero point (default 46), inferred from
      the original C++ optimized lookup path.
    - random_noise() follows the public Anguli source with FINE_EDGE disabled.
    - distortion() keeps the public Anguli math but uses safe NumPy clamping for
      array bounds instead of relying on legacy C++ indexing quirks.
    """

    def __init__(
        self,
        W: int = 256,
        H: int = 360,
        margin: int = 0,
        padding: int = 0,
        generation_seed: int = 42,
        distnct_f: int = 100,
        distnct_o: int = 180,
        density_dir: Optional[str] = None,
        filterbank_dir: Optional[str] = None,
        noise_blob_dir: Optional[str] = None,
        strict_assets: bool = True,
        filter_zero_point: int = FILTER_ZERO_POINT,
        use_gpu: bool = True,
    ):
        self.W = W
        self.H = H
        self.margin = margin
        self.padding = padding
        self.generation_seed = generation_seed
        self.rng = np.random.default_rng(generation_seed)
        self.py_rng = random.Random(generation_seed)

        self.pi = math.pi
        self.rad_deg_fact = 180.0 / math.pi
        self.deg_rad_fact = math.pi / 180.0

        self.distnct_f = distnct_f
        self.distnct_o = distnct_o
        self.strict_assets = strict_assets
        self.filter_zero_point = filter_zero_point

        self.density_dir = Path(density_dir) if density_dir else None
        self.filterbank_dir = Path(filterbank_dir) if filterbank_dir else None
        self.noise_blob_dir = Path(noise_blob_dir) if noise_blob_dir else None

        self.singularity_type = 1
        self.arch_fact1 = 1.0
        self.arch_fact2 = 1.0
        self.k_arch = 1.0

        self.l = [ComplexPoint(), ComplexPoint(), ComplexPoint()]
        self.d = [ComplexPoint(), ComplexPoint(), ComplexPoint()]
        self.g_cap = np.zeros((3, 3, 10), dtype=np.float32)

        shp = (self.H + self.margin, self.W + self.margin)
        self.orient = np.zeros(shp, dtype=np.float32)
        self.filt_map = np.zeros(shp, dtype=np.int32)
        self.f_print1_2Dmat = np.zeros(shp, dtype=np.int32)
        self.f_print2_2Dmat = np.zeros(shp, dtype=np.int32)
        self.temp_fp_2Dmat = np.zeros(shp, dtype=np.float32)
        self.f_den_2Dmat = np.zeros(shp, dtype=np.float32)
        self.f_den_int_2Dmat = np.zeros(shp, dtype=np.int32)
        self.orient_ind_2Dmat = np.zeros(shp, dtype=np.int32)
        self.freq_ind_2Dmat = np.zeros(shp, dtype=np.int32)
        self.f_mask_2Dmat = np.zeros(shp, dtype=np.int32)

        self.transmin = -3 * 3
        self.transmax = 3 * 3
        self.rotmin = (-self.pi / 60.0) * 3
        self.rotmax = (self.pi / 60.0) * 3
        self.max_noiseLevel = 0
        self.min_noiseLevel = 0

        self.max_filter_size = 0
        self.filterbank = None
        self.filter_sizes = None
        self.noise_blobs: List[NoiseBlob] = []
        self.density_files = index_densitymaps(self.density_dir, self.strict_assets)
        self.filterbank, self.filter_sizes, self.max_filter_size = load_filterbank_assets(
            self.filterbank_dir,
            self.strict_assets,
            self.distnct_o,
            filter_zero_point=self.filter_zero_point,
        )
        if self.noise_blob_dir is not None:
            self.noise_blobs = load_noise_blobs(self.noise_blob_dir, self.strict_assets)

        # -- GPU setup --------------------------------------------------------
        self._gpu_device = None
        self._fb_flat = None
        if use_gpu and _TORCH_AVAILABLE:
            try:
                device = get_device(use_gpu=True)
                self._gpu_device = device
                # Pre-build flat filterbank tensor on GPU:
                #   filterbank shape: (101, distnct_o, k, k)  dtype int32
                #   flatten freq×orient → n_kernels; flatten k×k → k²
                k = self.max_filter_size
                n_freq, n_orient = self.filterbank.shape[0], self.filterbank.shape[1]
                fb_reshaped = self.filterbank.reshape(n_freq * n_orient, k * k)  # (n, k²)
                self._fb_flat = torch.from_numpy(
                    fb_reshaped.astype(np.int32)
                ).to(device)  # stays on GPU for the entire session
                print(f'[AnguliFaithfulGenerator] GPU mode active — device: {device}')
            except Exception as exc:
                print(f'[AnguliFaithfulGenerator] GPU init failed ({exc}), falling back to CPU.')
                self._gpu_device = None
                self._fb_flat = None
        else:
            if use_gpu and not _TORCH_AVAILABLE:
                print('[AnguliFaithfulGenerator] torch not installed — running on CPU.')

    # ---- Density ---------------------------------------------------------
    def sel_n_merg_densitymap(self):
        self.f_den_2Dmat = select_and_merge_densitymap(
            self.density_files,
            self.py_rng,
            self.W + self.margin,
            self.H + self.margin,
        )
        self.f_den_int_2Dmat = (self.f_den_2Dmat * 100.0).astype(np.int32)
        return self.f_den_2Dmat

    # ---- Master post-processing -----------------------------------------
    def add_ridge_edge_roughness(self, img: np.ndarray, bumpiness: float = 1.2) -> np.ndarray:
        noise = self.rng.uniform(-1.0, 1.0, img.shape).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (7, 7), 3.0)
        noise = (noise - np.mean(noise)) / (np.std(noise) + 1e-5)
        noise = noise * (bumpiness * 30.0)
        
        base_blur = cv2.GaussianBlur(img, (5, 5), 0).astype(np.float32)
        
        perturbed = base_blur + noise
        out = (perturbed > 127).astype(np.uint8) * 255
        return out

    def add_sweat_pores(self, img: np.ndarray, mu: float = 20.0, sigma: float = 5.0) -> np.ndarray:
        binary_inv = (img < 127).astype(np.uint8) * 255
        skeleton = cv2.ximgproc.thinning(binary_inv)
        
        out = img.copy()
        num_labels, labels = cv2.connectedComponents(skeleton, connectivity=8)
        
        for label in range(1, num_labels):
            mask = (labels == label)
            pts_y, pts_x = np.nonzero(mask)
            if len(pts_y) < 10:
                continue
                
            pts = np.column_stack((pts_x, pts_y))
            self.rng.shuffle(pts)
            
            selected_pores = []
            for p in pts:
                if not selected_pores:
                    selected_pores.append(p)
                else:
                    dists = np.linalg.norm(np.array(selected_pores) - p, axis=1)
                    if np.min(dists) > max(10.0, self.rng.normal(mu, sigma)):
                        selected_pores.append(p)
                        
            for px, py in selected_pores:
                r = int(self.rng.choice([0, 1]))
                cv2.circle(out, (int(px), int(py)), r, 255, -1)
                    
        return out

    def add_scratches(self, img: np.ndarray, min_scratches: int = 5, max_scratches: int = 15) -> np.ndarray:
        out = img.copy()
        num_scratches = int(min_scratches + (max_scratches - min_scratches) * self.ahaq_rand())
        width = out.shape[1]
        height = out.shape[0]
        
        for _ in range(num_scratches):
            num_pts = int(self.rng.integers(3, 5))
            
            cx = self.rng.uniform(0, width)
            cy = self.rng.uniform(0, height)
            length = self.rng.uniform(20, 80)
            angle = self.rng.uniform(0, 2 * np.pi)
            
            t = np.linspace(-length/2, length/2, num_pts)
            dx = t * np.cos(angle)
            dy = t * np.sin(angle)
            
            noise = self.rng.normal(0, length * 0.1, num_pts)
            nx = -noise * np.sin(angle)
            ny = noise * np.cos(angle)
            
            pts_x = cx + dx + nx
            pts_y = cy + dy + ny
            
            try:
                tck, u = interp.splprep([pts_x, pts_y], s=0, k=min(3, num_pts-1))
                unew = np.linspace(0, 1, 100)
                out_x, out_y = interp.splev(unew, tck)
                curve_pts = np.vstack((out_x, out_y)).T.astype(np.int32)
            except Exception:
                curve_pts = np.vstack((pts_x, pts_y)).T.astype(np.int32)
            
            wide = int(self.rng.integers(1, 3))
            
            cv2.polylines(out, [curve_pts], isClosed=False, color=255, thickness=wide, lineType=cv2.LINE_AA)
            
        return out

    def paste_noise_blob(self, img: np.ndarray, x: int, y: int, blob_or_k):
        if not self.noise_blobs:
            return

        if isinstance(blob_or_k, int):
            idx = max(0, min(len(self.noise_blobs) - 1, int(blob_or_k) - 1))
            blob = self.noise_blobs[idx]
        else:
            blob = blob_or_k

        mask = blob.mask
        h, w = img.shape
        bh, bw = mask.shape
        shift_y = bh // 2
        shift_x = bw // 2

        for r in range(bh):
            rr = y + r - shift_y
            if rr < 0 or rr >= h:
                continue

            for c in range(bw):
                cc = x + c - shift_x
                if cc < 0 or cc >= w:
                    continue

                # faithful Anguli: paste where blob pixel is BLACK
                if int(mask[r, c]) == 0:
                    img[rr, cc] = 255

    def fixed_noise(self, img: np.ndarray, count: int = 600) -> np.ndarray:
        if not self.noise_blobs:
            return img.copy()

        out = img.copy()

        for _ in range(count):
            x = int(math.floor(self.ahaq_rand() * self.W))
            y = int(math.floor(self.ahaq_rand() * self.H))
            k = int(math.floor(1.0 + (int(self.ahaq_rand() * 100) % 2)))  # only 1 or 2
            self.paste_noise_blob(out, x + self.padding, y + self.padding, k)

        return out

    
    def _threshold_binary_255(self, img: np.ndarray) -> np.ndarray:
        return np.where(img > 127, 255, 0).astype(np.uint8)

    def _build_random_impression_mask(self) -> tuple[np.ndarray, np.ndarray]:
        mask = np.zeros((self.H + self.margin, self.W + self.margin), dtype=np.uint8)
        a1 = self.W * 0.45 + self.ahaq_rand() * self.W * 0.15
        a2 = (self.W + self.padding - a1)
        b1 = self.H * 0.40 + self.ahaq_rand() * self.H * 0.15
        b2 = b1 + b1 * 0.25 + b1 * self.ahaq_rand() * 0.25
        c1 = (self.H + self.padding - (b1 + b2))
        sx = (self.W / 2 - a2 + (a1 + a2) / 2) + self.padding
        sy = ((b2 + c1 / 2) + (self.H - (b1 + b2 + c1)) / 2) + self.padding

        cv2.ellipse(mask, (int(round(sx)), int(round(sy - c1 / 2))), (int(round(a1)), int(round(b2))), 0, 90, 180, 255, 1, 8)
        cv2.ellipse(mask, (int(round(sx)), int(round(sy - c1 / 2))), (int(round(a2)), int(round(b2))), 0, 0, 90, 255, 1, 8)
        cv2.ellipse(mask, (int(round(sx)), int(round(sy + c1 / 2))), (int(round(a1)), int(round(b1))), 0, 180, 270, 255, 1, 8)
        cv2.ellipse(mask, (int(round(sx)), int(round(sy + c1 / 2))), (int(round(a2)), int(round(b1))), 0, 270, 360, 255, 1, 8)
        cv2.line(mask, (int(round(sx - a1)), int(round(sy - c1 / 2))), (int(round(sx - a1)), int(round(sy + c1 / 2))), 255, 1, cv2.LINE_AA)
        cv2.line(mask, (int(round(sx + a2)), int(round(sy - c1 / 2))), (int(round(sx + a2)), int(round(sy + c1 / 2))), 255, 1, cv2.LINE_AA)

        mask_boundary = np.argwhere(mask == 255)
        flood = mask.copy()
        flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(flood, flood_mask, (self.W // 2 + self.padding, self.H // 2 + self.padding), 255)
        mask01 = (flood > 0).astype(np.uint8)
        self.f_mask_2Dmat = mask01.astype(np.int32)
        return mask01, mask_boundary

    def random_noise(self, img: np.ndarray, min_noise_level: int = 0, max_noise_level: int = 0) -> np.ndarray:
        if not self.noise_blobs:
            return img.copy()

        self.min_noiseLevel = int(min_noise_level)
        self.max_noiseLevel = int(max_noise_level)

        working = img.copy().astype(np.uint8)
        base_binary = self._threshold_binary_255(working)
        white_reference = working.copy()

        mask01, _mask_boundary = self._build_random_impression_mask()

        n_noise = int(
            self.min_noiseLevel * 4000
            + (self.min_noiseLevel - self.max_noiseLevel + 2) * math.floor(self.ahaq_rand() * 4000)
        )

        distance_max = math.sqrt((self.W / 2.0) ** 2 + (self.H / 2.0) ** 2)
        narea = 100
        _max_noise_blob = np.zeros(narea, dtype=np.int32)
        for i in range(narea):
            d = (i + 1) * (distance_max / narea)
            _max_noise_blob[i] = int(4 * 0.0003 * d * d * math.sqrt(d))

        cur_noise_blob = np.zeros(narea, dtype=np.int32)
        for _ in range(n_noise):
            x = int(math.floor(self.ahaq_rand() * self.W))
            y = int(math.floor(self.ahaq_rand() * self.H))
            k = int(math.floor(1.0 + (int(self.ahaq_rand() * 100) % 2)))  # faithful Anguli: 1 or 2

            distance = math.sqrt(
                (self.W / 2.0 - x) * (self.W / 2.0 - x)
                + (self.H / 2.0 - y) * (self.H / 2.0 - y)
            )
            area = int(math.floor(distance / (distance_max / narea)))
            area = max(0, min(narea - 1, area))

            self.paste_noise_blob(working, x + self.padding, y + self.padding, k)
            cur_noise_blob[area] += 1

        keep = (mask01 == 1) & (base_binary == 0)
        working = np.where(keep, working, 255).astype(np.uint8)
        working = cv2.GaussianBlur(working, (3, 3), 0)
        working[white_reference == 255] = 255
        return working

    def distortion(self, img: np.ndarray) -> np.ndarray:
        """Vectorized faithful port of public Anguli distortion().

        The original triple-nested Python loop is replaced by NumPy vectorized
        operations.  All arithmetic is identical; only the iteration strategy
        changes from pixel-by-pixel Python to NumPy array operations.
        """
        f_print1 = img.astype(np.uint8)
        height, width = f_print1.shape
        new_dim = max(width, height)
        aa = int(math.floor(new_dim / 2 - height / 2.0) + 1)
        cc = int(math.floor(new_dim / 2 - width / 2.0) + 1)

        kk = np.full((new_dim, new_dim), 255, dtype=np.float32)
        result_image = np.full((new_dim, new_dim), 255, dtype=np.float32)
        shapedist = np.full((new_dim, new_dim), -2.0, dtype=np.float64)

        r0 = max(0, aa - 1)
        c0 = max(0, cc - 1)
        kk[r0:r0 + height, c0:c0 + width] = f_print1.astype(np.float32)

        center_x = int(math.floor(new_dim / 2.0))
        center_y = int(math.floor(new_dim / 2.0))
        _dist_scale = new_dim / 360.0
        a1 = int(math.floor((100 / 3.0) * _dist_scale))
        a2 = a1
        b1 = int(math.floor((122 / 3.0) * _dist_scale))
        b2 = b1
        c = 0
        h = 0
        kshift = 0

        # ------------------------------------------------------------------
        # Build ellipse tables y1..y4 — kept as scalar loop (only 201 iters)
        # ------------------------------------------------------------------
        y1 = np.zeros(201, dtype=np.float64)
        y2 = np.zeros(201, dtype=np.float64)
        y3 = np.zeros(201, dtype=np.float64)
        y4 = np.zeros(201, dtype=np.float64)

        for i in range(1, 201):
            temp = 1 - ((i * i) / float(a1 * a1))
            temp = temp if temp > 0 else 0
            y1[i] = c / 2 + b1 * math.sqrt(temp)
            y3[i] = -c / 2 - b2 * math.sqrt(temp)
            j = i - 200
            temp = 1 - ((j * j) / float(a2 * a2))
            temp = temp if temp > 0 else 0
            y2[i] = c / 2 + b1 * math.sqrt(temp)
            y4[i] = -c / 2 - b2 * math.sqrt(temp)
            y1[i] += 200 + kshift
            y2[i] += 200 + kshift
            y3[i] += 200 + kshift
            y4[i] += 200 + kshift

        # ------------------------------------------------------------------
        # Fill shapedist — vectorized over the small ellipse index ranges
        # ------------------------------------------------------------------
        # Left half (j in [200-a2, 200])
        for j in range(200 - a2, 201):
            lo = int(math.floor(y4[j]))
            hi = int(math.floor(y2[j]))
            if lo > hi:
                lo, hi = hi, lo
            ls = np.arange(lo, hi + 1, dtype=np.int64)
            rr_vals = (400 - ls).astype(np.int64)
            cc2_vals = j + h
            valid = (rr_vals >= 0) & (rr_vals < new_dim) & (cc2_vals >= 0) & (cc2_vals < new_dim)
            if valid.any():
                shapedist[rr_vals[valid], cc2_vals] = 0

        # Right half (j in [1, a1])
        for j in range(1, a1 + 1):
            lo = int(math.floor(y3[j]))
            hi = int(math.floor(y1[j]))
            if lo > hi:
                lo, hi = hi, lo
            ls = np.arange(lo, hi + 1, dtype=np.int64)
            rr_vals = (400 - ls).astype(np.int64)
            cc2_vals = j + 200 + h
            valid = (rr_vals >= 0) & (rr_vals < new_dim) & (cc2_vals >= 0) & (cc2_vals < new_dim)
            if valid.any():
                shapedist[rr_vals[valid], cc2_vals] = 0

        # ------------------------------------------------------------------
        # Main distortion transform — fully vectorized with NumPy meshgrid
        # ------------------------------------------------------------------
        theta = self.rotmin + (self.rotmax - self.rotmin) * self.ahaq_rand()
        parak = 2
        trans_x = math.floor(self.transmin + (self.transmax - self.transmin) * self.ahaq_rand())
        trans_y = math.floor(self.transmin + (self.transmax - self.transmin) * self.ahaq_rand())

        # Pixel coordinate grids (1-indexed as in original: range 1..new_dim)
        jj, kk_idx = np.meshgrid(
            np.arange(1, new_dim + 1, dtype=np.float64),
            np.arange(1, new_dim + 1, dtype=np.float64),
            indexing='ij',
        )  # both (new_dim, new_dim)

        # shapedist at each pixel
        sd = shapedist[jj.astype(np.int64) - 1, kk_idx.astype(np.int64) - 1]

        # Compute temp3 (distortion weight) vectorized
        outside_mask = sd == -2
        temp1_raw = jj - center_x
        temp2_raw = kk_idx - center_y
        temp3_raw = np.where(
            outside_mask,
            np.sqrt(np.maximum(0.0,
                temp1_raw * (temp1_raw / float(a1 * a1)) +
                temp2_raw * (temp2_raw / float(b1 * b1))
            )) - 1.0,
            -2.0,
        )

        # Clamp and cos-blend
        temp3 = np.where(
            temp3_raw <= 0,
            0.0,
            np.where(
                temp3_raw <= parak,
                0.5 * (1 - np.cos((temp3_raw * self.pi) / parak)),
                1.0,
            ),
        )

        # Rotation + translation displacement
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        disp_x = (jj - center_x) * cos_t + (kk_idx - center_y) * sin_t + center_x + trans_x - jj
        disp_y = -(jj - center_x) * sin_t + (kk_idx - center_y) * cos_t + center_y + trans_y - kk_idx

        distortion_x = jj + disp_x * temp3    # (new_dim, new_dim)
        distortion_y = kk_idx + disp_y * temp3

        # Valid pixel mask (original condition: > 1 and < 400 inclusive check)
        valid = (
            (distortion_x > 1) & (distortion_y > 1) &
            (distortion_x < 400) & (distortion_y < 400)
        )

        # Source indices into kk (result indexed from 0, kk is 0-based)
        src_r = np.clip(jj.astype(np.int64) - 1, 0, new_dim - 1)
        src_c = np.clip(kk_idx.astype(np.int64) - 1, 0, new_dim - 1)
        # Destination indices in result_image
        dst_r = np.clip(distortion_x.astype(np.int64), 0, new_dim - 1)
        dst_c = np.clip(distortion_y.astype(np.int64), 0, new_dim - 1)

        # Apply scatter only for valid pixels
        valid_flat = valid.ravel()
        src_r_f = src_r.ravel()[valid_flat]
        src_c_f = src_c.ravel()[valid_flat]
        dst_r_f = dst_r.ravel()[valid_flat]
        dst_c_f = dst_c.ravel()[valid_flat]

        # Scatter: result_image[dst_r, dst_c] = kk[src_r, src_c]
        # Matches original: result_image[ind1, ind2] = kk[j-1, k-1]
        result_image[dst_r_f, dst_c_f] = kk[src_r_f, src_c_f]

        # ------------------------------------------------------------------
        # Crop back to original (height, width) — vectorized
        # ------------------------------------------------------------------
        rr_idx = np.clip(
            np.arange(height, dtype=np.int64) + aa - 1, 0, new_dim - 1
        )
        cc_idx = np.clip(
            np.arange(width, dtype=np.int64) + cc - 1, 0, new_dim - 1
        )
        cropped = np.clip(
            result_image[np.ix_(rr_idx, cc_idx)], 0, 255
        ).astype(np.uint8)
        return cropped

    def save_metadata(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            type_map = {
                1: 'Arch',
                2: 'Tented Arch',
                3: 'Right Loop',
                4: 'Left Loop',
                5: 'Double Loop',
                6: 'Whorl',
            }
            f.write(f'Type : {type_map.get(self.singularity_type, self.singularity_type)}\n')
            if self.singularity_type == 1:
                f.write(f'{self.k_arch}\n')
            if self.singularity_type in (2, 3, 4):
                f.write(f'Loop : {self.l[1].x}\t{self.l[1].y}\n')
                f.write(f'Delta : {self.d[1].x}\t{self.d[1].y}\n')
            if self.singularity_type in (5, 6):
                f.write(f'Loop : {self.l[1].x}\t{self.l[1].y}\n')
                f.write(f'Loop : {self.l[2].x}\t{self.l[2].y}\n')
                f.write(f'Delta : {self.d[1].x}\t{self.d[1].y}\n')
                f.write(f'Delta : {self.d[2].x}\t{self.d[2].y}\n')

    # ---- Debug saving ----------------------------------------------------
    def _save_u8(self, out_dir: Path, name: str, img_u8: np.ndarray):
        out_dir.mkdir(parents=True, exist_ok=True)
        if img_u8.dtype != np.uint8:
            img_u8 = img_u8.clip(0, 255).astype(np.uint8)
        cv2.imwrite(str(out_dir / name), img_u8)

    def _save_float01(self, out_dir: Path, name: str, arr: np.ndarray):
        self._save_u8(out_dir, name, (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8))

    def _save_int_scaled(self, out_dir: Path, name: str, arr: np.ndarray, max_value: int):
        img = np.clip(arr.astype(np.float32) * 255.0 / max_value, 0, 255).astype(np.uint8)
        self._save_u8(out_dir, name, img)

    def _save_orient_quiver(self, out_dir: Path, name: str, step: int = 12):
        ys, xs = np.mgrid[0:self.orient.shape[0], 0:self.orient.shape[1]]
        u = np.cos(self.orient)
        v = np.sin(self.orient)
        plt.figure(figsize=(7, 9))
        plt.quiver(
            xs[::step, ::step], ys[::step, ::step],
            u[::step, ::step], v[::step, ::step],
            headwidth=0, headlength=0, headaxislength=0,
            pivot='middle', scale=35, color='tab:blue', alpha=0.8,
        )
        if self.singularity_type in (2, 3, 4, 5, 6):
            plt.scatter(self.l[1].x, self.l[1].y, c='red', s=70, marker='o')
            plt.scatter(self.d[1].x, self.d[1].y, c='green', s=90, marker='^')
        if self.singularity_type in (5, 6):
            plt.scatter(self.l[2].x, self.l[2].y, c='orange', s=70, marker='o')
            plt.scatter(self.d[2].x, self.d[2].y, c='lime', s=90, marker='^')
        plt.xlim(0, self.W + self.margin)
        plt.ylim(self.H + self.margin, 0)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, linestyle='--', alpha=0.25)
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_dir / name), dpi=150, bbox_inches='tight')
        plt.close()

    def save_stage_snapshot(self, out_dir: Path, prefix: str, label: str):
        self._save_int_scaled(out_dir, f'{prefix}_{label}_f_print1.png', self.f_print1_2Dmat, 100)
        self._save_u8(out_dir, f'{prefix}_{label}_conv_norm.png', self.normalized_conv_preview())
        self._save_u8(out_dir, f'{prefix}_{label}_picturize_preview.png', self.picturize_preview())
        self._save_u8(out_dir, f'{prefix}_{label}_filt_map.png', (self.filt_map > 0).astype(np.uint8) * 255)

    # ---- Pipeline --------------------------------------------------------
    def run_until_global_filter_and_save(self, class_distribution: int = 0, out_dir: str = './anguli_pipeline_debug'):
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        self.choose_singularity_type(class_distribution)
        self.init_para()
        self.set_param()
        self.orientmap()

        self._save_orient_quiver(out_dir_path, '00_orient_quiver.png')
        self._save_u8(out_dir_path, '01_orient_gray.png', ((self.orient % math.pi) * 255.0 / math.pi).astype(np.uint8))

        self.seed_pos()
        self._save_u8(out_dir_path, '02_seed_filt_map.png', (self.filt_map > 0).astype(np.uint8) * 255)
        self._save_int_scaled(out_dir_path, '03_seed_f_print1.png', self.f_print1_2Dmat, 100)

        self.sel_n_merg_densitymap()
        self._save_float01(out_dir_path, '04_density_merged.png', self.f_den_2Dmat)

        self.pre_filtering()
        self._save_int_scaled(out_dir_path, '05_orient_idx.png', self.orient_ind_2Dmat, max_value=max(1, self.distnct_o - 1))
        self._save_int_scaled(out_dir_path, '06_freq_idx.png', self.freq_ind_2Dmat, max_value=100)

        self.set_filter_area()
        self._save_u8(out_dir_path, '07_filter_area_0.png', (self.filt_map > 0).astype(np.uint8) * 255)

        self.filter_image_firstpass()
        self.save_stage_snapshot(out_dir_path, '08', 'firstpass')

        self.set_filter_area()
        self._save_u8(out_dir_path, '09_filter_area_1.png', (self.filt_map > 0).astype(np.uint8) * 255)

        self.filter_image_withmap()
        self.save_stage_snapshot(out_dir_path, '10', 'withmap_pass1')

        self.set_filter_area()
        self._save_u8(out_dir_path, '11_filter_area_2.png', (self.filt_map > 0).astype(np.uint8) * 255)

        self.filter_image_withmap()
        self.save_stage_snapshot(out_dir_path, '12', 'withmap_pass2')

        self.set_filter_area()
        self._save_u8(out_dir_path, '13_filter_area_3.png', (self.filt_map > 0).astype(np.uint8) * 255)

        self.set_filter_area()
        self._save_u8(out_dir_path, '14_filter_area_extra.png', (self.filt_map > 0).astype(np.uint8) * 255)

        for step in range(1, 6):
            self.filter_image()
            self.save_stage_snapshot(out_dir_path, f'{14 + step}', f'global_pass{step}')


        return {
            'out_dir': str(out_dir_path),
            'singularity_type': int(self.singularity_type),
            'active_filter_area': int(np.count_nonzero(self.filt_map > 0)),
        }

    def generate_master(
        self,
        class_distribution: int = 0,
        save_debug: Optional[str] = None,
        out_size: Optional[tuple] = None,
    ) -> np.ndarray:
        """Generate the master fingerprint image.

        Args:
            class_distribution: Force a specific fingerprint type (0 = random).
            save_debug: If set, save intermediate images to this directory.
            out_size: Optional (W, H) to downscale the output after generation.
                      Use this for super-resolution rendering: instantiate with
                      W=512, H=720 and pass out_size=(256, 360) to get finer ridges.
                      cv2.INTER_AREA is used for clean anti-aliased downscaling.
        """
        self.choose_singularity_type(class_distribution)
        self.init_para()
        self.set_param()
        self.seed_pos()
        self.orientmap()
        self.sel_n_merg_densitymap()
        self.pre_filtering()
        self.set_filter_area()
        self.filter_image_firstpass()
        self.set_filter_area()
        for _ in range(2):
            self.filter_image_withmap()
            self.set_filter_area()
        self.set_filter_area()  # extra expansion before global passes
        for _ in range(5):
            self.filter_image()

        # faithful Anguli: no extra normalize+binarize here
        img = self.picturize()
        img = self.add_ridge_edge_roughness(img)
        img = self.add_sweat_pores(img)
        img = self.add_scratches(img)
        img = self.fixed_noise(img, count=600)

        if out_size is not None:
            img = cv2.resize(img, (out_size[0], out_size[1]), interpolation=cv2.INTER_AREA)

        if save_debug:
            out_dir = Path(save_debug)
            out_dir.mkdir(parents=True, exist_ok=True)
            self._save_u8(out_dir, 'master.png', img)
            self._save_float01(out_dir, 'density.png', self.f_den_2Dmat)
            self._save_u8(out_dir, 'seed_map.png', (self.filt_map > 0).astype(np.uint8) * 255)
            self._save_u8(out_dir, 'conv_norm.png', self.normalized_conv_preview())

        return img

    def generate_impressions(
        self,
        master_img: np.ndarray,
        out_dir: str | Path,
        n_impr: int = 4,
        min_noise_level: int = 0,
        max_noise_level: int = 0,
        save_debug: bool = True,
        out_size: Optional[tuple] = None,
    ) -> list[np.ndarray]:
        """Generate impression variants from a master fingerprint.

        Args:
            out_size: Optional (W, H) to downscale each impression after generation.
                      Should match the out_size used in generate_master().
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base_reference = master_img.copy().astype(np.uint8)
        base_binary = self._threshold_binary_255(base_reference)
        impressions: list[np.ndarray] = []
        self._save_u8(out_dir, '00_master_reference.png', base_reference)
        self._save_u8(out_dir, '01_master_binary.png', base_binary)

        for idx in range(n_impr):
            work = base_binary.copy()
            masked = self.random_noise(work, min_noise_level=min_noise_level, max_noise_level=max_noise_level)
            distorted = self.distortion(masked)
            if out_size is not None:
                distorted = cv2.resize(distorted, (out_size[0], out_size[1]), interpolation=cv2.INTER_AREA)
            impressions.append(distorted)
            self._save_u8(out_dir, f'impression_{idx + 1}.png', distorted)
            if save_debug:
                self._save_u8(out_dir, f'debug_{idx + 1:02d}_masked.png', masked)
                self._save_u8(out_dir, f'debug_{idx + 1:02d}_mask.png', (self.f_mask_2Dmat > 0).astype(np.uint8) * 255)
        return impressions


AnguliLikeGenerator = AnguliFaithfulGenerator
