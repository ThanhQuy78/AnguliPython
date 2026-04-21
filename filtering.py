import math

import numpy as np

from common import binarize_int_like_anguli, normalize_0_100_float, normalize_int_like_anguli


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

    def _apply_filter_pass(self, only_map: bool = False, first_pass: bool = False):
        Hm = self.H + self.margin
        Wm = self.W + self.margin

        if first_pass:
            self.f_print1_2Dmat = binarize_int_like_anguli(self.f_print1_2Dmat, 55)
            self.f_print1_2Dmat = normalize_int_like_anguli(self.f_print1_2Dmat)

        dst = self.f_print2_2Dmat.copy()

        # SỬA LỚN: Cho vòng lặp chạy tràn viền (từ 0 đến Hm, Wm) 
        # Thuật toán sẽ tự động vẽ vân tay bo góc đẹp mắt mà không cần copy mảng
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

                        # Thuật toán vốn đã rất an toàn, tự động clamp(giới hạn) khi đụng viền
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

        # Đã xóa toàn bộ logic "Bù viền" (Slicing) ở đây vì không còn cần thiết nữa

        self.f_print1_2Dmat = normalize_int_like_anguli(self.f_print1_2Dmat)
        self.f_print1_2Dmat = binarize_int_like_anguli(self.f_print1_2Dmat, 45)

        if not first_pass and not only_map:
            self.f_print1_2Dmat = normalize_int_like_anguli(self.f_print1_2Dmat)

        return self.f_print1_2Dmat

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
