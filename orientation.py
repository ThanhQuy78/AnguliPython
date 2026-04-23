import math

import numpy as np

from common import ComplexPoint, arg_complex


class OrientationMixin:
    def ahaq_rand(self) -> float:
        return float(self.rng.random())

    def choose_singularity_type(self, class_distribution: int = 0) -> int:
        if class_distribution != 0:
            self.singularity_type = class_distribution
            return self.singularity_type
        rand_num = int(self.ahaq_rand() * 1000)
        if rand_num <= 37:
            self.singularity_type = 1
        elif rand_num <= 66:
            self.singularity_type = 2
        elif rand_num <= 383:
            self.singularity_type = 3
        elif rand_num <= 721:
            self.singularity_type = 4
        elif rand_num <= 860:
            self.singularity_type = 5
        else:
            self.singularity_type = 6
        return self.singularity_type

    def init_para(self):
        W, H = self.W, self.H
        crx1 = math.floor(W * 0.4)
        crx2 = math.floor(W - W * 0.4)
        cry1 = math.floor(H * 0.35)
        cry2 = math.floor(H - H * 0.45)
        cwx1 = math.floor(W * 0.4)
        cwx2 = math.floor(W - W * 0.4)
        cwy1 = math.floor(H * 0.40)
        cwy2 = math.floor(H * 0.575)
        cw1x1 = math.floor(W * 0.4)
        cw1x2 = math.floor(W * 0.5)
        cw1y1 = math.floor(H * 0.325)
        cw1y2 = math.floor(H * 0.45)
        cw2x1 = math.floor(W * 0.5)
        cw2x2 = math.floor(W * 0.6)
        cw2y1 = math.floor(H * 0.475)
        cw2y2 = math.floor(H * 0.6)
        dw1x1 = math.floor(W * 0.05)
        dw1x2 = math.floor(W - W * 0.75)
        dw1y1 = math.floor(H * 0.625)
        dw1y2 = math.floor(H * 0.875)
        dw2x1 = math.floor(W * 0.75)
        dw2x2 = math.floor(W - W * 0.05)
        dw2y1 = math.floor(H * 0.65)
        dw2y2 = math.floor(H * 0.85)
        d0y1 = math.floor(H * 0.6)
        d0y2 = math.floor(H - H * 0.3)
        d1x1 = math.floor(W * 0.12)
        d1x2 = math.floor(W - W * 0.675)
        d1y1 = math.floor(H * 0.625)
        d1y2 = math.floor(H - H * 0.185)
        d2x1 = math.floor(W * 0.675)
        d2x2 = math.floor(W - W * 0.12)
        d2y1 = math.floor(H * 0.625)
        d2y2 = math.floor(H - H * 0.185)

        st = self.singularity_type
        if st == 1:
            self.arch_fact1 = 0.8 + 0.4 * self.ahaq_rand()
            self.arch_fact2 = 0.6 + 0.8 * self.ahaq_rand()
            self.k_arch = 1.2 + self.ahaq_rand() * 1.5
        elif st == 2:
            self.l[1].x = math.floor(crx1 + self.ahaq_rand() * (crx2 - crx1))
            self.l[1].y = math.floor(cry1 + self.ahaq_rand() * (cry2 - cry1))
            self.d[1].x = self.l[1].x
            self.d[1].y = math.floor(d0y1 + self.ahaq_rand() * (d0y2 - d0y1))
        elif st == 3:
            self.l[1].x = math.floor(crx1 + self.ahaq_rand() * (crx2 - crx1))
            self.l[1].y = math.floor(cry1 + self.ahaq_rand() * (cry2 - cry1))
            self.d[1].x = math.floor(d1x1 + self.ahaq_rand() * (d1x2 - d1x1))
            self.d[1].y = math.floor(d1y1 + self.ahaq_rand() * (d1y2 - d1y1))
        elif st == 4:
            self.l[1].x = math.floor(crx1 + self.ahaq_rand() * (crx2 - crx1))
            self.l[1].y = math.floor(cry1 + self.ahaq_rand() * (cry2 - cry1))
            self.d[1].x = math.floor(d2x1 + self.ahaq_rand() * (d2x2 - d2x1))
            self.d[1].y = math.floor(d2y1 + self.ahaq_rand() * (d2y2 - d2y1))
        elif st == 5:
            self.l[1].x = math.floor(cw1x1 + self.ahaq_rand() * (cw1x2 - cw1x1))
            self.l[1].y = math.floor(cw1y1 + self.ahaq_rand() * (cw1y2 - cw1y1))
            self.l[2].x = math.floor(cw2x1 + self.ahaq_rand() * (cw2x2 - cw2x1))
            self.l[2].y = math.floor(cw2y1 + self.ahaq_rand() * (cw2y2 - cw2y1))
            self.d[1].x = math.floor(dw1x1 + self.ahaq_rand() * (dw1x2 - dw1x1))
            self.d[1].y = math.floor(dw1y1 + self.ahaq_rand() * (dw1y2 - dw1y1))
            self.d[2].x = math.floor(dw2x1 + self.ahaq_rand() * (dw2x2 - dw2x1))
            self.d[2].y = math.floor(dw2y1 + self.ahaq_rand() * (dw2y2 - dw2y1))
        elif st == 6:
            self.l[1].x = math.floor(cwx1 + self.ahaq_rand() * (cwx2 - cwx1))
            self.l[2].x = self.l[1].x
            t1 = math.floor(cwy1 + self.ahaq_rand() * (cwy2 - cwy1))
            t2 = math.floor(cwy1 + self.ahaq_rand() * (cwy2 - cwy1))
            self.l[1].y, self.l[2].y = (t1, t2) if t1 < t2 else (t2, t1)
            self.d[1].x = math.floor(d1x1 + self.ahaq_rand() * (d1x2 - d1x1))
            self.d[2].x = math.floor(d2x1 + self.ahaq_rand() * (d2x2 - d2x1))
            t1 = math.floor(d1y1 + self.ahaq_rand() * (d1y2 - d1y1))
            t2 = math.floor(d2y1 + self.ahaq_rand() * (d2y2 - d2y1))
            self.d[1].y, self.d[2].y = (t1, t2) if t1 < t2 else (t2, t1)

        self.filt_map.fill(0)
        self.f_print1_2Dmat.fill(0)
        self.f_print2_2Dmat.fill(100)

    def set_param(self):
        st = self.singularity_type
        if st == 2:
            f1 = 2.0 / 3.0
            u = -90 * math.pi / 180.0 + self.ahaq_rand() * (45 * math.pi / 180.0)
            v = 45 * math.pi / 180.0 + self.ahaq_rand() * (45 * math.pi / 180.0)
            self._fill_gcap_pair(1, u, v, f1, f1)
            self._fill_gcap_delta_neutral(1)
        elif st == 3:
            f1 = 2.0 / 3.0
            u = -90 * math.pi / 180.0 + self.ahaq_rand() * (45 * math.pi / 180.0)
            v = 60 * math.pi / 180.0 + self.ahaq_rand() * (45 * math.pi / 180.0)
            self._fill_gcap_pair(1, u, v, f1, f1)
            self._fill_gcap_delta_neutral(1)
        elif st == 4:
            f1 = 2.0 / 3.0
            u = -120 * math.pi / 180.0 + self.ahaq_rand() * (45 * math.pi / 180.0)
            v = 50 * math.pi / 180.0 + self.ahaq_rand() * (45 * math.pi / 180.0)
            self._fill_gcap_pair(1, u, v, f1, f1)
            self._fill_gcap_delta_neutral(1)
        elif st in (5, 6):
            f1 = 2.0 / 3.0
            f2 = 2.0 / 3.0
            u = -60 * math.pi / 180.0 + self.ahaq_rand() * (15 * math.pi / 180.0)
            v = 40 * math.pi / 180.0 + self.ahaq_rand() * (15 * math.pi / 180.0)
            self._fill_gcap_pair(1, u, v, f1, f2)
            self._fill_gcap_delta_neutral(1)
            u = 10 * math.pi / 180.0 + self.ahaq_rand() * (15 * math.pi / 180.0)
            v = 30 * math.pi / 180.0 + self.ahaq_rand() * (15 * math.pi / 180.0)
            self._fill_gcap_pair(2, u, v, f1, f2)
            self._fill_gcap_delta_neutral(2)

    def _fill_gcap_pair(self, idx: int, u: float, v: float, f1: float, f2: float):
        self.g_cap[idx, 1, 1] = -math.pi + u
        self.g_cap[idx, 1, 2] = -3 * math.pi / 4 + f1 * u
        self.g_cap[idx, 1, 3] = -math.pi / 2
        self.g_cap[idx, 1, 4] = -math.pi / 4 + f1 * v
        self.g_cap[idx, 1, 5] = v
        self.g_cap[idx, 1, 6] = math.pi / 4 + f2 * v
        self.g_cap[idx, 1, 7] = math.pi / 2
        self.g_cap[idx, 1, 8] = 3 * math.pi / 4 + f2 * u
        self.g_cap[idx, 1, 9] = math.pi + u

    def _fill_gcap_delta_neutral(self, idx: int):
        u1 = 0.0
        v1 = 0.0
        self.g_cap[idx, 2, 1] = -math.pi + 2 * u1 / 3
        self.g_cap[idx, 2, 2] = -3 * math.pi / 4
        self.g_cap[idx, 2, 3] = -math.pi / 2
        self.g_cap[idx, 2, 4] = -math.pi / 4
        self.g_cap[idx, 2, 5] = 2 * v1 / 3
        self.g_cap[idx, 2, 6] = math.pi / 4 + v1
        self.g_cap[idx, 2, 7] = math.pi / 2 + 2 * (u1 + v1) / 3
        self.g_cap[idx, 2, 8] = 3 * math.pi / 4 + u1
        self.g_cap[idx, 2, 9] = math.pi + 2 * u1 / 3

    def _interp(self, a: int, b: int, alpha: float) -> float:
        alpha_temp = math.pi + alpha
        q = math.floor(4 * alpha_temp / math.pi)
        if q == 8:
            q = 7
        alpha_i = -math.pi + (math.pi * q) / 4.0
        return self.g_cap[a, b, q + 1] + ((4 * (alpha - alpha_i)) / math.pi) * (
            self.g_cap[a, b, q + 2] - self.g_cap[a, b, q + 1]
        )

    def g_loop1(self, alpha: float) -> float:
        return self._interp(1, 1, alpha)

    def g_delta1(self, alpha: float) -> float:
        return self._interp(1, 2, alpha)

    def g_loop2(self, alpha: float) -> float:
        return self._interp(2, 1, alpha)

    def g_delta2(self, alpha: float) -> float:
        return self._interp(2, 2, alpha)

    def get_orient(self, i: int, j: int) -> float:
        """Scalar fallback — kept for debugging and backward compatibility."""
        st = self.singularity_type
        local_orient = 0.0
        if st == 1:
            local_orient = math.atan(
                max(0.0, (self.k_arch - self.k_arch * i / (self.H * self.arch_fact2)))
                * math.cos(j * math.pi / (self.W * self.arch_fact1))
            )
        elif st in (2, 3, 4):
            z = ComplexPoint(float(j), float(i))
            v1 = ComplexPoint(z.x - self.d[1].x, z.y - self.d[1].y)
            u1 = ComplexPoint(z.x - self.l[1].x, z.y - self.l[1].y)
            local_orient = 0.5 * (self.g_delta1(arg_complex(v1)) - self.g_loop1(arg_complex(u1)))
        elif st in (5, 6):
            z = ComplexPoint(float(j), float(i))
            v1 = ComplexPoint(z.x - self.d[1].x, z.y - self.d[1].y)
            u1 = ComplexPoint(z.x - self.l[1].x, z.y - self.l[1].y)
            v2 = ComplexPoint(z.x - self.d[2].x, z.y - self.d[2].y)
            u2 = ComplexPoint(z.x - self.l[2].x, z.y - self.l[2].y)
            local_orient = 0.5 * (self.g_delta1(arg_complex(v1)) - self.g_loop1(arg_complex(u1)))
            local_orient += 0.5 * (self.g_delta2(arg_complex(v2)) - self.g_loop2(arg_complex(u2)))
        degrees = int(local_orient * self.rad_deg_fact)
        if degrees > 0:
            degrees = degrees % 180
        elif degrees < 0:
            degrees = -(((-1) * degrees) % 180) + 180
        return degrees * self.deg_rad_fact

    # ------------------------------------------------------------------
    # Vectorized interpolation helper for orientmap_vectorized
    # ------------------------------------------------------------------
    def _interp_vec(self, a: int, b: int, alpha: np.ndarray) -> np.ndarray:
        """Piecewise-linear interpolation over g_cap table, fully vectorized.

        Replicates the scalar _interp() but operates on a 2-D NumPy array of
        angles simultaneously — no Python loop over pixels.
        """
        alpha_temp = math.pi + alpha                    # shift to [0, 2π)
        q = np.floor(4.0 * alpha_temp / math.pi).astype(np.int32)
        q = np.clip(q, 0, 7)                            # q ∈ [0, 7]

        alpha_i = -math.pi + (math.pi * q) / 4.0       # base angle for segment

        # g_cap[a, b] has shape (10,); indices q+1 and q+2 (1-based, clamped)
        gcap_row = self.g_cap[a, b]                     # shape (10,)
        g1 = gcap_row[q + 1]                            # shape (H, W) broadcast
        g2 = gcap_row[np.minimum(q + 2, 9)]            # clamp to avoid OOB

        return g1 + ((4.0 * (alpha - alpha_i)) / math.pi) * (g2 - g1)

    def orientmap(self):
        """Fully vectorized orientation map — replaces the per-pixel Python loop.

        Computes self.orient[i, j] = get_orient(i - padding, j - padding) for
        every pixel simultaneously using NumPy broadcasting.  The scalar
        get_orient() is kept unchanged for debugging.
        """
        Hm = self.H + self.margin
        Wm = self.W + self.margin

        # Pixel coordinate grids (shifted by padding to match the original loop)
        i_grid, j_grid = np.meshgrid(
            np.arange(Hm, dtype=np.float64) - self.padding,
            np.arange(Wm, dtype=np.float64) - self.padding,
            indexing='ij',
        )  # both shape (Hm, Wm)

        st = self.singularity_type
        local_orient = np.zeros((Hm, Wm), dtype=np.float64)

        if st == 1:
            # atan(max(0, k*(1 - i/(H*f2))) * cos(j*π/(W*f1)))
            factor = np.maximum(
                0.0,
                self.k_arch - self.k_arch * i_grid / (self.H * self.arch_fact2),
            )
            local_orient = np.arctan(factor * np.cos(j_grid * math.pi / (self.W * self.arch_fact1)))

        elif st in (2, 3, 4):
            # alpha_v1 = arg(z - d1),  alpha_u1 = arg(z - l1)
            alpha_v1 = np.arctan2(i_grid - self.d[1].y, j_grid - self.d[1].x)
            alpha_u1 = np.arctan2(i_grid - self.l[1].y, j_grid - self.l[1].x)
            local_orient = 0.5 * (
                self._interp_vec(1, 2, alpha_v1) - self._interp_vec(1, 1, alpha_u1)
            )

        elif st in (5, 6):
            alpha_v1 = np.arctan2(i_grid - self.d[1].y, j_grid - self.d[1].x)
            alpha_u1 = np.arctan2(i_grid - self.l[1].y, j_grid - self.l[1].x)
            alpha_v2 = np.arctan2(i_grid - self.d[2].y, j_grid - self.d[2].x)
            alpha_u2 = np.arctan2(i_grid - self.l[2].y, j_grid - self.l[2].x)
            local_orient = 0.5 * (
                self._interp_vec(1, 2, alpha_v1) - self._interp_vec(1, 1, alpha_u1)
            )
            local_orient += 0.5 * (
                self._interp_vec(2, 2, alpha_v2) - self._interp_vec(2, 1, alpha_u2)
            )

        # Convert to degrees, apply modulo in the same way as the scalar path,
        # then convert back to radians.
        degrees = (local_orient * self.rad_deg_fact).astype(np.int64)
        pos_mask = degrees > 0
        neg_mask = degrees < 0
        degrees[pos_mask] = degrees[pos_mask] % 180
        degrees[neg_mask] = -((-degrees[neg_mask]) % 180) + 180

        self.orient = (degrees * self.deg_rad_fact).astype(np.float32)
        return self.orient

    def seed_pos(self):
        # Scale seed count proportionally with canvas area so coverage is
        # consistent regardless of whether we render at 1×, 2×, or larger scale.
        _ref_area = 256 * 360
        _canvas_area = (self.H + self.margin) * (self.W + self.margin)
        _area_scale = _canvas_area / _ref_area
        n_seeds = int((1200 + math.floor(self.ahaq_rand() * 150)) * _area_scale)
        for _ in range(n_seeds):
            i_blob = math.floor(5 + self.ahaq_rand() * (self.H + self.margin - 10))
            j_blob = math.floor(5 + self.ahaq_rand() * (self.W + self.margin - 10))
            self.filt_map[i_blob, j_blob] = 1
            self.f_print1_2Dmat[i_blob:i_blob + 4, j_blob:j_blob + 4] = 100
            self.filt_map[i_blob:i_blob + 4, j_blob:j_blob + 4] = 1
