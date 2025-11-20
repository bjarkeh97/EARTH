import numpy as np
from copy import deepcopy
from npearth._basis_function import BasisMatrix
from npearth._knotsearcher_base import KnotSearcherBase


class KnotSearcherSVD(KnotSearcherBase):

    def search_over_knots(
        self, ts: np.ndarray, lof_star: float
    ) -> tuple[float, float, np.ndarray]:
        t_star, coeffs_star = None, None
        for t in ts:
            g = deepcopy(self.g)
            g.add_split_end(self.m, self.v, t)
            w = np.sqrt(self.sample_weight)
            bx_weighted = g.bx * w[:, None]
            y_weighted = self.y * w
            coeffs, ssr, _, _ = np.linalg.lstsq(bx_weighted, y_weighted, rcond=None)
            residuals = y_weighted - np.dot(bx_weighted, coeffs)
            ssr = (residuals**2).sum()
            if ssr < lof_star:
                lof_star = ssr
                t_star = t
                coeffs_star = coeffs
        return lof_star, t_star, coeffs_star
