import numpy as np
from ._basis_function_fast import BasisFunction, BasisMatrix
from ._knot_search import KnotSearcher
from copy import deepcopy


class ForwardPasser:
    def __init__(self) -> None:
        self.coefs = []
        self.bx = None
        print("going fast")

    def forward_pass(
        self, X: np.ndarray, y: np.ndarray, M_max: float
    ) -> tuple[list, list[BasisFunction]]:
        bx = BasisMatrix(X=X)
        lof = np.inf
        M = 1
        m_star, v_star, t_star, coeffs_star = None, None, None, None
        while M <= M_max:
            last_lof = lof
            m_star_n, v_star_n, t_star_n, coeffs_star_n = None, None, None, None
            for m in range(M):
                vs_in_m = bx.basis[m].return_variables_used()
                vs_not_in_m = [v for v in range(bx.n) if v not in vs_in_m]
                for v in vs_not_in_m:
                    xv = X[:, v]
                    g = deepcopy(bx)
                    g.add_split_end(m, v, max(xv))
                    knts = KnotSearcher(g.bx, y, xv, m)
                    t, ssr, a = knts.search_over_knots()
                    if ssr <= lof and a is not None:
                        lof = ssr
                        m_star_n = m
                        v_star_n = v
                        t_star_n = t
                        coeffs_star_n = a
            if lof == last_lof:
                print(f"No improvement in LOF after {M} terms")
                self.coefs = coeffs_star
                self.bx = bx
                return self.coefs, bx.basis
            if m_star_n is None:
                print(f"No new coefficients after {M} terms")
                self.coefs = coeffs_star
                self.bx = bx
                return self.coefs, bx.basis
            bx.add_split_end(m_star_n, v_star_n, t_star_n)
            M += 2
            m_star, v_star, t_star, coeffs_star = (
                m_star_n,
                v_star_n,
                t_star_n,
                coeffs_star_n,
            )
        self.coefs = coeffs_star
        self.bx = bx
        return self.coefs, bx.basis
