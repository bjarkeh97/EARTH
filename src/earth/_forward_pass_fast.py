import numpy as np
from ._basis_function import BasisFunction, BasisMatrix
from copy import deepcopy


class ForwardPasser:
    def __init__(self) -> None:
        self.coefs = []
        self.bx = None

    def forward_pass(
        self, X: np.ndarray, y: np.ndarray, M_max: float
    ) -> tuple[list, list[BasisFunction]]:
        bx = BasisMatrix(X=X)
        lof = np.inf
        M = 1

        while M <= M_max:
            last_lof = lof
            m_star, v_star, t_star, coeffs_star = None, None, None, None
            for m in range(M):
                vs_in_m = bx.basis[m].return_variables_used()
                vs_not_in_m = [v for v in range(bx.n) if v not in vs_in_m]
                for v in vs_not_in_m:
                    # Find ts from set {x_vj | B_m(x_ij) > 0} with x_ij the j'th row in X
                    # We have B_m(X) stored as bx[:,m]
                    active_basis_mask = bx.bx[:, m] > 0
                    ts = X[active_basis_mask, v]
                    for t in ts:
                        g = deepcopy(bx)
                        g.add_split(m, v, t)
                        coeffs, ssr, _, _ = np.linalg.lstsq(g.bx, y, rcond=None)
                        residuals = y - np.dot(g.bx, coeffs)
                        ssr = np.pow(residuals, 2).sum()
                        if ssr < lof:
                            lof = ssr
                            m_star = m
                            v_star = v
                            t_star = t
                            coeffs_star = coeffs
            if lof == last_lof:
                print(f"No improvement in LOF after {M} terms")
                break
            bx.add_split(m_star, v_star, t_star)
            M += 2
        self.coefs = coeffs_star
        self.bx = bx
        return self.coefs, bx.basis
