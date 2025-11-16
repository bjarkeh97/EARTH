import numpy as np
from npearth._forward_pass_fast import ForwardPasser
from npearth._knotsearcher_base import KnotSearcherBase
from npearth._knotsearcher_cholesky import KnotSearcherCholesky


class EARTH:
    def __init__(self, M_max: int = 15) -> None:
        self.M_max = M_max

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        knot_searcher: KnotSearcherBase = KnotSearcherCholesky,
    ):
        self.coeffs, self.basis = ForwardPasser().forward_pass(
            X, y, self.M_max, knot_searcher
        )

    def predict(self, X) -> np.ndarray:
        l = []
        for i, c in enumerate(self.coeffs):
            l.append(self.basis[i].evaluate(X) * c)
        return sum(l)
