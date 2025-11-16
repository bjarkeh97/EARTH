import numpy as np
from npearth._forward_pass import ForwardPasser


class EARTH:
    def __init__(self, M_max: int = 15) -> None:
        self.M_max = M_max
        # comment

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.coeffs, self.basis = ForwardPasser().forward_pass(X, y, self.M_max)

    def predict(self, X) -> np.ndarray:
        l = []
        for i, c in enumerate(self.coeffs):
            l.append(self.basis[i].evaluate(X) * c)
        return sum(l)
