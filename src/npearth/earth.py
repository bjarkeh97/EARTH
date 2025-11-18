import numpy as np
from npearth._forward_pass import ForwardPasser
from npearth._knotsearcher_base import KnotSearcherBase
from npearth._knotsearcher_cholesky import KnotSearcherCholesky
from npearth._backward_pass import BackwardsStepwise


class EARTH:
    def __init__(
        self,
        M_max: int = 15,
        ridge: float = 1e-8,
        knot_searcher: KnotSearcherBase = KnotSearcherCholesky,
        prune_model: bool = True,
    ) -> None:
        self.M_max = M_max
        self.ridge = ridge
        self.knot_searcher = knot_searcher
        self.prune_model = prune_model

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        if sample_weight is None:
            sample_weight = np.ones_like(y)
        else:
            if len(sample_weight) != len(y):
                raise ValueError("Sample weights should have same dimensions as y")
        self.coefs, self.basis = ForwardPasser().forward_pass(
            X,
            y,
            self.M_max,
            self.knot_searcher,
            self.ridge,
        )
        if self.prune_model:
            print("Pruning model")
            pruner = BackwardsStepwise(ridge=self.ridge)
            self.basis, self.coefs = pruner.backward_pass(self.basis, X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.basis is None:
            raise RuntimeError("Model not fittet yet")
        bx = np.column_stack([b.evaluate(X) for b in self.basis])
        y = bx @ self.coefs
        return y

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return R^2 score"""
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean) ** 2).sum()
        if v > 0:
            return 1 - u / v
        else:
            return np.nan
