import numpy as np

from npearth._basis_function import BasisFunction
from npearth.earth import EARTH


class BackwardsStepwise:
    def __init__(self):
        self.best_basis = None
        self.best_coeffs = None

    def backward_pass(
        self, model: EARTH, X: np.ndarray, y: np.ndarray, ridge: float
    ) -> tuple[list[BasisFunction], list]:
        J_star = model.basis[:]  # Initial set of basis functions
        K_star = J_star[:]
        coeffs_star, lof_star = self.evaluate_basis(J_star, X, y, 1e-8)
        for M in range(len(J_star), 0, -1):  # M = M_max to 2
            b = np.inf
            L = K_star[:]
            for m in range(1, M):  # m = 2 to M
                K = L[:m] + L[m + 1 :]  # L - {m}
                a, lof = self.evaluate_basis(K, X, y, ridge)
                if lof < b:
                    b = lof
                    K_star = K
                if lof < lof_star:
                    lof_star = lof
                    J_star = K
                    coeffs_star = a
        self.best_basis = J_star
        self.best_coeffs = coeffs_star
        return J_star, coeffs_star

    def prune_model(self, model: EARTH) -> EARTH:
        if self.best_coeffs is None or self.best_basis is None:
            print("backward pass not run")
        else:
            model.basis = self.best_basis
            model.coeffs = self.best_coeffs

    def evaluate_basis(
        self, basis: list[BasisFunction], X: np.ndarray, y: np.ndarray, ridge: float
    ) -> tuple[list, float]:
        # Create Bx #O(NM) Evaluate N times for M functions
        bx = np.column_stack([b.evaluate(X) for b in basis])

        rank = np.linalg.matrix_rank(bx)
        C = rank + 1
        N, M = bx.shape
        coeffs, ssr, _, _ = np.linalg.lstsq(
            bx.T @ bx + ridge * np.eye(M), bx.T @ y, rcond=None
        )
        residuals = y - np.dot(bx, coeffs)
        ssr = (residuals**2).sum()
        lof = ssr / (1 - C / N) ** 2  # GCV score
        # Solve for a, ssr  O(M^3)
        return coeffs, lof


if __name__ == "__main__":
    import pickle

    with open(
        "/Users/bjarkehogdall/Code/EARTH/src/npearth/data/test_model.pickle", "rb"
    ) as f:
        model = pickle.load(f)

    with open(
        "/Users/bjarkehogdall/Code/EARTH/src/npearth/data/test_X.pickle", "rb"
    ) as f:
        X = pickle.load(f)

    with open(
        "/Users/bjarkehogdall/Code/EARTH/src/npearth/data/test_y.pickle", "rb"
    ) as f:
        y = pickle.load(f)

    backwardsstepwise = BackwardsStepwise()
    J_star, coeffs_star = backwardsstepwise.backward_pass(model, X, y, 1e-8)
    print("done")
