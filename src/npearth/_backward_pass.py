import numpy as np

from npearth._basis_function import BasisFunction


class BackwardsStepwise:
    def __init__(self, ridge: float, d: float):
        self.best_basis: list[BasisFunction] = None
        self.best_coeffs: np.ndarray = None
        self.ridge = ridge
        self.d = d

    def backward_pass(
        self,
        basis: list[BasisFunction],
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
    ) -> tuple[list[BasisFunction], list]:
        J_star = basis  # Initial set of basis functions
        K_star = J_star[:]
        coeffs_star, lof_star = self.evaluate_basis(J_star, X, y, sample_weight)
        for M in range(len(J_star), 0, -1):  # M = M_max to 2
            b = np.inf
            L = K_star[:]
            for m in range(1, M):  # m = 2 to M
                K = L[:m] + L[m + 1 :]  # L - {m}
                a, lof = self.evaluate_basis(K, X, y, sample_weight)
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

    def evaluate_basis(
        self,
        basis: list[BasisFunction],
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
    ) -> tuple[list, float]:
        # Create Bx #O(NM) Evaluate N times for M functions
        bx = np.column_stack([b.evaluate(X) for b in basis])
        N, M = bx.shape
        w = np.sqrt(sample_weight)  # sqrt(W)
        bx_weighted = bx * w[:, None]
        bx_squared = bx_weighted.T @ bx_weighted  # O(NM^2)
        G_r_inv = np.linalg.inv(bx_squared + self.ridge * np.eye(M))  # O(M^3)
        edof = np.trace(bx_squared @ G_r_inv)
        y_weighted = y * w

        C = (
            edof + 1 + self.d * (M - 1)
        )  # M-1 is number of non-constant basis functions. (Might actually not be if a knot is place at last x_i...)
        coeffs, ssr, _, _ = np.linalg.lstsq(
            bx_squared + self.ridge * np.eye(M),
            bx_weighted.T @ y_weighted,
            rcond=None,
        )
        residuals = y_weighted - np.dot(bx_weighted, coeffs)
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

    backwardsstepwise = BackwardsStepwise(ridge=1e-8)
    J_star, coeffs_star = backwardsstepwise.backward_pass(
        model.basis, X, y, np.ones_like(y)
    )
    print("done")
