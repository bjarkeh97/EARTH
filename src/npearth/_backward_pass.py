import numpy as np

from npearth._basis_function import BasisFunction


class BackwardsStepwise:
    def __init__(self, ridge: float, d: float):
        """
        Class for performing the pruning. Follows Algorithm 3 [1].

        Parameters
        ----------
        ridge : float
            The ridge factor delta in eq 54 [1]. Should be as small as possible

        d : float
            The smoothing parameter d from eq 32 [1]. Try range 2<=d<=4.
            The smoothing parameter penalizes model complexity functions to the models.

         References
        ----------

        [1] Friedman, Jerome. Multivariate Adaptive Regression Splines.
            Annals of Statistics. Volume 19, Number 1 (1991), 1-67.
        """
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
        """
        Implementation of Algorithm 3 [1].

        Parameters
        ----------
        basis : list[BasisFunction]
            The list of basis functions defining the model at the current
            pruning iteration.

        X : ndarray of shape (n_samples, n_features)
            Input design matrix.

        y : ndarray of shape (n_samples,)
            Target vector.

        sample_weight : ndarray of shape (n_samples,)
            Per-sample weights used in the weighted least squares fit.

        Returns
        -------
        J_star : list[BasisFunction]
            The optimal set of basis functions according the CGV score

        coeffs_star : list
            The coefficients for the optimal basis

        References
        ----------

        [1] Friedman, Jerome. Multivariate Adaptive Regression Splines.
            Annals of Statistics. Volume 19, Number 1 (1991), 1-67.
        """
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
    ) -> tuple[np.ndarray, float]:
        """
        Evaluate a set of basis functions and compute the Generalized Cross
        Validation (GCV) score used for backward stepwise pruning accoring to [1]
        Calculating effective degrees of freedom C(M) according to eq 31 [1]

        Parameters
        ----------
        basis : list[BasisFunction]
            The list of basis functions defining the model at the current
            pruning iteration.

        X : ndarray of shape (n_samples, n_features)
            Input design matrix.

        y : ndarray of shape (n_samples,)
            Target vector.

        sample_weight : ndarray of shape (n_samples,)
            Per-sample weights used in the weighted least squares fit.

        Returns
        -------
        coeffs : ndarray of shape (M,)
            The estimated coefficient vector for the given basis subset.

        lof : float
            The GCV lack-of-fit score associated with this basis.

        References
        ----------

        [1] Friedman, Jerome. Multivariate Adaptive Regression Splines.
            Annals of Statistics. Volume 19, Number 1 (1991), 1-67.
        """

        # Evaluate all basis functions → design matrix Bx  (O(NM))
        bx = np.column_stack([b.evaluate(X) for b in basis])
        N, M = bx.shape

        # Apply weighting: W^(1/2)
        w = np.sqrt(sample_weight)
        bx_weighted = bx * w[:, None]

        # Weighted Gram matrix Bᵀ W B  (O(NM²))
        bx_w_squared = bx_weighted.T @ bx_weighted

        # Ridge-regularized inverse (O(M³)) - Used for dof, coefficients
        G_r = bx_w_squared + self.ridge * np.eye(M)
        G_r_inv = np.linalg.solve(G_r, np.eye(M))

        # Effective degrees of freedom: trace(H) = trace(BᵀWB (BᵀWB + λI)⁻¹) from \hat{y} = Hy
        edof = np.trace(bx_w_squared @ G_r_inv)

        # Weighted target
        y_weighted = y * w

        # Complexity penalty C(M) eq 32 [1]
        C = edof + 1 + self.d * (M - 1)

        # Ridge solution for coefficients
        coeffs = G_r_inv @ (bx_weighted.T @ y_weighted)

        # Compute weighted SSR
        residuals = y_weighted - bx_weighted @ coeffs
        ssr = (residuals**2).sum()

        # GCV lack-of-fit eq 30 [1]. Guard against infinity
        if np.isclose(C, N):
            lof = np.inf
        else:
            lof = ssr / (1 - C / N) ** 2

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

    backwardsstepwise = BackwardsStepwise(ridge=1e-8, d=2)
    J_star, coeffs_star = backwardsstepwise.backward_pass(
        model.basis, X, y, np.ones_like(y)
    )
    print("done")
