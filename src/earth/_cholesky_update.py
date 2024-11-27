import numpy as np
from numba import njit


@njit
def cholupdate(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Perform a Cholesky rank-one update.

    Parameters:
    L : ndarray
        Lower triangular matrix (Cholesky factor).
    x : ndarray
        Vector used for the rank-one update.

    Returns:
    L : ndarray
        Updated lower triangular matrix.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
    """
    n = len(x)
    for k in range(n):
        r = np.sqrt(L[k, k] ** 2 + x[k] ** 2)
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if k < n - 1:
            L[k + 1 : n, k] = (L[k + 1 : n, k] + s * x[k + 1 : n]) / c
            x[k + 1 : n] = c * x[k + 1 : n] - s * L[k + 1 : n, k]
    return L


if __name__ == "__main__":
    # Example matrix A and its Cholesky factor L
    X = np.random.random((5000, 5000))
    A = np.matmul(X.transpose(), X)
    L = np.linalg.cholesky(A.copy())

    # Vector u for the update (rank-1 update)
    u = np.random.normal(size=A.shape[0])

    # Perform the rank-1 Cholesky update
    L_updated = cholupdate(L.copy(), u.copy())
    L_raw = np.linalg.cholesky(A.copy() + np.outer(u, u))

    assert np.all((L_updated - L_raw) ** 2 < 1e-16)
