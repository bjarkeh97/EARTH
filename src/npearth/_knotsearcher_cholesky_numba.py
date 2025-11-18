import numpy as np
from numba import njit
from copy import deepcopy, copy

from npearth._cholesky_update import cholupdate, choldowndate, cholsolve
from npearth._knotsearcher_base import KnotSearcherBase


@njit(fastmath=True, cache=True)
def search(
    bx_sorted: np.ndarray,  # 2D (N,M) float64, C-contiguous
    y_sorted: np.ndarray,  # 1D (N,) float64, C-contiguous
    xv_sorted: np.ndarray,  # 1D (N,) float64, C-contiguous
    ridge: float,
    m: int,
    ssr_min: float,
):
    # --- initial simple quantities (all scalar/array primitives) ---
    N = bx_sorted.shape[0]
    M = bx_sorted.shape[1]

    # y_square = y_sorted @ y_sorted  (compute as loop for numba friendliness)
    y_square = 0.0
    for ii in range(N):
        y_square += y_sorted[ii] * y_sorted[ii]

    # boolean mask (fine)
    active_basis_mask = np.empty(N, dtype=np.bool_)
    for ii in range(N):
        active_basis_mask[ii] = bx_sorted[ii, m] > 0.0

    # V = bx_sorted.T @ bx_sorted   (Numba handles dot for contiguous float64)
    V = bx_sorted.T @ bx_sorted  # resulting (M,M) float64

    # c = bx_sorted.T @ y_sorted
    c = bx_sorted.T @ y_sorted  # (M,)

    # L = cholesky(V + ridge*I)  (numba supports np.linalg.cholesky)
    # Build V + ridge*I in place to avoid temporary eye allocation:
    for ii in range(M):
        V[ii, ii] += ridge
    L = np.linalg.cholesky(V)  # note: V mutated (diagonal increased)

    # restore V to original without ridge for the algorithm (if you need)
    for ii in range(M):
        V[ii, ii] -= ridge

    # initialize working arrays
    V_m_old = np.empty(M, dtype=np.float64)
    for j in range(M):
        V_m_old[j] = V[j, M - 1]

    x_u = np.zeros(M, dtype=np.float64)
    x_d = np.zeros(M, dtype=np.float64)

    u = xv_sorted[0]
    last_i = 0
    last_last_i = 0
    a_best = np.full(M, np.nan, dtype=np.float64)
    t_star = np.nan

    C_u = 0.0
    V_u = np.zeros(M, dtype=np.float64)
    D_u = 0.0
    F_u = 0.0

    # --- main knot loop: use index-based loops and scalar accumulators ---
    for i in range(1, N - 1):  # skip first and last
        t = xv_sorted[i]

        # skip equal knots (no change) and inactive basis entries
        if t == u:
            continue
        if not active_basis_mask[i]:
            continue

        # ------- compute C_t = C_u + sum_{k=last_last_i}^{last_i-1} bx[k,m]*y[k] -------
        C_t = C_u
        if last_last_i < last_i:
            s = 0.0
            for k in range(last_last_i, last_i):
                s += bx_sorted[k, m] * y_sorted[k]
            C_t = C_u + s
        C_u = C_t

        # ------- update c[M-1] by adding sum_{k=last_i}^{i-1} bx[k,m]*y[k]*(xv[k]-t) + (u-t)*C_t -------
        sum_term = 0.0
        if last_i < i:
            for k in range(last_i, i):
                sum_term += bx_sorted[k, m] * y_sorted[k] * (xv_sorted[k] - t)
        c[M - 1] = c[M - 1] + sum_term + (u - t) * C_t

        # ------- update V: keep a copy of old last-column -------
        for jj in range(M):
            V_m_old[jj] = V[jj, M - 1]

        # for each column j != M-1 compute V_tj and the incremental update
        for j in range(M - 1):
            # V_tj = sum_{k=last_last_i}^{last_i-1} bx[k,m]*bx[k,j] + V_u[j]
            V_tj = V_u[j]
            if last_last_i < last_i:
                s_v = 0.0
                for k in range(last_last_i, last_i):
                    s_v += bx_sorted[k, m] * bx_sorted[k, j]
                V_tj = V_u[j] + s_v
            V_u[j] = V_tj

            # incremental term over k in [last_i, i)
            inc = 0.0
            if last_i < i:
                for k in range(last_i, i):
                    inc += bx_sorted[k, m] * bx_sorted[k, j] * (xv_sorted[k] - t)

            V[j, M - 1] = V[j, M - 1] + inc + (u - t) * V_tj
            V[M - 1, j] = V[j, M - 1]

        # ------- update diagonal entry V[M-1,M-1] -------
        # D_t = sum_{k=last_last_i}^{last_i-1} bx[k,m]^2 + D_u
        D_t = D_u
        if last_last_i < last_i:
            s_d = 0.0
            for k in range(last_last_i, last_i):
                val = bx_sorted[k, m]
                s_d += val * val
            D_t = D_u + s_d
        D_u = D_t

        # F_t = sum_{k=last_last_i}^{last_i-1} bx[k,m]^2 * xv[k] + F_u
        F_t = F_u
        if last_last_i < last_i:
            s_f = 0.0
            for k in range(last_last_i, last_i):
                val = bx_sorted[k, m]
                s_f += val * val * xv_sorted[k]
            F_t = F_u + s_f
        F_u = F_t

        # add contribution from k in [last_i, i)
        diag_inc = 0.0
        if last_i < i:
            for k in range(last_i, i):
                val = bx_sorted[k, m]
                diag_inc += (val * val) * ((xv_sorted[k] - t) ** 2)

        V[M - 1, M - 1] = (
            V[M - 1, M - 1] + diag_inc + (t * t - u * u) * D_t + 2.0 * (u - t) * F_t
        )

        # ------- check dVM and update chol factor L via update/downdate -------
        # dVM = V[:, M-1] - V_m_old
        all_zero = True
        for jj in range(M):
            if V[jj, M - 1] != V_m_old[jj]:
                all_zero = False
                break
        if all_zero:
            # nothing changed, continue
            u = t
            last_last_i = last_i
            last_i = i
            continue

        # compute dVM in place into x_u
        for jj in range(M):
            x_u[jj] = V[jj, M - 1] - V_m_old[jj]

        # guard against small/negative pivot
        denom = x_u[M - 1]
        if denom <= 0.0:
            # numerical problem; skip this knot
            u = t
            last_last_i = last_i
            last_i = i
            continue

        inv_sqrt = 1.0 / np.sqrt(denom)
        for jj in range(M):
            x_u[jj] = x_u[jj] * inv_sqrt

        # x_d = x_u but with last element zero
        for jj in range(M):
            x_d[jj] = x_u[jj]
        x_d[M - 1] = 0.0

        # update L using jitted chol update/downdate
        L = choldowndate(cholupdate(L, x_u), x_d)

        # solve for coefficients: a = cholsolve(L + ridge * I, c)
        # Build matrix T = L + ridge*I (we will modify a small temp diagonal)
        # We'll create a temporary copy of L to add ridge to diagonal — allocate once per knot
        T = np.empty((M, M), dtype=np.float64)
        for jj in range(M):
            for kk in range(M):
                T[jj, kk] = L[jj, kk]
        for jj in range(M):
            T[jj, jj] += ridge

        a = cholsolve(T, c)

        # SSR = y_square - a @ c  (compute dot by loop)
        ac = 0.0
        for jj in range(M):
            ac += a[jj] * c[jj]
        SSR = y_square - ac
        if SSR < 0.0:
            SSR = np.inf

        # update best
        if SSR < ssr_min:
            ssr_min = SSR
            # copy a into a_best (avoid referencing same array)
            for jj in range(M):
                a_best[jj] = a[jj]
            t_star = t

        # update moving indices
        u = t
        last_last_i = last_i
        last_i = i

    # return best found
    return ssr_min, t_star, a_best


class KnotSearcherCholeskyNumba(KnotSearcherBase):
    def __init__(
        self,
        bx: np.ndarray,
        y: np.ndarray,
        xv: np.ndarray,
        m: int,
        v: int,
        ridge: float = 1e-8,
    ) -> None:
        super().__init__(bx, y, xv, m, v, ridge)

    def search_over_knots(self, ts: np.ndarray, lof_star: float) -> None:
        ssr_min = lof_star  # Initialize LOF to LOF^*
        sort_idx = np.argsort(self.xv)[
            ::-1
        ]  # Indices for sorting xv in descending order. Need u>=t for update to work

        # We should search from largest knot so:
        gn = copy(self.g)
        gn.add_split_end(self.m, self.v, max(self.xv))

        # Sort all relevant matrices/vectors accordingly
        bx_sorted = np.ascontiguousarray(
            gn.bx[sort_idx, :].astype(np.float64)
        )  # Sort basis matrix accordingly
        y_sorted = np.ascontiguousarray(
            self.y[sort_idx].astype(np.float64)
        )  # Sort response vector accordingly
        # y_square = y_sorted @ y_sorted  # Precompute y^T y as is O(N)
        xv_sorted = np.ascontiguousarray(
            self.xv[sort_idx].astype(np.float64)
        )  # Sort predictor vector accordingly

        ssr_min, t_star, a_best = search(
            bx_sorted, y_sorted, xv_sorted, self.ridge, self.m, ssr_min
        )

        return ssr_min, t_star, a_best


if __name__ == "__main__":
    # bx = np.genfromtxt("C:/Users/Bruger/Code/EARTH/src/earth/data/bx.csv")
    # y = np.genfromtxt("C:/Users/Bruger/Code/EARTH/src/earth/data/y.csv")

    from npearth._basis_function import BasisMatrix

    X = np.reshape([5.0, 4.0, 3.0, 2.0, 1.0, 1.0], (-1, 1))
    v = 0
    xv = X[:, v]

    print(f"X: {X}")

    noise = 0 * np.random.normal(0, 0.1, size=xv.shape)
    y = 2.0 + 3.0 * np.maximum(0, xv - 3) + noise
    print(f"y: {y}")

    bm = BasisMatrix(X)

    knts2 = KnotSearcherCholeskyNumba(bm, y, xv, 0, 0)
    ssr, t, a = knts2.search_over_knots(xv, np.inf)
    print(f"t: {t}, ssr: {ssr}, a: {a}")

    for i, b in enumerate(bm.basis):
        print(b.svt)
