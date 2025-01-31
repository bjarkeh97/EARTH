import numpy as np
from copy import copy

# from ._cholesky_update import cholupdate


class KnotSearcher:
    def __init__(self, bx: np.ndarray, y: np.ndarray, xv: np.ndarray, m: int) -> None:
        self.y = y
        self.xv = xv
        self.L = None
        self.m = m
        self.M = bx.shape[1]
        sort_idx = np.argsort(xv)[::-1]
        self.ts = xv[sort_idx]
        self.B = bx#[sort_idx, :]
        self.u = self.ts[0]
        self.generate_V()
        self.generate_c()
        last_i = 0
        # For C vector updating
        self.cn0 = copy(self.c[-1])
        self.lowest = self.u
        self.cqueue1 = 0  # bx[k, m] * t * y[k]
        self.cqueue2 = 0  # bx[k, m] * y[k]
        self.ca2 = 0  # bx[k, m] * xv[k]
        self.ca1 = 0  # bx[k, m] * xv[k] * y[k]
        # For V matrix updating
        self.V0 = self.V[:, -1].copy()
        self.vqueue1 = np.zeros(self.M)  # B_ki * B_kj
        self.vqueue2 = np.zeros(self.M)  # B_ki * B_kj * x_k
        self.vqueueM = 0  # B_ki * B_kj * x_k * x_k
        self.va1 = np.zeros(self.M)  # B[k,m]*B[k,i]
        self.va2 = np.zeros(self.M)  # B[k,m]*B[k,i]*x[k]
        self.vaM = 0  # B[k,m]*B[k,i]*x[k]*x[k]

    def generate_V(self) -> None:
        self.V: np.ndarray = self.B.transpose() @ self.B

    def generate_c(self) -> None:
        self.c: np.ndarray = self.B.transpose() @ self.y

    def update_(self, i: int) -> None:
        if i == 0:
            return None
        t = self.ts[i]
        k = i - 1  # In future this could just be eg self.last_i : i-1 maybe
        if t < self.lowest:
            self.ca1 += self.B[k, self.m] * self.xv[k] * self.y[k] + self.cqueue1
            self.ca2 += self.B[k, self.m] * self.y[k] + self.cqueue2
            self.c[-1] = self.cn0 + self.ca1 - self.ca2 * t
            self.cqueue1 = self.cqueue2 = 0
        else:
            self.cqueue1 += self.B[k, self.m] * self.xv[k] * self.y[k]
            self.cqueue2 += self.B[k, self.m] * self.y[k]
        # update V[:-1,m]
        for n in range(self.M - 1):
            if t < self.lowest:
                self.va1[n] += self.B[k, self.m] * self.B[k, n] + self.vqueue1[n]
                self.va2[n] += (
                    self.B[k, self.m] * self.B[k, n] * self.xv[k] + self.vqueue2[n]
                )
                self.V[n, -1] = self.V0[n] + self.va2[n] - self.va1[n] * t
                self.V[-1, n] = self.V[n, -1]
                self.vqueue1[n] = 0
                self.vqueue2[n] = 0
            else:
                self.vqueue1[n] += self.B[k, self.m] * self.B[k, n]
                self.vqueue2[n] += self.B[k, self.m] * self.B[k, n] * self.xv[k]
        # update V[M,M]
        if t < self.lowest:
            self.va1[-1] += self.B[k, self.m] ** 2 + self.vqueue1[-1]
            self.va2[-1] += self.B[k, self.m] ** 2 * self.xv[k] + self.vqueue2[-1]
            self.vaM += self.B[k, self.m] ** 2 * self.xv[k] ** 2 + self.vqueueM
            self.V[-1, -1] = (
                self.V0[-1] + self.vaM + self.va1[-1] * t**2 - 2 * t * self.va2[-1]
            )
            self.vqueue1[-1] = 0
            self.vqueue2[-1] = 0
            self.vqueueM = 0
        else:
            self.vqueue1[-1] += self.B[k, self.m] ** 2
            self.vqueue2[-1] += self.B[k, self.m] ** 2 * self.xv[k]
            self.vqueueM += self.B[k, self.m] ** 2 * self.xv[k] ** 2
        self.lowest = t

    def solve_system(self) -> None:
        """
        Use cholesky decomposition to solve equation B^TBx = B^Ty for a
        """
        try:
            a = np.linalg.solve(self.V, self.c)
            SRR = self.y.T @ self.y - self.c.T @ a
        except Exception as e:
            # print(e)
            a = None
            SRR = np.inf
        return a, SRR

    def search_over_knots(self) -> None:
        ssr_min = np.inf
        t = self.ts[0]
        a_best = np.ones(self.M)
        N = len(self.ts)
        for i in range(N):
            self.update_(i)
            # if self.B[i, self.m] == 0:
            #     continue
            if i == N - 1 or i == 0:
                continue
            if self.xv[i] <= self.ts[-1]:
                continue
            a, ssr = self.solve_system()
            if a is None:
                # print("a is None")
                continue
            # residuals = self.y - np.dot(self.B, a)
            # ssr = np.sum(residuals**2)
            if ssr < ssr_min:
                ssr_min = ssr
                t = self.ts[i]
                a_best = a
        return t, ssr_min, a_best


if __name__ == "__main__":
    # bx = np.genfromtxt("C:/Users/Bruger/Code/EARTH/src/earth/data/bx.csv")
    # y = np.genfromtxt("C:/Users/Bruger/Code/EARTH/src/earth/data/y.csv")

    from earth._basis_function_fast import BasisMatrix

    X = np.reshape([5, 4, 3, 3, 1, 1], (-1, 1))
    y = np.arange(6) * 2 - 1 + (np.arange(6) > 2).astype(int)

    bm = BasisMatrix(X)
    bm.add_split_end(0, 0, 5)
    bx = bm.bx

    knts = KnotSearcher(bx, y, X[:, 0], 0)
    print("_" * 20)
    for i in range(6):
        print(i)
        knts.update_(i)
        print(knts.c)
        print(knts.V)
    print("done")

    knts2 = KnotSearcher(bx, y, X[:, 0], 0)
    t, ssr, a = knts2.search_over_knots()
    print(t, ssr, a)


# TODO fix update V part
