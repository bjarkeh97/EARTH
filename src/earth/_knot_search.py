import numpy as np
#from ._cholesky_update import cholupdate


class KnotSearcher:
    def __init__(self, bx: np.ndarray, y: np.ndarray, xv: np.ndarray, m: int) -> None:
        self.B = bx
        self.y = y
        self.xv = xv
        self.L = None
        self.m = m

    def generate_V(self) -> None:
        bx_mean = self.B.mean(axis=0)
        self.V = np.matmul(self.B.transpose(), self.B - bx_mean)

    def generate_c(self) -> None:
        y_mean = self.y.mean()
        self.y_mean=y_mean
        self.c = np.dot(self.B.transpose(), y - y_mean)

    def update_c(self, u: int, t: int) -> None:
        """
        Update c
        """
        mask_1 = np.where((self.xv>=t)&(self.xv<u))
        mask_2 = np.where(self.xv>=u)
        self.c[-1] + sum((y[mask_1]-self.y_mean)*self.B[m,mask_1]*(self.xv[mask_1]-t))

    def update_V(self) -> None:
        """
        Update covariance matrix
        """
        pass

    def solve_system(self) -> None:
        """
        Use cholesky decomposition to solve equation Va = c for a
        """
        pass


if __name__ == "__main__":
    bx = np.genfromtxt("C:/Users/Bruger/Code/EARTH/src/earth/data/bx.csv")
    y = np.genfromtxt("C:/Users/Bruger/Code/EARTH/src/earth/data/y.csv")

    knts = KnotSearcher(bx, y)
    knts.generate_V()
    knts.generate_c()
    knts.update_c(5,2)
    print("done")
