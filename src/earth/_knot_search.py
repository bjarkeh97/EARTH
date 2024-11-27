import numpy as np
from ._cholesky_update import cholupdate


class KnotSearcher:
    def __init__(self, bx: np.ndarray, y: np.ndarray, X: np.ndarray) -> None:
        self.B = bx
        self.y = y
        self.X = X
        self.L = None

    def generate_V(self) -> None:
        bx_mean = self.B.mean(axis=0)
        self.V = np.matmul(self.B.transpose(), self.B - bx_mean)

    def generate_y(self) -> None:
        y_mean = self.y.mean()
        self.c = np.dot(self.B.transpose(), y - y_mean)

    def update_c(self, u: int, t: int) -> None:
        """
        Update c
        """
        pass

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
    knts.generate_y()

    print("done")
