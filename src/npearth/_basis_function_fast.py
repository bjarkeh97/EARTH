import numpy as np
from typing import Literal
import math
from copy import deepcopy


class BasisFunction:
    def __init__(self) -> None:
        self.svt = []

    def add_step(self, s: Literal[-1, 0, 1], v: int, t: float):
        self.svt.append((s, v, t))
        return self

    def return_variables_used(self) -> list:
        if self.svt:
            return [svt[1] for svt in self.svt]
        else:
            return []

    def hinge(self, s: int, xv: np.ndarray, t: float) -> float:
        if s == 0:
            return xv
        else:
            return np.maximum(s * (xv - t), 0)

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        if self.svt:
            return math.prod([self.hinge(s, X[:, v], t) for s, v, t in self.svt])
        else:
            return np.ones(X.shape[0])  # Be a constant


class BasisMatrix:
    def __init__(self, X: np.ndarray) -> None:
        self.X = X
        self.m, self.n = X.shape
        self.bx = np.ones(shape=(self.m, 1))
        self.basis: list[BasisFunction] = [BasisFunction()]

    def add_split(self, m: int, v: int, t: float) -> None:
        xv = self.X[:, v]
        bm = self.basis[m]
        bxm = self.bx[:, m]  # Get the values for all x for the m'th term

        hinge_neg = bxm * bm.hinge(-1, xv, t)
        hinge_pos = bxm * bm.hinge(1, xv, t)

        self.bx = np.concatenate(
            [self.bx, hinge_neg[:, None], hinge_pos[:, None]], axis=1
        )
        self.basis.extend(
            [deepcopy(bm).add_step(-1, v, t), deepcopy(bm).add_step(1, v, t)]
        )

    def add_split_end(self, m: int, v: int, t: float) -> None:
        xv = self.X[:, v]
        bm = self.basis[m]
        bxm = self.bx[:, m]  # Get the values for all x for the m'th term

        hinge_neg = bxm * xv
        hinge_pos = bxm * bm.hinge(1, xv, t)

        self.bx = np.concatenate(
            [self.bx, hinge_neg[:, None], hinge_pos[:, None]], axis=1
        )
        self.basis.append(deepcopy(bm).add_step(0, v, 0))
        self.basis.append(deepcopy(bm).add_step(1, v, t))

        # self.bx = np.concatenate(
        #     [
        #         self.bx,
        #         (self.bx[:, m] * bm.hinge(-1, xv, t))
        #         .reshape((self.m, 1))
        #         .astype(float),
        #     ], axis=1
        # )
        # self.basis.append(deepcopy(bm).add_step(-1,v,t))
        # self.bx = np.concatenate(
        #     [
        #         self.bx,
        #         (self.bx[:, m] * bm.hinge(1, xv, t))
        #         .reshape((self.m, 1))
        #         .astype(float),
        #     ], axis=1
        # )
        # self.basis.append(deepcopy(bm).add_step(1,v,t))
