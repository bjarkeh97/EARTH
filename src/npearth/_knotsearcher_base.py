import numpy as np
from copy import deepcopy
from npearth._basis_function_fast import BasisMatrix


class KnotSearcherBase:
    def __init__(
        self, g: BasisMatrix, y: np.ndarray, xv: np.ndarray, m: int, v: int
    ) -> None:
        self.g = g
        self.y = y
        self.xv = xv
        self.m = m
        self.v = v

    def search_over_knots(
        self, ts: np.ndarray, lof_star: float
    ) -> tuple[float, float, np.ndarray]:
        return None, None, None
