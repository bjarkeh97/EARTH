from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from npearth._basis_function import BasisMatrix


class KnotSearcherBase(ABC):
    """
    Abstract base class for knot searching algorithms used in EARTH.
    Subclasses must implement the `search_over_knots` method.
    """

    def __init__(
        self,
        g: BasisMatrix,
        y: np.ndarray,
        xv: np.ndarray,
        m: int,
        v: int,
        sample_weight: np.ndarray,
        ridge: float,
    ) -> None:
        self.g = g  # BasisMatrix instance
        self.y = y  # Target vector
        self.xv = xv  # Sorted unique predictor values for feature v
        self.m = m  # Index of basis function being split
        self.v = v  # Predictive variable index for this split
        self.sample_weight = sample_weight  # Sample weights. The diagonal of the W matrix in WLS (eg 1/sigma^2). The weight in sum_i w_i L_i of the likelihood
        self.ridge = ridge  # Ridge term for the least square fit

    @abstractmethod
    def search_over_knots(
        self, ts: np.ndarray, lof_star: float
    ) -> tuple[float, float, np.ndarray]:
        """
        Must be implemented by all subclasses.

        Parameters
        ----------
        ts : np.ndarray
            Candidate knot positions.
        lof_star : float
            Current best lack-of-fit score.

        Returns
        -------
        best_lof : float
            Best LOF found.
        best_knot : float
            The knot location achieving best LOF.
        best_bx : np.ndarray
            The updated basis column for this knot.
        """
        raise NotImplementedError
