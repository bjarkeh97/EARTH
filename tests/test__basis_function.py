import numpy as np
import pytest
from copy import deepcopy

from npearth._basis_function import BasisFunction, BasisMatrix


def test_basis_function_add_and_return():
    """Test that no variables used in empty basis function"""
    bf = BasisFunction()
    assert bf.return_variables_used() == []


def test_basis_function_hinge():
    xv = np.array([0.0, 1.0, 1.5, 2.0])
    bf = BasisFunction()
    assert np.all(bf.hinge(0, xv, 1.0) == xv)  # s=0 => xv
    assert np.all(bf.hinge(1, xv, 1.0) == np.array([0, 0, 0.5, 1]))  # s=1 => (x - t)_+
    assert np.all(bf.hinge(-1, xv, 1.0) == np.array([1, 0, 0, 0]))  # s=-1 => (x - t)_-


def test_basis_function_evaluate():
    X = np.array([[0.0, 1.0], [2.0, 3.0]])
    bf = BasisFunction()
    # constant case
    assert np.all(bf.evaluate(X) == 1.0)
    bf.add_step(+1, 1, 2)
    assert np.all(bf.evaluate(X) == np.array([0.0, 1.0]))


def test_basis_matrix_add_split():
    X = np.array([[0, 1], [2, 3]])
    bm = BasisMatrix(X)
    assert bm.bx.shape == (2, 1)
    assert len(bm.basis) == 1


def test_basis_matrix_add_split_end():
    X = np.array([[0, 1], [2, 3]])
    bm = BasisMatrix(X)
    bm.add_split_end(0, 0, 1.0)
    # Should now have 3 columns: constant + 2 new (X[:,0] and (X[:,0]-1)_+)
    assert bm.bx.shape[1] == 3
    assert len(bm.basis) == 3
    assert np.all(bm.bx[:, 1:3] == np.array([[0, 0], [2, 1]]))


def test_basis_matrix_evaluate_consistency():
    X = np.array([[0, 1], [2, 3]])
    bm = BasisMatrix(X)
    bf = bm.basis[0]
    bf.add_step(1, 0, 0.5)
    # Manually compute
    manual = np.maximum(X[:, 0] - 0.5, 0)
    assert np.allclose(bf.evaluate(X), manual)
