#!/usr/bin/env python3
"""
Created on Mon Oct 10 14:21:23 2022.

@author: fabian
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import EQUAL, GREATER_EQUAL, Model


def test_basic_matrices() -> None:
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(name="y")

    m.add_constraints(1 * x + 10 * y, EQUAL, 0)

    obj = (10 * x + 5 * y).sum()
    m.add_objective(obj)

    assert m.matrices.A is not None
    assert m.matrices.A.shape == (*m.matrices.clabels.shape, *m.matrices.vlabels.shape)
    assert m.matrices.clabels.shape == m.matrices.sense.shape
    assert m.matrices.vlabels.shape == m.matrices.ub.shape
    assert m.matrices.vlabels.shape == m.matrices.lb.shape


def test_basic_matrices_masked() -> None:
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x")
    mask = pd.Series([True] * 8 + [False, False])
    y = m.add_variables(lower, name="y", mask=mask)

    m.add_constraints(x + y, GREATER_EQUAL, 10)

    m.add_constraints(y, GREATER_EQUAL, 0)

    m.add_objective(2 * x + y)

    assert m.matrices.A is not None
    assert m.matrices.A.shape == (*m.matrices.clabels.shape, *m.matrices.vlabels.shape)
    assert m.matrices.clabels.shape == m.matrices.sense.shape
    assert m.matrices.vlabels.shape == m.matrices.ub.shape
    assert m.matrices.vlabels.shape == m.matrices.lb.shape


def test_matrices_duplicated_variables() -> None:
    m = Model()

    x = m.add_variables(pd.Series([0, 0]), 1, name="x")
    y = m.add_variables(4, pd.Series([8, 10]), name="y")
    z = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4], [5, 6]]).T, name="z")
    m.add_constraints(x + x + y + y + z + z == 0)

    assert m.matrices.A is not None

    A = m.matrices.A.todense()
    assert A[0, 0] == 2
    assert np.isin(np.unique(np.array(A)), [0.0, 2.0]).all()


def test_matrices_drops_explicit_zeros() -> None:
    # https://github.com/PyPSA/linopy/issues/814
    # Expressions that broadcast against a dense coordinate store one coefficient
    # per pair, most of them structurally zero. Those must not reach A, whose
    # stored nnz drives the direct-solver handoff (e.g. highspy.addRows).
    m = Model()

    x = m.add_variables(coords=[range(4)], name="x")
    coeff = xr.DataArray(
        np.eye(4), dims=["j", "i"], coords={"j": range(4), "i": range(4)}
    )
    m.add_constraints((coeff * x.rename(dim_0="i")).sum("i") <= 1, name="c")

    A = m.matrices.A
    assert A is not None
    # 4 structural nonzeros on the diagonal; the 12 broadcast zeros are dropped.
    assert A.nnz == 4
    assert not (A.data == 0).any()
    assert np.array_equal(A.todense(), np.eye(4))


def test_matrices_float_c() -> None:
    # https://github.com/PyPSA/linopy/issues/200
    m = Model()

    x = m.add_variables(pd.Series([0, 0]), 1, name="x")
    m.add_objective(x * 1.5)

    c = m.matrices.c
    assert np.all(c == np.array([1.5, 1.5]))


@pytest.mark.parametrize("freeze", [False, True])
def test_matrices_scaled_masked_mixed_signs(freeze: bool) -> None:
    m = Model()
    i = pd.RangeIndex(3, name="i")
    x = m.add_variables(0, 4, coords=[i], name="x", scaling=[1.0, 2.0, 4.0])
    y = m.add_variables(coords=[i], name="y")
    z = m.add_variables(coords=[i], name="z", binary=True)
    sign = xr.DataArray(["<=", ">=", "="], coords=[i])
    m.add_constraints(
        2 * x + y - y,
        sign,
        xr.DataArray([1.0, 2.0, 3.0], coords=[i]),
        name="c",
        scaling=xr.DataArray([10.0, 1.0, 5.0], coords=[i]),
        mask=xr.DataArray([True, False, True], coords=[i]),
        freeze=freeze,
    )
    m.add_constraints(x + 3 * y >= 1, name="d", freeze=not freeze)
    m.add_indicator_constraints(z, 1, x <= 2, name="ind")
    M = m.matrices

    assert M.A is not None and M.indicator_A is not None
    col_scaling = np.array([1.0, 0.5, 0.25])
    expected = np.zeros((5, 9))
    expected[0, 0], expected[1, 2] = 20.0, 2.5
    expected[2:, :3] = np.diag(col_scaling)
    expected[2:, 3:6] = 3 * np.eye(3)
    np.testing.assert_array_equal(M.A.toarray(), expected)
    assert M.A.nnz == 8
    np.testing.assert_array_equal(M.b, [10.0, 15.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(M.sense, ["<", "=", ">", ">", ">"])
    np.testing.assert_array_equal(M.clabels, [0, 2, 3, 4, 5])
    expected_ind = np.zeros((3, 9))
    expected_ind[:, :3] = np.diag(col_scaling)
    np.testing.assert_array_equal(M.indicator_A.toarray(), expected_ind)
    np.testing.assert_array_equal(M.indicator_b, [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(M.indicator_binvar, [6, 7, 8])
    np.testing.assert_array_equal(M.lb[:3], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(M.ub[:3], [4.0, 8.0, 16.0])
