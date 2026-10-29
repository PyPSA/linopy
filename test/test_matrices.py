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
from linopy.matrices import Matrices


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


def test_matrices_drops_explicit_zeros(backend: Model) -> None:
    # https://github.com/PyPSA/linopy/issues/814
    # Expressions that broadcast against a dense coordinate store one coefficient
    # per pair, most of them structurally zero. Those must not reach A, whose
    # stored nnz drives the direct-solver handoff (e.g. highspy.addRows).
    m = backend

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


def test_matrices_sol_aligned_with_vlabels() -> None:
    m = Model()
    i = pd.RangeIndex(3, name="i")
    x = m.add_variables(coords=[i], name="x", mask=pd.Series([True, False, True], i))
    y = m.add_variables(coords=[i], name="y")
    m.add_constraints(x + y >= 0, name="c")
    with pytest.raises(ValueError, match="not optimized"):
        m.matrices.sol
    m._mock_solve()
    x.solution = xr.DataArray([1.0, np.nan, 3.0], coords=[i])
    y.solution = xr.DataArray([4.0, 5.0, 6.0], coords=[i])
    M = m.matrices
    np.testing.assert_array_equal(M.vlabels, [0, 2, 3, 4, 5])
    np.testing.assert_array_equal(M.sol, [1.0, 3.0, 4.0, 5.0, 6.0])


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_matrices_scaled_masked_mixed_signs(sparse: bool) -> None:
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
    )
    d = m.add_constraints(x + 3 * y >= 1, name="d")
    (m.constraints["c"] if sparse else d).to_sparse()
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


@pytest.mark.parametrize(
    ("rows", "cols"),
    [([0, 2], [1, 5, 6]), ([4, 1, 3], [7, 0, 2]), ([], [3])],
    ids=["subset", "reordered", "no-rows"],
)
def test_matrices_take(rows: list[int], cols: list[int]) -> None:
    m = Model()
    i = pd.RangeIndex(4, name="i")
    x = m.add_variables(0, 4, coords=[i], name="x", scaling=[1.0, 2.0, 4.0, 1.0])
    y = m.add_variables(coords=[i], name="y", integer=True)
    m.add_constraints(x + 2 * y >= 1, name="c", scaling=[1.0, 3.0, 1.0, 2.0])
    m.add_constraints(x.sum() <= 3, name="d")
    m.add_objective((x * x).sum() + 3 * y.sum(), sense="max")
    M = m.matrices
    r, c = np.array(rows, dtype=np.intp), np.array(cols, dtype=np.intp)

    taken = M.take(r, c)

    assert type(taken) is Matrices
    assert taken.A is not None and M.A is not None
    assert taken.Q is not None and M.Q is not None
    np.testing.assert_array_equal(taken.A.toarray(), M.A.toarray()[np.ix_(r, c)])
    np.testing.assert_array_equal(taken.Q.toarray(), M.Q.toarray()[np.ix_(c, c)])
    for column_side, full in [
        (taken.vlabels, M.vlabels),
        (taken.c, M.c),
        (taken.lb, M.lb),
        (taken.ub, M.ub),
        (taken.vtypes, M.vtypes),
        (taken.var_scaling, M.var_scaling),
    ]:
        np.testing.assert_array_equal(column_side, full[c])
    for row_side, full in [
        (taken.clabels, M.clabels),
        (taken.b, M.b),
        (taken.sense, M.sense),
        (taken.con_scaling, M.con_scaling),
    ]:
        np.testing.assert_array_equal(row_side, full[r])
    assert (taken.objective_scaling, taken.objective_sense) == (1.0, "max")


def test_matrices_take_rejects_indicator_constraints() -> None:
    m = Model()
    x = m.add_variables(0, 4, coords=[range(2)], name="x")
    z = m.add_variables(coords=[range(2)], name="z", binary=True)
    m.add_constraints(x >= 0, name="c")
    m.add_indicator_constraints(z, 1, x <= 2, name="ind")
    rows, cols = np.arange(1), np.arange(2)

    with pytest.raises(NotImplementedError, match="indicator"):
        m.matrices.take(rows, cols)
