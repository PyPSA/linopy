#!/usr/bin/env python3
"""
Created on Mon Oct 10 14:21:23 2022.

@author: fabian
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import EQUAL, GREATER_EQUAL, Model
from linopy.matrices import MatrixAccessor


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


@pytest.mark.parametrize("freeze", [False, True])
def test_matrices_drops_explicit_zeros(freeze: bool) -> None:
    # https://github.com/PyPSA/linopy/issues/814
    # Expressions that broadcast against a dense coordinate store one coefficient
    # per pair, most of them structurally zero. Those must not reach A, whose
    # stored nnz drives the direct-solver handoff (e.g. highspy.addRows).
    m = Model()

    x = m.add_variables(coords=[range(4)], name="x")
    coeff = xr.DataArray(
        np.eye(4), dims=["j", "i"], coords={"j": range(4), "i": range(4)}
    )
    m.add_constraints(
        (coeff * x.rename(dim_0="i")).sum("i") <= 1, name="c", freeze=freeze
    )

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


def _sparse_model() -> Model:
    m = Model(sparse=True)
    i = pd.RangeIndex(4, name="i")
    x = m.add_variables(0, 1, coords=[i], name="x")
    n = m.add_variables(0, 5, coords=[i], name="n", integer=True)
    m.add_constraints(x + n >= 1, name="c")
    m.add_constraints(x - n <= 1, name="d")
    m.add_objective((x + 2 * n).sum())
    return m


MUTATIONS = {
    "add_variables": lambda m: m.add_variables(coords=[m.variables["x"].indexes["i"]]),
    "add_constraints": lambda m: m.add_constraints(m.variables["x"] <= 0.5),
    "remove_constraints": lambda m: m.remove_constraints("c"),
    "update_bounds": lambda m: m.variables["x"].update(upper=2),
    "variable_scaling": lambda m: setattr(m.variables["x"], "scaling", 2),
    "relax": lambda m: m.variables["n"].relax(),
    "objective": lambda m: m.add_objective(-m.variables["x"].sum(), overwrite=True),
    "objective_scaling": lambda m: setattr(m.objective, "scaling", 2),
    "objective_coeffs": lambda m: setattr(
        m.objective.expression, "coeffs", m.objective.expression.coeffs * 2
    ),
    "objective_vars": lambda m: setattr(
        m.objective.expression, "vars", m.objective.expression.vars.roll(_term=1)
    ),
}


@pytest.mark.v1
def test_matrices_cached_for_frozen_model() -> None:
    m = _sparse_model()
    assert m.matrices is m.matrices


@pytest.mark.v1
def test_matrices_not_cached_with_mutable_constraint() -> None:
    m = _sparse_model()
    m.add_constraints(m.variables["x"] >= 0, name="mutable", freeze=False)
    assert m.matrices is not m.matrices


@pytest.mark.v1
@pytest.mark.parametrize("mutate", MUTATIONS.values(), ids=MUTATIONS.keys())
def test_matrices_cache_invalidated(mutate: Callable[[Model], Any]) -> None:
    m = _sparse_model()
    cached = m.matrices
    mutate(m)
    fresh = MatrixAccessor(m)
    assert m.matrices is not cached
    assert m.matrices is m.matrices
    for attr in ("vlabels", "clabels", "lb", "ub", "vtypes", "b", "sense", "c"):
        np.testing.assert_array_equal(getattr(m.matrices, attr), getattr(fresh, attr))
    assert m.matrices.A is not None and fresh.A is not None
    np.testing.assert_array_equal(m.matrices.A.toarray(), fresh.A.toarray())


@pytest.mark.v1
def test_matrices_cache_refreshes_solution() -> None:
    m = _sparse_model()
    cached = m.matrices
    m._mock_solve()
    assert m.matrices is not cached
    sol, dual = m.matrices.sol, m.matrices.dual
    for var in m.variables.data.values():
        var.solution = var.solution + 1
    for con in m.constraints.data.values():
        con.dual = con.dual + 1
    np.testing.assert_array_equal(m.matrices.sol, sol + 1)
    np.testing.assert_array_equal(m.matrices.dual, dual + 1)
