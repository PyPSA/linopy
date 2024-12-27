#!/usr/bin/env python3
"""
Test function defined in the Model class.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import gettempdir

import numpy as np
import pytest
import xarray as xr

from linopy import EQUAL, Model
from linopy.testing import assert_model_equal

target_shape: tuple[int, int] = (10, 10)


def test_model_repr() -> None:
    m: Model = Model()
    m.__repr__()


def test_model_force_dims_names() -> None:
    m: Model = Model(force_dim_names=True)
    with pytest.raises(ValueError):
        m.add_variables([-5], [10])


def test_model_solver_dir() -> None:
    d: str = gettempdir()
    m: Model = Model(solver_dir=d)
    assert m.solver_dir == Path(d)


def test_model_variable_getitem() -> None:
    m = Model()
    x = m.add_variables(name="x")
    assert m["x"].labels == x.labels


def test_coefficient_range() -> None:
    m: Model = Model()

    lower: xr.DataArray = xr.DataArray(
        np.zeros((10, 10)), coords=[range(10), range(10)]
    )
    upper: xr.DataArray = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper)
    y = m.add_variables()

    m.add_constraints(1 * x + 10 * y, EQUAL, 0)
    assert m.coefficientrange["min"].con0 == 1
    assert m.coefficientrange["max"].con0 == 10


def test_objective() -> None:
    m: Model = Model()

    lower: xr.DataArray = xr.DataArray(
        np.zeros((10, 10)), coords=[range(10), range(10)]
    )
    upper: xr.DataArray = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(lower, upper, name="y")

    obj1 = (10 * x + 5 * y).sum()
    m.add_objective(obj1)
    assert m.objective.vars.size == 200

    # test overwriting
    obj2 = (2 * x).sum()
    m.add_objective(obj2, overwrite=True)

    # test Tuple
    obj3 = [(2, x)]
    m.add_objective(obj3, overwrite=True)

    # test objective range
    assert m.objectiverange.min() == 2
    assert m.objectiverange.max() == 2

    # test objective with constant which is not supported
    with pytest.raises(ValueError):
        m.objective = m.objective + 3


def test_remove_variable() -> None:
    m: Model = Model()

    lower: xr.DataArray = xr.DataArray(
        np.zeros((10, 10)), coords=[range(10), range(10)]
    )
    upper: xr.DataArray = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(name="y")

    m.add_constraints(1 * x + 10 * y, EQUAL, 0)

    obj = (10 * x + 5 * y).sum()
    m.add_objective(obj)

    assert "x" in m.variables

    m.remove_variables("x")
    assert "x" not in m.variables

    assert not m.constraints.con0.vars.isin(x.labels).any()

    assert not m.objective.vars.isin(x.labels).any()


def test_remove_constraint() -> None:
    m: Model = Model()

    x = m.add_variables()
    m.add_constraints(x, EQUAL, 0, name="x")
    m.remove_constraints("x")
    assert not len(m.constraints.labels)


def test_remove_constraints_with_list() -> None:
    m: Model = Model()

    x = m.add_variables()
    y = m.add_variables()
    m.add_constraints(x, EQUAL, 0, name="constraint_x")
    m.add_constraints(y, EQUAL, 0, name="constraint_y")
    m.remove_constraints(["constraint_x", "constraint_y"])
    assert "constraint_x" not in m.constraints.labels
    assert "constraint_y" not in m.constraints.labels
    assert not len(m.constraints.labels)


def test_remove_objective() -> None:
    m: Model = Model()

    lower: xr.DataArray = xr.DataArray(np.zeros((2, 2)), coords=[range(2), range(2)])
    upper: xr.DataArray = xr.DataArray(np.ones((2, 2)), coords=[range(2), range(2)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(lower, upper, name="y")
    obj = (10 * x + 5 * y).sum()
    m.add_objective(obj)
    m.remove_objective()
    assert not len(m.objective.vars)


def test_assert_model_equal() -> None:
    m: Model = Model()

    lower: xr.DataArray = xr.DataArray(
        np.zeros((10, 10)), coords=[range(10), range(10)]
    )
    upper: xr.DataArray = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(name="y")

    m.add_constraints(1 * x + 10 * y, EQUAL, 0)

    obj = (10 * x + 5 * y).sum()
    m.add_objective(obj)

    assert_model_equal(m, m)
