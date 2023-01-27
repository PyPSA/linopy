#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test function defined in the Model class.
"""

from pathlib import Path
from tempfile import gettempdir

import numpy as np
import pytest
import xarray as xr

from linopy import EQUAL, Model

target_shape = (10, 10)


def test_model_repr():
    m = Model()
    m.__repr__()


def test_model_force_dims_names():
    m = Model(force_dim_names=True)
    with pytest.raises(ValueError):
        m.add_variables([-5], [10])


def test_model_solver_dir():
    d = gettempdir()
    m = Model(solver_dir=d)
    assert m.solver_dir == Path(d)


def test_model_variable_getitem():
    m = Model()
    x = m.add_variables(name="x")
    assert m["x"].values == x.values


def test_coefficient_range():
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper)
    y = m.add_variables()

    m.add_constraints(1 * x + 10 * y, EQUAL, 0)
    assert m.coefficientrange["min"].con0 == 1
    assert m.coefficientrange["max"].con0 == 10


def test_objective():
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(lower, upper, name="y")

    obj = (10 * x + 5 * y).sum()
    m.add_objective(obj)
    assert m.objective.vars.size == 200

    # test overwriting
    obj = (2 * x).sum()
    m.add_objective(obj, overwrite=True)

    # test Tuple
    obj = [(2, x)]
    m.add_objective(obj, overwrite=True)

    # test objective range
    assert m.objectiverange.min() == 2
    assert m.objectiverange.max() == 2


def test_remove_variable():
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(name="y")

    m.add_constraints(1 * x + 10 * y, EQUAL, 0)

    obj = (10 * x + 5 * y).sum()
    m.add_objective(obj)

    m.remove_variables("x")
    for attr in m.constraints.dataset_attrs:
        assert "x" not in getattr(m.constraints, attr)

    assert "con0" not in m.constraints.labels

    assert not m.objective.vars.isin(x.labels).any()


def test_remove_constraint():
    m = Model()

    x = m.add_variables()
    m.add_constraints(x, EQUAL, 0, name="x")
    m.remove_constraints("x")
    assert not len(m.constraints.labels)


def test_removed_eval_funcs():
    m = Model()

    with pytest.raises(NotImplementedError):
        m.vareval("")

    with pytest.raises(NotImplementedError):
        m.lineval("")

    with pytest.raises(NotImplementedError):
        m.coneval("")
