#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:23:13 2021.

@author: fabulous
"""

from pathlib import Path
from tempfile import gettempdir

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model

# Test model functions

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


def test_scalar_variable_assignment():
    m = Model()
    m.add_variables(-5, 10, name="x")
    assert "x" in m.variables.labels


def test_scalar_variable_assignment_default():
    m = Model()
    m.add_variables(name="x")
    assert m.variables.lower.x.item() == -np.inf
    assert m.variables.upper.x.item() == np.inf


def test_variable_getitem():
    m = Model()
    x = m.add_variables(name="x")
    assert m["x"].values == x.values


def test_scalar_variable_name_counter():
    m = Model()
    m.add_variables()
    m.add_variables()
    assert "var0" in m.variables
    assert "var1" in m.variables


def test_array_variable_assignment():
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    m.add_variables(lower, upper, name="x")
    assert m.variables.labels.x.shape == target_shape


def test_array_variable_assignment_broadcasted():
    m = Model()
    # setting only one dimension, the other has to be broadcasted
    lower = xr.DataArray(np.zeros((10)), coords=[range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    m.add_variables(lower, upper, name="x")
    assert m.variables.labels.x.shape == target_shape


def test_array_variable_assignment_no_coords():
    # setting bounds without explicit coords
    m = Model()
    lower = xr.DataArray(np.zeros((10)))
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    m.add_variables(lower, upper, name="x")
    assert m.variables.labels.x.shape == target_shape


def test_array_variable_assignment_pd_index():
    # setting bounds with pandas index
    m = Model()
    lower = xr.DataArray(np.zeros((10)), coords=[pd.Index(range(10))])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    m.add_variables(lower, upper, name="x")
    assert m.variables.labels.x.shape == target_shape


def test_array_variable_assignment_by_coords():
    # setting bounds with scalar and coords
    m = Model()
    lower = 0
    upper = 1
    coords = [pd.Index(range(10)), pd.Index(range(10))]
    m.add_variables(lower, upper, coords=coords, name="x")
    assert m.variables.labels.x.shape == target_shape


def test_array_variable_assignment_with_dataframes():
    # setting bounds with pd.DataFrames
    m = Model()
    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.DataFrame(np.ones((10, 10)))
    m.add_variables(lower, upper, name="x")
    assert m.variables.labels.x.shape == target_shape


def test_array_variable_assignment_with_dataframe_and_series():
    # setting bounds with one pd.DataFrame and one pd.Series
    m = Model()
    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.Series(np.ones((10)))
    m.add_variables(lower, upper, name="x")
    assert m.variables.labels.x.shape == target_shape


def test_array_variable_assignment_chunked():
    # setting bounds with one pd.DataFrame and one pd.Series
    m = Model(chunk=5)
    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.Series(np.ones((10)))
    m.add_variables(lower, upper, name="x")
    assert m.variables.labels.x.shape == target_shape
    assert isinstance(m.variables.labels.x.data, dask.array.core.Array)


def test_array_variable_assignment_different_coords():
    # set a variable with different set of coordinates, this should be properly
    # merged
    m = Model()
    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.Series(np.ones((10)))
    m.add_variables(lower, upper, name="x")

    lower = pd.DataFrame(np.zeros((20, 10)))
    upper = pd.Series(np.ones((20)))
    m.add_variables(lower, upper, name="y")
    assert m.variables.labels.y.shape == (20, 10)
    # x should now be aligned to new coords and contain 100 nans
    assert m.variables.labels.x.shape == (20, 10)
    assert (m.variables.labels.x != -1).sum() == 100


def test_wrong_variable_assignment_non_broadcastable():
    # setting with scalar and list
    m = Model()
    with pytest.raises(ValueError):
        m.add_variables(0, [1, 2])


def test_wrong_variable_assignment_repeated():
    # repeated variable assignment is forbidden
    m = Model()
    m.add_variables(name="x")
    with pytest.raises(ValueError):
        m.add_variables(name="x")


def test_masked_variables():
    m = Model()

    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.Series(np.ones((10)))
    mask = pd.Series([True] * 5 + [False] * 5)
    m.add_variables(lower, upper, mask=mask)
    assert m.variables.labels.var0[-1, -1].item() == -1


def test_variable_merging():
    """
    Test the merger of a variables with same dimension name but with different
    lengths.

    Missing values should be filled up with -1.
    """
    m = Model()

    upper = pd.Series(np.ones((10)))
    m.add_variables(upper)

    upper = pd.Series(np.ones((12)))
    m.add_variables(upper)
    assert m.variables.labels.var0[-1].item() == -1


def test_binary_assigment():
    m = Model()

    coords = [pd.Index(range(10)), pd.Index(range(10))]
    m.add_variables(coords=coords, binary=True)

    assert m.variables.labels.var0.shape == target_shape


def test_linexpr():
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(name="y")

    # select a variable by a scalar and broadcast if over another variable array
    expr = m.linexpr((1, "x"), (10, "y"))
    target = 1 * x + 10 * y
    # assert (expr._term == ['x', 'y']).all()
    assert (expr.to_dataset() == target.to_dataset()).all().to_array().all()


def test_constraint_assignment():
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper)
    y = m.add_variables()

    m.add_constraints(1 * x + 10 * y, "=", 0)

    for attr in m.constraints.dataset_attrs:
        assert "con0" in getattr(m.constraints, attr)

    assert m.constraints.labels.con0.shape == (10, 10)
    assert m.constraints.labels.con0.dtype == int
    assert m.constraints.coeffs.con0.dtype in (int, float)
    assert m.constraints.vars.con0.dtype in (int, float)
    assert m.constraints.rhs.con0.dtype in (int, float)


def test_anonymous_constraint_assignment():
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper)
    y = m.add_variables()
    con = 1 * x + 10 * y == 0
    m.add_constraints(con)

    for attr in m.constraints.dataset_attrs:
        assert "con0" in getattr(m.constraints, attr)

    assert m.constraints.labels.con0.shape == (10, 10)
    assert m.constraints.labels.con0.dtype == int
    assert m.constraints.coeffs.con0.dtype in (int, float)
    assert m.constraints.vars.con0.dtype in (int, float)
    assert m.constraints.rhs.con0.dtype in (int, float)


def test_coefficient_range():
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper)
    y = m.add_variables()

    m.add_constraints(1 * x + 10 * y, "=", 0)
    assert m.coefficientrange["min"].con0 == 1
    assert m.coefficientrange["max"].con0 == 10


def test_constraint_assignment_with_tuples():
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper)
    y = m.add_variables()

    m.add_constraints([(1, x), (10, y)], "=", 0, name="c")
    for attr in m.constraints.dataset_attrs:
        assert "c" in getattr(m.constraints, attr)
    assert m.constraints.labels.c.shape == (10, 10)


def test_constraint_assignment_chunked():
    # setting bounds with one pd.DataFrame and one pd.Series
    m = Model(chunk=5)
    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.Series(np.ones((10)))
    x = m.add_variables(lower, upper)
    m.add_constraints(x, ">=", 0, name="c")
    assert m.constraints.coeffs.c.data.shape == target_shape + (1,)
    assert isinstance(m.constraints.coeffs.c.data, dask.array.core.Array)


def test_wrong_constraint_assignment_wrong_sign():
    m = Model()
    x = m.add_variables()
    with pytest.raises(ValueError):
        m.add_constraints(x, "==", 0)


def test_wrong_constraint_assignment_repeated():
    # repeated variable assignment is forbidden
    m = Model()
    x = m.add_variables()
    m.add_constraints(x, "<=", 0, name="con")
    with pytest.raises(ValueError):
        m.add_constraints(x, "<=", 0, name="con")


def test_masked_constraints():
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper)
    y = m.add_variables()

    mask = pd.Series([True] * 5 + [False] * 5)
    m.add_constraints(1 * x + 10 * y, "=", 0, mask=mask)
    assert (m.constraints.labels.con0[0:5, :] != -1).all()
    assert (m.constraints.labels.con0[5:10, :] == -1).all()


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

    m.add_constraints(1 * x + 10 * y, "=", 0)

    obj = (10 * x + 5 * y).sum()
    m.add_objective(obj)

    m.remove_variables("x")
    for attr in m.constraints.dataset_attrs:
        assert "x" not in getattr(m.constraints, attr)

    assert "con0" not in m.constraints.labels

    assert not m.objective.vars.isin(x).any()


def test_remove_constraint():
    m = Model()

    x = m.add_variables()
    m.add_constraints(x, "=", 0, name="x")
    m.remove_constraints("x")
    assert not len(m.constraints.labels)
