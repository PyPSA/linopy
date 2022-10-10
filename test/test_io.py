#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 09:03:35 2021.

@author: fabian
"""

import pandas as pd
import pytest
import xarray as xr
from xarray.testing import assert_equal

from linopy import Model, available_solvers, read_netcdf
from linopy.io import float_to_str, int_to_str


def test_str_arrays():
    m = Model()

    x = m.add_variables(4, pd.Series([8, 10]))
    y = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4], [5, 6]]).T)

    da = int_to_str(x.values)
    assert da.dtype == object


def test_str_arrays_chunked():
    m = Model(chunk="auto")

    x = m.add_variables(4, pd.Series([8, 10]))
    y = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4], [5, 6]]).T)

    da = int_to_str(y.compute().values)
    assert da.dtype == object


def test_str_arrays_with_nans():
    m = Model()

    x = m.add_variables(4, pd.Series([8, 10]), name="x")
    # now expand the second dimension, expended values of x will be nan
    y = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4], [5, 6]]), name="y")
    assert m["x"].data[-1].item() == -1

    da = int_to_str(m["x"].values)
    assert da.dtype == object


def test_to_netcdf(tmp_path):
    m = Model()

    x = m.add_variables(4, pd.Series([8, 10]))
    y = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4], [5, 6]]))
    m.add_constraints(x + y, "<=", 10)
    m.add_objective(2 * x + 3 * y)

    fn = tmp_path / "test.nc"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    for k in m.scalar_attrs:
        if k != "objective_value":
            assert getattr(m, k) == getattr(p, k)
    for k in m.dataset_attrs:
        assert_equal(getattr(m, k), getattr(p, k))


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_file(tmp_path):
    import gurobipy

    m = Model()

    x = m.add_variables(4, pd.Series([8, 10]))
    y = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4], [5, 6]]))

    m.add_constraints(x + y, "<=", 10)

    m.add_objective(2 * x + 3 * y)

    fn = tmp_path / "test.lp"
    m.to_file(fn)

    gurobipy.read(str(fn))


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_gurobipy(tmp_path):
    m = Model()

    x = m.add_variables(4, pd.Series([8, 10]))
    y = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4], [5, 6]]))

    m.add_constraints(x + y, "<=", 10)

    m.add_objective(2 * x + 3 * y)

    m.to_gurobipy()


def test_to_blocks(tmp_path):
    m = Model()

    lower = pd.Series(range(20))
    upper = pd.Series(range(30, 50))
    x = m.add_variables(lower, upper)
    y = m.add_variables(lower, upper)

    m.add_constraints(x + y, "<=", 10)

    m.add_objective(2 * x + 3 * y)

    m.blocks = xr.DataArray([1] * 10 + [2] * 10)

    m.to_block_files(tmp_path)
