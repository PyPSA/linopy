#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 09:03:35 2021

@author: fabian
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.testing import assert_allclose, assert_equal

from linopy import LinearExpression, Model, available_solvers, read_netcdf
from linopy.io import to_int_str


def test_str_arrays():
    m = Model()

    x = m.add_variables(4, pd.Series([8, 10]))
    y = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4], [5, 6]]).T)

    da = to_int_str(x)
    assert da.dtype.type is np.str_


def test_str_arrays_chunked():
    m = Model(chunk=-1)

    x = m.add_variables(4, pd.Series([8, 10]))
    y = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4], [5, 6]]).T)

    da = to_int_str(y).compute()
    assert da.dtype.type is np.str_


def test_str_arrays_with_nans():
    m = Model()

    x = m.add_variables(4, pd.Series([8, 10]), name="x")
    # now expand the second dimension, expended values of x will be nan
    y = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4], [5, 6]]), name="y")
    assert m["x"][-1].item() == -1

    da = to_int_str(m["x"])
    assert da.dtype.type is np.str_


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
