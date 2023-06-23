#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:23:13 2021.

@author: fabulous
"""

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import EQUAL, GREATER_EQUAL, LESS_EQUAL, Model
from linopy.testing import assert_conequal

# Test model functions


def test_constraint_assignment():
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(name="y")

    con0 = m.add_constraints(1 * x + 10 * y, EQUAL, 0)

    for attr in m.constraints.dataset_attrs:
        assert "con0" in getattr(m.constraints, attr)

    assert m.constraints.labels.con0.shape == (10, 10)
    assert m.constraints.labels.con0.dtype == int
    assert m.constraints.coeffs.con0.dtype in (int, float)
    assert m.constraints.vars.con0.dtype in (int, float)
    assert m.constraints.rhs.con0.dtype in (int, float)

    assert_conequal(m.constraints.con0, con0)


def test_anonymous_constraint_assignment():
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(name="y")
    con = 1 * x + 10 * y == 0
    m.add_constraints(con)

    for attr in m.constraints.dataset_attrs:
        assert "con0" in getattr(m.constraints, attr)

    assert m.constraints.labels.con0.shape == (10, 10)
    assert m.constraints.labels.con0.dtype == int
    assert m.constraints.coeffs.con0.dtype in (int, float)
    assert m.constraints.vars.con0.dtype in (int, float)
    assert m.constraints.rhs.con0.dtype in (int, float)


def test_constraint_assignment_with_tuples():
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper)
    y = m.add_variables()

    m.add_constraints([(1, x), (10, y)], EQUAL, 0, name="c")
    for attr in m.constraints.dataset_attrs:
        assert "c" in getattr(m.constraints, attr)
    assert m.constraints.labels.c.shape == (10, 10)


def test_constraint_assignment_chunked():
    # setting bounds with one pd.DataFrame and one pd.Series
    m = Model(chunk=5)
    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.Series(np.ones((10)))
    x = m.add_variables(lower, upper)
    m.add_constraints(x, GREATER_EQUAL, 0, name="c")
    assert m.constraints.coeffs.c.data.shape == (
        10,
        10,
        1,
    )
    assert isinstance(m.constraints.coeffs.c.data, dask.array.core.Array)


def test_constraint_assignment_with_reindex():
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(name="y")

    m.add_constraints(1 * x + 10 * y, EQUAL, 0)

    shuffled_coords = [2, 1, 3, 4, 6, 5, 7, 9, 8, 0]

    con = x.loc[shuffled_coords] + y >= 10
    assert (con.coords["dim_0"].values == shuffled_coords).all()


def test_wrong_constraint_assignment_repeated():
    # repeated variable assignment is forbidden
    m = Model()
    x = m.add_variables()
    m.add_constraints(x, LESS_EQUAL, 0, name="con")
    with pytest.raises(ValueError):
        m.add_constraints(x, LESS_EQUAL, 0, name="con")


def test_masked_constraints():
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper)
    y = m.add_variables()

    mask = pd.Series([True] * 5 + [False] * 5)
    m.add_constraints(1 * x + 10 * y, EQUAL, 0, mask=mask)
    assert (m.constraints.labels.con0[0:5, :] != -1).all()
    assert (m.constraints.labels.con0[5:10, :] == -1).all()


def test_constraints_flat():
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper)
    y = m.add_variables()

    assert isinstance(m.constraints.flat, pd.DataFrame)
    assert m.constraints.flat.empty
    assert m.constraints.to_matrix() is None

    m.add_constraints(1 * x + 10 * y, EQUAL, 0)
    m.add_constraints(1 * x + 10 * y, LESS_EQUAL, 0)
    m.add_constraints(1 * x + 10 * y, GREATER_EQUAL, 0)

    assert isinstance(m.constraints.flat, pd.DataFrame)
    assert not m.constraints.flat.empty
