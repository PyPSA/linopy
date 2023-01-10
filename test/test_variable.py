#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 22:36:38 2021.

@author: fabian
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.testing import assert_equal

import linopy
from linopy import Model


@pytest.fixture
def m():
    m = Model()
    m.add_variables(coords=[pd.RangeIndex(10, name="first")], name="x")
    m.add_variables(coords=[pd.Index([1, 2, 3], name="second")], name="y")
    m.add_variables(0, 10, name="z")
    return m


@pytest.fixture
def x(m):
    return m.variables["x"]


@pytest.fixture
def z(m):
    return m.variables["z"]


def test_variable_repr(x):
    x.__repr__()


def test_wrong_variable_init(m, x):
    with pytest.raises(ValueError):
        linopy.Variable(x.labels.values, m)

    with pytest.raises(ValueError):
        linopy.Variable(x.labels, None)


def test_variable_getter(x, z):
    assert isinstance(x[0], linopy.variables.ScalarVariable)

    assert isinstance(z[0], linopy.variables.ScalarVariable)

    with pytest.raises(AssertionError):
        x[0, 0]

    with pytest.raises(AssertionError):
        x[0:5]

    with pytest.raises(AssertionError):
        x[[1, 2, 3]]


def test_variable_loc(x):
    assert isinstance(x.loc[[1, 2, 3]], linopy.Variable)


def test_variable_sel(x):
    assert isinstance(x.sel(first=[1, 2, 3]), linopy.Variable)


def test_variable_isel(x):
    assert isinstance(x.isel(first=[1, 2, 3]), linopy.Variable)
    assert_equal(
        x.isel(first=[0, 1]).labels,
        x.sel(first=[0, 1]).labels,
    )


def test_variable_upper_getter(z):
    assert z.upper.item() == 10


def test_variable_lower_getter(z):
    assert z.lower.item() == 0


def test_variable_upper_setter(z):
    z.upper = 20
    assert z.upper.item() == 20


def test_variable_lower_setter(z):
    z.lower = 8
    assert z.lower == 8


def test_variable_upper_setter_with_array(x):
    idx = pd.RangeIndex(10, name="first")
    upper = pd.Series(range(25, 35), index=idx)
    x.upper = upper
    assert isinstance(x.upper, xr.DataArray)
    assert (x.upper == upper).all()


def test_variable_upper_setter_with_array_invalid_dim(x):
    with pytest.raises(ValueError):
        upper = pd.Series(range(25, 35))
        x.upper = upper


def test_variable_lower_setter_with_array(x):
    idx = pd.RangeIndex(10, name="first")
    lower = pd.Series(range(15, 25), index=idx)
    x.lower = lower
    assert isinstance(x.lower, xr.DataArray)
    assert (x.lower == lower).all()


def test_variable_lower_setter_with_array_invalid_dim(x):
    with pytest.raises(ValueError):
        lower = pd.Series(range(15, 25))
        x.lower = lower


def test_variable_sum(x):
    res = x.sum()
    assert res.nterm == 10


def test_variable_where(x):
    x = x.where([True] * 4 + [False] * 6)
    assert isinstance(x, linopy.variables.Variable)
    assert x.values[9] == -1


def test_variable_shift(x):
    x = x.shift(first=3)
    assert isinstance(x, linopy.variables.Variable)
    assert x.values[0] == -1


def test_variable_bfill(x):
    result = x.bfill("first")
    assert isinstance(result, linopy.variables.Variable)


def test_variable_broadcast_like(x):
    result = x.broadcast_like(x.labels)
    assert isinstance(result, linopy.variables.Variable)


def test_variable_ffill(x):
    result = x.ffill("first")
    assert isinstance(result, linopy.variables.Variable)


def test_variable_fillna(x):
    result = x.fillna(-1)
    assert isinstance(result, linopy.variables.Variable)


def test_variable_sanitize(x):
    # convert intentionally to float with nans
    x = x.where([True] * 4 + [False] * 6, np.nan)
    x = x.sanitize()
    assert isinstance(x, linopy.variables.Variable)
    assert x.values[9] == -1
