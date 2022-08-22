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

import linopy
from linopy import Model


def test_variable_getter():
    m = Model()
    x = m.add_variables(coords=[range(10)])

    assert isinstance(x[0], linopy.variables.ScalarVariable)

    with pytest.raises(AssertionError):
        x[0, 0]

    with pytest.raises(AssertionError):
        x[0:5]

    with pytest.raises(AssertionError):
        x[[1, 2, 3]]


def test_variable_repr():
    m = Model()
    m.variables.__repr__()

    x = m.add_variables()
    x.__repr__()
    x._repr_html_()

    m.variables.__repr__()

    y = m.add_variables(coords=[pd.Index([1, 2, 3], name="time")], name="y")
    y.__repr__()
    y._repr_html_()

    m.variables.__repr__()


def test_variable_bound_accessor():
    m = Model()
    x = m.add_variables(0, 10)
    assert x.upper.item() == 10
    assert x.lower.item() == 0


def test_variable_modification():
    m = Model()
    x = m.add_variables(0, 10)
    x.upper = 20
    assert x.upper.item() == 20

    x.lower = 8
    assert x.lower == 8


def test_variable_modification_M():
    m = Model()
    lower = pd.Series(0, index=range(10))
    upper = pd.Series(range(10, 20), index=range(10))
    x = m.add_variables(lower, upper)

    new_upper = pd.Series(range(25, 35), index=range(10))
    x.upper = new_upper
    assert isinstance(x.upper, xr.DataArray)
    assert (x.upper == new_upper).all()

    new_lower = pd.Series(range(15, 25), index=range(10))
    x.lower = new_lower
    assert isinstance(x.lower, xr.DataArray)
    assert (x.lower == new_lower).all()


def test_variable_sum():
    m = Model()
    x = m.add_variables(coords=[range(10)])
    res = x.sum()
    assert res.nterm == 10


def test_nvars():
    m = Model()
    m.add_variables(coords=[range(10)])
    assert m.variables.nvars == 10

    mask = pd.Series([True] * 5 + [False] * 5)
    m.add_variables(coords=[range(10)], mask=mask)
    assert m.variables.nvars == 15


def test_variable_where():
    m = Model()
    x = m.add_variables(coords=[range(10)])
    x = x.where([True] * 4 + [False] * 6)
    assert isinstance(x, linopy.variables.Variable)
    assert x.loc[9].item() == -1


def test_variable_shift():
    m = Model()
    x = m.add_variables(coords=[range(10)])
    x = x.shift(dim_0=3)
    assert isinstance(x, linopy.variables.Variable)
    assert x.loc[0].item() == -1


def test_variable_sanitize():
    m = Model()
    x = m.add_variables(coords=[range(10)])
    # convert intentionally to float with nans
    x = x.where([True] * 4 + [False] * 6, np.nan)
    x = x.sanitize()
    assert isinstance(x, linopy.variables.Variable)
    assert x.loc[9].item() == -1


def test_variable_type_preservation():
    m = Model()
    x = m.add_variables(coords=[range(10)])

    assert isinstance(x.bfill("dim_0"), linopy.variables.Variable)
    assert isinstance(x.broadcast_like(x.to_array()), linopy.variables.Variable)
    assert isinstance(x.clip(max=20), linopy.variables.Variable)
    assert isinstance(x.ffill("dim_0"), linopy.variables.Variable)
    assert isinstance(x.fillna(-1), linopy.variables.Variable)


def test_variable_getter_without_model():
    data = xr.DataArray(range(10)).rename("var")
    v = linopy.variables.Variable(data)

    with pytest.raises(AttributeError):
        v.upper
    with pytest.raises(AttributeError):
        v.lower


def test_get_name_by_label():
    m = Model()
    m.add_variables(coords=[range(10)], name="x")
    m.add_variables(coords=[range(10)], name="asd")

    assert m.variables.get_name_by_label(4) == "x"
    assert m.variables.get_name_by_label(14) == "asd"

    with pytest.raises(ValueError):
        m.variables.get_name_by_label(30)

    with pytest.raises(ValueError):
        m.variables.get_name_by_label("asd")
