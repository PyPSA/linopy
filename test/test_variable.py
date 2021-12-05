#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 22:36:38 2021

@author: fabian
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import linopy
from linopy import LinearExpression, Model


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
    assert x.get_upper_bound().item() == 10
    assert x.get_lower_bound().item() == 0


def test_variable_sum():
    m = Model()
    x = m.add_variables(coords=[range(10)])
    res = x.sum()
    assert res.nterm == 10


def test_variable_where():
    m = Model()
    x = m.add_variables(coords=[range(10)])
    x = x.where([True] * 4 + [False] * 6)
    assert isinstance(x, linopy.variables.Variable)
    assert x.loc[9].item() == -1


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


def test_constraint_getter_without_model():

    data = xr.DataArray(range(10)).rename("var")
    v = linopy.variables.Variable(data)

    with pytest.raises(AttributeError):
        v.get_upper_bound()
    with pytest.raises(AttributeError):
        v.get_lower_bound()
        v.get_lower_bound()
