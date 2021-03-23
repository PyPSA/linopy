#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:06:36 2021

@author: fabian
"""

import xarray as xr
import numpy as np
import pandas as pd
from xarray.testing import assert_equal
from linopy import LinearExpression, Model


m = Model()

x = m.add_variables('x', pd.Series([0,0]), 1)
y = m.add_variables('y', 4, pd.Series([8,10]))
z = m.add_variables('z', 0, pd.DataFrame([[1,2], [3,4], [5,6]]).T)


def test_values():
    expr = m.linexpr((10, 'x'), (1, 'y'))
    target = xr.DataArray([[10, 1], [10, 1]],
                          coords=(('dim_0', [0, 1]), ('term_', [0, 1])))
    assert_equal(expr.coeffs, target)


def test_duplicated_index():
    expr = m.linexpr((10, 'x'), (-1, 'x'))
    assert (expr.term_ == [0,1]).all()


def test_variable_to_linexpr():
    expr = 1 * x
    assert isinstance(expr, LinearExpression)
    assert expr.nterm == 1
    assert len(expr.vars.dim_0) == x.shape[0]

    expr = 10 * x + y
    assert isinstance(expr, LinearExpression)
    assert_equal(expr, m.linexpr((10, 'x'), (1, 'y')))

    expr = x + 8 * y
    assert isinstance(expr, LinearExpression)
    assert_equal(expr, m.linexpr((1, 'x'), (8, 'y')))

    expr = x + y
    assert isinstance(expr, LinearExpression)
    assert_equal(expr, m.linexpr((1, 'x'), (1, 'y')))

    expr = x - y
    assert isinstance(expr, LinearExpression)
    assert_equal(expr, m.linexpr((1, 'x'), (-1, 'y')))

    expr = -x - 8*y
    assert isinstance(expr, LinearExpression)
    assert_equal(expr, m.linexpr((-1, 'x'), (-8, 'y')))


def test_term_labels():
    "Test that the term_ dimension is named after the variables."
    expr = 10 * x + y
    other = m.linexpr((2, 'y'), (1, 'z'))

    assert (expr.term_ == [0, 1]).all()
    assert (other.term_ == [0, 1]).all()


def test_add():
    expr = 10 * x + y
    other = 2 * y + z
    res = expr + other

    assert res.nterm == expr.nterm + other.nterm
    assert (res.coords['dim_0'] == expr.coords['dim_0']).all()
    assert (res.coords['dim_1'] == other.coords['dim_1']).all()
    assert res.notnull().all().to_array().all()


def test_sub():
    expr = 10 * x + y
    other = 2 * y + z
    res = expr - other

    assert res.nterm == expr.nterm + other.nterm
    assert (res.coords['dim_0'] == expr.coords['dim_0']).all()
    assert (res.coords['dim_1'] == other.coords['dim_1']).all()
    assert res.notnull().all().to_array().all()


def test_sum():
    expr = 10 * x + y + z
    res = expr.sum('dim_0')

    assert res.size == expr.size
    assert res.nterm == expr.nterm * len(expr.dim_0)

    res = expr.sum()
    assert res.size == expr.size
    assert res.nterm == expr.size
    assert res.notnull().all().to_array().all()


def test_groupby():
    expr = 10 * x + y + z
    group = xr.DataArray([1,1,2], dims='dim_1')
    expr = expr.group_terms(group)
    assert 'group' in expr.dims
    assert (expr.group == [1, 2]).all()
