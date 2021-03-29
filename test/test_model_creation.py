#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:23:13 2021

@author: fabulous
"""

import pytest
import xarray as xr
import numpy as np
from linopy import Model
import pandas as pd

# Test model functions

def test_add_variables_shape():
    target_shape = (10, 10)
    m = Model()

    lower = xr.DataArray(np.zeros((10,10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    m.add_variables(lower, upper)
    assert m.variables.var1.shape == target_shape

    # setting only one dimension, the other has to be broadcasted
    lower = xr.DataArray(np.zeros((10)), coords=[range(10)])
    m.add_variables(lower, upper)
    assert 'var2' in m.variables, 'Variable names are not set as expected.'
    assert m.variables.var2.shape == target_shape

    # setting bounds without explicit bounds
    lower = xr.DataArray(np.zeros((10)))
    m.add_variables(lower, upper)
    assert m.variables.var3.shape == target_shape

    # setting bounds with pandas index
    lower = xr.DataArray(np.zeros((10)), coords=[pd.Index(range(10))])
    m.add_variables(lower, upper)
    assert m.variables.var4.shape == target_shape

    # define variable without any further information, should lead to a
    # single variable between minus and plus inf
    m.add_variables()
    assert m.variables.var5.shape == ()
    assert m.variables_lower_bound.var5 == -np.inf
    assert m.variables_upper_bound.var5 == np.inf


    # setting bounds with scalar and no coords
    lower = 0
    upper = 1
    m.add_variables(lower, upper)
    assert m.variables.var6.shape == ()
    assert m.variables_lower_bound.var6 == 0
    assert m.variables_upper_bound.var6 == 1


    # setting bounds with scalar and coords
    lower = 0
    upper = 1
    coords = [pd.Index(range(10)), pd.Index(range(10))]
    m.add_variables(lower, upper, coords=coords)
    assert m.variables.var7.shape == target_shape


    # setting bounds with pd.DataFrames
    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.DataFrame(np.ones((10, 10)))
    m.add_variables(lower, upper)
    assert m.variables.var8.shape == target_shape


    # setting bounds with one pd.DataFrame and one pd.Series
    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.Series(np.ones((10)))
    m.add_variables(lower, upper)
    assert m.variables.var9.shape == target_shape


    # set a variable with different set of coordinates, this should be properly
    # merged
    lower = pd.DataFrame(np.zeros((20, 10)))
    upper = pd.Series(np.ones((20)))
    m.add_variables(lower, upper)
    assert m.variables.var10.shape == (20, 10)
    # var9 should now be aligned to new coords and contain 100 nans
    assert m.variables.var9.shape == (20, 10)
    assert m.variables.var9.notnull().sum() == 100


    # setting with scalar and list
    with pytest.raises(ValueError):
        m.add_variables(0, [1,2])


    # repeated variable assignment is forbidden
    with pytest.raises(AssertionError):
        m.add_variables(lower, upper, name='var9')



def test_linexpr():
    m = Model()

    lower = xr.DataArray(np.zeros((10,10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name='x')
    y = m.add_variables(name='y')

    # select a variable by a scalar and broadcast if over another variable array
    expr = m.linexpr((1, 'x'), (10, 'y'))
    # assert (expr._term == ['x', 'y']).all()
    assert (expr == 1 * x + 10 * y).all().to_array().all()



def test_constraints():
    m = Model()

    lower = xr.DataArray(np.zeros((10,10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper)
    y = m.add_variables()

    m.add_constraints(1 * x + 10 * y, '=', 0)

    assert m.constraints.con1.shape == (10, 10)
    assert m.constraints.con1.dtype == int
    assert m.constraints_lhs_coeffs.con1.dtype in (int, float)
    assert m.constraints_lhs_vars.con1.dtype in (int, float)
    assert m.constraints_rhs.con1.dtype in (int, float)


def test_objective():
    m = Model()

    lower = xr.DataArray(np.zeros((10,10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name='x')
    y = m.add_variables(lower, upper, name='y')

    obj = (10 * x + 5 * y).sum()
    m.add_objective(obj)
    assert m.objective.vars.size == 200


def test_variable_getitem():
    m = Model()
    x = m.add_variables(name='x')
    assert m['x'] == x


def test_remove_variable():
    m = Model()

    lower = xr.DataArray(np.zeros((10,10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name='x')
    y = m.add_variables(name='y')

    m.add_constraints(1 * x + 10 * y, '=', 0)

    m.remove_variables('x')
    assert 'x' not in m.variables
    assert 'x' not in m.variables_lower_bound
    assert 'x' not in m.variables_upper_bound

    assert 1 not in m.constraints_lhs_vars.con1

