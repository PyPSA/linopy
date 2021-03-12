#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 11:23:13 2021

@author: fabulous
"""

import pytest
import xarray as xr
import numpy as np
from linopy import Model, linexpr
import pandas as pd
import numpy as np

# Test model functions

def test_add_variables_shape():
    target_shape = (10, 10)
    m = Model()

    lower = xr.DataArray(np.zeros((10,10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    m.add_variables('var1', lower, upper)
    assert m.variables.var1.shape == target_shape

    # setting only one dimension, the other has to be broadcasted
    lower = xr.DataArray(np.zeros((10)), coords=[range(10)])
    m.add_variables('var2', lower, upper)
    assert m.variables.var2.shape == target_shape

    # setting bounds without explicit bounds
    lower = xr.DataArray(np.zeros((10)))
    m.add_variables('var3', lower, upper)
    assert m.variables.var3.shape == target_shape

    # setting bounds with pandas index
    lower = xr.DataArray(np.zeros((10)), coords=[pd.Index(range(10))])
    m.add_variables('var4', lower, upper)
    assert m.variables.var4.shape == target_shape

    # define variable without any further information, should lead to a
    # single variable between minus and plus inf
    m.add_variables('var5')
    assert m.variables.var5.shape == ()
    assert m.variables_lower_bounds.var5 == -np.inf
    assert m.variables_upper_bounds.var5 == np.inf


    # setting bounds with scalar and no coords
    lower = 0
    upper = 1
    m.add_variables('var6', lower, upper)
    assert m.variables.var6.shape == ()
    assert m.variables_lower_bounds.var6 == 0
    assert m.variables_upper_bounds.var6 == 1


    # setting bounds with scalar and coords
    lower = 0
    upper = 1
    coords = [pd.Index(range(10)), pd.Index(range(10))]
    m.add_variables('var7', lower, upper, coords=coords)
    assert m.variables.var7.shape == target_shape


    # setting bounds with pd.DataFrames
    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.DataFrame(np.ones((10, 10)))
    m.add_variables('var8', lower, upper)
    assert m.variables.var8.shape == target_shape


    # setting bounds with one pd.DataFrame and one pd.Series
    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.Series(np.ones((10)))
    m.add_variables('var9', lower, upper)
    assert m.variables.var9.shape == target_shape


    # repeated variable assignment is forbidden
    with pytest.raises(AssertionError):
        m.add_variables('var9', lower, upper)




def test_linexpr():
    m = Model()

    lower = xr.DataArray(np.zeros((10,10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    m.add_variables('var1', lower, upper)

    m.add_variables('var2')

    expr = m.linexpr((1, 'var1'), (10, 'var2'))
    assert expr.shape == (10, 10)
    assert expr.dtype == object


    # select a variable by a scalar and broadcast if over another variable array
    expr = m.linexpr((1, 1), (10, 'var1'))
    assert expr.shape == (10, 10)
    assert expr.dtype == object


    # select a variable by a scalar and weight it with different coefficients
    expr = m.linexpr((np.arange(0, 10), 1), (10, 'var1'))
    assert expr.shape == (10, 10)
    assert expr.dtype == object

    # select two explicit variables by scalars
    expr = m.linexpr((1, 1), (10, 2))
    assert expr.size == 1
    assert expr.dtype == object



def test_constraints():
    m = Model()

    lower = xr.DataArray(np.zeros((10,10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    m.add_variables('var1', lower, upper)
    m.add_variables('var2')

    lhs = m.linexpr((1, 'var1'), (10, 'var2'))

    m.add_constraints('con1', lhs, '==', 0)

    assert m.constraints.con1.shape == (10, 10)
    assert m.constraints.con1.dtype == int
    assert m.constraints_lhs.con1.dtype == object
    assert np.issubdtype(m.constraints_sign.con1.dtype, str) or \
        m.constraints_sign.con1.dtype == object
    assert m.constraints_rhs.con1.dtype in (int, float)


def test_objective():
    m = Model()

    lower = xr.DataArray(np.zeros((10,10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    m.add_variables('var1', lower, upper)
    m.add_variables('var2', lower, upper)

    obj = m.linexpr((np.arange(0, 20, 2), 'var1'), (10, 'var2')).sum()

    m.add_objective(obj)
    assert m.objective.size == 1
    assert isinstance(m.objective.compute().item(), str)

    with pytest.raises(AssertionError):
        m.add_objective(obj)

    obj2 = m.linexpr((np.arange(1, 21, 2), 'var1'), (10, 'var2')).sum()
    m.add_objective(obj2, overwrite=True)
    assert len(m.objective.compute().item()) == len(obj.compute().item())
    assert m.objective.compute() != obj.compute()

    m.add_objective(obj, extend=True)
    assert len(m.objective.compute().item()) > len(obj2.compute().item())

