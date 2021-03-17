#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:06:36 2021

@author: fabian
"""

import xarray as xr
import numpy as np
import pandas as pd

from linopy import LinearExpression, Model


m = Model()

m.add_variables('x', pd.Series([0,0]), 1)
m.add_variables('y', 4, pd.Series([8,10]))
m.add_variables('z', 0, pd.DataFrame([[1,2], [3,4], [5,6]]).T)

lhs = m.linexpr((1, 'x'), (4, 'y'))
other = m.linexpr((2, 'y'), (1, 'z'))

def test_term_labels():
    "Test that the term_ dimension is named after the variables."
    assert (lhs.coefficients.term_ == ['x', 'y']).all()
    assert (lhs.variables.term_ == ['x', 'y']).all()

    assert (other.coefficients.term_ == ['y', 'z']).all()
    assert (other.variables.term_ == ['y', 'z']).all()


def test_coords():
    "Make sure that the coords are the same for variables and coefficients."
    assert (lhs.coords['term_'] == lhs.coefficients.term_).all()
    assert (lhs.coords['dim_0'] == lhs.coefficients.dim_0).all()

    assert (lhs.coords['term_'] == lhs.variables.term_).all()
    assert (lhs.coords['dim_0'] == lhs.variables.dim_0).all()


def test_add():
    res = lhs + other
    assert len(res.coords['term_']) == len(lhs.coords['term_']) + len(other.coords['term_'])
    assert (res.coords['dim_0'] == lhs.coords['dim_0']).all()
    assert (res.coords['dim_1'] == other.coords['dim_1']).all()
    assert res.coefficients.notnull().all()
    # assert res.variables.notnull().all()


def test_sum():
    res = lhs.sum()
    assert res.size == lhs.size
    assert len(res.coords['term_']) == lhs.size

    res = other.sum('dim_1')
    assert res.size == other.size
    assert len(res.coords['term_']) == len(other.coords['term_']) * len(other.coords['dim_1'])

