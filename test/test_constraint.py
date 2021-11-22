#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 22:38:48 2021

@author: fabian
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.testing import assert_equal

import linopy
from linopy import Model


def test_constraint_repr():
    m = Model()

    x = m.add_variables()
    c = m.add_constraints(x, ">=", 0)
    c.__repr__()
    c._repr_html_()


def test_constraints_repr():
    m = Model()
    m.constraints.__repr__()
    x = m.add_variables()
    m.add_constraints(x, ">=", 0)
    m.constraints.__repr__()


def test_constraint_accessor():
    m = Model()
    x = m.add_variables()
    c = m.add_constraints(x, ">=", 0)
    assert c.get_rhs().item() == 0
    assert c.get_vars().item() == 0
    assert c.get_coeffs().item() == 1
    assert c.get_sign().item() == ">="


def test_constraints_accessor():
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper)
    y = m.add_variables()
    m.add_constraints(1 * x + 10 * y, "=", 0)
    assert m.constraints["con0"].shape == (10, 10)
    assert isinstance(m.constraints[["con0"]], linopy.constraints.Constraints)
    assert isinstance(m.constraints.inequalities, linopy.constraints.Constraints)
    assert isinstance(m.constraints.equalities, linopy.constraints.Constraints)


def test_constraint_getter_without_model():

    data = xr.DataArray(range(10)).rename("con")
    c = linopy.constraints.Constraint(data)

    with pytest.raises(AttributeError):
        c.get_coeffs()
    with pytest.raises(AttributeError):
        c.get_vars()
    with pytest.raises(AttributeError):
        c.get_sign()
    with pytest.raises(AttributeError):
        c.get_rhs()
