#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 22:38:48 2021.

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


def test_scalarconstraint():
    m = Model()
    coords = [pd.Index(range(10))]
    x = m.add_variables(coords=coords)
    con = m.add_constraints(x[0] >= 0)
    assert isinstance(con, linopy.constraints.Constraint)

    con = m.add_constraints(x[0] + x[1] >= 0)
    assert isinstance(con, linopy.constraints.Constraint)

    con = m.add_constraints(x[0], ">=", 0)
    assert isinstance(con, linopy.constraints.Constraint)

    con = m.add_constraints(x[0] + x[1], ">=", 0)
    assert isinstance(con, linopy.constraints.Constraint)


def test_constraint_accessor():
    m = Model()
    x = m.add_variables()
    y = m.add_variables()
    c = m.add_constraints(x, ">=", 0)
    assert c.rhs.item() == 0
    assert c.vars.item() == 0
    assert c.coeffs.item() == 1
    assert c.sign.item() == ">="

    c.rhs = 2
    assert c.rhs.item() == 2

    c.vars = y
    assert c.vars.item() == 1

    c.coeffs = 3
    assert c.coeffs.item() == 3

    c.sign = "="
    assert c.sign.item() == "="

    c.lhs = x + y
    assert len(c.vars) == 2
    assert len(c.coeffs) == 2
    assert c.vars.notnull().all().item()
    assert c.coeffs.notnull().all().item()


def test_constraint_accessor_M():
    m = Model()
    lower = pd.Series(range(10), range(10))
    upper = pd.Series(range(10, 20), range(10))
    x = m.add_variables(lower, upper)
    y = m.add_variables(lower, upper)
    c = m.add_constraints(x, ">=", 0)
    assert c.rhs.item() == 0
    assert (c.rhs == 0).all()

    assert c.vars.shape == (10, 1)
    assert c.coeffs.shape == (10, 1)

    assert c.sign.item() == ">="

    c.rhs = 2
    assert (c.rhs == 2).all().item()

    c.lhs = 3 * y
    assert (c.vars.squeeze() == y.data).all()
    assert (c.coeffs == 3).all()
    assert isinstance(c.lhs, linopy.LinearExpression)


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
        c.coeffs
    with pytest.raises(AttributeError):
        c.vars
    with pytest.raises(AttributeError):
        c.sign
    with pytest.raises(AttributeError):
        c.rhs


def test_constraint_sanitize_zeros():
    m = Model()
    x = m.add_variables(coords=[range(10)])
    y = m.add_variables()
    m.add_constraints(0 * x + y == 0)
    m.constraints.sanitize_zeros()
    assert m.constraints["con0"].vars[0, 0].item() == -1
    assert np.isnan(m.constraints["con0"].coeffs[0, 0].item())


def test_constraint_matrix():
    m = Model()
    x = m.add_variables(coords=[range(10)])
    y = m.add_variables()
    m.add_constraints(x, "=", 0)
    A = m.constraints.to_matrix()
    assert A.shape == (10, 11)


def test_constraint_matrix_masked_variables():
    """
    Test constraint matrix with missing variables.

    In this case the variables that are used in the constraints are
    missing. The matrix shoud not be built for constraints which have
    variables which are missing.
    """
    # now with missing variables
    m = Model()
    mask = pd.Series([False] * 5 + [True] * 5)
    x = m.add_variables(coords=[range(10)], mask=mask)
    m.add_variables()
    m.add_constraints(x, "=", 0)
    A = m.constraints.to_matrix(filter_missings=True)
    assert A.shape == (5, 6)
    assert A.shape == (m.ncons, m.nvars)

    A = m.constraints.to_matrix(filter_missings=False)
    assert A.shape == (m._cCounter, m._xCounter)


def test_constraint_matrix_masked_constraints():
    """
    Test constraint matrix with missing constraints.
    """
    # now with missing variables
    m = Model()
    mask = pd.Series([False] * 5 + [True] * 5)
    x = m.add_variables(coords=[range(10)])
    m.add_variables()
    m.add_constraints(x, "=", 0, mask=mask)
    A = m.constraints.to_matrix(filter_missings=True)
    assert A.shape == (5, 11)
    assert A.shape == (m.ncons, m.nvars)

    A = m.constraints.to_matrix(filter_missings=False)
    assert A.shape == (m._cCounter, m._xCounter)


def test_constraint_matrix_masked_constraints_and_variables():
    """
    Test constraint matrix with missing constraints.
    """
    # now with missing variables
    m = Model()
    mask = pd.Series([False] * 5 + [True] * 5)
    x = m.add_variables(coords=[range(10)], mask=mask)
    m.add_variables()
    m.add_constraints(x, "=", 0, mask=mask)
    A = m.constraints.to_matrix(filter_missings=True)
    assert A.shape == (5, 6)
    assert A.shape == (m.ncons, m.nvars)

    A = m.constraints.to_matrix(filter_missings=False)
    assert A.shape == (m._cCounter, m._xCounter)


def test_get_name_by_label():
    m = Model()
    x = m.add_variables(coords=[range(10)])
    y = m.add_variables(coords=[range(10)])

    m.add_constraints(x + y <= 10, name="first")
    m.add_constraints(x - y >= 5, name="second")

    assert m.constraints.get_name_by_label(4) == "first"
    assert m.constraints.get_name_by_label(14) == "second"

    with pytest.raises(ValueError):
        m.constraints.get_name_by_label(30)

    with pytest.raises(ValueError):
        m.constraints.get_name_by_label("first")
