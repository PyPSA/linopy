#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:47:32 2022.

@author: fabian
"""

import numpy as np
import pandas as pd
import pytest

from linopy import Model


def test_nan_in_variable_lower():
    m = Model()

    x = m.add_variables(lower=np.nan, name="x")
    y = m.add_variables(name="y")

    m.add_constraints(2 * x + 6 * y, ">=", 10)
    m.add_constraints(4 * x + 2 * y, ">=", 3)

    m.add_objective(2 * y + x)
    with pytest.raises(ValueError):
        m.solve()


def test_nan_in_variable_upper():
    m = Model()

    x = m.add_variables(upper=np.nan, name="x")
    y = m.add_variables(name="y")

    m.add_constraints(2 * x + 6 * y, ">=", 10)
    m.add_constraints(4 * x + 2 * y, ">=", 3)

    m.add_objective(2 * y + x)
    with pytest.raises(ValueError):
        m.solve()


def test_nan_in_constraint_coeffs():
    m = Model()

    x = m.add_variables(name="x")
    y = m.add_variables(name="y")

    m.add_constraints(np.nan * x + 6 * y, ">=", 10)
    m.add_constraints(4 * x + 2 * y, ">=", 3)

    m.add_objective(2 * y + x)
    with pytest.raises(ValueError):
        m.solve()


def test_nan_in_constraint_sign():
    m = Model()

    x = m.add_variables(name="x")
    y = m.add_variables(name="y")

    m.add_constraints(2 * x + 6 * y, np.nan, 10)
    m.add_constraints(4 * x + 2 * y, ">=", 3)

    m.add_objective(2 * y + x)
    with pytest.raises(ValueError):
        m.solve()


def test_nan_in_constraint_rhs():
    m = Model()

    x = m.add_variables(name="x")
    y = m.add_variables(name="y")

    m.add_constraints(2 * x + 6 * y, ">=", np.nan)
    m.add_constraints(4 * x + 2 * y, ">=", 3)

    m.add_objective(2 * y + x)
    with pytest.raises(ValueError):
        m.solve()


def test_nan_in_objective():
    m = Model()

    x = m.add_variables(name="x")
    y = m.add_variables(name="y")

    m.add_constraints(2 * x + 6 * y, ">=", np.nan)
    m.add_constraints(4 * x + 2 * y, ">=", 3)

    m.add_objective(np.nan * y + x)
    with pytest.raises(ValueError):
        m.solve()
