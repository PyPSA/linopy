#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import pytest

import linopy
from linopy import GREATER_EQUAL, Model


@pytest.fixture
def m():
    m = Model()
    m.add_variables(coords=[pd.RangeIndex(10, name="first")], name="x")
    return m


@pytest.fixture
def x(m):
    return m.variables["x"]


def test_scalar_constraint_repr(x):
    c = x[0] >= 0
    c.__repr__()


def test_scalar_constraint_initialization(m, x):
    c = x[0] >= 0
    assert isinstance(c, linopy.constraints.AnonymousScalarConstraint)

    c = m.add_constraints(x[0] >= 0)
    assert isinstance(c, linopy.constraints.Constraint)

    c = m.add_constraints(x[0] + x[1] >= 0)
    assert isinstance(c, linopy.constraints.Constraint)

    c = m.add_constraints(x[0], GREATER_EQUAL, 0)
    assert isinstance(c, linopy.constraints.Constraint)

    c = m.add_constraints(x[0] + x[1], GREATER_EQUAL, 0)
    assert isinstance(c, linopy.constraints.Constraint)
