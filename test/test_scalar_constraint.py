#!/usr/bin/env python3


import pandas as pd
import pytest

import linopy
from linopy import GREATER_EQUAL, Model, Variable
from linopy.constraints import AnonymousScalarConstraint, Constraint


@pytest.fixture
def m() -> Model:
    m = Model()
    m.add_variables(coords=[pd.RangeIndex(10, name="first")], name="x")
    return m


@pytest.fixture
def x(m: Model) -> Variable:
    return m.variables["x"]


def test_scalar_constraint_repr(x: Variable) -> None:
    c: AnonymousScalarConstraint = x.at[0] >= 0
    c.__repr__()


def test_anonymous_scalar_constraint_type(x: Variable) -> None:
    c: AnonymousScalarConstraint = x.at[0] >= 0
    assert isinstance(c, linopy.constraints.AnonymousScalarConstraint)


def test_simple_constraint_type(m: Model, x: Variable) -> None:
    c: Constraint = m.add_constraints(x.at[0] >= 0)
    assert isinstance(c, linopy.constraints.Constraint)


def test_compound_constraint_type(m: Model, x: Variable) -> None:
    c: Constraint = m.add_constraints(x.at[0] + x.at[1] >= 0)
    assert isinstance(c, linopy.constraints.Constraint)


def test_explicit_simple_constraint_type(m: Model, x: Variable) -> None:
    c: Constraint = m.add_constraints(x.at[0], GREATER_EQUAL, 0)
    assert isinstance(c, linopy.constraints.Constraint)


def test_explicit_compound_constraint_type(m: Model, x: Variable) -> None:
    c: Constraint = m.add_constraints(x.at[0] + x.at[1], GREATER_EQUAL, 0)
    assert isinstance(c, linopy.constraints.Constraint)
