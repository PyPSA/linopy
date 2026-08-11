#!/usr/bin/env python3
"""
This module aims at testing the correct behavior of the Expressions class.
"""

import pandas as pd
import pytest
import xarray as xr

from linopy import Model
from linopy.expressions import Expressions, LinearExpression, QuadraticExpression
from linopy.solvers import available_solvers
from linopy.testing import assert_linequal


@pytest.fixture
def m() -> Model:
    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(10, name="first")], name="x")
    y = m.add_variables(coords=[pd.Index([1, 2, 3], name="second")], name="y")
    m.add_expressions(x + 1, name="expr_x")
    m.add_expressions(x * y, name="expr_xy")
    return m


def test_expressions_repr(m: Model) -> None:
    m.expressions.__repr__()
    repr(Model())


def test_expressions_getitem(m: Model) -> None:
    assert isinstance(m.expressions["expr_x"], LinearExpression)

    subset = m.expressions[["expr_x"]]
    assert isinstance(subset, Expressions)
    assert len(subset) == 1


def test_expressions_getattr(m: Model) -> None:
    assert_linequal(m.expressions.expr_x, m.expressions["expr_x"])

    with pytest.raises(AttributeError):
        m.expressions.does_not_exist


def test_expressions_getattr_formatted() -> None:
    m = Model()
    x = m.add_variables(name="x")
    m.add_expressions(x + 1, name="e-0")
    assert_linequal(m.expressions.e_0, m.expressions["e-0"])


def test_expressions_dict_protocol(m: Model) -> None:
    assert len(m.expressions) == 2
    assert set(iter(m.expressions)) == {"expr_x", "expr_xy"}
    assert set(dict(m.expressions.items())) == {"expr_x", "expr_xy"}
    assert "expr_x" in m.expressions
    assert m.expressions._ipython_key_completions_() == list(m.expressions)
    assert "expr_x" in dir(m.expressions)


def test_expressions_name_counter() -> None:
    m = Model()
    x = m.add_variables(name="x")
    m.add_expressions(x + 1)
    m.add_expressions(x + 1)
    assert "expr0" in m.expressions
    assert "expr1" in m.expressions


def test_expressions_duplicate_name_raises(m: Model) -> None:
    x = m.variables["x"]
    with pytest.raises(ValueError, match="already assigned"):
        m.add_expressions(x + 1, name="expr_x")


def test_add_expressions_from_variable_and_tuples() -> None:
    m = Model()
    x = m.add_variables(name="x")

    expr = m.add_expressions(x, name="from_var")
    assert isinstance(expr, LinearExpression)
    assert_linequal(expr, x.to_linexpr())
    assert_linequal(expr, m.expressions["from_var"])

    expr = m.add_expressions([(2, x)], name="from_tuples")
    assert isinstance(expr, LinearExpression)
    assert_linequal(expr, 2 * x)
    assert_linequal(expr, m.expressions["from_tuples"])


def test_add_expressions_quadratic(m: Model) -> None:
    assert isinstance(m.expressions["expr_xy"], QuadraticExpression)


def test_add_expressions_mask() -> None:
    m = Model()
    idx = pd.RangeIndex(10, name="first")
    x = m.add_variables(coords=[idx], name="x")
    mask = xr.DataArray([True] * 5 + [False] * 5, coords=[idx])

    expr = m.add_expressions(x + 1, name="masked", mask=mask)
    assert_linequal(expr, (x + 1).where(mask))


def test_expressions_remove(m: Model) -> None:
    m.expressions.remove("expr_x")
    assert "expr_x" not in m.expressions

    with pytest.raises(KeyError):
        m.expressions.remove("expr_x")


def test_remove_expressions(m: Model) -> None:
    m.remove_expressions("expr_x")
    assert "expr_x" not in m.expressions
    assert "expr_xy" in m.expressions


def test_remove_expressions_with_list(m: Model) -> None:
    m.remove_expressions(["expr_x", "expr_xy"])
    assert len(m.expressions) == 0


def test_model_repr_contains_expressions(m: Model) -> None:
    r = repr(m)
    assert "Expressions:" in r
    assert "* expr_x" in r


@pytest.mark.skipif(not available_solvers, reason="No solver available")
def test_expressions_solution() -> None:
    m = Model()
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(3, name="first")], name="x")
    m.add_constraints(x >= 2)
    m.add_expressions(2 * x, name="double_x")
    m.add_objective(x.sum())
    m.solve(available_solvers[0])

    sol = m.expressions.solution
    assert isinstance(sol, xr.Dataset)
    assert "double_x" in sol
    assert (sol["double_x"] == 4).all()
