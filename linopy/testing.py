from __future__ import annotations

import numpy as np
from xarray.testing import assert_equal

from linopy.constants import TERM_DIM
from linopy.constraints import ConstraintBase, _con_unwrap
from linopy.expressions import LinearExpression, QuadraticExpression, _expr_unwrap
from linopy.model import Model
from linopy.variables import Variable, _var_unwrap


def _sort_by_vars_along_term(expr: LinearExpression) -> LinearExpression:
    """Sort a linear expression's terms by variable labels along _term."""
    ds = expr.data
    if TERM_DIM not in ds.dims:
        return expr
    order = np.argsort(ds["vars"].values, axis=-1, kind="stable")
    sorted_vars = np.take_along_axis(ds["vars"].values, order, axis=-1)
    sorted_coeffs = np.take_along_axis(ds["coeffs"].values, order, axis=-1)
    new_ds = ds.copy()
    new_ds["vars"] = (ds["vars"].dims, sorted_vars)
    new_ds["coeffs"] = (ds["coeffs"].dims, sorted_coeffs)
    return LinearExpression(new_ds, expr.model)


def assert_varequal(a: Variable, b: Variable) -> None:
    """Assert that two variables are equal."""
    return assert_equal(_var_unwrap(a), _var_unwrap(b))


def assert_linequal(
    a: LinearExpression | QuadraticExpression, b: LinearExpression | QuadraticExpression
) -> None:
    """
    Assert that two linear expressions are semantically equal.

    Terms are sorted by variable labels along _term before comparing,
    so expressions with different term orderings but identical mathematical
    meaning are considered equal.
    """
    assert isinstance(a, LinearExpression)
    assert isinstance(b, LinearExpression)
    a_sorted = _sort_by_vars_along_term(a)
    b_sorted = _sort_by_vars_along_term(b)
    return assert_equal(_expr_unwrap(a_sorted), _expr_unwrap(b_sorted))


def assert_quadequal(
    a: LinearExpression | QuadraticExpression, b: LinearExpression | QuadraticExpression
) -> None:
    """Assert that two quadratic or linear expressions are equal."""
    return assert_equal(_expr_unwrap(a), _expr_unwrap(b))


def assert_conequal(a: ConstraintBase, b: ConstraintBase, strict: bool = True) -> None:
    """
    Assert that two constraints are equal.

    Parameters
    ----------
        a: Constraint
            The first constraint.
        b: Constraint
            The second constraint.
        strict: bool
            Whether to compare the constraints strictly. If not, only compare mathematically relevant parts.
    """
    if strict:
        assert_equal(_con_unwrap(a), _con_unwrap(b))
    else:
        assert_linequal(a.lhs, b.lhs)
        assert_equal(a.sign, b.sign)
        assert_equal(a.rhs, b.rhs)


def assert_model_equal(a: Model, b: Model) -> None:
    """Assert that two models are equal."""
    for k in a.dataset_attrs:
        assert_equal(getattr(a, k), getattr(b, k))

    assert set(a.variables) == set(b.variables)
    assert set(a.constraints) == set(b.constraints)

    for v in a.variables:
        assert_varequal(a.variables[v], b.variables[v])

    for c in a.constraints:
        assert_conequal(a.constraints[c], b.constraints[c])

    assert_linequal(a.objective.expression, b.objective.expression)
    assert a.objective.sense == b.objective.sense
    assert a.objective.value == b.objective.value

    assert a.status == b.status
    assert a.termination_condition == b.termination_condition

    assert a.type == b.type
