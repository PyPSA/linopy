from __future__ import annotations

from xarray.testing import assert_equal

from linopy.constraints import Constraint, _con_unwrap
from linopy.expressions import LinearExpression, QuadraticExpression, _expr_unwrap
from linopy.model import Model
from linopy.variables import Variable, _var_unwrap


def assert_varequal(a: Variable, b: Variable) -> None:
    """Assert that two variables are equal."""
    return assert_equal(_var_unwrap(a), _var_unwrap(b))


def assert_linequal(
    a: LinearExpression | QuadraticExpression, b: LinearExpression | QuadraticExpression
) -> None:
    """Assert that two linear expressions are equal."""
    assert isinstance(a, LinearExpression)
    assert isinstance(b, LinearExpression)
    return assert_equal(_expr_unwrap(a), _expr_unwrap(b))


def assert_quadequal(
    a: LinearExpression | QuadraticExpression, b: LinearExpression | QuadraticExpression
) -> None:
    """Assert that two quadratic or linear expressions are equal."""
    return assert_equal(_expr_unwrap(a), _expr_unwrap(b))


def assert_conequal(a: Constraint, b: Constraint, strict: bool = True) -> None:
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
