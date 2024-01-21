from xarray.testing import assert_equal

from linopy.constraints import _con_unwrap
from linopy.expressions import _expr_unwrap
from linopy.variables import _var_unwrap


def assert_varequal(a, b):
    """Assert that two variables are equal."""
    return assert_equal(_var_unwrap(a), _var_unwrap(b))


def assert_linequal(a, b):
    """Assert that two linear expressions are equal."""
    return assert_equal(_expr_unwrap(a), _expr_unwrap(b))


def assert_conequal(a, b):
    """Assert that two constraints are equal."""
    return assert_equal(_con_unwrap(a), _con_unwrap(b))


def assert_model_equal(a, b):
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

    assert a.type == b.type
