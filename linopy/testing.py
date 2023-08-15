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
