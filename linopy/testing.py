from xarray.testing import assert_equal

from linopy.expressions import _expr_unwrap


def assert_linequal(a, b):
    return assert_equal(_expr_unwrap(a), _expr_unwrap(b))
