from xarray.testing import assert_equal
from .expressions import _expr_unwrap


def assert_linequal(a, b):
    return assert_equal(_expr_unrwap(a), _expr_unwrap(b))
