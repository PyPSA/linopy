import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model
from linopy.testing import assert_linequal


class SomeOtherDatatype:
    """
    A class that is not a subclass of xarray.DataArray, but stores data in a compatible way.
    It defines all necessary arithmetrics AND __array_ufunc__ to ensure that operations are
    performed on the active_data.
    """

    def __init__(self, data: xr.DataArray) -> None:
        self.data1 = data
        self.data2 = data.copy()
        self.active = 1

    def activate(self, active: int) -> None:
        self.active = active

    @property
    def active_data(self) -> xr.DataArray:
        return self.data1 if self.active == 1 else self.data2

    def __add__(self, other):
        return self.active_data + other

    def __sub__(self, other):
        return self.active_data - other

    def __mul__(self, other):
        return self.active_data * other

    def __truediv__(self, other):
        return self.active_data / other

    def __radd__(self, other):
        return other + self.active_data

    def __rsub__(self, other):
        return other - self.active_data

    def __rmul__(self, other):
        return other * self.active_data

    def __rtruediv__(self, other):
        return other / self.active_data

    def __neg__(self):
        return -self.active_data

    def __pos__(self):
        return +self.active_data

    def __abs__(self):
        return abs(self.active_data)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # Ensure we always use the active_data when interacting with numpy/xarray operations
        new_inputs = [
            inp.active_data if isinstance(inp, SomeOtherDatatype) else inp
            for inp in inputs
        ]
        return getattr(ufunc, method)(*new_inputs, **kwargs)


@pytest.fixture(
    params=[
        (pd.RangeIndex(10, name="first"),),
        (
            pd.Index(range(5), name="first"),
            pd.Index(range(3), name="second"),
            pd.Index(range(2), name="third"),
        ),
    ],
    ids=["single_dim", "multi_dim"],
)
def m(request) -> Model:
    m = Model()
    m.add_variables(coords=request.param, name="x")
    m.add_variables(0, 10, name="z")
    m.add_constraints(m.variables["x"] >= 0, name="c")
    return m


def test_arithmetric_operations_variable(m: Model) -> None:
    x = m.variables["x"]
    data = xr.DataArray(np.random.Generator(*x.shape), coords=x.coords)
    other_datatype = SomeOtherDatatype(data.copy())
    assert_linequal(x + data, x + other_datatype)
    assert_linequal(x - data, x - other_datatype)
    assert_linequal(x * data, x * other_datatype)
    assert_linequal(x / data, x / other_datatype)


def test_arithmetric_operations_con(m: Model) -> None:
    c = m.constraints["c"]
    x = m.variables["x"]
    data = xr.DataArray(np.random.Generator(*x.shape), coords=x.coords)
    other_datatype = SomeOtherDatatype(data.copy())
    assert_linequal(c.lhs + data, c.lhs + other_datatype)
    assert_linequal(c.lhs - data, c.lhs - other_datatype)
    assert_linequal(c.lhs * data, c.lhs * other_datatype)
    assert_linequal(c.lhs / data, c.lhs / other_datatype)
    assert_linequal(c.rhs + data, c.rhs + other_datatype)
    assert_linequal(c.rhs - data, c.rhs - other_datatype)
    assert_linequal(c.rhs * data, c.rhs * other_datatype)
    assert_linequal(c.rhs / data, c.rhs / other_datatype)
