from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.testing import assert_equal

from linopy import LESS_EQUAL, Model, Variable
from linopy.testing import assert_linequal, assert_quadequal


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

    def __add__(self, other: Any) -> xr.DataArray:
        return self.active_data + other

    def __sub__(self, other: Any) -> xr.DataArray:
        return self.active_data - other

    def __mul__(self, other: Any) -> xr.DataArray:
        return self.active_data * other

    def __truediv__(self, other: Any) -> xr.DataArray:
        return self.active_data / other

    def __radd__(self, other: Any) -> Any:
        return other + self.active_data

    def __rsub__(self, other: Any) -> Any:
        return other - self.active_data

    def __rmul__(self, other: Any) -> Any:
        return other * self.active_data

    def __rtruediv__(self, other: Any) -> Any:
        return other / self.active_data

    def __neg__(self) -> xr.DataArray:
        return -self.active_data

    def __pos__(self) -> xr.DataArray:
        return +self.active_data

    def __abs__(self) -> xr.DataArray:
        return abs(self.active_data)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):  # type: ignore
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
def m(request) -> Model:  # type: ignore
    m = Model()
    x = m.add_variables(coords=request.param, name="x")
    m.add_variables(0, 10, name="z")
    m.add_constraints(x, LESS_EQUAL, 0, name="c")
    return m


def test_arithmetric_operations_variable(m: Model) -> None:
    x: Variable = m.variables["x"]
    rng = np.random.default_rng()
    data = xr.DataArray(rng.random(x.shape), coords=x.coords)
    other_datatype = SomeOtherDatatype(data.copy())
    assert_linequal(x + data, x + other_datatype)
    assert_linequal(x - data, x - other_datatype)
    assert_linequal(x * data, x * other_datatype)
    assert_linequal(x / data, x / other_datatype)  # type: ignore
    assert_linequal(data * x, other_datatype * x)  # type: ignore
    assert x.__add__(object()) is NotImplemented
    assert x.__sub__(object()) is NotImplemented
    assert x.__mul__(object()) is NotImplemented
    assert x.__truediv__(object()) is NotImplemented  # type: ignore
    assert x.__pow__(object()) is NotImplemented  # type: ignore
    with pytest.raises(ValueError):
        x.__pow__(3)


def test_arithmetric_operations_expr(m: Model) -> None:
    x = m.variables["x"]
    expr = x + 3
    rng = np.random.default_rng()
    data = xr.DataArray(rng.random(x.shape), coords=x.coords)
    other_datatype = SomeOtherDatatype(data.copy())
    assert_linequal(expr + data, expr + other_datatype)
    assert_linequal(expr - data, expr - other_datatype)
    assert_linequal(expr * data, expr * other_datatype)
    assert_linequal(expr / data, expr / other_datatype)
    assert expr.__add__(object()) is NotImplemented
    assert expr.__sub__(object()) is NotImplemented
    assert expr.__mul__(object()) is NotImplemented
    assert expr.__truediv__(object()) is NotImplemented


def test_arithmetric_operations_vars_and_expr(m: Model) -> None:
    x = m.variables["x"]
    x_expr = x * 1.0

    assert_quadequal(x**2, x_expr**2)
    assert_quadequal(x**2 + x, x + x**2)
    assert_quadequal(x**2 * 2, x**2 * 2)
    with pytest.raises(TypeError):
        _ = x**2 * x


def test_arithmetric_operations_con(m: Model) -> None:
    c = m.constraints["c"]
    x = m.variables["x"]
    rng = np.random.default_rng()
    data = xr.DataArray(rng.random(x.shape), coords=x.coords)
    other_datatype = SomeOtherDatatype(data.copy())
    assert_linequal(c.lhs + data, c.lhs + other_datatype)
    assert_linequal(c.lhs - data, c.lhs - other_datatype)
    assert_linequal(c.lhs * data, c.lhs * other_datatype)
    assert_linequal(c.lhs / data, c.lhs / other_datatype)

    assert_equal(c.rhs + data, c.rhs + other_datatype)
    assert_equal(c.rhs - data, c.rhs - other_datatype)
    assert_equal(c.rhs * data, c.rhs * other_datatype)
    assert_equal(c.rhs / data, c.rhs / other_datatype)
