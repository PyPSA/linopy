#!/usr/bin/env python3
"""
Created on Tue Nov  2 22:36:38 2021.

@author: fabian
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
import xarray.core.indexes
import xarray.core.utils
from xarray.testing import assert_equal

import linopy
import linopy.variables
from linopy import Model
from linopy.testing import assert_linequal


@pytest.fixture
def m() -> Model:
    m = Model()
    m.add_variables(coords=[pd.RangeIndex(10, name="first")], name="x")
    m.add_variables(coords=[pd.Index([1, 2, 3], name="second")], name="y")
    m.add_variables(0, 10, name="z")
    return m


@pytest.fixture
def x(m: Model) -> linopy.Variable:
    return m.variables["x"]


@pytest.fixture
def z(m: Model) -> linopy.Variable:
    return m.variables["z"]


def test_variable_repr(x: linopy.Variable) -> None:
    x.__repr__()


def test_variable_inherited_properties(x: linopy.Variable) -> None:
    assert isinstance(x.attrs, dict)
    assert isinstance(x.coords, xr.Coordinates)
    assert isinstance(x.indexes, xarray.core.indexes.Indexes)
    assert isinstance(x.sizes, xarray.core.utils.Frozen)
    assert isinstance(x.shape, tuple)
    assert isinstance(x.size, int)
    assert isinstance(x.dims, tuple)
    assert isinstance(x.ndim, int)


def test_variable_labels(x: linopy.Variable) -> None:
    isinstance(x.labels, xr.DataArray)


def test_variable_data(x: linopy.Variable) -> None:
    isinstance(x.data, xr.DataArray)


def test_wrong_variable_init(m: Model, x: linopy.Variable) -> None:
    # wrong data type
    with pytest.raises(ValueError):
        linopy.Variable(x.labels.values, m, "")  # type: ignore

    # no model
    with pytest.raises(ValueError):
        linopy.Variable(x.labels, None, "")  # type: ignore


def test_variable_getter(x: linopy.Variable, z: linopy.Variable) -> None:
    assert isinstance(x[0], linopy.variables.Variable)

    assert isinstance(x.at[0], linopy.variables.ScalarVariable)


def test_variable_getter_slice(x: linopy.Variable) -> None:
    res = x[:5]
    assert isinstance(res, linopy.Variable)
    assert res.size == 5


def test_variable_getter_slice_with_step(x: linopy.Variable) -> None:
    res = x[::2]
    assert isinstance(res, linopy.Variable)
    assert res.size == 5


def test_variables_getter_list(x: linopy.Variable) -> None:
    res = x[[1, 2, 3]]
    assert isinstance(res, linopy.Variable)
    assert res.size == 3


def test_variable_getter_invalid_shape(x: linopy.Variable) -> None:
    with pytest.raises(AssertionError):
        x.at[0, 0]


def test_variable_loc(x: linopy.Variable) -> None:
    assert isinstance(x.loc[[1, 2, 3]], linopy.Variable)


def test_variable_sel(x: linopy.Variable) -> None:
    assert isinstance(x.sel(first=[1, 2, 3]), linopy.Variable)


def test_variable_isel(x: linopy.Variable) -> None:
    assert isinstance(x.isel(first=[1, 2, 3]), linopy.Variable)
    assert_equal(
        x.isel(first=[0, 1]).labels,
        x.sel(first=[0, 1]).labels,
    )


def test_variable_upper_getter(z: linopy.Variable) -> None:
    assert z.upper.item() == 10


def test_variable_lower_getter(z: linopy.Variable) -> None:
    assert z.lower.item() == 0


def test_variable_upper_setter(z: linopy.Variable) -> None:
    z.upper = 20
    assert z.upper.item() == 20


def test_variable_lower_setter(z: linopy.Variable) -> None:
    z.lower = 8
    assert z.lower == 8


def test_variable_upper_setter_with_array(x: linopy.Variable) -> None:
    idx = pd.RangeIndex(10, name="first")
    upper = pd.Series(range(25, 35), index=idx)
    x.upper = upper
    assert isinstance(x.upper, xr.DataArray)
    assert (x.upper == upper).all()


def test_variable_upper_setter_with_array_invalid_dim(x: linopy.Variable) -> None:
    with pytest.raises(ValueError):
        upper = pd.Series(range(25, 35))
        x.upper = upper


def test_variable_lower_setter_with_array(x: linopy.Variable) -> None:
    idx = pd.RangeIndex(10, name="first")
    lower = pd.Series(range(15, 25), index=idx)
    x.lower = lower
    assert isinstance(x.lower, xr.DataArray)
    assert (x.lower == lower).all()


def test_variable_lower_setter_with_array_invalid_dim(x: linopy.Variable) -> None:
    with pytest.raises(ValueError):
        lower = pd.Series(range(15, 25))
        x.lower = lower


def test_variable_sum(x: linopy.Variable) -> None:
    res = x.sum()
    assert res.nterm == 10


def test_variable_sum_warn_using_dims(x: linopy.Variable) -> None:
    with pytest.warns(DeprecationWarning):
        x.sum(dims="first")


def test_variable_sum_warn_unknown_kwargs(x: linopy.Variable) -> None:
    with pytest.raises(ValueError):
        x.sum(unknown_kwarg="first")


def test_fill_value() -> None:
    isinstance(linopy.variables.Variable._fill_value, dict)


def test_variable_where(x: linopy.Variable) -> None:
    x = x.where([True] * 4 + [False] * 6)
    assert isinstance(x, linopy.variables.Variable)
    assert x.labels[9] == x._fill_value["labels"]

    x = x.where([True] * 4 + [False] * 6, x.at[0])
    assert isinstance(x, linopy.variables.Variable)
    assert x.labels[9] == x.at[0].label

    x = x.where([True] * 4 + [False] * 6, x.loc[0])
    assert isinstance(x, linopy.variables.Variable)
    assert x.labels[9] == x.at[0].label

    with pytest.raises(ValueError):
        x.where([True] * 4 + [False] * 6, 0)  # type: ignore


def test_variable_shift(x: linopy.Variable) -> None:
    x = x.shift(first=3)
    assert isinstance(x, linopy.variables.Variable)
    assert x.labels[0] == -1


def test_variable_swap_dims(x: linopy.Variable) -> None:
    x = x.assign_coords({"second": ("first", x.indexes["first"] + 100)})
    x = x.swap_dims({"first": "second"})
    assert isinstance(x, linopy.variables.Variable)
    assert x.dims == ("second",)


def test_variable_set_index(x: linopy.Variable) -> None:
    x = x.assign_coords({"second": ("first", x.indexes["first"] + 100)})
    x = x.set_index({"multi": ["first", "second"]})
    assert isinstance(x, linopy.variables.Variable)
    assert x.dims == ("multi",)
    assert isinstance(x.indexes["multi"], pd.MultiIndex)


def test_isnull(x: linopy.Variable) -> None:
    x = x.where([True] * 4 + [False] * 6)
    assert isinstance(x.isnull(), xr.DataArray)
    assert (x.isnull() == [False] * 4 + [True] * 6).all()


def test_variable_fillna(x: linopy.Variable) -> None:
    x = x.where([True] * 4 + [False] * 6)

    isinstance(x.fillna(x.at[0]), linopy.variables.Variable)


def test_variable_bfill(x: linopy.Variable) -> None:
    x = x.where([False] * 4 + [True] * 6)
    x = x.bfill("first")
    assert isinstance(x, linopy.variables.Variable)
    assert x.labels[2] == x.labels[4]
    assert x.labels[2] != x.labels[5]


def test_variable_broadcast_like(x: linopy.Variable) -> None:
    result = x.broadcast_like(x.labels)
    assert isinstance(result, linopy.variables.Variable)


def test_variable_ffill(x: linopy.Variable) -> None:
    x = x.where([True] * 4 + [False] * 6)
    x = x.ffill("first")
    assert isinstance(x, linopy.variables.Variable)
    assert x.labels[9] == x.labels[3]
    assert x.labels[3] != x.labels[2]


def test_variable_expand_dims(x: linopy.Variable) -> None:
    result = x.expand_dims("new_dim")
    assert isinstance(result, linopy.variables.Variable)
    assert result.dims == ("new_dim", "first")


def test_variable_stack(x: linopy.Variable) -> None:
    result = x.expand_dims("new_dim").stack(new=("new_dim", "first"))
    assert isinstance(result, linopy.variables.Variable)
    assert result.dims == ("new",)


def test_variable_unstack(x: linopy.Variable) -> None:
    result = x.expand_dims("new_dim").stack(new=("new_dim", "first")).unstack("new")
    assert isinstance(result, linopy.variables.Variable)
    assert result.dims == ("new_dim", "first")


def test_variable_flat(x: linopy.Variable) -> None:
    result = x.flat
    assert isinstance(result, pd.DataFrame)
    assert len(result) == x.size


def test_variable_polars(x: linopy.Variable) -> None:
    result = x.to_polars()
    assert isinstance(result, pl.DataFrame)
    assert len(result) == x.size


def test_variable_sanitize(x: linopy.Variable) -> None:
    # convert intentionally to float with nans
    fill_value: dict[str, str | int | float] = {
        "labels": np.nan,
        "lower": np.nan,
        "upper": np.nan,
    }
    x = x.where([True] * 4 + [False] * 6, fill_value)
    x = x.sanitize()
    assert isinstance(x, linopy.variables.Variable)
    assert x.labels[9] == -1


def test_variable_iterate_slices(x: linopy.Variable) -> None:
    slices = x.iterate_slices(slice_size=2)
    for s in slices:
        assert isinstance(s, linopy.variables.Variable)
        assert s.size <= 2


def test_variable_addition(x: linopy.Variable) -> None:
    expr1 = x + 1
    assert isinstance(expr1, linopy.expressions.LinearExpression)
    expr2 = 1 + x
    assert isinstance(expr2, linopy.expressions.LinearExpression)
    assert_linequal(expr1, expr2)

    assert x.__radd__(object()) is NotImplemented
    assert x.__add__(object()) is NotImplemented


def test_variable_subtraction(x: linopy.Variable) -> None:
    expr1 = -x + 1
    assert isinstance(expr1, linopy.expressions.LinearExpression)
    expr2 = 1 - x
    assert isinstance(expr2, linopy.expressions.LinearExpression)
    assert_linequal(expr1, expr2)

    assert x.__rsub__(object()) is NotImplemented
    assert x.__sub__(object()) is NotImplemented


def test_variable_multiplication(x: linopy.Variable) -> None:
    expr1 = x * 2
    assert isinstance(expr1, linopy.expressions.LinearExpression)
    expr2 = 2 * x
    assert isinstance(expr2, linopy.expressions.LinearExpression)
    assert_linequal(expr1, expr2)

    expr3 = x * x
    assert isinstance(expr3, linopy.expressions.QuadraticExpression)

    assert x.__rmul__(object()) is NotImplemented
    assert x.__mul__(object()) is NotImplemented
