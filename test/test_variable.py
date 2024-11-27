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
from xarray.testing import assert_equal

import linopy
import linopy.variables
from linopy import Model


@pytest.fixture
def m():
    m = Model()
    m.add_variables(coords=[pd.RangeIndex(10, name="first")], name="x")
    m.add_variables(coords=[pd.Index([1, 2, 3], name="second")], name="y")
    m.add_variables(0, 10, name="z")
    return m


@pytest.fixture
def x(m):
    return m.variables["x"]


@pytest.fixture
def z(m):
    return m.variables["z"]


def test_variable_repr(x):
    x.__repr__()


def test_variable_inherited_properties(x):
    assert isinstance(x.attrs, dict)
    assert isinstance(x.coords, xr.Coordinates)
    assert isinstance(x.indexes, xr.core.indexes.Indexes)
    assert isinstance(x.sizes, xr.core.utils.Frozen)
    assert isinstance(x.shape, tuple)
    assert isinstance(x.size, int)
    assert isinstance(x.dims, tuple)
    assert isinstance(x.ndim, int)


def test_variable_labels(x):
    isinstance(x.labels, xr.DataArray)


def test_variable_data(x):
    isinstance(x.data, xr.DataArray)


def test_wrong_variable_init(m, x):
    # wrong data type
    with pytest.raises(ValueError):
        linopy.Variable(x.labels.values, m, "")

    # no model
    with pytest.raises(ValueError):
        linopy.Variable(x.labels, None, "")


def test_variable_getter(x, z):
    with pytest.warns(FutureWarning):
        assert isinstance(x[0], linopy.variables.ScalarVariable)
        assert isinstance(z[0], linopy.variables.ScalarVariable)

    assert isinstance(x.at[0], linopy.variables.ScalarVariable)


def test_variable_getter_slice(x):
    res = x[:5]
    assert isinstance(res, linopy.Variable)
    assert res.size == 5


def test_variable_getter_slice_with_step(x):
    res = x[::2]
    assert isinstance(res, linopy.Variable)
    assert res.size == 5


def test_variables_getter_list(x):
    res = x[[1, 2, 3]]
    assert isinstance(res, linopy.Variable)
    assert res.size == 3


def test_variable_getter_invalid_shape(x):
    with pytest.raises(AssertionError):
        x.at[0, 0]


def test_variable_loc(x):
    assert isinstance(x.loc[[1, 2, 3]], linopy.Variable)


def test_variable_sel(x):
    assert isinstance(x.sel(first=[1, 2, 3]), linopy.Variable)


def test_variable_isel(x):
    assert isinstance(x.isel(first=[1, 2, 3]), linopy.Variable)
    assert_equal(
        x.isel(first=[0, 1]).labels,
        x.sel(first=[0, 1]).labels,
    )


def test_variable_upper_getter(z):
    assert z.upper.item() == 10


def test_variable_lower_getter(z):
    assert z.lower.item() == 0


def test_variable_upper_setter(z):
    z.upper = 20
    assert z.upper.item() == 20


def test_variable_lower_setter(z):
    z.lower = 8
    assert z.lower == 8


def test_variable_upper_setter_with_array(x):
    idx = pd.RangeIndex(10, name="first")
    upper = pd.Series(range(25, 35), index=idx)
    x.upper = upper
    assert isinstance(x.upper, xr.DataArray)
    assert (x.upper == upper).all()


def test_variable_upper_setter_with_array_invalid_dim(x):
    with pytest.raises(ValueError):
        upper = pd.Series(range(25, 35))
        x.upper = upper


def test_variable_lower_setter_with_array(x):
    idx = pd.RangeIndex(10, name="first")
    lower = pd.Series(range(15, 25), index=idx)
    x.lower = lower
    assert isinstance(x.lower, xr.DataArray)
    assert (x.lower == lower).all()


def test_variable_lower_setter_with_array_invalid_dim(x):
    with pytest.raises(ValueError):
        lower = pd.Series(range(15, 25))
        x.lower = lower


def test_variable_sum(x):
    res = x.sum()
    assert res.nterm == 10


def test_variable_sum_warn_using_dims(x):
    with pytest.warns(DeprecationWarning):
        x.sum(dims="first")


def test_variable_sum_warn_unknown_kwargs(x):
    with pytest.raises(ValueError):
        x.sum(unknown_kwarg="first")


def test_fill_value():
    isinstance(linopy.variables.Variable._fill_value, dict)

    with pytest.warns(DeprecationWarning):
        linopy.variables.Variable.fill_value


def test_variable_where(x):
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
        x.where([True] * 4 + [False] * 6, 0)


def test_variable_shift(x):
    x = x.shift(first=3)
    assert isinstance(x, linopy.variables.Variable)
    assert x.labels[0] == -1


def test_variable_swap_dims(x):
    x = x.assign_coords({"second": ("first", x.indexes["first"] + 100)})
    x = x.swap_dims({"first": "second"})
    assert isinstance(x, linopy.variables.Variable)
    assert x.dims == ("second",)


def test_variable_set_index(x):
    x = x.assign_coords({"second": ("first", x.indexes["first"] + 100)})
    x = x.set_index({"multi": ["first", "second"]})
    assert isinstance(x, linopy.variables.Variable)
    assert x.dims == ("multi",)
    assert isinstance(x.indexes["multi"], pd.MultiIndex)


def test_isnull(x):
    x = x.where([True] * 4 + [False] * 6)
    assert isinstance(x.isnull(), xr.DataArray)
    assert (x.isnull() == [False] * 4 + [True] * 6).all()


def test_variable_fillna(x):
    x = x.where([True] * 4 + [False] * 6)

    isinstance(x.fillna(x.at[0]), linopy.variables.Variable)


def test_variable_bfill(x):
    x = x.where([False] * 4 + [True] * 6)
    x = x.bfill("first")
    assert isinstance(x, linopy.variables.Variable)
    assert x.labels[2] == x.labels[4]
    assert x.labels[2] != x.labels[5]


def test_variable_broadcast_like(x):
    result = x.broadcast_like(x.labels)
    assert isinstance(result, linopy.variables.Variable)


def test_variable_ffill(x):
    x = x.where([True] * 4 + [False] * 6)
    x = x.ffill("first")
    assert isinstance(x, linopy.variables.Variable)
    assert x.labels[9] == x.labels[3]
    assert x.labels[3] != x.labels[2]


def test_variable_expand_dims(x):
    result = x.expand_dims("new_dim")
    assert isinstance(result, linopy.variables.Variable)
    assert result.dims == ("new_dim", "first")


def test_variable_stack(x):
    result = x.expand_dims("new_dim").stack(new=("new_dim", "first"))
    assert isinstance(result, linopy.variables.Variable)
    assert result.dims == ("new",)


def test_variable_flat(x):
    result = x.flat
    assert isinstance(result, pd.DataFrame)
    assert len(result) == x.size


def test_variable_polars(x):
    result = x.to_polars()
    assert isinstance(result, pl.DataFrame)
    assert len(result) == x.size


def test_variable_sanitize(x):
    # convert intentionally to float with nans
    fill_value = {"labels": np.nan, "lower": np.nan, "upper": np.nan}
    x = x.where([True] * 4 + [False] * 6, fill_value)
    x = x.sanitize()
    assert isinstance(x, linopy.variables.Variable)
    assert x.labels[9] == -1


def test_variable_iterate_slices(x):
    slices = x.iterate_slices(slice_size=2)
    for s in slices:
        assert isinstance(s, linopy.variables.Variable)
        assert s.size <= 2
