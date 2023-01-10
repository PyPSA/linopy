#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:06:36 2021.

@author: fabian
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.testing import assert_equal

from linopy import LinearExpression, Model, merge
from linopy.testing import assert_linequal


@pytest.fixture
def m():
    m = Model()

    m.add_variables(pd.Series([0, 0]), 1, name="x")
    m.add_variables(4, pd.Series([8, 10]), name="y")
    m.add_variables(0, pd.DataFrame([[1, 2], [3, 4], [5, 6]]).T, name="z")
    m.add_variables(coords=[pd.RangeIndex(20, name="dim_2")], name="v")

    idx = pd.MultiIndex.from_product([[1, 2], ["a", "b"]], names=("level1", "level2"))
    idx.name = "dim_3"
    m.add_variables(coords=[idx], name="u")
    return m


@pytest.fixture
def x(m):
    return m.variables["x"]


@pytest.fixture
def y(m):
    return m.variables["y"]


@pytest.fixture
def z(m):
    return m.variables["z"]


@pytest.fixture
def v(m):
    return m.variables["v"]


@pytest.fixture
def u(m):
    return m.variables["u"]


def test_empty_linexpr(m):
    LinearExpression(None, m)


def test_linexpr_with_wrong_data(m):
    with pytest.raises(ValueError):
        LinearExpression(1, m)

    with pytest.raises(ValueError):
        LinearExpression(xr.Dataset({"a": [1]}), m)

    coeffs = xr.DataArray([1, 2], dims=["a"])
    vars = xr.DataArray([1, 2], dims=["a"])
    data = xr.Dataset({"coeffs": coeffs, "vars": vars})
    with pytest.raises(ValueError):
        LinearExpression(data, m)

    coords = [pd.Index([0], name="a"), pd.Index([1, 2], name="_term")]
    coeffs = xr.DataArray(np.array([[1, 2]]), coords=coords)
    vars = xr.DataArray(np.array([[1, 2]]), coords=coords)
    data = xr.Dataset({"coeffs": coeffs, "vars": vars})
    with pytest.raises(ValueError):
        LinearExpression(data, None)


def test_repr(m):
    expr = m.linexpr((10, "x"), (1, "y"))
    expr.__repr__()


def test_fill_value():
    isinstance(LinearExpression.fill_value, dict)


def test_linexpr_with_scalars(m):
    expr = m.linexpr((10, "x"), (1, "y"))
    target = xr.DataArray(
        [[10, 1], [10, 1]], coords={"dim_0": [0, 1]}, dims=["dim_0", "_term"]
    )
    assert_equal(expr.coeffs, target)


def test_linexpr_with_series(m, v):
    lhs = pd.Series(np.arange(20)), v
    expr = m.linexpr(lhs)
    isinstance(expr, LinearExpression)


def test_linexpr_with_dataframe(m, z):
    lhs = pd.DataFrame(z.labels), z
    expr = m.linexpr(lhs)
    isinstance(expr, LinearExpression)


def test_linexpr_duplicated_index(m):
    expr = m.linexpr((10, "x"), (-1, "x"))
    assert (expr.data._term == [0, 1]).all()


def test_linear_expression_with_multiplication(x):
    expr = 1 * x
    assert isinstance(expr, LinearExpression)
    assert expr.nterm == 1
    assert len(expr.vars.dim_0) == x.shape[0]

    expr = x * 1
    assert isinstance(expr, LinearExpression)

    expr = x / 1
    assert isinstance(expr, LinearExpression)

    expr = x / 1.0
    assert isinstance(expr, LinearExpression)

    expr = np.array([1, 2]) * x
    assert isinstance(expr, LinearExpression)


def test_linear_expression_with_addition(m, x, y):
    expr = 10 * x + y
    assert isinstance(expr, LinearExpression)
    assert_linequal(expr, m.linexpr((10, "x"), (1, "y")))

    expr = x + 8 * y
    assert isinstance(expr, LinearExpression)
    assert_linequal(expr, m.linexpr((1, "x"), (8, "y")))

    expr = x + y
    assert isinstance(expr, LinearExpression)
    assert_linequal(expr, m.linexpr((1, "x"), (1, "y")))


def test_linear_expression_with_subtraction(m, x, y):
    expr = x - y
    assert isinstance(expr, LinearExpression)
    assert_linequal(expr, m.linexpr((1, "x"), (-1, "y")))

    expr = -x - 8 * y
    assert isinstance(expr, LinearExpression)
    assert_linequal(expr, m.linexpr((-1, "x"), (-8, "y")))


def test_linear_expression_multi_indexed(u):
    expr = 3 * u + 1 * u
    assert isinstance(expr, LinearExpression)


def test_linear_expression_with_errors(m, x):
    with pytest.raises(TypeError):
        x.to_linexpr(x)

    with pytest.raises(TypeError):
        x + 10

    with pytest.raises(TypeError):
        x - 10

    with pytest.raises(TypeError):
        x * x

    with pytest.raises(TypeError):
        x / x

    with pytest.raises(TypeError):
        x * (1 * x)

    with pytest.raises(TypeError):
        x / (1 * x)

    with pytest.raises(TypeError):
        m.linexpr((10, x.labels), (1, "y"))


def test_linear_expression_from_rule(m, x, y):
    def bound(m, i):
        if i == 1:
            return (i - 1) * x[i - 1] + y[i] + 1 * x[i]
        else:
            return i * x[i] - y[i]

    expr = LinearExpression.from_rule(m, bound, x.coords)
    assert isinstance(expr, LinearExpression)
    assert expr.nterm == 3
    repr(expr)  # test repr


def test_linear_expression_from_rule_with_return_none(m, x, y):
    # with return type None
    def bound(m, i):
        if i == 1:
            return (i - 1) * x[i - 1] + y[i]

    expr = LinearExpression.from_rule(m, bound, x.coords)
    assert isinstance(expr, LinearExpression)
    assert (expr.vars[0] == -1).all()
    assert (expr.vars[1] != -1).all()
    assert expr.coeffs[0].isnull().all()
    assert expr.coeffs[1].notnull().all()
    repr(expr)  # test repr


def test_linear_expression_addition(x, y, z):
    expr = 10 * x + y
    other = 2 * y + z
    res = expr + other

    assert res.nterm == expr.nterm + other.nterm
    assert (res.coords["dim_0"] == expr.coords["dim_0"]).all()
    assert (res.coords["dim_1"] == other.coords["dim_1"]).all()
    assert res.data.notnull().all().to_array().all()

    assert isinstance(x - expr, LinearExpression)
    assert isinstance(x + expr, LinearExpression)


def test_linear_expression_addition_invalid(x, y, z):
    expr = 10 * x + y

    with pytest.raises(TypeError):
        expr + 10
    with pytest.raises(TypeError):
        expr - 10


def test_linear_expression_substraction(x, y, z):
    expr = 10 * x + y
    other = 2 * y - z
    res = expr - other

    assert res.nterm == expr.nterm + other.nterm
    assert (res.coords["dim_0"] == expr.coords["dim_0"]).all()
    assert (res.coords["dim_1"] == other.coords["dim_1"]).all()
    assert res.data.notnull().all().to_array().all()


def test_linear_expression_sum(x, y, z):
    expr = 10 * x + y + z
    res = expr.sum("dim_0")

    assert res.size == expr.size
    assert res.nterm == expr.nterm * len(expr.data.dim_0)

    res = expr.sum()
    assert res.size == expr.size
    assert res.nterm == expr.size
    assert res.data.notnull().all().to_array().all()

    assert_linequal(expr.sum(["dim_0", "_term"]), expr.sum("dim_0"))


def test_linear_expression_sum_drop_zeros(z):
    coeff = xr.zeros_like(z.labels)
    coeff[1, 0] = 3
    coeff[0, 2] = 5
    expr = coeff * z

    res = expr.sum("dim_0", drop_zeros=True)
    assert res.nterm == 1

    res = expr.sum("dim_1", drop_zeros=True)
    assert res.nterm == 1

    coeff[1, 2] = 4
    expr.data["coeffs"] = coeff
    res = expr.sum()

    res = expr.sum("dim_0", drop_zeros=True)
    assert res.nterm == 2

    res = expr.sum("dim_1", drop_zeros=True)
    assert res.nterm == 2


def test_linear_expression_multiplication(x, y, z):
    expr = 10 * x + y + z
    mexpr = expr * 10
    assert (mexpr.coeffs.sel(dim_1=0, dim_0=0, _term=0) == 100).item()

    mexpr = 10 * expr
    assert (mexpr.coeffs.sel(dim_1=0, dim_0=0, _term=0) == 100).item()

    mexpr = expr / 100
    assert (mexpr.coeffs.sel(dim_1=0, dim_0=0, _term=0) == 1 / 10).item()

    mexpr = expr / 100.0
    assert (mexpr.coeffs.sel(dim_1=0, dim_0=0, _term=0) == 1 / 10).item()


def test_linear_expression_multiplication_invalid(x, y, z):
    expr = 10 * x + y + z

    with pytest.raises(TypeError):
        expr = 10 * x + y + z
        expr * x

    with pytest.raises(TypeError):
        expr = 10 * x + y + z
        expr / x


def test_linear_expression_where(v):
    expr = np.arange(20) * v
    expr = expr.where(expr.coeffs >= 10)
    assert isinstance(expr, LinearExpression)
    assert expr.nterm == 1

    expr = np.arange(20) * v
    expr = expr.where(expr.coeffs >= 10, drop=True).sum()
    assert isinstance(expr, LinearExpression)
    assert expr.nterm == 10


def test_linear_expression_shift(v):
    shifted = v.to_linexpr().shift(dim_2=2)
    assert shifted.nterm == 1
    assert shifted.coeffs.loc[:1].isnull().all()
    assert (shifted.vars.loc[:1] == -1).all()


def test_linear_expression_diff(v):
    diff = v.to_linexpr().diff("dim_2")
    assert diff.nterm == 2


def test_linear_expression_diff(v):
    diff = v.diff("dim_2")
    assert diff.nterm == 2


def test_linear_expression_groupby(v):
    expr = 1 * v
    groups = xr.DataArray([1] * 10 + [2] * 10, coords=v.coords)
    grouped = expr.groupby(groups).sum()
    assert "group" in grouped.dims
    assert (grouped.data.group == [1, 2]).all()
    assert grouped.data._term.size == 10


def test_linear_expression_groupby_asymmetric(v):
    expr = 1 * v
    # now asymetric groups which result in different nterms
    groups = xr.DataArray([1] * 12 + [2] * 8, coords=v.coords)
    grouped = expr.groupby(groups).sum()
    assert "group" in grouped.dims
    # first group must be full with vars
    assert (grouped.data.sel(group=1) > 0).all()
    # the last 4 entries of the second group must be empty, i.e. -1
    assert (grouped.data.sel(group=2).isel(_term=slice(None, -4)).vars >= 0).all()
    assert (grouped.data.sel(group=2).isel(_term=slice(-4, None)).vars == -1).all()
    assert grouped.data._term.size == 12


def test_linear_expression_groupby_roll(v):
    expr = 1 * v
    groups = xr.DataArray([1] * 10 + [2] * 10, coords=v.coords)
    grouped = expr.groupby(groups).roll(dim_2=1)
    assert grouped.nterm == 1
    assert grouped.vars[0].item() == 19


def test_linear_expression_groupby_from_variable(v):
    groups = xr.DataArray([1] * 10 + [2] * 10, coords=v.coords)
    grouped = v.groupby(groups).sum()
    assert "group" in grouped.dims
    assert (grouped.data.group == [1, 2]).all()
    assert grouped.data._term.size == 10


def test_linear_expression_rolling(v):
    expr = 1 * v
    rolled = expr.rolling(dim_2=2).sum()
    assert rolled.nterm == 2

    rolled = expr.rolling(dim_2=3).sum()
    assert rolled.nterm == 3


def test_linear_expression_rolling_from_variable(v):
    rolled = v.rolling(dim_2=2).sum()
    assert rolled.nterm == 2


def test_linear_expression_sanitize(x, y, z):
    expr = 10 * x + y + z
    assert isinstance(expr.sanitize(), LinearExpression)


def test_merge(x, y, z):
    expr1 = (10 * x + y).sum("dim_0")
    expr2 = z.sum("dim_0")

    res = merge(expr1, expr2)
    assert res.nterm == 6

    res = merge([expr1, expr2])
    assert res.nterm == 6

    # now concat with same length of terms
    expr1 = z.sel(dim_0=0).sum("dim_1")
    expr2 = z.sel(dim_0=1).sum("dim_1")

    res = merge(expr1, expr2, dim="dim_1")
    assert res.nterm == 3

    # now with different length of terms
    expr1 = z.sel(dim_0=0, dim_1=slice(0, 1)).sum("dim_1")
    expr2 = z.sel(dim_0=1).sum("dim_1")

    res = merge(expr1, expr2, dim="dim_1")
    assert res.nterm == 3
    assert res.sel(dim_1=0).vars[2].item() == -1


# -------------------------------- deprecated -------------------------------- #


def test_rolling_sum_variable_deprecated(v):
    rolled = v.rolling_sum(dim_2=2)
    assert rolled.nterm == 2


def test_linear_expression_rolling_sum_deprecated(x, v):
    rolled = v.to_linexpr().rolling_sum(dim_2=2)
    assert rolled.nterm == 2

    # multi-dimensional rolling with non-scalar _term dimension
    expr = 10 * x + v
    rolled = expr.rolling_sum(dim_2=3)
    assert rolled.nterm == 6


def test_variable_groupby_sum_deprecated(v):
    groups = xr.DataArray([1] * 10 + [2] * 10, coords=v.coords)
    grouped = v.groupby_sum(groups)
    assert "group" in grouped.dims
    assert (grouped.data.group == [1, 2]).all()
    assert grouped.data._term.size == 10


def test_linear_expression_groupby_sum_deprecated(v):
    groups = xr.DataArray([1] * 10 + [2] * 10, coords=v.coords)
    grouped = v.to_linexpr().groupby_sum(groups)
    assert "group" in grouped.dims
    assert (grouped.data.group == [1, 2]).all()
    assert grouped.data._term.size == 10

    # now asymetric groups which result in different nterms
    groups = xr.DataArray([1] * 12 + [2] * 8, coords=v.coords)
    grouped = v.to_linexpr().groupby_sum(groups)
    assert "group" in grouped.dims
    # first group must be full with vars
    assert (grouped.data.sel(group=1) > 0).all()
    # the last 4 entries of the second group must be empty, i.e. -1
    assert (grouped.data.sel(group=2).isel(_term=slice(None, -4)).vars >= 0).all()
    assert (grouped.data.sel(group=2).isel(_term=slice(-4, None)).vars == -1).all()
    assert grouped.data._term.size == 12
