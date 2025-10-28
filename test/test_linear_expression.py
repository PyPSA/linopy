#!/usr/bin/env python3
"""
Created on Wed Mar 17 17:06:36 2021.

@author: fabian
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
from xarray.testing import assert_equal

from linopy import LinearExpression, Model, QuadraticExpression, Variable, merge
from linopy.constants import HELPER_DIMS, TERM_DIM
from linopy.expressions import ScalarLinearExpression
from linopy.testing import assert_linequal, assert_quadequal
from linopy.variables import ScalarVariable


@pytest.fixture
def m() -> Model:
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
def x(m: Model) -> Variable:
    return m.variables["x"]


@pytest.fixture
def y(m: Model) -> Variable:
    return m.variables["y"]


@pytest.fixture
def z(m: Model) -> Variable:
    return m.variables["z"]


@pytest.fixture
def v(m: Model) -> Variable:
    return m.variables["v"]


@pytest.fixture
def u(m: Model) -> Variable:
    return m.variables["u"]


def test_empty_linexpr(m: Model) -> None:
    LinearExpression(None, m)


def test_linexpr_with_wrong_data(m: Model) -> None:
    with pytest.raises(ValueError):
        LinearExpression(xr.Dataset({"a": [1]}), m)

    coeffs = xr.DataArray([1, 2], dims=["a"])
    vars = xr.DataArray([1, 2], dims=["a"])
    data = xr.Dataset({"coeffs": coeffs, "vars": vars})
    with pytest.raises(ValueError):
        LinearExpression(data, m)

    # with model as None
    coeffs = xr.DataArray(np.array([1, 2]), dims=[TERM_DIM])
    vars = xr.DataArray(np.array([1, 2]), dims=[TERM_DIM])
    data = xr.Dataset({"coeffs": coeffs, "vars": vars})
    with pytest.raises(ValueError):
        LinearExpression(data, None)  # type: ignore


def test_linexpr_with_helper_dims_as_coords(m: Model) -> None:
    coords = [pd.Index([0], name="a"), pd.Index([1, 2], name=TERM_DIM)]
    coeffs = xr.DataArray(np.array([[1, 2]]), coords=coords)
    vars = xr.DataArray(np.array([[1, 2]]), coords=coords)

    data = xr.Dataset({"coeffs": coeffs, "vars": vars})
    assert set(HELPER_DIMS).intersection(set(data.coords))

    expr = LinearExpression(data, m)
    assert not set(HELPER_DIMS).intersection(set(expr.data.coords))


def test_linexpr_with_data_without_coords(m: Model) -> None:
    lhs = 1 * m["x"]
    vars = xr.DataArray(lhs.vars.values, dims=["dim_0", TERM_DIM])
    coeffs = xr.DataArray(lhs.coeffs.values, dims=["dim_0", TERM_DIM])
    data = xr.Dataset({"vars": vars, "coeffs": coeffs})
    expr = LinearExpression(data, m)
    assert_linequal(expr, lhs)


def test_linexpr_from_constant_dataarray(m: Model) -> None:
    const = xr.DataArray([1, 2], dims=["dim_0"])
    expr = LinearExpression(const, m)
    assert (expr.const == const).all()
    assert expr.nterm == 0


def test_linexpr_from_constant_pl_series(m: Model) -> None:
    const = pl.Series([1, 2])
    expr = LinearExpression(const, m)
    assert (expr.const == const.to_numpy()).all()
    assert expr.nterm == 0


def test_linexpr_from_constant_pandas_series(m: Model) -> None:
    const = pd.Series([1, 2], index=pd.RangeIndex(2, name="dim_0"))
    expr = LinearExpression(const, m)
    assert (expr.const == const).all()
    assert expr.nterm == 0


def test_linexpr_from_constant_pandas_dataframe(m: Model) -> None:
    const = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    expr = LinearExpression(const, m)
    assert (expr.const == const).all()
    assert expr.nterm == 0


def test_linexpr_from_constant_numpy_array(m: Model) -> None:
    const = np.array([1, 2])
    expr = LinearExpression(const, m)
    assert (expr.const == const).all()
    assert expr.nterm == 0


def test_linexpr_from_constant_scalar(m: Model) -> None:
    const = 1
    expr = LinearExpression(const, m)
    assert (expr.const == const).all()
    assert expr.nterm == 0


def test_repr(m: Model) -> None:
    expr = m.linexpr((10, "x"), (1, "y"))
    expr.__repr__()


def test_fill_value() -> None:
    isinstance(LinearExpression._fill_value, dict)


def test_linexpr_with_scalars(m: Model) -> None:
    expr = m.linexpr((10, "x"), (1, "y"))
    target = xr.DataArray(
        [[10, 1], [10, 1]], coords={"dim_0": [0, 1]}, dims=["dim_0", TERM_DIM]
    )
    assert_equal(expr.coeffs, target)


def test_linexpr_with_variables_and_constants(
    m: Model, x: Variable, y: Variable
) -> None:
    expr = m.linexpr((10, x), (1, y), 2)
    assert (expr.const == 2).all()


def test_linexpr_with_series(m: Model, v: Variable) -> None:
    lhs = pd.Series(np.arange(20)), v
    expr = m.linexpr(lhs)
    isinstance(expr, LinearExpression)


def test_linexpr_with_dataframe(m: Model, z: Variable) -> None:
    lhs = pd.DataFrame(z.labels), z
    expr = m.linexpr(lhs)
    isinstance(expr, LinearExpression)


def test_linexpr_duplicated_index(m: Model) -> None:
    expr = m.linexpr((10, "x"), (-1, "x"))
    assert (expr.data._term == [0, 1]).all()


def test_linear_expression_with_multiplication(x: Variable) -> None:
    expr = 1 * x
    assert isinstance(expr, LinearExpression)
    assert expr.nterm == 1
    assert len(expr.vars.dim_0) == x.shape[0]

    expr = x * 1
    assert isinstance(expr, LinearExpression)

    expr2 = x.mul(1)
    assert_linequal(expr, expr2)

    expr3 = expr.mul(1)
    assert_linequal(expr, expr3)

    expr = x / 1
    assert isinstance(expr, LinearExpression)

    expr = x / 1.0
    assert isinstance(expr, LinearExpression)

    expr2 = x.div(1)
    assert_linequal(expr, expr2)

    expr3 = expr.div(1)
    assert_linequal(expr, expr3)

    expr = np.array([1, 2]) * x
    assert isinstance(expr, LinearExpression)

    expr = np.array(1) * x
    assert isinstance(expr, LinearExpression)

    expr = xr.DataArray(np.array([[1, 2], [2, 3]])) * x
    assert isinstance(expr, LinearExpression)

    expr = pd.Series([1, 2], index=pd.RangeIndex(2, name="dim_0")) * x
    assert isinstance(expr, LinearExpression)

    quad = x * x
    assert isinstance(quad, QuadraticExpression)

    with pytest.raises(TypeError):
        quad * quad

    expr = x * 1
    assert isinstance(expr, LinearExpression)
    assert expr.__mul__(object()) is NotImplemented
    assert expr.__rmul__(object()) is NotImplemented


def test_linear_expression_with_addition(m: Model, x: Variable, y: Variable) -> None:
    expr = 10 * x + y
    assert isinstance(expr, LinearExpression)
    assert_linequal(expr, m.linexpr((10, "x"), (1, "y")))

    expr = x + 8 * y
    assert isinstance(expr, LinearExpression)
    assert_linequal(expr, m.linexpr((1, "x"), (8, "y")))

    expr = x + y
    assert isinstance(expr, LinearExpression)
    assert_linequal(expr, m.linexpr((1, "x"), (1, "y")))

    expr2 = x.add(y)
    assert_linequal(expr, expr2)

    expr3 = (x * 1).add(y)
    assert_linequal(expr, expr3)

    expr3 = x + (x * x)
    assert isinstance(expr3, QuadraticExpression)


def test_linear_expression_with_raddition(m: Model, x: Variable) -> None:
    expr = x * 1.0
    expr_2: LinearExpression = 10.0 + expr
    assert isinstance(expr, LinearExpression)
    expr_3: LinearExpression = expr + 10.0
    assert_linequal(expr_2, expr_3)


def test_linear_expression_with_subtraction(m: Model, x: Variable, y: Variable) -> None:
    expr = x - y
    assert isinstance(expr, LinearExpression)
    assert_linequal(expr, m.linexpr((1, "x"), (-1, "y")))

    expr2 = x.sub(y)
    assert_linequal(expr, expr2)

    expr3: LinearExpression = x * 1
    expr4 = expr3.sub(y)
    assert_linequal(expr, expr4)

    expr = -x - 8 * y
    assert isinstance(expr, LinearExpression)
    assert_linequal(expr, m.linexpr((-1, "x"), (-8, "y")))


def test_linear_expression_rsubtraction(x: Variable, y: Variable) -> None:
    expr = x * 1.0
    expr_2: LinearExpression = 10.0 - expr
    assert isinstance(expr_2, LinearExpression)
    expr_3: LinearExpression = (expr - 10.0) * -1
    assert_linequal(expr_2, expr_3)
    assert expr.__rsub__(object()) is NotImplemented


def test_linear_expression_with_constant(m: Model, x: Variable, y: Variable) -> None:
    expr = x + 1
    assert isinstance(expr, LinearExpression)
    assert (expr.const == 1).all()

    expr = -x - 8 * y - 10
    assert isinstance(expr, LinearExpression)
    assert (expr.const == -10).all()
    assert expr.nterm == 2


def test_linear_expression_with_constant_multiplication(
    m: Model, x: Variable, y: Variable
) -> None:
    expr = x + 1

    obs = expr * 10
    assert isinstance(obs, LinearExpression)
    assert (obs.const == 10).all()

    obs = expr * pd.Series([1, 2, 3], index=pd.RangeIndex(3, name="new_dim"))
    assert isinstance(obs, LinearExpression)
    assert obs.shape == (2, 3, 1)


def test_linear_expression_multi_indexed(u: Variable) -> None:
    expr = 3 * u + 1 * u
    assert isinstance(expr, LinearExpression)


def test_linear_expression_with_errors(m: Model, x: Variable) -> None:
    with pytest.raises(TypeError):
        x / x

    with pytest.raises(TypeError):
        x / (1 * x)

    with pytest.raises(TypeError):
        m.linexpr((10, x.labels), (1, "y"))

    with pytest.raises(TypeError):
        m.linexpr(a=2)  # type: ignore


def test_linear_expression_from_rule(m: Model, x: Variable, y: Variable) -> None:
    def bound(m: Model, i: int) -> ScalarLinearExpression:
        return (
            (i - 1) * x.at[i - 1] + y.at[i] + 1 * x.at[i]
            if i == 1
            else i * x.at[i] - y.at[i]
        )

    expr = LinearExpression.from_rule(m, bound, x.coords)
    assert isinstance(expr, LinearExpression)
    assert expr.nterm == 3
    repr(expr)  # test repr


def test_linear_expression_from_rule_with_return_none(
    m: Model, x: Variable, y: Variable
) -> None:
    # with return type None
    def bound(m: Model, i: int) -> ScalarLinearExpression | None:
        if i == 1:
            return (i - 1) * x.at[i - 1] + y.at[i]
        return None

    expr = LinearExpression.from_rule(m, bound, x.coords)
    assert isinstance(expr, LinearExpression)
    assert (expr.vars[0] == -1).all()
    assert (expr.vars[1] != -1).all()
    assert expr.coeffs[0].isnull().all()
    assert expr.coeffs[1].notnull().all()
    repr(expr)  # test repr


def test_linear_expression_addition(x: Variable, y: Variable, z: Variable) -> None:
    expr = 10 * x + y
    other = 2 * y + z
    res = expr + other

    assert res.nterm == expr.nterm + other.nterm
    assert (res.coords["dim_0"] == expr.coords["dim_0"]).all()
    assert (res.coords["dim_1"] == other.coords["dim_1"]).all()
    assert res.data.notnull().all().to_array().all()

    res2 = expr.add(other)
    assert_linequal(res, res2)

    assert isinstance(x - expr, LinearExpression)
    assert isinstance(x + expr, LinearExpression)


def test_linear_expression_addition_with_constant(
    x: Variable, y: Variable, z: Variable
) -> None:
    expr = 10 * x + y + 10
    assert (expr.const == 10).all()

    expr = 10 * x + y + np.array([2, 3])
    assert list(expr.const) == [2, 3]

    expr = 10 * x + y + pd.Series([2, 3])
    assert list(expr.const) == [2, 3]


def test_linear_expression_subtraction(x: Variable, y: Variable, z: Variable) -> None:
    expr = 10 * x + y - 10
    assert (expr.const == -10).all()

    expr = 10 * x + y - np.array([2, 3])
    assert list(expr.const) == [-2, -3]

    expr = 10 * x + y - pd.Series([2, 3])
    assert list(expr.const) == [-2, -3]


def test_linear_expression_substraction(
    x: Variable, y: Variable, z: Variable, v: Variable
) -> None:
    expr = 10 * x + y
    other = 2 * y - z
    res = expr - other

    assert res.nterm == expr.nterm + other.nterm
    assert (res.coords["dim_0"] == expr.coords["dim_0"]).all()
    assert (res.coords["dim_1"] == other.coords["dim_1"]).all()
    assert res.data.notnull().all().to_array().all()


def test_linear_expression_sum(
    x: Variable, y: Variable, z: Variable, v: Variable
) -> None:
    expr = 10 * x + y + z
    res = expr.sum("dim_0")

    assert res.size == expr.size
    assert res.nterm == expr.nterm * len(expr.data.dim_0)

    res = expr.sum()
    assert res.size == expr.size
    assert res.nterm == expr.size
    assert res.data.notnull().all().to_array().all()

    assert_linequal(expr.sum(["dim_0", TERM_DIM]), expr.sum("dim_0"))

    # test special case otherride coords
    expr = v.loc[:9] + v.loc[10:]
    assert expr.nterm == 2
    assert len(expr.coords["dim_2"]) == 10


def test_linear_expression_sum_with_const(
    x: Variable, y: Variable, z: Variable, v: Variable
) -> None:
    expr = 10 * x + y + z + 10
    res = expr.sum("dim_0")

    assert res.size == expr.size
    assert res.nterm == expr.nterm * len(expr.data.dim_0)
    assert (res.const == 20).all()

    res = expr.sum()
    assert res.size == expr.size
    assert res.nterm == expr.size
    assert res.data.notnull().all().to_array().all()
    assert (res.const == 60).item()

    assert_linequal(expr.sum(["dim_0", TERM_DIM]), expr.sum("dim_0"))

    # test special case otherride coords
    expr = v.loc[:9] + v.loc[10:]
    assert expr.nterm == 2
    assert len(expr.coords["dim_2"]) == 10


def test_linear_expression_sum_drop_zeros(z: Variable) -> None:
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


def test_linear_expression_sum_warn_using_dims(z: Variable) -> None:
    with pytest.warns(DeprecationWarning):
        (1 * z).sum(dims="dim_0")


def test_linear_expression_sum_warn_unknown_kwargs(z: Variable) -> None:
    with pytest.raises(ValueError):
        (1 * z).sum(unknown_kwarg="dim_0")


def test_linear_expression_power(x: Variable) -> None:
    expr: LinearExpression = x * 1.0
    qd_expr = expr**2
    assert isinstance(qd_expr, QuadraticExpression)

    qd_expr2 = expr.pow(2)
    assert_quadequal(qd_expr, qd_expr2)

    with pytest.raises(ValueError):
        expr**3


def test_linear_expression_multiplication(
    x: Variable, y: Variable, z: Variable
) -> None:
    expr = 10 * x + y + z
    mexpr = expr * 10
    assert (mexpr.coeffs.sel(dim_1=0, dim_0=0, _term=0) == 100).item()

    mexpr = 10 * expr
    assert (mexpr.coeffs.sel(dim_1=0, dim_0=0, _term=0) == 100).item()

    mexpr = expr / 100
    assert (mexpr.coeffs.sel(dim_1=0, dim_0=0, _term=0) == 1 / 10).item()

    mexpr = expr / 100.0
    assert (mexpr.coeffs.sel(dim_1=0, dim_0=0, _term=0) == 1 / 10).item()


def test_matmul_variable_and_const(x: Variable, y: Variable) -> None:
    const = np.array([1, 2])
    expr = x @ const
    assert expr.nterm == 2
    assert_linequal(expr, (x * const).sum())

    assert_linequal(x @ const, (x * const).sum())

    assert_linequal(x.dot(const), x @ const)


def test_matmul_expr_and_const(x: Variable, y: Variable) -> None:
    expr = 10 * x + y
    const = np.array([1, 2])
    res = expr @ const
    target = (10 * x) @ const + y @ const
    assert res.nterm == 4
    assert_linequal(res, target)

    assert_linequal(expr.dot(const), target)


def test_matmul_wrong_input(x: Variable, y: Variable, z: Variable) -> None:
    expr = 10 * x + y + z
    with pytest.raises(TypeError):
        expr @ expr


def test_linear_expression_multiplication_invalid(
    x: Variable, y: Variable, z: Variable
) -> None:
    expr = 10 * x + y + z

    with pytest.raises(TypeError):
        expr = 10 * x + y + z
        expr * expr

    with pytest.raises(TypeError):
        expr = 10 * x + y + z
        expr / x


def test_expression_inherited_properties(x: Variable, y: Variable) -> None:
    expr = 10 * x + y
    assert isinstance(expr.attrs, dict)
    assert isinstance(expr.coords, xr.Coordinates)
    assert isinstance(expr.indexes, xr.core.indexes.Indexes)
    assert isinstance(expr.sizes, xr.core.utils.Frozen)


def test_linear_expression_getitem_single(x: Variable, y: Variable) -> None:
    expr = 10 * x + y + 3
    sel = expr[0]
    assert isinstance(sel, LinearExpression)
    assert sel.nterm == 2
    # one expression with two terms (constant is not counted)
    assert sel.size == 2


def test_linear_expression_getitem_slice(x: Variable, y: Variable) -> None:
    expr = 10 * x + y + 3
    sel = expr[:1]

    assert isinstance(sel, LinearExpression)
    assert sel.nterm == 2
    # one expression with two terms (constant is not counted)
    assert sel.size == 2


def test_linear_expression_getitem_list(x: Variable, y: Variable, z: Variable) -> None:
    expr = 10 * x + z + 10
    sel = expr[:, [0, 2]]
    assert isinstance(sel, LinearExpression)
    assert sel.nterm == 2
    # four expressions with two terms (constant is not counted)
    assert sel.size == 8


def test_linear_expression_loc(x: Variable, y: Variable) -> None:
    expr = x + y
    assert expr.loc[0].size < expr.loc[:5].size


def test_linear_expression_empty(v: Variable) -> None:
    expr = 7 * v
    assert not expr.empty
    assert expr.loc[[]].empty

    with pytest.warns(DeprecationWarning, match="use `.empty` property instead"):
        assert expr.loc[[]].empty()


def test_linear_expression_isnull(v: Variable) -> None:
    expr = np.arange(20) * v
    filter = (expr.coeffs >= 10).any(TERM_DIM)
    expr = expr.where(filter)
    assert expr.isnull().sum() == 10


def test_linear_expression_flat(v: Variable) -> None:
    coeff = np.arange(1, 21)  # use non-zero coefficients
    expr = coeff * v
    df = expr.flat
    assert isinstance(df, pd.DataFrame)
    assert (df.coeffs == coeff).all()


def test_iterate_slices(x: Variable, y: Variable) -> None:
    expr = x + 10 * y
    for s in expr.iterate_slices(slice_size=2):
        assert isinstance(s, LinearExpression)
        assert s.nterm == expr.nterm
        assert s.coord_dims == expr.coord_dims


def test_linear_expression_to_polars(v: Variable) -> None:
    coeff = np.arange(1, 21)  # use non-zero coefficients
    expr = coeff * v
    df = expr.to_polars()
    assert isinstance(df, pl.DataFrame)
    assert (df["coeffs"].to_numpy() == coeff).all()


def test_linear_expression_where(v: Variable) -> None:
    expr = np.arange(20) * v
    filter = (expr.coeffs >= 10).any(TERM_DIM)
    expr = expr.where(filter)
    assert isinstance(expr, LinearExpression)
    assert expr.nterm == 1

    expr = np.arange(20) * v
    expr = expr.where(filter, drop=True).sum()
    assert isinstance(expr, LinearExpression)
    assert expr.nterm == 10


def test_linear_expression_where_with_const(v: Variable) -> None:
    expr = np.arange(20) * v + 10
    filter = (expr.coeffs >= 10).any(TERM_DIM)
    expr = expr.where(filter)
    assert isinstance(expr, LinearExpression)
    assert expr.nterm == 1
    assert expr.const[:10].isnull().all()
    assert (expr.const[10:] == 10).all()

    expr = np.arange(20) * v + 10
    expr = expr.where(filter, drop=True).sum()
    assert isinstance(expr, LinearExpression)
    assert expr.nterm == 10
    assert expr.const == 100


def test_linear_expression_where_scalar_fill_value(v: Variable) -> None:
    expr = np.arange(20) * v + 10
    filter = (expr.coeffs >= 10).any(TERM_DIM)
    expr = expr.where(filter, 200)
    assert isinstance(expr, LinearExpression)
    assert expr.nterm == 1
    assert (expr.const[:10] == 200).all()
    assert (expr.const[10:] == 10).all()


def test_linear_expression_where_array_fill_value(v: Variable) -> None:
    expr = np.arange(20) * v + 10
    filter = (expr.coeffs >= 10).any(TERM_DIM)
    other = expr.coeffs
    expr = expr.where(filter, other)
    assert isinstance(expr, LinearExpression)
    assert expr.nterm == 1
    assert (expr.const[:10] == other[:10]).all()
    assert (expr.const[10:] == 10).all()


def test_linear_expression_where_expr_fill_value(v: Variable) -> None:
    expr = np.arange(20) * v + 10
    expr2 = np.arange(20) * v + 5
    filter = (expr.coeffs >= 10).any(TERM_DIM)
    res = expr.where(filter, expr2)
    assert isinstance(res, LinearExpression)
    assert res.nterm == 1
    assert (res.const[:10] == expr2.const[:10]).all()
    assert (res.const[10:] == 10).all()


def test_where_with_helper_dim_false(v: Variable) -> None:
    expr = np.arange(20) * v
    with pytest.raises(ValueError):
        filter = expr.coeffs >= 10
        expr.where(filter)


def test_linear_expression_shift(v: Variable) -> None:
    shifted = v.to_linexpr().shift(dim_2=2)
    assert shifted.nterm == 1
    assert shifted.coeffs.loc[:1].isnull().all()
    assert (shifted.vars.loc[:1] == -1).all()


def test_linear_expression_swap_dims(v: Variable) -> None:
    expr = v.to_linexpr()
    expr = expr.assign_coords({"second": ("dim_2", expr.indexes["dim_2"] + 100)})
    expr = expr.swap_dims({"dim_2": "second"})
    assert isinstance(expr, LinearExpression)
    assert expr.coord_dims == ("second",)


def test_linear_expression_set_index(v: Variable) -> None:
    expr = v.to_linexpr()
    expr = expr.assign_coords({"second": ("dim_2", expr.indexes["dim_2"] + 100)})
    expr = expr.set_index({"multi": ["dim_2", "second"]})
    assert isinstance(expr, LinearExpression)
    assert expr.coord_dims == ("multi",)
    assert isinstance(expr.indexes["multi"], pd.MultiIndex)


def test_linear_expression_fillna(v: Variable) -> None:
    expr = np.arange(20) * v + 10
    assert expr.const.sum() == 200

    filter = (expr.coeffs >= 10).any(TERM_DIM)
    filtered = expr.where(filter)
    assert isinstance(filtered, LinearExpression)
    assert filtered.const.sum() == 100

    filled = filtered.fillna(10)
    assert isinstance(filled, LinearExpression)
    assert filled.const.sum() == 200
    assert filled.coeffs.isnull().sum() == 10


def test_variable_expand_dims(v: Variable) -> None:
    result = v.to_linexpr().expand_dims("new_dim")
    assert isinstance(result, LinearExpression)
    assert result.coord_dims == ("dim_2", "new_dim")


def test_variable_stack(v: Variable) -> None:
    result = v.to_linexpr().expand_dims("new_dim").stack(new=("new_dim", "dim_2"))
    assert isinstance(result, LinearExpression)
    assert result.coord_dims == ("new",)


def test_linear_expression_unstack(v: Variable) -> None:
    result = v.to_linexpr().expand_dims("new_dim").stack(new=("new_dim", "dim_2"))
    result = result.unstack("new")
    assert isinstance(result, LinearExpression)
    assert result.coord_dims == ("new_dim", "dim_2")


def test_linear_expression_diff(v: Variable) -> None:
    diff = v.to_linexpr().diff("dim_2")
    assert diff.nterm == 2


@pytest.mark.parametrize("use_fallback", [True, False])
def test_linear_expression_groupby(v: Variable, use_fallback: bool) -> None:
    expr = 1 * v
    dim = v.dims[0]
    groups = xr.DataArray([1] * 10 + [2] * 10, coords=v.coords, name=dim)
    grouped = expr.groupby(groups).sum(use_fallback=use_fallback)
    assert dim in grouped.dims
    assert (grouped.data[dim] == [1, 2]).all()
    assert grouped.nterm == 10


@pytest.mark.parametrize("use_fallback", [True, False])
def test_linear_expression_groupby_on_same_name_as_target_dim(
    v: Variable, use_fallback: bool
) -> None:
    expr = 1 * v
    groups = xr.DataArray([1] * 10 + [2] * 10, coords=v.coords)
    grouped = expr.groupby(groups).sum(use_fallback=use_fallback)
    assert "group" in grouped.dims
    assert (grouped.data.group == [1, 2]).all()
    assert grouped.nterm == 10


@pytest.mark.parametrize("use_fallback", [True])
def test_linear_expression_groupby_ndim(z: Variable, use_fallback: bool) -> None:
    # TODO: implement fallback for n-dim groupby, see https://github.com/PyPSA/linopy/issues/299
    expr = 1 * z
    groups = xr.DataArray([[1, 1, 2], [1, 3, 3]], coords=z.coords)
    grouped = expr.groupby(groups).sum(use_fallback=use_fallback)
    assert "group" in grouped.dims
    # there are three groups, 1, 2 and 3, the largest group has 3 elements
    assert (grouped.data.group == [1, 2, 3]).all()
    assert grouped.nterm == 3


@pytest.mark.parametrize("use_fallback", [True, False])
def test_linear_expression_groupby_with_name(v: Variable, use_fallback: bool) -> None:
    expr = 1 * v
    groups = xr.DataArray([1] * 10 + [2] * 10, coords=v.coords, name="my_group")
    grouped = expr.groupby(groups).sum(use_fallback=use_fallback)
    assert "my_group" in grouped.dims
    assert (grouped.data.my_group == [1, 2]).all()
    assert grouped.nterm == 10


@pytest.mark.parametrize("use_fallback", [True, False])
def test_linear_expression_groupby_with_series(v: Variable, use_fallback: bool) -> None:
    expr = 1 * v
    groups = pd.Series([1] * 10 + [2] * 10, index=v.indexes["dim_2"])
    grouped = expr.groupby(groups).sum(use_fallback=use_fallback)
    assert "group" in grouped.dims
    assert (grouped.data.group == [1, 2]).all()
    assert grouped.nterm == 10


@pytest.mark.parametrize("use_fallback", [True, False])
def test_linear_expression_groupby_series_with_name(
    v: Variable, use_fallback: bool
) -> None:
    expr = 1 * v
    groups = pd.Series([1] * 10 + [2] * 10, index=v.indexes[v.dims[0]], name="my_group")
    grouped = expr.groupby(groups).sum(use_fallback=use_fallback)
    assert "my_group" in grouped.dims
    assert (grouped.data.my_group == [1, 2]).all()
    assert grouped.nterm == 10


@pytest.mark.parametrize("use_fallback", [True, False])
def test_linear_expression_groupby_with_series_with_same_group_name(
    v: Variable, use_fallback: bool
) -> None:
    """
    Test that the group by works with a series whose name is the same as
    the dimension to group.
    """
    expr = 1 * v
    groups = pd.Series([1] * 10 + [2] * 10, index=v.indexes["dim_2"])
    groups.name = "dim_2"
    grouped = expr.groupby(groups).sum(use_fallback=use_fallback)
    assert "dim_2" in grouped.dims
    assert (grouped.data.dim_2 == [1, 2]).all()
    assert grouped.nterm == 10


@pytest.mark.parametrize("use_fallback", [True, False])
def test_linear_expression_groupby_with_series_on_multiindex(
    u: Variable, use_fallback: bool
) -> None:
    expr = 1 * u
    len_grouped_dim = len(u.data["dim_3"])
    groups = pd.Series([1] * len_grouped_dim, index=u.indexes["dim_3"])
    grouped = expr.groupby(groups).sum(use_fallback=use_fallback)
    assert "group" in grouped.dims
    assert (grouped.data.group == [1]).all()
    assert grouped.nterm == len_grouped_dim


@pytest.mark.parametrize("use_fallback", [True, False])
def test_linear_expression_groupby_with_dataframe(
    v: Variable, use_fallback: bool
) -> None:
    expr = 1 * v
    groups = pd.DataFrame(
        {"a": [1] * 10 + [2] * 10, "b": list(range(4)) * 5}, index=v.indexes["dim_2"]
    )
    if use_fallback:
        with pytest.raises(ValueError):
            expr.groupby(groups).sum(use_fallback=use_fallback)
        return

    grouped = expr.groupby(groups).sum(use_fallback=use_fallback)
    index = pd.MultiIndex.from_frame(groups)
    assert "group" in grouped.dims
    assert set(grouped.data.group.values) == set(index.values)
    assert grouped.nterm == 3


@pytest.mark.parametrize("use_fallback", [True, False])
def test_linear_expression_groupby_with_dataframe_with_same_group_name(
    v: Variable, use_fallback: bool
) -> None:
    """
    Test that the group by works with a dataframe whose column name is the same as
    the dimension to group.
    """
    expr = 1 * v
    groups = pd.DataFrame(
        {"dim_2": [1] * 10 + [2] * 10, "b": list(range(4)) * 5},
        index=v.indexes["dim_2"],
    )
    if use_fallback:
        with pytest.raises(ValueError):
            expr.groupby(groups).sum(use_fallback=use_fallback)
        return

    grouped = expr.groupby(groups).sum(use_fallback=use_fallback)
    index = pd.MultiIndex.from_frame(groups)
    assert "group" in grouped.dims
    assert set(grouped.data.group.values) == set(index.values)
    assert grouped.nterm == 3


@pytest.mark.parametrize("use_fallback", [True, False])
def test_linear_expression_groupby_with_dataframe_on_multiindex(
    u: Variable, use_fallback: bool
) -> None:
    expr = 1 * u
    len_grouped_dim = len(u.data["dim_3"])
    groups = pd.DataFrame({"a": [1] * len_grouped_dim}, index=u.indexes["dim_3"])

    if use_fallback:
        with pytest.raises(ValueError):
            expr.groupby(groups).sum(use_fallback=use_fallback)
        return
    grouped = expr.groupby(groups).sum(use_fallback=use_fallback)
    assert "group" in grouped.dims
    assert isinstance(grouped.indexes["group"], pd.MultiIndex)
    assert grouped.nterm == len_grouped_dim


@pytest.mark.parametrize("use_fallback", [True, False])
def test_linear_expression_groupby_with_dataarray(
    v: Variable, use_fallback: bool
) -> None:
    expr = 1 * v
    df = pd.DataFrame(
        {"a": [1] * 10 + [2] * 10, "b": list(range(4)) * 5}, index=v.indexes["dim_2"]
    )
    groups = xr.DataArray(df)

    # this should not be the case, see https://github.com/PyPSA/linopy/issues/351
    if use_fallback:
        with pytest.raises((KeyError, IndexError)):
            expr.groupby(groups).sum(use_fallback=use_fallback)
        return

    grouped = expr.groupby(groups).sum(use_fallback=use_fallback)
    index = pd.MultiIndex.from_frame(df)
    assert "group" in grouped.dims
    assert set(grouped.data.group.values) == set(index.values)
    assert grouped.nterm == 3


def test_linear_expression_groupby_with_dataframe_non_aligned(v: Variable) -> None:
    expr = 1 * v
    groups = pd.DataFrame(
        {"a": [1] * 10 + [2] * 10, "b": list(range(4)) * 5}, index=v.indexes["dim_2"]
    )
    target = expr.groupby(groups).sum()

    groups_non_aligned = groups[::-1]
    grouped = expr.groupby(groups_non_aligned).sum()
    assert_linequal(grouped, target)


@pytest.mark.parametrize("use_fallback", [True, False])
def test_linear_expression_groupby_with_const(v: Variable, use_fallback: bool) -> None:
    expr = 1 * v + 15
    groups = xr.DataArray([1] * 10 + [2] * 10, coords=v.coords)
    grouped = expr.groupby(groups).sum(use_fallback=use_fallback)
    assert "group" in grouped.dims
    assert (grouped.data.group == [1, 2]).all()
    assert grouped.nterm == 10
    assert (grouped.const == 150).all()


@pytest.mark.parametrize("use_fallback", [True, False])
def test_linear_expression_groupby_asymmetric(v: Variable, use_fallback: bool) -> None:
    expr = 1 * v
    # now asymetric groups which result in different nterms
    groups = xr.DataArray([1] * 12 + [2] * 8, coords=v.coords)
    grouped = expr.groupby(groups).sum(use_fallback=use_fallback)
    assert "group" in grouped.dims
    # first group must be full with vars
    assert (grouped.data.sel(group=1) > 0).all()
    # the last 4 entries of the second group must be empty, i.e. -1
    assert (grouped.data.sel(group=2).isel(_term=slice(None, -4)).vars >= 0).all()
    assert (grouped.data.sel(group=2).isel(_term=slice(-4, None)).vars == -1).all()
    assert grouped.nterm == 12


@pytest.mark.parametrize("use_fallback", [True, False])
def test_linear_expression_groupby_asymmetric_with_const(
    v: Variable, use_fallback: bool
) -> None:
    expr = 1 * v + 15
    # now asymetric groups which result in different nterms
    groups = xr.DataArray([1] * 12 + [2] * 8, coords=v.coords)
    grouped = expr.groupby(groups).sum(use_fallback=use_fallback)
    assert "group" in grouped.dims
    # first group must be full with vars
    assert (grouped.data.sel(group=1) > 0).all()
    # the last 4 entries of the second group must be empty, i.e. -1
    assert (grouped.data.sel(group=2).isel(_term=slice(None, -4)).vars >= 0).all()
    assert (grouped.data.sel(group=2).isel(_term=slice(-4, None)).vars == -1).all()
    assert grouped.nterm == 12
    assert list(grouped.const) == [180, 120]


def test_linear_expression_groupby_roll(v: Variable) -> None:
    expr = 1 * v
    groups = xr.DataArray([1] * 10 + [2] * 10, coords=v.coords)
    grouped = expr.groupby(groups).roll(dim_2=1)
    assert grouped.nterm == 1
    assert grouped.vars[0].item() == 19


def test_linear_expression_groupby_roll_with_const(v: Variable) -> None:
    expr = 1 * v + np.arange(20)
    groups = xr.DataArray([1] * 10 + [2] * 10, coords=v.coords)
    grouped = expr.groupby(groups).roll(dim_2=1)
    assert grouped.nterm == 1
    assert grouped.vars[0].item() == 19
    assert grouped.const[0].item() == 9


def test_linear_expression_groupby_from_variable(v: Variable) -> None:
    groups = xr.DataArray([1] * 10 + [2] * 10, coords=v.coords)
    grouped = v.groupby(groups).sum()
    assert "group" in grouped.dims
    assert (grouped.data.group == [1, 2]).all()
    assert grouped.nterm == 10


def test_linear_expression_rolling(v: Variable) -> None:
    expr = 1 * v
    rolled = expr.rolling(dim_2=2).sum()
    assert rolled.nterm == 2

    rolled = expr.rolling(dim_2=3).sum()
    assert rolled.nterm == 3

    with pytest.raises(ValueError):
        expr.rolling().sum()


def test_linear_expression_rolling_with_const(v: Variable) -> None:
    expr = 1 * v + 15
    rolled = expr.rolling(dim_2=2).sum()
    assert rolled.nterm == 2
    assert rolled.const[0].item() == 15
    assert (rolled.const[1:] == 30).all()

    rolled = expr.rolling(dim_2=3).sum()
    assert rolled.nterm == 3
    assert rolled.const[0].item() == 15
    assert rolled.const[1].item() == 30
    assert (rolled.const[2:] == 45).all()


def test_linear_expression_rolling_from_variable(v: Variable) -> None:
    rolled = v.rolling(dim_2=2).sum()
    assert rolled.nterm == 2


def test_linear_expression_from_tuples(x: Variable, y: Variable) -> None:
    expr = LinearExpression.from_tuples((10, x), (1, y))
    assert isinstance(expr, LinearExpression)

    with pytest.warns(DeprecationWarning):
        expr2 = LinearExpression.from_tuples((10, x), (1,))
    assert isinstance(expr2, LinearExpression)
    assert (expr2.const == 1).all()

    expr3 = LinearExpression.from_tuples((10, x), 1)
    assert isinstance(expr3, LinearExpression)
    assert_linequal(expr2, expr3)

    expr4 = LinearExpression.from_tuples((10, x), (1, y), 1)
    assert isinstance(expr4, LinearExpression)
    assert (expr4.const == 1).all()

    expr5 = LinearExpression.from_tuples(1, model=x.model)
    assert isinstance(expr5, LinearExpression)


def test_linear_expression_from_tuples_bad_calls(
    m: Model, x: Variable, y: Variable
) -> None:
    with pytest.raises(ValueError):
        LinearExpression.from_tuples((10, x), (1, y), x)

    with pytest.raises(ValueError):
        LinearExpression.from_tuples((10, x, 3), (1, y), 1)

    sv = ScalarVariable(label=0, model=m)
    with pytest.raises(TypeError):
        LinearExpression.from_tuples((np.array([1, 1]), sv))

    with pytest.raises(TypeError):
        LinearExpression.from_tuples((x, x))

    with pytest.raises(ValueError):
        LinearExpression.from_tuples(10)


def test_linear_expression_sanitize(x: Variable, y: Variable, z: Variable) -> None:
    expr = 10 * x + y + z
    assert isinstance(expr.sanitize(), LinearExpression)


def test_merge(x: Variable, y: Variable, z: Variable) -> None:
    expr1 = (10 * x + y).sum("dim_0")
    expr2 = z.sum("dim_0")

    res = merge([expr1, expr2], cls=LinearExpression)
    assert res.nterm == 6

    res: LinearExpression = merge([expr1, expr2])  # type: ignore
    assert isinstance(res, LinearExpression)

    # now concat with same length of terms
    expr1 = z.sel(dim_0=0).sum("dim_1")
    expr2 = z.sel(dim_0=1).sum("dim_1")

    res = merge([expr1, expr2], dim="dim_1", cls=LinearExpression)
    assert res.nterm == 3

    # now with different length of terms
    expr1 = z.sel(dim_0=0, dim_1=slice(0, 1)).sum("dim_1")
    expr2 = z.sel(dim_0=1).sum("dim_1")

    res = merge([expr1, expr2], dim="dim_1", cls=LinearExpression)
    assert res.nterm == 3
    assert res.sel(dim_1=0).vars[2].item() == -1

    with pytest.warns(DeprecationWarning):
        merge(expr1, expr2)


def test_linear_expression_outer_sum(x: Variable, y: Variable) -> None:
    expr = x + y
    expr2: LinearExpression = sum([x, y])  # type: ignore
    assert_linequal(expr, expr2)

    expr = 1 * x + 2 * y
    expr2: LinearExpression = sum([1 * x, 2 * y])  # type: ignore
    assert_linequal(expr, expr2)

    assert isinstance(expr.sum(), LinearExpression)


def test_rename(x: Variable, y: Variable, z: Variable) -> None:
    expr = 10 * x + y + z
    renamed = expr.rename({"dim_0": "dim_5"})
    assert set(renamed.dims) == {"dim_1", "dim_5", TERM_DIM}
    assert renamed.nterm == 3

    renamed = expr.rename({"dim_0": "dim_1", "dim_1": "dim_2"})
    assert set(renamed.dims) == {"dim_1", "dim_2", TERM_DIM}
    assert renamed.nterm == 3


@pytest.mark.parametrize("multiple", [1.0, 0.5, 2.0, 0.0])
def test_cumsum(m: Model, multiple: float) -> None:
    # Test cumsum on variable x
    var = m.variables["x"]
    cumsum = (multiple * var).cumsum()
    cumsum.nterm == 2

    # Test cumsum on sum of variables
    expr = m.variables["x"] + m.variables["y"]
    cumsum = (multiple * expr).cumsum()
    cumsum.nterm == 2
