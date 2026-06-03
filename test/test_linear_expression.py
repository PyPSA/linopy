#!/usr/bin/env python3
"""
Created on Wed Mar 17 17:06:36 2021.

@author: fabian
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
from xarray.core.types import JoinOptions
from xarray.testing import assert_equal

from linopy import (
    EvolvingAPIWarning,
    LinearExpression,
    Model,
    QuadraticExpression,
    Variable,
    merge,
)
from linopy.constants import HELPER_DIMS, TERM_DIM
from linopy.expressions import ScalarLinearExpression
from linopy.testing import assert_linequal, assert_quadequal
from linopy.variables import ScalarVariable


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
    assert not set(HELPER_DIMS).intersection(set(expr.coords))


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


def test_multiply_expression_by_multiindex_level_constant(u: Variable) -> None:
    """
    Expression over a MultiIndex dim times a single-level constant.

    Mirrors PyPSA's ``soc_delta * storage_weightings``: ``u`` is indexed by
    the (level1, level2) MultiIndex ``dim_3``; the weighting is indexed only
    by ``level1``. The product must not raise, and each ``dim_3`` entry must
    take the weight of its ``level1``.
    """
    by_level1 = xr.DataArray([10.0, 20.0], coords={"level1": [1, 2]}, dims=["level1"])

    with pytest.warns(EvolvingAPIWarning, match=r"broadcasting level subset"):
        expr = (1 * u) * by_level1

    coeffs = expr.coeffs.squeeze("_term")
    assert coeffs.sel(dim_3=(1, "a")).item() == 10.0
    assert coeffs.sel(dim_3=(1, "b")).item() == 10.0
    assert coeffs.sel(dim_3=(2, "a")).item() == 20.0
    assert coeffs.sel(dim_3=(2, "b")).item() == 20.0


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


def test_matmul_contracts_only_shared_dims(z: Variable) -> None:
    """
    A @ b contracts the genuinely shared dims and keeps the rest.

    ``z`` has dims (dim_0, dim_1); ``b`` has (dim_1, location). Only dim_1
    is shared, so the result must keep dim_0 and location. A conversion that
    broadcast ``b`` to ``z``'s coords would expand dim_0 into ``b`` and
    contract it away too — collapsing the result to (location,) only.
    """
    expr = 1 * z
    b = xr.DataArray(
        np.ones((3, 2)),
        coords={"dim_1": expr.indexes["dim_1"], "location": ["L1", "L2"]},
        dims=["dim_1", "location"],
    )

    res = expr @ b

    assert set(res.coord_dims) == {"dim_0", "location"}
    assert_linequal(res, (expr * b).sum("dim_1"))


def test_matmul_contracts_all_dims_when_const_covers_them(z: Variable) -> None:
    """B covering all of a's dims (and more) contracts a's dims, keeping b's extras."""
    expr = 1 * z  # dims (dim_0, dim_1)
    b = xr.DataArray(
        np.ones((2, 3, 2)),
        coords={
            "dim_0": expr.indexes["dim_0"],
            "dim_1": expr.indexes["dim_1"],
            "location": ["L1", "L2"],
        },
        dims=["dim_0", "dim_1", "location"],
    )

    res = expr @ b

    assert set(res.coord_dims) == {"location"}
    assert_linequal(res, (expr * b).sum(["dim_0", "dim_1"]))


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


class TestCoordinateAlignment:
    @pytest.fixture(params=["da", "series"])
    def subset(self, request: Any) -> xr.DataArray | pd.Series:
        if request.param == "da":
            return xr.DataArray([10.0, 30.0], dims=["dim_2"], coords={"dim_2": [1, 3]})
        return pd.Series([10.0, 30.0], index=pd.Index([1, 3], name="dim_2"))

    @pytest.fixture(params=["da", "series"])
    def superset(self, request: Any) -> xr.DataArray | pd.Series:
        if request.param == "da":
            return xr.DataArray(
                np.arange(25, dtype=float),
                dims=["dim_2"],
                coords={"dim_2": range(25)},
            )
        return pd.Series(
            np.arange(25, dtype=float), index=pd.Index(range(25), name="dim_2")
        )

    @pytest.fixture
    def expected_fill(self) -> np.ndarray:
        arr = np.zeros(20)
        arr[1] = 10.0
        arr[3] = 30.0
        return arr

    @pytest.fixture(params=["xarray", "pandas_series"], ids=["da", "series"])
    def nan_constant(self, request: Any) -> xr.DataArray | pd.Series:
        vals = np.arange(20, dtype=float)
        vals[0] = np.nan
        vals[5] = np.nan
        vals[19] = np.nan
        if request.param == "xarray":
            return xr.DataArray(vals, dims=["dim_2"], coords={"dim_2": range(20)})
        return pd.Series(vals, index=pd.Index(range(20), name="dim_2"))

    class TestSubset:
        @pytest.mark.parametrize("operand", ["var", "expr"])
        def test_mul_subset_fills_zeros(
            self,
            v: Variable,
            subset: xr.DataArray,
            expected_fill: np.ndarray,
            operand: str,
        ) -> None:
            target = v if operand == "var" else 1 * v
            result = target * subset
            assert result.sizes["dim_2"] == v.sizes["dim_2"]
            assert not np.isnan(result.coeffs.values).any()
            np.testing.assert_array_equal(result.coeffs.squeeze().values, expected_fill)

        @pytest.mark.parametrize("operand", ["var", "expr"])
        def test_add_subset_fills_zeros(
            self,
            v: Variable,
            subset: xr.DataArray,
            expected_fill: np.ndarray,
            operand: str,
        ) -> None:
            if operand == "var":
                result = v + subset
                expected = expected_fill
            else:
                result = (v + 5) + subset
                expected = expected_fill + 5
            assert result.sizes["dim_2"] == v.sizes["dim_2"]
            assert not np.isnan(result.const.values).any()
            np.testing.assert_array_equal(result.const.values, expected)

        @pytest.mark.parametrize("operand", ["var", "expr"])
        def test_sub_subset_fills_negated(
            self,
            v: Variable,
            subset: xr.DataArray,
            expected_fill: np.ndarray,
            operand: str,
        ) -> None:
            if operand == "var":
                result = v - subset
                expected = -expected_fill
            else:
                result = (v + 5) - subset
                expected = 5 - expected_fill
            assert result.sizes["dim_2"] == v.sizes["dim_2"]
            assert not np.isnan(result.const.values).any()
            np.testing.assert_array_equal(result.const.values, expected)

        @pytest.mark.parametrize("operand", ["var", "expr"])
        def test_div_subset_inverts_nonzero(
            self, v: Variable, subset: xr.DataArray, operand: str
        ) -> None:
            target = v if operand == "var" else 1 * v
            result = target / subset
            assert result.sizes["dim_2"] == v.sizes["dim_2"]
            assert not np.isnan(result.coeffs.values).any()
            assert result.coeffs.squeeze().sel(dim_2=1).item() == pytest.approx(0.1)
            assert result.coeffs.squeeze().sel(dim_2=0).item() == pytest.approx(1.0)

        def test_subset_add_var_coefficients(
            self, v: Variable, subset: xr.DataArray
        ) -> None:
            result = subset + v
            np.testing.assert_array_equal(result.coeffs.squeeze().values, np.ones(20))

        def test_subset_sub_var_coefficients(
            self, v: Variable, subset: xr.DataArray
        ) -> None:
            result = subset - v
            np.testing.assert_array_equal(result.coeffs.squeeze().values, -np.ones(20))

    class TestSuperset:
        def test_add_superset_pins_to_lhs_coords(
            self, v: Variable, superset: xr.DataArray
        ) -> None:
            result = v + superset
            assert result.sizes["dim_2"] == v.sizes["dim_2"]
            assert not np.isnan(result.const.values).any()

        def test_add_var_commutative(self, v: Variable, superset: xr.DataArray) -> None:
            assert_linequal(superset + v, v + superset)

        def test_sub_var_commutative(self, v: Variable, superset: xr.DataArray) -> None:
            assert_linequal(superset - v, -v + superset)

        def test_mul_var_commutative(self, v: Variable, superset: xr.DataArray) -> None:
            assert_linequal(superset * v, v * superset)

        def test_mul_superset_pins_to_lhs_coords(
            self, v: Variable, superset: xr.DataArray
        ) -> None:
            result = v * superset
            assert result.sizes["dim_2"] == v.sizes["dim_2"]
            assert not np.isnan(result.coeffs.values).any()

        def test_div_superset_pins_to_lhs_coords(self, v: Variable) -> None:
            superset_nonzero = xr.DataArray(
                np.arange(1, 26, dtype=float),
                dims=["dim_2"],
                coords={"dim_2": range(25)},
            )
            result = v / superset_nonzero
            assert result.sizes["dim_2"] == v.sizes["dim_2"]
            assert not np.isnan(result.coeffs.values).any()

    class TestDisjoint:
        def test_add_disjoint_fills_zeros(self, v: Variable) -> None:
            disjoint = xr.DataArray(
                [100.0, 200.0], dims=["dim_2"], coords={"dim_2": [50, 60]}
            )
            result = v + disjoint
            assert result.sizes["dim_2"] == v.sizes["dim_2"]
            assert not np.isnan(result.const.values).any()
            np.testing.assert_array_equal(result.const.values, np.zeros(20))

        def test_mul_disjoint_fills_zeros(self, v: Variable) -> None:
            disjoint = xr.DataArray(
                [10.0, 20.0], dims=["dim_2"], coords={"dim_2": [50, 60]}
            )
            result = v * disjoint
            assert result.sizes["dim_2"] == v.sizes["dim_2"]
            assert not np.isnan(result.coeffs.values).any()
            np.testing.assert_array_equal(result.coeffs.squeeze().values, np.zeros(20))

        def test_div_disjoint_preserves_coeffs(self, v: Variable) -> None:
            disjoint = xr.DataArray(
                [10.0, 20.0], dims=["dim_2"], coords={"dim_2": [50, 60]}
            )
            result = v / disjoint
            assert result.sizes["dim_2"] == v.sizes["dim_2"]
            assert not np.isnan(result.coeffs.values).any()
            np.testing.assert_array_equal(result.coeffs.squeeze().values, np.ones(20))

    class TestCommutativity:
        @pytest.mark.parametrize(
            "make_lhs,make_rhs",
            [
                (lambda v, s: s * v, lambda v, s: v * s),
                (lambda v, s: s * (1 * v), lambda v, s: (1 * v) * s),
                (lambda v, s: s + v, lambda v, s: v + s),
                (lambda v, s: s + (v + 5), lambda v, s: (v + 5) + s),
            ],
            ids=["subset*var", "subset*expr", "subset+var", "subset+expr"],
        )
        def test_commutativity(
            self,
            v: Variable,
            subset: xr.DataArray,
            make_lhs: Any,
            make_rhs: Any,
        ) -> None:
            assert_linequal(make_lhs(v, subset), make_rhs(v, subset))

        def test_sub_var_anticommutative(
            self, v: Variable, subset: xr.DataArray
        ) -> None:
            assert_linequal(subset - v, -v + subset)

        def test_sub_expr_anticommutative(
            self, v: Variable, subset: xr.DataArray
        ) -> None:
            expr = v + 5
            assert_linequal(subset - expr, -(expr - subset))

        def test_add_commutativity_full_coords(self, v: Variable) -> None:
            full = xr.DataArray(
                np.arange(20, dtype=float),
                dims=["dim_2"],
                coords={"dim_2": range(20)},
            )
            assert_linequal(v + full, full + v)

    class TestQuadratic:
        def test_quadexpr_add_subset(
            self,
            v: Variable,
            subset: xr.DataArray,
            expected_fill: np.ndarray,
        ) -> None:
            qexpr = v * v
            result = qexpr + subset
            assert isinstance(result, QuadraticExpression)
            assert result.sizes["dim_2"] == v.sizes["dim_2"]
            assert not np.isnan(result.const.values).any()
            np.testing.assert_array_equal(result.const.values, expected_fill)

        def test_quadexpr_sub_subset(
            self,
            v: Variable,
            subset: xr.DataArray,
            expected_fill: np.ndarray,
        ) -> None:
            qexpr = v * v
            result = qexpr - subset
            assert isinstance(result, QuadraticExpression)
            assert result.sizes["dim_2"] == v.sizes["dim_2"]
            assert not np.isnan(result.const.values).any()
            np.testing.assert_array_equal(result.const.values, -expected_fill)

        def test_quadexpr_mul_subset(
            self,
            v: Variable,
            subset: xr.DataArray,
            expected_fill: np.ndarray,
        ) -> None:
            qexpr = v * v
            result = qexpr * subset
            assert isinstance(result, QuadraticExpression)
            assert result.sizes["dim_2"] == v.sizes["dim_2"]
            assert not np.isnan(result.coeffs.values).any()
            np.testing.assert_array_equal(result.coeffs.squeeze().values, expected_fill)

        def test_subset_mul_quadexpr(
            self,
            v: Variable,
            subset: xr.DataArray,
            expected_fill: np.ndarray,
        ) -> None:
            qexpr = v * v
            result = subset * qexpr
            assert isinstance(result, QuadraticExpression)
            assert result.sizes["dim_2"] == v.sizes["dim_2"]
            assert not np.isnan(result.coeffs.values).any()
            np.testing.assert_array_equal(result.coeffs.squeeze().values, expected_fill)

        def test_subset_add_quadexpr(self, v: Variable, subset: xr.DataArray) -> None:
            qexpr = v * v
            assert_quadequal(subset + qexpr, qexpr + subset)

    class TestMissingValues:
        """
        Same shape as variable but with NaN entries in the constant.

        NaN values are filled with operation-specific neutral elements:
        - Addition/subtraction: NaN -> 0 (additive identity)
        - Multiplication: NaN -> 0 (zeroes out the variable)
        - Division: NaN -> 1 (multiplicative identity, no scaling)
        """

        NAN_POSITIONS = [0, 5, 19]

        @pytest.mark.parametrize("operand", ["var", "expr"])
        def test_add_nan_filled(
            self,
            v: Variable,
            nan_constant: xr.DataArray | pd.Series,
            operand: str,
        ) -> None:
            base_const = 0.0 if operand == "var" else 5.0
            target = v if operand == "var" else v + 5
            result = target + nan_constant
            assert result.sizes["dim_2"] == 20
            assert not np.isnan(result.const.values).any()
            # At NaN positions, const should be unchanged (added 0)
            for i in self.NAN_POSITIONS:
                assert result.const.values[i] == base_const

        @pytest.mark.parametrize("operand", ["var", "expr"])
        def test_sub_nan_filled(
            self,
            v: Variable,
            nan_constant: xr.DataArray | pd.Series,
            operand: str,
        ) -> None:
            base_const = 0.0 if operand == "var" else 5.0
            target = v if operand == "var" else v + 5
            result = target - nan_constant
            assert result.sizes["dim_2"] == 20
            assert not np.isnan(result.const.values).any()
            # At NaN positions, const should be unchanged (subtracted 0)
            for i in self.NAN_POSITIONS:
                assert result.const.values[i] == base_const

        @pytest.mark.parametrize("operand", ["var", "expr"])
        def test_mul_nan_filled(
            self,
            v: Variable,
            nan_constant: xr.DataArray | pd.Series,
            operand: str,
        ) -> None:
            target = v if operand == "var" else 1 * v
            result = target * nan_constant
            assert result.sizes["dim_2"] == 20
            assert not np.isnan(result.coeffs.squeeze().values).any()
            # At NaN positions, coeffs should be 0 (variable zeroed out)
            for i in self.NAN_POSITIONS:
                assert result.coeffs.squeeze().values[i] == 0.0

        @pytest.mark.parametrize("operand", ["var", "expr"])
        def test_div_nan_filled(
            self,
            v: Variable,
            nan_constant: xr.DataArray | pd.Series,
            operand: str,
        ) -> None:
            target = v if operand == "var" else 1 * v
            result = target / nan_constant
            assert result.sizes["dim_2"] == 20
            assert not np.isnan(result.coeffs.squeeze().values).any()
            # At NaN positions, coeffs should be unchanged (divided by 1)
            original_coeffs = (1 * v).coeffs.squeeze().values
            for i in self.NAN_POSITIONS:
                assert result.coeffs.squeeze().values[i] == original_coeffs[i]

        def test_add_commutativity(
            self,
            v: Variable,
            nan_constant: xr.DataArray | pd.Series,
        ) -> None:
            result_a = v + nan_constant
            result_b = nan_constant + v
            assert not np.isnan(result_a.const.values).any()
            assert not np.isnan(result_b.const.values).any()
            np.testing.assert_array_equal(result_a.const.values, result_b.const.values)
            np.testing.assert_array_equal(
                result_a.coeffs.values, result_b.coeffs.values
            )

        def test_mul_commutativity(
            self,
            v: Variable,
            nan_constant: xr.DataArray | pd.Series,
        ) -> None:
            result_a = v * nan_constant
            result_b = nan_constant * v
            assert not np.isnan(result_a.coeffs.values).any()
            assert not np.isnan(result_b.coeffs.values).any()
            np.testing.assert_array_equal(
                result_a.coeffs.values, result_b.coeffs.values
            )

        def test_quadexpr_add_nan(
            self,
            v: Variable,
            nan_constant: xr.DataArray | pd.Series,
        ) -> None:
            qexpr = v * v
            result = qexpr + nan_constant
            assert isinstance(result, QuadraticExpression)
            assert result.sizes["dim_2"] == 20
            assert not np.isnan(result.const.values).any()

    class TestExpressionWithNaN:
        """Test that NaN in expression's own const/coeffs doesn't propagate."""

        def test_shifted_expr_add_scalar(self, v: Variable) -> None:
            expr = (1 * v).shift(dim_2=1)
            result = expr + 5
            assert not np.isnan(result.const.values).any()
            assert result.const.values[0] == 5.0

        def test_shifted_expr_mul_scalar(self, v: Variable) -> None:
            expr = (1 * v).shift(dim_2=1)
            result = expr * 2
            assert not np.isnan(result.coeffs.squeeze().values).any()
            assert result.coeffs.squeeze().values[0] == 0.0

        def test_shifted_expr_add_array(self, v: Variable) -> None:
            arr = np.arange(v.sizes["dim_2"], dtype=float)
            expr = (1 * v).shift(dim_2=1)
            result = expr + arr
            assert not np.isnan(result.const.values).any()
            assert result.const.values[0] == 0.0

        def test_shifted_expr_mul_array(self, v: Variable) -> None:
            arr = np.arange(v.sizes["dim_2"], dtype=float) + 1
            expr = (1 * v).shift(dim_2=1)
            result = expr * arr
            assert not np.isnan(result.coeffs.squeeze().values).any()
            assert result.coeffs.squeeze().values[0] == 0.0

        def test_shifted_expr_div_scalar(self, v: Variable) -> None:
            expr = (1 * v).shift(dim_2=1)
            result = expr / 2
            assert not np.isnan(result.coeffs.squeeze().values).any()
            assert result.coeffs.squeeze().values[0] == 0.0

        def test_shifted_expr_sub_scalar(self, v: Variable) -> None:
            expr = (1 * v).shift(dim_2=1)
            result = expr - 3
            assert not np.isnan(result.const.values).any()
            assert result.const.values[0] == -3.0

        def test_shifted_expr_div_array(self, v: Variable) -> None:
            arr = np.arange(v.sizes["dim_2"], dtype=float) + 1
            expr = (1 * v).shift(dim_2=1)
            result = expr / arr
            assert not np.isnan(result.coeffs.squeeze().values).any()
            assert result.coeffs.squeeze().values[0] == 0.0

        def test_variable_to_linexpr_nan_coefficient(self, v: Variable) -> None:
            nan_coeff = np.ones(v.sizes["dim_2"])
            nan_coeff[0] = np.nan
            result = v.to_linexpr(nan_coeff)
            assert not np.isnan(result.coeffs.squeeze().values).any()
            assert result.coeffs.squeeze().values[0] == 0.0

    class TestMultiDim:
        def test_multidim_subset_mul(self, m: Model) -> None:
            coords_a = pd.RangeIndex(4, name="a")
            coords_b = pd.RangeIndex(5, name="b")
            w = m.add_variables(coords=[coords_a, coords_b], name="w")

            subset_2d = xr.DataArray(
                [[2.0, 3.0], [4.0, 5.0]],
                dims=["a", "b"],
                coords={"a": [1, 3], "b": [0, 4]},
            )
            result = w * subset_2d
            assert result.sizes["a"] == 4
            assert result.sizes["b"] == 5
            assert not np.isnan(result.coeffs.values).any()
            assert result.coeffs.squeeze().sel(a=1, b=0).item() == pytest.approx(2.0)
            assert result.coeffs.squeeze().sel(a=3, b=4).item() == pytest.approx(5.0)
            assert result.coeffs.squeeze().sel(a=0, b=0).item() == pytest.approx(0.0)
            assert result.coeffs.squeeze().sel(a=1, b=2).item() == pytest.approx(0.0)

        def test_multidim_subset_add(self, m: Model) -> None:
            coords_a = pd.RangeIndex(4, name="a")
            coords_b = pd.RangeIndex(5, name="b")
            w = m.add_variables(coords=[coords_a, coords_b], name="w")

            subset_2d = xr.DataArray(
                [[2.0, 3.0], [4.0, 5.0]],
                dims=["a", "b"],
                coords={"a": [1, 3], "b": [0, 4]},
            )
            result = w + subset_2d
            assert result.sizes["a"] == 4
            assert result.sizes["b"] == 5
            assert not np.isnan(result.const.values).any()
            assert result.const.sel(a=1, b=0).item() == pytest.approx(2.0)
            assert result.const.sel(a=3, b=4).item() == pytest.approx(5.0)
            assert result.const.sel(a=0, b=0).item() == pytest.approx(0.0)

    class TestXarrayCompat:
        def test_da_eq_da_still_works(self) -> None:
            da1 = xr.DataArray([1, 2, 3])
            da2 = xr.DataArray([1, 2, 3])
            result = da1 == da2
            assert result.values.all()

        def test_da_eq_scalar_still_works(self) -> None:
            da = xr.DataArray([1, 2, 3])
            result = da == 2
            np.testing.assert_array_equal(result.values, [False, True, False])

        def test_da_truediv_var_raises(self, v: Variable) -> None:
            da = xr.DataArray(np.ones(20), dims=["dim_2"], coords={"dim_2": range(20)})
            with pytest.raises(TypeError):
                da / v  # type: ignore[operator]


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


class TestHasTerms:
    """has_terms: true at slots with at least one live term, regardless of the constant."""

    def test_basic_and_masking(self, v: Variable) -> None:
        expr = np.arange(20) * v
        assert expr.has_terms.all()

        filter = (expr.coeffs >= 10).any(TERM_DIM)
        masked = expr.where(filter)
        assert_equal(masked.has_terms, filter.rename("has_terms"))

    def test_ignores_const(self, v: Variable) -> None:
        # has_terms differs from isnull() at slots whose constant was revived by
        # fillna: no longer null, but still without terms
        expr = np.arange(20) * v
        filter = (expr.coeffs >= 10).any(TERM_DIM)
        masked = expr.where(filter)
        assert_equal(masked.isnull(), ~masked.has_terms)

        filled = masked.fillna(0)
        assert not filled.isnull().any()
        assert_equal(filled.has_terms, filter.rename("has_terms"))

    def test_merge_reindex(self, x: Variable, y: Variable) -> None:
        # the nodal-balance pattern: outer merge, then reindex to a superset of
        # coordinates; slots beyond the original coordinates carry no terms
        lhs = merge([1 * x, 1 * y], join="outer").reindex(
            dim_0=pd.RangeIndex(4, name="dim_0")
        )
        assert lhs.has_terms.values.tolist() == [True, True, False, False]

    def test_constant_only(self, m: Model) -> None:
        expr = LinearExpression(xr.DataArray([1, 2], dims=["dim_0"]), m)
        assert expr.nterm == 0
        assert not expr.has_terms.any()

    def test_quadratic(self, v: Variable) -> None:
        # linear terms inside a quadratic expression carry one factor == -1;
        # they must still count as live terms
        quad = v * v + 2 * v
        assert quad.has_terms.all()
        assert TERM_DIM not in quad.has_terms.dims

        filter = xr.DataArray(
            np.arange(20) >= 10, dims="dim_2", coords={"dim_2": range(20)}
        )
        masked = quad.where(filter)
        assert_equal(masked.has_terms, filter.rename("has_terms"))


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


class TestMultiKeyFastPath:
    """
    Group a LinearExpression by a list of coordinate names: takes the fast
    reindex path and returns one dimension per key, like the xarray fallback.
    """

    @staticmethod
    def _expr(period_vals: list, season_vals: list) -> LinearExpression:
        n = len(period_vals)
        s = pd.RangeIndex(n, name="s")
        m = Model()
        x = m.add_variables(coords=[s], name="x")
        return (1.0 * x).assign_coords(
            period=xr.DataArray(period_vals, dims="s", coords={"s": s}, name="period"),
            season=xr.DataArray(season_vals, dims="s", coords={"s": s}, name="season"),
        )

    @pytest.mark.parametrize("spelling", [list, tuple], ids=["list", "tuple"])
    def test_matches_fallback(self, spelling: type) -> None:
        # the fast path must equal the slow fallback, sparse cells included
        expr = self._expr([2020, 2020, 2030, 2030, 2030], list("wswws"))
        group = spelling(["period", "season"])

        fast = expr.groupby(group).sum()
        slow = expr.groupby(group).sum(use_fallback=True)

        assert_linequal(fast, slow)

    def test_separate_dims_not_stacked(self) -> None:
        # built via a stacked index internally, but returns one dim per key
        expr = self._expr([2020, 2020, 2030, 2030], list("wsws"))

        grouped = expr.groupby(["period", "season"]).sum()

        assert {"period", "season"} <= set(grouped.dims)
        assert "group" not in grouped.dims
        assert not isinstance(grouped.data.indexes.get("period"), pd.MultiIndex)

    def test_sparse_combination_filled(self) -> None:
        # (2020, "s") never occurs -> empty term in the grid
        expr = self._expr([2020, 2020, 2030, 2030], list("wwws"))

        grouped = expr.groupby(["period", "season"]).sum()

        cell = grouped.sel(period=2020, season="s")
        assert (cell.vars == -1).all()
        assert cell.coeffs.isnull().all()

    def test_dataframe_grouper_stays_compact(self) -> None:
        # the DataFrame grouper keeps the stacked observed-only group dim
        expr = self._expr([2020, 2020, 2030, 2030], list("wwws"))
        df = expr.data[["period", "season"]].to_dataframe()[["period", "season"]]

        grouped = expr.groupby(df).sum()

        assert "group" in grouped.dims
        assert isinstance(grouped.data.indexes["group"], pd.MultiIndex)
        assert grouped.sizes["group"] == 3  # observed, not the 2x2=4 grid

    def test_blowup_warns_when_sparse(self) -> None:
        # 200 observed combos, 200x200 grid -> nudge toward observed=True
        expr = self._expr(list(range(200)), list(range(200)))

        with pytest.warns(UserWarning, match="dense .* grid"):
            expr.groupby(["period", "season"]).sum()

    def test_no_warning_when_dense(self) -> None:
        expr = self._expr([2020, 2020, 2030, 2030], list("wsws"))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            expr.groupby(["period", "season"]).sum()

    def test_observed_keeps_stacked(self) -> None:
        # observed=True skips the unstack: compact stacked MultiIndex,
        # identical to the DataFrame grouper output
        expr = self._expr([2020, 2020, 2030, 2030], list("wwws"))
        df = expr.data[["period", "season"]].to_dataframe()[["period", "season"]]

        grouped = expr.groupby(["period", "season"]).sum(observed=True)

        assert_linequal(grouped, expr.groupby(df).sum())
        assert grouped.sizes["group"] == 3  # observed, not the 2x2=4 grid

    def test_observed_silences_blowup_warning(self) -> None:
        expr = self._expr(list(range(200)), list(range(200)))

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            grouped = expr.groupby(["period", "season"]).sum(observed=True)

        assert grouped.sizes["group"] == 200

    def test_observed_with_fallback_raises(self) -> None:
        expr = self._expr([2020, 2020], list("ws"))

        with pytest.raises(ValueError, match="observed"):
            expr.groupby(["period", "season"]).sum(use_fallback=True, observed=True)


class TestGroupbyByAttachedCoordinate:
    """
    Group by an attached non-dimension coordinate.

    Asserts grouping against hard-coded ``vars``/``coeffs`` to catch regressions.
    """

    @pytest.fixture
    def t(self) -> pd.RangeIndex:
        return pd.RangeIndex(4, name="t")

    @pytest.fixture
    def period(self, t: pd.RangeIndex) -> xr.DataArray:
        return xr.DataArray(
            [2020, 2020, 2030, 2030], dims="t", coords={"t": t}, name="period"
        )

    @pytest.fixture
    def season(self, t: pd.RangeIndex) -> xr.DataArray:
        return xr.DataArray(list("wsws"), dims="t", coords={"t": t}, name="season")

    @pytest.fixture
    def expr(
        self, t: pd.RangeIndex, period: xr.DataArray, season: xr.DataArray
    ) -> LinearExpression:
        m = Model()
        x = m.add_variables(coords=[t], name="x")
        return (2.0 * x).assign_coords(period=period, season=season)

    @pytest.mark.parametrize("use_fallback", [True, False])
    @pytest.mark.parametrize("by", ["name", "dataarray"])
    def test_single_key(
        self,
        expr: LinearExpression,
        period: xr.DataArray,
        by: str,
        use_fallback: bool,
    ) -> None:
        group = "period" if by == "name" else period

        grouped = expr.groupby(group).sum(use_fallback=use_fallback)

        assert grouped.data.period.values.tolist() == [2020, 2030]
        assert grouped.vars.transpose("period", TERM_DIM).values.tolist() == [
            [0, 1],
            [2, 3],
        ]
        assert grouped.coeffs.transpose("period", TERM_DIM).values.tolist() == [
            [2.0, 2.0],
            [2.0, 2.0],
        ]

    @pytest.mark.parametrize("spelling", [list, tuple], ids=["list", "tuple"])
    def test_multi_key(self, expr: LinearExpression, spelling: type) -> None:
        # A multi-key group always goes through the xarray fallback (a list is
        # not a fast-path type), so there is no separate use_fallback case.
        group = spelling(["period", "season"])

        grouped = expr.groupby(group).sum()

        assert dict(grouped.sizes) == {"period": 2, "season": 2, TERM_DIM: 1}
        assert grouped.data.period.values.tolist() == [2020, 2030]
        assert grouped.data.season.values.tolist() == ["s", "w"]
        assert grouped.vars.transpose("period", "season", TERM_DIM).values.tolist() == [
            [[1], [0]],
            [[3], [2]],
        ]
        assert (grouped.coeffs == 2.0).all()

    def test_extra_aux_coord_does_not_change_result(
        self, t: pd.RangeIndex, period: xr.DataArray
    ) -> None:
        # A second auxiliary coord on the grouped dimension must neither break
        # the reshape (it raised ``KeyError`` before the fix) nor change the sum.
        m = Model()
        x = m.add_variables(coords=[t], name="x")
        timestep = xr.DataArray(
            list("abab"), dims="t", coords={"t": t}, name="timestep"
        )
        expr = (2.0 * x).assign_coords(period=period, timestep=timestep)

        grouped = expr.groupby("period").sum()

        assert "timestep" not in grouped.coords
        assert grouped.vars.transpose("period", TERM_DIM).values.tolist() == [
            [0, 1],
            [2, 3],
        ]
        assert (grouped.coeffs == 2.0).all()

    @pytest.mark.parametrize("by", ["name", "dataarray"])
    def test_two_dimensional(self, by: str) -> None:
        # Grouping one dimension of a 2-D variable by an aux coord must keep the
        # other dimension intact and pair up the right variable labels.
        m = Model()
        snapshot = pd.RangeIndex(4, name="snapshot")
        gen = pd.Index(["g1", "g2"], name="gen")
        y = m.add_variables(coords=[snapshot, gen], name="y")  # labels 0..7
        period = xr.DataArray(
            [2020, 2020, 2030, 2030],
            dims="snapshot",
            coords={"snapshot": snapshot},
            name="period",
        )
        expr = (1.0 * y).assign_coords(period=period)
        group = "period" if by == "name" else period

        grouped = expr.groupby(group).sum()

        assert grouped.data.period.values.tolist() == [2020, 2030]
        assert grouped.data.gen.values.tolist() == ["g1", "g2"]
        assert grouped.vars.transpose("period", "gen", TERM_DIM).values.tolist() == [
            [[0, 2], [1, 3]],
            [[4, 6], [5, 7]],
        ]
        assert (grouped.coeffs == 1.0).all()

    @pytest.mark.parametrize("use_fallback", [True, False])
    def test_dimension_coordinate_by_name(self, use_fallback: bool) -> None:
        # A dimension coordinate may also be grouped by name; it collapses that
        # dimension and keeps the other one.
        m = Model()
        snapshot = pd.RangeIndex(4, name="snapshot")
        gen = pd.Index(["g1", "g2"], name="gen")
        y = m.add_variables(coords=[snapshot, gen], name="y")  # labels 0..7

        grouped = (1 * y).groupby("gen").sum(use_fallback=use_fallback)

        assert grouped.data.gen.values.tolist() == ["g1", "g2"]
        assert grouped.sizes["snapshot"] == 4
        assert grouped.vars.transpose("gen", "snapshot", TERM_DIM).values.tolist() == [
            [[0], [2], [4], [6]],
            [[1], [3], [5], [7]],
        ]

    @pytest.mark.parametrize("use_fallback", [True, False])
    def test_single_element_list_groups_like_scalar(
        self, expr: LinearExpression, use_fallback: bool
    ) -> None:
        # ``groupby(["period"])`` groups like the scalar key, mirroring xarray.
        grouped = expr.groupby(["period"]).sum(use_fallback=use_fallback)

        assert grouped.data.period.values.tolist() == [2020, 2030]
        assert grouped.vars.transpose("period", TERM_DIM).values.tolist() == [
            [0, 1],
            [2, 3],
        ]
        assert (grouped.coeffs == 2.0).all()

    def test_multi_key_dataarrays_unsupported(
        self, expr: LinearExpression, period: xr.DataArray, season: xr.DataArray
    ) -> None:
        # Multi-key grouping must be spelled with names; a list of DataArrays
        # is unhashable and raises in xarray itself, so linopy mirrors that.
        with pytest.raises(TypeError, match="unhashable"):
            expr.groupby([period, season]).sum()

    @pytest.mark.parametrize("use_fallback", [True, False])
    @pytest.mark.parametrize(
        "level, values, vars_",
        [
            ("period", [2020, 2030], [[0, 1, 2], [3, 4, 5]]),
            ("timestep", ["t1", "t2", "t3"], [[0, 3], [1, 4], [2, 5]]),
        ],
    )
    def test_multiindex_level(
        self, level: str, values: list, vars_: list, use_fallback: bool
    ) -> None:
        # Grouping by a level of a real ``MultiIndex`` dimension (the
        # pydata/xarray#6836 case, fixed upstream) works through linopy.
        m = Model()
        mi = pd.MultiIndex.from_product(
            [[2020, 2030], ["t1", "t2", "t3"]], names=["period", "timestep"]
        )
        x = m.add_variables(coords={"snapshot": mi}, name="x")  # labels 0..5

        grouped = (1 * x).groupby(level).sum(use_fallback=use_fallback)

        assert grouped.data[level].values.tolist() == values
        assert grouped.vars.transpose(level, TERM_DIM).values.tolist() == vars_


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


def test_linear_expression_groupby_skewed_unsorted_groups(v: Variable) -> None:
    """
    The scatter-based fast path must match the xarray fallback for groups that
    are unsorted, non-contiguous and of very different sizes.
    """
    expr = 2 * v + 5
    # 'b' appears 14 times, 'c' 5 times, 'a' once, scattered over the dimension
    labels = ["b"] * 4 + ["c", "a"] + ["b"] * 5 + ["c"] * 4 + ["b"] * 5
    groups = pd.Series(labels, index=v.indexes["dim_2"], name="letter")

    grouped = expr.groupby(groups).sum()
    fallback = expr.groupby(groups.to_xarray()).sum(use_fallback=True)

    assert list(grouped.data.letter) == ["a", "b", "c"]
    # padded to the largest group times the number of terms of the input
    assert grouped.nterm == 14 * expr.nterm
    assert_linequal(grouped, fallback)

    # every group must carry exactly the variables of its members, the rest is fill
    for letter in ["a", "b", "c"]:
        members = np.where(np.array(labels) == letter)[0]
        vars_of_group = grouped.data.vars.sel(letter=letter).values
        assert set(vars_of_group[vars_of_group >= 0]) == set(v.labels.values[members])
        assert (vars_of_group >= 0).sum() == len(members) * expr.nterm
        assert grouped.const.sel(letter=letter).item() == 5 * len(members)


def test_linear_expression_groupby_chunked(v: Variable) -> None:
    """Chunked (dask-backed) expressions group via xarray's unstack machinery."""
    pytest.importorskip("dask")
    expr = 2 * v + 5
    groups = pd.Series([1] * 12 + [2] * 8, index=v.indexes["dim_2"], name="group")

    chunked = LinearExpression(expr.data.chunk({"dim_2": 5}), expr.model)
    grouped_chunked = chunked.groupby(groups).sum()
    grouped = expr.groupby(groups).sum()

    assert grouped_chunked.nterm == grouped.nterm
    assert_linequal(
        LinearExpression(grouped_chunked.data.compute(), expr.model), grouped
    )


def test_linear_expression_groupby_with_nan_groups(v: Variable) -> None:
    expr = 1 * v
    groups = pd.Series([1.0, np.nan] * 10, index=v.indexes["dim_2"], name="with_nans")
    with pytest.raises(ValueError, match="NaN"):
        expr.groupby(groups).sum()


@pytest.mark.parametrize(
    "case",
    [
        "skewed_int_groups",
        "multidim_with_const",
        "nan_const",
        "masked_vars",
        "quadratic",
        "single_group",
        "identity_groups",
    ],
)
def test_linear_expression_groupby_scatter_equals_unstack(case: str) -> None:
    """
    Lock the two groupby-sum kernels together.

    The fast path of groupby(...).sum() scatters terms into numpy arrays
    (_sum_by_scatter); the xarray unstack implementation (_sum_by_unstack) is
    kept for chunked data and exotic coordinates. Both must stay
    interchangeable — if an xarray/pandas update changes the unstack output or
    an edge case diverges, this fails.
    """
    m = Model()
    rng = np.random.default_rng(0)
    idx = pd.RangeIndex(60, name="elem")
    skewed = pd.Series(rng.choice(8, 60, p=[0.5] + [0.5 / 7] * 7), index=idx, name="g")
    groups = skewed

    if case == "skewed_int_groups":
        x = m.add_variables(coords=[idx], name="x")
        expr: LinearExpression | QuadraticExpression = 3 * x - 2 * x + 7
    elif case == "multidim_with_const":
        other = pd.Index(list("abc"), name="other")
        y = m.add_variables(coords=[other, idx], name="y")
        const = xr.DataArray(rng.normal(size=(3, 60)), coords=[other, idx])
        expr = 2 * y + 1 * y + const
    elif case == "nan_const":
        x = m.add_variables(coords=[idx], name="x")
        expr = 1 * x + np.where(np.arange(60) % 3, np.nan, 5.0)
    elif case == "masked_vars":
        mask = xr.DataArray(np.arange(60) % 4 != 0, coords=[idx])
        x = m.add_variables(coords=[idx], name="x", mask=mask)
        expr = 1 * x
    elif case == "quadratic":
        x = m.add_variables(coords=[idx], name="x")
        expr = x * x + 2 * x
    elif case == "single_group":
        x = m.add_variables(coords=[idx], name="x")
        expr = 1 * x
        groups = pd.Series(1, index=idx, name="g")
    else:  # identity_groups
        x = m.add_variables(coords=[idx], name="x")
        expr = 1 * x
        groups = pd.Series(np.arange(60), index=idx, name="g")

    gb = expr.groupby(groups)
    assert gb._can_sum_by_scatter(groups)
    scatter = LinearExpression(gb._sum_by_scatter(groups).rename(_group="g"), m)
    unstack = LinearExpression(gb._sum_by_unstack(groups).rename(_group="g"), m)

    # identical structure: dims, dim order, coordinates
    assert scatter.data.coeffs.dims == unstack.data.coeffs.dims
    assert scatter.data.const.dims == unstack.data.const.dims
    assert list(scatter.data.coords) == list(unstack.data.coords)
    for name in scatter.data.coords:
        assert_equal(scatter.data[name], unstack.data[name])

    # identical values: vars and coeffs bit-exact, including padding positions
    np.testing.assert_array_equal(scatter.vars.values, unstack.vars.values)
    np.testing.assert_array_equal(scatter.coeffs.values, unstack.coeffs.values)
    # constants may differ by floating-point summation order
    np.testing.assert_allclose(scatter.const.values, unstack.const.values, rtol=1e-12)


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


def test_linear_expression_from_constant_scalar(m: Model) -> None:
    expr = LinearExpression.from_constant(model=m, constant=10)
    assert expr.is_constant
    assert isinstance(expr, LinearExpression)
    assert (expr.const == 10).all()


def test_linear_expression_from_constant_1D(m: Model) -> None:
    arr = pd.Series(index=pd.Index([0, 1], name="t"), data=[10, 20])
    expr = LinearExpression.from_constant(model=m, constant=arr)
    assert isinstance(expr, LinearExpression)
    assert list(expr.coords.keys())[0] == "t"
    assert expr.nterm == 0
    assert (expr.const.values == [10, 20]).all()
    assert expr.is_constant


def test_constant_linear_expression_to_polars_2D(m: Model) -> None:
    index_a = pd.Index([0, 1], name="a")
    index_b = pd.Index([0, 1, 2], name="b")
    arr = np.array([[10, 20, 30], [40, 50, 60]])
    const = xr.DataArray(data=arr, coords=[index_a, index_b])

    le_variable = m.add_variables(name="var", coords=[index_a, index_b]) * 1 + const
    assert not le_variable.is_constant
    le_const = LinearExpression.from_constant(model=m, constant=const)
    assert le_const.is_constant

    var_pol = le_variable.to_polars()
    const_pol = le_const.to_polars()
    assert var_pol.shape == const_pol.shape
    assert var_pol.columns == const_pol.columns
    assert all(const_pol["const"] == var_pol["const"])
    assert all(const_pol["coeffs"].is_null())
    assert all(const_pol["vars"].is_null())


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


def test_simplify_basic(x: Variable) -> None:
    """Test basic simplification with duplicate terms."""
    expr = 2 * x + 3 * x + 1 * x
    simplified = expr.simplify()
    assert simplified.nterm == 1, f"Expected 1 term, got {simplified.nterm}"

    x_len = len(x.coords["dim_0"])
    # Check that the coefficient is 6 (2 + 3 + 1)
    coeffs: np.ndarray = simplified.coeffs.values
    assert len(coeffs) == x_len, f"Expected {x_len} coefficients, got {len(coeffs)}"
    assert all(coeffs == 6.0), f"Expected coefficient 6.0, got {coeffs[0]}"


def test_simplify_multiple_dimensions() -> None:
    model = Model()
    a_index = pd.Index([0, 1, 2, 3], name="a")
    b_index = pd.Index([0, 1, 2], name="b")
    coords = [a_index, b_index]
    x = model.add_variables(name="x", coords=coords)

    expr = 2 * x + 3 * x + x
    # Simplify
    simplified = expr.simplify()
    assert simplified.nterm == 1, f"Expected 1 term, got {simplified.nterm}"
    assert simplified.ndim == 2, f"Expected 2 dimensions, got {simplified.ndim}"
    assert all(simplified.coeffs.values.reshape(-1) == 6), (
        f"Expected coefficients of 6, got {simplified.coeffs.values}"
    )


def test_simplify_with_different_variables(x: Variable, y: Variable) -> None:
    """Test that different variables are kept separate."""
    # Create expression: 2*x + 3*x + 4*y
    expr = 2 * x + 3 * x + 4 * y

    # Simplify
    simplified = expr.simplify()
    # Should have 2 terms (one for x with coeff 5, one for y with coeff 4)
    assert simplified.nterm == 2, f"Expected 2 terms, got {simplified.nterm}"

    coeffs: list[float] = simplified.coeffs.values.flatten().tolist()
    assert set(coeffs) == {5.0, 4.0}, (
        f"Expected coefficients {{5.0, 4.0}}, got {set(coeffs)}"
    )


def test_simplify_with_constant(x: Variable) -> None:
    """Test that constants are preserved."""
    expr = 2 * x + 3 * x + 10

    # Simplify
    simplified = expr.simplify()

    # Check constant is preserved
    assert all(simplified.const.values == 10.0), (
        f"Expected constant 10.0, got {simplified.const.values}"
    )

    # Check coefficients
    assert all(simplified.coeffs.values == 5.0), (
        f"Expected coefficient 5.0, got {simplified.coeffs.values}"
    )


def test_simplify_cancellation(x: Variable) -> None:
    """Test that terms cancel out correctly when coefficients sum to zero."""
    expr = x - x
    simplified = expr.simplify()

    assert simplified.nterm == 0, f"Expected 0 terms, got {simplified.nterm}"
    assert simplified.coeffs.values.size == 0
    assert simplified.vars.values.size == 0


def test_simplify_partial_cancellation(x: Variable, y: Variable) -> None:
    """Test partial cancellation where some terms cancel but others remain."""
    expr = 2 * x - 2 * x + 3 * y
    simplified = expr.simplify()

    assert simplified.nterm == 1, f"Expected 1 term, got {simplified.nterm}"
    assert all(simplified.coeffs.values == 3.0), (
        f"Expected coefficient 3.0, got {simplified.coeffs.values}"
    )


def test_constant_only_expression_mul_dataarray(m: Model) -> None:
    const_arr = xr.DataArray([2, 3], dims=["dim_0"])
    const_expr = LinearExpression(const_arr, m)
    assert const_expr.is_constant
    assert const_expr.nterm == 0

    data_arr = xr.DataArray([10, 20], dims=["dim_0"])
    expected_const = const_arr * data_arr

    result = const_expr * data_arr
    assert isinstance(result, LinearExpression)
    assert result.is_constant
    assert (result.const == expected_const).all()

    result_rev = data_arr * const_expr
    assert isinstance(result_rev, LinearExpression)
    assert result_rev.is_constant
    assert (result_rev.const == expected_const).all()


def test_constant_only_expression_mul_linexpr_with_vars(m: Model, x: Variable) -> None:
    const_arr = xr.DataArray([2, 3], dims=["dim_0"])
    const_expr = LinearExpression(const_arr, m)
    assert const_expr.is_constant
    assert const_expr.nterm == 0

    expr_with_vars = 1 * x + 5
    expected_coeffs = const_arr
    expected_const = const_arr * 5

    result = const_expr * expr_with_vars
    assert isinstance(result, LinearExpression)
    assert (result.coeffs == expected_coeffs).all()
    assert (result.const == expected_const).all()

    result_rev = expr_with_vars * const_expr
    assert isinstance(result_rev, LinearExpression)
    assert (result_rev.coeffs == expected_coeffs).all()
    assert (result_rev.const == expected_const).all()


def test_constant_only_expression_mul_constant_only(m: Model) -> None:
    const_arr = xr.DataArray([2, 3], dims=["dim_0"])
    const_arr2 = xr.DataArray([4, 5], dims=["dim_0"])
    const_expr = LinearExpression(const_arr, m)
    const_expr2 = LinearExpression(const_arr2, m)
    assert const_expr.is_constant
    assert const_expr2.is_constant

    expected_const = const_arr * const_arr2

    result = const_expr * const_expr2
    assert isinstance(result, LinearExpression)
    assert result.is_constant
    assert (result.const == expected_const).all()

    result_rev = const_expr2 * const_expr
    assert isinstance(result_rev, LinearExpression)
    assert result_rev.is_constant
    assert (result_rev.const == expected_const).all()


def test_constant_only_expression_mul_linexpr_with_vars_and_const(
    m: Model, x: Variable
) -> None:
    const_arr = xr.DataArray([2, 3], dims=["dim_0"])
    const_expr = LinearExpression(const_arr, m)
    assert const_expr.is_constant

    expr_with_vars_and_const = 4 * x + 10
    expected_coeffs = const_arr * 4
    expected_const = const_arr * 10

    result = const_expr * expr_with_vars_and_const
    assert isinstance(result, LinearExpression)
    assert not result.is_constant
    assert (result.coeffs == expected_coeffs).all()
    assert (result.const == expected_const).all()

    result_rev = expr_with_vars_and_const * const_expr
    assert isinstance(result_rev, LinearExpression)
    assert not result_rev.is_constant
    assert (result_rev.coeffs == expected_coeffs).all()
    assert (result_rev.const == expected_const).all()


def test_variable_names() -> None:
    m = Model()
    time = pd.Index(range(3), name="time")

    a = m.add_variables(name="a", coords=[time])
    b = m.add_variables(name="b", coords=[time])

    expr = a + b
    assert expr.nterm == 2
    assert expr.variable_names == {"a", "b"}

    mask = xr.DataArray(False, coords=[time])
    expr = a + (b * 1).where(mask)
    assert expr.nterm == 2
    assert expr.variable_names == {"a"}

    expr = (b * 1).where(mask)
    assert expr.nterm == 1
    assert expr.variable_names == set()

    expr = LinearExpression.from_constant(model=m, constant=5)
    assert expr.nterm == 0
    assert expr.variable_names == set()

    # Single variable expression
    expr = 1 * a
    assert expr.variable_names == {"a"}

    # Repeated variable across terms (a + a)
    expr = a + a
    assert expr.variable_names == {"a"}


def test_nterm() -> None:
    m = Model()
    time = pd.Index(range(3), name="time")
    all_false = xr.DataArray(False, coords=[time])
    not_0 = xr.DataArray([False, True, True], coords=[time])
    not_1 = xr.DataArray([True, False, True], coords=[time])
    not_2 = xr.DataArray([True, True, False], coords=[time])

    a = m.add_variables(name="a", coords=[time])
    b = m.add_variables(name="b", coords=[time])
    c = m.add_variables(name="c", coords=[time])

    expr = (a.where(not_0) + b.where(not_1) + c.where(not_2)).densify_terms()
    assert expr.nterm == 3

    expr = a + b.where(all_false)
    assert expr.nterm == 2

    expr = expr.simplify()
    assert expr.nterm == 1


class TestJoinParameter:
    @pytest.fixture
    def m2(self) -> Model:
        m = Model()
        m.add_variables(coords=[pd.Index([0, 1, 2], name="i")], name="a")
        m.add_variables(coords=[pd.Index([1, 2, 3], name="i")], name="b")
        m.add_variables(coords=[pd.Index([0, 1, 2], name="i")], name="c")
        return m

    @pytest.fixture
    def a(self, m2: Model) -> Variable:
        return m2.variables["a"]

    @pytest.fixture
    def b(self, m2: Model) -> Variable:
        return m2.variables["b"]

    @pytest.fixture
    def c(self, m2: Model) -> Variable:
        return m2.variables["c"]

    class TestAddition:
        def test_add_join_none_preserves_default(
            self, a: Variable, b: Variable
        ) -> None:
            result_default = a.to_linexpr() + b.to_linexpr()
            result_none = a.to_linexpr().add(b.to_linexpr(), join=None)
            assert_linequal(result_default, result_none)

        def test_add_expr_join_inner(self, a: Variable, b: Variable) -> None:
            result = a.to_linexpr().add(b.to_linexpr(), join="inner")
            assert list(result.indexes["i"]) == [1, 2]

        def test_add_expr_join_outer(self, a: Variable, b: Variable) -> None:
            result = a.to_linexpr().add(b.to_linexpr(), join="outer")
            assert list(result.indexes["i"]) == [0, 1, 2, 3]

        def test_add_expr_join_left(self, a: Variable, b: Variable) -> None:
            result = a.to_linexpr().add(b.to_linexpr(), join="left")
            assert list(result.indexes["i"]) == [0, 1, 2]

        def test_add_expr_join_right(self, a: Variable, b: Variable) -> None:
            result = a.to_linexpr().add(b.to_linexpr(), join="right")
            assert list(result.indexes["i"]) == [1, 2, 3]

        def test_add_constant_join_inner(self, a: Variable) -> None:
            const = xr.DataArray([10, 20, 30], dims=["i"], coords={"i": [1, 2, 3]})
            result = a.to_linexpr().add(const, join="inner")
            assert list(result.indexes["i"]) == [1, 2]

        def test_add_constant_join_outer(self, a: Variable) -> None:
            const = xr.DataArray([10, 20, 30], dims=["i"], coords={"i": [1, 2, 3]})
            result = a.to_linexpr().add(const, join="outer")
            assert list(result.indexes["i"]) == [0, 1, 2, 3]

        def test_add_constant_join_override(self, a: Variable, c: Variable) -> None:
            expr = a.to_linexpr()
            const = xr.DataArray([10, 20, 30], dims=["i"], coords={"i": [0, 1, 2]})
            result = expr.add(const, join="override")
            assert list(result.indexes["i"]) == [0, 1, 2]
            assert (result.const.values == const.values).all()

        def test_add_same_coords_all_joins(self, a: Variable, c: Variable) -> None:
            expr_a = 1 * a + 5
            const = xr.DataArray([1, 2, 3], dims=["i"], coords={"i": [0, 1, 2]})
            joins: list[JoinOptions] = ["override", "outer", "inner"]
            for join in joins:
                result = expr_a.add(const, join=join)
                assert list(result.coords["i"].values) == [0, 1, 2]
                np.testing.assert_array_equal(result.const.values, [6, 7, 8])

        def test_add_scalar_with_explicit_join(self, a: Variable) -> None:
            expr = 1 * a + 5
            result = expr.add(10, join="override")
            np.testing.assert_array_equal(result.const.values, [15, 15, 15])
            assert list(result.coords["i"].values) == [0, 1, 2]

    class TestSubtraction:
        def test_sub_expr_join_inner(self, a: Variable, b: Variable) -> None:
            result = a.to_linexpr().sub(b.to_linexpr(), join="inner")
            assert list(result.indexes["i"]) == [1, 2]

        def test_sub_constant_override(self, a: Variable) -> None:
            expr = 1 * a + 5
            other = xr.DataArray([10, 20, 30], dims=["i"], coords={"i": [5, 6, 7]})
            result = expr.sub(other, join="override")
            assert list(result.coords["i"].values) == [0, 1, 2]
            np.testing.assert_array_equal(result.const.values, [-5, -15, -25])

    class TestMultiplication:
        def test_mul_constant_join_inner(self, a: Variable) -> None:
            const = xr.DataArray([2, 3, 4], dims=["i"], coords={"i": [1, 2, 3]})
            result = a.to_linexpr().mul(const, join="inner")
            assert list(result.indexes["i"]) == [1, 2]

        def test_mul_constant_join_outer(self, a: Variable) -> None:
            const = xr.DataArray([2, 3, 4], dims=["i"], coords={"i": [1, 2, 3]})
            result = a.to_linexpr().mul(const, join="outer")
            assert list(result.indexes["i"]) == [0, 1, 2, 3]
            assert result.coeffs.sel(i=0).item() == 0
            assert result.coeffs.sel(i=1).item() == 2
            assert result.coeffs.sel(i=2).item() == 3

        def test_mul_expr_with_join_raises(self, a: Variable, b: Variable) -> None:
            with pytest.raises(TypeError, match="join parameter is not supported"):
                a.to_linexpr().mul(b.to_linexpr(), join="inner")

    class TestDivision:
        def test_div_constant_join_inner(self, a: Variable) -> None:
            const = xr.DataArray([2, 3, 4], dims=["i"], coords={"i": [1, 2, 3]})
            result = a.to_linexpr().div(const, join="inner")
            assert list(result.indexes["i"]) == [1, 2]

        def test_div_constant_join_outer(self, a: Variable) -> None:
            const = xr.DataArray([2, 3, 4], dims=["i"], coords={"i": [1, 2, 3]})
            result = a.to_linexpr().div(const, join="outer")
            assert list(result.indexes["i"]) == [0, 1, 2, 3]

        def test_div_expr_with_join_raises(self, a: Variable, b: Variable) -> None:
            with pytest.raises(TypeError):
                a.to_linexpr().div(b.to_linexpr(), join="outer")

    class TestVariableOperations:
        def test_variable_add_join(self, a: Variable, b: Variable) -> None:
            result = a.add(b, join="inner")
            assert list(result.indexes["i"]) == [1, 2]

        def test_variable_sub_join(self, a: Variable, b: Variable) -> None:
            result = a.sub(b, join="inner")
            assert list(result.indexes["i"]) == [1, 2]

        def test_variable_mul_join(self, a: Variable) -> None:
            const = xr.DataArray([2, 3, 4], dims=["i"], coords={"i": [1, 2, 3]})
            result = a.mul(const, join="inner")
            assert list(result.indexes["i"]) == [1, 2]

        def test_variable_div_join(self, a: Variable) -> None:
            const = xr.DataArray([2, 3, 4], dims=["i"], coords={"i": [1, 2, 3]})
            result = a.div(const, join="inner")
            assert list(result.indexes["i"]) == [1, 2]

        def test_variable_add_outer_values(self, a: Variable, b: Variable) -> None:
            result = a.add(b, join="outer")
            assert isinstance(result, LinearExpression)
            assert set(result.coords["i"].values) == {0, 1, 2, 3}
            assert result.nterm == 2

        def test_variable_mul_override(self, a: Variable) -> None:
            other = xr.DataArray([2, 3, 4], dims=["i"], coords={"i": [5, 6, 7]})
            result = a.mul(other, join="override")
            assert isinstance(result, LinearExpression)
            assert list(result.coords["i"].values) == [0, 1, 2]
            np.testing.assert_array_equal(result.coeffs.squeeze().values, [2, 3, 4])

        def test_variable_div_override(self, a: Variable) -> None:
            other = xr.DataArray([2.0, 5.0, 10.0], dims=["i"], coords={"i": [5, 6, 7]})
            result = a.div(other, join="override")
            assert isinstance(result, LinearExpression)
            assert list(result.coords["i"].values) == [0, 1, 2]
            np.testing.assert_array_almost_equal(
                result.coeffs.squeeze().values, [0.5, 0.2, 0.1]
            )

        def test_same_shape_add_join_override(self, a: Variable, c: Variable) -> None:
            result = a.to_linexpr().add(c.to_linexpr(), join="override")
            assert list(result.indexes["i"]) == [0, 1, 2]

    class TestMerge:
        def test_merge_join_parameter(self, a: Variable, b: Variable) -> None:
            result = merge(
                [a.to_linexpr(), b.to_linexpr()], cls=LinearExpression, join="inner"
            )
            assert list(result.indexes["i"]) == [1, 2]

        def test_merge_outer_join(self, a: Variable, b: Variable) -> None:
            result = merge(
                [a.to_linexpr(), b.to_linexpr()], cls=LinearExpression, join="outer"
            )
            assert set(result.coords["i"].values) == {0, 1, 2, 3}

        def test_merge_join_left(self, a: Variable, b: Variable) -> None:
            result = merge(
                [a.to_linexpr(), b.to_linexpr()], cls=LinearExpression, join="left"
            )
            assert list(result.indexes["i"]) == [0, 1, 2]

        def test_merge_join_right(self, a: Variable, b: Variable) -> None:
            result = merge(
                [a.to_linexpr(), b.to_linexpr()], cls=LinearExpression, join="right"
            )
            assert list(result.indexes["i"]) == [1, 2, 3]

    class TestValueVerification:
        def test_add_expr_outer_const_values(self, a: Variable, b: Variable) -> None:
            expr_a = 1 * a + 5
            expr_b = 2 * b + 10
            result = expr_a.add(expr_b, join="outer")
            assert set(result.coords["i"].values) == {0, 1, 2, 3}
            assert result.const.sel(i=0).item() == 5
            assert result.const.sel(i=1).item() == 15
            assert result.const.sel(i=2).item() == 15
            assert result.const.sel(i=3).item() == 10

        def test_add_expr_inner_const_values(self, a: Variable, b: Variable) -> None:
            expr_a = 1 * a + 5
            expr_b = 2 * b + 10
            result = expr_a.add(expr_b, join="inner")
            assert list(result.coords["i"].values) == [1, 2]
            assert result.const.sel(i=1).item() == 15
            assert result.const.sel(i=2).item() == 15

        def test_add_constant_outer_fill_values(self, a: Variable) -> None:
            expr = 1 * a + 5
            const = xr.DataArray([10, 20], dims=["i"], coords={"i": [1, 3]})
            result = expr.add(const, join="outer")
            assert set(result.coords["i"].values) == {0, 1, 2, 3}
            assert result.const.sel(i=0).item() == 5
            assert result.const.sel(i=1).item() == 15
            assert result.const.sel(i=2).item() == 5
            assert result.const.sel(i=3).item() == 20

        def test_add_constant_inner_fill_values(self, a: Variable) -> None:
            expr = 1 * a + 5
            const = xr.DataArray([10, 20], dims=["i"], coords={"i": [1, 3]})
            result = expr.add(const, join="inner")
            assert list(result.coords["i"].values) == [1]
            assert result.const.sel(i=1).item() == 15

        def test_add_constant_override_positional(self, a: Variable) -> None:
            expr = 1 * a + 5
            other = xr.DataArray([10, 20, 30], dims=["i"], coords={"i": [5, 6, 7]})
            result = expr.add(other, join="override")
            assert list(result.coords["i"].values) == [0, 1, 2]
            np.testing.assert_array_equal(result.const.values, [15, 25, 35])

        def test_sub_expr_outer_const_values(self, a: Variable, b: Variable) -> None:
            expr_a = 1 * a + 5
            expr_b = 2 * b + 10
            result = expr_a.sub(expr_b, join="outer")
            assert set(result.coords["i"].values) == {0, 1, 2, 3}
            assert result.const.sel(i=0).item() == 5
            assert result.const.sel(i=1).item() == -5
            assert result.const.sel(i=2).item() == -5
            assert result.const.sel(i=3).item() == -10

        def test_mul_constant_override_positional(self, a: Variable) -> None:
            expr = 1 * a + 5
            other = xr.DataArray([2, 3, 4], dims=["i"], coords={"i": [5, 6, 7]})
            result = expr.mul(other, join="override")
            assert list(result.coords["i"].values) == [0, 1, 2]
            np.testing.assert_array_equal(result.const.values, [10, 15, 20])
            np.testing.assert_array_equal(result.coeffs.squeeze().values, [2, 3, 4])

        def test_mul_constant_outer_fill_values(self, a: Variable) -> None:
            expr = 1 * a + 5
            other = xr.DataArray([2, 3], dims=["i"], coords={"i": [1, 3]})
            result = expr.mul(other, join="outer")
            assert set(result.coords["i"].values) == {0, 1, 2, 3}
            assert result.const.sel(i=0).item() == 0
            assert result.const.sel(i=1).item() == 10
            assert result.const.sel(i=2).item() == 0
            assert result.const.sel(i=3).item() == 0
            assert result.coeffs.squeeze().sel(i=1).item() == 2
            assert result.coeffs.squeeze().sel(i=0).item() == 0

        def test_div_constant_override_positional(self, a: Variable) -> None:
            expr = 1 * a + 10
            other = xr.DataArray([2.0, 5.0, 10.0], dims=["i"], coords={"i": [5, 6, 7]})
            result = expr.div(other, join="override")
            assert list(result.coords["i"].values) == [0, 1, 2]
            np.testing.assert_array_equal(result.const.values, [5.0, 2.0, 1.0])

        def test_div_constant_outer_fill_values(self, a: Variable) -> None:
            expr = 1 * a + 10
            other = xr.DataArray([2.0, 5.0], dims=["i"], coords={"i": [1, 3]})
            result = expr.div(other, join="outer")
            assert set(result.coords["i"].values) == {0, 1, 2, 3}
            assert result.const.sel(i=1).item() == pytest.approx(5.0)
            assert result.coeffs.squeeze().sel(i=1).item() == pytest.approx(0.5)
            assert result.const.sel(i=0).item() == pytest.approx(10.0)
            assert result.coeffs.squeeze().sel(i=0).item() == pytest.approx(1.0)

    class TestQuadratic:
        def test_quadratic_add_constant_join_inner(
            self, a: Variable, b: Variable
        ) -> None:
            quad = a.to_linexpr() * b.to_linexpr()
            const = xr.DataArray([10, 20, 30], dims=["i"], coords={"i": [1, 2, 3]})
            result = quad.add(const, join="inner")
            assert list(result.indexes["i"]) == [1, 2, 3]

        def test_quadratic_add_expr_join_inner(self, a: Variable) -> None:
            quad = a.to_linexpr() * a.to_linexpr()
            const = xr.DataArray([10, 20], dims=["i"], coords={"i": [0, 1]})
            result = quad.add(const, join="inner")
            assert list(result.indexes["i"]) == [0, 1]

        def test_quadratic_mul_constant_join_inner(
            self, a: Variable, b: Variable
        ) -> None:
            quad = a.to_linexpr() * b.to_linexpr()
            const = xr.DataArray([2, 3, 4], dims=["i"], coords={"i": [1, 2, 3]})
            result = quad.mul(const, join="inner")
            assert list(result.indexes["i"]) == [1, 2, 3]
