#!/usr/bin/env python3

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.sparse import csc_matrix
from xarray import DataArray

from linopy import Model, Variable, merge
from linopy.constants import FACTOR_DIM, TERM_DIM
from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.testing import assert_quadequal


@pytest.fixture
def model() -> Model:
    return Model()


@pytest.fixture
def x(model: Model) -> Variable:
    return model.add_variables(pd.Series([0, 0]), 1, name="x")


@pytest.fixture
def y(model: Model) -> Variable:
    return model.add_variables(4, pd.Series([8, 10]), name="y")


@pytest.fixture
def z(model: Model) -> Variable:
    return model.add_variables(4, pd.Series([8, 10]), name="z")


def test_quadratic_expression_from_variables_multiplication(
    x: Variable, y: Variable
) -> None:
    quad_expr = x * y
    assert isinstance(quad_expr, QuadraticExpression)
    assert quad_expr.data.sizes[FACTOR_DIM] == 2


def test_adding_quadratic_expressions(x: Variable) -> None:
    quad_expr = x * x
    double_quad = quad_expr + quad_expr
    assert isinstance(double_quad, QuadraticExpression)
    assert double_quad.__add__(object()) is NotImplemented


def test_quadratic_expression_from_variables_power(x: Variable) -> None:
    power_expr = x**2
    target: QuadraticExpression = x * x  # type: ignore
    assert isinstance(power_expr, QuadraticExpression)
    assert power_expr.data.sizes[FACTOR_DIM] == 2
    assert_quadequal(power_expr, target)
    assert_quadequal(x.pow(2), target)


def test_quadratic_expression_from_linexpr_multiplication(
    x: Variable, y: Variable
) -> None:
    mult_expr = (10 * x + y) * y
    target: QuadraticExpression = 10 * x * y + y * y  # type: ignore
    assert isinstance(mult_expr, QuadraticExpression)
    assert mult_expr.data.sizes[FACTOR_DIM] == 2
    assert mult_expr.nterm == 2
    assert_quadequal(mult_expr, target)


def test_quadratic_expression_from_linexpr_power(x: Variable) -> None:
    expr = (10 * x) ** 2
    assert isinstance(expr, QuadraticExpression)
    assert expr.data.sizes[FACTOR_DIM] == 2
    assert expr.nterm == 1


def test_quadratic_expression_from_linexpr_with_constant_power(x: Variable) -> None:
    expr = (10 * x + 5) ** 2
    target: QuadraticExpression = 100 * x * x + 50 * x + 50 * x + 25  # type: ignore
    assert isinstance(expr, QuadraticExpression)
    assert expr.data.sizes[FACTOR_DIM] == 2
    assert expr.nterm == 3
    assert_quadequal(expr, target)


def test_quadratic_expression_from_linexpr_with_constant_multiplation(
    x: Variable, y: Variable
) -> None:
    expr: QuadraticExpression = (10 * x + 5) * (y + 5)  # type: ignore
    target: QuadraticExpression = 10 * x * y + 5 * y + 50 * x + 25  # type: ignore
    assert isinstance(expr, QuadraticExpression)
    assert expr.data.sizes[FACTOR_DIM] == 2
    assert expr.nterm == 3
    assert_quadequal(expr, target)


def test_quadratic_expression_scalar_mul_dot_right(x: Variable, y: Variable) -> None:
    expr: QuadraticExpression = 10 * x @ y  # type: ignore
    assert expr.nterm == 2
    target: QuadraticExpression = (10 * x * y).sum()  # type: ignore
    assert_quadequal(expr, target)


def test_quadratic_expression_scalar_mul_dot_left(x: Variable, y: Variable) -> None:
    expr: QuadraticExpression = y @ (10 * x)  # type: ignore
    assert expr.nterm == 2
    target: QuadraticExpression = (y * 10 * x).sum()  # type: ignore
    assert_quadequal(expr, target)


def test_quadratic_expression_dot_method(x: Variable, y: Variable) -> None:
    expr: QuadraticExpression = x.dot(y)  # type: ignore
    target: QuadraticExpression = x @ y  # type: ignore
    assert_quadequal(expr, target)


def test_matmul_expr_and_expr(x: Variable, y: Variable, z: Variable) -> None:
    expr: QuadraticExpression = (2 * x + 5) @ (3 * y + 10)  # type: ignore
    target: QuadraticExpression = (
        2 * 3 * x @ y + 5 * 3 * y.sum() + 2 * 10 * x.sum() + 5 * 10 * 2  # type: ignore
    )
    assert expr.nterm == 6
    assert_quadequal(expr, target)

    with pytest.raises(TypeError):
        (x**2) @ (y**2)


def test_matmul_with_const(x: Variable) -> None:
    expr = x * x
    const = DataArray([2.0, 1.0], dims=["dim_0"])
    expr2 = expr @ const
    assert isinstance(expr2, QuadraticExpression)
    assert expr2.nterm == 2
    assert expr2.data.sizes[FACTOR_DIM] == 2


def test_quadratic_expression_dot_and_matmul(x: Variable, y: Variable) -> None:
    matmul_expr: QuadraticExpression = 10 * x @ y  # type: ignore
    dot_expr: QuadraticExpression = 10 * x.dot(y)  # type: ignore
    assert_quadequal(matmul_expr, dot_expr)


def test_quadratic_expression_wrong_assignment(x: Variable, y: Variable) -> None:
    with pytest.raises(ValueError):
        QuadraticExpression((x + y).data, x.model)

    with pytest.raises(ValueError):
        QuadraticExpression((x + y).data.expand_dims(FACTOR_DIM), x.model)


def test_quadratic_expression_addition(model: Model, x: Variable, y: Variable) -> None:
    expr = x * y + x + 5
    assert isinstance(expr, QuadraticExpression)
    assert (expr.const == 5).all()
    assert expr.nterm == 2


def test_quadratic_expression_raddition(x: Variable, y: Variable) -> None:
    expr = x + x * y + 5
    assert isinstance(expr, QuadraticExpression)
    assert (expr.const == 5).all()
    assert expr.nterm == 2

    expr_2 = 5 + x * y + x
    assert isinstance(expr_2, QuadraticExpression)
    assert (expr_2.const == 5).all()
    assert expr_2.nterm == 2

    assert_quadequal(expr, expr_2)


def test_quadratic_expression_subtraction(x: Variable, y: Variable) -> None:
    expr = x * y - x - 5
    assert isinstance(expr, QuadraticExpression)
    assert (expr.const == -5).all()
    assert expr.nterm == 2
    assert expr.__sub__(object()) is NotImplemented


def test_quadratic_expression_rsubtraction(x: Variable, y: Variable) -> None:
    expr = x - x * y - 5
    assert isinstance(expr, QuadraticExpression)
    assert (expr.const == -5).all()
    assert expr.nterm == 2

    expr2 = 5 - x * y
    assert isinstance(expr2, QuadraticExpression)
    assert (expr2.const == 5).all()
    assert expr2.nterm == 1


def test_quadratic_expression_sum(x: Variable, y: Variable) -> None:
    base_expr = x * y + x + 5

    summed_expr = base_expr.sum(dim="dim_0")
    assert isinstance(summed_expr, QuadraticExpression)
    assert not summed_expr.coord_dims

    summed_expr_all = base_expr.sum()
    assert isinstance(summed_expr_all, QuadraticExpression)
    assert not summed_expr_all.coord_dims


def test_quadratic_expression_sum_warn_using_dims(x: Variable) -> None:
    with pytest.warns(DeprecationWarning):
        (x**2).sum(dims="dim_0")


def test_quadratic_expression_sum_warn_unknown_kwargs(x: Variable) -> None:
    with pytest.raises(ValueError):
        (x**2).sum(unknown_kwarg="dim_0")


def test_quadratic_expression_wrong_multiplication(x: Variable, y: Variable) -> None:
    with pytest.raises(TypeError):
        x * x * y

    quad = x * x
    with pytest.raises(TypeError):
        quad * quad


def merge_raise_deprecation_warning(x: Variable, y: Variable) -> None:
    expr: QuadraticExpression = x * y  # type: ignore
    with pytest.warns(DeprecationWarning):
        merge(expr, expr)  # type: ignore


def test_merge_linear_expression_and_quadratic_expression(
    x: Variable, y: Variable
) -> None:
    linexpr: LinearExpression = 10 * x + y + 5
    quadexpr: QuadraticExpression = x * y  # type: ignore

    merge([linexpr.to_quadexpr(), quadexpr], cls=QuadraticExpression)
    with pytest.raises(ValueError):
        merge([linexpr, quadexpr], cls=QuadraticExpression)

    new_quad_ex = merge([linexpr.to_quadexpr(), quadexpr])  # type: ignore
    assert isinstance(new_quad_ex, QuadraticExpression)

    with pytest.warns(DeprecationWarning):
        merge(quadexpr, quadexpr, cls=QuadraticExpression)  # type: ignore

    quadexpr_2 = linexpr.to_quadexpr()
    merged_expr = merge([quadexpr_2, quadexpr], cls=QuadraticExpression)
    assert isinstance(merged_expr, QuadraticExpression)
    assert merged_expr.nterm == 3
    assert merged_expr.const.sum() == 10
    assert FACTOR_DIM not in merged_expr.coeffs.dims
    assert FACTOR_DIM not in merged_expr.const.dims

    first_term = merged_expr.data.isel({TERM_DIM: 0})
    assert (first_term.vars.isel({FACTOR_DIM: 1}) == -1).all()

    qdexpr = merge([x**2, y**2], cls=QuadraticExpression)
    assert isinstance(qdexpr, QuadraticExpression)

    with pytest.raises(ValueError):
        merge([x**2, y**2], cls=LinearExpression)


def test_quadratic_expression_loc(x: Variable) -> None:
    expr = x * x
    assert expr.loc[0].size < expr.loc[:5].size


def test_quadratic_expression_isnull(x: Variable) -> None:
    test_expr = np.arange(2) * x * x
    filter = (test_expr.coeffs > 0).any(TERM_DIM)
    filtered_expr = test_expr.where(filter)
    isnull = filtered_expr.isnull()
    assert isinstance(isnull, DataArray)
    assert isnull.sum() == 1


def test_quadratic_expression_flat(x: Variable, y: Variable) -> None:
    expr = x * y + x + 5
    df = expr.flat
    assert isinstance(df, pd.DataFrame)

    expr = x * y + x * y
    assert expr.nterm == 2
    assert (expr.flat.coeffs == 2).all()
    assert len(expr.flat) == 2


def test_linear_expression_to_polars(x: Variable, y: Variable) -> None:
    expr = x * y + x + 5
    df = expr.to_polars()
    assert isinstance(df, pl.DataFrame)
    assert "vars1" in df.columns
    assert "vars2" in df.columns
    assert len(df) == expr.nterm * 2


def test_quadratic_expression_to_matrix(model: Model, x: Variable, y: Variable) -> None:
    expr: QuadraticExpression = x * y + x + 5  # type: ignore

    Q = expr.to_matrix()
    assert isinstance(Q, csc_matrix)
    assert Q.shape == (model.nvars, model.nvars)


def test_matrices_matrix(model: Model, x: Variable, y: Variable) -> None:
    expr = 10 * x * y
    model.objective = expr

    Q = model.matrices.Q
    assert isinstance(Q, csc_matrix)
    assert Q.shape == (model.nvars, model.nvars)


def test_matrices_matrix_mixed_linear_and_quadratic(
    model: Model, x: Variable, y: Variable
) -> None:
    quad_expr = x * y + x
    model.objective = quad_expr + x

    Q = model.matrices.Q
    assert isinstance(Q, csc_matrix)
    assert Q.shape == (model._xCounter, model._xCounter)

    c = model.matrices.c
    assert isinstance(c, np.ndarray)
    assert c.shape == (model.nvars,)


def test_quadratic_to_constraint(x: Variable, y: Variable) -> None:
    with pytest.raises(NotImplementedError):
        x * y <= 10


def test_power_of_three(x: Variable) -> None:
    with pytest.raises(TypeError):
        x * x * x
    with pytest.raises(TypeError):
        (x * 1) * (x * x)
    with pytest.raises(TypeError):
        (x * x) * (x * 1)
    with pytest.raises(ValueError):
        x**3
    with pytest.raises(TypeError):
        (x * x) * (x * x)
