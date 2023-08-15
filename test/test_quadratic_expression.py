#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csc_matrix

from linopy import Model, merge
from linopy.constants import FACTOR_DIM, TERM_DIM
from linopy.expressions import QuadraticExpression


@pytest.fixture
def model():
    return Model()


@pytest.fixture
def x(model):
    return model.add_variables(pd.Series([0, 0]), 1, name="x")


@pytest.fixture
def y(model):
    return model.add_variables(4, pd.Series([8, 10]), name="y")


@pytest.fixture
def z(model):
    return model.add_variables(4, pd.Series([8, 10]), name="z")


def test_quadratic_expression_from_variables_multiplication(x, y):
    expr = x * y
    assert isinstance(expr, QuadraticExpression)
    assert expr.data.dims[FACTOR_DIM] == 2


def test_quadratic_expression_from_linexpr_multiplication(x, y):
    expr = (10 * x + y) * y
    assert isinstance(expr, QuadraticExpression)
    assert expr.data.dims[FACTOR_DIM] == 2
    assert expr.nterm == 2


def test_quadratic_expression_wrong_assignment(x, y):
    with pytest.raises(ValueError):
        QuadraticExpression((x + y).data, x.model)

    with pytest.raises(ValueError):
        QuadraticExpression((x + y).data.expand_dims(FACTOR_DIM), x.model)


def test_quadratic_expression_addition(model, x, y):
    expr = x * y + x + 5
    assert isinstance(expr, QuadraticExpression)
    assert (expr.const == 5).all()
    assert expr.nterm == 2


def test_quadratic_expression_raddition(x, y):
    expr = x + x * y + 5
    assert isinstance(expr, QuadraticExpression)
    assert (expr.const == 5).all()
    assert expr.nterm == 2

    with pytest.raises(TypeError):
        5 + x * y + x


def test_quadratic_expression_subtraction(x, y):
    expr = x * y - x - 5
    assert isinstance(expr, QuadraticExpression)
    assert (expr.const == -5).all()
    assert expr.nterm == 2


def test_quadratic_expression_rsubtraction(x, y):
    expr = x - x * y - 5
    assert isinstance(expr, QuadraticExpression)
    assert (expr.const == -5).all()
    assert expr.nterm == 2


def test_quadratic_expression_sum(x, y):
    expr = x * y + x + 5

    summed_expr = expr.sum(dims="dim_0")
    assert isinstance(summed_expr, QuadraticExpression)
    assert not summed_expr.coord_dims

    summed_expr_all = expr.sum()
    assert isinstance(summed_expr_all, QuadraticExpression)
    assert not summed_expr_all.coord_dims


def test_quadratic_expression_wrong_multiplication(x, y):
    with pytest.raises(TypeError):
        x * x * y


def test_merge_linear_expression_and_quadratic_expression(x, y):
    linexpr = 10 * x + y + 5
    quadexpr = x * y

    with pytest.raises(ValueError):
        expr = merge(linexpr, quadexpr, cls=QuadraticExpression)

    linexpr = linexpr.to_quadexpr()
    expr = merge(linexpr, quadexpr, cls=QuadraticExpression)
    assert isinstance(expr, QuadraticExpression)
    assert expr.nterm == 3
    assert expr.const.sum() == 10
    assert FACTOR_DIM not in expr.coeffs.dims
    assert FACTOR_DIM not in expr.const.dims

    first_term = expr.data.isel({TERM_DIM: 0})
    assert (first_term.vars.isel({FACTOR_DIM: 1}) == -1).all()


def test_quadratic_expression_loc(x):
    expr = x * x
    assert expr.loc[0].size < expr.loc[:5].size


def test_quadratic_expression_flat(x, y):
    expr = x * y + x + 5
    df = expr.flat
    assert isinstance(df, pd.DataFrame)

    expr = x * y + x * y
    assert expr.nterm == 2
    assert (expr.flat.coeffs == 2).all()
    assert len(expr.flat) == 2


def test_quadratic_expression_to_matrix(model, x, y):
    expr = x * y + x + 5

    Q = expr.to_matrix()
    assert isinstance(Q, csc_matrix)
    assert Q.shape == (model.nvars, model.nvars)


def test_matrices_matrix(model, x, y):
    expr = 10 * x * y
    model.objective = expr

    Q = model.matrices.Q
    assert isinstance(Q, csc_matrix)
    assert Q.shape == (model.nvars, model.nvars)


def test_matrices_matrix_mixed_linear_and_quadratic(model, x, y):
    expr = x * y + x
    model.objective = expr + x

    Q = model.matrices.Q
    assert isinstance(Q, csc_matrix)
    assert Q.shape == (model._xCounter, model._xCounter)

    c = model.matrices.c
    assert isinstance(c, np.ndarray)
    assert c.shape == (model.nvars,)


def test_quadratic_to_constraint(x, y):
    with pytest.raises(NotImplementedError):
        x * y <= 10
