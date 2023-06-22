#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csc_matrix

from linopy import Model
from linopy.constants import FACTOR_DIM
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


def test_quadratic_expression_addition(x, y):
    expr = x * y
    q = expr + x + 5
    assert isinstance(q, QuadraticExpression)
    assert (q.const == 5).all()


def test_quadratic_expression_sum(x, y):
    expr = x * y
    q = expr + x + 5

    summed_expr = q.sum(dims="dim_0")
    assert isinstance(summed_expr, QuadraticExpression)
    assert not summed_expr.coord_dims

    summed_expr_all = q.sum()
    assert isinstance(summed_expr_all, QuadraticExpression)
    assert not summed_expr_all.coord_dims


def test_quadratic_expression_flat(x, y):
    expr = x * y
    q = expr + x + 5

    df = q.flat
    assert isinstance(df, pd.DataFrame)


def test_quadratic_expression_to_matrix(model, x, y):
    expr = x * y
    q = expr + x + 5

    Q = q.to_matrix()
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
    model.objective = expr + x + 5

    Q = model.matrices.Q
    assert isinstance(Q, csc_matrix)
    assert Q.shape == (model._xCounter, model._xCounter)

    c = model.matrices.c
    assert isinstance(c, np.ndarray)
    assert c.shape == (model.nvars,)


# m = Model()
# x = m.add_variables(lower=[0,0], name="x")
# y = m.add_variables(upper=[10, 10], name="y")

# m.objective = x * x
