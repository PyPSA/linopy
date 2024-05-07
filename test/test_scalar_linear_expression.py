import pandas as pd
import pytest

from linopy import Model
from linopy.expressions import ScalarLinearExpression


@pytest.fixture
def m():
    m = Model()

    m.add_variables(pd.Series([0, 0]), 1, name="x")
    m.add_variables(4, pd.Series([8, 10]), name="y")
    m.add_variables(0, pd.DataFrame([[1, 2], [3, 4], [5, 6]]).T, name="z")
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


def test_scalar_expression_initialization(x, y, z):
    expr = 10 * x.at[0]
    assert isinstance(expr, ScalarLinearExpression)

    expr = 10 * x.at[0] + y.at[1] + z.at[1, 1]
    assert isinstance(expr, ScalarLinearExpression)


def test_scalar_expression_multiplication(x):
    expr = 10 * x.at[0]
    expr2 = 2 * expr
    assert isinstance(expr2, ScalarLinearExpression)
    assert expr2.coeffs == (20,)


def test_scalar_expression_division(x):
    expr = 10 * x.at[0]
    expr2 = expr / 2
    assert isinstance(expr2, ScalarLinearExpression)
    assert expr2.coeffs == (5,)

    expr2 = expr / 2.0
    assert isinstance(expr2, ScalarLinearExpression)
    assert expr2.coeffs == (5,)


def test_scalar_expression_negation(x):
    expr = 10 * x.at[0]
    expr3 = -expr
    assert isinstance(expr3, ScalarLinearExpression)
    assert expr3.coeffs == (-10,)


def test_scalar_expression_multiplication_raises_type_error(x):
    with pytest.raises(TypeError):
        x.at[1] * x.at[1]


def test_scalar_expression_division_raises_type_error(x):
    with pytest.raises(TypeError):
        x.at[1] / x.at[1]


def test_scalar_expression_sum(x, y, z):
    target = 10 * x.at[0] + y.at[1] + z.at[1, 1]
    expr = sum((10 * x.at[0], y.at[1], z.at[1, 1]))
    assert isinstance(expr, ScalarLinearExpression)
    assert expr.vars == target.vars
    assert expr.coeffs == target.coeffs


def test_scalar_expression_sum_from_variables(x, y):
    target = x.at[0] + y.at[0]
    expr = sum((x.at[0], y.at[0]))
    assert isinstance(expr, ScalarLinearExpression)
    assert expr.vars == target.vars
    assert expr.coeffs == target.coeffs
