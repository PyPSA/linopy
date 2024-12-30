import pandas as pd
import pytest

from linopy import Model, Variable
from linopy.expressions import ScalarLinearExpression


@pytest.fixture
def m() -> Model:
    m = Model()

    m.add_variables(pd.Series([0, 0]), 1, name="x")
    m.add_variables(4, pd.Series([8, 10]), name="y")
    m.add_variables(0, pd.DataFrame([[1, 2], [3, 4], [5, 6]]).T, name="z")
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


def test_scalar_expression_initialization(
    x: Variable, y: Variable, z: Variable
) -> None:
    expr: ScalarLinearExpression = 10 * x.at[0]
    assert isinstance(expr, ScalarLinearExpression)

    expr2: ScalarLinearExpression = 10 * x.at[0] + y.at[1] + z.at[1, 1]
    assert isinstance(expr2, ScalarLinearExpression)


def test_scalar_expression_multiplication(x: Variable) -> None:
    expr: ScalarLinearExpression = 10 * x.at[0]
    expr2: ScalarLinearExpression = 2 * expr
    assert isinstance(expr2, ScalarLinearExpression)
    assert expr2.coeffs == (20,)


def test_scalar_expression_division(x: Variable) -> None:
    expr: ScalarLinearExpression = 10 * x.at[0]
    expr2: ScalarLinearExpression = expr / 2
    assert isinstance(expr2, ScalarLinearExpression)
    assert expr2.coeffs == (5,)

    expr3: ScalarLinearExpression = expr / 2.0
    assert isinstance(expr3, ScalarLinearExpression)
    assert expr3.coeffs == (5,)


def test_scalar_expression_negation(x: Variable) -> None:
    expr: ScalarLinearExpression = 10 * x.at[0]
    expr3: ScalarLinearExpression = -expr
    assert isinstance(expr3, ScalarLinearExpression)
    assert expr3.coeffs == (-10,)


def test_scalar_expression_multiplication_raises_type_error(x: Variable) -> None:
    with pytest.raises(TypeError):
        x.at[1] * x.at[1]  # type: ignore


def test_scalar_expression_division_raises_type_error(x: Variable) -> None:
    with pytest.raises(TypeError):
        x.at[1] / x.at[1]  # type: ignore


def test_scalar_expression_sum(x: Variable, y: Variable, z: Variable) -> None:
    target: ScalarLinearExpression = 10 * x.at[0] + y.at[1] + z.at[1, 1]
    expr: ScalarLinearExpression = sum((10 * x.at[0], y.at[1], z.at[1, 1]))  # type: ignore
    assert isinstance(expr, ScalarLinearExpression)
    assert expr.vars == target.vars
    assert expr.coeffs == target.coeffs


def test_scalar_expression_sum_from_variables(x: Variable, y: Variable) -> None:
    target: ScalarLinearExpression = x.at[0] + y.at[0]
    expr: ScalarLinearExpression = sum((x.at[0], y.at[0]))  # type: ignore
    assert isinstance(expr, ScalarLinearExpression)
    assert expr.vars == target.vars
    assert expr.coeffs == target.coeffs
