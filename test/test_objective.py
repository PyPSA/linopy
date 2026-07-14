import pandas as pd
import pytest
import xarray as xr
import xarray.core.indexes
import xarray.core.utils
from scipy.sparse import csc_matrix

from linopy import Model
from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.objective import Objective


@pytest.fixture
def linear_objective() -> Objective:
    m = Model()
    v = m.add_variables(coords=[[1, 2, 3]])
    m.objective = Objective(1 * v, m, sense="min")
    return m.objective


@pytest.fixture
def quadratic_objective() -> Objective:
    m = Model()
    v = m.add_variables(coords=[[1, 2, 3]])
    m.objective = Objective(v * v, m, sense="max")
    return m.objective


def test_model(linear_objective: Objective, quadratic_objective: Objective) -> None:
    assert isinstance(linear_objective.model, Model)
    assert isinstance(quadratic_objective.model, Model)


def test_add_objective_from_variable() -> None:
    m = Model()
    v = m.add_variables(coords=[[1, 2, 3]])
    m.add_objective(v)
    assert isinstance(m.objective, Objective)


def test_sense(linear_objective: Objective, quadratic_objective: Objective) -> None:
    assert linear_objective.sense == "min"
    assert quadratic_objective.sense == "max"

    assert linear_objective.model.sense == "min"
    assert quadratic_objective.model.sense == "max"


def test_set_sense(linear_objective: Objective, quadratic_objective: Objective) -> None:
    linear_objective.sense = "max"
    quadratic_objective.sense = "min"

    assert linear_objective.sense == "max"
    assert quadratic_objective.sense == "min"

    assert linear_objective.model.sense == "max"
    assert quadratic_objective.model.sense == "min"


def test_set_sense_via_model(
    linear_objective: Objective, quadratic_objective: Objective
) -> None:
    linear_objective.model.sense = "max"
    quadratic_objective.model.sense = "min"

    assert linear_objective.sense == "max"
    assert quadratic_objective.sense == "min"


def test_sense_setter_error(linear_objective: Objective) -> None:
    with pytest.raises(ValueError):
        linear_objective.sense = "not min or max"


def test_variables_inherited_properties(linear_objective: Objective) -> None:
    assert isinstance(linear_objective.attrs, dict)
    assert isinstance(linear_objective.coords, xr.Coordinates)
    assert isinstance(linear_objective.indexes, xarray.core.indexes.Indexes)
    assert isinstance(linear_objective.sizes, xarray.core.utils.Frozen)

    assert isinstance(linear_objective.flat, pd.DataFrame)
    assert isinstance(linear_objective.vars, xr.DataArray)
    assert isinstance(linear_objective.coeffs, xr.DataArray)
    assert isinstance(linear_objective.nterm, int)


def test_expression(
    linear_objective: Objective, quadratic_objective: Objective
) -> None:
    assert isinstance(linear_objective.expression, LinearExpression)
    assert isinstance(quadratic_objective.expression, QuadraticExpression)


def test_value(linear_objective: Objective, quadratic_objective: Objective) -> None:
    assert linear_objective.value is None
    assert quadratic_objective.value is None


def test_set_value(linear_objective: Objective, quadratic_objective: Objective) -> None:
    linear_objective.set_value(1)
    quadratic_objective.set_value(2)
    assert linear_objective.value == 1
    assert quadratic_objective.value == 2


def test_set_value_error(linear_objective: Objective) -> None:
    with pytest.raises(ValueError):
        linear_objective.set_value("not a number")  # type: ignore


def test_assign(linear_objective: Objective) -> None:
    assert isinstance(linear_objective.assign(one=1), Objective)


def test_sel(linear_objective: Objective) -> None:
    assert isinstance(linear_objective.sel(_term=[]), Objective)


def test_is_linear(linear_objective: Objective, quadratic_objective: Objective) -> None:
    assert linear_objective.is_linear is True
    assert quadratic_objective.is_linear is False


def test_is_quadratic(
    linear_objective: Objective, quadratic_objective: Objective
) -> None:
    assert linear_objective.is_quadratic is False
    assert quadratic_objective.is_quadratic is True


def test_to_matrix(linear_objective: Objective, quadratic_objective: Objective) -> None:
    with pytest.raises(ValueError):
        linear_objective.to_matrix()
    assert isinstance(quadratic_objective.to_matrix(), csc_matrix)


def test_add(linear_objective: Objective, quadratic_objective: Objective) -> None:
    obj = linear_objective + quadratic_objective
    assert isinstance(obj, Objective)
    assert isinstance(obj.expression, QuadraticExpression)


def test_add_expr(linear_objective: Objective, quadratic_objective: Objective) -> None:
    obj = linear_objective + quadratic_objective.expression
    assert isinstance(obj, Objective)
    assert isinstance(obj.expression, QuadraticExpression)


def test_sub(linear_objective: Objective, quadratic_objective: Objective) -> None:
    obj = quadratic_objective - linear_objective
    assert isinstance(obj, Objective)
    assert isinstance(obj.expression, QuadraticExpression)


def test_sub_epxr(linear_objective: Objective, quadratic_objective: Objective) -> None:
    obj = quadratic_objective - linear_objective.expression
    assert isinstance(obj, Objective)
    assert isinstance(obj.expression, QuadraticExpression)


def test_mul(quadratic_objective: Objective) -> None:
    obj = quadratic_objective * 2
    assert isinstance(obj, Objective)
    assert isinstance(obj.expression, QuadraticExpression)


def test_neg(quadratic_objective: Objective) -> None:
    obj = -quadratic_objective
    assert isinstance(obj, Objective)
    assert isinstance(obj.expression, QuadraticExpression)


def test_truediv(quadratic_objective: Objective) -> None:
    obj = quadratic_objective / 2
    assert isinstance(obj, Objective)
    assert isinstance(obj.expression, QuadraticExpression)


def test_truediv_false(quadratic_objective: Objective) -> None:
    with pytest.raises(ValueError):
        quadratic_objective / quadratic_objective


def test_repr(linear_objective: Objective, quadratic_objective: Objective) -> None:
    assert isinstance(linear_objective.__repr__(), str)
    assert isinstance(quadratic_objective.__repr__(), str)

    assert "Linear" in linear_objective.__repr__()
    assert "Quadratic" in quadratic_objective.__repr__()


def test_objective_constant() -> None:
    m = Model()
    linear_expr = LinearExpression(None, m) + 1
    with pytest.raises(ValueError):
        m.objective = Objective(linear_expr, m)
