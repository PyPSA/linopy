#!/usr/bin/env python3
"""
Test function defined in the Model class.
"""

from __future__ import annotations

import copy as pycopy
import weakref
from collections.abc import Callable
from pathlib import Path
from tempfile import gettempdir

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import EQUAL, Model, available_solvers
from linopy.testing import (
    assert_conequal,
    assert_equal,
    assert_linequal,
    assert_model_equal,
)

target_shape: tuple[int, int] = (10, 10)


def test_model_repr() -> None:
    m: Model = Model()
    m.__repr__()


def test_model_force_dims_names() -> None:
    m: Model = Model(force_dim_names=True)
    with pytest.raises(ValueError):
        m.add_variables([-5], [10])


def test_model_solver_dir() -> None:
    d: str = gettempdir()
    m: Model = Model(solver_dir=d)
    assert m.solver_dir == Path(d)


def test_model_config_defaults() -> None:
    m = Model(freeze_constraints=True, set_names_in_solver_io=False)
    assert m.freeze_constraints is True
    assert m.set_names_in_solver_io is False


def test_model_copy_preserves_config() -> None:
    m = Model(freeze_constraints=True, set_names_in_solver_io=False)
    copied = m.copy()
    assert copied.freeze_constraints is True
    assert copied.set_names_in_solver_io is False


def test_model_is_weakrefable() -> None:
    m: Model = Model()
    ref = weakref.ref(m)
    assert ref() is m


def test_model_weakkeydict_use_case() -> None:
    # third-party extensions rely on WeakKeyDictionary for per-instance storage
    registry: weakref.WeakKeyDictionary[Model, str] = weakref.WeakKeyDictionary()
    m: Model = Model()
    registry[m] = "extension-state"
    assert registry[m] == "extension-state"
    del m
    import gc

    gc.collect()
    assert len(registry) == 0


def test_model_variable_getitem() -> None:
    m = Model()
    x = m.add_variables(name="x")
    assert m["x"].labels == x.labels


def test_coefficient_range() -> None:
    m: Model = Model()

    lower: xr.DataArray = xr.DataArray(
        np.zeros((10, 10)), coords=[range(10), range(10)]
    )
    upper: xr.DataArray = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper)
    y = m.add_variables()

    m.add_constraints(1 * x + 10 * y, EQUAL, 0)
    assert m.coefficientrange["min"].con0 == 1
    assert m.coefficientrange["max"].con0 == 10


def test_objective() -> None:
    m: Model = Model()

    lower: xr.DataArray = xr.DataArray(
        np.zeros((10, 10)), coords=[range(10), range(10)]
    )
    upper: xr.DataArray = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(lower, upper, name="y")

    obj1 = (10 * x + 5 * y).sum()
    m.add_objective(obj1)
    assert m.objective.vars.size == 200

    # test overwriting
    obj2 = (2 * x).sum()
    m.add_objective(obj2, overwrite=True)

    # test Tuple
    obj3 = [(2, x)]
    m.add_objective(obj3, overwrite=True)

    # test objective range
    assert m.objectiverange.min() == 2
    assert m.objectiverange.max() == 2

    # test objective with constant which is not supported
    with pytest.raises(ValueError):
        m.objective = m.objective + 3


def test_solve_without_objective_raises() -> None:
    # https://github.com/PyPSA/linopy/issues/668
    m: Model = Model()
    m.add_variables(lower=0, upper=10, name="myvar")
    with pytest.raises(ValueError, match="No objective has been set"):
        m.solve()


def test_remove_variable() -> None:
    m: Model = Model()

    lower: xr.DataArray = xr.DataArray(
        np.zeros((10, 10)), coords=[range(10), range(10)]
    )
    upper: xr.DataArray = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(name="y")

    m.add_constraints(1 * x + 10 * y, EQUAL, 0)

    obj = (10 * x + 5 * y).sum()
    m.add_objective(obj)

    assert "x" in m.variables

    with pytest.warns(UserWarning, match="con0"):
        m.remove_variables("x")
    assert "x" not in m.variables

    assert "con0" not in m.constraints

    assert not m.objective.vars.isin(x.labels).any()


@pytest.mark.parametrize("freeze", [False, True])
@pytest.mark.parametrize("quadratic", [False, True])
def test_remove_masked_variable_keeps_unrelated_constraints(
    freeze: bool, quadratic: bool
) -> None:
    # https://github.com/PyPSA/linopy/issues/883
    m: Model = Model(freeze_constraints=freeze)

    i = pd.Index(range(3), name="i")
    mask = [True, False, True]
    a = m.add_variables(coords=[i], name="a", mask=mask)
    b = m.add_variables(coords=[i], name="b", mask=mask)
    c = m.add_variables(coords=[i], name="c")

    # `b` is masked, so the constraint carries empty term slots (-1), but it
    # never references `a`
    without_a = m.add_constraints(b.sum() + c, EQUAL, 0, name="without_a")
    assert not without_a.has_variable(a)

    with_a = m.add_constraints(a.sum() + c, EQUAL, 0, name="with_a")
    assert with_a.has_variable(a)

    if quadratic:
        m.add_objective((a * c).sum() + (c * c).sum())
    else:
        m.add_objective((1 * a).sum() + (1 * c).sum())

    with pytest.warns(UserWarning, match="with_a"):
        m.remove_variables("a")

    assert "without_a" in m.constraints
    assert "with_a" not in m.constraints
    assert not m.objective.vars.isin(a.labels[a.labels != -1]).any()
    assert m.objective.vars.isin(c.labels).any()


def test_remove_constraint() -> None:
    m: Model = Model()

    x = m.add_variables()
    m.add_constraints(x, EQUAL, 0, name="x")
    m.remove_constraints("x")
    assert not len(m.constraints.labels)


def test_remove_constraints_with_list() -> None:
    m: Model = Model()

    x = m.add_variables()
    y = m.add_variables()
    m.add_constraints(x, EQUAL, 0, name="constraint_x")
    m.add_constraints(y, EQUAL, 0, name="constraint_y")
    m.remove_constraints(["constraint_x", "constraint_y"])
    assert "constraint_x" not in m.constraints.labels
    assert "constraint_y" not in m.constraints.labels
    assert not len(m.constraints.labels)


def test_remove_objective() -> None:
    m: Model = Model()

    lower: xr.DataArray = xr.DataArray(np.zeros((2, 2)), coords=[range(2), range(2)])
    upper: xr.DataArray = xr.DataArray(np.ones((2, 2)), coords=[range(2), range(2)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(lower, upper, name="y")
    obj = (10 * x + 5 * y).sum()
    m.add_objective(obj)
    m.remove_objective()
    assert not len(m.objective.vars)


def test_assert_model_equal() -> None:
    m: Model = Model()

    lower: xr.DataArray = xr.DataArray(
        np.zeros((10, 10)), coords=[range(10), range(10)]
    )
    upper: xr.DataArray = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(name="y")

    m.add_constraints(1 * x + 10 * y, EQUAL, 0)

    obj = (10 * x + 5 * y).sum()
    m.add_objective(obj)

    assert_model_equal(m, m)


@pytest.fixture(scope="module")
def copy_test_model() -> Model:
    """Small representative model used across copy tests."""
    m: Model = Model()

    lower: xr.DataArray = xr.DataArray(
        np.zeros((10, 10)), coords=[range(10), range(10)]
    )
    upper: xr.DataArray = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(name="y")

    m.add_constraints(1 * x + 10 * y, EQUAL, 0)
    m.add_objective((10 * x + 5 * y).sum())

    return m


@pytest.fixture(scope="module")
def solved_copy_test_model(copy_test_model: Model) -> Model:
    """Solved representative model used across solved-copy tests."""
    m = copy_test_model.copy(deep=True)
    m.solve()
    return m


def test_model_copy_unsolved(copy_test_model: Model) -> None:
    """Copy of unsolved model is structurally equal and independent."""
    m = copy_test_model.copy(deep=True)
    c = m.copy(include_solution=False)

    assert_model_equal(m, c)

    # independence: mutating copy does not affect source
    c.add_variables(name="z")
    assert "z" not in m.variables


def test_model_copy_unsolved_with_solution_flag(copy_test_model: Model) -> None:
    """Unsolved model with include_solution=True has no extra solve artifacts."""
    m = copy_test_model.copy(deep=True)

    c_include_solution = m.copy(include_solution=True)
    c_exclude_solution = m.copy(include_solution=False)

    assert_model_equal(c_include_solution, c_exclude_solution)
    assert c_include_solution.status == "initialized"
    assert c_include_solution.termination_condition == ""
    assert c_include_solution.objective.value is None


def test_model_copy_shallow(copy_test_model: Model) -> None:
    """Shallow copy has independent wrappers sharing underlying data buffers."""
    m = copy_test_model.copy(deep=True)
    c = m.copy(deep=False)

    assert c is not m
    assert c.variables is not m.variables
    assert c.constraints is not m.constraints
    assert c.objective is not m.objective

    # wrappers are distinct, but shallow copy shares payload buffers
    c.variables["x"].lower.values[0, 0] = 123.0
    assert m.variables["x"].lower.values[0, 0] == 123.0


def test_model_deepcopy_protocol(copy_test_model: Model) -> None:
    """copy.deepcopy(model) dispatches to Model.__deepcopy__ and stays independent."""
    m = copy_test_model.copy(deep=True)
    c = pycopy.deepcopy(m)

    assert_model_equal(m, c)

    # Test independence: mutations to copy do not affect source
    # 1. Variable mutation: add new variable
    c.add_variables(name="z")
    assert "z" not in m.variables

    # 2. Variable data mutation (bounds): verify buffers are independent
    original_lower = m.variables["x"].lower.values[0, 0].item()
    new_lower = 999
    c.variables["x"].lower.values[0, 0] = new_lower
    assert c.variables["x"].lower.values[0, 0] == new_lower
    assert m.variables["x"].lower.values[0, 0] == original_lower

    # 3. Constraint coefficient mutation: deep copy must not leak back
    original_con_coeff = m.constraints["con0"].coeffs.values.flat[0].item()
    new_con_coeff = original_con_coeff + 42
    c.constraints["con0"].coeffs.values.flat[0] = new_con_coeff
    assert c.constraints["con0"].coeffs.values.flat[0] == new_con_coeff
    assert m.constraints["con0"].coeffs.values.flat[0] == original_con_coeff

    # 4. Objective expression coefficient mutation: deep copy must not leak back
    original_obj_coeff = m.objective.expression.coeffs.values.flat[0].item()
    new_obj_coeff = original_obj_coeff + 20
    c.objective.expression.coeffs.values.flat[0] = new_obj_coeff
    assert c.objective.expression.coeffs.values.flat[0] == new_obj_coeff
    assert m.objective.expression.coeffs.values.flat[0] == original_obj_coeff

    # 5. Objective sense mutation
    original_sense = m.objective.sense
    c.objective.sense = "max"
    assert c.objective.sense == "max"
    assert m.objective.sense == original_sense


@pytest.fixture(scope="module")
def copy_test_model_with_expressions() -> Model:
    """Representative model with named linear and quadratic expressions."""
    m: Model = Model()

    lower: xr.DataArray = xr.DataArray(
        np.zeros((10, 10)), coords=[range(10), range(10)]
    )
    upper: xr.DataArray = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(name="y")

    m.add_expressions(x + 1, name="lin")
    m.add_expressions(x * y, name="quad")

    m.add_constraints(1 * x + 10 * y, EQUAL, 0)
    m.add_objective((10 * x + 5 * y).sum())

    return m


def test_copy_model_with_expressions(
    copy_test_model_with_expressions: Model,
) -> None:
    """Model.copy(), copy.copy() and copy.deepcopy() all preserve expressions."""
    m = copy_test_model_with_expressions.copy(deep=True)

    for c in (m.copy(), pycopy.copy(m), pycopy.deepcopy(m)):
        assert_model_equal(m, c)
        assert c.expressions["lin"].model is c
        assert c.expressions["quad"].model is c

    deep = pycopy.deepcopy(m)
    original_coeff = m.expressions["lin"].coeffs.values.flat[0].item()
    deep.expressions["lin"].coeffs.values.flat[0] = original_coeff + 42
    assert m.expressions["lin"].coeffs.values.flat[0] == original_coeff


@pytest.mark.parametrize(
    "copy_fn",
    [lambda m: m.copy(), pycopy.copy, pycopy.deepcopy],
    ids=["copy", "shallowcopy", "deepcopy"],
)
@pytest.mark.xfail(
    strict=True, reason="issue #903: copy downgrades a quadratic objective to linear"
)
def test_model_copy_preserves_quadratic_objective(
    copy_fn: Callable[[Model], Model],
) -> None:
    """Copying a model keeps its objective quadratic instead of linearizing it."""
    m: Model = Model()
    x = m.add_variables(lower=-5, upper=5, name="x")
    m.add_objective(x * x + 2 * x)

    c = copy_fn(m)

    assert type(c.objective.expression) is type(m.objective.expression)
    assert c.type == m.type == "QP"
    assert_model_equal(m, c)


@pytest.mark.skipif(not available_solvers, reason="No solver installed")
class TestModelCopySolved:
    def test_model_deepcopy_protocol_excludes_solution(
        self, solved_copy_test_model: Model
    ) -> None:
        """copy.deepcopy on solved model drops solve state by default."""
        m = solved_copy_test_model

        c = pycopy.deepcopy(m)

        assert c.status == "initialized"
        assert c.termination_condition == ""
        assert c.objective.value is None

        for v in m.variables:
            assert_equal(
                c.variables[v].data[c.variables.dataset_attrs],
                m.variables[v].data[m.variables.dataset_attrs],
            )
        for con in m.constraints:
            assert_conequal(c.constraints[con], m.constraints[con], strict=False)
        assert_linequal(c.objective.expression, m.objective.expression)
        assert c.objective.sense == m.objective.sense

    def test_model_copy_solved_with_solution(
        self, solved_copy_test_model: Model
    ) -> None:
        """Copy with include_solution=True preserves solve state."""
        m = solved_copy_test_model

        c = m.copy(include_solution=True)
        assert_model_equal(m, c)

    def test_model_copy_solved_without_solution(
        self, solved_copy_test_model: Model
    ) -> None:
        """Copy with include_solution=False (default) drops solve state but preserves problem structure."""
        m = solved_copy_test_model

        c = m.copy(include_solution=False)

        # solve state is dropped
        assert c.status == "initialized"
        assert c.termination_condition == ""
        assert c.objective.value is None

        # problem structure is preserved — compare only dataset_attrs to exclude solution/dual
        for v in m.variables:
            assert_equal(
                c.variables[v].data[c.variables.dataset_attrs],
                m.variables[v].data[m.variables.dataset_attrs],
            )
        for con in m.constraints:
            assert_conequal(c.constraints[con], m.constraints[con], strict=False)
        assert_linequal(c.objective.expression, m.objective.expression)
        assert c.objective.sense == m.objective.sense
