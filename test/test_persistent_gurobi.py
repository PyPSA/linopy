from __future__ import annotations

import numpy as np
import pytest

from linopy import Model
from linopy.persistent import RebuildReason
from linopy.solvers import Gurobi

pytest.importorskip("gurobipy")


def _base_model() -> Model:
    m = Model()
    x = m.add_variables(0, 10, coords=[range(3)], name="x")
    y = m.add_variables(0, 10, coords=[range(3)], name="y")
    m.add_constraints(x + y >= 4, name="c1")
    m.add_constraints(2 * x + y <= 20, name="c2")
    m.add_objective(x.sum() + 2 * y.sum())
    return m


def _built(model: Model) -> Gurobi:
    s = Gurobi(model=model, io_api="direct", track_updates=True)
    s.options = {"OutputFlag": 0}
    s._build()
    return s


def _solve_and_assign(solver: Gurobi, model: Model) -> float:
    result = solver.solve(model, assign=True)
    return float(result.solution.objective)


def test_var_lb_in_place() -> None:
    m = _base_model()
    s = _built(m)
    s.solve(assign=True)
    assert s._rebuilds == 0
    assert s._in_place_updates == 0
    base_obj = float(m.objective.value)

    m.variables["x"].lower.values[...] = 5.0
    obj = _solve_and_assign(s, m)
    assert s._rebuilds == 0
    assert s._in_place_updates == 1
    assert s._last_rebuild_reason is RebuildReason.NONE
    assert obj > base_obj


def test_var_ub_in_place() -> None:
    m = _base_model()
    s = _built(m)
    s.solve(assign=True)

    m.variables["x"].upper.values[...] = 1.0
    _solve_and_assign(s, m)
    assert s._in_place_updates == 1
    assert s._rebuilds == 0


def test_rhs_only_in_place() -> None:
    m = _base_model()
    s = _built(m)
    s.solve(assign=True)
    base_obj = float(m.objective.value)

    c = m.constraints["c1"]
    c.rhs = 8.0
    assert c._coef_dirty is False
    obj = _solve_and_assign(s, m)
    assert s._in_place_updates == 1
    assert s._rebuilds == 0
    assert obj > base_obj


def test_constraint_coef_change_in_place() -> None:
    m = _base_model()
    s = _built(m)
    s.solve(assign=True)
    base_obj = float(m.objective.value)

    c = m.constraints["c1"]
    new_coeffs = c.coeffs * 2
    c.coeffs = new_coeffs
    obj = _solve_and_assign(s, m)
    assert s._in_place_updates == 1
    assert s._rebuilds == 0
    assert obj != base_obj


def test_objective_linear_change_in_place() -> None:
    m = _base_model()
    s = _built(m)
    s.solve(assign=True)
    base_obj = float(m.objective.value)

    x = m.variables["x"]
    y = m.variables["y"]
    m.objective.expression = 3 * x.sum() + 7 * y.sum()
    obj = _solve_and_assign(s, m)
    assert s._in_place_updates == 1
    assert s._rebuilds == 0
    assert obj != base_obj


def test_objective_sense_flip_in_place() -> None:
    m = _base_model()
    s = _built(m)
    s.solve(assign=True)
    min_obj = float(m.objective.value)

    m.objective.sense = "max"
    max_obj = _solve_and_assign(s, m)
    assert s._in_place_updates == 1
    assert s._rebuilds == 0
    assert max_obj > min_obj


def test_sparsity_change_triggers_rebuild() -> None:
    m = _base_model()
    s = _built(m)
    s.solve(assign=True)

    x = m.variables["x"]
    m.add_constraints(x <= 5, name="c3")
    s.solve(m, assign=True)
    assert s._rebuilds == 1
    assert s._last_rebuild_reason is RebuildReason.STRUCTURAL_CONTAINERS


def test_cross_model_in_place() -> None:
    m1 = _base_model()
    s = _built(m1)
    s.solve(assign=True)

    m2 = _base_model()
    m2.constraints["c1"].rhs = 8.0

    s.solve(m2, assign=True)
    assert s._in_place_updates == 1
    assert s._rebuilds == 0

    fresh_obj = m2.objective.value
    m3 = _base_model()
    m3.constraints["c1"].rhs = 8.0
    s_fresh = _built(m3)
    s_fresh.solve(assign=True)
    assert np.isclose(float(fresh_obj), float(m3.objective.value))
