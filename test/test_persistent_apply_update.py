from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from linopy import Model
from linopy.persistent import RebuildReason
from linopy.solvers import Gurobi, Highs, Mosek, Solver, Xpress

_BACKENDS: dict[str, tuple[type[Solver], dict[str, Any]]] = {
    "gurobi": (Gurobi, {"OutputFlag": 0}),
    "highs": (Highs, {"output_flag": False}),
    "xpress": (Xpress, {"OUTPUTLOG": 0}),
    "mosek": (Mosek, {"MSK_IPAR_LOG": 0}),
}

_SIGN_CHANGE_IN_PLACE: dict[str, bool] = {
    "gurobi": True,
    "highs": False,
    "xpress": True,
    "mosek": False,
}


def _have(name: str) -> bool:
    cls = _BACKENDS[name][0]
    if not cls.is_available():
        return False
    try:
        cls._license_probe()
    except Exception:
        return False
    if name == "xpress":
        try:
            import xpress

            xpress.problem()
        except Exception:
            return False
    return True


SOLVER_PARAMS = [
    pytest.param(
        name,
        marks=pytest.mark.skipif(not _have(name), reason=f"{name} not installed"),
    )
    for name in _BACKENDS
]


def _base_model() -> Model:
    m = Model()
    x = m.add_variables(0, 10, coords=[range(3)], name="x")
    y = m.add_variables(0, 10, coords=[range(3)], name="y")
    m.add_constraints(x + y >= 4, name="c1")
    m.add_constraints(2 * x + y <= 20, name="c2")
    m.add_objective(x.sum() + 2 * y.sum())
    return m


def _built(solver_name: str, model: Model) -> Solver:
    cls, opts = _BACKENDS[solver_name]
    s = cls(model=model, io_api="direct", track_updates=True)
    s.options = opts
    s._build()
    return s


def _solve(solver: Solver, model: Model) -> float:
    result = solver.solve(model, assign=True)
    assert result.solution is not None
    return float(result.solution.objective)


def _obj(model: Model) -> float:
    value = model.objective.value
    assert value is not None
    return float(value)


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_var_lb_in_place(solver_name: str) -> None:
    m = _base_model()
    s = _built(solver_name, m)
    s.solve(assign=True)
    base_obj = _obj(m)

    m.variables["x"].lower.values[...] = 5.0
    obj = _solve(s, m)
    assert s._in_place_updates == 1
    assert s._rebuilds == 0
    assert s._last_rebuild_reason is None
    assert obj > base_obj


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_var_ub_in_place(solver_name: str) -> None:
    m = _base_model()
    s = _built(solver_name, m)
    s.solve(assign=True)

    m.variables["x"].upper.values[...] = 1.0
    _solve(s, m)
    assert s._in_place_updates == 1
    assert s._rebuilds == 0


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_rhs_only_in_place(solver_name: str) -> None:
    m = _base_model()
    s = _built(solver_name, m)
    s.solve(assign=True)
    base_obj = _obj(m)

    c = m.constraints["c1"]
    c.rhs = 8.0
    assert c._coef_dirty is False
    obj = _solve(s, m)
    assert s._in_place_updates == 1
    assert s._rebuilds == 0
    assert obj > base_obj


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_constraint_coef_change_in_place(solver_name: str) -> None:
    m = _base_model()
    s = _built(solver_name, m)
    s.solve(assign=True)
    base_obj = _obj(m)

    c = m.constraints["c1"]
    c.coeffs = c.coeffs * 2
    obj = _solve(s, m)
    assert s._in_place_updates == 1
    assert s._rebuilds == 0
    assert not np.isclose(obj, base_obj)


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_objective_linear_change_in_place(solver_name: str) -> None:
    m = _base_model()
    s = _built(solver_name, m)
    s.solve(assign=True)
    base_obj = _obj(m)

    x = m.variables["x"]
    y = m.variables["y"]
    m.objective.expression = 5 * x.sum() + 3 * y.sum()
    obj = _solve(s, m)
    assert s._in_place_updates == 1
    assert s._rebuilds == 0
    assert not np.isclose(obj, base_obj)


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_objective_sense_flip_in_place(solver_name: str) -> None:
    m = _base_model()
    s = _built(solver_name, m)
    s.solve(assign=True)
    min_obj = _obj(m)

    m.objective.sense = "max"
    max_obj = _solve(s, m)
    assert s._in_place_updates == 1
    assert s._rebuilds == 0
    assert max_obj > min_obj


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_sparsity_change_triggers_rebuild(solver_name: str) -> None:
    m = _base_model()
    s = _built(solver_name, m)
    s.solve(assign=True)

    x = m.variables["x"]
    m.add_constraints(x <= 5, name="c3")
    s.solve(m, assign=True)
    assert s._rebuilds == 1
    assert s._last_rebuild_reason in {
        RebuildReason.STRUCTURAL_LABELS,
        RebuildReason.STRUCTURAL_CONTAINERS,
    }


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_cross_model_in_place(solver_name: str) -> None:
    m1 = _base_model()
    s = _built(solver_name, m1)
    s.solve(assign=True)

    m2 = _base_model()
    m2.constraints["c1"].rhs = 8.0

    s.solve(m2, assign=True)
    assert s._in_place_updates == 1
    assert s._rebuilds == 0

    cross_obj = _obj(m2)
    m3 = _base_model()
    m3.constraints["c1"].rhs = 8.0
    s_fresh = _built(solver_name, m3)
    s_fresh.solve(assign=True)
    assert np.isclose(cross_obj, _obj(m3))


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_sign_flip(solver_name: str) -> None:
    m = _base_model()
    s = _built(solver_name, m)
    s.solve(assign=True)

    m.constraints["c1"].sign = "<="
    s.solve(m, assign=True)
    if _SIGN_CHANGE_IN_PLACE[solver_name]:
        assert s._in_place_updates == 1
        assert s._rebuilds == 0
    else:
        assert s._rebuilds == 1
        assert s._last_rebuild_reason is RebuildReason.BACKEND_REJECTED
