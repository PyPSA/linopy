from __future__ import annotations

import pickle
import threading
from typing import Any

import numpy as np
import pytest

from linopy import Model
from linopy.persistent import ModelDiff, RebuildReason, UpdatesDisabledError
from linopy.solvers import Gurobi, Highs, Solver

_BACKENDS: dict[str, tuple[type[Solver], dict[str, Any]]] = {
    "gurobi": (Gurobi, {"OutputFlag": 0}),
    "highs": (Highs, {"output_flag": False}),
}


def _have(name: str) -> bool:
    try:
        if name == "gurobi":
            import gurobipy  # noqa: F401
        elif name == "highs":
            import highspy  # noqa: F401
        return True
    except ImportError:
        return False


SOLVER_PARAMS = [
    pytest.param(
        "gurobi",
        marks=pytest.mark.skipif(not _have("gurobi"), reason="gurobipy not installed"),
    ),
    pytest.param(
        "highs",
        marks=pytest.mark.skipif(not _have("highs"), reason="highspy not installed"),
    ),
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


def _obj(model: Model) -> float:
    value = model.objective.value
    assert value is not None
    return float(value)


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_noop_resolve_increments_in_place(solver_name: str) -> None:
    m = _base_model()
    s = _built(solver_name, m)
    s.solve(assign=True)
    first_obj = _obj(m)

    s.solve(m, assign=True)
    assert s._in_place_updates == 1
    assert s._rebuilds == 0
    assert np.isclose(_obj(m), first_obj)


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_two_consecutive_solves_no_stale_state(solver_name: str) -> None:
    m = _base_model()
    s = _built(solver_name, m)
    s.solve(assign=True)
    first_status = s.status

    m.variables["x"].lower.values[...] = 5.0
    s.solve(m, assign=True)
    assert s.status is not first_status
    assert s.solution is not None
    assert np.isclose(float(s.solution.objective), _obj(m))


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_cross_model_scenario_sweep(solver_name: str) -> None:
    m1 = _base_model()
    m2 = _base_model()
    m2.constraints["c1"].rhs = 6.0
    m3 = _base_model()
    m3.variables["x"].lower.values[...] = 2.0

    s = _built(solver_name, m1)
    s.solve(assign=True)
    obj1 = _obj(m1)
    sol1 = m1.solution

    s.solve(m2, assign=True)
    s.solve(m3, assign=True)

    assert s._rebuilds == 0
    assert s._in_place_updates >= 2

    assert m1.objective._value == obj1
    np.testing.assert_array_equal(m1.solution.x.values, sol1.x.values)
    assert m2.objective._value is not None
    assert m3.objective._value is not None

    for mk in (m2, m3):
        fresh = _base_model()
        if mk is m2:
            fresh.constraints["c1"].rhs = 6.0
        else:
            fresh.variables["x"].lower.values[...] = 2.0
        s_fresh = _built(solver_name, fresh)
        s_fresh.solve(assign=True)
        assert np.isclose(_obj(mk), _obj(fresh))
        s_fresh.close()


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_cross_model_sparsity_change_rebuilds(solver_name: str) -> None:
    def build(include_y_in_c1: bool) -> Model:
        m = Model()
        x = m.add_variables(0, 10, coords=[range(3)], name="x")
        y = m.add_variables(0, 10, coords=[range(3)], name="y")
        if include_y_in_c1:
            m.add_constraints(x + y >= 4, name="c1")
        else:
            m.add_constraints(2 * x >= 4, name="c1")
        m.add_constraints(2 * x + y <= 20, name="c2")
        m.add_objective(x.sum() + 2 * y.sum())
        return m

    m1 = build(include_y_in_c1=True)
    s = _built(solver_name, m1)
    s.solve(assign=True)

    m2 = build(include_y_in_c1=False)

    s.solve(m2, assign=True)
    assert s._rebuilds == 1
    assert s._last_rebuild_reason in {
        RebuildReason.SPARSITY,
        RebuildReason.STRUCTURAL_LABELS,
        RebuildReason.STRUCTURAL_CONTAINERS,
    }


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_cross_model_structural_mismatch_rebuilds(solver_name: str) -> None:
    m1 = _base_model()
    s = _built(solver_name, m1)
    s.solve(assign=True)

    m2 = _base_model()
    m2.add_variables(0, 5, coords=[range(3)], name="z")

    s.solve(m2, assign=True)
    assert s._rebuilds == 1
    assert s.model is m2


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_dirty_flag_ignored_across_models(solver_name: str) -> None:
    m1 = _base_model()
    s = _built(solver_name, m1)
    s.solve(assign=True)

    m2 = _base_model()
    c = m2.constraints["c1"]
    c.coeffs = c.coeffs * 3
    c._coef_dirty = False

    s.solve(m2, assign=True)
    assert s._rebuilds == 0
    assert s._in_place_updates == 1

    fresh = _base_model()
    cf = fresh.constraints["c1"]
    cf.coeffs = cf.coeffs * 3
    s_fresh = _built(solver_name, fresh)
    s_fresh.solve(assign=True)
    assert np.isclose(_obj(m2), _obj(fresh))
    s_fresh.close()


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_solver_pickle_round_trip_drops_native(solver_name: str) -> None:
    m = _base_model()
    s = _built(solver_name, m)
    s.solve(assign=True)

    state = s.__getstate__()
    for key in ("solver_model", "env", "_env_stack", "snapshot", "_lock"):
        assert key not in state

    restored = pickle.loads(pickle.dumps(s))
    assert restored.solver_model is None
    assert restored.snapshot is None
    assert restored._env_stack is None
    assert isinstance(restored._lock, type(threading.Lock()))


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_model_pickle_round_trip_no_native_handle(solver_name: str) -> None:
    m = _base_model()
    s = _built(solver_name, m)
    s.solve(assign=True)

    m2 = pickle.loads(pickle.dumps(m))
    s2 = _built(solver_name, m2)
    assert s2.solver_model is not None
    s2.solve(assign=True)
    assert s2._rebuilds == 0
    assert np.isclose(_obj(m), _obj(m2))
    s2.close()


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_backend_exception_during_apply_rebuilds(
    solver_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = _base_model()
    s = _built(solver_name, m)
    s.solve(assign=True)

    c = m.constraints["c1"]
    c.coeffs = c.coeffs * 2
    assert c._coef_dirty is True

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("simulated backend failure")

    monkeypatch.setattr(s, "apply_update", _boom)

    dirty_at_rebuild: list[bool] = []
    original_build = s._build

    def _spy_build(**kwargs: Any) -> None:
        dirty_at_rebuild.append(m.constraints["c1"]._coef_dirty)
        original_build(**kwargs)

    monkeypatch.setattr(s, "_build", _spy_build)

    s.solve(m, assign=True)
    assert s._rebuilds == 1
    assert s._last_rebuild_reason is RebuildReason.BACKEND_REJECTED
    assert dirty_at_rebuild == [True]


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_concurrent_solves_serialize(solver_name: str) -> None:
    m = _base_model()
    s = _built(solver_name, m)
    s.solve(assign=True)
    expected = _obj(m)

    barrier = threading.Barrier(2)
    results: list[float] = []
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            barrier.wait()
            res = s.solve(m, assign=True)
            assert res.solution is not None
            results.append(float(res.solution.objective))
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=_run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(results) == 2
    for r in results:
        assert np.isclose(r, expected)


_SCENARIO_PARAMS = [
    "bound_only",
    "rhs_only",
    "single_cell_coef",
    "multi_row_coef",
    "mixed",
]


def _apply_scenario(model: Model, scenario: str) -> None:
    if scenario == "bound_only":
        model.variables["x"].lower.values[...] = 3.0
    elif scenario == "rhs_only":
        model.constraints["c1"].rhs = 7.0
    elif scenario == "single_cell_coef":
        c = model.constraints["c1"]
        new = c.coeffs.copy()
        new.values[0, 0] = 5.0
        c.coeffs = new
    elif scenario == "multi_row_coef":
        c = model.constraints["c2"]
        c.coeffs = c.coeffs * 2
    elif scenario == "mixed":
        model.variables["x"].lower.values[...] = 1.0
        model.constraints["c1"].rhs = 6.0
        c = model.constraints["c2"]
        new = c.coeffs.copy()
        new.values[0, 0] = 4.0
        c.coeffs = new
    else:
        raise ValueError(scenario)


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
@pytest.mark.parametrize("scenario", _SCENARIO_PARAMS)
@pytest.mark.parametrize("same_model", [True, False])
def test_scenario_sweep_in_place(
    solver_name: str, scenario: str, same_model: bool
) -> None:
    m = _base_model()
    s = _built(solver_name, m)
    s.solve(assign=True)

    target = m if same_model else _base_model()
    _apply_scenario(target, scenario)
    s.solve(target, assign=True)

    assert s._rebuilds == 0
    assert s._in_place_updates == 1
    assert s._last_rebuild_reason is None

    fresh = _base_model()
    _apply_scenario(fresh, scenario)
    s_fresh = _built(solver_name, fresh)
    s_fresh.solve(assign=True)
    assert np.isclose(_obj(target), _obj(fresh))
    s_fresh.close()


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_disallow_rebuild_raises_on_structural_change(solver_name: str) -> None:
    from linopy.persistent import RebuildRequiredError

    m = _base_model()
    s = _built(solver_name, m)
    s.solve(assign=True)

    m2 = _base_model()
    m2.add_variables(0, 5, coords=[range(3)], name="z")

    with pytest.raises(RebuildRequiredError):
        s.solve(m2, disallow_rebuild=True, assign=True)
    assert s._rebuilds == 0


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_disallow_rebuild_passes_when_update_works(solver_name: str) -> None:
    m = _base_model()
    s = _built(solver_name, m)
    s.solve(assign=True)

    m.constraints["c1"].rhs = 6.0
    s.solve(m, disallow_rebuild=True, assign=True)
    assert s._in_place_updates == 1
    assert s._rebuilds == 0


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_solve_without_assign_does_not_mutate_model(solver_name: str) -> None:
    m = _base_model()
    s = _built(solver_name, m)

    assert m.objective._value is None
    s.solve()
    assert m.objective._value is None

    s.solve(assign=True)
    assert m.objective._value is not None


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_track_updates_false_skips_snapshot(solver_name: str) -> None:
    cls, opts = _BACKENDS[solver_name]
    m = _base_model()
    s = cls(model=m, io_api="direct", track_updates=False)
    s.options = opts
    s._build()
    assert s.snapshot is None
    s.solve(assign=True)
    assert s.snapshot is None


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_track_updates_false_rejects_resolve_with_model(solver_name: str) -> None:
    cls, opts = _BACKENDS[solver_name]
    m = _base_model()
    s = cls(model=m, io_api="direct", track_updates=False)
    s.options = opts
    s._build()
    s.solve(assign=True)

    m.variables["x"].lower.values[...] = 6.0
    with pytest.raises(UpdatesDisabledError, match="track_updates=False"):
        s.solve(m, assign=True)


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_track_updates_false_rejects_update(solver_name: str) -> None:
    cls, opts = _BACKENDS[solver_name]
    m = _base_model()
    s = cls(model=m, io_api="direct", track_updates=False)
    s.options = opts
    s._build()
    with pytest.raises(UpdatesDisabledError, match="track_updates=False"):
        s.update(m)


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_track_updates_false_cross_instance_resolve(solver_name: str) -> None:
    cls, opts = _BACKENDS[solver_name]
    m1 = _base_model()
    s = cls(model=m1, io_api="direct", track_updates=False)
    s.options = opts
    s._build()
    s.solve(assign=True)
    base_obj = _obj(m1)

    m2 = _base_model()
    m2.constraints["c1"].rhs = 8.0
    result = s.solve(m2, assign=True)
    assert s._in_place_updates == 1
    assert s._rebuilds == 0
    assert s.snapshot is None
    assert s.model is m2
    assert result.solution is not None
    assert float(result.solution.objective) > base_obj


@pytest.mark.parametrize("solver_name", SOLVER_PARAMS)
def test_track_updates_false_cross_instance_update(solver_name: str) -> None:
    cls, opts = _BACKENDS[solver_name]
    m1 = _base_model()
    s = cls(model=m1, io_api="direct", track_updates=False)
    s.options = opts
    s._build()
    s.solve(assign=True)

    m2 = _base_model()
    m2.constraints["c1"].rhs = 8.0
    diff = s.update(m2, apply=False)
    assert isinstance(diff, ModelDiff)
    assert diff.summary()["con_rhs"] == 3
    assert "c1" in diff.changed_constraints
    assert s.snapshot is None
