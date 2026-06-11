from __future__ import annotations

import pickle
from typing import Any

import pytest

from linopy import Model
from linopy.constants import (
    Result,
    Solution,
    SolverStatus,
    Status,
    TerminationCondition,
)
from linopy.persistent import ModelDiff, RebuildReason
from linopy.solvers import Solver, SolverFeature


class FakeSolver(Solver[None]):
    display_name = "Fake"
    features = frozenset({SolverFeature.DIRECT_API})
    accepted_io_apis = frozenset({"direct"})
    supports_persistent_update = False

    @classmethod
    def is_available(cls) -> bool:  # type: ignore[override]
        return True

    @property
    def solver_name(self) -> Any:
        class _N:
            value = "fake"

        return _N()

    def _validate_model(self) -> None:
        return None

    def _build_direct(self, **kwargs: Any) -> None:
        self.solver_model = object()

    def _run_direct(self, **kwargs: Any) -> Result:
        status = Status(SolverStatus.ok, TerminationCondition.optimal)
        return Result(
            status=status, solution=Solution(objective=0.0), solver_name="fake"
        )


@pytest.fixture
def model() -> Model:
    m = Model()
    x = m.add_variables(0, 10, coords=[range(3)], name="x")
    m.add_constraints(2 * x >= 4, name="c1")
    m.add_objective(x.sum())
    return m


@pytest.fixture
def other_model() -> Model:
    m = Model()
    x = m.add_variables(0, 10, coords=[range(3)], name="x")
    m.add_constraints(2 * x >= 4, name="c1")
    m.add_objective(x.sum())
    return m


def _built(model: Model) -> FakeSolver:
    s = FakeSolver(model=model, io_api="direct", track_updates=True)
    s._build()
    return s


def test_unsupported_falls_through_to_rebuild(model: Model, other_model: Model) -> None:
    s = _built(model)
    assert s._rebuilds == 0
    s.solve(other_model)
    assert s._rebuilds == 1
    assert s._last_rebuild_reason is RebuildReason.BACKEND_REJECTED
    assert s.model is other_model


def test_update_apply_false_returns_diff(model: Model) -> None:
    s = _built(model)
    diff = s.update(model, apply=False)
    assert isinstance(diff, ModelDiff)
    assert s._in_place_updates == 0
    assert s._rebuilds == 0


def test_solve_no_model_still_works(model: Model) -> None:
    s = _built(model)
    result = s.solve()
    assert result.status.status is SolverStatus.ok


def test_getstate_drops_native_fields(model: Model) -> None:
    s = _built(model)
    state = s.__getstate__()
    for k in ("solver_model", "env", "_env_stack", "snapshot", "_lock"):
        assert k not in state
    restored = pickle.loads(pickle.dumps(s))
    assert restored.solver_model is None
    assert restored.snapshot is None


def test_update_without_snapshot_raises(model: Model) -> None:
    s = FakeSolver(model=model, io_api="direct")
    with pytest.raises(RuntimeError, match="not been built"):
        s.update(model)


def test_unmutated_resolve_diff_is_empty(model: Model) -> None:
    s = _built(model)
    diff = s.update(model, apply=False)
    assert isinstance(diff, ModelDiff)
    assert diff.is_empty


class FakePersistentSolver(FakeSolver):
    supports_persistent_update = True

    def apply_update(
        self, diff: ModelDiff, var_label_index: Any, con_label_index: Any
    ) -> None:
        return None


def _built_persistent(model: Model) -> FakePersistentSolver:
    s = FakePersistentSolver(model=model, io_api="direct", track_updates=True)
    s._build()
    return s


def test_build_clears_coef_dirty(model: Model) -> None:
    c = model.constraints["c1"]
    c.update(coeffs=c.coeffs * 2)
    assert c._coef_dirty is True
    _built_persistent(model)
    assert c._coef_dirty is False


def test_in_place_update_adopts_diff_snapshot(model: Model) -> None:
    s = _built_persistent(model)
    c = model.constraints["c1"]
    c.update(coeffs=c.coeffs * 2)
    diff = s.update(model)
    assert isinstance(diff, ModelDiff)
    assert s.snapshot is diff.snapshot
    assert c._coef_dirty is False
    rediff = s.update(model, apply=False)
    assert isinstance(rediff, ModelDiff)
    assert rediff.is_empty


def test_update_apply_false_leaves_state_untouched(model: Model) -> None:
    s = _built_persistent(model)
    snap_before = s.snapshot
    c = model.constraints["c1"]
    c.update(coeffs=c.coeffs * 2)
    diff = s.update(model, apply=False)
    assert isinstance(diff, ModelDiff)
    assert c._coef_dirty is True
    assert s.snapshot is snap_before
