from __future__ import annotations

import numpy as np
import pytest

from linopy import Model
from linopy.persistent import (
    CoefPattern,
    ModelDiff,
    ModelSnapshot,
    RebuildReason,
    StructuralKey,
    compute_diff,
)


@pytest.fixture
def baseline() -> Model:
    m = Model()
    x = m.add_variables(0, 10, coords=[range(3)], name="x")
    y = m.add_variables(0, 5, coords=[range(2)], name="y")
    m.add_constraints(2 * x + 1 >= 4, name="c1")
    m.add_constraints(x.sum() + y.sum() <= 20, name="c2")
    m.add_objective(x.sum() + 2 * y.sum())
    return m


def test_capture_structural_key(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    assert isinstance(snap, ModelSnapshot)
    assert isinstance(snap.structural_key, StructuralKey)
    assert snap.structural_key.var_container_names == ("x", "y")
    assert snap.structural_key.con_container_names == ("c1", "c2")
    np.testing.assert_array_equal(
        snap.structural_key.vlabels, baseline.variables.label_index.vlabels
    )
    np.testing.assert_array_equal(
        snap.structural_key.clabels, baseline.constraints.label_index.clabels
    )
    assert isinstance(snap.con_coef_pattern["c1"], CoefPattern)


def test_is_empty_on_unmutated(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    diff = compute_diff(snap, baseline)
    assert diff.is_empty
    assert diff.rebuild_reason is RebuildReason.NONE
    assert not diff.rebuild_required


def test_bounds_only_mutation(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    baseline.variables["x"].lower = 1
    diff = compute_diff(snap, baseline)
    assert diff.rebuild_reason is RebuildReason.NONE
    assert "x" in diff.var_lb
    assert "x" not in diff.var_ub


def test_rhs_only_mutation(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    baseline.constraints["c1"].rhs = 9
    diff = compute_diff(snap, baseline)
    assert diff.rebuild_reason is RebuildReason.NONE
    assert "c1" in diff.con_rhs
    assert not diff.con_coef_updates


def test_objective_linear_change(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    x = baseline.variables["x"]
    y = baseline.variables["y"]
    baseline.add_objective(3 * x.sum() + 2 * y.sum(), overwrite=True)
    diff = compute_diff(snap, baseline)
    assert diff.rebuild_reason is RebuildReason.NONE
    assert diff.obj_linear is not None


def test_objective_sense_flip(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    baseline.objective.sense = "max"
    diff = compute_diff(snap, baseline)
    assert diff.rebuild_reason is RebuildReason.NONE
    assert diff.obj_sense == "max"


def test_add_constraints_is_structural(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    x = baseline.variables["x"]
    baseline.add_constraints(x.sum() <= 99, name="c3")
    diff = compute_diff(snap, baseline)
    assert diff.rebuild_reason in (
        RebuildReason.STRUCTURAL_LABELS,
        RebuildReason.STRUCTURAL_CONTAINERS,
    )


def test_remove_variables_is_structural(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    baseline.remove_variables("y")
    diff = compute_diff(snap, baseline)
    assert diff.rebuild_reason in (
        RebuildReason.STRUCTURAL_LABELS,
        RebuildReason.STRUCTURAL_CONTAINERS,
    )


def test_coef_value_change_same_sparsity(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    c = baseline.constraints["c1"]
    c.coeffs = c.coeffs * 3
    diff = compute_diff(snap, baseline)
    assert diff.rebuild_reason is RebuildReason.NONE
    assert "c1" in diff.con_coef_updates
    values = diff.con_coef_updates["c1"]
    np.testing.assert_array_equal(values, np.full_like(values, 6.0))


def test_coef_sparsity_change(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    x = baseline.variables["x"]
    baseline.constraints["c2"].lhs = 2 * x.sum()
    diff = compute_diff(snap, baseline)
    assert diff.rebuild_reason is RebuildReason.SPARSITY


def test_deep_copy_invariant(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    baseline.variables["x"].lower.values[...] = 99
    diff = compute_diff(snap, baseline)
    assert "x" in diff.var_lb


def test_same_model_false_ignores_dirty_flag(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    c = baseline.constraints["c1"]
    c.coeffs = c.coeffs * 5
    c._coef_dirty = False
    diff_fast = compute_diff(snap, baseline, same_model=True)
    assert "c1" not in diff_fast.con_coef_updates
    diff_full = compute_diff(snap, baseline, same_model=False)
    assert "c1" in diff_full.con_coef_updates


def test_modeldiff_default_is_empty() -> None:
    d = ModelDiff()
    assert d.is_empty
    assert not d.rebuild_required
