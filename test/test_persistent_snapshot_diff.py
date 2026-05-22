from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from linopy import Model
from linopy.persistent import (
    ContainerConBuffers,
    ContainerVarBuffers,
    ModelDiff,
    ModelSnapshot,
    RebuildReason,
    StructuralKey,
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
    assert isinstance(snap.var_buffers["x"], ContainerVarBuffers)
    assert isinstance(snap.con_buffers["c1"], ContainerConBuffers)


def test_is_empty_on_unmutated(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert diff.is_empty
    assert diff.rebuild_reason is RebuildReason.NONE
    assert not diff.rebuild_required


def test_bounds_only_mutation(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    baseline.variables["x"].lower = 1
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert diff.rebuild_reason is RebuildReason.NONE
    assert "x" in diff.changed_variables
    assert "y" not in diff.changed_variables
    sl = diff.var_slices["x"].bounds
    np.testing.assert_array_equal(diff.var_bounds_lower[sl], np.ones(3))


def test_rhs_only_mutation(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    baseline.constraints["c1"].rhs = 9
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert diff.rebuild_reason is RebuildReason.NONE
    assert "c1" in diff.changed_constraints
    sl = diff.con_slices["c1"]
    assert sl.rhs.stop > sl.rhs.start
    assert sl.coef.stop == sl.coef.start


def test_objective_linear_change(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    x = baseline.variables["x"]
    y = baseline.variables["y"]
    baseline.add_objective(3 * x.sum() + 2 * y.sum(), overwrite=True)
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert diff.rebuild_reason is RebuildReason.NONE
    assert diff.obj_c_indices is not None
    assert diff.obj_c_values is not None


def test_objective_sense_flip(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    baseline.objective.sense = "max"
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert diff.rebuild_reason is RebuildReason.NONE
    assert diff.obj_sense == "max"


def test_add_constraints_is_structural(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    x = baseline.variables["x"]
    baseline.add_constraints(x.sum() <= 99, name="c3")
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert diff.rebuild_reason in (
        RebuildReason.STRUCTURAL_LABELS,
        RebuildReason.STRUCTURAL_CONTAINERS,
    )


def test_remove_variables_is_structural(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    baseline.remove_variables("y")
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert diff.rebuild_reason in (
        RebuildReason.STRUCTURAL_LABELS,
        RebuildReason.STRUCTURAL_CONTAINERS,
    )


def test_coef_value_change_same_sparsity(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    c = baseline.constraints["c1"]
    c.coeffs = c.coeffs * 3
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert diff.rebuild_reason is RebuildReason.NONE
    assert "c1" in diff.changed_constraints
    sl = diff.con_slices["c1"].coef
    vals = diff.con_coef_vals[sl]
    np.testing.assert_array_equal(vals, np.full(vals.size, 6.0))


def test_coef_sparsity_change(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    x = baseline.variables["x"]
    baseline.constraints["c2"].lhs = 2 * x.sum()
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert diff.rebuild_reason is RebuildReason.SPARSITY


def test_deep_copy_invariant(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    baseline.variables["x"].lower.values[...] = 99
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert "x" in diff.changed_variables


def test_same_model_false_ignores_dirty_flag(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    c = baseline.constraints["c1"]
    c.coeffs = c.coeffs * 5
    c._coef_dirty = False
    diff_fast = ModelDiff.from_snapshot(snap, baseline, same_model=True)
    fast_coef = diff_fast.con_slices.get("c1")
    assert fast_coef is None or fast_coef.coef.stop == fast_coef.coef.start
    diff_full = ModelDiff.from_snapshot(snap, baseline, same_model=False)
    full_coef = diff_full.con_slices["c1"].coef
    assert full_coef.stop > full_coef.start


def test_modeldiff_default_is_empty() -> None:
    d = ModelDiff()
    assert d.is_empty
    assert not d.rebuild_required


def test_from_models_diffs_two_models() -> None:
    m1 = Model()
    x1 = m1.add_variables(0, 10, coords=[range(3)], name="x")
    m1.add_constraints(2 * x1 >= 4, name="c1")
    m1.add_objective(x1.sum())

    m2 = Model()
    x2 = m2.add_variables(0, 10, coords=[range(3)], name="x")
    m2.add_constraints(2 * x2 >= 7, name="c1")
    m2.add_objective(x2.sum())

    diff = ModelDiff.from_models(m1, m2)
    assert diff.rebuild_reason is RebuildReason.NONE
    assert "c1" in diff.changed_constraints
    sl = diff.con_slices["c1"].rhs
    np.testing.assert_array_equal(diff.con_rhs_values[sl], np.full(3, 7.0))


def test_ignore_dims_detects_coord_change() -> None:
    m1 = Model()
    m1.add_variables(0, 10, coords=[pd.Index([0, 1, 2], name="t")], name="x")
    m1.add_constraints(m1.variables["x"] >= 0, name="c1")
    m1.add_objective(m1.variables["x"].sum())
    snap = ModelSnapshot.capture(m1)

    m2 = Model()
    m2.add_variables(0, 10, coords=[pd.Index([10, 11, 12], name="t")], name="x")
    m2.add_constraints(m2.variables["x"] >= 0, name="c1")
    m2.add_objective(m2.variables["x"].sum())

    assert ModelDiff.from_snapshot(snap, m2).rebuild_reason is (
        RebuildReason.COORD_REINDEX
    )
    assert ModelDiff.from_snapshot(snap, m2, ignore_dims={"t"}).rebuild_reason is (
        RebuildReason.NONE
    )
