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
    VarKind,
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
    assert isinstance(diff, ModelDiff)
    assert diff.is_empty


def test_bounds_only_mutation(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    baseline.variables["x"].lower = 1
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert isinstance(diff, ModelDiff)
    assert "x" in diff.changed_variables
    assert "y" not in diff.changed_variables
    sl = diff.var_slices["x"].bounds
    np.testing.assert_array_equal(diff.var_bounds_lower[sl], np.ones(3))


def test_rhs_only_mutation(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    baseline.constraints["c1"].rhs = 9
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert isinstance(diff, ModelDiff)
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
    assert isinstance(diff, ModelDiff)
    assert diff.obj_c_indices is not None
    assert diff.obj_c_values is not None


def test_objective_sense_flip(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    baseline.objective.sense = "max"
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert isinstance(diff, ModelDiff)
    assert diff.obj_sense == "max"


def test_add_constraints_is_structural(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    x = baseline.variables["x"]
    baseline.add_constraints(x.sum() <= 99, name="c3")
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert diff in (
        RebuildReason.STRUCTURAL_LABELS,
        RebuildReason.STRUCTURAL_CONTAINERS,
    )


def test_remove_variables_is_structural(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    baseline.remove_variables("y")
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert diff in (
        RebuildReason.STRUCTURAL_LABELS,
        RebuildReason.STRUCTURAL_CONTAINERS,
    )


def test_coef_value_change_same_sparsity(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    c = baseline.constraints["c1"]
    c.coeffs = c.coeffs * 3
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert isinstance(diff, ModelDiff)
    assert "c1" in diff.changed_constraints
    sl = diff.con_slices["c1"].coef
    vals = diff.con_coef_vals[sl]
    np.testing.assert_array_equal(vals, np.full(vals.size, 6.0))


def test_coef_changes_across_containers(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    c1 = baseline.constraints["c1"]
    c2 = baseline.constraints["c2"]
    c1.update(coeffs=c1.coeffs * 3)
    c2.update(coeffs=c2.coeffs * 2)
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert isinstance(diff, ModelDiff)
    sl1 = diff.con_slices["c1"].coef
    sl2 = diff.con_slices["c2"].coef
    assert diff.n_coef_updates == (sl1.stop - sl1.start) + (sl2.stop - sl2.start)
    np.testing.assert_array_equal(
        diff.con_coef_vals[sl1], np.full(sl1.stop - sl1.start, 6.0)
    )
    np.testing.assert_array_equal(
        diff.con_coef_vals[sl2], np.full(sl2.stop - sl2.start, 2.0)
    )


def test_coef_sparsity_change(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    x = baseline.variables["x"]
    baseline.constraints["c2"].lhs = 2 * x.sum()
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert diff is RebuildReason.SPARSITY


def test_deep_copy_invariant(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    baseline.variables["x"].lower.values[...] = 99
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert isinstance(diff, ModelDiff)
    assert "x" in diff.changed_variables


def test_same_model_false_ignores_dirty_flag(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    c = baseline.constraints["c1"]
    c.coeffs = c.coeffs * 5
    c._coef_dirty = False
    diff_fast = ModelDiff.from_snapshot(snap, baseline, same_model=True)
    assert isinstance(diff_fast, ModelDiff)
    fast_coef = diff_fast.con_slices.get("c1")
    assert fast_coef is None or fast_coef.coef.stop == fast_coef.coef.start
    diff_full = ModelDiff.from_snapshot(snap, baseline, same_model=False)
    assert isinstance(diff_full, ModelDiff)
    full_coef = diff_full.con_slices["c1"].coef
    assert full_coef.stop > full_coef.start


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
    assert isinstance(diff, ModelDiff)
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

    assert ModelDiff.from_snapshot(snap, m2) is RebuildReason.COORD_REINDEX
    assert isinstance(ModelDiff.from_snapshot(snap, m2, ignore_dims={"t"}), ModelDiff)


def _assert_snapshot_equal(a: ModelSnapshot, b: ModelSnapshot) -> None:
    assert a.structural_key == b.structural_key
    assert a.var_buffers.keys() == b.var_buffers.keys()
    assert a.con_buffers.keys() == b.con_buffers.keys()
    for name, va in a.var_buffers.items():
        vb = b.var_buffers[name]
        np.testing.assert_array_equal(va.lower, vb.lower)
        np.testing.assert_array_equal(va.upper, vb.upper)
        np.testing.assert_array_equal(va.active_labels, vb.active_labels)
        assert va.type is vb.type
    for name, ca in a.con_buffers.items():
        cb = b.con_buffers[name]
        for attr in ("indptr", "indices", "data", "rhs", "sign", "active_labels"):
            np.testing.assert_array_equal(getattr(ca, attr), getattr(cb, attr))
    for coords_a, coords_b in (
        (a.var_coords, b.var_coords),
        (a.con_coords, b.con_coords),
    ):
        assert coords_a.keys() == coords_b.keys()
        for name in coords_a:
            assert coords_a[name].keys() == coords_b[name].keys()
            for dim in coords_a[name]:
                np.testing.assert_array_equal(coords_a[name][dim], coords_b[name][dim])
    np.testing.assert_array_equal(a.obj_c, b.obj_c)
    assert a.obj_quad_present == b.obj_quad_present
    assert a.obj_sense == b.obj_sense


def test_capture_is_pure(baseline: Model) -> None:
    c = baseline.constraints["c1"]
    c.update(coeffs=c.coeffs * 2)
    assert c._coef_dirty is True
    ModelSnapshot.capture(baseline)
    assert c._coef_dirty is True


@pytest.mark.parametrize(
    "mutate", ["none", "rhs", "bounds", "coeffs", "objective", "combined"]
)
def test_diff_snapshot_matches_capture(baseline: Model, mutate: str) -> None:
    snap = ModelSnapshot.capture(baseline)
    x = baseline.variables["x"]
    y = baseline.variables["y"]
    if mutate in ("rhs", "combined"):
        baseline.constraints["c1"].update(rhs=9)
    if mutate in ("bounds", "combined"):
        x.update(lower=1)
    if mutate in ("coeffs", "combined"):
        c2 = baseline.constraints["c2"]
        c2.update(coeffs=c2.coeffs * 3)
    if mutate in ("objective", "combined"):
        baseline.add_objective(3 * x.sum() + 2 * y.sum(), overwrite=True)
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert isinstance(diff, ModelDiff)
    _assert_snapshot_equal(diff.snapshot, ModelSnapshot.capture(baseline))


def test_diff_snapshot_matches_capture_under_ignore_dims() -> None:
    def build(t0: int) -> Model:
        m = Model()
        t = pd.Index(range(t0, t0 + 3), name="t")
        m.add_variables(0, 10, coords=[t], name="x")
        m.add_constraints(m.variables["x"] >= 0, name="c1")
        m.add_objective(m.variables["x"].sum())
        return m

    m1, m2 = build(0), build(10)
    snap = ModelSnapshot.capture(m1)
    diff = ModelDiff.from_snapshot(snap, m2, ignore_dims={"t"})
    assert isinstance(diff, ModelDiff)
    _assert_snapshot_equal(diff.snapshot, ModelSnapshot.capture(m2))


def test_from_models_snapshot_matches_capture() -> None:
    def build(rhs: float) -> Model:
        m = Model()
        x = m.add_variables(0, 10, coords=[range(3)], name="x")
        m.add_constraints(2 * x >= rhs, name="c1")
        m.add_objective(x.sum())
        return m

    m1, m2 = build(4.0), build(7.0)
    diff = ModelDiff.from_models(m1, m2)
    assert isinstance(diff, ModelDiff)
    _assert_snapshot_equal(diff.snapshot, ModelSnapshot.capture(m2))


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"binary": True}, VarKind.BINARY),
        ({"lower": 0, "upper": 10, "integer": True}, VarKind.INTEGER),
        ({"lower": 1, "upper": 10, "semi_continuous": True}, VarKind.SEMI_CONTINUOUS),
    ],
)
def test_variable_kind_captured(kwargs: dict, expected: VarKind) -> None:
    m = Model()
    m.add_variables(coords=[range(2)], name="x", **kwargs)
    m.add_objective(m.variables["x"].sum())
    snap = ModelSnapshot.capture(m)
    assert snap.var_buffers["x"].type is expected


def test_variable_type_change_via_from_models() -> None:
    def build(integer: bool) -> Model:
        m = Model()
        m.add_variables(0, 10, coords=[range(3)], name="x", integer=integer)
        m.add_constraints(m.variables["x"] >= 1, name="c1")
        m.add_objective(m.variables["x"].sum())
        return m

    diff = ModelDiff.from_models(build(False), build(True))
    assert isinstance(diff, ModelDiff)
    sl = diff.var_slices["x"].type
    assert sl.stop > sl.start
    assert diff.var_type_kinds[sl][0] is VarKind.INTEGER


def test_quadratic_objective_triggers_rebuild() -> None:
    m = Model()
    x = m.add_variables(0, 10, coords=[range(3)], name="x")
    m.add_constraints(x >= 1, name="c1")
    m.add_objective((x * x).sum())
    snap = ModelSnapshot.capture(m)
    x.update(lower=2)
    assert ModelDiff.from_snapshot(snap, m) is RebuildReason.QUAD_OBJ


def test_variable_count_change_is_structural() -> None:
    def build(n: int) -> Model:
        m = Model()
        x = m.add_variables(0, 10, coords=[range(n)], name="x")
        m.add_constraints(x >= 1, name="c1")
        m.add_objective(x.sum())
        return m

    assert ModelDiff.from_models(build(3), build(4)) is RebuildReason.STRUCTURAL_LABELS


def test_constraint_count_change_is_structural() -> None:
    def build(aggregate: bool) -> Model:
        m = Model()
        x = m.add_variables(0, 10, coords=[range(3)], name="x")
        m.add_constraints(x.sum() >= 1 if aggregate else x >= 1, name="c1")
        m.add_objective(x.sum())
        return m

    diff = ModelDiff.from_models(build(False), build(True))
    assert diff is RebuildReason.STRUCTURAL_LABELS


def test_indices_change_triggers_sparsity() -> None:
    def build(on: int) -> Model:
        m = Model()
        x = m.add_variables(0, 10, coords=[range(2)], name="x")
        m.add_constraints(x.loc[on] >= 1, name="c1")
        m.add_objective(x.sum())
        return m

    assert ModelDiff.from_models(build(0), build(1)) is RebuildReason.SPARSITY


def test_sign_only_mutation(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    baseline.constraints["c1"].update(sign="<=")
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert isinstance(diff, ModelDiff)
    sl = diff.con_slices["c1"]
    assert sl.sign.stop > sl.sign.start
    assert sl.coef.stop == sl.coef.start


def test_inspect_and_repr(baseline: Model) -> None:
    snap = ModelSnapshot.capture(baseline)
    assert repr(ModelDiff.from_snapshot(snap, baseline)) == "ModelDiff(empty)"

    baseline.variables["x"].update(lower=1)
    c1 = baseline.constraints["c1"]
    c1.update(coeffs=c1.coeffs * 2, rhs=9, sign="<=")
    diff = ModelDiff.from_snapshot(snap, baseline)
    assert isinstance(diff, ModelDiff)

    var_info = diff.inspect_variable("x")
    assert "lower" in var_info and "bounds_indices" in var_info
    con_info = diff.inspect_constraint("c1")
    assert {"coef_vals", "rhs_values", "sign_values"} <= con_info.keys()

    assert diff.inspect_variable("missing") == {}
    assert diff.inspect_constraint("missing") == {}
    assert repr(diff).startswith("ModelDiff(") and "empty" not in repr(diff)
