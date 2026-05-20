from __future__ import annotations

import numpy as np
import pytest

from linopy import Model
from linopy.persistent import ModelDiff, ModelSnapshot, RebuildReason
from linopy.persistent.snapshot import (
    _canonicalize_rows,
    _extract_con_buffers,
)


def test_canonicalize_rows_sorts_by_var_label() -> None:
    vars_in = np.array([[5, 2, 9], [1, 3, 0]], dtype=np.int64)
    coeffs_in = np.array([[0.5, 0.2, 0.9], [0.1, 0.3, 0.0]], dtype=np.float64)
    vars_out, coeffs_out = _canonicalize_rows(vars_in, coeffs_in)
    np.testing.assert_array_equal(vars_out, [[2, 5, 9], [0, 1, 3]])
    np.testing.assert_array_equal(coeffs_out, [[0.2, 0.5, 0.9], [0.0, 0.1, 0.3]])


def test_canonicalize_rows_minus_one_to_right() -> None:
    vars_in = np.array([[5, -1, 2], [-1, 0, -1]], dtype=np.int64)
    coeffs_in = np.array([[0.5, 0.0, 0.2], [0.0, 0.1, 0.0]], dtype=np.float64)
    vars_out, coeffs_out = _canonicalize_rows(vars_in, coeffs_in)
    np.testing.assert_array_equal(vars_out[:, 0], [2, 0])
    assert (vars_out[:, -1] == -1).all()


def test_canonicalize_empty_buffers_round_trip() -> None:
    vars_in = np.empty((0, 3), dtype=np.int64)
    coeffs_in = np.empty((0, 3), dtype=np.float64)
    vars_out, coeffs_out = _canonicalize_rows(vars_in, coeffs_in)
    assert vars_out.shape == (0, 3)
    assert coeffs_out.shape == (0, 3)


def _build_permuted_pair() -> tuple[Model, Model]:
    m1 = Model()
    x1 = m1.add_variables(0, 10, coords=[range(3)], name="x")
    y1 = m1.add_variables(0, 5, coords=[range(2)], name="y")
    m1.add_constraints(2 * x1 + 3 * y1.sum() >= 4, name="c1")
    m1.add_objective(x1.sum())

    m2 = Model()
    x2 = m2.add_variables(0, 10, coords=[range(3)], name="x")
    y2 = m2.add_variables(0, 5, coords=[range(2)], name="y")
    m2.add_constraints(3 * y2.sum() + 2 * x2 >= 4, name="c1")
    m2.add_objective(x2.sum())
    return m1, m2


def test_permuted_term_order_produces_equal_buffers() -> None:
    m1, m2 = _build_permuted_pair()
    s1 = ModelSnapshot.capture(m1)
    s2 = ModelSnapshot.capture(m2)
    np.testing.assert_array_equal(s1.con_buffers["c1"].vars, s2.con_buffers["c1"].vars)
    np.testing.assert_array_equal(
        s1.con_buffers["c1"].coeffs, s2.con_buffers["c1"].coeffs
    )


def test_active_labels_match_label_index(baseline_model: Model) -> None:
    snap = ModelSnapshot.capture(baseline_model)
    expected = baseline_model.constraints.label_index.clabels
    concatenated = np.concatenate(
        [buf.active_labels for buf in snap.con_buffers.values()]
    )
    np.testing.assert_array_equal(concatenated, expected)


@pytest.fixture
def baseline_model() -> Model:
    m = Model()
    x = m.add_variables(0, 10, coords=[range(3)], name="x")
    y = m.add_variables(0, 5, coords=[range(2)], name="y")
    m.add_constraints(2 * x >= 4, name="c1")
    m.add_constraints(x.sum() + y.sum() <= 20, name="c2")
    m.add_objective(x.sum())
    return m


def test_shape_mismatch_triggers_sparsity_rebuild(baseline_model: Model) -> None:
    snap = ModelSnapshot.capture(baseline_model)
    # Mutate to widen the term dim of c1 via lhs replacement
    x = baseline_model.variables["x"]
    y = baseline_model.variables["y"]
    baseline_model.constraints["c1"].lhs = 2 * x + 0 * y.sum()
    diff = ModelDiff.from_snapshot(snap, baseline_model)
    assert diff.rebuild_reason in {
        RebuildReason.SPARSITY,
        RebuildReason.STRUCTURAL_LABELS,
    }


def test_zero_row_container_capture() -> None:
    m = Model()
    m.add_variables(0, 10, coords=[range(2)], name="x")
    m.add_objective(0.0 * m.variables["x"].sum())
    snap = ModelSnapshot.capture(m)
    assert snap.con_buffers == {}
    diff = ModelDiff.from_snapshot(snap, m)
    assert diff.is_empty


def test_con_buffers_rhs_and_sign_dtypes(baseline_model: Model) -> None:
    snap = ModelSnapshot.capture(baseline_model)
    buf = snap.con_buffers["c1"]
    assert buf.rhs.dtype == np.float64
    assert buf.sign.dtype.kind == "U"
    assert buf.coeffs.dtype == np.float64
    assert buf.vars.dtype == np.int64


def test_masked_rows_excluded_from_active_labels() -> None:
    m = Model()
    x = m.add_variables(0, 10, coords=[range(4)], name="x")
    mask = np.array([True, False, True, True])
    m.add_constraints(2 * x >= 1, mask=mask, name="c1")
    m.add_objective(x.sum())
    snap = ModelSnapshot.capture(m)
    buf = snap.con_buffers["c1"]
    assert buf.active_labels.size == 3
    var_l2p = m.variables.label_index.label_to_pos
    rebuilt = _extract_con_buffers(m.constraints["c1"], var_l2p)
    np.testing.assert_array_equal(rebuilt.active_labels, buf.active_labels)
