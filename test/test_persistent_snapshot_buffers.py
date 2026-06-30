from __future__ import annotations

import numpy as np
import pytest

from linopy import Model
from linopy.persistent import ModelDiff, ModelSnapshot, RebuildReason
from linopy.persistent.snapshot import _extract_con_buffers


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
    b1 = s1.con_buffers["c1"]
    b2 = s2.con_buffers["c1"]
    np.testing.assert_array_equal(b1.indptr, b2.indptr)
    np.testing.assert_array_equal(b1.indices, b2.indices)
    np.testing.assert_array_equal(b1.data, b2.data)


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
    x = baseline_model.variables["x"]
    y = baseline_model.variables["y"]
    baseline_model.constraints["c1"].lhs = 2 * x + 0 * y.sum()
    diff = ModelDiff.from_snapshot(snap, baseline_model)
    assert diff in {
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
    assert isinstance(diff, ModelDiff)
    assert diff.is_empty


def test_con_buffers_dtypes(baseline_model: Model) -> None:
    snap = ModelSnapshot.capture(baseline_model)
    buf = snap.con_buffers["c1"]
    assert buf.rhs.dtype == np.float64
    assert buf.sign.dtype == np.dtype("U1")
    assert buf.data.dtype == np.float64
    assert np.issubdtype(buf.indices.dtype, np.integer)
    assert np.issubdtype(buf.indptr.dtype, np.integer)


def test_masked_rows_excluded_from_active_labels() -> None:
    m = Model()
    x = m.add_variables(0, 10, coords=[range(4)], name="x")
    mask = np.array([True, False, True, True])
    m.add_constraints(2 * x >= 1, mask=mask, name="c1")
    m.add_objective(x.sum())
    snap = ModelSnapshot.capture(m)
    buf = snap.con_buffers["c1"]
    assert buf.active_labels.size == 3
    rebuilt = _extract_con_buffers(m.constraints["c1"], m.variables.label_index)
    np.testing.assert_array_equal(rebuilt.active_labels, buf.active_labels)


def test_csr_capture_deterministic(baseline_model: Model) -> None:
    s1 = ModelSnapshot.capture(baseline_model)
    s2 = ModelSnapshot.capture(baseline_model)
    for name in s1.con_buffers:
        b1, b2 = s1.con_buffers[name], s2.con_buffers[name]
        np.testing.assert_array_equal(b1.indptr, b2.indptr)
        np.testing.assert_array_equal(b1.indices, b2.indices)
        np.testing.assert_array_equal(b1.data, b2.data)


def test_duplicate_variable_terms_summed() -> None:
    m1 = Model()
    x1 = m1.add_variables(0, 10, coords=[range(3)], name="x")
    m1.add_constraints(2 * x1 + 3 * x1 >= 1, name="c1")
    m1.add_objective(x1.sum())

    m2 = Model()
    x2 = m2.add_variables(0, 10, coords=[range(3)], name="x")
    m2.add_constraints(5 * x2 >= 1, name="c1")
    m2.add_objective(x2.sum())

    diff = ModelDiff.from_models(m1, m2)
    assert isinstance(diff, ModelDiff)
    assert diff.is_empty
