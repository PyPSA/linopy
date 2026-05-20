from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from collections.abc import Iterable

from linopy.persistent.snapshot import (
    ModelSnapshot,
    _coord_snapshot,
    _extract_con_buffers,
    _extract_var_buffers,
    _objective_linear_vector,
)

if TYPE_CHECKING:
    from linopy.model import Model


class RebuildReason(enum.Enum):
    NONE = "none"
    STRUCTURAL_LABELS = "vlabels/clabels mismatch"
    STRUCTURAL_CONTAINERS = "container set changed"
    COORD_REINDEX = "coordinates changed"
    SPARSITY = "coefficient sparsity changed"
    QUAD_OBJ = "quadratic objective changed"
    BACKEND_REJECTED = "backend raised UnsupportedUpdate"


@dataclass
class ContainerVarUpdate:
    """
    In-place variable bounds / type update for one container.

    Bounds payloads share ``bounds_indices``. When only ``lower`` (or only
    ``upper``) changes, both arrays are still populated from the new model so
    backends with a single batched call (HiGHS ``changeColsBounds``) can be
    fed directly.
    """

    bounds_indices: np.ndarray | None = None
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    type_change: str | None = None


@dataclass
class ContainerRowUpdate:
    """
    Per-row constraint update.

    Holds views into the new model's canonicalised buffers; the orchestrator
    diffs and applies under the same lock, so aliasing is bounded.
    """

    coef_row_indices: np.ndarray | None = None
    coef_vars: np.ndarray | None = None
    coef_values: np.ndarray | None = None
    rhs_row_indices: np.ndarray | None = None
    rhs_values: np.ndarray | None = None
    rhs_signs: np.ndarray | None = None
    sign_row_indices: np.ndarray | None = None
    sign_values: np.ndarray | None = None


@dataclass
class ModelDiff:
    rebuild_reason: RebuildReason = RebuildReason.NONE
    vars: dict[str, ContainerVarUpdate] = field(default_factory=dict)
    cons: dict[str, ContainerRowUpdate] = field(default_factory=dict)
    obj_c_indices: np.ndarray | None = None
    obj_c_values: np.ndarray | None = None
    obj_sense: str | None = None

    @property
    def is_empty(self) -> bool:
        return (
            self.rebuild_reason is RebuildReason.NONE
            and not self.vars
            and not self.cons
            and self.obj_c_indices is None
            and self.obj_sense is None
        )

    @property
    def rebuild_required(self) -> bool:
        return self.rebuild_reason is not RebuildReason.NONE

    @property
    def changed_variables(self) -> set[str]:
        return set(self.vars)

    @property
    def changed_constraints(self) -> set[str]:
        return set(self.cons)

    @property
    def n_coef_updates(self) -> int:
        total = 0
        for upd in self.cons.values():
            if upd.coef_vars is not None:
                total += int((upd.coef_vars != -1).sum())
        return total

    def summary(self) -> dict[str, int | bool | str | None]:
        n_var_lb = sum(1 for u in self.vars.values() if u.lower is not None)
        n_var_ub = sum(1 for u in self.vars.values() if u.upper is not None)
        n_var_type = sum(1 for u in self.vars.values() if u.type_change is not None)
        n_con_rhs = sum(1 for u in self.cons.values() if u.rhs_values is not None)
        n_con_sign = sum(1 for u in self.cons.values() if u.sign_values is not None)
        n_con_coef = sum(1 for u in self.cons.values() if u.coef_values is not None)
        return {
            "rebuild_reason": self.rebuild_reason.value,
            "var_lb": n_var_lb,
            "var_ub": n_var_ub,
            "var_type": n_var_type,
            "con_rhs": n_con_rhs,
            "con_sign": n_con_sign,
            "con_coef_updates": n_con_coef,
            "n_coef_values": self.n_coef_updates,
            "obj_linear_changed": self.obj_c_indices is not None,
            "obj_sense_changed_to": self.obj_sense,
        }

    def inspect_variable(self, name: str) -> dict[str, object]:
        if name not in self.vars:
            return {}
        u = self.vars[name]
        entry: dict[str, object] = {}
        if u.lower is not None:
            entry["lower"] = u.lower
        if u.upper is not None:
            entry["upper"] = u.upper
        if u.type_change is not None:
            entry["type"] = u.type_change
        return entry

    def inspect_constraint(self, name: str) -> dict[str, object]:
        if name not in self.cons:
            return {}
        u = self.cons[name]
        entry: dict[str, object] = {}
        if u.rhs_values is not None:
            entry["rhs"] = u.rhs_values
        if u.sign_values is not None:
            entry["sign"] = u.sign_values
        if u.coef_values is not None:
            entry["coef_values"] = u.coef_values
        return entry

    def __repr__(self) -> str:
        if self.is_empty:
            return "ModelDiff(empty)"
        if self.rebuild_required:
            return f"ModelDiff(rebuild_required={self.rebuild_reason.value!r})"
        s = self.summary()
        parts = [
            f"{k}={v}"
            for k, v in s.items()
            if k != "rebuild_reason" and v not in (0, False, None)
        ]
        return "ModelDiff(" + ", ".join(parts) + ")"


def _coords_equal(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray], ignored: frozenset[str]
) -> bool:
    keys_a = set(a) - ignored
    keys_b = set(b) - ignored
    if keys_a != keys_b:
        return False
    return all(np.array_equal(a[k], b[k]) for k in keys_a)


def compute_diff(
    snapshot: ModelSnapshot,
    model: Model,
    same_model: bool = True,
    ignore_dims: Iterable[str] | None = None,
) -> ModelDiff:
    """Compute a ``ModelDiff`` between ``snapshot`` and ``model``.

    Coordinate values are not compared by default. Pass ``ignore_dims``
    (e.g. ``ignore_dims=()`` or ``ignore_dims={"snapshot"}``) to opt into
    per-container coord-equality on every dim *not* in the set — a mismatch
    triggers ``RebuildReason.COORD_REINDEX``.
    """
    check_coords = ignore_dims is not None
    ignored = frozenset(ignore_dims) if ignore_dims is not None else frozenset()
    diff = ModelDiff()

    var_names = tuple(model.variables)
    con_names = tuple(model.constraints)
    if (
        snapshot.structural_key.var_container_names != var_names
        or snapshot.structural_key.con_container_names != con_names
    ):
        diff.rebuild_reason = RebuildReason.STRUCTURAL_CONTAINERS
        return diff

    var_label_index = model.variables.label_index
    con_label_index = model.constraints.label_index
    if not np.array_equal(snapshot.structural_key.vlabels, var_label_index.vlabels):
        diff.rebuild_reason = RebuildReason.STRUCTURAL_LABELS
        return diff
    if not np.array_equal(snapshot.structural_key.clabels, con_label_index.clabels):
        diff.rebuild_reason = RebuildReason.STRUCTURAL_LABELS
        return diff

    var_l2p = var_label_index.label_to_pos
    con_l2p = con_label_index.label_to_pos

    for name, var in model.variables.items():
        snap_buf = snapshot.var_buffers[name]
        new_buf = _extract_var_buffers(var)
        if new_buf.lower.shape != snap_buf.lower.shape:
            diff.rebuild_reason = RebuildReason.COORD_REINDEX
            return diff
        if not np.array_equal(new_buf.active_labels, snap_buf.active_labels):
            diff.rebuild_reason = RebuildReason.STRUCTURAL_LABELS
            return diff
        if check_coords and not _coords_equal(
            snapshot.var_coords[name], _coord_snapshot(var), ignored
        ):
            diff.rebuild_reason = RebuildReason.COORD_REINDEX
            return diff

        lower_diff = new_buf.lower != snap_buf.lower
        upper_diff = new_buf.upper != snap_buf.upper
        type_changed = new_buf.type != snap_buf.type

        bound_mask = lower_diff | upper_diff
        if not (bound_mask.any() or type_changed):
            continue

        update = ContainerVarUpdate(type_change=new_buf.type if type_changed else None)
        if bound_mask.any():
            local_idx = np.flatnonzero(bound_mask)
            update.bounds_indices = var_l2p[
                new_buf.active_labels[local_idx]
            ].astype(np.int32, copy=False)
            update.lower = new_buf.lower[local_idx]
            update.upper = new_buf.upper[local_idx]
        diff.vars[name] = update

    for name, con in model.constraints.items():
        snap_buf = snapshot.con_buffers[name]
        new_buf = _extract_con_buffers(con, var_l2p)

        if new_buf.coeffs.shape != snap_buf.coeffs.shape:
            diff.rebuild_reason = RebuildReason.SPARSITY
            return diff
        if not np.array_equal(new_buf.active_labels, snap_buf.active_labels):
            diff.rebuild_reason = RebuildReason.STRUCTURAL_LABELS
            return diff
        if check_coords and not _coords_equal(
            snapshot.con_coords[name], _coord_snapshot(con), ignored
        ):
            diff.rebuild_reason = RebuildReason.COORD_REINDEX
            return diff

        n_rows = new_buf.active_labels.size
        if n_rows == 0:
            continue

        skip_coef_compare = same_model and not con._coef_dirty
        if skip_coef_compare:
            row_value_changed = np.zeros(n_rows, dtype=bool)
            row_struct_changed = np.zeros(n_rows, dtype=bool)
        else:
            row_struct_changed = np.any(new_buf.vars != snap_buf.vars, axis=-1)
            row_value_changed = np.any(new_buf.coeffs != snap_buf.coeffs, axis=-1)

        if row_struct_changed.any():
            diff.rebuild_reason = RebuildReason.SPARSITY
            return diff

        rhs_changed = new_buf.rhs != snap_buf.rhs
        sign_changed = new_buf.sign != snap_buf.sign

        if not (row_value_changed.any() or rhs_changed.any() or sign_changed.any()):
            continue

        update = ContainerRowUpdate()
        if row_value_changed.any():
            idx = np.flatnonzero(row_value_changed)
            update.coef_row_indices = con_l2p[
                new_buf.active_labels[idx]
            ].astype(np.int32, copy=False)
            update.coef_vars = new_buf.vars[idx]
            update.coef_values = new_buf.coeffs[idx]
        if rhs_changed.any():
            idx = np.flatnonzero(rhs_changed)
            update.rhs_row_indices = con_l2p[
                new_buf.active_labels[idx]
            ].astype(np.int32, copy=False)
            update.rhs_values = new_buf.rhs[idx]
            update.rhs_signs = new_buf.sign[idx]
        if sign_changed.any():
            idx = np.flatnonzero(sign_changed)
            update.sign_row_indices = con_l2p[
                new_buf.active_labels[idx]
            ].astype(np.int32, copy=False)
            update.sign_values = new_buf.sign[idx]
        diff.cons[name] = update

    obj_quad_present = model.objective.is_quadratic
    if obj_quad_present != snapshot.obj_quad_present:
        diff.rebuild_reason = RebuildReason.QUAD_OBJ
        return diff
    if obj_quad_present:
        diff.rebuild_reason = RebuildReason.QUAD_OBJ
        return diff

    obj_c = _objective_linear_vector(model)
    if obj_c.shape != snapshot.obj_c.shape:
        diff.rebuild_reason = RebuildReason.COORD_REINDEX
        return diff
    obj_diff_mask = obj_c != snapshot.obj_c
    if obj_diff_mask.any():
        idx = np.flatnonzero(obj_diff_mask).astype(np.int32, copy=False)
        diff.obj_c_indices = idx
        diff.obj_c_values = obj_c[idx]

    if model.objective.sense != snapshot.obj_sense:
        diff.obj_sense = model.objective.sense

    return diff
