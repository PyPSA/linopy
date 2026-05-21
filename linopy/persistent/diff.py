from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from linopy.constants import short_GREATER_EQUAL, short_LESS_EQUAL
from linopy.persistent.snapshot import (
    ContainerConBuffers,
    ContainerVarBuffers,
    ModelSnapshot,
    VarKind,
    _coord_snapshot,
    _extract_con_buffers,
    _extract_var_buffers,
    _objective_linear_vector,
)

if TYPE_CHECKING:
    from linopy.constraints import ConstraintBase
    from linopy.model import Model
    from linopy.variables import Variable, VariableLabelIndex


_EMPTY_I32 = np.empty(0, dtype=np.int32)
_EMPTY_F64 = np.empty(0, dtype=np.float64)
_EMPTY_U1 = np.empty(0, dtype="U1")
_EMPTY_KIND: np.ndarray = np.empty(0, dtype=object)


class RebuildReason(enum.Enum):
    NONE = "none"
    STRUCTURAL_LABELS = "vlabels/clabels mismatch"
    STRUCTURAL_CONTAINERS = "container set changed"
    COORD_REINDEX = "coordinates changed"
    SPARSITY = "coefficient sparsity changed"
    QUAD_OBJ = "quadratic objective changed"
    BACKEND_REJECTED = "backend raised UnsupportedUpdate"


@dataclass(frozen=True)
class VarSlice:
    bounds: slice
    type: slice


@dataclass(frozen=True)
class ConSlice:
    coef: slice
    rhs: slice
    sign: slice


def _empty_slice() -> slice:
    return slice(0, 0)


@dataclass
class _DiffBuilder:
    var_bounds_idx: list[np.ndarray] = field(default_factory=list)
    var_bounds_lo: list[np.ndarray] = field(default_factory=list)
    var_bounds_up: list[np.ndarray] = field(default_factory=list)
    var_type_pos: list[np.ndarray] = field(default_factory=list)
    var_type_kinds: list[np.ndarray] = field(default_factory=list)

    con_coef_rows: list[np.ndarray] = field(default_factory=list)
    con_coef_cols: list[np.ndarray] = field(default_factory=list)
    con_coef_vals: list[np.ndarray] = field(default_factory=list)

    con_rhs_idx: list[np.ndarray] = field(default_factory=list)
    con_rhs_vals: list[np.ndarray] = field(default_factory=list)
    con_rhs_signs: list[np.ndarray] = field(default_factory=list)

    con_sign_idx: list[np.ndarray] = field(default_factory=list)
    con_sign_vals: list[np.ndarray] = field(default_factory=list)

    var_slices: dict[str, VarSlice] = field(default_factory=dict)
    con_slices: dict[str, ConSlice] = field(default_factory=dict)

    obj_c_indices: np.ndarray | None = None
    obj_c_values: np.ndarray | None = None
    obj_sense: str | None = None

    _vb_cur: int = 0
    _vt_cur: int = 0
    _cc_cur: int = 0
    _cr_cur: int = 0
    _cs_cur: int = 0

    def push_var(
        self,
        name: str,
        bounds_idx: np.ndarray | None,
        lower: np.ndarray | None,
        upper: np.ndarray | None,
        type_positions: np.ndarray | None,
        type_kind: VarKind | None,
    ) -> None:
        b_start = self._vb_cur
        if bounds_idx is not None:
            self.var_bounds_idx.append(bounds_idx)
            self.var_bounds_lo.append(lower)
            self.var_bounds_up.append(upper)
            self._vb_cur += bounds_idx.size
        t_start = self._vt_cur
        if type_positions is not None:
            self.var_type_pos.append(type_positions)
            self.var_type_kinds.append(
                np.full(type_positions.size, type_kind, dtype=object)
            )
            self._vt_cur += type_positions.size
        self.var_slices[name] = VarSlice(
            bounds=slice(b_start, self._vb_cur),
            type=slice(t_start, self._vt_cur),
        )

    def push_con(
        self,
        name: str,
        coef_rows: np.ndarray | None,
        coef_cols: np.ndarray | None,
        coef_vals: np.ndarray | None,
        rhs_idx: np.ndarray | None,
        rhs_vals: np.ndarray | None,
        rhs_signs: np.ndarray | None,
        sign_idx: np.ndarray | None,
        sign_vals: np.ndarray | None,
    ) -> None:
        c_start = self._cc_cur
        if coef_rows is not None:
            self.con_coef_rows.append(coef_rows)
            self.con_coef_cols.append(coef_cols)
            self.con_coef_vals.append(coef_vals)
            self._cc_cur += coef_rows.size
        r_start = self._cr_cur
        if rhs_idx is not None:
            self.con_rhs_idx.append(rhs_idx)
            self.con_rhs_vals.append(rhs_vals)
            self.con_rhs_signs.append(rhs_signs)
            self._cr_cur += rhs_idx.size
        s_start = self._cs_cur
        if sign_idx is not None:
            self.con_sign_idx.append(sign_idx)
            self.con_sign_vals.append(sign_vals)
            self._cs_cur += sign_idx.size
        self.con_slices[name] = ConSlice(
            coef=slice(c_start, self._cc_cur),
            rhs=slice(r_start, self._cr_cur),
            sign=slice(s_start, self._cs_cur),
        )

    def set_objective(
        self,
        c_indices: np.ndarray | None,
        c_values: np.ndarray | None,
        sense: str | None,
    ) -> None:
        self.obj_c_indices = c_indices
        self.obj_c_values = c_values
        self.obj_sense = sense

    def finalize(self, diff: ModelDiff) -> None:
        diff.obj_c_indices = self.obj_c_indices
        diff.obj_c_values = self.obj_c_values
        diff.obj_sense = self.obj_sense
        diff.var_bounds_indices = _cat(self.var_bounds_idx, np.int32)
        diff.var_bounds_lower = _cat(self.var_bounds_lo, np.float64)
        diff.var_bounds_upper = _cat(self.var_bounds_up, np.float64)
        diff.var_type_positions = _cat(self.var_type_pos, np.int32)
        diff.var_type_kinds = _cat_obj(self.var_type_kinds)
        diff.con_coef_rows = _cat(self.con_coef_rows, np.int32)
        diff.con_coef_cols = _cat(self.con_coef_cols, np.int32)
        diff.con_coef_vals = _cat(self.con_coef_vals, np.float64)
        diff.con_rhs_indices = _cat(self.con_rhs_idx, np.int32)
        diff.con_rhs_values = _cat(self.con_rhs_vals, np.float64)
        diff.con_rhs_signs = _cat_str(self.con_rhs_signs)
        diff.con_sign_indices = _cat(self.con_sign_idx, np.int32)
        diff.con_sign_values = _cat_str(self.con_sign_vals)
        diff.var_slices = {
            n: s
            for n, s in self.var_slices.items()
            if s.bounds.stop > s.bounds.start or s.type.stop > s.type.start
        }
        diff.con_slices = {
            n: s
            for n, s in self.con_slices.items()
            if s.coef.stop > s.coef.start
            or s.rhs.stop > s.rhs.start
            or s.sign.stop > s.sign.start
        }


def _cat(parts: list[np.ndarray], dtype: type) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=dtype)
    return np.concatenate(parts).astype(dtype, copy=False)


def _cat_obj(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return _EMPTY_KIND
    return np.concatenate(parts)


def _cat_str(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return _EMPTY_U1
    return np.concatenate(parts)


@dataclass
class ModelDiff:
    rebuild_reason: RebuildReason = RebuildReason.NONE

    var_bounds_indices: np.ndarray = field(default_factory=lambda: _EMPTY_I32)
    var_bounds_lower: np.ndarray = field(default_factory=lambda: _EMPTY_F64)
    var_bounds_upper: np.ndarray = field(default_factory=lambda: _EMPTY_F64)
    var_type_positions: np.ndarray = field(default_factory=lambda: _EMPTY_I32)
    var_type_kinds: np.ndarray = field(default_factory=lambda: _EMPTY_KIND)

    con_coef_rows: np.ndarray = field(default_factory=lambda: _EMPTY_I32)
    con_coef_cols: np.ndarray = field(default_factory=lambda: _EMPTY_I32)
    con_coef_vals: np.ndarray = field(default_factory=lambda: _EMPTY_F64)

    con_rhs_indices: np.ndarray = field(default_factory=lambda: _EMPTY_I32)
    con_rhs_values: np.ndarray = field(default_factory=lambda: _EMPTY_F64)
    con_rhs_signs: np.ndarray = field(default_factory=lambda: _EMPTY_U1)

    con_sign_indices: np.ndarray = field(default_factory=lambda: _EMPTY_I32)
    con_sign_values: np.ndarray = field(default_factory=lambda: _EMPTY_U1)

    obj_c_indices: np.ndarray | None = None
    obj_c_values: np.ndarray | None = None
    obj_sense: str | None = None

    var_slices: dict[str, VarSlice] = field(default_factory=dict)
    con_slices: dict[str, ConSlice] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return (
            self.rebuild_reason is RebuildReason.NONE
            and self.var_bounds_indices.size == 0
            and self.var_type_positions.size == 0
            and self.con_coef_rows.size == 0
            and self.con_rhs_indices.size == 0
            and self.con_sign_indices.size == 0
            and self.obj_c_indices is None
            and self.obj_sense is None
        )

    @property
    def rebuild_required(self) -> bool:
        return self.rebuild_reason is not RebuildReason.NONE

    @property
    def changed_variables(self) -> set[str]:
        return set(self.var_slices)

    @property
    def changed_constraints(self) -> set[str]:
        return set(self.con_slices)

    @property
    def n_coef_updates(self) -> int:
        return int(self.con_coef_vals.size)

    def con_rhs_as_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) row-bounds form of the RHS updates."""
        vals = self.con_rhs_values
        signs = self.con_rhs_signs
        lower = np.where(signs == short_LESS_EQUAL, -np.inf, vals)
        upper = np.where(signs == short_GREATER_EQUAL, np.inf, vals)
        return lower, upper

    def summary(self) -> dict[str, int | bool | str | None]:
        return {
            "rebuild_reason": self.rebuild_reason.value,
            "var_bounds": int(self.var_bounds_indices.size),
            "var_type": int(self.var_type_positions.size),
            "con_rhs": int(self.con_rhs_indices.size),
            "con_sign": int(self.con_sign_indices.size),
            "con_coef_updates": int(self.con_coef_vals.size),
            "obj_linear_changed": self.obj_c_indices is not None,
            "obj_sense_changed_to": self.obj_sense,
        }

    def inspect_variable(self, name: str) -> dict[str, object]:
        sl = self.var_slices.get(name)
        if sl is None:
            return {}
        entry: dict[str, object] = {}
        if sl.bounds.stop > sl.bounds.start:
            entry["bounds_indices"] = self.var_bounds_indices[sl.bounds]
            entry["lower"] = self.var_bounds_lower[sl.bounds]
            entry["upper"] = self.var_bounds_upper[sl.bounds]
        if sl.type.stop > sl.type.start:
            entry["type_positions"] = self.var_type_positions[sl.type]
            entry["type_kinds"] = self.var_type_kinds[sl.type]
        return entry

    def inspect_constraint(self, name: str) -> dict[str, object]:
        sl = self.con_slices.get(name)
        if sl is None:
            return {}
        entry: dict[str, object] = {}
        if sl.coef.stop > sl.coef.start:
            entry["coef_rows"] = self.con_coef_rows[sl.coef]
            entry["coef_cols"] = self.con_coef_cols[sl.coef]
            entry["coef_vals"] = self.con_coef_vals[sl.coef]
        if sl.rhs.stop > sl.rhs.start:
            entry["rhs_indices"] = self.con_rhs_indices[sl.rhs]
            entry["rhs_values"] = self.con_rhs_values[sl.rhs]
            entry["rhs_signs"] = self.con_rhs_signs[sl.rhs]
        if sl.sign.stop > sl.sign.start:
            entry["sign_indices"] = self.con_sign_indices[sl.sign]
            entry["sign_values"] = self.con_sign_values[sl.sign]
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

    @classmethod
    def from_snapshot(
        cls,
        snapshot: ModelSnapshot,
        model: Model,
        same_model: bool = True,
        ignore_dims: Iterable[str] | None = None,
    ) -> ModelDiff:
        """
        Diff ``model`` against a captured ``snapshot``.

        Coordinate values are not compared by default. Pass ``ignore_dims``
        (e.g. ``ignore_dims=()`` or ``ignore_dims={"snapshot"}``) to opt into
        per-container coord-equality on every dim *not* in the set — a
        mismatch triggers ``RebuildReason.COORD_REINDEX``.
        """
        check_coords = ignore_dims is not None
        ignored = frozenset(ignore_dims) if ignore_dims is not None else frozenset()
        diff = cls()

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
        builder = _DiffBuilder()

        for name, var in model.variables.items():
            base_coords = snapshot.var_coords[name] if check_coords else None
            reason = _diff_var_container(
                builder, name, var, snapshot.var_buffers[name],
                base_coords, var_l2p, ignored, check_coords,
            )
            if reason is not None:
                diff.rebuild_reason = reason
                return diff

        for name, con in model.constraints.items():
            base_coords = snapshot.con_coords[name] if check_coords else None
            skip_coef_compare = same_model and not con._coef_dirty
            reason = _diff_con_container(
                builder, name, con, snapshot.con_buffers[name],
                base_coords, var_label_index, con_l2p, ignored, check_coords,
                skip_coef_compare,
            )
            if reason is not None:
                diff.rebuild_reason = reason
                return diff

        reason = _diff_objective(
            builder, model,
            snapshot.obj_c, snapshot.obj_quad_present, snapshot.obj_sense,
        )
        if reason is not None:
            diff.rebuild_reason = reason
            return diff

        builder.finalize(diff)
        return diff

    @classmethod
    def from_models(
        cls,
        model_a: Model,
        model_b: Model,
        ignore_dims: Iterable[str] | None = None,
    ) -> ModelDiff:
        """
        Diff two linopy models directly, without capturing a snapshot.

        ``model_a`` is the baseline, ``model_b`` is the target. The
        coefficient comparison runs unconditionally — no ``_coef_dirty``
        shortcut applies between independently-built models.
        """
        check_coords = ignore_dims is not None
        ignored = frozenset(ignore_dims) if ignore_dims is not None else frozenset()
        diff = cls()

        var_names_a = tuple(model_a.variables)
        con_names_a = tuple(model_a.constraints)
        if (
            var_names_a != tuple(model_b.variables)
            or con_names_a != tuple(model_b.constraints)
        ):
            diff.rebuild_reason = RebuildReason.STRUCTURAL_CONTAINERS
            return diff

        var_idx_a = model_a.variables.label_index
        con_idx_a = model_a.constraints.label_index
        var_idx_b = model_b.variables.label_index
        con_idx_b = model_b.constraints.label_index
        if not np.array_equal(var_idx_a.vlabels, var_idx_b.vlabels):
            diff.rebuild_reason = RebuildReason.STRUCTURAL_LABELS
            return diff
        if not np.array_equal(con_idx_a.clabels, con_idx_b.clabels):
            diff.rebuild_reason = RebuildReason.STRUCTURAL_LABELS
            return diff

        var_l2p = var_idx_b.label_to_pos
        con_l2p = con_idx_b.label_to_pos
        builder = _DiffBuilder()

        for name, var_b in model_b.variables.items():
            var_a = model_a.variables[name]
            base_buf = _extract_var_buffers(var_a)
            base_coords = _coord_snapshot(var_a) if check_coords else None
            reason = _diff_var_container(
                builder, name, var_b, base_buf,
                base_coords, var_l2p, ignored, check_coords,
            )
            if reason is not None:
                diff.rebuild_reason = reason
                return diff

        for name, con_b in model_b.constraints.items():
            con_a = model_a.constraints[name]
            base_buf = _extract_con_buffers(con_a, var_idx_a)
            base_coords = _coord_snapshot(con_a) if check_coords else None
            reason = _diff_con_container(
                builder, name, con_b, base_buf,
                base_coords, var_idx_b, con_l2p, ignored, check_coords,
                skip_coef_compare=False,
            )
            if reason is not None:
                diff.rebuild_reason = reason
                return diff

        reason = _diff_objective(
            builder, model_b,
            _objective_linear_vector(model_a),
            model_a.objective.is_quadratic,
            model_a.objective.sense,
        )
        if reason is not None:
            diff.rebuild_reason = reason
            return diff

        builder.finalize(diff)
        return diff


def _coords_equal(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray], ignored: frozenset[str]
) -> bool:
    keys = a.keys() - ignored
    if keys != b.keys() - ignored:
        return False
    return all(np.array_equal(a[k], b[k]) for k in keys)


def _active_container_positions(
    var: Variable, var_l2p: np.ndarray
) -> np.ndarray:
    labels = var.labels.values.ravel()
    active = labels[labels != -1]
    return var_l2p[active].astype(np.int32, copy=False)


def _diff_var_container(
    builder: _DiffBuilder,
    name: str,
    var: Variable,
    base_buf: ContainerVarBuffers,
    base_coords: dict[str, np.ndarray] | None,
    var_l2p: np.ndarray,
    ignored: frozenset[str],
    check_coords: bool,
) -> RebuildReason | None:
    new_buf = _extract_var_buffers(var)
    if new_buf.lower.shape != base_buf.lower.shape:
        return RebuildReason.COORD_REINDEX
    if not np.array_equal(new_buf.active_labels, base_buf.active_labels):
        return RebuildReason.STRUCTURAL_LABELS
    if check_coords and not _coords_equal(base_coords, _coord_snapshot(var), ignored):
        return RebuildReason.COORD_REINDEX

    lower_diff = new_buf.lower != base_buf.lower
    upper_diff = new_buf.upper != base_buf.upper
    type_changed = new_buf.type != base_buf.type

    bound_mask = lower_diff | upper_diff
    if not (bound_mask.any() or type_changed):
        return None

    bounds_idx = lower = upper = None
    if bound_mask.any():
        local_idx = np.flatnonzero(bound_mask)
        bounds_idx = var_l2p[
            new_buf.active_labels[local_idx]
        ].astype(np.int32, copy=False)
        lower = new_buf.lower[local_idx].astype(np.float64, copy=False)
        upper = new_buf.upper[local_idx].astype(np.float64, copy=False)

    type_positions = None
    type_kind: VarKind | None = None
    if type_changed:
        type_positions = _active_container_positions(var, var_l2p)
        type_kind = new_buf.type

    builder.push_var(name, bounds_idx, lower, upper, type_positions, type_kind)
    return None


def _diff_con_container(
    builder: _DiffBuilder,
    name: str,
    con: ConstraintBase,
    base_buf: ContainerConBuffers,
    base_coords: dict[str, np.ndarray] | None,
    var_label_index: VariableLabelIndex,
    con_l2p: np.ndarray,
    ignored: frozenset[str],
    check_coords: bool,
    skip_coef_compare: bool,
) -> RebuildReason | None:
    new_buf = _extract_con_buffers(con, var_label_index)
    if new_buf.indptr.shape != base_buf.indptr.shape:
        return RebuildReason.COORD_REINDEX
    if not np.array_equal(new_buf.active_labels, base_buf.active_labels):
        return RebuildReason.STRUCTURAL_LABELS
    if check_coords and not _coords_equal(base_coords, _coord_snapshot(con), ignored):
        return RebuildReason.COORD_REINDEX
    if not np.array_equal(new_buf.indptr, base_buf.indptr):
        return RebuildReason.SPARSITY
    if not np.array_equal(new_buf.indices, base_buf.indices):
        return RebuildReason.SPARSITY

    n_rows = new_buf.active_labels.size
    if n_rows == 0:
        return None

    if skip_coef_compare:
        row_value_changed = np.zeros(n_rows, dtype=bool)
        data_diff = None
    else:
        data_diff = new_buf.data != base_buf.data
        if data_diff.any():
            nnz_per_row = np.diff(new_buf.indptr)
            row_idx_per_nnz = np.repeat(np.arange(n_rows), nnz_per_row)
            row_value_changed = np.zeros(n_rows, dtype=bool)
            row_value_changed[row_idx_per_nnz[data_diff]] = True
        else:
            row_value_changed = np.zeros(n_rows, dtype=bool)

    rhs_changed = new_buf.rhs != base_buf.rhs
    sign_changed = new_buf.sign != base_buf.sign

    if not (row_value_changed.any() or rhs_changed.any() or sign_changed.any()):
        return None

    coef_rows = coef_cols = coef_vals = None
    if row_value_changed.any():
        coef_rows, coef_cols, coef_vals = _expand_coefs_coo(
            new_buf, con_l2p, row_value_changed
        )

    rhs_idx = rhs_vals = rhs_signs_arr = None
    if rhs_changed.any():
        idx = np.flatnonzero(rhs_changed)
        rhs_idx = con_l2p[
            new_buf.active_labels[idx]
        ].astype(np.int32, copy=False)
        rhs_vals = new_buf.rhs[idx].astype(np.float64, copy=False)
        rhs_signs_arr = new_buf.sign[idx]

    sign_idx = sign_vals = None
    if sign_changed.any():
        idx = np.flatnonzero(sign_changed)
        sign_idx = con_l2p[
            new_buf.active_labels[idx]
        ].astype(np.int32, copy=False)
        sign_vals = new_buf.sign[idx]

    builder.push_con(
        name,
        coef_rows, coef_cols, coef_vals,
        rhs_idx, rhs_vals, rhs_signs_arr,
        sign_idx, sign_vals,
    )
    return None


def _expand_coefs_coo(
    new_buf: ContainerConBuffers,
    con_l2p: np.ndarray,
    row_value_changed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.flatnonzero(row_value_changed)
    row_positions = con_l2p[
        new_buf.active_labels[idx]
    ].astype(np.int32, copy=False)
    indptr = new_buf.indptr
    nnz_per_changed = (indptr[idx + 1] - indptr[idx]).astype(np.int32)
    total_nnz = int(nnz_per_changed.sum())
    rows = np.repeat(row_positions, nnz_per_changed)
    cols = np.empty(total_nnz, dtype=np.int32)
    vals = np.empty(total_nnz, dtype=np.float64)
    cursor = 0
    for i in idx:
        s, e = int(indptr[i]), int(indptr[i + 1])
        n = e - s
        cols[cursor:cursor + n] = new_buf.indices[s:e]
        vals[cursor:cursor + n] = new_buf.data[s:e]
        cursor += n
    return rows, cols, vals


def _diff_objective(
    builder: _DiffBuilder,
    model: Model,
    base_obj_c: np.ndarray,
    base_obj_quad: bool,
    base_obj_sense: str,
) -> RebuildReason | None:
    if model.objective.is_quadratic or base_obj_quad:
        return RebuildReason.QUAD_OBJ

    obj_c = _objective_linear_vector(model)
    if obj_c.shape != base_obj_c.shape:
        return RebuildReason.COORD_REINDEX
    c_indices = c_values = None
    obj_diff_mask = obj_c != base_obj_c
    if obj_diff_mask.any():
        c_indices = np.flatnonzero(obj_diff_mask).astype(np.int32, copy=False)
        c_values = obj_c[c_indices].astype(np.float64, copy=False)

    sense = (
        model.objective.sense if model.objective.sense != base_obj_sense else None
    )
    builder.set_objective(c_indices, c_values, sense)
    return None
