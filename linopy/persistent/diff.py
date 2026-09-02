from __future__ import annotations

import enum
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np

from linopy.constants import short_GREATER_EQUAL, short_LESS_EQUAL
from linopy.constraints import Constraint
from linopy.persistent.snapshot import (
    ContainerConBuffers,
    ContainerVarBuffers,
    ModelSnapshot,
    StructuralKey,
    _coord_snapshot,
    _extract_con_buffers,
    _extract_var_buffers,
    _objective_linear_vector,
)

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from linopy.common import ConstraintLabelIndex, VariableLabelIndex
    from linopy.constraints import ConstraintBase
    from linopy.model import Model
    from linopy.variables import Variable


class RebuildReason(enum.Enum):
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


def _cat(parts: list[np.ndarray], dtype: DTypeLike) -> np.ndarray:
    if not parts:
        return np.empty(0, dtype=dtype)
    return np.concatenate(parts).astype(dtype, copy=False)


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return a is b or np.array_equal(a, b)


def _coords_equal(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray], ignored: frozenset[str]
) -> bool:
    keys = a.keys() - ignored
    if keys != b.keys() - ignored:
        return False
    return all(np.array_equal(a[k], b[k]) for k in keys)


def _structural_reason(base: StructuralKey, model: Model) -> RebuildReason | None:
    if base.var_container_names != tuple(
        model.variables
    ) or base.con_container_names != tuple(model.constraints):
        return RebuildReason.STRUCTURAL_CONTAINERS
    if not np.array_equal(base.vlabels, model.variables.label_index.vlabels):
        return RebuildReason.STRUCTURAL_LABELS
    if not np.array_equal(base.clabels, model.constraints.label_index.clabels):
        return RebuildReason.STRUCTURAL_LABELS
    return None


@dataclass(frozen=True)
class _CoefDelta:
    """Coefficient changes of one container, expanded to COO lazily."""

    buf: ContainerConBuffers
    changed_rows: np.ndarray
    row_positions: np.ndarray
    nnz: int


@dataclass
class ModelDiff:
    """
    Flat-native delta between two structurally identical model states.

    Instances are produced by :meth:`from_snapshot` / :meth:`from_models`;
    any condition that cannot be expressed as an in-place delta is returned
    as a :class:`RebuildReason` instead of a diff.

    Coefficient changes are stored per container as ``coef_deltas``
    (changed rows referencing the container's CSR buffers) and expanded to
    COO triplets — ``con_coef_rows`` / ``con_coef_cols`` / ``con_coef_vals``
    — on first access.
    """

    var_bounds_indices: np.ndarray
    var_bounds_lower: np.ndarray
    var_bounds_upper: np.ndarray
    var_type_positions: np.ndarray
    var_type_kinds: np.ndarray

    coef_deltas: list[_CoefDelta]
    n_coef_updates: int

    con_rhs_indices: np.ndarray
    con_rhs_values: np.ndarray
    con_rhs_signs: np.ndarray

    con_sign_indices: np.ndarray
    con_sign_values: np.ndarray

    obj_c_indices: np.ndarray | None
    obj_c_values: np.ndarray | None
    obj_sense: str | None

    var_slices: dict[str, VarSlice]
    con_slices: dict[str, ConSlice]

    #: Snapshot of the diffed (target) model state, assembled from the
    #: buffers the diff walk already extracted — adopting it after a
    #: successful apply replaces a full re-capture. Note: holding a diff
    #: therefore pins all container buffers for its lifetime.
    snapshot: ModelSnapshot

    @property
    def is_empty(self) -> bool:
        return (
            self.var_bounds_indices.size == 0
            and self.var_type_positions.size == 0
            and self.n_coef_updates == 0
            and self.con_rhs_indices.size == 0
            and self.con_sign_indices.size == 0
            and self.obj_c_indices is None
            and self.obj_sense is None
        )

    @property
    def changed_variables(self) -> set[str]:
        return set(self.var_slices)

    @property
    def changed_constraints(self) -> set[str]:
        return set(self.con_slices)

    @cached_property
    def _coef_coo(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.empty(self.n_coef_updates, dtype=np.int32)
        cols = np.empty(self.n_coef_updates, dtype=np.int32)
        vals = np.empty(self.n_coef_updates, dtype=np.float64)
        cursor = 0
        for delta in self.coef_deltas:
            indptr = delta.buf.indptr
            starts = indptr[delta.changed_rows]
            counts = indptr[delta.changed_rows + 1] - starts
            run_offsets = np.repeat(np.cumsum(counts) - counts, counts)
            flat = np.repeat(starts, counts) + np.arange(delta.nnz) - run_offsets
            sl = slice(cursor, cursor + delta.nnz)
            rows[sl] = np.repeat(delta.row_positions, counts)
            cols[sl] = delta.buf.indices[flat]
            vals[sl] = delta.buf.data[flat]
            cursor += delta.nnz
        return rows, cols, vals

    @property
    def con_coef_rows(self) -> np.ndarray:
        return self._coef_coo[0]

    @property
    def con_coef_cols(self) -> np.ndarray:
        return self._coef_coo[1]

    @property
    def con_coef_vals(self) -> np.ndarray:
        return self._coef_coo[2]

    def con_rhs_as_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) row-bounds form of the RHS updates."""
        vals = self.con_rhs_values
        signs = self.con_rhs_signs
        lower = np.where(signs == short_LESS_EQUAL, -np.inf, vals)
        upper = np.where(signs == short_GREATER_EQUAL, np.inf, vals)
        return lower, upper

    def summary(self) -> dict[str, int | bool | str | None]:
        return {
            "var_bounds": int(self.var_bounds_indices.size),
            "var_type": int(self.var_type_positions.size),
            "con_rhs": int(self.con_rhs_indices.size),
            "con_sign": int(self.con_sign_indices.size),
            "con_coef_updates": self.n_coef_updates,
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
        parts = [
            f"{k}={v}" for k, v in self.summary().items() if v not in (0, False, None)
        ]
        return "ModelDiff(" + ", ".join(parts) + ")"

    @classmethod
    def from_snapshot(
        cls,
        snapshot: ModelSnapshot,
        model: Model,
        same_model: bool = False,
        ignore_dims: Iterable[str] = (),
    ) -> ModelDiff | RebuildReason:
        """
        Diff ``model`` against a captured ``snapshot``.

        Returns a :class:`ModelDiff` when the change is expressible in
        place, or the :class:`RebuildReason` that prevents it.

        Coordinate values are compared on every dim *not* in
        ``ignore_dims``; a mismatch triggers
        ``RebuildReason.COORD_REINDEX``. Pass ``ignore_dims={"snapshot"}``
        for rolling-horizon use cases where the snapshot coord
        legitimately shifts between solves.

        ``same_model`` is a perf hint, **default False**. When True, the
        diff trusts ``Constraint._coef_dirty`` to short-circuit the CSR
        walk for unchanged containers. That's only safe if every
        coefficient mutation went through ``Constraint.update`` (or the
        setters that forward there) — direct ``c.coeffs.values[...]``
        writes bypass the flag and would silently miss changes. Pass
        ``same_model=True`` only when you own the mutation contract.
        """
        reason = _structural_reason(snapshot.structural_key, model)
        if reason is not None:
            return reason

        builder = _DiffBuilder(
            model.variables.label_index,
            model.constraints.label_index,
            frozenset(ignore_dims),
            structural_key=snapshot.structural_key,
        )

        for name, var in model.variables.items():
            reason = builder.diff_var(
                name, var, snapshot.var_buffers[name], snapshot.var_coords[name]
            )
            if reason is not None:
                return reason

        for name, con in model.constraints.items():
            skip = same_model and isinstance(con, Constraint) and not con._coef_dirty
            reason = builder.diff_con(
                name,
                con,
                snapshot.con_buffers[name],
                snapshot.con_coords[name],
                skip_coef_compare=skip,
            )
            if reason is not None:
                return reason

        reason = builder.diff_objective(
            model, snapshot.obj_c, snapshot.obj_quad_present, snapshot.obj_sense
        )
        if reason is not None:
            return reason

        return builder.finalize()

    @classmethod
    def from_models(
        cls,
        model_a: Model,
        model_b: Model,
        ignore_dims: Iterable[str] = (),
    ) -> ModelDiff | RebuildReason:
        """
        Diff two linopy models directly, without capturing a snapshot.

        ``model_a`` is the baseline, ``model_b`` is the target. The
        coefficient comparison runs unconditionally — no ``_coef_dirty``
        shortcut applies between independently-built models. Returns a
        :class:`ModelDiff` or the :class:`RebuildReason` preventing an
        in-place update.

        Captures a snapshot of ``model_a`` and defers to
        :meth:`from_snapshot` with ``same_model=False``.
        """
        return cls.from_snapshot(
            ModelSnapshot.capture(model_a),
            model_b,
            same_model=False,
            ignore_dims=ignore_dims,
        )


class _DiffBuilder:
    """Accumulates per-container deltas and finalizes them into a ModelDiff."""

    def __init__(
        self,
        var_label_index: VariableLabelIndex,
        con_label_index: ConstraintLabelIndex,
        ignored: frozenset[str],
        structural_key: StructuralKey,
    ) -> None:
        self.var_label_index = var_label_index
        self.var_l2p = var_label_index.label_to_pos
        self.con_l2p = con_label_index.label_to_pos
        self.ignored = ignored
        self.structural_key = structural_key

        # Target-state material for the snapshot assembled in finalize().
        self.var_buffers: dict[str, ContainerVarBuffers] = {}
        self.con_buffers: dict[str, ContainerConBuffers] = {}
        self.var_coords: dict[str, dict[str, np.ndarray]] = {}
        self.con_coords: dict[str, dict[str, np.ndarray]] = {}
        self._snap_obj_c: np.ndarray | None = None
        self._snap_obj_sense: str | None = None

        self.var_bounds_idx: list[np.ndarray] = []
        self.var_bounds_lo: list[np.ndarray] = []
        self.var_bounds_up: list[np.ndarray] = []
        self.var_type_pos: list[np.ndarray] = []
        self.var_type_kinds: list[np.ndarray] = []

        self.coef_deltas: list[_CoefDelta] = []
        self.con_rhs_idx: list[np.ndarray] = []
        self.con_rhs_vals: list[np.ndarray] = []
        self.con_rhs_signs: list[np.ndarray] = []
        self.con_sign_idx: list[np.ndarray] = []
        self.con_sign_vals: list[np.ndarray] = []

        self.var_slices: dict[str, VarSlice] = {}
        self.con_slices: dict[str, ConSlice] = {}

        self.obj_c_indices: np.ndarray | None = None
        self.obj_c_values: np.ndarray | None = None
        self.obj_sense: str | None = None

        self._vb_cur = 0
        self._vt_cur = 0
        self._cc_cur = 0
        self._cr_cur = 0
        self._cs_cur = 0

    def diff_var(
        self,
        name: str,
        var: Variable,
        base_buf: ContainerVarBuffers,
        base_coords: dict[str, np.ndarray],
    ) -> RebuildReason | None:
        new_buf = _extract_var_buffers(var)
        new_coords = _coord_snapshot(var)
        self.var_buffers[name] = new_buf
        self.var_coords[name] = new_coords
        if not _coords_equal(base_coords, new_coords, self.ignored):
            return RebuildReason.COORD_REINDEX

        bound_mask = (new_buf.lower != base_buf.lower) | (
            new_buf.upper != base_buf.upper
        )
        bounds_changed = bool(bound_mask.any())
        type_changed = new_buf.type != base_buf.type
        if not (bounds_changed or type_changed):
            return None

        b_start, t_start = self._vb_cur, self._vt_cur
        if bounds_changed:
            local_idx = np.flatnonzero(bound_mask)
            positions = self.var_l2p[new_buf.active_labels[local_idx]]
            self.var_bounds_idx.append(positions.astype(np.int32, copy=False))
            self.var_bounds_lo.append(
                new_buf.lower[local_idx].astype(np.float64, copy=False)
            )
            self.var_bounds_up.append(
                new_buf.upper[local_idx].astype(np.float64, copy=False)
            )
            self._vb_cur += local_idx.size
        if type_changed:
            positions = self.var_l2p[new_buf.active_labels].astype(np.int32, copy=False)
            self.var_type_pos.append(positions)
            self.var_type_kinds.append(
                np.full(positions.size, new_buf.type, dtype=object)
            )
            self._vt_cur += positions.size
        self.var_slices[name] = VarSlice(
            bounds=slice(b_start, self._vb_cur),
            type=slice(t_start, self._vt_cur),
        )
        return None

    def diff_con(
        self,
        name: str,
        con: ConstraintBase,
        base_buf: ContainerConBuffers,
        base_coords: dict[str, np.ndarray],
        skip_coef_compare: bool,
    ) -> RebuildReason | None:
        new_buf = _extract_con_buffers(con, self.var_label_index)
        new_coords = _coord_snapshot(con)
        self.con_buffers[name] = new_buf
        self.con_coords[name] = new_coords
        if not _coords_equal(base_coords, new_coords, self.ignored):
            return RebuildReason.COORD_REINDEX
        if not _same(new_buf.indptr, base_buf.indptr):
            return RebuildReason.SPARSITY
        if not _same(new_buf.indices, base_buf.indices):
            return RebuildReason.SPARSITY

        n_rows = new_buf.active_labels.size
        if n_rows == 0:
            return None

        changed_rows = None
        if not (skip_coef_compare or new_buf.data is base_buf.data):
            data_diff = new_buf.data != base_buf.data
            if data_diff.any():
                nnz_per_row = np.diff(new_buf.indptr)
                row_of_nnz = np.repeat(np.arange(n_rows), nnz_per_row)
                changed_rows = np.unique(row_of_nnz[data_diff])

        rhs_idx = None
        if new_buf.rhs is not base_buf.rhs:
            rhs_idx = np.flatnonzero(new_buf.rhs != base_buf.rhs)
            if rhs_idx.size == 0:
                rhs_idx = None
        sign_idx = None
        if new_buf.sign is not base_buf.sign:
            sign_idx = np.flatnonzero(new_buf.sign != base_buf.sign)
            if sign_idx.size == 0:
                sign_idx = None

        if changed_rows is None and rhs_idx is None and sign_idx is None:
            return None

        c_start, r_start, s_start = self._cc_cur, self._cr_cur, self._cs_cur
        if changed_rows is not None:
            row_positions = self.con_l2p[new_buf.active_labels[changed_rows]].astype(
                np.int32, copy=False
            )
            indptr = new_buf.indptr
            nnz = int((indptr[changed_rows + 1] - indptr[changed_rows]).sum())
            self.coef_deltas.append(
                _CoefDelta(new_buf, changed_rows, row_positions, nnz)
            )
            self._cc_cur += nnz
        if rhs_idx is not None:
            positions = self.con_l2p[new_buf.active_labels[rhs_idx]]
            self.con_rhs_idx.append(positions.astype(np.int32, copy=False))
            self.con_rhs_vals.append(
                new_buf.rhs[rhs_idx].astype(np.float64, copy=False)
            )
            self.con_rhs_signs.append(new_buf.sign[rhs_idx])
            self._cr_cur += rhs_idx.size
        if sign_idx is not None:
            positions = self.con_l2p[new_buf.active_labels[sign_idx]]
            self.con_sign_idx.append(positions.astype(np.int32, copy=False))
            self.con_sign_vals.append(new_buf.sign[sign_idx])
            self._cs_cur += sign_idx.size
        self.con_slices[name] = ConSlice(
            coef=slice(c_start, self._cc_cur),
            rhs=slice(r_start, self._cr_cur),
            sign=slice(s_start, self._cs_cur),
        )
        return None

    def diff_objective(
        self,
        model: Model,
        base_obj_c: np.ndarray,
        base_obj_quad: bool,
        base_obj_sense: str,
    ) -> RebuildReason | None:
        if model.objective.is_quadratic or base_obj_quad:
            return RebuildReason.QUAD_OBJ

        obj_c = _objective_linear_vector(model)
        self._snap_obj_c = obj_c
        self._snap_obj_sense = model.objective.sense
        obj_diff_mask = obj_c != base_obj_c
        if obj_diff_mask.any():
            self.obj_c_indices = np.flatnonzero(obj_diff_mask).astype(
                np.int32, copy=False
            )
            self.obj_c_values = obj_c[self.obj_c_indices].astype(np.float64, copy=False)
        if model.objective.sense != base_obj_sense:
            self.obj_sense = model.objective.sense
        return None

    def finalize(self) -> ModelDiff:
        assert self._snap_obj_c is not None and self._snap_obj_sense is not None
        snapshot = ModelSnapshot(
            structural_key=self.structural_key,
            var_buffers=self.var_buffers,
            con_buffers=self.con_buffers,
            var_coords=self.var_coords,
            con_coords=self.con_coords,
            obj_c=self._snap_obj_c,
            obj_quad_present=False,
            obj_sense=self._snap_obj_sense,
        )
        return ModelDiff(
            snapshot=snapshot,
            var_bounds_indices=_cat(self.var_bounds_idx, np.int32),
            var_bounds_lower=_cat(self.var_bounds_lo, np.float64),
            var_bounds_upper=_cat(self.var_bounds_up, np.float64),
            var_type_positions=_cat(self.var_type_pos, np.int32),
            var_type_kinds=_cat(self.var_type_kinds, object),
            coef_deltas=self.coef_deltas,
            n_coef_updates=self._cc_cur,
            con_rhs_indices=_cat(self.con_rhs_idx, np.int32),
            con_rhs_values=_cat(self.con_rhs_vals, np.float64),
            con_rhs_signs=_cat(self.con_rhs_signs, "U1"),
            con_sign_indices=_cat(self.con_sign_idx, np.int32),
            con_sign_values=_cat(self.con_sign_vals, "U1"),
            obj_c_indices=self.obj_c_indices,
            obj_c_values=self.obj_c_values,
            obj_sense=self.obj_sense,
            var_slices=self.var_slices,
            con_slices=self.con_slices,
        )
