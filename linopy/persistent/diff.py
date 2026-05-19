from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from linopy.persistent.snapshot import (
    CoefPattern,
    ModelSnapshot,
    _canonical_csr,
    _coord_snapshot,
    _objective_linear_vector,
    _variable_type,
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
class ModelDiff:
    rebuild_reason: RebuildReason = RebuildReason.NONE

    var_lb: dict[str, xr.DataArray] = field(default_factory=dict)
    var_ub: dict[str, xr.DataArray] = field(default_factory=dict)
    var_type: dict[str, str] = field(default_factory=dict)
    con_rhs: dict[str, xr.DataArray] = field(default_factory=dict)
    con_sign: dict[str, xr.DataArray] = field(default_factory=dict)
    con_coef_updates: dict[str, np.ndarray] = field(default_factory=dict)

    obj_linear: xr.DataArray | None = None
    obj_sense: str | None = None

    @property
    def is_empty(self) -> bool:
        return (
            self.rebuild_reason is RebuildReason.NONE
            and not self.var_lb
            and not self.var_ub
            and not self.var_type
            and not self.con_rhs
            and not self.con_sign
            and not self.con_coef_updates
            and self.obj_linear is None
            and self.obj_sense is None
        )

    @property
    def rebuild_required(self) -> bool:
        return self.rebuild_reason is not RebuildReason.NONE


def _coords_equal(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    if a.keys() != b.keys():
        return False
    return all(np.array_equal(a[k], b[k]) for k in a)


def _any_diff(a: xr.DataArray, b: xr.DataArray) -> bool:
    return bool((a != b).any().item())


def compute_diff(
    snapshot: ModelSnapshot, model: Model, same_model: bool = True
) -> ModelDiff:
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

    for name, var in model.variables.items():
        if not _coords_equal(snapshot.var_coords[name], _coord_snapshot(var)):
            diff.rebuild_reason = RebuildReason.COORD_REINDEX
            return diff
        if _any_diff(snapshot.var_lb[name], var.lower):
            diff.var_lb[name] = var.lower.copy(deep=True)
        if _any_diff(snapshot.var_ub[name], var.upper):
            diff.var_ub[name] = var.upper.copy(deep=True)
        vtype = _variable_type(var)
        if snapshot.var_type[name] != vtype:
            diff.var_type[name] = vtype

    for name, con in model.constraints.items():
        if not _coords_equal(snapshot.con_coords[name], _coord_snapshot(con)):
            diff.rebuild_reason = RebuildReason.COORD_REINDEX
            return diff
        if _any_diff(snapshot.con_rhs[name], con.rhs):
            diff.con_rhs[name] = con.rhs.copy(deep=True)
        if _any_diff(snapshot.con_sign[name], con.sign):
            diff.con_sign[name] = con.sign.copy(deep=True)

    if same_model:
        dirty_names = [n for n, c in model.constraints.items() if c._coef_dirty]
    else:
        dirty_names = list(con_names)

    for name in dirty_names:
        con = model.constraints[name]
        indptr, indices, data = _canonical_csr(con, var_label_index)
        pattern = CoefPattern(indptr=indptr, indices=indices)
        if pattern == snapshot.con_coef_pattern[name]:
            diff.con_coef_updates[name] = data
        else:
            diff.rebuild_reason = RebuildReason.SPARSITY
            return diff

    obj_quad_present = model.objective.is_quadratic
    if obj_quad_present != snapshot.obj_quad_present:
        diff.rebuild_reason = RebuildReason.QUAD_OBJ
        return diff
    if obj_quad_present:
        diff.rebuild_reason = RebuildReason.QUAD_OBJ
        return diff

    obj_linear = _objective_linear_vector(model)
    if not np.array_equal(
        obj_linear.values, snapshot.obj_linear.values
    ) or not np.array_equal(
        obj_linear["vlabel"].values, snapshot.obj_linear["vlabel"].values
    ):
        diff.obj_linear = obj_linear.copy(deep=True)

    if model.objective.sense != snapshot.obj_sense:
        diff.obj_sense = model.objective.sense

    return diff
