from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from linopy import expressions

if TYPE_CHECKING:
    from linopy.constraints import ConstraintBase
    from linopy.model import Model
    from linopy.variables import Variable


_INT64_MAX = np.iinfo(np.int64).max


class VarKind(enum.Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"
    INTEGER = "integer"
    SEMI_CONTINUOUS = "semi_continuous"


def _variable_type(var: Variable) -> VarKind:
    attrs = var.attrs
    if attrs.get("binary"):
        return VarKind.BINARY
    if attrs.get("integer"):
        return VarKind.INTEGER
    if attrs.get("semi_continuous"):
        return VarKind.SEMI_CONTINUOUS
    return VarKind.CONTINUOUS


def _objective_linear_vector(model: Model) -> np.ndarray:
    vlabels = model.variables.label_index.vlabels
    label_to_pos = model.variables.label_index.label_to_pos
    result = np.zeros(len(vlabels), dtype=np.float64)
    expr = model.objective.expression
    if isinstance(expr, expressions.QuadraticExpression):
        vars_2d = expr.data.vars.values
        coeffs_all = expr.data.coeffs.values.ravel()
        vars1, vars2 = vars_2d[0], vars_2d[1]
        linear = (vars1 == -1) | (vars2 == -1)
        var_labels = np.where(vars1[linear] != -1, vars1[linear], vars2[linear])
        coeffs = coeffs_all[linear]
    else:
        var_labels = expr.data.vars.values.ravel()
        coeffs = expr.data.coeffs.values.ravel()
    mask = var_labels != -1
    np.add.at(result, label_to_pos[var_labels[mask]], coeffs[mask])
    return result


def _canonicalize_rows(
    vars_arr: np.ndarray, coeffs_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Sort each row jointly by var index. -1 sentinels sort to the right."""
    vars_i64 = np.ascontiguousarray(vars_arr, dtype=np.int64)
    coeffs_f64 = np.ascontiguousarray(coeffs_arr, dtype=np.float64)
    if vars_i64.size == 0:
        return vars_i64, coeffs_f64
    sort_key = np.where(vars_i64 == -1, _INT64_MAX, vars_i64)
    if vars_i64.shape[1] <= 1 or np.all(np.diff(sort_key, axis=1) >= 0):
        return vars_i64, coeffs_f64
    order = np.argsort(sort_key, axis=1, kind="stable")
    rows = np.arange(vars_i64.shape[0])[:, None]
    return vars_i64[rows, order], coeffs_f64[rows, order]


def _extract_var_buffers(var: Variable) -> ContainerVarBuffers:
    labels_flat = var.labels.values.ravel()
    mask = labels_flat != -1
    return ContainerVarBuffers(
        lower=np.ascontiguousarray(var.lower.values.ravel()[mask], dtype=np.float64),
        upper=np.ascontiguousarray(var.upper.values.ravel()[mask], dtype=np.float64),
        type=_variable_type(var),
        active_labels=np.ascontiguousarray(labels_flat[mask], dtype=np.int64),
    )


def _extract_con_buffers(
    con: ConstraintBase, var_l2p: np.ndarray
) -> ContainerConBuffers:
    labels_flat = con.labels.values.ravel()
    vars_vals = con.vars.values
    coeffs_vals = con.coeffs.values
    n_rows = len(labels_flat)
    if n_rows > 0:
        vars_2d = vars_vals.reshape(n_rows, -1)
        coeffs_2d = coeffs_vals.reshape(vars_2d.shape)
    else:
        n_term = max(1, vars_vals.size)
        vars_2d = vars_vals.reshape(0, n_term)
        coeffs_2d = coeffs_vals.reshape(0, n_term)

    row_mask = (labels_flat != -1) & (vars_2d != -1).any(axis=1)
    active_labels = labels_flat[row_mask].astype(np.int64, copy=True)

    vars_active = vars_2d[row_mask]
    coeffs_active = coeffs_2d[row_mask].astype(np.float64, copy=True)

    valid = vars_active != -1
    col_indices = np.full(vars_active.shape, -1, dtype=np.int64)
    col_indices[valid] = var_l2p[vars_active[valid]]
    coeffs_clean = np.where(valid, coeffs_active, 0.0)

    vars_sorted, coeffs_sorted = _canonicalize_rows(col_indices, coeffs_clean)

    return ContainerConBuffers(
        coeffs=coeffs_sorted,
        vars=vars_sorted,
        rhs=con.rhs.values.ravel()[row_mask].astype(np.float64, copy=True),
        sign=con.sign.values.ravel()[row_mask].astype("U2", copy=True),
        active_labels=active_labels,
    )


@dataclass(frozen=True)
class StructuralKey:
    var_container_names: tuple[str, ...]
    con_container_names: tuple[str, ...]
    vlabels: np.ndarray
    clabels: np.ndarray

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, StructuralKey)
            and self.var_container_names == other.var_container_names
            and self.con_container_names == other.con_container_names
            and np.array_equal(self.vlabels, other.vlabels)
            and np.array_equal(self.clabels, other.clabels)
        )

    __hash__ = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ContainerVarBuffers:
    lower: np.ndarray
    upper: np.ndarray
    type: VarKind
    active_labels: np.ndarray


@dataclass(frozen=True)
class ContainerConBuffers:
    coeffs: np.ndarray
    vars: np.ndarray
    rhs: np.ndarray
    sign: np.ndarray
    active_labels: np.ndarray


def _coord_snapshot(obj: Variable | ConstraintBase) -> dict[str, np.ndarray]:
    return {str(name): np.asarray(idx) for name, idx in obj.indexes.items()}


@dataclass
class ModelSnapshot:
    structural_key: StructuralKey
    var_buffers: dict[str, ContainerVarBuffers] = field(default_factory=dict)
    con_buffers: dict[str, ContainerConBuffers] = field(default_factory=dict)
    var_coords: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    con_coords: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    obj_c: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    obj_quad_present: bool = False
    obj_sense: str = "min"

    @classmethod
    def capture(cls, model: Model) -> ModelSnapshot:
        var_label_index = model.variables.label_index
        con_label_index = model.constraints.label_index
        var_l2p = var_label_index.label_to_pos

        structural_key = StructuralKey(
            var_container_names=tuple(model.variables),
            con_container_names=tuple(model.constraints),
            vlabels=var_label_index.vlabels,
            clabels=con_label_index.clabels,
        )

        var_buffers = {
            name: _extract_var_buffers(var) for name, var in model.variables.items()
        }
        con_buffers = {
            name: _extract_con_buffers(con, var_l2p)
            for name, con in model.constraints.items()
        }
        var_coords = {
            name: _coord_snapshot(var) for name, var in model.variables.items()
        }
        con_coords = {
            name: _coord_snapshot(con) for name, con in model.constraints.items()
        }

        for con in model.constraints.data.values():
            con._coef_dirty = False

        return cls(
            structural_key=structural_key,
            var_buffers=var_buffers,
            con_buffers=con_buffers,
            var_coords=var_coords,
            con_coords=con_coords,
            obj_c=_objective_linear_vector(model),
            obj_quad_present=model.objective.is_quadratic,
            obj_sense=model.objective.sense,
        )
