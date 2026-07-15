from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from linopy import expressions
from linopy.constraints import Constraint

if TYPE_CHECKING:
    from linopy.constraints import ConstraintBase
    from linopy.model import Model
    from linopy.variables import Variable, VariableLabelIndex


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


def _extract_var_buffers(var: Variable) -> ContainerVarBuffers:
    # Boolean masking copies, so the buffers never alias the live model
    # arrays — the snapshot stays a valid baseline even after in-place
    # ``.values[...]`` mutations.
    labels_flat = var.labels.values.ravel()
    mask = labels_flat != -1
    return ContainerVarBuffers(
        lower=var.lower.values.ravel()[mask].astype(np.float64, copy=False),
        upper=var.upper.values.ravel()[mask].astype(np.float64, copy=False),
        type=_variable_type(var),
        active_labels=labels_flat[mask].astype(np.int64, copy=False),
    )


def _extract_con_buffers(
    con: ConstraintBase, var_label_index: VariableLabelIndex
) -> ContainerConBuffers:
    """
    Extract flat constraint buffers without copying.

    Mutable ``Constraint`` objects build fresh arrays in
    ``to_matrix_with_rhs``, so the buffers are exclusively owned.
    ``CSRConstraint`` returns its stored arrays — the buffers share memory
    with the constraint, every mutation path rebinds whole arrays
    (copy-on-write), and the diff uses object identity to skip comparisons
    on untouched containers.
    """
    csr, con_labels, b, sense = con.to_matrix_with_rhs(var_label_index)
    return ContainerConBuffers(
        indptr=csr.indptr,
        indices=csr.indices,
        data=np.asarray(csr.data, dtype=np.float64),
        rhs=np.asarray(b, dtype=np.float64),
        sign=np.asarray(sense, dtype="U1"),
        active_labels=np.asarray(con_labels, dtype=np.int64),
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
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    rhs: np.ndarray
    sign: np.ndarray
    active_labels: np.ndarray


def _coord_snapshot(obj: Variable | ConstraintBase) -> dict[str, np.ndarray]:
    return {str(name): np.asarray(idx) for name, idx in obj.indexes.items()}


def clear_coef_dirty(model: Model) -> None:
    """
    Reset ``Constraint._coef_dirty`` on every constraint of ``model``.

    Must be called exactly when a snapshot reflecting the model's current
    state is adopted by a tracking solver — clearing without adopting makes
    a later ``same_model=True`` diff silently skip changed coefficients.
    """
    for con in model.constraints.data.values():
        if isinstance(con, Constraint):
            con._coef_dirty = False


@dataclass
class ModelSnapshot:
    structural_key: StructuralKey
    var_buffers: dict[str, ContainerVarBuffers] = field(default_factory=dict)
    con_buffers: dict[str, ContainerConBuffers] = field(default_factory=dict)
    var_coords: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    con_coords: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    obj_c: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    obj_quad_present: bool = False
    obj_sense: str = "min"

    @classmethod
    def capture(cls, model: Model) -> ModelSnapshot:
        var_label_index = model.variables.label_index
        con_label_index = model.constraints.label_index

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
            name: _extract_con_buffers(con, var_label_index)
            for name, con in model.constraints.items()
        }
        var_coords = {
            name: _coord_snapshot(var) for name, var in model.variables.items()
        }
        con_coords = {
            name: _coord_snapshot(con) for name, con in model.constraints.items()
        }

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
