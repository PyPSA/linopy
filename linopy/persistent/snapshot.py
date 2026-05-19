from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from linopy import expressions

if TYPE_CHECKING:
    from linopy.constraints import ConstraintBase
    from linopy.model import Model
    from linopy.variables import Variable


def _variable_type(var: Variable) -> str:
    attrs = var.attrs
    if attrs.get("binary"):
        return "binary"
    if attrs.get("integer"):
        return "integer"
    if attrs.get("semi_continuous"):
        return "semi_continuous"
    return "continuous"


def _coord_snapshot(obj: Variable | ConstraintBase) -> dict[str, np.ndarray]:
    return {str(name): np.asarray(idx) for name, idx in obj.indexes.items()}


def _canonical_csr(
    constraint: ConstraintBase, label_index
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    csr, _ = constraint.to_matrix(label_index)
    csr.sort_indices()
    csr.eliminate_zeros()
    indptr = csr.indptr.astype(np.int64)
    indices = csr.indices.astype(np.int64)
    return indptr, indices, csr.data


def _objective_linear_vector(model: Model) -> xr.DataArray:
    vlabels = model.variables.label_index.vlabels
    label_to_pos = model.variables.label_index.label_to_pos
    result = np.zeros(len(vlabels))
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
    return xr.DataArray(result, dims="vlabel", coords={"vlabel": vlabels})


@dataclass(frozen=True)
class CoefPattern:
    indptr: np.ndarray
    indices: np.ndarray

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CoefPattern)
            and np.array_equal(self.indptr, other.indptr)
            and np.array_equal(self.indices, other.indices)
        )

    __hash__ = None  # type: ignore[assignment]


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


@dataclass
class ModelSnapshot:
    structural_key: StructuralKey

    var_lb: dict[str, xr.DataArray] = field(default_factory=dict)
    var_ub: dict[str, xr.DataArray] = field(default_factory=dict)
    var_type: dict[str, str] = field(default_factory=dict)
    var_coords: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    con_rhs: dict[str, xr.DataArray] = field(default_factory=dict)
    con_sign: dict[str, xr.DataArray] = field(default_factory=dict)
    con_coords: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    con_coef_pattern: dict[str, CoefPattern] = field(default_factory=dict)

    obj_linear: xr.DataArray = field(default_factory=lambda: xr.DataArray([]))
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

        var_lb: dict[str, xr.DataArray] = {}
        var_ub: dict[str, xr.DataArray] = {}
        var_type: dict[str, str] = {}
        var_coords: dict[str, dict[str, np.ndarray]] = {}
        for name, var in model.variables.items():
            var_lb[name] = var.lower.copy(deep=True)
            var_ub[name] = var.upper.copy(deep=True)
            var_type[name] = _variable_type(var)
            var_coords[name] = _coord_snapshot(var)

        con_rhs: dict[str, xr.DataArray] = {}
        con_sign: dict[str, xr.DataArray] = {}
        con_coords: dict[str, dict[str, np.ndarray]] = {}
        con_coef_pattern: dict[str, CoefPattern] = {}
        for name, con in model.constraints.items():
            con_rhs[name] = con.rhs.copy(deep=True)
            con_sign[name] = con.sign.copy(deep=True)
            con_coords[name] = _coord_snapshot(con)
            indptr, indices, _ = _canonical_csr(con, var_label_index)
            con_coef_pattern[name] = CoefPattern(indptr=indptr, indices=indices)

        obj_linear = _objective_linear_vector(model).copy(deep=True)
        obj_quad_present = model.objective.is_quadratic
        obj_sense = model.objective.sense

        return cls(
            structural_key=structural_key,
            var_lb=var_lb,
            var_ub=var_ub,
            var_type=var_type,
            var_coords=var_coords,
            con_rhs=con_rhs,
            con_sign=con_sign,
            con_coords=con_coords,
            con_coef_pattern=con_coef_pattern,
            obj_linear=obj_linear,
            obj_quad_present=obj_quad_present,
            obj_sense=obj_sense,
        )
