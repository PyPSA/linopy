"""
Window a model's flattened matrices.

:class:`MatrixWindow` is built once from a parent model. :meth:`MatrixWindow.select`
selects every container's labels with :meth:`xarray.DataArray.sel`, maps the
selected labels onto rows and columns of the parent's flattened matrices and
folds the out-of-window columns into the right-hand side at their fixed values.
No container is rebuilt.

Indicator constraints, quadratic objectives and native SOS constraints raise.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse
from xarray import DataArray

from linopy.alignment import as_dataarray

if TYPE_CHECKING:
    from linopy.model import Model

__all__ = ["MatrixWindow", "RestrictionError", "WindowMatrices"]


class RestrictionError(ValueError):
    """
    Raised when a model cannot be windowed.

    A window requires dimensions that exist in the model, carriers that contain
    every selected label, fixed values aligned with their variables, and folded
    boundary terms whose constant rows are satisfiable.
    """


@dataclass(frozen=True)
class WindowMatrices:
    """
    One window of a parent model, as flat arrays.

    Columns follow the parent variable labels in ``vlabels``; rows follow the
    parent constraint labels in ``clabels``. ``variables`` and ``constraints``
    hold each container's selected labels, onto which flat results are
    scattered. Out-of-window objective terms are dropped, so the objective value
    differs from the parent's by a constant.
    """

    A: scipy.sparse.csr_array
    b: np.ndarray
    sense: np.ndarray
    c: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    vlabels: np.ndarray
    clabels: np.ndarray
    vtypes: np.ndarray
    var_scaling: np.ndarray
    con_scaling: np.ndarray
    objective_scaling: float
    objective_sense: str
    variables: dict[str, DataArray]
    constraints: dict[str, DataArray]

    @property
    def Q(self) -> None:
        """No quadratic objective: matrix windows are linear."""
        return None

    @property
    def indicator_A(self) -> None:
        """No indicator constraints: matrix windows reject them."""
        return None


@dataclass(frozen=True)
class WindowResult:
    """Flat result of solving one window."""

    objective: float
    primal: np.ndarray
    dual: np.ndarray


def _select(containers: Any, selection: Mapping[str, Any]) -> dict[str, DataArray]:
    """Each container's labels, selected along the dimensions it carries."""
    selected = {}
    for name, item in containers.items():
        labels = item.labels
        indexers = {d: v for d, v in selection.items() if d in labels.dims}
        try:
            selected[name] = labels.sel(indexers)
        except KeyError as error:
            if (labels.values != -1).any():
                raise RestrictionError(
                    f"Cannot window {name!r}: it does not carry the selected "
                    f"labels ({error})."
                ) from error
            selected[name] = labels
    return selected


def _positions(selected: dict[str, DataArray], label_to_pos: np.ndarray) -> np.ndarray:
    """Parent matrix positions of the selected active labels, in container order."""
    empty = np.empty(0, dtype=label_to_pos.dtype)
    flat = np.concatenate([empty, *(lab.values.ravel() for lab in selected.values())])
    positions = label_to_pos[flat[flat != -1]]
    positions = positions[positions >= 0]
    if np.unique(positions).size != positions.size:
        raise RestrictionError("Cannot restrict model: the selection repeats labels.")
    return positions


def _align(given: Any, labels: DataArray, name: str) -> DataArray:
    """Fixed values of one variable, aligned and broadcast onto its labels."""
    try:
        value = (
            given
            if isinstance(given, DataArray)
            else as_dataarray(given, coords=labels.coords, dims=labels.dims)
        )
        shared = {d: labels.indexes[d] for d in value.indexes if d in labels.indexes}
        return value.sel(shared).broadcast_like(labels).transpose(*labels.dims)
    except (KeyError, ValueError) as error:
        raise RestrictionError(
            f"Fixed values for variable {name!r} do not match its coordinates "
            f"({error})."
        ) from error


def _check_dropped(
    rhs: np.ndarray, sense: np.ndarray, magnitude: np.ndarray, clabels: np.ndarray
) -> None:
    """Raise when a row that lost every term violates its sign."""
    tolerance = 1e-9 * np.maximum(1.0, magnitude)
    violated = (
        ((sense == "=") & (np.abs(rhs) > tolerance))
        | ((sense == "<") & (rhs < -tolerance))
        | ((sense == ">") & (rhs > tolerance))
    )
    if violated.any():
        raise RestrictionError(
            "Cannot window the matrices: folding the dangling terms makes "
            f"constraint labels {clabels[violated].tolist()} unsatisfiable "
            f"(folded rhs {rhs[violated].tolist()} against sign "
            f"{sense[violated].tolist()})."
        )


class MatrixWindow:
    """
    A matrix-level view of a model, windowed by label selection.

    Build once, then call :meth:`select` per window. The parent model is read
    only; every selection allocates its own arrays.
    """

    def __init__(self, model: Model) -> None:
        if model.variables.sos:
            raise NotImplementedError(
                "Matrix windows do not support unreformulated SOS constraints: "
                "call `model.apply_sos_reformulation()` first."
            )
        matrices = model.matrices
        if matrices.Q is not None:
            raise NotImplementedError(
                "Matrix windows do not support quadratic objectives."
            )
        if matrices.indicator_A is not None and matrices.indicator_A.nnz:
            raise NotImplementedError(
                "Matrix windows do not support indicator constraints."
            )
        if matrices.A is None:
            raise NotImplementedError("Matrix windows require at least one constraint.")
        self._model = model
        self._matrices = matrices
        self._A = matrices.A.tocsr()

    def select(
        self, selection: Mapping[str, Any], fix: Mapping[str, Any] | None = None
    ) -> WindowMatrices:
        """
        The window's matrices.

        ``selection`` maps dimensions to labels, as for :meth:`Model.restrict`.
        ``fix`` maps variable names to the values of out-of-window terms, which
        are folded into the right-hand side at their actual coefficient; a row
        that loses every term is dropped when its folded right-hand side
        satisfies its sign.
        """
        m, mat = self._model, self._matrices
        carried = {d for item in m._coordinate_carriers() for d in item.sizes}
        unknown = [d for d in selection if d not in carried | set(m.parameters.sizes)]
        if unknown:
            raise RestrictionError(
                f"Cannot restrict the model along {unknown}: the dimension does "
                "not exist in the model."
            )
        variables = _select(m.variables, selection)
        constraints = _select(m.constraints, selection)
        cols = _positions(variables, m.variables.label_index.label_to_pos)
        rows = _positions(constraints, m.constraints.label_index.label_to_pos)
        if not cols.size:
            raise RestrictionError(
                f"Cannot restrict model: the selection {dict(selection)} is empty."
            )

        outside = np.ones(mat.vlabels.size, dtype=bool)
        outside[cols] = False
        boundary = np.where(outside, self._fixed(fix), 0.0)
        missing = np.isnan(boundary)
        boundary[missing] = 0.0
        A = self._A[rows]
        magnitude = abs(A)
        unfixed = rows[magnitude @ missing.astype(float) > 0]
        if unfixed.size:
            bad = mat.clabels[unfixed]
            owners = [
                n for n, lab in constraints.items() if np.isin(lab.values, bad).any()
            ]
            raise RestrictionError(
                f"Cannot restrict model: constraint(s) {owners} have dangling "
                "terms on variables outside the selection without fixed values. "
                "Pass their values via `fix`."
            )
        rhs = mat.b[rows] - A @ boundary
        inner = A[:, cols]
        keep = np.diff(inner.indptr) > 0
        dropped = rows[~keep]
        _check_dropped(
            rhs[~keep],
            mat.sense[dropped],
            (magnitude @ np.abs(boundary))[~keep],
            mat.clabels[dropped],
        )
        kept = rows[keep]
        return WindowMatrices(
            A=inner[keep],
            b=rhs[keep],
            sense=mat.sense[kept],
            c=mat.c[cols],
            lb=mat.lb[cols],
            ub=mat.ub[cols],
            vlabels=mat.vlabels[cols],
            clabels=mat.clabels[kept],
            vtypes=mat.vtypes[cols],
            var_scaling=mat.var_scaling[cols],
            con_scaling=mat.con_scaling[kept],
            objective_scaling=float(m.objective.scaling),
            objective_sense=m.objective.sense,
            variables=variables,
            constraints=constraints,
        )

    def _fixed(self, fix: Mapping[str, Any] | None) -> np.ndarray:
        """Fixed values on the parent's scaled column space, NaN where unfixed."""
        by_label = np.full(self._model._xCounter, np.nan)
        for name, given in (fix or {}).items():
            if name not in self._model.variables:
                raise RestrictionError(
                    f"Cannot fix {name!r}: it does not name a variable of the model."
                )
            labels = self._model.variables[name].labels
            active = labels.values != -1
            by_label[labels.values[active]] = _align(given, labels, name).values[active]
        return by_label[self._matrices.vlabels] * self._matrices.var_scaling
