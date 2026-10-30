"""
Window a model's flattened matrices.

:class:`MatrixWindow` is built once from a parent model. :meth:`MatrixWindow.select`
resolves the selection to positions once per distinct coordinate index, as
:meth:`xarray.DataArray.sel` would, selects every container's labels with
:meth:`xarray.DataArray.isel`, maps the selected labels onto rows and columns of
the parent's flattened matrices and folds the out-of-window columns into the right-hand side at their fixed values.
The window is a :class:`~linopy.matrices.Matrices` taken from the parent's with
:meth:`~linopy.matrices.Matrices.take`. No container is rebuilt.

Indicator constraints, quadratic objectives and native SOS constraints raise.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.sparse
from xarray import DataArray
from xarray.indexes import PandasIndex

from linopy.alignment import as_dataarray
from linopy.common import lookup_table
from linopy.csr import indptr_from_counts, repeat_per_row
from linopy.matrices import Matrices

if TYPE_CHECKING:
    from linopy.model import Model

__all__ = ["MatrixWindow", "RestrictionError"]


class RestrictionError(ValueError):
    """
    Raised when a model cannot be windowed.

    A window requires dimensions that exist in the model, carriers that contain
    every selected label, fixed values aligned with their variables, and folded
    boundary terms whose constant rows are satisfiable.
    """


def _indexer(index: pd.Index, dim: str, value: Any) -> Any:
    """Positional indexer of ``value`` on ``index``, a slice when contiguous."""
    indexer = PandasIndex(index, dim).sel({dim: value}).dim_indexers[dim]
    if not isinstance(indexer, np.ndarray) or indexer.ndim != 1 or not indexer.size:
        return indexer
    if (np.diff(indexer) == 1).all():
        return slice(int(indexer[0]), int(indexer[-1]) + 1)
    return indexer


class _Indexers:
    """Positional indexers of a selection, resolved once per distinct index."""

    def __init__(self, selection: Mapping[str, Any]) -> None:
        self._selection = selection
        self._cache: dict[str, list[tuple[pd.Index, Any]]] = {}
        self.distinct = True

    def __call__(self, labels: DataArray) -> dict[str, Any]:
        return {
            d: self._resolve(labels.indexes[d], d)
            for d in self._selection
            if d in labels.dims
        }

    def _resolve(self, index: pd.Index, dim: str) -> Any:
        cached = self._cache.setdefault(dim, [])
        for known, indexer in cached:
            if known is index or known.equals(index):
                return indexer
        indexer = _indexer(index, dim, self._selection[dim])
        self.distinct &= isinstance(indexer, int | slice)
        cached.append((index, indexer))
        return indexer


def _select(containers: Any, indexers: _Indexers) -> dict[str, DataArray]:
    """Each container's labels, selected along the dimensions it carries."""
    selected = {}
    for name, item in containers.items():
        labels = item.labels
        try:
            selected[name] = labels.isel(indexers(labels))
        except KeyError as error:
            if (labels.values != -1).any():
                raise RestrictionError(
                    f"Cannot window {name!r}: it does not carry the selected "
                    f"labels ({error})."
                ) from error
            selected[name] = labels
    return selected


def _positions(
    selected: dict[str, DataArray], label_to_pos: np.ndarray, distinct: bool
) -> np.ndarray:
    """Parent matrix positions of the selected active labels, in container order."""
    empty = np.empty(0, dtype=label_to_pos.dtype)
    flat = np.concatenate([empty, *(lab.values.ravel() for lab in selected.values())])
    positions = label_to_pos[flat[flat != -1]]
    positions = positions[positions >= 0]
    if not distinct and np.unique(positions).size != positions.size:
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
        same_shape = value.dims == labels.dims and value.shape == labels.shape
        if same_shape and _equal(value, shared):
            return value
        return value.sel(shared).broadcast_like(labels).transpose(*labels.dims)
    except (KeyError, ValueError) as error:
        raise RestrictionError(
            f"Fixed values for variable {name!r} do not match its coordinates "
            f"({error})."
        ) from error


def _equal(value: DataArray, indexes: dict[str, pd.Index]) -> bool:
    """Whether ``value`` carries exactly ``indexes`` on every dimension."""
    return value.indexes.keys() == indexes.keys() and all(
        value.indexes[d].equals(index) for d, index in indexes.items()
    )


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
        if matrices.indicator_b.size:
            raise NotImplementedError(
                "Matrix windows do not support indicator constraints."
            )
        if matrices.A is None:
            raise NotImplementedError("Matrix windows require at least one constraint.")
        self._model = model
        self._matrices = matrices
        self._A = matrices.A
        frame = {f.name: vars(matrices)[f.name] for f in fields(Matrices)}
        self._frame = Matrices(**{**frame, "A": None})
        self._dims = {d for item in model._coordinate_carriers() for d in item.sizes}
        self._dims |= set(model.parameters.sizes)

    def select(
        self, selection: Mapping[str, Any], fix: Mapping[str, Any] | None = None
    ) -> tuple[Matrices, dict[str, DataArray], dict[str, DataArray]]:
        """
        The window's matrices and each container's selected labels.

        ``selection`` maps dimensions to labels, as for :meth:`Model.restrict`.
        ``fix`` maps variable names to the values of out-of-window terms, which
        are folded into the right-hand side at their actual coefficient; a row
        that loses every term is dropped when its folded right-hand side
        satisfies its sign.

        Columns follow the parent variable labels in ``vlabels``; rows follow the
        parent constraint labels in ``clabels``. The selected labels are where
        flat results are scattered back. Out-of-window objective terms are
        dropped, so the objective value differs from the parent's by a constant.
        """
        m, mat = self._model, self._matrices
        unknown = [d for d in selection if d not in self._dims]
        if unknown:
            raise RestrictionError(
                f"Cannot restrict the model along {unknown}: the dimension does "
                "not exist in the model."
            )
        indexers = _Indexers(selection)
        variables = _select(m.variables, indexers)
        constraints = _select(m.constraints, indexers)
        cols = _positions(
            variables, m.variables.label_index.label_to_pos, indexers.distinct
        )
        rows = _positions(
            constraints, m.constraints.label_index.label_to_pos, indexers.distinct
        )
        if not cols.size:
            raise RestrictionError(
                f"Cannot restrict model: the selection {dict(selection)} is empty."
            )

        A = self._A[rows]
        position = lookup_table(cols, mat.vlabels.size, dtype=A.indices.dtype)
        mapped = position[A.indices]
        inside = mapped >= 0
        entry_rows = repeat_per_row(np.arange(rows.size), A.indptr)
        outer_rows = entry_rows[~inside]
        outer_data = A.data[~inside]
        fixed = self._fixed(fix, A.indices[~inside])
        missing = np.isnan(fixed) & (outer_data != 0)
        if missing.any():
            bad = mat.clabels[np.unique(rows[outer_rows[missing]])]
            owners = [
                n for n, lab in constraints.items() if np.isin(lab.values, bad).any()
            ]
            raise RestrictionError(
                f"Cannot restrict model: constraint(s) {owners} have dangling "
                "terms on variables outside the selection without fixed values. "
                "Pass their values via `fix`."
            )
        folded = outer_data * np.nan_to_num(fixed)
        rhs = mat.b[rows] - np.bincount(outer_rows, folded, minlength=rows.size)
        counts = np.bincount(entry_rows[inside], minlength=rows.size)
        keep = counts > 0
        magnitude = np.bincount(outer_rows, np.abs(folded), minlength=rows.size)
        dropped = rows[~keep]
        _check_dropped(
            rhs[~keep], mat.sense[dropped], magnitude[~keep], mat.clabels[dropped]
        )
        indptr = indptr_from_counts(counts[keep], A.indptr.dtype)
        inner = scipy.sparse.csr_array(
            (A.data[inside], mapped[inside], indptr), shape=(indptr.size - 1, cols.size)
        )
        window = replace(self._frame.take(rows[keep], cols), A=inner, b=rhs[keep])
        return window, variables, constraints

    def _fixed(self, fix: Mapping[str, Any] | None, cols: np.ndarray) -> np.ndarray:
        """Fixed values of the given parent columns, scaled, NaN where unfixed."""
        mat = self._matrices
        labels = mat.vlabels[cols]
        values = np.full(cols.size, np.nan)
        for name, given in (fix or {}).items():
            if name not in self._model.variables:
                raise RestrictionError(
                    f"Cannot fix {name!r}: it does not name a variable of the model."
                )
            variable = self._model.variables[name]
            start, stop = variable.range
            own = (labels >= start) & (labels < stop)
            aligned = _align(given, variable.labels, name).values.ravel()
            values[own] = aligned[labels[own] - start]
        return values * mat.var_scaling[cols]
