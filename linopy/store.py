"""
The storage behind a :class:`~linopy.expressions.LinearExpression`.

A :class:`Store` holds the terms and the constant of a linear expression on a
fixed coordinate :class:`~linopy.csr.Grid`. It is an immutable value: every
operation returns a new store and the expression changes only by swapping
its store. :class:`DenseStore` keeps the dense ``coeffs``/``vars``/``const``
dataset, :class:`~linopy.csr.CSRStore` the sparse ``A @ x + c`` form. Flat
arrays, e.g. ``const`` or the ``rows`` of :meth:`Store.take`, run over the
grid cells in C order. Dense and sparse results of the same operation agree
under :func:`linopy.testing.assert_equivalent`, not in their term layout.
"""

from __future__ import annotations

import functools
import math
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Self, TypeAlias, cast, final

import numpy as np
import pandas as pd
import scipy.sparse
import xarray as xr
from xarray import Dataset

from linopy.alignment import as_dataarray, fill_missing_coords
from linopy.common import assign_fields
from linopy.constants import FACTOR_DIM, HELPER_DIMS, TERM_DIM
from linopy.semantics import absorb_absence, enforce_aux_conflict
from linopy.types import CONSTANT_TYPES

if TYPE_CHECKING:
    from linopy.csr import AuxCoords, CSRStore, Grid
    from linopy.model import Model

Reason: TypeAlias = str
"""Why a store refuses an operation, e.g. ``"reindex(method=...)"``."""


class Store(ABC):
    """
    Terms and constant of a linear expression on a fixed coordinate grid.

    A method returning ``Self | Reason`` may refuse a case with the reason;
    :class:`DenseStore` never does.
    """

    model: Model

    if TYPE_CHECKING:

        @property
        def grid(self) -> Grid: ...

        @property
        def const(self) -> np.ndarray: ...

    @property
    def dims(self) -> tuple[str, ...]:
        return self.grid.dims

    @classmethod
    @abstractmethod
    def from_dense(cls, ds: Dataset, model: Model) -> Self | Reason:
        """Build from a dense expression dataset on its own grid."""

    @classmethod
    @abstractmethod
    def from_labels(cls, ds: Dataset, model: Model) -> Self:
        """One-term store from variable labels ``vars`` and ``coeffs``, constant zero."""

    @classmethod
    @abstractmethod
    def from_grouper(
        cls,
        source: Store,
        model: Model,
        grouper: pd.Series | pd.DataFrame,
        group_dim: str,
        stacked: bool,
        coord_dims: tuple[str, ...],
    ) -> Self:
        """The grouped sum of ``source``, see :meth:`linopy.csr.CSRStore.from_grouper`."""

    @abstractmethod
    def to_dense(self, reason: Reason | None = None) -> DenseStore:
        """The dense equivalent, with a densify notice naming ``reason`` if given."""

    @abstractmethod
    def to_sparse(self, reason: Reason | None = None) -> CSRStore:
        """The sparse equivalent."""

    @abstractmethod
    def to_dataset(self) -> Dataset:
        """The dense layout as a dataset, without a notice."""

    @property
    @abstractmethod
    def nterm(self) -> int:
        """Number of terms per cell of the stored layout."""

    @property
    @abstractmethod
    def n_cells(self) -> int:
        """Number of grid cells."""

    @abstractmethod
    def has_terms(self) -> np.ndarray:
        """Per cell, whether it holds a term, the constant ignored."""

    @abstractmethod
    def cell(self, indices: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray, float]:
        """Coefficients, variable labels and constant of the cell at ``indices``."""

    @abstractmethod
    def terms(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cell, variable label and coefficient of the live terms."""

    @abstractmethod
    def solution(
        self, label_solution: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        """Per-cell value from the solution indexed by label, ``-1`` reading NaN."""

    @abstractmethod
    def scale(
        self,
        factor: float | np.ndarray,
        op: Callable[[Any, Any], Any] = operator.mul,
    ) -> Self:
        """
        Apply ``op`` with a scalar or per-cell ``factor`` to coefficients and
        constant; a cell made absent drops its terms.
        """

    @abstractmethod
    def assign_const(self, const: np.ndarray) -> Self:
        """Replace the per-cell constant; a cell made absent drops its terms."""

    def fillna(self, value: float) -> Self:
        """Resolve absent cells (NaN const) to the constant ``value``."""
        const = self.const
        return self.assign_const(np.where(np.isnan(const), value, const))

    @abstractmethod
    def prune(self) -> Self:
        """Drop the terms with a zero coefficient."""

    @abstractmethod
    def canonical(self) -> Self:
        """The canonical form a constraint is built from."""

    @abstractmethod
    def with_grid(self, grid: Grid) -> Self:
        """The same cells on ``grid``, a grid of the same shape."""

    @abstractmethod
    def take(self, rows: np.ndarray, grid: Grid) -> Self:
        """
        Gather cell ``rows[i]`` into cell ``i`` of ``grid``, ``-1`` leaving it
        absent without terms.
        """

    def broadcast(self, grid: Grid) -> Self:
        """Spread the cells over ``grid``, whose dims include this grid's."""
        rows, valid = self.grid.indexer(grid)
        return self.take(np.where(valid, rows, -1), self.grid.conform(grid))

    @abstractmethod
    def reindex(self, grid: Grid, fill: float = np.nan) -> Self:
        """Remap the cells onto ``grid``, new cells taking ``fill`` as constant."""

    @abstractmethod
    def rename(self, names: Mapping[str, str]) -> Self:
        """Relabel grid dims."""

    @abstractmethod
    def sum(self, dims: Iterable[str]) -> Self:
        """Sum over grid dims, every kept cell present."""

    @abstractmethod
    def aggregate(self, grid: Grid, rows: np.ndarray) -> Self:
        """Sum cell ``i`` into cell ``rows[i]`` of ``grid``."""

    @abstractmethod
    def add(self, *others: Self) -> Self:
        """
        Merge along the term dimension, all operands on the same grid; absent
        cells drop their terms.
        """

    @abstractmethod
    def concat(self, others: Iterable[Self], dim: str, grid: Grid) -> Self | Reason:
        """Stack along the grid dim ``dim`` into the cells of ``grid``."""

    @abstractmethod
    def dot(
        self,
        matrix: scipy.sparse.csr_array,
        contracted_dims: Iterable[str],
        new_indexes: Iterable[pd.Index],
    ) -> Self:
        """Contract grid dims against a sparse constant, see :meth:`linopy.csr.CSRStore.dot`."""


def expression_dataset(data: Dataset | Any | None, model: Model) -> Dataset:
    """Normalize ``data`` to the dataset layout of an expression."""
    if data is None:
        da = xr.DataArray([], dims=[TERM_DIM])
        data = Dataset({"coeffs": da, "vars": da, "const": 0.0})
    elif isinstance(data, CONSTANT_TYPES):
        const = as_dataarray(data)
        da = xr.DataArray([], dims=[TERM_DIM])
        data = Dataset({"coeffs": da, "vars": da, "const": const})
    elif not isinstance(data, Dataset):
        supported_types = ", ".join(
            map(lambda s: s.__qualname__, (*CONSTANT_TYPES, Dataset))
        )
        raise ValueError(
            f"data must be an instance of {supported_types}, got {type(data)}"
        )

    data = cast(Dataset, data)
    if not set(data).issuperset({"coeffs", "vars"}):
        raise ValueError("data must contain the fields 'coeffs' and 'vars' or 'const'")

    if np.issubdtype(data.vars, np.floating):
        data = assign_fields(
            data, vars=data.vars.fillna(-1).astype(model._dtypes["labels"])
        )
    if not np.issubdtype(data.coeffs, np.floating):
        data["coeffs"].values = data.coeffs.values.astype(float)

    data = fill_missing_coords(data)

    if TERM_DIM not in data.dims:
        raise ValueError("data must contain one dimension ending with '_term'")

    if "const" not in data:
        data = data.assign(const=0.0)
    elif not np.issubdtype(data.const, np.floating):
        data = assign_fields(data, const=data.const.astype(float))

    (data,) = xr.broadcast(data, exclude=HELPER_DIMS)
    (coeffs_vars,) = xr.broadcast(data[["coeffs", "vars"]], exclude=[FACTOR_DIM])
    coeffs_vars_dict = {str(k): v for k, v in coeffs_vars.items()}
    data = assign_fields(data, **coeffs_vars_dict)

    data = data.transpose(..., TERM_DIM)

    if drop_dims := set(HELPER_DIMS).intersection(data.coords):
        data = data.drop_vars(drop_dims)

    return data.assign_attrs(name=None)


def merged_grid(stores: Sequence[Store]) -> Grid:
    """
    The grid of the first store carrying the auxiliary coordinates of all,
    earlier stores taking precedence; conflicting ones raise (§11).
    """
    enforce_aux_conflict([Dataset(coords=s.grid.aux) for s in stores])
    aux: AuxCoords = {}
    for store in reversed(stores):
        aux |= store.grid.aux
    return replace(stores[0].grid, aux=aux)


@final
@dataclass(frozen=True, eq=False)
class DenseStore(Store):
    """
    A dense expression dataset: ``coeffs`` and ``vars`` on the grid dims
    followed by ``_term``, ``const`` on the grid dims. The grid is read off
    the dataset.
    """

    ds: Dataset
    model: Model

    @functools.cached_property
    def grid(self) -> Grid:
        from linopy.csr import Grid

        grid = Grid.from_dataset(self.ds, self.dims)
        if lost := set(map(str, self.ds.coords)) - {*grid.dims, *grid.aux}:
            raise ValueError(f"The grid cannot hold the coordinates {sorted(lost)}.")
        return grid

    @property
    def dims(self) -> tuple[str, ...]:
        return tuple(str(d) for d in self.ds.coeffs.dims if d != TERM_DIM)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.ds.sizes[d] for d in self.dims)

    @property
    def const(self) -> np.ndarray:
        return self.ds.const.transpose(*self.dims).to_numpy().reshape(-1)

    @property
    def nterm(self) -> int:
        return self.ds.sizes[TERM_DIM]

    @property
    def n_cells(self) -> int:
        return math.prod(self.shape)

    @classmethod
    def from_arrays(
        cls,
        grid: Grid,
        coeffs: np.ndarray,
        vars_: np.ndarray,
        const: np.ndarray,
        model: Model,
    ) -> DenseStore:
        """Build from per-cell term arrays ``(n_cells, nterm)`` and a flat constant."""
        shape = (*grid.shape, coeffs.shape[1])
        dims = (*grid.dims, TERM_DIM)
        ds = Dataset(
            {
                "coeffs": (dims, coeffs.reshape(shape)),
                "vars": (dims, vars_.reshape(shape)),
                "const": (grid.dims, const.reshape(grid.shape)),
            },
            coords=grid.indexes | grid.aux,
        )
        return cls(expression_dataset(ds, model), model)

    def _term_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Coefficients and variable labels as ``(n_cells, nterm)`` arrays."""
        dims = (*self.dims, TERM_DIM)
        shape = (self.n_cells, self.nterm)
        coeffs = self.ds.coeffs.transpose(*dims).to_numpy().reshape(shape)
        return coeffs, self.ds.vars.transpose(*dims).to_numpy().reshape(shape)

    def _with(
        self,
        grid: Grid,
        coeffs: np.ndarray | None = None,
        vars_: np.ndarray | None = None,
        const: np.ndarray | None = None,
    ) -> DenseStore:
        """This store on ``grid`` with the given arrays replaced."""
        own_coeffs, own_vars = self._term_arrays()
        return self.from_arrays(
            grid,
            own_coeffs if coeffs is None else coeffs,
            own_vars if vars_ is None else vars_,
            self.const if const is None else const,
            self.model,
        )

    @classmethod
    def from_dense(cls, ds: Dataset, model: Model) -> DenseStore:
        return cls(ds, model)

    @classmethod
    def from_labels(cls, ds: Dataset, model: Model) -> DenseStore:
        labels, coeffs = xr.broadcast(ds.vars, ds.coeffs)
        const = xr.where(labels == -1, np.nan, 0.0)
        terms = Dataset({"vars": labels, "coeffs": coeffs.astype(float)})
        dense = terms.expand_dims(TERM_DIM, -1).assign(const=const)
        return cls(expression_dataset(absorb_absence(dense), model), model)

    @classmethod
    def from_grouper(
        cls,
        source: Store,
        model: Model,
        grouper: pd.Series | pd.DataFrame,
        group_dim: str,
        stacked: bool,
        coord_dims: tuple[str, ...],
    ) -> DenseStore:
        from linopy.csr import CSRStore

        args = (grouper, group_dim, stacked, coord_dims)
        return CSRStore.from_grouper(source, model, *args).to_dense()

    def to_dense(self, reason: Reason | None = None) -> DenseStore:
        return self

    def to_sparse(self, reason: Reason | None = None) -> CSRStore:
        from linopy.csr import CSRStore

        return CSRStore.from_dense(self.ds, self.model)

    def to_dataset(self) -> Dataset:
        return self.ds

    def has_terms(self) -> np.ndarray:
        return (self._term_arrays()[1] != -1).any(axis=1)

    def cell(self, indices: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray, float]:
        ds = self.ds
        coeffs, vars_ = ds.coeffs.values[indices], ds.vars.values[indices]
        return coeffs, vars_, float(ds.const.values[indices])

    def terms(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coeffs, vars_ = self._term_arrays()
        present = ~np.isnan(self.const)[:, None]
        keep = present & (vars_ != -1) & (coeffs != 0)
        cells = np.broadcast_to(np.arange(self.n_cells)[:, None], keep.shape)
        return cells[keep], vars_[keep], coeffs[keep]

    def solution(
        self, label_solution: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        coeffs, vars_ = self._term_arrays()
        values = label_solution(vars_)[vars_]
        return np.nansum(coeffs * values, axis=1) + self.const

    def scale(
        self,
        factor: float | np.ndarray,
        op: Callable[[Any, Any], Any] = operator.mul,
    ) -> DenseStore:
        factor = np.broadcast_to(factor, (self.n_cells,))
        coeffs = op(self._term_arrays()[0], factor[:, None])
        const = op(self.const, factor)
        return self._with(self.grid, coeffs=coeffs, const=const).absorbed()

    def assign_const(self, const: np.ndarray) -> DenseStore:
        return self._with(self.grid, const=const).absorbed()

    def absorbed(self) -> DenseStore:
        """Drop the terms of absent cells."""
        return DenseStore(absorb_absence(self.ds), self.model)

    def prune(self) -> DenseStore:
        coeffs, vars_ = self._term_arrays()
        zero = coeffs == 0
        pruned = np.where(zero, np.nan, coeffs)
        return self._with(self.grid, pruned, np.where(zero, -1, vars_))

    def canonical(self) -> DenseStore:
        return self

    def with_grid(self, grid: Grid) -> DenseStore:
        return self._with(grid)

    def take(self, rows: np.ndarray, grid: Grid) -> DenseStore:
        present = rows >= 0
        safe = np.where(present, rows, 0)
        coeffs, vars_ = self._term_arrays()
        const = np.where(present, self.const[safe], np.nan)
        coeffs = np.where(present[:, None], coeffs[safe], np.nan)
        vars_ = np.where(present[:, None], vars_[safe], -1)
        return self.from_arrays(grid, coeffs, vars_, const, self.model)

    def reindex(self, grid: Grid, fill: float = np.nan) -> DenseStore:
        row_map, valid = grid.indexer(self.grid)
        source = np.full(grid.size, -1, dtype=np.int64)
        source[row_map[valid]] = np.flatnonzero(valid)
        taken = self.take(source, self.grid.conform(grid))
        return taken.assign_const(np.where(source < 0, fill, taken.const))

    def rename(self, names: Mapping[str, str]) -> DenseStore:
        return DenseStore(self.ds.rename(names), self.model)

    def sum(self, dims: Iterable[str]) -> DenseStore:
        dims = set(dims)
        kept = [d for d in self.dims if d not in dims]
        order = (*kept, *(d for d in self.dims if d in dims), TERM_DIM)
        grid = self.grid.select(kept)
        members = self.n_cells // max(grid.size, 1)
        shape = (grid.size, members * self.nterm)
        coeffs = self.ds.coeffs.transpose(*order).to_numpy().reshape(shape)
        vars_ = self.ds.vars.transpose(*order).to_numpy().reshape(shape)
        const = self.ds.const.transpose(*order[:-1]).to_numpy()
        const = np.nansum(const.reshape(grid.size, members), axis=1)
        return self.from_arrays(grid, coeffs, vars_, const, self.model)

    def aggregate(self, grid: Grid, rows: np.ndarray) -> DenseStore:
        return self.to_sparse().aggregate(grid, rows).to_dense()

    def add(self, *others: DenseStore) -> DenseStore:
        parts = (self, *others)
        terms = [p._term_arrays() for p in parts]
        coeffs = np.hstack([c for c, _ in terms])
        vars_ = np.hstack([v for _, v in terms])
        const = functools.reduce(np.add, (p.const for p in parts))
        merged = self.from_arrays(merged_grid(parts), coeffs, vars_, const, self.model)
        return merged.absorbed()

    def concat(self, others: Iterable[DenseStore], dim: str, grid: Grid) -> DenseStore:
        parts = (self, *others)
        nterm = max(p.nterm for p in parts)
        axis = self.dims.index(dim)
        padded = [p._padded(nterm) for p in parts]
        coeffs, vars_, const = (
            np.concatenate([a[i] for a in padded], axis=axis) for i in range(3)
        )
        shape = (grid.size, nterm)
        coeffs, vars_ = coeffs.reshape(shape), vars_.reshape(shape)
        return self.from_arrays(grid, coeffs, vars_, const.reshape(-1), self.model)

    def _padded(self, nterm: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Coefficients, labels and constant on the grid shape, padded to ``nterm``."""
        coeffs, vars_ = self._term_arrays()
        pad = ((0, 0), (0, nterm - self.nterm))
        shape = (*self.shape, nterm)
        return (
            np.pad(coeffs, pad, constant_values=np.nan).reshape(shape),
            np.pad(vars_, pad, constant_values=-1).reshape(shape),
            self.const.reshape(self.shape),
        )

    def dot(
        self,
        matrix: scipy.sparse.csr_array,
        contracted_dims: Iterable[str],
        new_indexes: Iterable[pd.Index],
    ) -> DenseStore:
        return self.to_sparse().dot(matrix, contracted_dims, new_indexes).to_dense()
