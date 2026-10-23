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
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Self, TypeAlias, cast, final

import numpy as np
import pandas as pd
import scipy.sparse
import xarray as xr
from xarray import DataArray, Dataset

from linopy.alignment import as_dataarray, fill_missing_coords
from linopy.common import assign_fields, check_common_keys_values
from linopy.constants import FACTOR_DIM, HELPER_DIMS, STACKED_TERM_DIM, TERM_DIM
from linopy.semantics import absorb_absence, enforce_aux_conflict, enforce_merge_dims
from linopy.types import CONSTANT_TYPES, DimsLike

if TYPE_CHECKING:
    from xarray.core.types import JoinOptions

    from linopy.csr import AuxCoords, CSRStore, Grid
    from linopy.model import Model

Reason: TypeAlias = str
"""Why a store refuses an operation, e.g. ``"reindex(method=...)"``."""

FILL_VALUE = {"vars": -1, "coeffs": np.nan, "const": np.nan}
"""What a dense expression dataset holds at a cell or term it lacks."""


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

    @classmethod
    @abstractmethod
    def from_matmul(
        cls,
        source: CSRStore,
        matrix: scipy.sparse.csr_array,
        contracted_dims: Iterable[str],
        new_indexes: Iterable[pd.Index],
    ) -> Self:
        """The contraction ``source @ matrix``, see :meth:`linopy.csr.CSRStore.dot`."""

    @abstractmethod
    def to_dense(self, reason: Reason | None = None) -> DenseStore:
        """The dense equivalent, with a densify notice naming ``reason`` if given."""

    @abstractmethod
    def to_sparse(self, reason: Reason | None = None) -> CSRStore | Reason:
        """
        The sparse equivalent, with a sparsify notice naming ``reason`` if
        given; the reason a dense store cannot convert instead.
        """

    @abstractmethod
    def to_dataset(self) -> Dataset:
        """The dense layout as a dataset, without a notice."""

    @abstractmethod
    def copy(self, model: Model, deep: bool = True) -> Self:
        """The same store on ``model``, its arrays copied if ``deep``."""

    @property
    @abstractmethod
    def nterm(self) -> int:
        """Number of terms per cell of the stored layout."""

    @property
    @abstractmethod
    def n_cells(self) -> int:
        """Number of grid cells."""

    @abstractmethod
    def dataarray(self, values: np.ndarray, name: str) -> DataArray:
        """Wrap one value per cell as a DataArray on the store's coordinates."""

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
    def combine(
        self,
        const: DataArray,
        op: Callable[[Any, Any], Any],
        factor: DataArray | None = None,
        reindex: bool = False,
    ) -> Self | Reason:
        """
        Set the constant to ``const``, an aligned constant whose labels the
        cells are first remapped onto if ``reindex``, the new cells without
        terms; with a ``factor`` on those labels, apply ``op`` with it to the
        coefficients too.
        """

    @abstractmethod
    def absorbed(self) -> Self:
        """Drop the terms of absent cells."""

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
    def sum(self, dims: Iterable[str] | None = None) -> Self:
        """
        Sum over grid dims, all if None; every kept cell present unless no
        dim is summed.
        """

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
    followed by ``_term``, ``const`` on the grid dims. The operations run on
    the dataset with xarray, so coordinates the grid cannot hold, e.g.
    two-dimensional auxiliary ones, survive them; the grid is read off the
    dataset where a caller asks for it.
    """

    ds: Dataset
    model: Model

    @functools.cached_property
    def grid(self) -> Grid:
        from linopy.csr import Grid

        if lost := self._off_grid():
            raise ValueError(f"The grid cannot hold the coordinates {lost}.")
        return Grid.from_dataset(self.ds, self.dims)

    def _off_grid(self) -> list[str]:
        """The coordinates the grid cannot hold, e.g. two-dimensional ones."""
        from linopy.csr import _aux_coords

        own = set(self.dims)
        return sorted(
            set(map(str, self.ds.coords)) - own - set(_aux_coords(self.ds, own))
        )

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

    @classmethod
    def merged(
        cls,
        stores: Sequence[DenseStore],
        dim: str,
        join: JoinOptions | None,
        fill: float,
        kwargs: Mapping[str, Any],
    ) -> DenseStore:
        """The dense merge of ``stores`` along ``dim``, see :func:`merge_datasets`."""
        model = stores[0].model
        ds = merge_datasets([s.ds for s in stores], dim, join, fill, kwargs, True)
        return cls(expression_dataset(ds, model), model)

    def _derived(self, ds: Dataset) -> DenseStore:
        """A store on this model holding ``ds``, normalized."""
        return DenseStore(expression_dataset(ds, self.model), self.model)

    def _on_cells(self, values: float | np.ndarray) -> float | DataArray:
        """A flat per-cell array as a DataArray over the dims; a scalar as is."""
        if isinstance(values, np.ndarray) and values.ndim:
            return DataArray(values.reshape(self.shape), dims=self.dims)
        return float(values)

    def _term_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Coefficients and variable labels as ``(n_cells, nterm)`` arrays."""
        dims = (*self.dims, TERM_DIM)
        shape = (self.n_cells, self.nterm)
        coeffs = self.ds.coeffs.transpose(*dims).to_numpy().reshape(shape)
        return coeffs, self.ds.vars.transpose(*dims).to_numpy().reshape(shape)

    @classmethod
    def from_dense(cls, ds: Dataset, model: Model) -> DenseStore:
        return cls(ds, model)

    @classmethod
    def from_labels(cls, ds: Dataset, model: Model) -> DenseStore:
        labels, coeffs = xr.broadcast(ds.vars, ds.coeffs)
        const = xr.where(labels == -1, np.nan, 0.0)
        terms = Dataset({"vars": labels, "coeffs": coeffs})
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

    @classmethod
    def from_matmul(
        cls,
        source: CSRStore,
        matrix: scipy.sparse.csr_array,
        contracted_dims: Iterable[str],
        new_indexes: Iterable[pd.Index],
    ) -> DenseStore:
        from linopy.csr import CSRStore

        args = (matrix, contracted_dims, new_indexes)
        return CSRStore.from_matmul(source, *args).to_dense()

    def to_dense(self, reason: Reason | None = None) -> DenseStore:
        return self

    def to_sparse(self, reason: Reason | None = None) -> CSRStore | Reason:
        from linopy.csr import CSRStore, _sparsify_notice

        if lost := self._off_grid():
            return f"coordinates the Grid cannot hold: {', '.join(lost)}"
        if reason is not None:
            _sparsify_notice(reason, self.model)
        return CSRStore.from_dense(self.ds, self.model)

    def to_dataset(self) -> Dataset:
        return self.ds

    def copy(self, model: Model, deep: bool = True) -> DenseStore:
        return DenseStore(self.ds.copy(deep=deep), model)

    def dataarray(self, values: np.ndarray, name: str) -> DataArray:
        const = self.ds.const.transpose(*self.dims)
        return const.copy(data=values.reshape(const.shape)).rename(name)

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
        labels = self.ds.vars
        values = labels.copy(data=label_solution(labels.values)[labels.values])
        sol = (self.ds.coeffs * values).sum(TERM_DIM) + self.ds.const
        return sol.transpose(*self.dims).to_numpy().reshape(-1)

    def scale(
        self,
        factor: float | np.ndarray,
        op: Callable[[Any, Any], Any] = operator.mul,
    ) -> DenseStore:
        ds, per_cell = self.ds, self._on_cells(factor)
        coeffs, const = op(ds.coeffs, per_cell), op(ds.const, per_cell)
        return self._derived(
            absorb_absence(assign_fields(ds, coeffs=coeffs, const=const))
        )

    def assign_const(self, const: np.ndarray) -> DenseStore:
        per_cell = self._on_cells(const)
        return self._derived(absorb_absence(assign_fields(self.ds, const=per_cell)))

    def fillna(self, value: float) -> DenseStore:
        return self._derived(self.ds.fillna({"const": value}))

    def combine(
        self,
        const: DataArray,
        op: Callable[[Any, Any], Any],
        factor: DataArray | None = None,
        reindex: bool = False,
    ) -> DenseStore:
        ds = self.ds.reindex_like(const, fill_value=FILL_VALUE) if reindex else self.ds
        fields = {"const": const}
        if factor is not None:
            fields["coeffs"] = op(ds.coeffs, factor)
        return self._derived(absorb_absence(assign_fields(ds, **fields)))

    def absorbed(self) -> DenseStore:
        return self._derived(absorb_absence(self.ds))

    def prune(self) -> DenseStore:
        pruned = pruned_terms(self.ds)
        return self if pruned is self.ds else self._derived(pruned)

    def canonical(self) -> DenseStore:
        return self

    def with_grid(self, grid: Grid) -> DenseStore:
        coeffs, vars_ = self._term_arrays()
        return self.from_arrays(grid, coeffs, vars_, self.const, self.model)

    def take(self, rows: np.ndarray, grid: Grid) -> DenseStore:
        present = rows >= 0
        safe = np.where(present, rows, 0)
        coeffs, vars_ = self._term_arrays()
        const = np.where(present, self.const[safe], np.nan)
        coeffs = np.where(present[:, None], coeffs[safe], np.nan)
        vars_ = np.where(present[:, None], vars_[safe], -1)
        return self.from_arrays(grid, coeffs, vars_, const, self.model)

    def reindex(self, grid: Grid, fill: float = np.nan) -> DenseStore:
        fill_value = FILL_VALUE | {"const": fill}
        ds = self.ds.reindex(grid.indexes, fill_value=fill_value)
        return self._derived(ds.transpose(*grid.dims, TERM_DIM))

    def rename(self, names: Mapping[str, str]) -> DenseStore:
        return self._derived(self.ds.rename(names))

    def sum(self, dims: Iterable[str] | None = None) -> DenseStore:
        return self._derived(summed(self.ds, None if dims is None else list(dims)))

    def aggregate(self, grid: Grid, rows: np.ndarray) -> DenseStore:
        return as_sparse(self).aggregate(grid, rows).to_dense()

    def add(self, *others: DenseStore) -> DenseStore:
        return self.merged((self, *others), TERM_DIM, "override", 0.0, {})

    def concat(self, others: Iterable[DenseStore], dim: str, grid: Grid) -> DenseStore:
        return self.merged((self, *others), dim, "override", np.nan, {})


def as_sparse(store: Store) -> CSRStore:
    """``store`` as a sparse store, raising where it cannot convert."""
    res = store.to_sparse()
    if isinstance(res, str):
        raise ValueError(f"Cannot convert to a sparse store: {res}.")
    return res


def _stack_into_term_dim(ds: Dataset, dims: list[Hashable]) -> Dataset:
    """
    Stack ``dims`` into the term dimension.

    ``Dataset.stack`` reshapes with an inferred ``-1`` that numpy cannot
    resolve on zero-element data (https://github.com/PyPSA/linopy/issues/906),
    so an empty dataset is stacked onto a term dimension of size 0 instead.
    """
    if ds.vars.size:
        return ds.stack({TERM_DIM: dims}, create_index=False)

    for name, da in ds.data_vars.items():
        kept = [d for d in da.dims if d not in dims]
        shape = [da.sizes[d] for d in kept] + [0]
        ds[name] = DataArray(da.values.reshape(shape), dims=kept + [TERM_DIM])
    return ds


def _drop_coords_on_dims(ds: Dataset, dims: Iterable[Hashable]) -> Dataset:
    """
    Drop every coordinate touching the given dimensions.

    A coordinate on a dimension that is being reduced away must not survive
    the reduction: an auxiliary coordinate left in place would ride a
    subsequent stack onto the helper term dimension and break later
    alignment (https://github.com/PyPSA/linopy/issues/295).
    """
    dims = set(dims)
    return ds.drop_vars(k for k, c in ds.coords.items() if dims & set(c.dims))


def summed(data: Dataset, dim: DimsLike | None = None) -> Dataset:
    """
    Sum an expression dataset over ``dim``, all dims if None or ``...``: the
    terms of the summed cells stacked into the term dimension, the constant
    the NaN-skipping sum.
    """
    if isinstance(dim, str):
        dim = [dim]
    elif isinstance(dim, EllipsisType):
        dim = None

    if dim is None:
        vars = DataArray(data.vars.data.ravel(), dims=TERM_DIM)
        coeffs = DataArray(data.coeffs.data.ravel(), dims=TERM_DIM)
        const = data.const.sum()
        return Dataset({"vars": vars, "coeffs": coeffs, "const": const})

    dim = [d for d in dim if d != TERM_DIM]
    ds = _drop_coords_on_dims(data[["coeffs", "vars"]], dim).rename(
        {TERM_DIM: STACKED_TERM_DIM}
    )
    ds = _stack_into_term_dim(ds, [STACKED_TERM_DIM] + dim)
    return assign_fields(ds, const=data.const.sum(dim))


def pruned_terms(ds: Dataset) -> Dataset:
    """
    ``ds`` with the nonzero terms moved to the front of the term dimension
    and the all-zero tail cut off; ``ds`` itself if no coefficient is zero.
    """
    coeffs = ds.coeffs.transpose(..., TERM_DIM)
    vars = ds.vars.transpose(*coeffs.dims, ...)
    cdata = coeffs.data
    mask = cdata != 0
    if mask.all():
        return ds

    lead = cdata.shape[:-1]
    old_nterm = cdata.shape[-1]
    trailing = vars.shape[cdata.ndim :]
    mask = mask.reshape(-1, old_nterm)
    nrows = mask.shape[0]
    counts = mask.sum(1)
    nterm = int(counts.max()) if nrows else 0
    rows = np.repeat(np.arange(nrows), counts)
    pos = np.arange(rows.size) - np.repeat(np.cumsum(counts) - counts, counts)

    new_coeffs = np.zeros((nrows, nterm), dtype=cdata.dtype)
    new_coeffs[rows, pos] = cdata.reshape(nrows, old_nterm)[mask]
    new_vars = np.full((nrows, nterm, *trailing), -1, dtype=vars.dtype)
    new_vars[rows, pos] = vars.data.reshape(nrows, old_nterm, *trailing)[mask]

    new: dict[Hashable, Any] = {
        "coeffs": (coeffs.dims, new_coeffs.reshape(*lead, nterm)),
        "vars": (vars.dims, new_vars.reshape(*lead, nterm, *trailing)),
    }
    data_vars = {k: new.get(k, ds[k]) for k in ds.data_vars}
    return Dataset(data_vars, coords=ds.coords, attrs=ds.attrs)


def coord_sizes_agree(objs: Sequence[Any]) -> bool:
    """Whether all ``objs`` carry the same non-helper dimension sizes."""
    sizes = [{k: v for k, v in o.sizes.items() if k not in HELPER_DIMS} for o in objs]
    return check_common_keys_values(sizes)


def merge_datasets(
    data: Sequence[Dataset],
    dim: str,
    join: JoinOptions | None,
    fill: float,
    kwargs: Mapping[str, Any],
    linear: bool,
) -> Dataset:
    """
    The dense merge of expression datasets with ``xr.concat`` along ``dim``:
    along the term dimension the constants add up, along the factor
    dimension coefficients and constants multiply, along any other dimension
    the cells stack. The cells the join creates carry ``fill`` as constant;
    ``kwargs`` replace the default ``xr.concat`` arguments. An absent cell of
    any operand stays absent (§6).
    """
    data = [fill_missing_coords(ds, fill_helper_dims=True) for ds in data]

    if join is None:
        enforce_merge_dims(data, concat_dim=dim)
    enforce_aux_conflict(data, concat_dim=dim)

    if join is not None:
        override = join == "override"
    elif dim in HELPER_DIMS:
        override = coord_sizes_agree(data)
    else:
        override = False

    concat_kwargs = dict(kwargs)
    if not concat_kwargs:
        concat_kwargs = {"coords": "minimal", "compat": "override"}
        if linear:
            concat_kwargs["fill_value"] = FILL_VALUE
        if join is not None:
            concat_kwargs["join"] = join
        else:
            concat_kwargs["join"] = "override" if override else "outer"

    if dim == TERM_DIM:
        ds = xr.concat([d[["coeffs", "vars"]] for d in data], dim, **concat_kwargs)
        subkwargs = {**concat_kwargs, "fill_value": {"const": fill}}
        const = xr.concat([d[["const"]] for d in data], dim, **subkwargs)["const"]
        const = const.sum(TERM_DIM, skipna=False)
        ds = assign_fields(ds, const=const)
    elif dim == FACTOR_DIM:
        ds = xr.concat([d[["vars"]] for d in data], dim, **concat_kwargs)
        coeffs = xr.concat([d["coeffs"] for d in data], dim, **concat_kwargs)
        coeffs = coeffs.prod(FACTOR_DIM, skipna=False)
        const = xr.concat([d["const"] for d in data], dim, **concat_kwargs)
        const = const.prod(FACTOR_DIM, skipna=False)
        ds = assign_fields(ds, coeffs=coeffs, const=const)
    else:
        ds = xr.concat(data, dim, **concat_kwargs)

    for d in set(HELPER_DIMS) & set(ds.coords):
        ds = ds.reset_index(d, drop=True)

    return absorb_absence(ds)


def _merge_dims(store: Store, union: set[str]) -> list[str]:
    """
    The dims ``store`` brings to a merge over ``union``, in the order the
    dense merge lays them out: first appearance over its coordinates, a
    scalar one named after a dim of another operand standing for that dim.
    A sparse store's coordinates are taken in the order of its dense
    equivalent, which xarray's broadcast sets.
    """
    if isinstance(store, DenseStore):
        coords = store.ds.coords
    else:
        coords = xr.broadcast(store.grid.to_dataset())[0].coords
    dims = [str(d) for n, c in coords.items() for d in (c.dims or [n])]
    return [d for d in dims if d in union] + list(store.dims)


def merge_stores(
    stores: Sequence[Store],
    dim: str,
    join: JoinOptions | None,
    fill: float,
    kwargs: Mapping[str, Any],
) -> Store | Reason:
    """
    Merge the stores along ``dim``, the cells the join creates carrying
    ``fill`` as constant. Dense stores merge densely
    (:func:`merge_datasets`, ``kwargs`` passed to ``xr.concat``). Once one
    store is sparse, all are merged as sparse stores: along the term
    dimension as sparse matrix addition on the joined grid
    (:func:`align_stores`), along a grid dimension as a row stack
    (:func:`concat_stores`); the dense stores convert up through
    :meth:`Store.to_sparse`, with one sparsify notice for all. Returns the
    reason where the dense merge owns the semantics instead.
    """
    dense = [s for s in stores if isinstance(s, DenseStore)]
    if len(dense) == len(stores):
        return DenseStore.merged(dense, dim, join, fill, kwargs)
    if kwargs:
        return "merge(extra arguments)"
    union = {d for s in stores for d in s.dims}
    grid_dims = tuple(dict.fromkeys(d for s in stores for d in _merge_dims(s, union)))
    if dim != TERM_DIM and any(len(s.dims) != len(grid_dims) for s in stores):
        return "merge(operands over different dimensions)"
    if dim != TERM_DIM and dim not in grid_dims:
        return "merge(new dimension)"
    last = max(
        (i for i, s in enumerate(stores) if isinstance(s, DenseStore)), default=-1
    )
    csrs: list[CSRStore] = []
    for i, s in enumerate(stores):
        csr = s.to_sparse("merge with a sparse operand" if i == last else None)
        if isinstance(csr, str):
            return f"merge({csr})"
        csrs.append(csr)
    if dim != TERM_DIM:
        order = csrs[0].grid.dims
        reordered = [
            p if p.grid.dims == order else p.reindex(p.grid.select(order)) for p in csrs
        ]
        return concat_stores(reordered, dim, join)
    if not all(csrs[0].same_grid(p) for p in csrs[1:]):
        enforce_aux_conflict([Dataset(coords=p.grid.aux) for p in csrs])
        aligned = align_stores(csrs, grid_dims, join, fill)
        if isinstance(aligned, str):
            return aligned
        csrs = aligned
    return csrs[0].add(*csrs[1:])


def align_stores(
    stores: Sequence[CSRStore],
    dims: tuple[str, ...],
    join: JoinOptions | None,
    fill: float,
) -> list[CSRStore] | Reason:
    """
    Conform the stores to the grid on ``dims`` that ``join`` produces, per
    dim among the stores carrying it, each spread over the dims it lacks.
    The cells the join creates carry ``fill`` as constant. Returns the reason
    where the dense path owns the semantics instead: the auto-detected join
    on labels differing among operands that not all carry the dim,
    ``override`` on differing sizes, any join on non-unique labels. Labels
    differing where the dense path raises, ``exact`` or the auto-detected
    join on a dim all operands carry, raise here too.
    """
    from linopy.csr import Grid

    metadata = [p.grid.to_dataset() for p in stores]
    if join is None:
        enforce_merge_dims(metadata, concat_dim=TERM_DIM)
    if join == "exact":
        xr.align(*metadata, join="exact")
    if any(not p.grid.is_unique for p in stores):
        return "merge(non-unique labels)"
    indexes: dict[str, pd.Index] = {}
    for d in dims:
        own = [p.grid.indexes[d] for p in stores if d in p.grid.indexes]
        if join is None and not all(own[0].equals(i) for i in own):
            return "merge(differing grids with join=None)"
        if join == "override" and len({len(i) for i in own}) > 1:
            return 'merge(join="override" with differently shaped grids)'
        if join in ("outer", "inner"):
            combine = pd.Index.union if join == "outer" else pd.Index.intersection
            indexes[d] = pd.Index(functools.reduce(combine, own), name=d)
        else:
            indexes[d] = own[-1] if join == "right" else own[0]
    grid = Grid(indexes)
    aligned = []
    for p in stores:
        target = grid.select(p.grid.dims)
        if join == "override":
            p = p.with_grid(p.grid.with_indexes(target.indexes))
        elif not p.grid.same_layout(target):
            p = p.reindex(target, fill)
        aligned.append(p if p.grid.dims == grid.dims else p.broadcast(grid))
    return aligned


def concat_stores(
    stores: Sequence[CSRStore], dim: str, join: JoinOptions | None
) -> CSRStore | Reason:
    """
    Stack stores sharing one dim order along the grid dim ``dim``, in order.
    The result's coordinates are what the dense ``xr.concat`` yields on the
    grids alone: labels and auxiliary coordinates concatenated along
    ``dim``, the other dims joined as ``join`` says (``outer`` by default,
    after the §8 label check for the auto-detected join), with the cells the
    join creates absent. Returns the reason where the dense path owns the
    semantics instead: a join over non-unique labels.
    """
    from linopy.csr import Grid

    dims = stores[0].grid.dims
    metadata = [p.grid.to_dataset() for p in stores]
    if join is None:
        enforce_merge_dims(metadata, concat_dim=dim)
    enforce_aux_conflict(metadata, concat_dim=dim)
    combined = xr.concat(
        metadata, dim, join=join or "outer", coords="minimal", compat="override"
    )
    grid = Grid.from_dataset(combined, dims)
    aligned = []
    for p in stores:
        target = grid.with_indexes({dim: p.grid.indexes[dim]})
        if join == "override" or p.grid.same_layout(target):
            aligned.append(p.with_grid(target))
        elif p.grid.is_unique:
            aligned.append(p.reindex(target))
        else:
            return "merge(non-unique labels)"
    return aligned[0].concat(aligned[1:], dim, grid)
