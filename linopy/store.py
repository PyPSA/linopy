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
from typing import TYPE_CHECKING, Any, Self, TypeAlias, TypeVar, cast, final

import numpy as np
import pandas as pd
import scipy.sparse
import xarray as xr
from xarray import DataArray, Dataset

from linopy.alignment import as_dataarray, fill_missing_coords
from linopy.common import assign_fields, check_common_keys_values
from linopy.constants import FACTOR_DIM, HELPER_DIMS, STACKED_TERM_DIM, TERM_DIM
from linopy.semantics import (
    absorb_absence,
    check_arithmetic_nan,
    check_user_nan,
    enforce_aux_conflict,
    enforce_merge_dims,
    enforce_no_multiindex,
    is_nan_scalar,
)
from linopy.types import CONSTANT_TYPES, DimsLike

if TYPE_CHECKING:
    from xarray.core.types import JoinOptions

    from linopy.common import VariableLabelIndex
    from linopy.csr import AuxCoords, CSRStore, Grid
    from linopy.model import Model

Reason: TypeAlias = str
"""Why a store refuses an operation, e.g. ``"reindex(method=...)"``."""

FILL_VALUE = {"vars": -1, "coeffs": np.nan, "const": np.nan}
"""What a dense expression dataset holds at a cell or term it lacks."""

STORE_KIND_ATTR = "_linopy_store_kind"
CSR_COLUMNS_ATTR = "_linopy_n_columns"
NETCDF_STORE_ATTRS = frozenset((STORE_KIND_ATTR, CSR_COLUMNS_ATTR))
"""The netcdf attributes a store writes next to its arrays."""


class Store(ABC):
    """
    Terms and constant of a linear expression on a fixed coordinate grid.

    A method returning ``Self | Reason`` may refuse a case with the reason;
    :class:`DenseStore` does so only in :meth:`to_sparse`.
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

    @abstractmethod
    def to_netcdf_vars(self) -> Dataset:
        """
        The arrays for netcdf with ``const`` on the grid dims and the store
        kind in the ``_linopy_store_kind`` attribute, read by :meth:`from_netcdf_vars`.
        """

    @classmethod
    @abstractmethod
    def read_netcdf_vars(cls, ds: Dataset, model: Model) -> Self:
        """Rebuild this store kind from the arrays of :meth:`to_netcdf_vars`."""

    @staticmethod
    def from_netcdf_vars(ds: Dataset, model: Model) -> Store:
        """The store written by :meth:`to_netcdf_vars`, by its ``_linopy_store_kind``."""
        from linopy.csr import CSRStore

        kinds: dict[str, type[Store]] = {
            "dense": DenseStore,
            "csr": CSRStore,
        }
        kind = ds.attrs.get(STORE_KIND_ATTR)
        if kind not in kinds:
            raise ValueError(
                f"Unknown or missing {STORE_KIND_ATTR!r} attribute {kind!r} in "
                f"dataset with variables {list(ds.data_vars)}; expected one of {sorted(kinds)}."
            )
        return kinds[kind].read_netcdf_vars(ds, model)

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
    def variable_labels(self) -> np.ndarray:
        """
        Variable labels of all stored terms, zero coefficients included, as
        :meth:`positional_csr` reads them; ``-1`` may mark an empty slot.
        """

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
        constant; a NaN it produces in a live cell raises.
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
        coefficients too, a NaN it produces at a live term raising.
        """

    @abstractmethod
    def absorbed(self) -> Self:
        """Drop the terms of absent cells."""

    @abstractmethod
    def prune(self, tol: float = 0.0) -> Self:
        """Drop the terms with a coefficient of at most ``tol`` in magnitude."""

    @abstractmethod
    def canonical(self) -> Self:
        """The canonical form a constraint is built from."""

    @abstractmethod
    def nnz(self, active_rows: np.ndarray) -> int:
        """
        Upper bound on the nonzeros :meth:`positional_csr` returns for the
        cells selected by the boolean ``active_rows``.
        """

    @abstractmethod
    def positional_csr(
        self, label_index: VariableLabelIndex, active_rows: np.ndarray, name: str
    ) -> scipy.sparse.csr_array:
        """
        The cells selected by the boolean ``active_rows`` as CSR rows over the
        dense variable positions of ``label_index``, raising for labels of
        removed variables with the constraint ``name``.
        """

    @abstractmethod
    def with_grid(self, grid: Grid) -> Self:
        """The same cells on ``grid``, a grid of the same shape."""

    @abstractmethod
    def take(self, rows: np.ndarray, grid: Grid) -> Self:
        """
        Gather cell ``rows[i]`` into cell ``i`` of ``grid``, ``-1`` leaving it
        absent without terms.
        """

    def _take_positions(self, positions: DataArray) -> Self:
        from linopy.csr import Grid

        dims = tuple(map(str, positions.dims))
        return self.take(positions.values.ravel(), Grid.from_dataset(positions, dims))

    def isel(self, indexers: Mapping[Hashable, slice]) -> Self:
        """The cells at the positional slices ``indexers`` of grid dims."""
        grid = self.grid
        return self._take_positions(grid.dataarray(np.arange(grid.size)).isel(indexers))

    def transpose(self, dims: Sequence[str]) -> Self:
        """The cells on the grid dims reordered to ``dims``."""
        if tuple(dims) == self.dims:
            return self
        grid = self.grid
        return self._take_positions(
            grid.dataarray(np.arange(grid.size)).transpose(*dims)
        )

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


_FIELDS = ("const", "coeffs", "vars")
_SCALARS = (int, float, np.number)


_Laid = TypeVar("_Laid", xr.Variable, DataArray)


def transposed(variable: _Laid, dims: tuple[Hashable, ...]) -> _Laid:
    """``variable`` on ``dims``, itself when already laid out so."""
    return variable if variable.dims == dims else variable.transpose(*dims)


def coords_within(value: DataArray, like: DataArray | Dataset) -> bool:
    """Whether every coordinate of ``value`` equals the one ``like`` holds under its name."""
    held = like.coords.variables
    return all(
        k in held and v.equals(held[k]) for k, v in value.coords.variables.items()
    )


def on_coords(
    fields: Mapping[str, xr.Variable],
    coords: xr.Coordinates,
    attrs: Mapping[str, Any] | None = None,
) -> Dataset:
    """
    A dataset holding ``fields`` on ``coords`` and ``attrs``. ``fields`` are
    taken as lying on ``coords``: nothing is aligned, and the indexes of
    ``coords`` are reused.
    """
    return Dataset(dict(fields), coords=coords, attrs=dict(attrs) if attrs else None)


def on_cells(
    value: Any, dims: tuple[Hashable, ...], like: DataArray | Dataset
) -> float | np.ndarray | None:
    """
    ``value`` as a number or an array over ``dims``: a numeric scalar, or a
    DataArray lying on a subset of ``dims`` with the coordinates of ``like``;
    else None. A NaN in it raises (§5).
    """
    if isinstance(value, _SCALARS):
        if is_nan_scalar(value):
            check_user_nan()
        return float(value)
    if not isinstance(value, DataArray):
        return None
    values = laid_on(value, dims, like)
    if values is not None and values.dtype.kind == "f" and np.isnan(values).any():
        check_user_nan()
    return values


def laid_on(
    value: DataArray, dims: tuple[Hashable, ...], like: DataArray | Dataset
) -> np.ndarray | None:
    """The DataArray case of :func:`on_cells` with a NaN kept, for a caller that reads NaN as its own."""
    if value.dtype.kind not in "biuf" or not set(value.dims) <= set(dims):
        return None
    enforce_no_multiindex(value, context="The constant")
    sizes = like.sizes
    if any(value.sizes[d] != sizes[d] for d in value.dims):
        return None
    if not coords_within(value, like):
        return None
    ordered = tuple(d for d in dims if d in value.dims)
    values = transposed(value.variable, ordered).values
    shape = tuple(sizes[d] if d in value.dims else 1 for d in dims)
    return np.broadcast_to(values.reshape(shape), tuple(sizes[d] for d in dims))


def _is_expression_dataset(data: Dataset, model: Model) -> bool:
    """Whether ``data`` already has the layout :func:`expression_dataset` makes."""
    if set(data.data_vars) != set(_FIELDS):
        return False
    variables = data.variables
    const, coeffs, vars_ = variables["const"], variables["coeffs"], variables["vars"]
    if (
        vars_.dtype != model._dtypes["labels"]
        or coeffs.dtype.kind != "f"
        or const.dtype.kind != "f"
    ):
        return False
    grid = tuple(d for d in data.dims if d not in HELPER_DIMS)
    factor = (FACTOR_DIM,) if FACTOR_DIM in data.dims else ()
    if (
        const.dims != grid
        or coeffs.dims != (*grid, TERM_DIM)
        or vars_.dims != (*grid, *factor, TERM_DIM)
    ):
        return False
    coords = data.coords
    return all(d in coords for d in grid) and not any(d in coords for d in HELPER_DIMS)


def expression_dataset(data: Dataset | Any | None, model: Model) -> Dataset:
    """Normalize ``data`` to the dataset layout of an expression."""
    if isinstance(data, Dataset) and _is_expression_dataset(data, model):
        if tuple(data.data_vars) != _FIELDS:
            data = data[list(_FIELDS)]
        if data.attrs.get("name", 0) is None:
            return data
        return data.assign_attrs(name=None)
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
        return tuple(str(d) for d in self.ds.variables["coeffs"].dims if d != TERM_DIM)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.ds.sizes[d] for d in self.dims)

    @property
    def const(self) -> np.ndarray:
        return transposed(self.ds.variables["const"], self.dims).values.reshape(-1)

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
        return cls.normalized(ds, model)

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
        ds = merge_datasets([s.ds for s in stores], dim, join, fill, kwargs, True)
        return cls.normalized(ds, stores[0].model)

    @classmethod
    def normalized(cls, ds: Dataset, model: Model) -> DenseStore:
        """A store on ``model`` holding ``ds`` in the expression layout."""
        return cls(expression_dataset(ds, model), model)

    def subtracted(self, other: DenseStore) -> DenseStore | None:
        """``self - other`` where the two stack along the term dimension as :func:`merge_datasets` would, else None."""
        data = [self.ds, other.ds]
        if not (_overriding(data, TERM_DIM, None) and _stackable(data)):
            return None
        return self._derived(absorb_absence(_stacked_terms(data, [False, True])))

    def _derived(self, ds: Dataset) -> DenseStore:
        """A store on this model holding ``ds``, normalized."""
        return self.normalized(ds, self.model)

    def _own(self, **fields: xr.Variable) -> Dataset:
        """This store's dataset with ``fields`` replaced, on its own coordinates."""
        ds = self.ds
        held = {str(k): ds.variables[k] for k in ds.data_vars}
        return on_coords(held | fields, ds.coords, ds.attrs)

    def _laid_fields(self) -> tuple[xr.Variable, xr.Variable, xr.Variable]:
        """``const``, ``coeffs`` and ``vars``, the terms on the dims of the constant."""
        variables = self.ds.variables
        const = variables["const"]
        dims = (*const.dims, TERM_DIM)
        coeffs, vars_ = variables["coeffs"], variables["vars"]
        return const, transposed(coeffs, dims), transposed(vars_, dims)

    def _term_views(self) -> tuple[np.ndarray, np.ndarray]:
        """Coefficients and variable labels on ``(*dims, _term)``, uncopied."""
        dims = (*self.dims, TERM_DIM)
        variables = self.ds.variables
        return (
            transposed(variables["coeffs"], dims).values,
            transposed(variables["vars"], dims).values,
        )

    def _term_rows(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Coefficients and variable labels of the cells ``rows``, ``(len(rows), nterm)``."""
        index = np.unravel_index(rows, (1, *self.shape))
        coeffs, vars_ = self._term_views()
        return coeffs[None][index], vars_[None][index]

    def _term_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Coefficients and variable labels as ``(n_cells, nterm)`` arrays."""
        shape = (self.n_cells, self.nterm)
        coeffs, vars_ = self._term_views()
        return coeffs.reshape(shape), vars_.reshape(shape)

    @classmethod
    def from_dense(cls, ds: Dataset, model: Model) -> DenseStore:
        return cls(ds, model)

    @classmethod
    def from_labels(cls, ds: Dataset, model: Model) -> DenseStore:
        terms = ds[["vars", "coeffs"]].expand_dims(TERM_DIM, -1)
        absent = ds.vars == -1
        if absent.any():
            const = xr.where(absent, np.nan, 0.0)
            terms = terms.assign(coeffs=terms.coeffs.where(~absent), const=const)
        return cls.normalized(terms, model)

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

    def to_netcdf_vars(self) -> Dataset:
        ds = Dataset({k: self.ds[k] for k in ("coeffs", "vars", "const")})
        return ds.assign_attrs({STORE_KIND_ATTR: "dense"})

    @classmethod
    def read_netcdf_vars(cls, ds: Dataset, model: Model) -> DenseStore:
        return cls(Dataset({k: ds[k] for k in ("coeffs", "vars", "const")}), model)

    def dataarray(self, values: np.ndarray, name: str) -> DataArray:
        const = transposed(self.ds.const, self.dims)
        return const.copy(deep=False, data=values.reshape(const.shape)).rename(name)

    def has_terms(self) -> np.ndarray:
        vars_ = self._term_views()[1]
        if vars_.shape[-1] > 8:
            return (vars_ != -1).any(axis=-1).reshape(-1)
        live = np.zeros(vars_.shape[:-1], dtype=bool)
        for i in range(vars_.shape[-1]):
            live |= vars_[..., i] != -1
        return live.reshape(-1)

    def cell(self, indices: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray, float]:
        ds = self.ds
        coeffs, vars_ = ds.coeffs.values[indices], ds.vars.values[indices]
        return coeffs, vars_, float(ds.const.values[indices])

    def terms(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coeffs, vars_, keep = self._active_mask(~np.isnan(self.const))
        cells = np.flatnonzero(keep)
        cells //= self.nterm
        return cells, vars_[keep], coeffs[keep]

    def variable_labels(self) -> np.ndarray:
        return self.ds.vars.values.ravel()

    def solution(
        self, label_solution: Callable[[np.ndarray], np.ndarray]
    ) -> np.ndarray:
        labels = self.ds.vars
        values = labels.copy(
            deep=False, data=label_solution(labels.values)[labels.values]
        )
        sol = (self.ds.coeffs * values).sum(TERM_DIM) + self.ds.const
        return sol.transpose(*self.dims).to_numpy().reshape(-1)

    def _fits(self, field: str, value: DataArray) -> bool:
        """Whether ``value`` lies on this store's own layout of ``field``: same dims, sizes and labels, no coordinate of its own."""
        own = self.ds.variables[field]
        if value.dims != own.dims or value.shape != own.shape:
            return False
        return coords_within(value, self.ds)

    def on_cells(self, value: Any) -> float | np.ndarray | None:
        """:func:`on_cells` on this store's grid."""
        return on_cells(value, self.ds.variables["const"].dims, self.ds)

    def combined(
        self, operand: float | np.ndarray, op: Callable[[Any, Any], Any], scale: bool
    ) -> DenseStore:
        """The store with ``operand``, laid out by :meth:`on_cells`, combined into the constant by ``op``, and into the coefficients too if ``scale``."""
        const, coeffs, vars_ = self._laid_fields()
        with np.errstate(all="ignore"):
            result = op(const.values, operand)
            check_arithmetic_nan(result, op, operand, const.values)
            fields = {"const": xr.Variable(const.dims, result)}
            if scale:
                per_term = (
                    operand[..., None] if isinstance(operand, np.ndarray) else operand
                )
                scaled = op(coeffs.values, per_term)
                check_arithmetic_nan(scaled, op, per_term, labels=vars_.values)
                fields |= {"coeffs": xr.Variable(coeffs.dims, scaled), "vars": vars_}
        return self._derived(absorb_absence(self._own(**fields)))

    def moved(self, rhs: float | np.ndarray) -> tuple[DenseStore, DataArray]:
        """``rhs``, laid out by :meth:`on_cells`, moved onto this side: the store with a zero constant, and the constant the other side keeps."""
        store = self.absorbed()
        const = store.ds.variables["const"]
        with np.errstate(all="ignore"):
            remainder = const.values - rhs
        check_arithmetic_nan(remainder, operator.sub, rhs, const.values)
        zero = xr.Variable(const.dims, np.broadcast_to(0.0, const.shape))
        kept = DataArray(xr.Variable(const.dims, -remainder), coords=store.ds.coords)
        return store._derived(store._own(const=zero)), kept

    def _assigned(self, **fields: DataArray) -> DenseStore:
        """The store with ``fields`` replaced, on its own coordinates where they fit, else through :func:`assign_fields`."""
        if all(self._fits(k, v) for k, v in fields.items()):
            ds = self._own(**{k: v.variable for k, v in fields.items()})
        else:
            ds = assign_fields(self.ds, **fields)
        return self._derived(absorb_absence(ds))

    def scale(
        self,
        factor: float | np.ndarray,
        op: Callable[[Any, Any], Any] = operator.mul,
    ) -> DenseStore:
        if isinstance(factor, np.ndarray) and factor.ndim:
            return self.combined(factor.reshape(self.shape), op, True)
        return self.combined(float(factor), op, True)

    def assign_const(self, const: np.ndarray) -> DenseStore:
        if np.ndim(const) == 0:
            const = np.full(self.n_cells, float(const))
        return self._assigned(const=self.dataarray(const, "const"))

    def fillna(self, value: float) -> DenseStore:
        return self._assigned(const=self.ds.const.fillna(value))

    def combine(
        self,
        const: DataArray,
        op: Callable[[Any, Any], Any],
        factor: DataArray | None = None,
        reindex: bool = False,
    ) -> DenseStore:
        store = self
        if reindex:
            ds = self.ds.reindex_like(const, fill_value=FILL_VALUE)
            store = DenseStore(ds, self.model)
        fields = {"const": const}
        if factor is not None:
            fields["coeffs"] = op(store.ds.coeffs, factor)
            check_arithmetic_nan(fields["coeffs"], op, factor, labels=store.ds.vars)
        return store._assigned(**fields)

    def masked(self, keep: float | np.ndarray) -> DenseStore:
        """The store with every cell ``keep`` is False at made absent, ``keep`` laid out by :meth:`on_cells`."""
        const, coeffs, vars_ = self._laid_fields()
        per_term = np.asarray(keep)[..., None]
        fields = {
            "const": xr.Variable(const.dims, np.where(keep, const.values, np.nan)),
            "coeffs": xr.Variable(
                coeffs.dims, np.where(per_term, coeffs.values, np.nan)
            ),
            "vars": xr.Variable(vars_.dims, np.where(per_term, vars_.values, -1)),
        }
        return self._derived(self._own(**fields))

    def absorbed(self) -> DenseStore:
        return self._derived(absorb_absence(self.ds))

    def prune(self, tol: float = 0.0) -> DenseStore:
        pruned = pruned_terms(self.ds, tol)
        return self if pruned is self.ds else self._derived(pruned)

    def canonical(self) -> DenseStore:
        return self

    def _active_mask(
        self, active_rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        coeffs, vars_ = self._term_views()
        valid = vars_ != -1
        valid &= coeffs != 0
        valid &= active_rows.reshape(valid.shape[:-1])[..., None]
        return coeffs, vars_, valid

    def nnz(self, active_rows: np.ndarray) -> int:
        return int(np.count_nonzero(self._active_mask(active_rows)[2]))

    def active_terms(
        self, active_rows: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Coefficients, raw variable labels and row pointers of the live terms
        of the cells selected by the boolean ``active_rows``, in CSR layout.
        """
        from linopy.csr import index_dtype

        coeffs, vars_, valid = self._active_mask(active_rows)
        labels, data = vars_[valid], coeffs[valid]
        counts = valid.sum(axis=-1).reshape(-1)[active_rows]
        indptr = np.empty(
            len(counts) + 1, index_dtype(len(data), counts.shape, self.model)
        )
        indptr[0] = 0
        np.cumsum(counts, out=indptr[1:])
        return data, labels, indptr

    def positional_csr(
        self, label_index: VariableLabelIndex, active_rows: np.ndarray, name: str
    ) -> scipy.sparse.csr_array:
        from linopy.csr import _csr_from_label_columns

        data, labels, indptr = self.active_terms(active_rows)
        csr = _csr_from_label_columns(
            data, labels, indptr, len(indptr) - 1, label_index, name
        )
        csr.sum_duplicates()
        return csr

    def with_grid(self, grid: Grid) -> DenseStore:
        coeffs, vars_ = self._term_arrays()
        return self.from_arrays(grid, coeffs, vars_, self.const, self.model)

    def take(self, rows: np.ndarray, grid: Grid) -> DenseStore:
        present = rows >= 0
        safe = np.where(present, rows, 0)
        coeffs, vars_ = self._term_rows(safe)
        const = np.where(present, self.const[safe], np.nan)
        coeffs[~present] = np.nan
        vars_[~present] = -1
        return self.from_arrays(grid, coeffs, vars_, const, self.model)

    def isel(self, indexers: Mapping[Hashable, slice]) -> DenseStore:
        return DenseStore(self.ds.isel(indexers), self.model)

    def transpose(self, dims: Sequence[str]) -> DenseStore:
        return DenseStore(self.ds.transpose(*dims, TERM_DIM), self.model)

    def reindex(self, grid: Grid, fill: float = np.nan) -> DenseStore:
        fill_value = FILL_VALUE | {"const": fill}
        ds = self.ds.reindex(grid.indexes, fill_value=fill_value)
        return self._derived(ds.transpose(*grid.dims, TERM_DIM))

    def rename(self, names: Mapping[str, str]) -> DenseStore:
        return self._derived(self.ds.rename(names))

    def sum(self, dims: Iterable[str] | None = None) -> DenseStore:
        if dims is None:
            return self._derived(summed(self.ds, None))
        over = [d for d in dims if d != TERM_DIM]
        if not over:
            return self._derived(summed(self.ds, over))
        ds = self.ds
        variables = ds.variables
        const = variables["const"]
        kept = [d for d in self.dims if d not in over]
        axes = tuple(const.dims.index(d) for d in over)
        rows = tuple(const.sizes[d] for d in kept)
        nterm = (
            self.nterm * math.prod(const.sizes[d] for d in over) if const.size else 0
        )

        sizes = {**const.sizes, TERM_DIM: self.nterm}
        layout = {d: sizes[d] for d in (*kept, TERM_DIM, *over)}

        def stacked(field: str) -> xr.Variable:
            values = variables[field].set_dims(layout).values
            return xr.Variable((*kept, TERM_DIM), values.reshape(*rows, nterm))

        gone = set(over)
        names = [k for k, v in ds.coords.variables.items() if not gone & set(v.dims)]
        coords = xr.Coordinates(
            {k: ds.coords.variables[k] for k in names},
            indexes={k: i for k, i in ds.xindexes.items() if k in names},
        )
        fields = {
            "const": xr.Variable(kept, np.nansum(const.values, axis=axes)),
            "coeffs": stacked("coeffs"),
            "vars": stacked("vars"),
        }
        return self._derived(on_coords(fields, coords, ds.attrs))

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


def pruned_terms(ds: Dataset, tol: float = 0.0) -> Dataset:
    """
    ``ds`` with the terms of a coefficient above ``tol`` in magnitude moved
    to the front of the term dimension and the tail cut off; ``ds`` itself
    if there is none to drop.
    """
    coeffs = ds.coeffs.transpose(..., TERM_DIM)
    vars = ds.vars.transpose(*coeffs.dims, ...)
    cdata = coeffs.data
    mask = ~(np.abs(cdata) <= tol)
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
    override = _overriding(data, dim, join)
    if dim == TERM_DIM and override and not kwargs and _stackable(data):
        return absorb_absence(_stacked_terms(data))

    data = [fill_missing_coords(ds, fill_helper_dims=True) for ds in data]

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


def _overriding(data: Sequence[Dataset], dim: str, join: JoinOptions | None) -> bool:
    """
    Whether a merge of ``data`` along ``dim`` under ``join`` overrides onto
    the coordinates of the first, its operands checked for a dimension
    mismatch and a conflicting auxiliary coordinate.
    """
    if join is None:
        enforce_merge_dims(data, concat_dim=dim)
    enforce_aux_conflict(data, concat_dim=dim)
    if join is not None:
        return join == "override"
    return dim in HELPER_DIMS and coord_sizes_agree(data)


def _stackable(data: Sequence[Dataset]) -> bool:
    """
    Whether every dataset holds exactly the three fields in the linear
    layout with just its dimension coordinates, every dimension at one size
    throughout, so a merge along the term dimension that overrides onto the
    first coordinates is a stack of terms broadcast over the union of dims.
    """
    sizes: dict[Hashable, int] = {}
    for ds in data:
        if set(ds.data_vars) != set(_FIELDS):
            return False
        variables = ds.variables
        grid = variables["const"].dims
        terms = (*grid, TERM_DIM)
        if (
            set(ds.coords) != set(grid)
            or variables["coeffs"].dims != terms
            or variables["vars"].dims != terms
        ):
            return False
        for d, n in zip(grid, variables["const"].shape):
            if sizes.setdefault(d, n) != n:
                return False
    return True


def _stacked(parts: Sequence[np.ndarray], negate: Sequence[bool]) -> np.ndarray:
    """
    ``parts`` stacked along their last axis, a part ``negate`` flags
    negated, a two-term part copied term by term, where that runs faster.
    """
    width = sum(part.shape[-1] for part in parts)
    out = np.empty((*parts[0].shape[:-1], width), np.result_type(*parts))
    j = 0
    for part, flip in zip(parts, negate):
        nterm = part.shape[-1]
        slabs = [part[..., i : i + 1] for i in range(nterm)] if nterm == 2 else [part]
        for slab in slabs:
            target = out[..., j : j + slab.shape[-1]]
            if flip:
                np.negative(slab, out=target)
            else:
                np.copyto(target, slab)
            j += slab.shape[-1]
    return out


def _stacked_terms(
    data: Sequence[Dataset], negate: Sequence[bool] | None = None
) -> Dataset:
    """
    The datasets' terms broadcast over the union of their dims, in order of
    first appearance, stacked along the term dimension with their constants
    summed, on the coordinates of the first dataset carrying each dim. A
    dataset ``negate`` flags enters subtracted.
    """
    negate = [False] * len(data) if negate is None else negate
    consts = [ds.variables["const"] for ds in data]
    sizes = {d: n for c in consts for d, n in zip(c.dims, c.shape)}
    grid = tuple(sizes)
    held = {d: ds for ds in reversed(data) for d in ds.coords}
    coords = xr.Coordinates(
        {d: held[d].coords.variables[d] for d in grid if d in held},
        indexes={d: held[d].xindexes[d] for d in grid if d in held},
    )

    def on_grid(ds: Dataset, field: str) -> np.ndarray:
        variable = ds.variables[field]
        dims = (
            {**sizes, TERM_DIM: variable.sizes[TERM_DIM]} if field != "const" else sizes
        )
        return variable.set_dims(dims).values

    coeffs = _stacked([on_grid(ds, "coeffs") for ds in data], negate)
    vars_ = _stacked([on_grid(ds, "vars") for ds in data], [False] * len(data))
    parts = [on_grid(ds, "const") for ds in data]
    const = parts[0]
    for part, flip in zip(parts[1:], negate[1:]):
        const = const - part if flip else const + part
    fields = {
        "const": xr.Variable(grid, const),
        "coeffs": xr.Variable((*grid, TERM_DIM), coeffs),
        "vars": xr.Variable((*grid, TERM_DIM), vars_),
    }
    return on_coords(fields, coords, data[0].attrs)


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
    :meth:`Store.to_sparse`, with one sparsify notice once the sparse merge
    succeeds. Returns the reason where the dense merge owns the semantics
    instead.
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
    csrs: list[CSRStore] = []
    for s in stores:
        csr = s.to_sparse()
        if isinstance(csr, str):
            return f"merge({csr})"
        csrs.append(csr)
    res = _merged_sparse(csrs, dim, join, fill, grid_dims)
    if dense and not isinstance(res, str):
        from linopy.csr import _sparsify_notice

        _sparsify_notice("merge with a sparse operand", stores[0].model)
    return res


def _merged_sparse(
    csrs: list[CSRStore],
    dim: str,
    join: JoinOptions | None,
    fill: float,
    grid_dims: tuple[str, ...],
) -> CSRStore | Reason:
    """The sparse merge of :func:`merge_stores` on the converted stores."""
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
