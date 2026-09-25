"""
The sparse backing of a LinearExpression: ``A @ x + c`` in CSR form.

Under v1 semantics, ``expr.groupby(g).sum(sparse=True)`` (or
``linopy.options["sparse_groupby"]``) returns an ordinary
:class:`~linopy.expressions.LinearExpression` backed by a
:class:`CSRLinearExpression`: same public type, different backing, akin to
dask-backed xarray objects. The CSR form is canonical (duplicate variables
summed, terms label-ordered) and ragged along ``_term``, with no fixed term
count per row; grouping, ``sum``, ``merge``/``+``/``-``, scaling and
``@``/``dot`` (:meth:`contracted`) are sparse linear algebra. Zero policy: the
structural operations (grouping and ``sum`` via :meth:`aggregated`, merge via
:meth:`added`, scaling, reindexing) go through COO and keep explicit zero
coefficients; only the product with a constant matrix, ``@``/``dot``, prunes
them. Either way cell activeness is carried by ``const`` alone, independent of
term layout.
Any operation without a sparse branch expands the expression through
``.data`` to the mathematically identical dense rectangle in canonical term
layout; this is valid because v1 semantics do not fix the term layout.

This module documents the CSR structure only, working on plain datasets. The
one bridge back to a dense type is :meth:`CSRLinearExpression.to_dense`, which
wraps the expanded dataset in a :class:`~linopy.expressions.LinearExpression`.
The reverse bridges live at the dense call sites, in
:class:`~linopy.expressions.LinearExpression` and
:meth:`linopy.constraints.CSRConstraint.from_csr`.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import pandas as pd
import scipy.sparse
from xarray import DataArray, Dataset

from linopy.common import coords_from_dataset, coords_to_dataset_vars
from linopy.config import options
from linopy.constants import HELPER_DIMS, TERM_DIM, PerformanceWarning
from linopy.semantics import (
    absorb_absence,
    enforce_aux_conflict,
    warn_outside_linopy,
)

if TYPE_CHECKING:
    from linopy.expressions import LinearExpression
    from linopy.model import Model

CONTRACTION_CHUNK = 64
"""Kept-axis block size of the chunked Kronecker product in ``contracted``."""

AuxCoords: TypeAlias = dict[str, tuple[str | tuple[()], np.ndarray]]
"""Auxiliary coordinates as ``name -> (grid dim, values)``, dim ``()`` for a scalar."""

_AUX_PREFIX = "_aux"
_SCALAR_DIM = "_scalar"


@dataclass(frozen=True, eq=False)
class Grid:
    """
    The flat coordinate grid a CSR row layout is defined on.

    ``indexes`` maps each grid dimension to its labels, ordered as the
    dimensions are, so row ``i`` of a CSR matrix is cell ``i`` of the
    C-order flattening of that grid. ``aux`` holds the auxiliary
    coordinates as ``name -> (grid dim, values)``, e.g. the key levels of a
    grouped result kept stacked over the observed key combinations; a
    scalar coordinate, e.g. left by a scalar selection, has dim ``()``.
    """

    indexes: dict[str, pd.Index]
    aux: AuxCoords = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "aux", {n: (d, _readonly(v)) for n, (d, v) in self.aux.items()}
        )

    @classmethod
    def from_coords(cls, coords: Iterable[pd.Index]) -> Grid:
        """Build from one index per dimension, each named after its dim."""
        return cls({str(c.name): c for c in coords})

    @classmethod
    def from_dataset(cls, ds: Dataset | DataArray, dims: Iterable[str]) -> Grid:
        """Build from the indexes and auxiliary coordinates ``ds`` carries on ``dims``."""
        dims = tuple(dims)
        indexes = {d: ds.get_index(d).rename(d) for d in dims}
        return cls(indexes, _aux_coords(ds, set(dims)))

    @classmethod
    def from_netcdf_vars(cls, ds: Dataset, dims: Iterable[str]) -> Grid:
        """Read back a grid written by :meth:`to_netcdf_vars`."""
        aux: AuxCoords = {
            da.attrs["name"]: (da.attrs.get("dim", ()), da.to_numpy())
            for k, da in ds.data_vars.items()
            if str(k).startswith(_AUX_PREFIX)
        }
        return cls(cls.from_coords(coords_from_dataset(ds, list(dims))).indexes, aux)

    def to_netcdf_vars(self) -> dict[str, DataArray]:
        """
        The indexes and auxiliary coordinates as plain data variables for
        netcdf, named by position with the coordinate names as attributes.
        """
        aux = {
            f"{_AUX_PREFIX}{j}": DataArray(
                v,
                dims=[f"{_AUX_PREFIX}dim{j}"] if isinstance(d, str) else [],
                attrs={"name": n} | ({"dim": d} if isinstance(d, str) else {}),
            )
            for j, (n, (d, v)) in enumerate(self.aux.items())
        }
        return coords_to_dataset_vars(self.coords) | aux

    @property
    def dims(self) -> tuple[str, ...]:
        return tuple(self.indexes)

    @property
    def coords(self) -> list[pd.Index]:
        return list(self.indexes.values())

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(i) for i in self.indexes.values())

    def to_dataset(self) -> Dataset:
        """The grid's indexes and auxiliary coordinates as a coordinate-only Dataset."""
        return Dataset(coords=self.indexes | self.aux)

    @property
    def size(self) -> int:
        """Number of flat cells; one for the zero-dimensional grid."""
        shape = self.shape
        return int(np.prod(shape, dtype=np.int64)) if shape else 1

    @property
    def strides(self) -> dict[str, int]:
        """C-order row stride per dimension."""
        shape = self.shape
        return {
            d: int(np.prod(shape[i + 1 :], dtype=np.int64))
            for i, d in enumerate(self.dims)
        }

    @property
    def is_unique(self) -> bool:
        """Whether every dimension's labels are unique."""
        return all(i.is_unique for i in self.indexes.values())

    def indexer(self, other: Grid) -> tuple[np.ndarray, np.ndarray]:
        """
        Map ``other``'s cells onto this grid: the flat target row of each
        source cell, and a mask of the cells whose labels all survive.
        """
        strides = self.strides
        positions = {
            d: self.indexes[d].get_indexer(i) for d, i in other.indexes.items()
        }
        valid = _outer_sum([pos == -1 for pos in positions.values()]) == 0
        row_map = _outer_sum([pos * strides[d] for d, pos in positions.items()])
        return row_map, valid

    def renamed(self, names: Mapping[str, str]) -> Grid:
        """Relabel dimensions; the cell layout is unchanged."""
        indexes = {
            names.get(d, d): i.rename(names.get(d, d)) for d, i in self.indexes.items()
        }
        aux = {
            n: (names.get(d, d) if isinstance(d, str) else d, v)
            for n, (d, v) in self.aux.items()
        }
        return Grid(indexes, aux)

    def reordered(self, dims: Iterable[str]) -> Grid:
        """
        Select and order the given dimensions; labels unchanged, auxiliary
        coordinates on dropped dimensions dropped, scalar ones kept.
        """
        dims = tuple(dims)
        aux = {
            n: (d, v)
            for n, (d, v) in self.aux.items()
            if not isinstance(d, str) or d in dims
        }
        return Grid({d: self.indexes[d] for d in dims}, aux)

    def with_indexes(self, indexers: Mapping[Any, Any]) -> Grid:
        """
        Replace the labels of the named dimensions; the rest, and the
        auxiliary coordinates, unchanged.
        """
        indexes = {
            d: pd.Index(indexers[d], name=d) if d in indexers else i
            for d, i in self.indexes.items()
        }
        return replace(self, indexes=indexes)

    def conformed(self, target: Grid) -> Grid:
        """``target`` carrying this grid's auxiliary coordinates, reindexed onto its labels."""
        aux = dict(self.aux)
        for name, (d, values) in self.aux.items():
            if isinstance(d, str):
                series = pd.Series(values, index=self.indexes[d])
                aux[name] = (d, series.reindex(target.indexes[d]).to_numpy())
        return Grid(target.indexes, aux)

    def combined(self, others: Iterable[Grid], how: str) -> Grid:
        """
        Join with ``others`` along shared dimensions: per dimension the union
        (``how="outer"``) or intersection (``how="inner"``) of labels, kept in
        this grid's dimension order. Auxiliary coordinates are not carried.
        """
        combine = pd.Index.union if how == "outer" else pd.Index.intersection
        others = list(others)
        indexes = {}
        for d in self.dims:
            index = self.indexes[d]
            for other in others:
                index = combine(index, other.indexes[d])
            indexes[d] = pd.Index(index, name=d)
        return Grid(indexes)

    def same_layout(self, other: Grid) -> bool:
        """Whether both grids have the same dims and labels, auxiliary coordinates aside."""
        return self.dims == other.dims and all(
            i.equals(other.indexes[d]) for d, i in self.indexes.items()
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        if not self.same_layout(other) or self.aux.keys() != other.aux.keys():
            return False
        for name, (d, values) in self.aux.items():
            other_d, other_values = other.aux[name]
            if d != other_d:
                return False
            if not pd.Index(np.atleast_1d(values)).equals(
                pd.Index(np.atleast_1d(other_values))
            ):
                return False
        return True

    def dataarray(self, values: np.ndarray, name: str | None = None) -> DataArray:
        """Wrap one value per flat cell as a DataArray on the grid's coordinates."""
        coords = self.to_dataset().coords
        return DataArray(
            values.reshape(self.shape), coords=coords, dims=self.dims, name=name
        )


@dataclass(frozen=True)
class CSRLinearExpression:
    """
    An expression as ``A @ x + c`` over a fixed coordinate grid.

    ``csr`` has one row per flat cell of ``grid`` and one column per raw
    variable label — label columns stay valid when variables are added to
    the model later; realization maps them to dense positions. ``const`` is
    the per-cell constant, NaN for an absent cell. Auxiliary coordinates
    live on the ``grid``.
    """

    csr: scipy.sparse.csr_array
    const: np.ndarray
    grid: Grid
    model: Model

    def __post_init__(self) -> None:
        csr = self.csr
        dtype = index_dtype(csr.nnz, csr.shape, self.model)
        if csr.indices.dtype != dtype or csr.indptr.dtype != dtype:
            indices, indptr = csr.indices.astype(dtype), csr.indptr.astype(dtype)
            csr = scipy.sparse.csr_array((csr.data, indices, indptr), shape=csr.shape)
            object.__setattr__(self, "csr", csr)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.grid.shape

    @property
    def n_cells(self) -> int:
        return self.csr.shape[0]

    @property
    def nterm(self) -> int:
        return csr_nterm(self.csr)

    def cell(self, indices: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Coefficients, label-ordered variable labels and constant of the grid
        cell at ``indices``. An absent cell returns empty coefficient and
        label arrays.
        """
        row = int(np.ravel_multi_index(indices, self.grid.shape)) if indices else 0
        const = float(self.const[row])
        start, end = self.csr.indptr[row], self.csr.indptr[row + 1]
        if np.isnan(const):
            end = start
        vars_, coeffs = self.csr.indices[start:end], self.csr.data[start:end]
        order = np.argsort(vars_, kind="stable")
        return coeffs[order], vars_[order], const

    @classmethod
    def from_grouper(
        cls,
        source: Dataset | CSRLinearExpression,
        model: Model,
        grouper: pd.Series | pd.DataFrame,
        group_dim: str,
        stacked: bool,
        coord_dims: tuple[str, ...],
    ) -> CSRLinearExpression:
        """
        Build the grouped sum in CSR form.

        The grouper is conformed to the expression's member index by label
        (upstream alignment checks guarantee equal label sets) and group
        labels are sorted. A DataFrame grouper (one column per key) yields
        one grid dim per key -- the cartesian grid, absent combinations being
        empty cells -- or, ``stacked``, a single ``group_dim`` over the
        observed key combinations only, the key values attached as auxiliary
        coordinates. The new dims take the member dim's slot in
        ``coord_dims``. ``source`` is a dense expression dataset or an
        already CSR-backed expression, which is regrouped through
        :meth:`aggregated`.
        """
        ds = source if isinstance(source, Dataset) else source.grid.to_dataset()
        member_dim = str(grouper.index.name)
        if member_dim in ds.indexes:
            grouper = grouper.reindex(ds.indexes[member_dim])
        elif len(grouper) != ds.sizes[member_dim]:
            raise ValueError(f"grouper length does not match dimension {member_dim!r}")
        if grouper.isna().to_numpy().any():
            raise ValueError(
                "Cannot group by a pandas object containing NaN values. "
                "Drop or fill the corresponding entries before grouping."
            )
        frame = grouper if isinstance(grouper, pd.DataFrame) else grouper.to_frame()
        keys = [str(k) for k in frame.columns]
        scatter_codes: dict[str, np.ndarray] = {}
        indexes: dict[str, pd.Index] = {}
        aux: AuxCoords = {}
        if len(keys) == 1:
            codes, uniques = pd.factorize(frame.iloc[:, 0], sort=True)
            scatter_codes[group_dim] = codes
            indexes[group_dim] = pd.Index(uniques, name=group_dim)
        elif stacked:
            codes, uniques = pd.factorize(pd.MultiIndex.from_frame(frame), sort=True)
            scatter_codes[group_dim] = codes
            indexes[group_dim] = pd.RangeIndex(len(uniques), name=group_dim)
            aux = {
                k: (group_dim, uniques.get_level_values(i).to_numpy())
                for i, k in enumerate(keys)
            }
        else:
            for col, k in zip(frame.columns, keys):
                codes, uniques = pd.factorize(frame[col], sort=True)
                scatter_codes[k] = codes
                indexes[k] = pd.Index(uniques, name=k)
        new_dims = tuple(scatter_codes)
        grid_dims = tuple(
            d for dim in coord_dims for d in (new_dims if dim == member_dim else (dim,))
        )
        for d in coord_dims:
            if d != member_dim:
                indexes[d] = ds.get_index(d).rename(d)
        aux |= _aux_coords(ds, set(coord_dims) - {member_dim})
        grid = Grid({d: indexes[d] for d in grid_dims}, aux)
        if isinstance(source, Dataset):
            return cls._from_scatter(ds, model, grid, member_dim, scatter_codes, True)
        member_rows = _member_rows(grid, scatter_codes, ds.sizes[member_dim])
        rows = _cell_rows(grid, member_rows, member_dim, source.grid.dims)
        return source.aggregated(grid, rows)

    @classmethod
    def from_dense(cls, ds: Dataset, model: Model) -> CSRLinearExpression:
        """Convert a dense expression to CSR form on its own coordinate grid."""
        grid_dims = tuple(str(d) for d in ds.coeffs.dims if d not in HELPER_DIMS)
        grid = Grid.from_dataset(ds, grid_dims)
        if not grid_dims:
            scalar = ds.expand_dims(_SCALAR_DIM)
            return cls._from_scatter(scalar, model, grid, _SCALAR_DIM, {}, False)
        first = grid_dims[0]
        codes = {first: np.arange(len(grid.indexes[first]))}
        return cls._from_scatter(ds, model, grid, first, codes, False)

    @classmethod
    def _from_scatter(
        cls,
        ds: Dataset,
        model: Model,
        grid: Grid,
        member_dim: str,
        scatter_codes: dict[str, np.ndarray],
        skipna: bool,
    ) -> CSRLinearExpression:
        """
        Scatter an expression's terms into grid rows (conceptually ``G @ A``):
        ``member_dim`` lands in the contiguous block of grid dims named by
        ``scatter_codes`` (one row-position array per dim), every other grid
        dim maps one-to-one, and the COO to CSR conversion sums duplicate
        variables, giving the group sum. Cells no member lands in stay absent
        (NaN const). With ``skipna``, NaN member constants count as 0 in the
        sum; without it, a NaN member constant propagates and leaves the cell
        absent (NaN const).
        """
        grid_dims = grid.dims
        slot = min((grid_dims.index(d) for d in scatter_codes), default=0)
        transposed = [d for d in grid_dims if d not in scatter_codes]
        transposed.insert(slot, member_dim)
        member_rows = _member_rows(grid, scatter_codes, ds.sizes[member_dim])
        cell_rows = _cell_rows(grid, member_rows, member_dim, transposed)

        coeffs = ds.coeffs.transpose(*transposed, TERM_DIM).to_numpy().reshape(-1)
        vars_ = ds.vars.transpose(*transposed, TERM_DIM).to_numpy().reshape(-1)
        rows = np.repeat(cell_rows, ds.sizes[TERM_DIM])
        keep = (vars_ != -1) & ~np.isnan(coeffs)

        full_size = grid.size
        csr = coo_to_csr(
            coeffs[keep], rows[keep], vars_[keep], (full_size, model._xCounter), model
        )

        const_vals = ds.const.transpose(*transposed).to_numpy().reshape(-1)
        if skipna:
            const_vals = np.where(np.isnan(const_vals), 0.0, const_vals)
        const = np.full(full_size, np.nan)
        const[cell_rows] = 0.0
        np.add.at(const, cell_rows, const_vals)

        return cls(csr, const, grid, model)

    def aggregated(self, grid: Grid, rows: np.ndarray) -> CSRLinearExpression:
        """
        Sum source rows into the cells of ``grid``, row ``i`` landing in cell
        ``rows[i]`` (conceptually ``G @ A``). Goes through COO, so duplicate
        variables are summed and explicit zeros kept, as by :meth:`added`. The
        constant is the NaN-skipping sum of its rows' constants (NaN counts as
        0); cells no row lands in are absent. Auxiliary coordinates are
        ``grid``'s.
        """
        coo = self.csr.tocoo()
        shape = (grid.size, self.csr.shape[1])
        rows_ = rows[coo.coords[0]]
        csr = coo_to_csr(coo.data, rows_, coo.coords[1], shape, self.model)
        weights = np.nan_to_num(self.const)
        const = np.bincount(rows, weights=weights, minlength=grid.size).astype(float)
        const[np.bincount(rows, minlength=grid.size) == 0] = np.nan
        return replace(self, csr=csr, const=const, grid=grid)

    def summed(self, dims: Iterable[str]) -> CSRLinearExpression:
        """
        Sum over grid dimensions. The kept dims stay in grid order with their
        auxiliary coordinates, and every kept cell is present, its constant
        the NaN-skipping sum of its members.
        """
        dims = set(dims)
        grid = self.grid.reordered(d for d in self.grid.dims if d not in dims)
        stride = grid.strides
        rows = _outer_sum(
            [
                np.zeros(n, dtype=np.int64) if d in dims else np.arange(n) * stride[d]
                for d, n in zip(self.grid.dims, self.grid.shape)
            ]
        )
        return self.aggregated(grid, rows).filled(0.0)

    def live_terms(self) -> np.ndarray:
        """
        Mask over the stored terms: the cell is present and the coefficient
        is nonzero.
        """
        present = np.repeat(~np.isnan(self.const), np.diff(self.csr.indptr))
        return present & (self.csr.data != 0)

    def pruned(self) -> CSRLinearExpression:
        """Drop explicit zero coefficients; cell activeness stays with ``const``."""
        csr = self.csr.copy()
        csr.eliminate_zeros()
        return replace(self, csr=csr)

    def scaled(
        self,
        factor: float | np.ndarray,
        op: Callable[[Any, Any], Any] = operator.mul,
    ) -> CSRLinearExpression:
        """
        Apply ``op`` with ``factor`` -- a scalar or one value per cell -- to
        every coefficient and to the constant. Explicit zeros are kept, as by
        :meth:`added`.
        """
        factor = np.broadcast_to(factor, (self.n_cells,))
        per_term = np.repeat(factor, np.diff(self.csr.indptr))
        csr = scipy.sparse.csr_array(
            (op(self.csr.data, per_term), self.csr.indices, self.csr.indptr),
            shape=self.csr.shape,
        )
        return replace(self, csr=csr).with_const(op(self.const, factor))

    def with_const(self, const: np.ndarray) -> CSRLinearExpression:
        """
        Replace the per-cell constant, leaving the terms alone. A cell made
        absent (NaN) has its terms dropped: an absent cell carries no terms.
        """
        absent = np.isnan(const)
        counts = np.diff(self.csr.indptr)
        if not counts[absent].any():
            return replace(self, const=const)
        keep = np.repeat(~absent, counts)
        indptr = np.concatenate([[0], np.cumsum(np.where(absent, 0, counts))])
        csr = scipy.sparse.csr_array(
            (self.csr.data[keep], self.csr.indices[keep], indptr),
            shape=self.csr.shape,
        )
        return replace(self, csr=csr, const=const)

    def taken(self, rows: np.ndarray, grid: Grid) -> CSRLinearExpression:
        """
        Gather rows into the cells of ``grid``, cell ``i`` taking row
        ``rows[i]`` with its terms, explicit zeros included, and its constant;
        ``-1`` leaves the cell absent. Auxiliary coordinates are ``grid``'s.
        """
        present = rows >= 0
        const = np.where(present, self.const[rows], np.nan)
        csr = self.csr[np.where(present, rows, 0)]
        return replace(self, csr=csr, grid=grid).with_const(const)

    def reindexed(self, grid: Grid, fill: float = np.nan) -> CSRLinearExpression:
        """
        Remap rows onto a new grid, possibly in a new dim order: dropped
        labels vanish, new labels get ``fill`` as their constant (NaN: absent
        cells). Auxiliary coordinates follow the rows; those of ``grid`` are
        ignored.
        """
        row_map, valid = grid.indexer(self.grid)

        coo = self.csr.tocoo()
        keep = valid[coo.coords[0]]
        rows = row_map[coo.coords[0][keep]]
        cols = coo.coords[1][keep]
        n_cells = grid.size
        shape = (n_cells, self.csr.shape[1])
        csr = coo_to_csr(coo.data[keep], rows, cols, shape, self.model)
        const = np.full(n_cells, fill)
        const[row_map[valid]] = self.const[valid]
        return replace(
            self,
            csr=csr,
            const=const,
            grid=self.grid.conformed(grid),
        )

    def filled(self, value: float) -> CSRLinearExpression:
        """Resolve absent cells (NaN const) to a constant; terms untouched."""
        const = np.where(np.isnan(self.const), value, self.const)
        return replace(self, const=const)

    def renamed(self, names: dict[str, str]) -> CSRLinearExpression:
        """Relabel grid dims; the CSR row layout is unchanged."""
        return replace(self, grid=self.grid.renamed(names))

    def same_grid(self, other: CSRLinearExpression) -> bool:
        """Whether both live on the same cells, auxiliary coordinates aside."""
        return self.grid.same_layout(other.grid)

    def added(self, other: CSRLinearExpression) -> CSRLinearExpression:
        """
        Sparse matrix addition == merge along the term dimension. Goes through
        COO so explicit zero coefficients survive (scipy's ``+`` drops them),
        keeping a cell with only zero-coefficient terms distinguishable from
        an empty cell. A cell absent in either operand is absent in the sum
        and carries no terms. Auxiliary coordinates propagate and conflicting
        ones raise (§11).
        """
        const = self.const + other.const
        a, b = self.csr.tocoo(), other.csr.tocoo()
        shape = (self.n_cells, max(a.shape[1], b.shape[1]))
        rows = np.concatenate([a.coords[0], b.coords[0]])
        cols = np.concatenate([a.coords[1], b.coords[1]])
        data = np.concatenate([a.data, b.data])
        present = ~np.isnan(const)[rows]
        csr = coo_to_csr(data[present], rows[present], cols[present], shape, self.model)
        enforce_aux_conflict([Dataset(coords=p.grid.aux) for p in (self, other)])
        grid = replace(self.grid, aux=other.grid.aux | self.grid.aux)
        return replace(self, csr=csr, const=const, grid=grid)

    def contracted(
        self,
        matrix: scipy.sparse.csr_array,
        contracted_dims: Iterable[str],
        new_indexes: Iterable[pd.Index],
    ) -> CSRLinearExpression:
        """
        Contract grid dimensions against a sparse constant (``expr @ C``).

        ``matrix`` is the constant flattened to
        ``(prod(contracted shape), prod(new shape))`` in C order over
        ``contracted_dims``, which are given in grid order. Each entry of
        ``new_indexes`` must be named after the dim it creates, which is read
        off its ``name``. The result lives on the kept grid dims followed by
        ``new_indexes`` and is
        ``kron(I_kept, matrix.T) @ csr``, evaluated in chunks of the kept axis
        so the operator never grows with the kept size.

        The result is in compact canonical form: duplicate variables summed,
        terms label-ordered and explicit zeros pruned -- unlike :meth:`added`,
        the sparse product drops them, so cell activeness is carried by
        ``const`` alone. Auxiliary coordinates on kept dims propagate, those
        on contracted dims drop.
        """
        contracted_dims = tuple(contracted_dims)
        kept = tuple(d for d in self.grid.dims if d not in contracted_dims)
        target = kept + contracted_dims
        source = (
            self
            if self.grid.dims == target
            else self.reindexed(self.grid.reordered(target))
        )

        n_contracted = matrix.shape[0]
        kept_grid = source.grid.reordered(kept)
        n_kept = kept_grid.size
        const = np.nan_to_num(source.const)
        chunk = min(CONTRACTION_CHUNK, n_kept)
        operator = scipy.sparse.kron(
            scipy.sparse.eye_array(chunk), matrix.T, format="csr"
        )

        blocks = []
        const_blocks = []
        for start in range(0, n_kept, chunk):
            size = min(chunk, n_kept - start)
            block = (
                operator
                if size == chunk
                else scipy.sparse.kron(
                    scipy.sparse.eye_array(size), matrix.T, format="csr"
                )
            )
            rows = slice(start * n_contracted, (start + size) * n_contracted)
            blocks.append(block @ source.csr[rows])
            const_blocks.append(block @ const[rows])

        indexes = kept_grid.indexes | {str(i.name): i for i in new_indexes}
        return replace(
            source,
            csr=scipy.sparse.csr_array(scipy.sparse.vstack(blocks, format="csr")),
            const=np.concatenate(const_blocks),
            grid=Grid(indexes, kept_grid.aux),
        )

    def to_dense(self) -> LinearExpression:
        """
        Expand to the dense equivalent in canonical form: terms label-ordered,
        duplicates summed, padded to the widest cell with the usual fill.
        Absent cells (NaN const) carry no terms. The expanded dataset is
        wrapped in a :class:`LinearExpression`.
        """
        from linopy.expressions import LinearExpression

        csr = self.csr.copy()
        csr.sort_indices()
        nterm = self.nterm
        vars_flat, coeffs_flat = csr_to_term_arrays(
            csr, nterm, self.model._dtypes["labels"]
        )

        shape = self.grid.shape
        dims = (*self.grid.dims, TERM_DIM)
        ds = Dataset(
            {
                "coeffs": (dims, coeffs_flat.reshape(*shape, nterm)),
                "vars": (dims, vars_flat.reshape(*shape, nterm)),
                "const": (self.grid.dims, self.const.reshape(shape)),
            },
            coords=self.grid.indexes | self.grid.aux,
        )
        return LinearExpression(absorb_absence(ds), self.model)


def index_dtype(nnz: int, shape: tuple[int, ...], model: Model) -> np.dtype:
    """
    Index dtype of a CSR store: the model's label dtype, widened to int64
    only when the nonzeros or the shape outgrow it.
    """
    dtype = np.dtype(model._dtypes["labels"])
    return dtype if max(nnz, *shape) <= np.iinfo(dtype).max else np.dtype(np.int64)


def coo_to_csr(
    data: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    shape: tuple[int, int],
    model: Model,
) -> scipy.sparse.csr_array:
    """Build a CSR store from COO triplets in the model's index dtype."""
    dtype = index_dtype(len(data), shape, model)
    coords = (rows.astype(dtype, copy=False), cols.astype(dtype, copy=False))
    return scipy.sparse.csr_array(scipy.sparse.coo_array((data, coords), shape=shape))


def csr_nterm(csr: scipy.sparse.csr_array) -> int:
    """Widest CSR row, floored at one term."""
    return max(int(np.diff(csr.indptr).max(initial=0)), 1)


def csr_to_term_arrays(
    csr: scipy.sparse.csr_array,
    nterm: int,
    label_dtype: Any,
    n_rows: int | None = None,
    row_positions: np.ndarray | None = None,
    coeff_dtype: Any = float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand CSR rows into padded ``(n_rows, nterm)`` term arrays.

    Returns variable labels (absent terms filled with ``-1``) and coefficients
    (absent terms filled with NaN). ``row_positions`` places CSR row ``i`` at
    output row ``row_positions[i]`` of an ``n_rows``-row output, for a CSR that
    holds only the active rows of a larger grid.
    """
    counts = np.diff(csr.indptr)
    n_rows = csr.shape[0] if n_rows is None else n_rows
    vars_ = np.full((n_rows, nterm), -1, dtype=label_dtype)
    coeffs = np.full((n_rows, nterm), np.nan, dtype=coeff_dtype)
    if csr.nnz:
        positions = np.arange(csr.shape[0]) if row_positions is None else row_positions
        rows = np.repeat(positions, counts)
        cols = np.arange(csr.nnz) - np.repeat(csr.indptr[:-1], counts)
        vars_[rows, cols] = csr.indices
        coeffs[rows, cols] = csr.data
    return vars_, coeffs


def _densify_notice(reason: str) -> None:
    """
    Emit a :class:`~linopy.constants.PerformanceWarning` naming why a sparse
    (CSR) backing is dropped, if ``options["warn_on_densify"]`` is set.
    """
    if options["warn_on_densify"]:
        warn_outside_linopy(
            f"Sparse (CSR) backing densified: {reason}.", PerformanceWarning
        )


def _readonly(values: np.ndarray) -> np.ndarray:
    """``values`` as a read-only array, copied first if it is writable."""
    if not values.flags.writeable:
        return values
    values = values.copy()
    values.setflags(write=False)
    return values


def _aux_coords(ds: Dataset | DataArray, dims: set[str]) -> AuxCoords:
    """Scalar and one-dimensional auxiliary coordinates of ``ds`` lying on ``dims``."""
    aux: AuxCoords = {}
    for n, c in ds.coords.items():
        if n in ds.xindexes:
            continue
        if c.ndim == 0:
            aux[str(n)] = ((), c.to_numpy())
        elif c.ndim == 1 and str(c.dims[0]) in dims:
            aux[str(n)] = (str(c.dims[0]), c.to_numpy())
    return aux


def _member_rows(
    grid: Grid, scatter_codes: dict[str, np.ndarray], n_members: int
) -> np.ndarray:
    """Row offset in ``grid`` of each member, from its per-dim group codes."""
    stride = grid.strides
    member_rows = np.zeros(n_members, dtype=np.int64)
    for d, codes in scatter_codes.items():
        member_rows += codes * stride[d]
    return member_rows


def _cell_rows(
    grid: Grid, member_rows: np.ndarray, member_dim: str, dims: Iterable[str]
) -> np.ndarray:
    """
    Grid row of every cell over ``dims`` in C order, ``member_dim`` landing on
    ``member_rows`` and every other dim mapping one-to-one onto ``grid``.
    """
    stride = grid.strides
    return _outer_sum(
        [
            member_rows
            if d == member_dim
            else np.arange(len(grid.indexes[d])) * stride[d]
            for d in dims
        ]
    )


def _outer_sum(axis_positions: list[np.ndarray]) -> np.ndarray:
    """Outer sum of per-axis offsets, flattened in C order."""
    cells = np.zeros((), dtype=np.int64)
    for pos in axis_positions:
        cells = cells[..., None] + pos
    return cells.reshape(-1)
