"""
The sparse backing of a LinearExpression: ``A @ x + c`` in CSR form.

``expr.groupby(g).sum(sparse=True)`` (or ``linopy.options["sparse_groupby"]``
under v1) returns an ordinary :class:`~linopy.expressions.LinearExpression`
backed by a :class:`CSRLinearExpression` instead of the dense dataset — same
public type, different backing, akin to dask-backed xarray objects. The CSR
form is canonical (duplicate variables summed, terms label-ordered) and ragged
along ``_term``, so the group-size padding of issue #745 has no analog;
grouping, ``merge``/``+``/``-`` and scaling become sparse linear algebra.
Anything without a sparse branch expands through ``.data`` to the
mathematically identical dense rectangle in canonical term layout — the reason
the feature is v1-gated, where term layout is non-contractual.

This module documents the CSR structure only: it works on plain datasets and
knows nothing about the dense types. The bridges live at the dense call sites,
in :class:`~linopy.expressions.LinearExpression` and
:meth:`linopy.constraints.CSRConstraint.from_csr`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.sparse
from xarray import Dataset

from linopy.constants import HELPER_DIMS, TERM_DIM
from linopy.semantics import absorb_absence, enforce_aux_conflict

if TYPE_CHECKING:
    from linopy.model import Model


@dataclass(frozen=True, eq=False)
class Grid:
    """
    The flat coordinate grid a CSR row layout is defined on.

    ``indexes`` maps each grid dimension to its labels, ordered as the
    dimensions are, so row ``i`` of a CSR matrix is cell ``i`` of the
    C-order flattening of that grid.
    """

    indexes: dict[str, pd.Index]

    @classmethod
    def from_coords(cls, coords: Iterable[pd.Index]) -> Grid:
        """Build from one index per dimension, each named after its dim."""
        return cls({str(c.name): c for c in coords})

    @classmethod
    def from_dataset(cls, ds: Dataset, dims: Iterable[str]) -> Grid:
        """Build from the indexes ``ds`` carries on ``dims``."""
        return cls({d: ds.get_index(d).rename(d) for d in dims})

    @property
    def dims(self) -> tuple[str, ...]:
        return tuple(self.indexes)

    @property
    def coords(self) -> list[pd.Index]:
        return list(self.indexes.values())

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(i) for i in self.indexes.values())

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
        return Grid(
            {
                names.get(d, d): i.rename(names.get(d, d))
                for d, i in self.indexes.items()
            }
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        return self.dims == other.dims and all(
            i.equals(other.indexes[d]) for d, i in self.indexes.items()
        )


@dataclass(frozen=True)
class CSRLinearExpression:
    """
    An expression as ``A @ x + c`` over a fixed coordinate grid.

    ``csr`` has one row per flat cell of ``grid`` and one column per raw
    variable label — label columns stay valid when variables are added to
    the model later; realization maps them to dense positions. ``const`` is
    the per-cell constant, NaN for an absent cell. ``coords`` holds auxiliary
    coordinates as ``name -> (grid dim, values)``, e.g. the key levels of a
    grouped result kept stacked over the observed key combinations.
    """

    csr: scipy.sparse.csr_array
    const: np.ndarray
    grid: Grid
    model: Model
    coords: dict[str, tuple[str, np.ndarray]] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.grid.shape

    @property
    def n_cells(self) -> int:
        return self.csr.shape[0]

    @property
    def nterm(self) -> int:
        return csr_nterm(self.csr)

    @classmethod
    def from_grouper(
        cls,
        ds: Dataset,
        model: Model,
        grouper: pd.Series | pd.DataFrame,
        group_dim: str,
        stacked: bool,
        coord_dims: tuple[str, ...],
    ) -> CSRLinearExpression:
        """
        Build the grouped sum directly in CSR form (no padded rectangle).

        The grouper is conformed to the expression's member index by label
        (upstream alignment checks guarantee equal label sets) and group
        labels are sorted, matching the dense kernel's output grid. A
        DataFrame grouper (one column per key) yields one grid dim per key
        -- the cartesian grid, absent combinations being empty cells -- or,
        ``stacked``, a single ``group_dim`` over the observed key combinations
        only, the key values attached as auxiliary coordinates. The new dims
        take the member dim's slot in ``coord_dims``, as on the dense path.
        """
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
        coords: dict[str, tuple[str, np.ndarray]] = {}
        if len(keys) == 1:
            codes, uniques = pd.factorize(frame.iloc[:, 0], sort=True)
            scatter_codes[group_dim] = codes
            indexes[group_dim] = pd.Index(uniques, name=group_dim)
        elif stacked:
            codes, uniques = pd.factorize(pd.MultiIndex.from_frame(frame), sort=True)
            scatter_codes[group_dim] = codes
            indexes[group_dim] = pd.RangeIndex(len(uniques), name=group_dim)
            coords = {
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
        coords |= _aux_coords(ds, set(coord_dims) - {member_dim})
        grid = Grid({d: indexes[d] for d in grid_dims})
        return cls._from_scatter(
            ds, model, grid, member_dim, scatter_codes, True, coords
        )

    @classmethod
    def from_dense(cls, ds: Dataset, model: Model) -> CSRLinearExpression:
        """Convert a dense expression to CSR form on its own coordinate grid."""
        grid_dims = tuple(str(d) for d in ds.coeffs.dims if d not in HELPER_DIMS)
        grid = Grid.from_dataset(ds, grid_dims)
        first = grid_dims[0]
        codes = {first: np.arange(len(grid.indexes[first]))}
        coords = _aux_coords(ds, set(grid_dims))
        return cls._from_scatter(ds, model, grid, first, codes, False, coords)

    @classmethod
    def _from_scatter(
        cls,
        ds: Dataset,
        model: Model,
        grid: Grid,
        member_dim: str,
        scatter_codes: dict[str, np.ndarray],
        skipna: bool,
        coords: dict[str, tuple[str, np.ndarray]],
    ) -> CSRLinearExpression:
        """
        Scatter an expression's terms into grid rows (conceptually ``G @ A``):
        ``member_dim`` lands in the contiguous block of grid dims named by
        ``scatter_codes`` (one row-position array per dim), every other grid
        dim maps one-to-one, and the COO→CSR conversion sums duplicates --
        which is the group sum. Cells no member lands in stay absent (NaN
        const). With ``skipna`` the constant is reduced as by the dense group
        kernel (NaN members count as 0); without it an absent cell (NaN const)
        stays absent, as on the dense v1 merge path.
        """
        grid_dims = grid.dims
        stride = grid.strides

        slot = min(grid_dims.index(d) for d in scatter_codes)
        transposed = [d for d in grid_dims if d not in scatter_codes]
        transposed.insert(slot, member_dim)
        member_rows = np.zeros(ds.sizes[member_dim], dtype=np.int64)
        for d, codes in scatter_codes.items():
            member_rows += codes * stride[d]
        cell_rows = _outer_sum(
            [
                member_rows
                if d == member_dim
                else np.arange(len(grid.indexes[d])) * stride[d]
                for d in transposed
            ]
        )

        coeffs = ds.coeffs.transpose(*transposed, TERM_DIM).to_numpy().reshape(-1)
        vars_ = ds.vars.transpose(*transposed, TERM_DIM).to_numpy().reshape(-1)
        rows = np.repeat(cell_rows, ds.sizes[TERM_DIM])
        keep = (vars_ != -1) & ~np.isnan(coeffs)

        full_size = grid.size
        coo = scipy.sparse.coo_array(
            (coeffs[keep], (rows[keep], vars_[keep])),
            shape=(full_size, model._xCounter),
        )

        const_vals = ds.const.transpose(*transposed).to_numpy().reshape(-1)
        if skipna:
            const_vals = np.where(np.isnan(const_vals), 0.0, const_vals)
        const = np.full(full_size, np.nan)
        const[cell_rows] = 0.0
        np.add.at(const, cell_rows, const_vals)

        return cls(scipy.sparse.csr_array(coo), const, grid, model, coords)

    def scaled(self, factor: float) -> CSRLinearExpression:
        return replace(self, csr=self.csr * factor, const=self.const * factor)

    def reindexed(self, grid: Grid, fill: float = np.nan) -> CSRLinearExpression:
        """
        Remap rows onto a new grid, possibly in a new dim order, without the
        dense rectangle: dropped labels vanish, new labels get ``fill`` as
        their constant (NaN: absent cells).
        """
        row_map, valid = grid.indexer(self.grid)

        coo = self.csr.tocoo()
        keep = valid[coo.coords[0]]
        rows = row_map[coo.coords[0][keep]]
        cols = coo.coords[1][keep]
        n_cells = grid.size
        coo = scipy.sparse.coo_array(
            (coo.data[keep], (rows, cols)), shape=(n_cells, self.csr.shape[1])
        )
        const = np.full(n_cells, fill)
        const[row_map[valid]] = self.const[valid]
        coords = {
            name: (
                d,
                pd.Series(v, index=self.grid.indexes[d])
                .reindex(grid.indexes[d])
                .to_numpy(),
            )
            for name, (d, v) in self.coords.items()
        }
        return replace(
            self,
            csr=scipy.sparse.csr_array(coo),
            const=const,
            grid=grid,
            coords=coords,
        )

    def filled(self, value: float) -> CSRLinearExpression:
        """Resolve absent cells (NaN const) to a constant; terms untouched."""
        const = np.where(np.isnan(self.const), value, self.const)
        return replace(self, const=const)

    def renamed(self, names: dict[str, str]) -> CSRLinearExpression:
        """Relabel grid dims; the CSR row layout is unchanged."""
        coords = {n: (names.get(d, d), v) for n, (d, v) in self.coords.items()}
        return replace(self, grid=self.grid.renamed(names), coords=coords)

    def same_grid(self, other: CSRLinearExpression) -> bool:
        return self.grid == other.grid

    def added(self, other: CSRLinearExpression) -> CSRLinearExpression:
        """
        Sparse matrix addition == merge along the term dimension. Goes through
        COO so explicit zero coefficients survive (scipy's ``+`` drops them),
        keeping a cell with only zero-coefficient terms distinguishable from
        an empty cell, as on the dense path. A cell absent in either operand
        is absent in the sum and carries no terms (v1 dead-term invariant).
        Auxiliary coordinates propagate and conflicting ones raise (§11), as
        on the dense path.
        """
        const = self.const + other.const
        a, b = self.csr.tocoo(), other.csr.tocoo()
        shape = (self.n_cells, max(a.shape[1], b.shape[1]))
        rows = np.concatenate([a.coords[0], b.coords[0]])
        cols = np.concatenate([a.coords[1], b.coords[1]])
        data = np.concatenate([a.data, b.data])
        present = ~np.isnan(const)[rows]
        coo = scipy.sparse.coo_array(
            (data[present], (rows[present], cols[present])), shape=shape
        )
        enforce_aux_conflict([Dataset(coords=p.coords) for p in (self, other)])
        coords = other.coords | self.coords
        return replace(
            self, csr=scipy.sparse.csr_array(coo), const=const, coords=coords
        )

    def to_dense(self) -> Dataset:
        """
        Expand to the dense rectangle in canonical form: terms label-ordered,
        duplicates summed, padded to the widest cell with the usual fill.
        Absent cells (NaN const) carry no terms, per the v1 dead-term invariant.
        """
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
            coords=self.grid.indexes | self.coords,
        )
        return absorb_absence(ds)


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


def _aux_coords(ds: Dataset, dims: set[str]) -> dict[str, tuple[str, np.ndarray]]:
    """One-dimensional auxiliary coordinates of ``ds`` lying on ``dims``."""
    return {
        str(n): (str(c.dims[0]), c.to_numpy())
        for n, c in ds.coords.items()
        if n not in ds.dims and len(c.dims) == 1 and str(c.dims[0]) in dims
    }


def _outer_sum(axis_positions: list[np.ndarray]) -> np.ndarray:
    """Outer sum of per-axis offsets, flattened in C order."""
    cells = np.zeros((), dtype=np.int64)
    for pos in axis_positions:
        cells = cells[..., None] + pos
    return cells.reshape(-1)
