"""
The sparse payload behind a LinearExpression: ``A @ x + c`` in CSR form.

``expr.groupby(g).sum(sparse=True)`` (or ``linopy.options["sparse_groupby"]``
under v1) returns an ordinary :class:`~linopy.expressions.LinearExpression`
backed by a :class:`CSRPayload` instead of the dense dataset — same public
type, different backing, akin to dask-backed xarray objects. The CSR form is
canonical (duplicate variables summed, terms label-ordered) and ragged along
``_term``, so the group-size padding of issue #745 has no analog; grouping,
``merge``/``+``/``-`` and scaling become sparse linear algebra. Anything
without a sparse branch expands through ``.data`` to the mathematically
identical dense rectangle in canonical term layout — the reason the feature
is v1-gated, where term layout is non-contractual.

This module covers the expression layer only. Stapling sign and rhs onto a
payload to form a :class:`~linopy.constraints.CSRConstraint` lives in
:meth:`linopy.constraints.CSRConstraint.from_payload`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.sparse
from xarray import Dataset
from xarray.core.types import JoinOptions

from linopy.constants import TERM_DIM
from linopy.semantics import FillValueLike, join_fill

if TYPE_CHECKING:
    from linopy.expressions import LinearExpression
    from linopy.model import Model


@dataclass(frozen=True)
class CSRPayload:
    """
    An expression as ``A @ x + c`` over a fixed coordinate grid.

    ``csr`` has one row per flat grid cell (C order over ``grid_dims``) and
    one column per raw variable label — label columns stay valid when
    variables are added to the model later; realization maps them to dense
    positions. ``const`` is the per-cell constant.
    """

    csr: scipy.sparse.csr_array
    const: np.ndarray
    grid_dims: tuple[str, ...]
    indexes: dict[str, pd.Index]
    model: Model

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(self.indexes[d]) for d in self.grid_dims)

    @property
    def n_cells(self) -> int:
        return self.csr.shape[0]

    @classmethod
    def from_grouper(
        cls, expr: LinearExpression, grouper: pd.Series, group_dim: str
    ) -> CSRPayload:
        """
        Build the grouped sum directly in CSR form (no padded rectangle).

        The grouper is conformed to the expression's member index by label
        (upstream alignment checks guarantee equal label sets) and group
        labels are sorted, matching the dense kernel's output grid.
        """
        member_dim = str(grouper.index.name)
        if member_dim in expr.data.indexes:
            grouper = grouper.reindex(expr.data.indexes[member_dim])
        elif len(grouper) != expr.data.sizes[member_dim]:
            raise ValueError(f"grouper length does not match dimension {member_dim!r}")
        codes, uniques = pd.factorize(grouper, sort=True)
        if (codes == -1).any():
            raise ValueError(
                "Cannot group by a pandas object containing NaN values. "
                "Drop or fill the corresponding entries before grouping."
            )
        grid_dims = tuple(
            group_dim if d == member_dim else str(d) for d in expr.coord_dims
        )
        indexes: dict[str, pd.Index] = {
            str(d): expr.data.get_index(d).rename(d)
            for d in expr.coord_dims
            if d != member_dim
        }
        indexes[group_dim] = pd.Index(uniques, name=group_dim)
        return cls._from_scatter(
            expr, grid_dims, indexes, group_dim, member_dim, codes, True
        )

    @classmethod
    def from_expression(cls, expr: LinearExpression) -> CSRPayload:
        """Convert a dense expression to CSR form on its own coordinate grid."""
        grid_dims = tuple(str(d) for d in expr.coord_dims)
        indexes = {d: expr.data.get_index(d).rename(d) for d in grid_dims}
        first = grid_dims[0]
        codes = np.arange(len(indexes[first]))
        return cls._from_scatter(expr, grid_dims, indexes, first, first, codes, False)

    @classmethod
    def _from_scatter(
        cls,
        expr: LinearExpression,
        grid_dims: tuple[str, ...],
        indexes: dict[str, pd.Index],
        scatter_dim: str,
        member_dim: str,
        codes: np.ndarray,
        skipna: bool,
    ) -> CSRPayload:
        """
        Scatter an expression's terms into grid rows (conceptually ``G @ A``):
        ``member_dim`` lands in the grid dim ``scatter_dim`` at row positions
        ``codes``, every other grid dim maps one-to-one, and the COO→CSR
        conversion sums duplicates — which is the group sum. With ``skipna``
        the constant is reduced as by the dense group kernel (NaN members
        count as 0); without it an absent cell (NaN const) stays absent, as
        on the dense v1 merge path.
        """
        ds = expr.data
        shape, strides = _grid_layout(grid_dims, indexes)

        transposed = [member_dim if d == scatter_dim else d for d in grid_dims]
        cell_rows = _flat_cells(
            [
                codes * stride if d == scatter_dim else np.arange(n) * stride
                for d, n, stride in zip(grid_dims, shape, strides)
            ]
        )

        coeffs = ds.coeffs.transpose(*transposed, TERM_DIM).to_numpy().reshape(-1)
        vars_ = ds.vars.transpose(*transposed, TERM_DIM).to_numpy().reshape(-1)
        rows = np.repeat(cell_rows, ds.sizes[TERM_DIM])
        keep = (vars_ != -1) & ~np.isnan(coeffs)

        full_size = int(np.prod(shape, dtype=np.int64)) if shape else 1
        coo = scipy.sparse.coo_array(
            (coeffs[keep], (rows[keep], vars_[keep])),
            shape=(full_size, expr.model._xCounter),
        )

        const_vals = ds.const.transpose(*transposed).to_numpy().reshape(-1)
        if skipna:
            const_vals = np.where(np.isnan(const_vals), 0.0, const_vals)
        const = np.zeros(full_size)
        np.add.at(const, cell_rows, const_vals)

        return cls(scipy.sparse.csr_array(coo), const, grid_dims, indexes, expr.model)

    def scaled(self, factor: float) -> CSRPayload:
        return replace(self, csr=self.csr * factor, const=self.const * factor)

    def reindexed(
        self,
        indexes: dict[str, pd.Index],
        grid_dims: tuple[str, ...] | None = None,
        fill: float = np.nan,
    ) -> CSRPayload:
        """
        Remap rows onto new per-dim indexes, optionally in a new dim order,
        without the dense rectangle: dropped labels vanish, new labels get
        ``fill`` as their constant (NaN: absent cells).
        """
        grid_dims = grid_dims or self.grid_dims
        shape, strides = _grid_layout(grid_dims, indexes)
        stride_of = dict(zip(grid_dims, strides))
        positions = [indexes[d].get_indexer(self.indexes[d]) for d in self.grid_dims]
        valid = _flat_cells([pos == -1 for pos in positions]) == 0
        row_map = _flat_cells(
            [pos * stride_of[d] for pos, d in zip(positions, self.grid_dims)]
        )

        coo = self.csr.tocoo()
        keep = valid[coo.coords[0]]
        rows = row_map[coo.coords[0][keep]]
        cols = coo.coords[1][keep]
        n_cells = int(np.prod(shape, dtype=np.int64))
        coo = scipy.sparse.coo_array(
            (coo.data[keep], (rows, cols)), shape=(n_cells, self.csr.shape[1])
        )
        const = np.full(n_cells, fill)
        const[row_map[valid]] = self.const[valid]
        return replace(
            self,
            csr=scipy.sparse.csr_array(coo),
            const=const,
            grid_dims=grid_dims,
            indexes=indexes,
        )

    def filled(self, value: float) -> CSRPayload:
        """Resolve absent cells (NaN const) to a constant; terms untouched."""
        const = np.where(np.isnan(self.const), value, self.const)
        return replace(self, const=const)

    def renamed(self, names: dict[str, str]) -> CSRPayload:
        """Relabel grid dims; the CSR row layout is unchanged."""
        grid_dims = tuple(names.get(d, d) for d in self.grid_dims)
        indexes = {
            names.get(d, d): self.indexes[d].rename(names.get(d, d))
            for d in self.grid_dims
        }
        return replace(self, grid_dims=grid_dims, indexes=indexes)

    def same_grid(self, other: CSRPayload) -> bool:
        return self.grid_dims == other.grid_dims and all(
            self.indexes[d].equals(other.indexes[d]) for d in self.grid_dims
        )

    def add(self, other: CSRPayload) -> CSRPayload:
        """
        Sparse matrix addition == merge along the term dimension. Goes through
        COO so explicit zero coefficients survive (scipy's ``+`` drops them),
        keeping a cell with only zero-coefficient terms distinguishable from
        an empty cell, as on the dense path. A cell absent in either operand
        is absent in the sum and carries no terms (v1 dead-term invariant).
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
        return replace(self, csr=scipy.sparse.csr_array(coo), const=const)

    def materialize(self) -> LinearExpression:
        """
        Expand to the dense rectangle in canonical form: terms label-ordered,
        duplicates summed, padded to the widest cell with the usual fill.
        Absent cells (NaN const) carry no terms, per the v1 dead-term invariant.
        """
        from linopy.expressions import LinearExpression
        from linopy.semantics import absorb_absence

        csr = self.csr.copy()
        csr.sort_indices()
        lengths = np.diff(csr.indptr)
        nterm = max(int(lengths.max(initial=0)), 1)

        vars_flat = np.full(
            (self.n_cells, nterm), -1, dtype=self.model._dtypes["labels"]
        )
        coeffs_flat = np.full((self.n_cells, nterm), np.nan)
        rows = np.repeat(np.arange(self.n_cells), lengths)
        pos = np.arange(csr.nnz) - np.repeat(csr.indptr[:-1], lengths)
        vars_flat[rows, pos] = csr.indices
        coeffs_flat[rows, pos] = csr.data

        dims = (*self.grid_dims, TERM_DIM)
        ds = Dataset(
            {
                "coeffs": (dims, coeffs_flat.reshape(*self.shape, nterm)),
                "vars": (dims, vars_flat.reshape(*self.shape, nterm)),
                "const": (self.grid_dims, self.const.reshape(self.shape)),
            },
            coords={d: self.indexes[d] for d in self.grid_dims},
        )
        return LinearExpression(absorb_absence(ds), self.model)


def _grid_layout(
    grid_dims: tuple[str, ...], indexes: dict[str, pd.Index]
) -> tuple[tuple[int, ...], list[int]]:
    """C-order shape and row strides of the grid."""
    shape = tuple(len(indexes[d]) for d in grid_dims)
    strides = [int(np.prod(shape[i + 1 :], dtype=np.int64)) for i in range(len(shape))]
    return shape, strides


def _flat_cells(axis_positions: list[np.ndarray]) -> np.ndarray:
    """Outer sum of per-axis offsets, flattened in C order."""
    cells = np.zeros((), dtype=np.int64)
    for pos in axis_positions:
        cells = cells[..., None] + pos
    return cells.reshape(-1)


def _aligned(
    payloads: list[CSRPayload], join: JoinOptions | None, fill: float
) -> list[CSRPayload] | None:
    """
    Conform the payloads to the grid an explicit join produces, cells the
    join creates carrying ``fill`` as constant. None where the dense path
    owns the semantics: ``exact`` and the auto-detected join raise there on
    differing grids, ``override`` on differing shapes, any join on
    non-unique labels.
    """
    template = payloads[0]
    dims = template.grid_dims
    if any(not p.indexes[d].is_unique for p in payloads for d in dims):
        return None
    if join == "override":
        if any(p.grid_dims != dims or p.shape != template.shape for p in payloads):
            return None
        return [replace(p, indexes=template.indexes) for p in payloads]
    if join == "left":
        indexes = template.indexes
    elif join == "right":
        indexes = payloads[-1].indexes
    elif join in ("outer", "inner"):
        combine = pd.Index.union if join == "outer" else pd.Index.intersection
        indexes = {}
        for d in dims:
            index = template.indexes[d]
            for p in payloads[1:]:
                index = combine(index, p.indexes[d])
            indexes[d] = pd.Index(index, name=d)
    else:
        return None
    return [p.reindexed(indexes, dims, fill) for p in payloads]


def try_csr_merge(
    exprs: Any,
    dim: str,
    join: JoinOptions | None,
    fill_value: FillValueLike,
    kwargs: dict[str, Any],
) -> LinearExpression | None:
    """
    Sparse branch of :func:`linopy.expressions.merge`: combine plain
    LinearExpressions over one set of grid dimensions (CSR-backed or
    dense-convertible) as sparse matrix addition. Grids that differ in
    their labels are aligned row-wise onto the joined grid, the cells the
    join creates carrying the fill of the dense path (zero, or NaN for
    ``fill_value=ABSENT``). Returns None to fall through to the dense path.
    """
    from linopy.expressions import LinearExpression

    if dim != TERM_DIM or kwargs:
        return None
    if not all(type(e) is LinearExpression for e in exprs):
        return None
    if all(e._payload is None for e in exprs):
        return None
    dims = set(exprs[0].coord_dims)
    if any(set(e.coord_dims) != dims for e in exprs[1:]):
        return None
    if any(e._payload is None and set(e.data.coords) - dims for e in exprs):
        return None

    payloads = [e._payload or CSRPayload.from_expression(e) for e in exprs]
    template = payloads[0]
    if not all(template.same_grid(p) for p in payloads[1:]):
        aligned = _aligned(payloads, join, join_fill(fill_value, 0.0))
        if aligned is None:
            return None
        payloads = aligned

    combined = payloads[0]
    for payload in payloads[1:]:
        combined = combined.add(payload)
    return LinearExpression._from_payload(combined, exprs[0].model)
