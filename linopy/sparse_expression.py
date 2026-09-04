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

from linopy.constants import TERM_DIM

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
        return cls._from_scatter(expr, grid_dims, indexes, group_dim, member_dim, codes)

    @classmethod
    def from_expression(
        cls, expr: LinearExpression, template: CSRPayload
    ) -> CSRPayload | None:
        """Convert a dense expression on the template's grid, else None."""
        if set(expr.coord_dims) != set(template.grid_dims):
            return None
        for d in expr.coord_dims:
            if not expr.data.get_index(d).equals(template.indexes[str(d)]):
                return None
        first = template.grid_dims[0]
        codes = np.arange(len(template.indexes[first]))
        return cls._from_scatter(
            expr, template.grid_dims, template.indexes, first, first, codes
        )

    @classmethod
    def _from_scatter(
        cls,
        expr: LinearExpression,
        grid_dims: tuple[str, ...],
        indexes: dict[str, pd.Index],
        scatter_dim: str,
        member_dim: str,
        codes: np.ndarray,
    ) -> CSRPayload:
        """
        Scatter an expression's terms into grid rows (conceptually ``G @ A``):
        ``member_dim`` lands in the grid dim ``scatter_dim`` at row positions
        ``codes``, every other grid dim maps one-to-one, and the COO→CSR
        conversion sums duplicates — which is the group sum. The constant is
        reduced with the dense kernel's skipna semantics.
        """
        ds = expr.data
        shape = tuple(len(indexes[d]) for d in grid_dims)
        strides = [
            int(np.prod(shape[i + 1 :], dtype=np.int64)) for i in range(len(shape))
        ]

        transposed = [member_dim if d == scatter_dim else d for d in grid_dims]
        axis_positions = [
            codes * stride if d == scatter_dim else np.arange(n) * stride
            for d, n, stride in zip(grid_dims, shape, strides)
        ]
        cell_rows = axis_positions[0]
        for pos in axis_positions[1:]:
            cell_rows = cell_rows[..., None] + pos
        cell_rows = cell_rows.reshape(-1)

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
        const = np.zeros(full_size)
        np.add.at(const, cell_rows, np.where(np.isnan(const_vals), 0.0, const_vals))

        return cls(scipy.sparse.csr_array(coo), const, grid_dims, indexes, expr.model)

    def scaled(self, factor: float) -> CSRPayload:
        return replace(self, csr=self.csr * factor, const=self.const * factor)

    def same_grid(self, other: CSRPayload) -> bool:
        return self.grid_dims == other.grid_dims and all(
            self.indexes[d].equals(other.indexes[d]) for d in self.grid_dims
        )

    def add(self, other: CSRPayload) -> CSRPayload:
        """
        Sparse matrix addition == merge along the term dimension. Goes through
        COO so explicit zero coefficients survive (scipy's ``+`` drops them),
        keeping a cell with only zero-coefficient terms distinguishable from
        an empty cell, as on the dense path.
        """
        a, b = self.csr.tocoo(), other.csr.tocoo()
        shape = (self.n_cells, max(a.shape[1], b.shape[1]))
        rows = np.concatenate([a.coords[0], b.coords[0]])
        cols = np.concatenate([a.coords[1], b.coords[1]])
        data = np.concatenate([a.data, b.data])
        coo = scipy.sparse.coo_array((data, (rows, cols)), shape=shape)
        return replace(
            self, csr=scipy.sparse.csr_array(coo), const=self.const + other.const
        )

    def materialize(self) -> LinearExpression:
        """
        Expand to the dense rectangle in canonical form: terms label-ordered,
        duplicates summed, padded to the widest cell with the usual fill.
        """
        from linopy.expressions import LinearExpression

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
        return LinearExpression(ds, self.model)


def try_csr_merge(
    exprs: Any, dim: str, join: Any, kwargs: dict
) -> LinearExpression | None:
    """
    Sparse branch of :func:`linopy.expressions.merge`: combine plain
    LinearExpressions on one shared grid (CSR-backed or dense-convertible),
    where any join produces the identical result. Returns None to fall
    through to the dense path.
    """
    from linopy.expressions import LinearExpression

    if dim != TERM_DIM or kwargs:
        return None
    if not all(type(e) is LinearExpression for e in exprs):
        return None
    payloads = [e._payload for e in exprs if e._payload is not None]
    if not payloads:
        return None
    template = payloads[0]
    if not all(template.same_grid(p) for p in payloads[1:]):
        return None

    combined: CSRPayload | None = None
    for e in exprs:
        payload = e._payload or CSRPayload.from_expression(e, template)
        if payload is None:
            return None
        combined = payload if combined is None else combined.add(payload)
    assert combined is not None
    return LinearExpression._from_payload(combined, exprs[0].model)
