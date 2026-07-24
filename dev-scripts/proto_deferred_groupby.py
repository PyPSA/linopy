"""
Prototype: deferred groupby().sum() realized directly as a CSRConstraint.

Idea (issue #745 / #756): the padded rectangle produced by
``expr.groupby(g).sum()`` is pure intermediate waste whenever the result ends
up as a constraint — the LP/matrix export un-pads it again. Instead of
materializing the grouped expression, hold (ungrouped expr, grouper) as a
deferred node and realize the constraint straight from the ungrouped long
triplets:

    row  = flat position of (group, *rest_dims) in the constraint grid
    col  = label_to_pos[var_label]
    data = coeff

scipy's COO->CSR conversion sums duplicate (row, col) entries, which *is* the
group sum. No `_term` padding ever exists; peak memory is O(nnz), independent
of the group-size distribution.

This module deliberately builds a `CSRConstraint` — the existing second
constraint backend (#630 / freeze_constraints) — so the whole downstream
pipeline (LP writer via `to_polars`, matrix export via `to_matrix_with_rhs`)
works unchanged.

Prototype limitations (documented, not fundamental):
- all deferred parts must share the same non-grouped ("rest") dims,
- the grouped result is only usable as a constraint LHS (no further
  expression arithmetic; that path would fall back to the dense kernel),
- signs are uniform per constraint, masks are not supported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse
import xarray as xr

from linopy.constants import TERM_DIM
from linopy.constraints import CSRConstraint
from linopy.expressions import LinearExpression
from linopy.model import Model


@dataclass
class DeferredGroupbySum:
    """
    An unmaterialized ``expr.groupby(grouper).sum()``.

    ``grouper`` maps the labels of one of ``expr``'s dims (its index, whose
    ``name`` must be that dim) to group labels.
    """

    expr: LinearExpression
    grouper: pd.Series

    @property
    def member_dim(self) -> str:
        return str(self.grouper.index.name)

    @property
    def rest_dims(self) -> list[str]:
        return [d for d in self.expr.coord_dims if d != self.member_dim]

    def materialize(self) -> LinearExpression:
        """Fall back to the dense scatter kernel (today's behavior)."""
        return self.expr.groupby(self.grouper).sum()


def _part_triplets(
    part: DeferredGroupbySum,
    group_index: pd.Index,
    rest_dims: list[str],
    label_to_pos: np.ndarray,
    n_rest: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Long-form (row, col, coeff) triplets of one part, plus its grouped const.

    Rows are flat positions in the (group, *rest_dims) constraint grid.
    """
    ds = part.expr.data
    member_dim = part.member_dim

    grouper = part.grouper.reindex(ds.indexes[member_dim])
    if grouper.isna().any():
        raise ValueError(f"grouper does not cover all {member_dim!r} labels")
    gcodes = group_index.get_indexer(grouper.to_numpy())
    if (gcodes == -1).any():
        raise ValueError("grouper contains labels outside the group index")

    coeffs = ds.coeffs.transpose(member_dim, *rest_dims, TERM_DIM).to_numpy()
    vars_ = ds.vars.transpose(member_dim, *rest_dims, TERM_DIM).to_numpy()
    nterm = ds.sizes[TERM_DIM]

    # flat row id of each (member, rest...) cell in the constraint grid
    cell_rows = gcodes[:, None] * n_rest + np.arange(n_rest)[None, :]
    rows = np.repeat(cell_rows.reshape(-1), nterm)
    cols_label = vars_.reshape(-1)
    data = coeffs.reshape(-1)

    keep = (cols_label != -1) & (data != 0) & ~np.isnan(data)
    rows, cols_label, data = rows[keep], cols_label[keep], data[keep]
    cols = label_to_pos[cols_label]

    # group-sum of the constant (a true numeric reduction, already cheap)
    const = (
        ds.const.transpose(member_dim, *rest_dims)
        .to_numpy()
        .reshape(len(gcodes), n_rest)
    )
    const_grid = np.zeros((len(group_index), n_rest))
    np.add.at(const_grid, gcodes, np.where(np.isnan(const), 0.0, const))

    return rows, cols, data, const_grid.reshape(-1)


def add_deferred_constraints(
    model: Model,
    parts: list[DeferredGroupbySum],
    sign: str,
    rhs: float | xr.DataArray,
    name: str,
    group_index: pd.Index,
) -> CSRConstraint:
    """
    Realize ``sum(parts) sign rhs`` as a CSRConstraint, no dense rectangle.

    Equivalent to
    ``model.add_constraints(sum(p.materialize() for p in parts), sign, rhs, name)``
    for constraint rows over ``(group_index, *rest_dims)``.
    """
    rest_dims = parts[0].rest_dims
    for p in parts[1:]:
        if p.rest_dims != rest_dims:
            raise ValueError("all parts must share the same non-grouped dims")
    rest_indexes = [parts[0].expr.data.indexes[d] for d in rest_dims]
    n_rest = int(np.prod([len(ix) for ix in rest_indexes])) if rest_indexes else 1
    full_size = len(group_index) * n_rest

    label_index = model.variables.label_index
    label_to_pos = label_index.label_to_pos

    rows_l, cols_l, data_l = [], [], []
    const_flat = np.zeros(full_size)
    for part in parts:
        rows, cols, data, const = _part_triplets(
            part, group_index, rest_dims, label_to_pos, n_rest
        )
        rows_l.append(rows)
        cols_l.append(cols)
        data_l.append(data)
        const_flat += const

    coo = scipy.sparse.coo_array(
        (np.concatenate(data_l), (np.concatenate(rows_l), np.concatenate(cols_l))),
        shape=(full_size, label_index.n_active_vars),
    )
    csr = scipy.sparse.csr_array(coo)  # sums duplicates == the group sum
    csr.eliminate_zeros()

    # rhs over the grid; expression constants move to the right-hand side
    if isinstance(rhs, xr.DataArray):
        grid = {str(group_index.name): group_index} | dict(zip(rest_dims, rest_indexes))
        rhs_flat = (
            rhs.reindex(grid)  # align by label to the constraint grid
            .transpose(str(group_index.name), *rest_dims)
            .to_numpy()
            .reshape(-1)
        )
    else:
        rhs_flat = np.full(full_size, float(rhs))
    rhs_flat = rhs_flat - const_flat

    # label allocation, mirroring Model._allocate_constraint_labels
    cindex = model._cCounter
    model._cCounter += full_size
    active = np.diff(csr.indptr) > 0
    con_labels = np.arange(cindex, cindex + full_size)[active]

    con = CSRConstraint(
        csr[active],
        con_labels,
        rhs_flat[active],
        sign,
        coords=[group_index, *rest_indexes],
        model=model,
        name=name,
        cindex=cindex,
    )
    model.constraints.add(con)
    return con
