"""
Lazy groupby-sum: an internal long-format representation for grouped sums.

``expr.groupby(g).sum(lazy=True)`` (or ``linopy.options["lazy_groupby"] =
True``) returns an ordinary :class:`~linopy.expressions.LinearExpression`
whose payload is a :class:`LazyGroupSum` — a list of ungrouped parts plus
their groupers — instead of the materialized dense dataset. Modeled on
dask-backed xarray objects: same public type, different backing, and any
operation without a lazy branch transparently materializes through the
``.data`` property (yielding exactly today's result, so laziness can only
delay the dense rectangle, never change semantics).

The payoff comes at ``Model.add_constraints`` with ``freeze=True`` (or
``Model.freeze_constraints``): a still-lazy left-hand side is realized
directly as a :class:`~linopy.constraints.CSRConstraint` from the ungrouped
long triplets — scipy's COO->CSR duplicate summation *is* the group sum —
so the ``group-size-padded`` ``_term`` rectangle (issue #745) never exists.

Operations with a lazy branch (everything else materializes):

- unary minus and multiplication by a scalar (scales the ungrouped parts),
- ``merge`` along ``_term`` — and therefore ``+``/``-`` — with other lazy
  expressions on the same grid or with dense expressions already on the
  result grid,
- comparison with a constant rhs (``==``, ``<=``, ``>=``), carried lazily
  on the resulting :class:`~linopy.constraints.Constraint`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.sparse
from xarray import DataArray

from linopy.constants import HELPER_DIMS, TERM_DIM

if TYPE_CHECKING:
    from linopy.constraints import CSRConstraint
    from linopy.expressions import LinearExpression
    from linopy.model import Model


@dataclass(frozen=True)
class LazyPart:
    """
    One summand: an ungrouped expression plus its grouper.

    ``grouper`` maps the labels of ``expr``'s member dim (the grouper's
    index, named after that dim) to group labels. ``None`` marks an
    identity part that already lives on the result grid.
    """

    expr: LinearExpression
    grouper: pd.Series | None


@dataclass(frozen=True)
class LazyGroupSum:
    """
    A sum of grouped/dense parts over a fixed result grid.

    ``grid_dims`` is the result's coordinate dim order (the grouped dim in
    the member dim's position, as the dense kernel produces it) and
    ``indexes`` maps each grid dim to its pandas index.
    """

    parts: tuple[LazyPart, ...]
    group_dim: str
    grid_dims: tuple[str, ...]
    indexes: dict[str, pd.Index]

    @classmethod
    def from_grouper(
        cls, expr: LinearExpression, grouper: pd.Series, group_dim: str
    ) -> LazyGroupSum:
        member_dim = str(grouper.index.name)
        if member_dim in expr.data.indexes:
            # labels checked equal upstream; conform the order
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
        indexes = {
            d: expr.data.get_index(d).rename(d)
            for d in expr.coord_dims
            if d != member_dim
        }
        indexes[group_dim] = pd.Index(uniques, name=group_dim)
        return cls((LazyPart(expr, grouper),), group_dim, grid_dims, indexes)

    @property
    def group_index(self) -> pd.Index:
        return self.indexes[self.group_dim]

    def scaled(self, factor: float) -> LazyGroupSum:
        parts = tuple(LazyPart(p.expr * factor, p.grouper) for p in self.parts)
        return replace(self, parts=parts)

    def same_grid(self, other: LazyGroupSum) -> bool:
        return (
            self.group_dim == other.group_dim
            and self.grid_dims == other.grid_dims
            and all(self.indexes[d].equals(other.indexes[d]) for d in self.grid_dims)
        )

    def wrap_dense(self, expr: LinearExpression) -> LazyPart | None:
        """Wrap a dense expression already on the result grid as identity part."""
        if set(expr.coord_dims) != set(self.grid_dims):
            return None
        for d in expr.coord_dims:
            if not expr.data.get_index(d).equals(self.indexes[str(d)]):
                return None
        return LazyPart(expr, None)

    def materialize(self) -> LinearExpression:
        """Left-fold the parts through today's dense kernel."""
        result: LinearExpression | None = None
        for part in self.parts:
            if part.grouper is None:
                term = part.expr
            else:
                term = part.expr.groupby(part.grouper).sum(lazy=False)
            result = term if result is None else result + term
        assert result is not None
        return result


def try_lazy_merge(
    exprs: Any, dim: str, join: Any, kwargs: dict
) -> LinearExpression | None:
    """
    Lazy branch of :func:`linopy.expressions.merge`.

    Returns a lazy combined expression when every input is a plain
    LinearExpression on the same result grid (lazy, or dense wrappable as
    an identity part) — where any join produces the identical result — and
    None otherwise to fall through to the dense path.
    """
    from linopy.expressions import LinearExpression

    if dim != TERM_DIM or kwargs:
        return None
    if not all(type(e) is LinearExpression for e in exprs):
        return None
    lazies = [e._lazy for e in exprs if e._lazy is not None]
    if not lazies:
        return None
    template = lazies[0]
    if not all(template.same_grid(lz) for lz in lazies[1:]):
        return None

    parts: list[LazyPart] = []
    for e in exprs:
        if e._lazy is not None:
            parts.extend(e._lazy.parts)
        else:
            part = template.wrap_dense(e)
            if part is None:
                return None
            parts.append(part)
    combined = replace(template, parts=tuple(parts))
    return LinearExpression._from_lazy(combined, exprs[0].model)


def extract_lazy(lhs: Any, sign: Any, rhs: Any) -> tuple[LazyGroupSum, str, Any] | None:
    """Return (payload, sign, rhs) if lhs is a realizable lazy constraint."""
    from linopy.common import is_constant
    from linopy.constraints import Constraint
    from linopy.expressions import LinearExpression

    if (
        isinstance(lhs, Constraint)
        and lhs._lazy is not None
        and sign is None
        and rhs is None
    ):
        expr, sign, rhs = lhs._lazy
        if expr._lazy is None:  # already materialized elsewhere
            return None
        lhs = expr
    if not (isinstance(lhs, LinearExpression) and lhs._lazy is not None):
        return None
    if not isinstance(sign, str) or rhs is None or not is_constant(rhs):
        return None
    lazy = lhs._lazy
    rhs_da = _as_rhs_dataarray(rhs)
    if rhs_da is None or not set(rhs_da.dims) <= set(lazy.grid_dims):
        return None
    return lazy, sign, rhs


def _as_rhs_dataarray(rhs: Any) -> DataArray | None:
    from linopy.alignment import as_dataarray

    try:
        da = as_dataarray(rhs)
    except (TypeError, ValueError):
        return None
    if set(da.dims) & set(HELPER_DIMS):
        return None
    return da


def _part_entries(
    part: LazyPart, lazy: LazyGroupSum, label_to_pos: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Long-form (rows, cols, coeffs) triplets of one part plus its const.

    Rows are flat positions in the ``grid_dims`` constraint grid (C order).
    Returns (rows_per_term, cols, coeffs, cell_rows_flat, const_flat_values).
    """
    ds = part.expr.data
    shape = tuple(len(lazy.indexes[d]) for d in lazy.grid_dims)
    strides = np.array(
        [int(np.prod(shape[i + 1 :], dtype=np.int64)) for i in range(len(shape))]
    )

    # per grid dim: the position of each of the part's cells along that dim
    transposed: list[str] = []
    axis_positions: list[np.ndarray] = []
    for d, stride in zip(lazy.grid_dims, strides):
        if part.grouper is not None and d == lazy.group_dim:
            member_dim = str(part.grouper.index.name)
            transposed.append(member_dim)
            codes = lazy.group_index.get_indexer(part.grouper.to_numpy())
            axis_positions.append(codes * stride)
        else:
            transposed.append(d)
            axis_positions.append(np.arange(shape[lazy.grid_dims.index(d)]) * stride)

    cell_rows = axis_positions[0]
    for pos in axis_positions[1:]:
        cell_rows = cell_rows[..., None] + pos
    cell_rows = cell_rows.reshape(-1)

    coeffs = ds.coeffs.transpose(*transposed, TERM_DIM).to_numpy().reshape(-1)
    vars_ = ds.vars.transpose(*transposed, TERM_DIM).to_numpy().reshape(-1)
    nterm = ds.sizes[TERM_DIM]
    rows = np.repeat(cell_rows, nterm)

    keep = (vars_ != -1) & (coeffs != 0) & ~np.isnan(coeffs)
    rows, vars_, coeffs = rows[keep], vars_[keep], coeffs[keep]

    const = ds.const.transpose(*transposed).to_numpy().reshape(-1)
    return rows, label_to_pos[vars_], coeffs, cell_rows, const


def realize_lazy_constraint(
    model: Model, lazy: LazyGroupSum, sign: str, rhs: Any, name: str
) -> CSRConstraint:
    """
    Realize ``lazy sign rhs`` directly as a CSRConstraint.

    Equivalent to materializing the lazy sum, adding the constraint and
    freezing it — but the padded dense ``_term`` rectangle never exists;
    peak memory is proportional to the number of nonzero terms.
    """
    from linopy.common import maybe_replace_sign
    from linopy.constraints import CSRConstraint
    from linopy.semantics import check_user_nan, is_v1

    sign = maybe_replace_sign(sign)
    shape = tuple(len(lazy.indexes[d]) for d in lazy.grid_dims)
    full_size = int(np.prod(shape, dtype=np.int64)) if shape else 1

    label_index = model.variables.label_index
    label_to_pos = label_index.label_to_pos

    rows_l, cols_l, data_l = [], [], []
    const_flat = np.zeros(full_size)
    for part in lazy.parts:
        rows, cols, coeffs, cell_rows, const = _part_entries(part, lazy, label_to_pos)
        rows_l.append(rows)
        cols_l.append(cols)
        data_l.append(coeffs)
        np.add.at(const_flat, cell_rows, np.where(np.isnan(const), 0.0, const))

    coo = scipy.sparse.coo_array(
        (np.concatenate(data_l), (np.concatenate(rows_l), np.concatenate(cols_l))),
        shape=(full_size, label_index.n_active_vars),
    )
    csr = scipy.sparse.csr_array(coo)  # sums duplicates == the group sum
    csr.eliminate_zeros()

    # rhs aligned by label to the grid; expression constants move to the rhs
    rhs_da = _as_rhs_dataarray(rhs)
    assert rhs_da is not None
    if is_v1() and bool(rhs_da.isnull().any()):
        check_user_nan()  # §5: NaN in a user constant raises under v1
    if is_v1():
        # §8 parity with the dense path: a reordered or differing rhs index
        # raises; the user aligns explicitly with .sel/.reindex
        for d in rhs_da.dims:
            if not rhs_da.get_index(d).equals(lazy.indexes[str(d)]):
                raise ValueError(
                    f"Coordinate mismatch on shared dimension {d!r} between "
                    "the rhs and the grouped result. Align the rhs with "
                    ".sel(...) / .reindex(...) before combining (§8)."
                )
    rhs_da = rhs_da.reindex(
        {d: lazy.indexes[str(d)] for d in rhs_da.dims}, fill_value=np.nan
    )
    missing = {d: lazy.indexes[d] for d in lazy.grid_dims if d not in rhs_da.dims}
    if missing:
        rhs_da = rhs_da.expand_dims(missing)
    rhs_flat = rhs_da.transpose(*lazy.grid_dims).to_numpy().reshape(-1) - const_flat

    # label allocation, mirroring Model._allocate_constraint_labels; rows
    # without terms or with NaN rhs are inactive, as in the frozen dense path
    cindex = model._cCounter
    model._cCounter += full_size
    active = (np.diff(csr.indptr) > 0) & ~np.isnan(rhs_flat)
    con_labels = np.arange(cindex, cindex + full_size)[active]

    return CSRConstraint(
        csr[active],
        con_labels,
        rhs_flat[active],
        sign,
        coords=[lazy.indexes[d] for d in lazy.grid_dims],
        model=model,
        name=name,
        cindex=cindex,
    )
