"""
CSR-backed expressions: a sparse internal representation for grouped sums.

``expr.groupby(g).sum(sparse=True)`` (or ``linopy.options["sparse_groupby"]``
under v1) returns an ordinary :class:`~linopy.expressions.LinearExpression`
backed by a :class:`CSRPayload` instead of the dense dataset — same public
type, different backing, akin to dask-backed xarray objects. The CSR form is
canonical (duplicate variables summed, terms label-ordered) and ragged along
``_term``, so the group-size padding of issue #745 has no analog; grouping,
``merge``/``+``/``-`` and scaling become sparse linear algebra, and
``Model.add_constraints(..., freeze=True)`` staples sign and rhs on to form a
:class:`~linopy.constraints.CSRConstraint` directly. Anything without a
sparse branch expands through ``.data`` to the mathematically identical dense
rectangle in canonical term layout — the reason the feature is v1-gated,
where term layout is non-contractual.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.sparse
from xarray import DataArray, Dataset

from linopy.constants import HELPER_DIMS, TERM_DIM

if TYPE_CHECKING:
    from linopy.constraints import CSRConstraint
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
        indexes = {
            d: expr.data.get_index(d).rename(d)
            for d in expr.coord_dims
            if d != member_dim
        }
        indexes[group_dim] = pd.Index(uniques, name=group_dim)
        return _assemble(expr, grid_dims, indexes, group_dim, member_dim, codes)

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
        return _assemble(
            expr, template.grid_dims, template.indexes, first, first, codes
        )

    def scaled(self, factor: float) -> CSRPayload:
        return replace(self, csr=self.csr * factor, const=self.const * factor)

    def same_grid(self, other: CSRPayload) -> bool:
        return self.grid_dims == other.grid_dims and all(
            self.indexes[d].equals(other.indexes[d]) for d in self.grid_dims
        )

    def add(self, other: CSRPayload) -> CSRPayload:
        """Sparse matrix addition == merge along the term dimension."""
        width = max(self.csr.shape[1], other.csr.shape[1])
        csr = _widen(self.csr, width) + _widen(other.csr, width)
        return replace(self, csr=csr, const=self.const + other.const)

    def materialize(self) -> LinearExpression:
        """
        Expand to the dense rectangle in canonical form: terms label-ordered,
        duplicates summed, padded to the widest cell with the usual fill.
        """
        from linopy.expressions import LinearExpression

        csr = self.csr.copy()
        csr.eliminate_zeros()
        csr.sort_indices()
        lengths = np.diff(csr.indptr)
        nterm = max(int(lengths.max()) if len(lengths) else 0, 1)

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


def _widen(csr: scipy.sparse.csr_array, width: int) -> scipy.sparse.csr_array:
    if csr.shape[1] == width:
        return csr
    csr = csr.copy()
    csr.resize((csr.shape[0], width))
    return csr


def _assemble(
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
    strides = [int(np.prod(shape[i + 1 :], dtype=np.int64)) for i in range(len(shape))]

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
    keep = (vars_ != -1) & (coeffs != 0) & ~np.isnan(coeffs)

    full_size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    coo = scipy.sparse.coo_array(
        (coeffs[keep], (rows[keep], vars_[keep])),
        shape=(full_size, expr.model._xCounter),
    )

    const_vals = ds.const.transpose(*transposed).to_numpy().reshape(-1)
    const = np.zeros(full_size)
    np.add.at(const, cell_rows, np.where(np.isnan(const_vals), 0.0, const_vals))

    return CSRPayload(
        scipy.sparse.csr_array(coo), const, grid_dims, indexes, expr.model
    )


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
    payloads = [e._csr for e in exprs if e._csr is not None]
    if not payloads:
        return None
    template = payloads[0]
    if not all(template.same_grid(p) for p in payloads[1:]):
        return None

    combined: CSRPayload | None = None
    for e in exprs:
        payload = e._csr or CSRPayload.from_expression(e, template)
        if payload is None:
            return None
        combined = payload if combined is None else combined.add(payload)
    assert combined is not None
    return LinearExpression._from_csr(combined, exprs[0].model)


def extract_pending(
    lhs: Any, sign: Any, rhs: Any
) -> tuple[CSRPayload, str, Any] | None:
    """Return (payload, sign, rhs) if lhs is a realizable CSR constraint."""
    from linopy.common import is_constant
    from linopy.constraints import Constraint
    from linopy.expressions import LinearExpression

    if (
        isinstance(lhs, Constraint)
        and lhs._pending is not None
        and sign is None
        and rhs is None
    ):
        lhs, sign, rhs = lhs._pending
    if not (isinstance(lhs, LinearExpression) and lhs._csr is not None):
        return None
    if not isinstance(sign, str) or rhs is None or not is_constant(rhs):
        return None
    rhs_da = _as_rhs_dataarray(rhs)
    if rhs_da is None or not set(rhs_da.dims) <= set(lhs._csr.grid_dims):
        return None
    return lhs._csr, sign, rhs


def _as_rhs_dataarray(rhs: Any) -> DataArray | None:
    from linopy.alignment import as_dataarray

    try:
        da = as_dataarray(rhs)
    except (TypeError, ValueError):
        return None
    return None if set(da.dims) & set(HELPER_DIMS) else da


def _rhs_grid_values(payload: CSRPayload, rhs: Any) -> np.ndarray:
    """
    Align the rhs to the payload grid by label and flatten it, with v1
    parity: NaN in the rhs raises (§5) and a reordered or differing index
    on a shared dim raises (§8), as on the dense path.
    """
    from linopy.semantics import check_user_nan, is_v1

    rhs_da = _as_rhs_dataarray(rhs)
    assert rhs_da is not None
    if is_v1():
        if bool(rhs_da.isnull().any()):
            check_user_nan()
        for d in rhs_da.dims:
            if not rhs_da.get_index(d).equals(payload.indexes[str(d)]):
                raise ValueError(
                    f"Coordinate mismatch on shared dimension {d!r} between "
                    "the rhs and the grouped result. Align the rhs with "
                    ".sel(...) / .reindex(...) before combining (§8)."
                )
    rhs_da = rhs_da.reindex(
        {d: payload.indexes[str(d)] for d in rhs_da.dims}, fill_value=np.nan
    )
    missing = {d: payload.indexes[d] for d in payload.grid_dims if d not in rhs_da.dims}
    if missing:
        rhs_da = rhs_da.expand_dims(missing)
    return rhs_da.transpose(*payload.grid_dims).to_numpy().reshape(-1)


def realize_csr_constraint(
    model: Model, payload: CSRPayload, sign: str, rhs: Any, name: str
) -> CSRConstraint:
    """
    Staple sign and rhs onto a CSR-backed lhs to form a CSRConstraint.

    Label columns are mapped to dense variable positions, the payload's
    constant moves to the rhs, labels are allocated as in
    ``Model._allocate_constraint_labels``, and rows without terms or with a
    NaN rhs are inactive — all as on the frozen dense path.
    """
    from linopy.common import maybe_replace_sign
    from linopy.constraints import CSRConstraint

    sign = maybe_replace_sign(sign)
    full_size = payload.n_cells

    label_index = model.variables.label_index
    coo = payload.csr.tocoo()
    csr = scipy.sparse.csr_array(
        scipy.sparse.coo_array(
            (coo.data, (coo.coords[0], label_index.label_to_pos[coo.coords[1]])),
            shape=(full_size, label_index.n_active_vars),
        )
    )
    csr.eliminate_zeros()

    rhs_flat = _rhs_grid_values(payload, rhs) - payload.const

    cindex = model._cCounter
    model._cCounter += full_size
    active = (np.diff(csr.indptr) > 0) & ~np.isnan(rhs_flat)

    return CSRConstraint(
        csr[active],
        np.arange(cindex, cindex + full_size)[active],
        rhs_flat[active],
        sign,
        coords=[payload.indexes[d] for d in payload.grid_dims],
        model=model,
        name=name,
        cindex=cindex,
    )
