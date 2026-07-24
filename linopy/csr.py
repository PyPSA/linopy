"""
CSR-backed expressions: a sparse internal representation for grouped sums.

``expr.groupby(g).sum(sparse=True)`` (or ``linopy.options["sparse_groupby"]
= True`` under v1) returns an ordinary
:class:`~linopy.expressions.LinearExpression` whose payload is a
:class:`CSRPayload` — the expression as a scipy CSR matrix (rows = flat
cells of the coordinate grid, columns = variable labels, values =
coefficients) plus a dense const vector — instead of the materialized
dense dataset. Modeled on dask-backed xarray objects: same public type,
different backing; any operation without a sparse branch transparently
converts to the dense form through the ``.data`` property.

The CSR form is canonical: duplicate variables within a cell are summed
(``2x + 3x -> 5x``) and terms are ordered by variable label, so the
``_term`` axis is ragged by construction — the group-size padding of issue
#745 has no analog. Operations become sparse linear algebra:

- grouping scatters members into group rows (conceptually ``G @ A`` with a
  0/1 grouping matrix),
- ``merge`` along ``_term`` — and therefore ``+``/``-`` — is sparse matrix
  addition,
- unary minus and scalar multiplication scale the values,
- ``== rhs`` with a constant is carried as a pending payload on the
  :class:`~linopy.constraints.Constraint`, and ``Model.add_constraints``
  with ``freeze=True`` staples sign and rhs on to produce a
  :class:`~linopy.constraints.CSRConstraint` directly.

Converting back to the dense rectangle yields the mathematically identical
expression in canonical form (not the eager kernel's exact term layout);
this is why the feature is gated behind v1 semantics, where the term
layout is explicitly non-contractual.

Columns are raw variable labels (not positions), so payloads stay valid
when variables are added to the model later; realization maps labels to
dense positions via the model's label index.
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

    ``csr`` has one row per flat grid cell (C order over ``grid_dims``)
    and one column per variable label; ``const`` is the per-cell constant.
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
        """Build the grouped sum directly in CSR form (no padded rectangle)."""
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
        return _assemble(expr, grid_dims, indexes, group_dim, member_dim, codes)

    @classmethod
    def from_expression(
        cls, expr: LinearExpression, template: CSRPayload
    ) -> CSRPayload | None:
        """Convert a dense expression on the template's grid to CSR form."""
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
        a, b = self.csr, other.csr
        width = max(a.shape[1], b.shape[1])
        a, b = _widen(a, width), _widen(b, width)
        return replace(self, csr=a + b, const=self.const + other.const)

    def materialize(self) -> LinearExpression:
        """
        Expand to the dense rectangle in canonical form.

        Terms are variable-label-ordered and duplicates are summed; the
        term axis is padded to the widest cell with the usual fill
        (vars=-1, coeffs=NaN).
        """
        from linopy.expressions import LinearExpression

        csr = self.csr.copy()
        csr.eliminate_zeros()
        csr.sort_indices()
        lengths = np.diff(csr.indptr)
        nterm = max(int(lengths.max()) if len(lengths) else 0, 1)

        labels_dtype = self.model._dtypes["labels"]
        vars_flat = np.full((self.n_cells, nterm), -1, dtype=labels_dtype)
        coeffs_flat = np.full((self.n_cells, nterm), np.nan)
        rows = np.repeat(np.arange(self.n_cells), lengths)
        pos = np.arange(csr.nnz) - np.repeat(csr.indptr[:-1], lengths)
        vars_flat[rows, pos] = csr.indices
        coeffs_flat[rows, pos] = csr.data

        shape = self.shape
        dims = (*self.grid_dims, TERM_DIM)
        coords = {d: self.indexes[d] for d in self.grid_dims}
        ds = Dataset(
            {
                "coeffs": (dims, coeffs_flat.reshape(*shape, nterm)),
                "vars": (dims, vars_flat.reshape(*shape, nterm)),
                "const": (self.grid_dims, self.const.reshape(shape)),
            },
            coords=coords,
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
    Scatter an expression's terms into grid rows (conceptually G @ A).

    ``expr``'s ``member_dim`` scatters into the grid dim ``scatter_dim``
    at the row positions given by ``codes``; every other grid dim maps
    one-to-one.
    """
    ds = expr.data
    shape = tuple(len(indexes[d]) for d in grid_dims)
    strides = np.array(
        [int(np.prod(shape[i + 1 :], dtype=np.int64)) for i in range(len(shape))]
    )

    transposed: list[str] = []
    axis_positions: list[np.ndarray] = []
    for i, (d, stride) in enumerate(zip(grid_dims, strides)):
        if d == scatter_dim:
            transposed.append(member_dim)
            axis_positions.append(codes * stride)
        else:
            transposed.append(d)
            axis_positions.append(np.arange(shape[i]) * stride)

    cell_rows = axis_positions[0]
    for pos in axis_positions[1:]:
        cell_rows = cell_rows[..., None] + pos
    cell_rows = cell_rows.reshape(-1)

    coeffs = ds.coeffs.transpose(*transposed, TERM_DIM).to_numpy().reshape(-1)
    vars_ = ds.vars.transpose(*transposed, TERM_DIM).to_numpy().reshape(-1)
    nterm = ds.sizes[TERM_DIM]
    rows = np.repeat(cell_rows, nterm)

    keep = (vars_ != -1) & (coeffs != 0) & ~np.isnan(coeffs)
    full_size = int(np.prod(shape, dtype=np.int64)) if shape else 1
    width = expr.model._xCounter
    coo = scipy.sparse.coo_array(
        (coeffs[keep], (rows[keep], vars_[keep])), shape=(full_size, width)
    )
    csr = scipy.sparse.csr_array(coo)  # sums duplicates == the group sum

    const_vals = ds.const.transpose(*transposed).to_numpy().reshape(-1)
    const = np.zeros(full_size)
    np.add.at(const, cell_rows, np.where(np.isnan(const_vals), 0.0, const_vals))

    return CSRPayload(csr, const, grid_dims, indexes, expr.model)


def try_csr_merge(
    exprs: Any, dim: str, join: Any, kwargs: dict
) -> LinearExpression | None:
    """
    Sparse branch of :func:`linopy.expressions.merge`.

    Returns a CSR-backed combined expression when every input is a plain
    LinearExpression on the same result grid (CSR-backed, or dense and
    convertible) — where any join produces the identical result — and None
    otherwise to fall through to the dense path.
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
        payload = e._csr
        if payload is None:
            payload = CSRPayload.from_expression(e, template)
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
        expr, sign, rhs = lhs._pending
        if expr._csr is None:  # already materialized elsewhere
            return None
        lhs = expr
    if not (isinstance(lhs, LinearExpression) and lhs._csr is not None):
        return None
    if not isinstance(sign, str) or rhs is None or not is_constant(rhs):
        return None
    payload = lhs._csr
    rhs_da = _as_rhs_dataarray(rhs)
    if rhs_da is None or not set(rhs_da.dims) <= set(payload.grid_dims):
        return None
    return payload, sign, rhs


def _as_rhs_dataarray(rhs: Any) -> DataArray | None:
    from linopy.alignment import as_dataarray

    try:
        da = as_dataarray(rhs)
    except (TypeError, ValueError):
        return None
    if set(da.dims) & set(HELPER_DIMS):
        return None
    return da


def realize_csr_constraint(
    model: Model, payload: CSRPayload, sign: str, rhs: Any, name: str
) -> CSRConstraint:
    """
    Staple sign and rhs onto a CSR-backed lhs to form a CSRConstraint.

    Equivalent to materializing the sparse sum, adding the constraint and
    freezing it — but the padded dense ``_term`` rectangle never exists;
    peak memory is proportional to the number of nonzero terms.
    """
    from linopy.common import maybe_replace_sign
    from linopy.constraints import CSRConstraint
    from linopy.semantics import check_user_nan, is_v1

    sign = maybe_replace_sign(sign)
    full_size = payload.n_cells

    # map label columns to dense positions in the active variable array
    label_index = model.variables.label_index
    label_to_pos = label_index.label_to_pos
    coo = payload.csr.tocoo()
    csr = scipy.sparse.csr_array(
        scipy.sparse.coo_array(
            (coo.data, (coo.coords[0], label_to_pos[coo.coords[1]])),
            shape=(full_size, label_index.n_active_vars),
        )
    )
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
    rhs_flat = (
        rhs_da.transpose(*payload.grid_dims).to_numpy().reshape(-1) - payload.const
    )

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
        coords=[payload.indexes[d] for d in payload.grid_dims],
        model=model,
        name=name,
        cindex=cindex,
    )
