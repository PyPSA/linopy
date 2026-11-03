#!/usr/bin/env python3
"""
Created on Mon Oct 10 13:33:55 2022.

@author: fabian
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property, partial
from typing import TYPE_CHECKING, NamedTuple, TypedDict

import numpy as np
import scipy.sparse
from numpy import ndarray

from linopy import expressions
from linopy.csr import index_dtype, keep_entries, repeat_per_row

if TYPE_CHECKING:
    from linopy.common import VariableLabelIndex
    from linopy.constraints import Constraint
    from linopy.model import Model


#: Solvers whose dual sign convention was verified against ordinary rows for
#: recovering the duals of rows folded into bounds (``as_bound=True``).
BOUND_ROW_DUAL_SOLVERS = frozenset({"highs", "gurobi", "cplex", "xpress", "copt"})


class _RowBlock(NamedTuple):
    """
    Active rows of one constraint. ``positional`` returns their positional
    CSR, built only while the rows are gathered; ``nnz`` bounds its nonzeros.
    """

    positional: Callable[[], scipy.sparse.csr_array]
    n_rows: int
    nnz: int
    rhs: ndarray
    sign: ndarray
    scaling: ndarray


def _non_unit(scaling: ndarray) -> ndarray | None:
    """``scaling`` where any factor differs from 1, else None."""
    return None if (scaling == 1).all() else scaling


def _concat(arrays: list[ndarray]) -> ndarray:
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=float)


def _row_block(
    con: Constraint, active: ndarray, label_index: VariableLabelIndex
) -> _RowBlock:
    store = con.lhs._store
    return _RowBlock(
        partial(store.positional_csr, label_index, active, con.name),
        int(active.sum()),
        store.nnz(active),
        con.rhs.values.ravel()[active],
        con.sign.values.ravel()[active].astype("U1"),
        con.scaling.values.ravel()[active],
    )


def _stack(
    blocks: list[_RowBlock],
    label_index: VariableLabelIndex,
    col_scaling: ndarray | None,
    model: Model,
) -> tuple[scipy.sparse.csr_array | None, ndarray, ndarray]:
    """
    Gather the scaled row blocks into one preallocated CSR, rhs and sense.

    With solver variables y = Scol * x, constraints A x = b become
    Srow * A * Scol^-1 * y = Srow * b.

    Explicit zeros are dropped: expressions that broadcast against a dense
    coordinate store one coefficient per pair, most of them zero, and a zero
    coefficient never changes a constraint. Keeping them only inflates the
    stored nnz handed to the solvers/writers (e.g. ``highspy.addRows`` scales
    with stored nnz), so we prune them once, centrally, for every backend.
    """
    if not blocks:
        return None, np.array([]), np.array([], dtype=object)
    nnz = sum(block.nnz for block in blocks)
    n_rows = sum(block.n_rows for block in blocks)
    n_cols = label_index.n_active_vars
    dtype = index_dtype(nnz, (n_rows, n_cols), model)
    data = np.empty(nnz, dtype=float)
    indices = np.empty(nnz, dtype=dtype)
    indptr = np.empty(n_rows + 1, dtype=dtype)
    b = np.empty(n_rows, dtype=float)
    sense = np.empty(n_rows, dtype="U1")
    indptr[0] = 0
    pos = row = 0
    for block in blocks:
        source = block.positional()
        count = int(np.count_nonzero(source.data))
        block_data, block_indices, block_indptr = (
            source.data,
            source.indices,
            source.indptr,
        )
        if count < source.nnz:
            block_data, block_indices, block_indptr = keep_entries(
                block_data, block_indices, block_indptr, block_data != 0
            )
        rows = slice(row, row + source.shape[0])
        entries = slice(pos, pos + count)
        row_scaling = _non_unit(block.scaling)
        if row_scaling is None:
            data[entries] = block_data
            b[rows] = block.rhs
        else:
            per_row = repeat_per_row(row_scaling, block_indptr)
            np.multiply(block_data, per_row, out=data[entries])
            np.multiply(block.rhs, row_scaling, out=b[rows])
        if col_scaling is not None:
            data[entries] /= col_scaling[block_indices]
        indices[entries] = block_indices
        block_offsets = indptr[rows.start + 1 : rows.stop + 1]
        block_offsets[:] = block_indptr[1:]
        block_offsets += pos
        sense[rows] = block.sign
        pos, row = entries.stop, rows.stop
    A = scipy.sparse.csr_array(
        (data[:pos], indices[:pos], indptr), shape=(n_rows, n_cols)
    )
    return A, b, sense


@dataclass(frozen=True, eq=False)
class BoundRows:
    """
    Constraint rows lowered to column bounds (``as_bound=True``).

    Each folded row ``coeffs[k] * x[cols[k]] (sign) rhs`` tightened the
    bounds of column ``cols[k]`` (a position aligned with ``vlabels``) to
    ``rhs / coeffs[k]``. ``lower_row`` and ``upper_row`` hold, per column,
    the label of the folded row that sets the lower / upper bound, or -1 where
    the variable's own bound is tightest. Coefficients are in user units.
    """

    labels: ndarray
    cols: ndarray
    coeffs: ndarray
    lower_row: ndarray
    upper_row: ndarray


def _tightest(
    cols: ndarray, values: ndarray, labels: ndarray, bound: ndarray, lower: bool
) -> ndarray:
    """
    Tighten ``bound`` in place with ``values`` on ``cols``; return the label
    of the row that sets each bound, -1 where no row does. A row wins over an
    equal bound of the variable itself, the first row over later equal rows.
    """
    keep = np.isfinite(values)
    cols, values, labels = cols[keep], values[keep], labels[keep]
    (np.maximum if lower else np.minimum).at(bound, cols, values)
    win = np.flatnonzero(values == bound[cols])
    first = np.full(bound.size, labels.size)
    np.minimum.at(first, cols[win], win)
    return np.append(labels, -1)[first]


def _fold_bound_rows(
    m: Model, lb: ndarray, ub: ndarray, vtypes: ndarray
) -> BoundRows | None:
    """
    Tighten ``lb`` and ``ub`` (user units, aligned with vlabels) in place with
    the rows of constraints added with ``as_bound=True``.
    """
    cons = [c for c in m.constraints.data.values() if c.as_bound]
    if not cons:
        return None
    label_index = m.variables.label_index
    parts = []
    for con in cons:
        active = con.active_row_mask()
        block = _row_block(con, active, label_index)
        csr = block.positional()
        data, cols, indptr = keep_entries(
            csr.data, csr.indices, csr.indptr, csr.data != 0
        )
        con.check_bound_rows(np.diff(indptr))
        parts.append((con._label_values()[active], cols, data, block.rhs, block.sign))
    labels, cols, coeffs, rhs, sign = (
        np.concatenate([p[i] for p in parts]) for i in range(5)
    )
    cols = cols.astype(np.intp)
    semi = vtypes[cols] == "S"
    if semi.any():
        raise ValueError(
            "Constraints lowered to bounds (as_bound=True) cannot act on "
            "semi-continuous variables, since a bound changes their semantics; "
            f"see constraint label {int(labels[semi][0])}."
        )
    values = rhs / coeffs
    flip, equal = coeffs < 0, sign == "="
    lower = equal | ((sign == ">") != flip)
    upper = equal | ((sign == "<") != flip)
    return BoundRows(
        labels=labels,
        cols=cols,
        coeffs=coeffs,
        lower_row=_tightest(cols[lower], values[lower], labels[lower], lb, True),
        upper_row=_tightest(cols[upper], values[upper], labels[upper], ub, False),
    )


def _label_take(values: ndarray, labels: ndarray) -> ndarray:
    """``values[labels]``, NaN where a label lies beyond ``values``."""
    out = np.full(labels.size, np.nan)
    inside = labels < values.size
    out[inside] = values[labels[inside]]
    return out


def bound_row_duals(M: Matrices, primal: ndarray, dual: ndarray) -> ndarray:
    """
    Duals of the rows folded into bounds, in user units, aligned with
    ``M.bound_rows.labels``.

    ``primal`` and ``dual`` are the label-indexed solver-unit arrays of a
    :class:`~linopy.constants.Solution`. The reduced costs of the columns are
    reconstructed from the duals of the remaining rows as
    ``d = c + Q x - A^T y``, the stationarity condition under linopy's dual
    sign convention. The row setting the active bound of a column then
    carries ``d_j / coeff``, which is the dual the same row would get as an
    ordinary constraint; every other folded row is not binding and gets 0.
    """
    rows = M.bound_rows
    if rows is None:
        return np.empty(0)
    x = _label_take(primal, M.vlabels)
    gradient = M.c.astype(float, copy=True)
    if M.Q is not None:
        gradient += M.Q @ x
    if M.A is not None and M.A.shape[0]:
        gradient -= M.A.T @ _label_take(dual, M.clabels)
    reduced = gradient * M.var_scaling / M.objective_scaling
    d = reduced[rows.cols]
    direction = d if M.objective_sense == "min" else -d
    at_lower = (direction > 0) & (rows.lower_row[rows.cols] == rows.labels)
    at_upper = (direction < 0) & (rows.upper_row[rows.cols] == rows.labels)
    result = np.where(at_lower | at_upper, d / rows.coeffs, 0.0)
    result[np.isnan(d)] = np.nan
    return result


def _indicator_rows(con: Constraint, active: ndarray) -> tuple[ndarray, ndarray]:
    """Binary variable labels and triggering values of the active indicator rows."""
    if con.binary_var is None or con.binary_val is None:
        raise ValueError(f"Constraint {con.name!r} is not an indicator constraint.")
    var_labels = con.binary_var.values.ravel()[active]
    return var_labels, np.ravel(con.binary_val)[active].astype(np.intp)


@dataclass(frozen=True, eq=False)
class Matrices:
    """
    A model as flat arrays.

    All arrays are compact — only active (non-masked) entries are included.
    Position i in variable-side arrays corresponds to vlabels[i].
    Position i in constraint-side arrays corresponds to clabels[i].

    The matrix members (``A``, ``b``, ``c``, ``Q``, ``lb``, ``ub``) are in
    solver-scaled units. ``con_scaling`` is the row scaling applied to ``A``,
    aligned with ``clabels``; consumers that report duals need it to invert the
    scaling on the way back.

    Rows of constraints added with ``as_bound=True`` are not part of ``A`` and
    ``clabels``; they are folded into ``lb`` and ``ub``, and ``bound_rows``
    records which row sets which bound, see :class:`BoundRows`.
    """

    vlabels: ndarray
    clabels: ndarray
    A: scipy.sparse.csr_array | None
    b: ndarray
    sense: ndarray
    c: ndarray
    Q: scipy.sparse.csc_matrix | None
    lb: ndarray
    ub: ndarray
    vtypes: ndarray
    var_scaling: ndarray
    con_scaling: ndarray
    objective_scaling: float
    objective_sense: str
    indicator_A: scipy.sparse.csr_array | None
    indicator_b: ndarray
    indicator_sense: ndarray
    indicator_binvar: ndarray
    indicator_binval: ndarray
    bound_rows: BoundRows | None = None

    @property
    def type(self) -> str:
        """Short string identifying the problem type, as :attr:`Model.type`."""
        discrete = self.vtypes != "C"
        variable_type = ("I" if discrete.all() else "MI") if discrete.any() else ""
        return f"{variable_type}{'L' if self.Q is None else 'Q'}P"

    def take(self, rows: ndarray, cols: ndarray) -> Matrices:
        """
        The matrices restricted to the given row and column positions.

        Raises when indicator constraints are present, since their binary
        variables would have to be remapped onto the new columns.
        """
        if self.indicator_b.size:
            raise NotImplementedError(
                "Taking rows and columns of matrices with indicator constraints "
                "is not supported."
            )
        return Matrices(
            vlabels=self.vlabels[cols],
            clabels=self.clabels[rows],
            A=None if self.A is None else self.A[rows][:, cols],
            b=self.b[rows],
            sense=self.sense[rows],
            c=self.c[cols],
            Q=None if self.Q is None else self.Q[cols][:, cols],
            lb=self.lb[cols],
            ub=self.ub[cols],
            vtypes=self.vtypes[cols],
            var_scaling=self.var_scaling[cols],
            con_scaling=self.con_scaling[rows],
            objective_scaling=self.objective_scaling,
            objective_sense=self.objective_sense,
            indicator_A=None,
            indicator_b=self.indicator_b,
            indicator_sense=self.indicator_sense,
            indicator_binvar=self.indicator_binvar,
            indicator_binval=self.indicator_binval,
            # windows keep the folded bounds in lb/ub; no dual recovery there
            bound_rows=None,
        )


class _VarArrays(TypedDict):
    vlabels: ndarray
    var_scaling: ndarray
    lb: ndarray
    ub: ndarray
    vtypes: ndarray
    bound_rows: BoundRows | None


class _ConArrays(TypedDict):
    clabels: ndarray
    con_scaling: ndarray
    A: scipy.sparse.csr_array | None
    b: ndarray
    sense: ndarray
    indicator_A: scipy.sparse.csr_array | None
    indicator_b: ndarray
    indicator_sense: ndarray
    indicator_binvar: ndarray
    indicator_binval: ndarray


def _build_vars(m: Model) -> _VarArrays:
    vlabels = m.variables.label_index.vlabels
    n = len(vlabels)
    var_scaling = np.empty(n, dtype=float)
    lb = np.empty(n, dtype=float)
    ub = np.empty(n, dtype=float)
    vtypes = np.empty(n, dtype="U1")

    pos = 0
    for name, var in m.variables.items():
        mask = var.labels.values.ravel() != -1
        cols = slice(pos, pos + int(np.count_nonzero(mask)))
        if name in m.binaries:
            vtype = "B"
        elif name in m.integers:
            vtype = "I"
        elif name in m.semi_continuous:
            vtype = "S"
        else:
            vtype = "C"
        vtypes[cols] = vtype
        var_scaling[cols] = var.solver_scaling.values.ravel()[mask]
        lb[cols] = var.lower.values.ravel()[mask]
        ub[cols] = var.upper.values.ravel()[mask]
        pos = cols.stop

    bound_rows = _fold_bound_rows(m, lb, ub, vtypes)
    return _VarArrays(
        vlabels=vlabels,
        var_scaling=var_scaling,
        lb=lb,
        ub=ub,
        vtypes=vtypes,
        bound_rows=bound_rows,
    )


def folded_bounds(m: Model) -> tuple[ndarray, ndarray] | None:
    """
    Label-indexed lower and upper bounds in user units with the rows of
    ``as_bound=True`` constraints folded in, or None without such rows.
    """
    if not any(c.as_bound for c in m.constraints.data.values()):
        return None
    arrays = _build_vars(m)
    lower, upper = np.full(m._xCounter, np.nan), np.full(m._xCounter, np.nan)
    lower[arrays["vlabels"]] = arrays["lb"]
    upper[arrays["vlabels"]] = arrays["ub"]
    return lower, upper


def _build_cons(m: Model, col_scaling: ndarray | None) -> _ConArrays:
    label_index = m.variables.label_index
    label_to_pos = label_index.label_to_pos

    regular, indicator = [], []
    binvar, binval = [], []
    for c in m.constraints.data.values():
        if c.as_bound:
            continue
        active = c.active_row_mask()
        block = _row_block(c, active, label_index)
        if not c.is_indicator:
            regular.append(block)
            continue
        indicator.append(block)
        var_labels, val = _indicator_rows(c, active)
        binvar.append(label_to_pos[var_labels])
        binval.append(val)

    A, b, sense = _stack(regular, label_index, col_scaling, m)
    indicator_A, indicator_b, indicator_sense = _stack(
        indicator, label_index, col_scaling, m
    )
    label_dtype = m._dtypes["labels"]
    return _ConArrays(
        clabels=m.constraints.label_index.clabels,
        con_scaling=_concat([block.scaling for block in regular]),
        A=A,
        b=b,
        sense=sense,
        indicator_A=indicator_A,
        indicator_b=indicator_b,
        indicator_sense=indicator_sense,
        indicator_binvar=np.concatenate([np.array([], dtype=label_dtype), *binvar]),
        indicator_binval=(
            np.concatenate(binval) if binval else np.array([], dtype=np.intp)
        ),
    )


def _objective(
    m: Model, vlabels: ndarray, var_scaling: ndarray
) -> tuple[ndarray, scipy.sparse.csc_matrix | None]:
    """Linear coefficients and quadratic matrix of the objective, aligned with vlabels."""
    c = np.zeros(len(vlabels))
    var_labels, coeffs = m.objective.linear_terms()
    positions = m.variables.label_index.label_to_pos[var_labels]
    np.add.at(c, positions, coeffs / var_scaling[positions] * m.objective.scaling)

    expr = m.objective.expression
    if not isinstance(expr, expressions.QuadraticExpression):
        return c, None
    q = expr.to_matrix()[vlabels][:, vlabels]
    if q.nnz:
        q = q.multiply(1 / var_scaling[:, np.newaxis])
        q = q.multiply(1 / var_scaling[np.newaxis, :])
        q = q * m.objective.scaling
    return c, q.tocsc()


class MatrixAccessor(Matrices):
    """
    The flat matrices of a model, with read-back of its solution and duals.

    The read-back members (``sol``, ``dual``) are in user units, so ``A @ sol``
    and ``c @ sol`` do not reconstruct ``b`` or the objective when scaling
    factors differ from 1.
    """

    _parent: Model

    def __init__(self, model: Model) -> None:
        object.__setattr__(self, "_parent", model)
        variables = _build_vars(model)
        col_scaling = _non_unit(variables["var_scaling"])
        if col_scaling is not None:
            variables["lb"] *= col_scaling
            variables["ub"] *= col_scaling
        c, Q = _objective(model, variables["vlabels"], variables["var_scaling"])
        super().__init__(
            **variables,
            **_build_cons(model, col_scaling),
            c=c,
            Q=Q,
            objective_scaling=model.objective.scaling,
            objective_sense=model.objective.sense,
        )

    @cached_property
    def sol(self) -> ndarray:
        """Solution values aligned with vlabels."""
        if not self._parent.status == "ok":
            raise ValueError("Model is not optimized.")
        values = [
            var.solution.values.ravel()[var.labels.values.ravel() != -1]
            for _, var in self._parent.variables.items()
        ]
        result = _concat(values)
        if result.size != len(self.vlabels):
            raise ValueError(
                f"Solution holds {result.size} active values, expected "
                f"{len(self.vlabels)} aligned with vlabels."
            )
        return result

    @cached_property
    def dual(self) -> ndarray:
        """Dual values aligned with clabels."""
        if not self._parent.status == "ok":
            raise ValueError("Model is not optimized.")
        m = self._parent
        dual_list = []
        has_dual = False
        for c in m.constraints.data.values():
            if c.is_indicator or c.as_bound:
                continue
            active = c.active_row_mask()
            if "dual" in c:
                dual_list.append(c.dual.values.ravel()[active])
                has_dual = True
            else:
                dual_list.append(np.full(int(active.sum()), np.nan))
        if not has_dual:
            raise AttributeError(
                "Underlying is optimized but does not have dual values stored."
            )
        return np.concatenate(dual_list) if dual_list else np.array([])
