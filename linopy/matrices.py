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
        )


class _VarArrays(TypedDict):
    vlabels: ndarray
    var_scaling: ndarray
    lb: ndarray
    ub: ndarray
    vtypes: ndarray


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

    if _non_unit(var_scaling) is not None:
        lb *= var_scaling
        ub *= var_scaling
    return _VarArrays(
        vlabels=vlabels, var_scaling=var_scaling, lb=lb, ub=ub, vtypes=vtypes
    )


def _build_cons(m: Model, col_scaling: ndarray | None) -> _ConArrays:
    label_index = m.variables.label_index
    label_to_pos = label_index.label_to_pos

    regular, indicator = [], []
    binvar, binval = [], []
    for c in m.constraints.data.values():
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
        c, Q = _objective(model, variables["vlabels"], variables["var_scaling"])
        super().__init__(
            **variables,
            **_build_cons(model, _non_unit(variables["var_scaling"])),
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
            if c.is_indicator:
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
