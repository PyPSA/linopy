#!/usr/bin/env python3
"""
Created on Mon Oct 10 13:33:55 2022.

@author: fabian
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cached_property, partial
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
import scipy.sparse
from numpy import ndarray

from linopy import expressions
from linopy.constraints import CSRConstraint
from linopy.csr import index_dtype

if TYPE_CHECKING:
    from linopy.common import VariableLabelIndex
    from linopy.constraints import ConstraintBase
    from linopy.model import Model


class _RowBlock(NamedTuple):
    """
    Rows of one constraint. ``positional`` returns its positional CSR; for a
    frozen constraint it is built only while the rows are gathered.
    """

    positional: Callable[[], scipy.sparse.csr_array]
    n_rows: int
    nnz: int
    rhs: ndarray
    sign: str | ndarray
    row_scaling: ndarray | None


def _unit_or(scaling: ndarray) -> ndarray | None:
    return None if (scaling == 1).all() else scaling


def _row_block(con: ConstraintBase, label_index: VariableLabelIndex) -> _RowBlock:
    if isinstance(con, CSRConstraint):
        stored = con._csr
        return _RowBlock(
            partial(con._to_positional_csr, label_index),
            stored.shape[0],
            int(np.count_nonzero(stored.data)),
            con._rhs,
            con._sign,
            _unit_or(con._scaling),
        )
    csr, _, rhs, sense = con.to_matrix_with_rhs(label_index)
    scaling = _unit_or(con.scaling.values.ravel())
    return _RowBlock(
        lambda: csr,
        csr.shape[0],
        int(np.count_nonzero(csr.data)),
        rhs,
        sense,
        None if scaling is None else scaling[con.active_row_mask()],
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
        count = block.nnz
        block_data, block_indices, block_indptr = (
            source.data,
            source.indices,
            source.indptr,
        )
        if count < source.nnz:
            keep = block_data != 0
            block_data, block_indices = block_data[keep], block_indices[keep]
            block_indptr = np.concatenate([[0], np.cumsum(keep)])[block_indptr]
        rows = slice(row, row + source.shape[0])
        entries = slice(pos, pos + count)
        if block.row_scaling is None:
            data[entries] = block_data
            b[rows] = block.rhs
        else:
            row_counts = np.diff(block_indptr)
            np.multiply(
                block_data, np.repeat(block.row_scaling, row_counts), out=data[entries]
            )
            np.multiply(block.rhs, block.row_scaling, out=b[rows])
        if col_scaling is not None:
            data[entries] /= col_scaling[block_indices]
        indices[entries] = block_indices
        block_offsets = indptr[rows.start + 1 : rows.stop + 1]
        block_offsets[:] = block_indptr[1:]
        block_offsets += pos
        sense[rows] = block.sign
        pos, row = entries.stop, rows.stop
    A = scipy.sparse.csr_array((data, indices, indptr), shape=(n_rows, n_cols))
    return A, b, sense


def _binval_per_row(binval: int | np.ndarray, n: int) -> ndarray:
    """Broadcast an indicator triggering value to one entry per active row."""
    if np.ndim(binval) == 0:
        return np.full(n, int(binval), dtype=np.intp)
    return np.asarray(binval, dtype=np.intp).ravel()


class MatrixAccessor:
    """
    Helper class to quickly access model related vectors and matrices.

    All arrays are compact — only active (non-masked) entries are included.
    Position i in variable-side arrays corresponds to vlabels[i].
    Position i in constraint-side arrays corresponds to clabels[i].

    The matrix members (``A``, ``b``, ``c``, ``Q``, ``lb``, ``ub``) are in
    solver-scaled units while the read-back members (``sol``, ``dual``) are in
    user units, so ``A @ sol`` and ``c @ sol`` do not reconstruct ``b`` or the
    objective when scaling factors differ from 1.
    """

    def __init__(self, model: Model) -> None:
        self._parent = model
        self._build_vars()
        self._build_cons()

    def _build_vars(self) -> None:
        m = self._parent
        self.vlabels: ndarray = m.variables.label_index.vlabels
        n = len(self.vlabels)
        self.var_scaling: ndarray = np.empty(n, dtype=float)
        self.lb: ndarray = np.empty(n, dtype=float)
        self.ub: ndarray = np.empty(n, dtype=float)
        self.vtypes: ndarray = np.empty(n, dtype="U1")

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
            self.vtypes[cols] = vtype
            self.var_scaling[cols] = var.solver_scaling.values.ravel()[mask]
            self.lb[cols] = var.lower.values.ravel()[mask]
            self.ub[cols] = var.upper.values.ravel()[mask]
            pos = cols.stop

        if not (self.var_scaling == 1).all():
            self.lb *= self.var_scaling
            self.ub *= self.var_scaling

    def _build_cons(self) -> None:
        m = self._parent
        label_index = m.variables.label_index
        label_to_pos = label_index.label_to_pos
        col_scaling = None if (self.var_scaling == 1).all() else self.var_scaling

        regular, indicator = [], []
        binvar, binval = [], []
        for c in m.constraints.data.values():
            if not c.is_indicator:
                regular.append(_row_block(c, label_index))
                continue
            cc = c if isinstance(c, CSRConstraint) else c.freeze()
            indicator.append(_row_block(cc, label_index))
            binvar.append(label_to_pos[cc._binvar_labels])
            cc_binval = cast("int | np.ndarray", cc._binval)
            binval.append(_binval_per_row(cc_binval, len(cc._rhs)))

        self.clabels: ndarray = m.constraints.label_index.clabels
        self.A: scipy.sparse.csr_array | None
        self.A, self.b, self.sense = _stack(regular, label_index, col_scaling, m)
        self.indicator_A: scipy.sparse.csr_array | None
        self.indicator_A, self.indicator_b, self.indicator_sense = _stack(
            indicator, label_index, col_scaling, m
        )
        label_dtype = m._dtypes["labels"]
        self.indicator_binvar: ndarray = np.concatenate(
            [np.array([], dtype=label_dtype), *binvar]
        )
        self.indicator_binval: ndarray = (
            np.concatenate(binval) if binval else np.array([], dtype=np.intp)
        )

    @cached_property
    def c(self) -> ndarray:
        """Objective coefficients aligned with vlabels."""
        m = self._parent
        result = np.zeros(len(self.vlabels))

        label_index = m.variables.label_index
        label_to_pos = label_index.label_to_pos
        var_labels, coeffs = m.objective.linear_terms()
        positions = label_to_pos[var_labels]
        scaled_coeffs = coeffs / self.var_scaling[positions]
        scaled_coeffs = scaled_coeffs * m.objective.scaling
        np.add.at(result, positions, scaled_coeffs)
        return result

    @cached_property
    def Q(self) -> scipy.sparse.csc_matrix | None:
        """Quadratic objective matrix, shape (n_active_vars, n_active_vars)."""
        m = self._parent
        expr = m.objective.expression
        if not isinstance(expr, expressions.QuadraticExpression):
            return None
        q = expr.to_matrix()[self.vlabels][:, self.vlabels]
        if q.nnz:
            # Quadratic coefficients get one inverse column scaling per factor.
            q = q.multiply(1 / self.var_scaling[:, np.newaxis])
            q = q.multiply(1 / self.var_scaling[np.newaxis, :])
            q = q * self._parent.objective.scaling
        return q.tocsc()

    @cached_property
    def sol(self) -> ndarray:
        """Solution values aligned with vlabels."""
        if not self._parent.status == "ok":
            raise ValueError("Model is not optimized.")
        m = self._parent
        result = np.full(len(self.vlabels), np.nan)
        label_index = m.variables.label_index
        label_to_pos = label_index.label_to_pos
        for _, var in m.variables.items():
            labels = var.labels.values.ravel()
            mask = labels != -1
            positions = label_to_pos[labels[mask]]
            result[positions] = var.solution.values.ravel()[mask]
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
            if isinstance(c, CSRConstraint):
                # _dual is active-only
                if c._dual is not None:
                    dual_list.append(c._dual)
                    has_dual = True
                else:
                    dual_list.append(np.full(c.ncons, np.nan))
            else:
                active_rows = np.flatnonzero(c.active_row_mask())
                if "dual" in c.data:
                    dual_list.append(c.dual.values.ravel()[active_rows])
                    has_dual = True
                else:
                    dual_list.append(np.full(len(active_rows), np.nan))
        if not has_dual:
            raise AttributeError(
                "Underlying is optimized but does not have dual values stored."
            )
        return np.concatenate(dual_list) if dual_list else np.array([])
