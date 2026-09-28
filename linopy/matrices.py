#!/usr/bin/env python3
"""
Created on Mon Oct 10 13:33:55 2022.

@author: fabian
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, cast

import numpy as np
import scipy.sparse
from numpy import ndarray

from linopy import expressions
from linopy.constraints import CSRConstraint
from linopy.scaling import constraint_scaling_lookup, variable_scaling_lookup

if TYPE_CHECKING:
    from linopy.model import Model


def _stack(csrs: list) -> scipy.sparse.csr_array | None:
    """
    Vertically stack CSR blocks, or None when there are none.

    Explicit zeros are dropped: expressions that broadcast against a dense
    coordinate store one coefficient per pair, most of them zero, and a zero
    coefficient never changes a constraint. Keeping them only inflates the
    stored nnz handed to the solvers/writers (e.g. ``highspy.addRows`` scales
    with stored nnz), so we prune them once, centrally, for every backend.
    """
    if not csrs:
        return None
    stacked = cast(scipy.sparse.csr_array, scipy.sparse.vstack(csrs, format="csr"))
    stacked.eliminate_zeros()
    return stacked


def _concat(arrays: list, dtype: type | None = None) -> ndarray:
    """Concatenate arrays, or an empty array when there are none."""
    return np.concatenate(arrays) if arrays else np.array([], dtype=dtype)


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
        label_index = m.variables.label_index
        self.vlabels: ndarray = label_index.vlabels
        var_scaling_by_label = variable_scaling_lookup(m)
        self.var_scaling: ndarray = (
            var_scaling_by_label[self.vlabels]
            if len(self.vlabels)
            else np.array([], dtype=float)
        )

        lb_list = []
        ub_list = []
        vtypes_list = []

        for name, var in m.variables.items():
            labels = var.labels.values.ravel()
            mask = labels != -1

            if name in m.binaries:
                vtype = "B"
            elif name in m.integers:
                vtype = "I"
            elif name in m.semi_continuous:
                vtype = "S"
            else:
                vtype = "C"

            lb_list.append(var.lower.values.ravel()[mask])
            ub_list.append(var.upper.values.ravel()[mask])
            vtypes_list.append(np.full(mask.sum(), vtype))

        if lb_list:
            self.lb: ndarray = np.concatenate(lb_list) * self.var_scaling
            self.ub: ndarray = np.concatenate(ub_list) * self.var_scaling
            self.vtypes: ndarray = np.concatenate(vtypes_list)
        else:
            self.lb = np.array([])
            self.ub = np.array([])
            self.vtypes = np.array([], dtype=object)

    def _build_cons(self) -> None:
        m = self._parent
        label_index = m.variables.label_index
        label_to_pos = label_index.label_to_pos
        con_scaling_by_label = constraint_scaling_lookup(m)
        unit_cols = bool((self.var_scaling == 1).all())

        def scale_rows_and_cols(
            csr: scipy.sparse.csr_array, con_labels: np.ndarray, b: np.ndarray
        ) -> tuple[scipy.sparse.csr_array, np.ndarray]:
            if csr.shape[0] == 0:
                return csr, b
            row_scaling = con_scaling_by_label[con_labels]
            unit_rows = bool((row_scaling == 1).all())
            if unit_rows and unit_cols:
                return csr, b
            # With solver variables y = Scol * x, constraints A x = b become
            # Srow * A * Scol^-1 * y = Srow * b.
            data = csr.data
            if not unit_rows:
                data = data * np.repeat(row_scaling, np.diff(csr.indptr))
            if not unit_cols:
                data = data / self.var_scaling[csr.indices]
            scaled = scipy.sparse.csr_array(
                (data, csr.indices, csr.indptr), shape=csr.shape
            )
            return scaled, b * row_scaling

        reg_csrs, reg_b, reg_sense = [], [], []
        ind_csrs, ind_b, ind_sense, ind_binvar, ind_binval = [], [], [], [], []
        for c in m.constraints.data.values():
            if c.is_indicator:
                cc = c if isinstance(c, CSRConstraint) else c.freeze()
                csr, con_labels, b, sense = cc.to_matrix_with_rhs(label_index)
                csr, b = scale_rows_and_cols(csr, con_labels, b)
                ind_csrs.append(csr)
                ind_b.append(b)
                ind_sense.append(sense)
                ind_binvar.append(label_to_pos[cc._binvar_labels])
                binval = cast("int | np.ndarray", cc._binval)
                ind_binval.append(_binval_per_row(binval, len(b)))
            else:
                csr, con_labels, b, sense = c.to_matrix_with_rhs(label_index)
                csr, b = scale_rows_and_cols(csr, con_labels, b)
                reg_csrs.append(csr)
                reg_b.append(b)
                reg_sense.append(sense)

        self.clabels: ndarray = m.constraints.label_index.clabels
        self.A: scipy.sparse.csr_array | None = _stack(reg_csrs)
        self.b: ndarray = _concat(reg_b)
        self.sense: ndarray = _concat(reg_sense, dtype=object)
        self.indicator_A: scipy.sparse.csr_array | None = _stack(ind_csrs)
        self.indicator_b: ndarray = _concat(ind_b)
        self.indicator_sense: ndarray = _concat(ind_sense, dtype=object)
        self.indicator_binvar: ndarray = _concat(ind_binvar, dtype=np.intp)
        self.indicator_binval: ndarray = _concat(ind_binval, dtype=np.intp)

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
