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

if TYPE_CHECKING:
    from linopy.model import Model


def _stack(csrs: list) -> scipy.sparse.csr_array | None:
    """Vertically stack CSR blocks, or None when there are none."""
    if not csrs:
        return None
    return cast(scipy.sparse.csr_array, scipy.sparse.vstack(csrs, format="csr"))


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
    """

    def __init__(self, model: Model) -> None:
        self._parent = model
        self._build_vars()
        self._build_cons()

    def _build_vars(self) -> None:
        m = self._parent
        label_index = m.variables.label_index
        self.vlabels: ndarray = label_index.vlabels

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
            self.lb: ndarray = np.concatenate(lb_list)
            self.ub: ndarray = np.concatenate(ub_list)
            self.vtypes: ndarray = np.concatenate(vtypes_list)
        else:
            self.lb = np.array([])
            self.ub = np.array([])
            self.vtypes = np.array([], dtype=object)

    def _build_cons(self) -> None:
        m = self._parent
        label_index = m.variables.label_index
        label_to_pos = label_index.label_to_pos

        reg_csrs, reg_b, reg_sense = [], [], []
        ind_csrs, ind_b, ind_sense, ind_binvar, ind_binval = [], [], [], [], []
        for c in m.constraints.data.values():
            if c.is_indicator:
                cc = c if isinstance(c, CSRConstraint) else c.freeze()
                csr, _, b, sense = cc.to_matrix_with_rhs(label_index)
                ind_csrs.append(csr)
                ind_b.append(b)
                ind_sense.append(sense)
                ind_binvar.append(label_to_pos[cc._binvar_labels])
                binval = cast("int | np.ndarray", cc._binval)
                ind_binval.append(_binval_per_row(binval, len(b)))
            else:
                csr, _, b, sense = c.to_matrix_with_rhs(label_index)
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
        expr = m.objective.expression
        if isinstance(expr, expressions.QuadraticExpression):
            # vars has shape (_factor=2, _term); linear terms have one factor == -1
            vars_2d = expr.data.vars.values  # shape (2, n_term)
            coeffs_all = expr.data.coeffs.values.ravel()
            vars1, vars2 = vars_2d[0], vars_2d[1]
            linear = (vars1 == -1) | (vars2 == -1)
            var_labels = np.where(vars1[linear] != -1, vars1[linear], vars2[linear])
            coeffs = coeffs_all[linear]
        else:
            var_labels = expr.data.vars.values.ravel()
            coeffs = expr.data.coeffs.values.ravel()

        mask = var_labels != -1
        np.add.at(result, label_to_pos[var_labels[mask]], coeffs[mask])
        return result

    @cached_property
    def Q(self) -> scipy.sparse.csc_matrix | None:
        """Quadratic objective matrix, shape (n_active_vars, n_active_vars)."""
        m = self._parent
        expr = m.objective.expression
        if not isinstance(expr, expressions.QuadraticExpression):
            return None
        return expr.to_matrix()[self.vlabels][:, self.vlabels]

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
        label_index = m.variables.label_index
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
                    dual_list.append(np.full(len(c._con_labels), np.nan))
            else:
                csr, _ = c.to_matrix(label_index)
                nonempty = np.diff(csr.indptr).astype(bool)
                active_rows = np.flatnonzero(nonempty)
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
