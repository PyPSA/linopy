#!/usr/bin/env python3
"""
Created on Mon Oct 10 13:33:55 2022.

@author: fabian
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import scipy.sparse
from numpy import ndarray

from linopy import expressions
from linopy.constraints import CSRConstraint

if TYPE_CHECKING:
    from linopy.model import Model


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

        if not len(m.constraints):
            self.clabels: ndarray = np.array([], dtype=np.intp)
            self.b: ndarray = np.array([])
            self.sense: ndarray = np.array([], dtype=object)
            self.A: scipy.sparse.csr_array | None = None
            return

        label_index = m.variables.label_index
        csrs = []
        clabels_list = []
        b_list = []
        sense_list = []
        for c in m.constraints.data.values():
            csr, con_labels, b, sense = c.to_matrix_with_rhs(label_index)
            csrs.append(csr)
            clabels_list.append(con_labels)
            b_list.append(b)
            sense_list.append(sense)

        self.A = cast(scipy.sparse.csr_array, scipy.sparse.vstack(csrs, format="csr"))
        self.clabels = np.concatenate(clabels_list)
        self.b = np.concatenate(b_list) if b_list else np.array([])
        self.sense = (
            np.concatenate(sense_list) if sense_list else np.array([], dtype=object)
        )

    @property
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

    @property
    def Q(self) -> scipy.sparse.csc_matrix | None:
        """Quadratic objective matrix, shape (n_active_vars, n_active_vars)."""
        m = self._parent
        expr = m.objective.expression
        if not isinstance(expr, expressions.QuadraticExpression):
            return None
        return expr.to_matrix()[self.vlabels][:, self.vlabels]

    @property
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

    @property
    def dual(self) -> ndarray:
        """Dual values aligned with clabels."""
        if not self._parent.status == "ok":
            raise ValueError("Model is not optimized.")
        m = self._parent
        label_index = m.variables.label_index
        dual_list = []
        has_dual = False
        for c in m.constraints.data.values():
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
