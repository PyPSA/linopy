#!/usr/bin/env python3
"""
Created on Mon Oct 10 13:33:55 2022.

@author: fabian
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy import ndarray
from scipy.sparse._csc import csc_matrix

from linopy import expressions
from linopy.constraints import Constraint

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

        labels_list = []
        lb_list = []
        ub_list = []
        vtypes_list = []

        for name, var in m.variables.items():
            labels = var.labels.values.ravel()
            mask = labels != -1
            active_labels = labels[mask]

            if name in m.binaries:
                vtype = "B"
            elif name in m.integers:
                vtype = "I"
            elif name in m.semi_continuous:
                vtype = "S"
            else:
                vtype = "C"

            labels_list.append(active_labels)
            lb_list.append(var.lower.values.ravel()[mask])
            ub_list.append(var.upper.values.ravel()[mask])
            vtypes_list.append(np.full(mask.sum(), vtype))

        if labels_list:
            vlabels = np.concatenate(labels_list)
            order = np.argsort(vlabels)
            self.vlabels: ndarray = vlabels[order]
            self.lb: ndarray = np.concatenate(lb_list)[order]
            self.ub: ndarray = np.concatenate(ub_list)[order]
            self.vtypes: ndarray = np.concatenate(vtypes_list)[order]
        else:
            self.vlabels = np.array([], dtype=np.intp)
            self.lb = np.array([])
            self.ub = np.array([])
            self.vtypes = np.array([], dtype=object)

    def _build_cons(self) -> None:
        m = self._parent

        if not len(m.constraints):
            self.clabels: ndarray = np.array([], dtype=np.intp)
            self.b: ndarray = np.array([])
            self.sense: ndarray = np.array([], dtype=object)
            self.A: csc_matrix | None = None
            return

        A_full, clabels, _ = m.constraints.to_matrix(filter_missings=False)
        self.A = A_full[:, self.vlabels]
        self.clabels = clabels

        b_list = []
        sense_list = []
        for c in m.constraints.data.values():
            csr = c.to_matrix()
            nonempty = np.diff(csr.indptr).astype(bool)
            active_rows = np.flatnonzero(nonempty)

            if isinstance(c, Constraint):
                b_list.append(c._rhs[active_rows])
                sense_list.append(np.full(len(active_rows), c._sign[0]))
            else:
                b_list.append(c.rhs.values.ravel()[active_rows])
                sign_flat = c.sign.values.ravel()[active_rows]
                sense_list.append(np.array([s[0] for s in sign_flat]))

        self.b = np.concatenate(b_list) if b_list else np.array([])
        self.sense = (
            np.concatenate(sense_list) if sense_list else np.array([], dtype=object)
        )

    @property
    def c(self) -> ndarray:
        """Objective coefficients aligned with vlabels."""
        m = self._parent
        result = np.zeros(len(self.vlabels))

        ds = m.objective.flat
        if isinstance(m.objective.expression, expressions.QuadraticExpression):
            ds = ds[(ds.vars1 == -1) | (ds.vars2 == -1)].copy()
            ds["vars"] = ds.vars1.where(ds.vars1 != -1, ds.vars2)

        var_labels = ds.vars.values
        coeffs = ds.coeffs.values
        mask = var_labels != -1
        active_labels = var_labels[mask]
        positions = np.searchsorted(self.vlabels, active_labels)
        valid = (positions < len(self.vlabels)) & (
            self.vlabels[positions] == active_labels
        )
        np.add.at(result, positions[valid], coeffs[mask][valid])
        return result

    @property
    def Q(self) -> csc_matrix | None:
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
        for _, var in m.variables.items():
            labels = var.labels.values.ravel()
            mask = labels != -1
            active_labels = labels[mask]
            positions = np.searchsorted(self.vlabels, active_labels)
            result[positions] = var.solution.values.ravel()[mask]
        return result

    @property
    def dual(self) -> ndarray:
        """Dual values aligned with clabels."""
        if not self._parent.status == "ok":
            raise ValueError("Model is not optimized.")
        m = self._parent
        dual_list = []
        has_dual = False
        for c in m.constraints.data.values():
            csr = c.to_matrix()
            nonempty = np.diff(csr.indptr).astype(bool)
            active_rows = np.flatnonzero(nonempty)
            if isinstance(c, Constraint):
                if c._dual is not None:
                    dual_list.append(c._dual[active_rows])
                    has_dual = True
                else:
                    dual_list.append(np.full(len(active_rows), np.nan))
            else:
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
