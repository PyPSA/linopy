#!/usr/bin/env python3
"""
Created on Mon Oct 10 13:33:55 2022.

@author: fabian
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas.core.indexes.base import Index
from pandas.core.series import Series
from scipy.sparse._csc import csc_matrix

from linopy import expressions
from linopy.constants import FACTOR_DIM

if TYPE_CHECKING:
    from linopy.model import Model


def create_vector(
    indices: Series | Index,
    values: Series | ndarray,
    fill_value: str | float | int = np.nan,
    shape: int | None = None,
) -> ndarray:
    """Create a vector of a size equal to the maximum index plus one."""
    if shape is None:
        max_value = indices.max()
        if not isinstance(max_value, np.integer | int):
            raise ValueError("Indices must be integers.")
        shape = max_value + 1
    vector = np.full(shape, fill_value)
    vector[indices] = values
    return vector


class MatrixAccessor:
    """
    Helper class to quickly access model related vectors and matrices.
    """

    def __init__(self, model: Model) -> None:
        self._parent = model

    def clean_cached_properties(self) -> None:
        """Clear the cache for all cached properties of an object"""

        for cached_prop in [
            "flat_vars",
            "flat_cons",
            "sol",
            "dual",
            "_variable_data",
            "_constraint_data",
        ]:
            # check existence of cached_prop without creating it
            if cached_prop in self.__dict__:
                delattr(self, cached_prop)

    @cached_property
    def _variable_data(self) -> tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
        """Dense-by-key variable vectors and label->key map."""
        m = self._parent

        label_to_key = np.full(m._xCounter, -1, dtype=np.int64)
        labels_parts: list[np.ndarray] = []
        lb_parts: list[np.ndarray] = []
        ub_parts: list[np.ndarray] = []
        vtype_parts: list[np.ndarray] = []
        next_key = 0

        for _, variable in m.variables.items():
            labels = variable.labels.values.reshape(-1)
            mask = labels != -1
            labels = labels[mask]
            n = labels.size
            if not n:
                continue

            label_to_key[labels] = np.arange(next_key, next_key + n)
            next_key += n

            lb = np.broadcast_to(variable.lower.values, variable.labels.shape).reshape(
                -1
            )
            ub = np.broadcast_to(variable.upper.values, variable.labels.shape).reshape(
                -1
            )

            labels_parts.append(labels)
            lb_parts.append(lb[mask])
            ub_parts.append(ub[mask])

            if variable.attrs["binary"]:
                vtype = "B"
            elif variable.attrs["integer"]:
                vtype = "I"
            else:
                vtype = "C"
            vtype_parts.append(np.full(n, vtype, dtype="<U1"))

        if labels_parts:
            return (
                np.concatenate(labels_parts),
                np.concatenate(lb_parts),
                np.concatenate(ub_parts),
                np.concatenate(vtype_parts),
                label_to_key,
            )
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype="<U1"),
            label_to_key,
        )

    @cached_property
    def _constraint_data(self) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        """Dense-by-key constraint vectors and label->key map."""
        m = self._parent

        label_to_key = np.full(m._cCounter, -1, dtype=np.int64)
        labels_parts: list[np.ndarray] = []
        rhs_parts: list[np.ndarray] = []
        sense_parts: list[np.ndarray] = []
        next_key = 0

        for _, constraint in m.constraints.items():
            labels = constraint.labels.values.reshape(-1)
            vars_arr = constraint.vars.values
            coeffs_arr = constraint.coeffs.values

            term_axis = constraint.vars.get_axis_num(constraint.term_dim)
            if term_axis != vars_arr.ndim - 1:
                vars_arr = np.moveaxis(vars_arr, term_axis, -1)
                coeffs_arr = np.moveaxis(coeffs_arr, term_axis, -1)

            active = ((vars_arr != -1) & (coeffs_arr != 0)).any(axis=-1).reshape(-1)
            mask = (labels != -1) & active
            labels = labels[mask]
            n = labels.size
            if not n:
                continue

            label_to_key[labels] = np.arange(next_key, next_key + n)
            next_key += n

            rhs = np.broadcast_to(
                constraint.rhs.values, constraint.labels.shape
            ).reshape(-1)
            sign = np.broadcast_to(
                constraint.sign.values, constraint.labels.shape
            ).reshape(-1)
            sign = sign[mask]
            sense = np.full(sign.shape, "=", dtype="<U1")
            sense[sign == "<="] = "<"
            sense[sign == ">="] = ">"

            labels_parts.append(labels)
            rhs_parts.append(rhs[mask])
            sense_parts.append(sense)

        if labels_parts:
            return (
                np.concatenate(labels_parts),
                np.concatenate(sense_parts),
                np.concatenate(rhs_parts),
                label_to_key,
            )
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype="<U1"),
            np.array([], dtype=float),
            label_to_key,
        )

    @cached_property
    def flat_vars(self) -> pd.DataFrame:
        m = self._parent
        return m.variables.flat

    @cached_property
    def flat_cons(self) -> pd.DataFrame:
        m = self._parent
        return m.constraints.flat

    @property
    def vlabels(self) -> ndarray:
        """Vector of labels of all non-missing variables."""
        labels, *_ = self._variable_data
        return labels

    @property
    def vtypes(self) -> ndarray:
        """Vector of types of all non-missing variables."""
        _, _, _, vtypes, _ = self._variable_data
        return vtypes

    @property
    def lb(self) -> ndarray:
        """Vector of lower bounds of all non-missing variables."""
        _, lb, _, _, _ = self._variable_data
        return lb

    @cached_property
    def sol(self) -> ndarray:
        """Vector of solution values of all non-missing variables."""
        if not self._parent.status == "ok":
            raise ValueError("Model is not optimized.")
        if "solution" not in self.flat_vars:
            del self.flat_vars  # clear cache
        df: pd.DataFrame = self.flat_vars
        return create_vector(df.key, df.solution, fill_value=np.nan)

    @cached_property
    def dual(self) -> ndarray:
        """Vector of dual values of all non-missing constraints."""
        if not self._parent.status == "ok":
            raise ValueError("Model is not optimized.")
        if "dual" not in self.flat_cons:
            del self.flat_cons  # clear cache
        df: pd.DataFrame = self.flat_cons
        if "dual" not in df:
            raise AttributeError(
                "Underlying is optimized but does not have dual values stored."
            )
        return create_vector(df.key, df.dual, fill_value=np.nan)

    @property
    def ub(self) -> ndarray:
        """Vector of upper bounds of all non-missing variables."""
        _, _, ub, _, _ = self._variable_data
        return ub

    @property
    def clabels(self) -> ndarray:
        """Vector of labels of all non-missing constraints."""
        labels, _, _, _ = self._constraint_data
        return labels

    @property
    def A(self) -> csc_matrix | None:
        """Constraint matrix of all non-missing constraints and variables."""
        m = self._parent
        if not len(m.constraints):
            return None
        return m.constraints.to_matrix(filter_missings=True)

    @property
    def sense(self) -> ndarray:
        """Vector of senses of all non-missing constraints."""
        _, sense, _, _ = self._constraint_data
        return sense

    @property
    def b(self) -> ndarray:
        """Vector of right-hand-sides of all non-missing constraints."""
        _, _, rhs, _ = self._constraint_data
        return rhs

    @property
    def c(self) -> ndarray:
        """Vector of objective coefficients of all non-missing variables."""
        m = self._parent
        _, _, _, _, label_to_key = self._variable_data
        nvars = len(self.vlabels)
        if nvars == 0:
            return np.array([], dtype=float)

        expr = m.objective.expression

        if isinstance(expr, expressions.QuadraticExpression):
            vars_arr = expr.data.vars.values
            coeffs = expr.data.coeffs.values.reshape(-1)
            factor_axis = expr.data.vars.get_axis_num(FACTOR_DIM)

            vars1 = np.take(vars_arr, 0, axis=factor_axis).reshape(-1)
            vars2 = np.take(vars_arr, 1, axis=factor_axis).reshape(-1)
            mask = ((vars1 == -1) ^ (vars2 == -1)) & (coeffs != 0)
            lin_vars = np.where(vars1 == -1, vars2, vars1)
            labels = lin_vars[mask]
            coeffs = coeffs[mask]
        else:
            vars_arr = expr.vars.values.reshape(-1)
            coeffs = expr.coeffs.values.reshape(-1)
            mask = (vars_arr != -1) & (coeffs != 0)
            labels = vars_arr[mask]
            coeffs = coeffs[mask]

        keys = label_to_key[labels]
        valid = keys != -1
        if not np.any(valid):
            return np.zeros(nvars, dtype=float)
        return np.bincount(keys[valid], weights=coeffs[valid], minlength=nvars).astype(
            float,
            copy=False,
        )

    @property
    def Q(self) -> csc_matrix | None:
        """Matrix objective coefficients of quadratic terms of all non-missing variables."""
        m = self._parent
        expr = m.objective.expression
        if not isinstance(expr, expressions.QuadraticExpression):
            return None
        return expr.to_matrix()[self.vlabels][:, self.vlabels]
