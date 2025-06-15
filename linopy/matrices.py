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

        for cached_prop in ["flat_vars", "flat_cons", "sol", "dual"]:
            # check existence of cached_prop without creating it
            if cached_prop in self.__dict__:
                delattr(self, cached_prop)

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
        df: pd.DataFrame = self.flat_vars
        return create_vector(df.key, df.labels, -1)

    @property
    def vtypes(self) -> ndarray:
        """Vector of types of all non-missing variables."""
        m = self._parent
        df: pd.DataFrame = self.flat_vars
        specs = []
        for name in m.variables:
            if name in m.binaries:
                val = "B"
            elif name in m.integers:
                val = "I"
            else:
                val = "C"
            specs.append(pd.Series(val, index=m.variables[name].flat.labels))

        ds = pd.concat(specs)
        ds = df.set_index("key").labels.map(ds)
        return create_vector(ds.index, ds.to_numpy(), fill_value="")

    @property
    def lb(self) -> ndarray:
        """Vector of lower bounds of all non-missing variables."""
        df: pd.DataFrame = self.flat_vars
        return create_vector(df.key, df.lower)

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
        df: pd.DataFrame = self.flat_vars
        return create_vector(df.key, df.upper)

    @property
    def clabels(self) -> ndarray:
        """Vector of labels of all non-missing constraints."""
        df: pd.DataFrame = self.flat_cons
        if df.empty:
            return np.array([], dtype=int)
        return create_vector(df.key, df.labels, fill_value=-1)

    @property
    def A(self) -> csc_matrix | None:
        """Constraint matrix of all non-missing constraints and variables."""
        m = self._parent
        if not len(m.constraints):
            return None
        A: csc_matrix = m.constraints.to_matrix(filter_missings=False)
        return A[self.clabels][:, self.vlabels]

    @property
    def sense(self) -> ndarray:
        """Vector of senses of all non-missing constraints."""
        df: pd.DataFrame = self.flat_cons
        return create_vector(df.key, df.sign.astype(np.dtype("<U1")), fill_value="")

    @property
    def b(self) -> ndarray:
        """Vector of right-hand-sides of all non-missing constraints."""
        df: pd.DataFrame = self.flat_cons
        return create_vector(df.key, df.rhs)

    @property
    def c(self) -> ndarray:
        """Vector of objective coefficients of all non-missing variables."""
        m = self._parent
        ds = m.objective.flat
        if isinstance(m.objective.expression, expressions.QuadraticExpression):
            ds = ds[(ds.vars1 == -1) | (ds.vars2 == -1)]
            ds["vars"] = ds.vars1.where(ds.vars1 != -1, ds.vars2)

        vars: pd.Series = ds.vars.map(self.flat_vars.set_index("labels").key)
        shape: int = self.flat_vars.key.max() + 1
        return create_vector(vars, ds.coeffs, fill_value=0.0, shape=shape)

    @property
    def Q(self) -> csc_matrix | None:
        """Matrix objective coefficients of quadratic terms of all non-missing variables."""
        m = self._parent
        expr = m.objective.expression
        if not isinstance(expr, expressions.QuadraticExpression):
            return None
        return expr.to_matrix()[self.vlabels][:, self.vlabels]
