#!/usr/bin/env python3
"""
Linopy objective module.

This module contains definition related to objective expressions.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
from numpy import float64
from pandas.core.frame import DataFrame
from scipy.sparse._csc import csc_matrix
from xarray.core.coordinates import DatasetCoordinates
from xarray.core.dataarray import DataArray
from xarray.core.dataset import Dataset
from xarray.core.indexes import Indexes
from xarray.core.utils import Frozen

from linopy import expressions
from linopy.types import ConstantLike

if TYPE_CHECKING:
    from linopy.expressions import LinearExpression, QuadraticExpression
    from linopy.model import Model


def objwrap(
    method: Callable, *default_args: Any, **new_default_kwargs: Any
) -> Callable:
    @functools.wraps(method)
    def _objwrap(obj: Objective, *args: Any, **kwargs: Any) -> Objective:
        for k, v in new_default_kwargs.items():
            kwargs.setdefault(k, v)
        return obj.__class__(
            method(obj.expression, *default_args, *args, **kwargs), obj.model, obj.sense
        )

    _objwrap.__doc__ = (
        f"Wrapper for the expression {method} function for linopy.Objective."
    )
    if new_default_kwargs:
        _objwrap.__doc__ += f" with default arguments: {new_default_kwargs}"

    return _objwrap


class Objective:
    """
    An objective expression containing all relevant information.
    """

    __slots__ = ("_expression", "_model", "_sense", "_value")
    __array_ufunc__ = None
    __array_priority__ = 10000
    __pandas_priority__ = 10000

    _fill_value: dict[str, float | int] = {"vars": -1, "coeffs": np.nan, "const": 0}

    def __init__(
        self,
        expression: expressions.LinearExpression | expressions.QuadraticExpression,
        model: Model,
        sense: str = "min",
    ) -> None:
        self._model: Model = model
        self._value: float | None = None

        self.sense: str = sense
        self.expression: (
            expressions.LinearExpression | expressions.QuadraticExpression
        ) = expression

    def __repr__(self) -> str:
        sense_string = f"Sense: {self.sense}"
        expr_string = self.expression.__repr__()
        expr_string = "\n".join(
            expr_string.split("\n")[2:]
        )  # drop first two lines of expression string
        expr_string = self.expression.type + ": " + expr_string
        value_string = f"Value: {self.value}"

        return f"Objective:\n----------\n{expr_string}\n{sense_string}\n{value_string}"

    @property
    def attrs(self) -> dict[str, Any]:
        """
        Returns the attributes of the objective.
        """
        return self.expression.attrs

    @property
    def coords(self) -> DatasetCoordinates:
        """
        Returns the coordinates of the objective.
        """
        return self.expression.coords

    @property
    def indexes(self) -> Indexes:
        """
        Returns the indexes of the objective.
        """
        return self.expression.indexes

    @property
    def sizes(self) -> Frozen:
        """
        Returns the sizes of the objective.
        """
        return self.expression.sizes

    @property
    def flat(self) -> DataFrame:
        """
        Returns the flattened objective.
        """
        return self.expression.flat

    def to_polars(self, **kwargs: Any) -> pl.DataFrame:
        """
        Returns the objective as a polars DataFrame.
        """
        return self.expression.to_polars(**kwargs)

    @property
    def coeffs(self) -> DataArray:
        """
        Returns the coefficients of the objective.
        """
        return self.expression.coeffs

    @property
    def vars(self) -> DataArray:
        """
        Returns the variables of the objective.
        """
        return self.expression.vars

    @property
    def data(self) -> Dataset:
        """
        Returns the data of the objective.
        """
        return self.expression.data

    @property
    def nterm(self) -> int:
        """
        Returns the number of terms in the objective.
        """
        return self.expression.nterm

    @property
    def expression(
        self,
    ) -> expressions.LinearExpression | expressions.QuadraticExpression:
        """
        Returns the expression of the objective.
        """
        return self._expression

    @expression.setter
    def expression(
        self,
        expr: expressions.LinearExpression
        | expressions.QuadraticExpression
        | Sequence[tuple],
    ) -> None:
        """
        Sets the expression of the objective.
        """
        if isinstance(expr, list | tuple):
            expr = self.model.linexpr(*expr)

        if not isinstance(
            expr, expressions.LinearExpression | expressions.QuadraticExpression
        ):
            raise ValueError(
                f"Invalid type of `expr` ({type(expr)})."
                " Must be a LinearExpression or QuadraticExpression."
            )

        if len(expr.coord_dims):
            expr = expr.sum()

        if (expr.const != 0.0) and not np.isnan(expr.const):
            raise ValueError("Constant values in objective function not supported.")

        self._expression = expr

    @property
    def model(self) -> Model:
        """
        Returns the model of the objective.
        """
        return self._model

    @property
    def sense(self) -> str:
        """
        Returns the sense of the objective.
        """
        return self._sense

    @sense.setter
    def sense(self, sense: str) -> None:
        """
        Sets the sense of the objective.
        """
        if sense not in ("min", "max"):
            raise ValueError("Invalid sense. Must be 'min' or 'max'.")
        self._sense = sense

    @property
    def value(self) -> float64 | float | None:
        """
        Returns the value of the objective.
        """
        return self._value

    def set_value(self, value: float) -> None:
        """
        Sets the value of the objective.
        """
        self._value = float(value)

    @property
    def is_linear(self) -> bool:
        return type(self.expression) is expressions.LinearExpression

    @property
    def is_quadratic(self) -> bool:
        return type(self.expression) is expressions.QuadraticExpression

    def to_matrix(self, *args: Any, **kwargs: Any) -> csc_matrix:
        """Wrapper for expression.to_matrix"""
        if not isinstance(self.expression, expressions.QuadraticExpression):
            raise ValueError("Cannot convert linear objective to matrix.")
        return self.expression.to_matrix(*args, **kwargs)

    assign = objwrap(expressions.LinearExpression.assign)

    sel = objwrap(expressions.LinearExpression.sel)

    def __add__(
        self, expr: int | QuadraticExpression | LinearExpression | Objective
    ) -> Objective:
        if isinstance(expr, Objective):
            expr = expr.expression

        return Objective(self.expression + expr, self.model, self.sense)

    def __sub__(self, expr: LinearExpression | Objective) -> Objective:
        if not isinstance(expr, Objective):
            expr = Objective(expr, self.model)
        return Objective(self.expression - expr.expression, self.model, self.sense)

    def __mul__(self, expr: ConstantLike) -> Objective:
        # only allow scalar multiplication
        if not isinstance(expr, int | float | np.floating | np.integer):
            raise ValueError("Invalid type for multiplication.")
        return Objective(self.expression * expr, self.model, self.sense)

    def __neg__(self) -> Objective:
        return Objective(-self.expression, self.model, self.sense)

    def __truediv__(self, expr: ConstantLike) -> Objective:
        # only allow scalar division
        if not isinstance(expr, int | float | np.floating | np.integer):
            raise ValueError("Invalid type for division.")
        return Objective(self.expression / expr, self.model, self.sense)
