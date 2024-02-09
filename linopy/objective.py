#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linopy objective module.

This module contains definition related to objective expressions.
"""

import functools
from typing import Union

import numpy as np

from linopy import expressions
from linopy.common import forward_as_properties


def objwrap(method, *default_args, **new_default_kwargs):
    @functools.wraps(method)
    def _objwrap(obj, *args, **kwargs):
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


@forward_as_properties(
    expression=[
        "attrs",
        "coords",
        "indexes",
        "sizes",
        "flat",
        "coeffs",
        "vars",
        "data",
        "nterm",
    ]
)
class Objective:
    """
    An objective expression containing all relevant information.
    """

    __slots__ = ("_expression", "_model", "_sense", "_value")
    __array_ufunc__ = None
    __array_priority__ = 10000

    _fill_value = {"vars": -1, "coeffs": np.nan, "const": 0}

    def __init__(
        self,
        expression: Union[
            expressions.LinearExpression, expressions.QuadraticExpression
        ],
        model,
        sense="min",
    ):
        from linopy.model import Model

        self._model = model
        self._value = None

        self.sense = sense
        self.expression = expression

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
    def expression(self):
        """
        Returns the expression of the objective.
        """
        return self._expression

    @expression.setter
    def expression(self, expr):
        """
        Sets the expression of the objective.
        """
        if isinstance(expr, (list, tuple)):
            expr = self.model.linexpr(*expr)

        if not isinstance(
            expr, (expressions.LinearExpression, expressions.QuadraticExpression)
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
    def model(self):
        """
        Returns the model of the objective.
        """
        return self._model

    @property
    def sense(self):
        """
        Returns the sense of the objective.
        """
        return self._sense

    @sense.setter
    def sense(self, sense):
        """
        Sets the sense of the objective.
        """
        if sense not in ("min", "max"):
            raise ValueError("Invalid sense. Must be 'min' or 'max'.")
        self._sense = sense

    @property
    def value(self):
        """
        Returns the value of the objective.
        """
        return self._value

    def set_value(self, value: float):
        """
        Sets the value of the objective.
        """
        self._value = float(value)

    @property
    def is_linear(self):
        return type(self.expression) is expressions.LinearExpression

    @property
    def is_quadratic(self):
        return type(self.expression) is expressions.QuadraticExpression

    def to_matrix(self, *args, **kwargs):
        "Wrapper for expression.to_matrix"
        if self.is_linear:
            raise ValueError("Cannot convert linear objective to matrix.")
        return self.expression.to_matrix(*args, **kwargs)

    assign = objwrap(expressions.LinearExpression.assign)

    sel = objwrap(expressions.LinearExpression.sel)

    def __add__(self, expr):
        if not isinstance(expr, Objective):
            expr = Objective(expr, self.model, self.sense)
        return Objective(self.expression + expr.expression, self.model, self.sense)

    def __sub__(self, expr):
        if not isinstance(expr, Objective):
            expr = Objective(expr, self.model)
        return Objective(self.expression - expr.expression, self.model, self.sense)

    def __mul__(self, expr):
        # only allow scalar multiplication
        if not isinstance(expr, (int, float, np.floating, np.integer)):
            raise ValueError("Invalid type for multiplication.")
        return Objective(self.expression * expr, self.model, self.sense)

    def __neg__(self):
        return Objective(-self.expression, self.model, self.sense)

    def __truediv__(self, expr):
        # only allow scalar division
        if not isinstance(expr, (int, float, np.floating, np.integer)):
            raise ValueError("Invalid type for division.")
        return Objective(self.expression / expr, self.model, self.sense)
