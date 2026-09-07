"""
What an expression node evaluates to, and how absence is spelled at each position.

Absence is positional: one missing parameter row is a zero in a coefficient,
a refusal in ``bounds:`` and false in a ``where`` operand, so there is no
single fill applied once and each position states its own answer. The
convention underneath is linopy v1's, which a spec-built model requires.
"""

from __future__ import annotations

import xarray as xr

from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.variables import Variable

Term = Variable | LinearExpression | QuadraticExpression
Array = xr.DataArray | Term
Value = float | Array


def present(variable: Variable) -> xr.DataArray:
    """The coordinates the variable occupies; ``-1`` is linopy's marker for an absent slot."""
    return variable.labels != -1


def variable_term(variable: Variable, absence: str) -> Term:
    """The variable as it enters a built expression, carrying its declared ``absence:``."""
    return variable.fillna(0) if absence == "zero" else variable


def solution(variable: Variable, absence: str) -> xr.DataArray:
    """The solved variable as it enters a fold, carrying its declared ``absence:``."""
    return variable.solution.fillna(0) if absence == "zero" else variable.solution


def coefficient(parameter: xr.DataArray) -> xr.DataArray:
    """A parameter in a coefficient position, its uncovered slots at zero."""
    return parameter.fillna(0.0)
