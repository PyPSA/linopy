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


def unmapped(key: object) -> bool:
    """Whether a lookup left this member in no group: ``None``, or the NaN that never equals itself."""
    return key is None or key != key


def variable_term(variable: Variable, absence: str) -> Term:
    """The variable as it enters a built expression, carrying its declared ``absence:``."""
    return variable.fillna(0) if absence == "zero" else variable


def solution(variable: Variable, absence: str) -> xr.DataArray:
    """The solved variable as it enters a fold, carrying its declared ``absence:``."""
    return variable.solution.fillna(0) if absence == "zero" else variable.solution


def coefficient(parameter: xr.DataArray) -> xr.DataArray:
    """A parameter in a coefficient position, its uncovered slots at zero."""
    return parameter.fillna(0.0)


def filled(expression: Array, fill: float) -> Array:
    """*expression* with every absence in it standing as *fill*."""
    if isinstance(expression, Variable):
        expression = expression.to_linexpr()
    return expression.fillna(fill)


def vacated(
    shifted: Array, operand: Array, over: str, vacated: xr.DataArray, fill: float
) -> Array:
    """
    *shifted*, with the positions the shift vacated filled, and only those.

    The fill lands where the shift vacated and the operand carries the
    coordinate; every other slot keeps the absence it arrived with, so no row
    is invented at a coordinate the operand never had.
    """
    carried = (~operand.isnull()).any(over)
    keep = carried & (~shifted.isnull() | vacated)
    return filled(shifted, fill).where(keep)
