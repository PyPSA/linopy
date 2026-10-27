"""
What evaluating a node reads beyond the node, and what it evaluates to.

A :class:`Context` holds the model, the program, the coordinates, the
relations and ``parameter``, which gives a parameter by name as an
:class:`~linopy.spec.attach.Parameter`: its values filled the way its
``missing:`` says beside where the data has a row. What a node evaluates
to is a :data:`Value`: a number, an array or a linopy term.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import overload

import pandas as pd
import xarray as xr
from mathspec import program as ms

from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.model import Model
from linopy.spec.attach import Parameter
from linopy.variables import Variable

Term = Variable | LinearExpression | QuadraticExpression
Array = xr.DataArray | Term
Value = float | Array
Expression = xr.DataArray | LinearExpression | QuadraticExpression


@overload
def as_expression(value: Term) -> LinearExpression | QuadraticExpression: ...
@overload
def as_expression(value: Array) -> Expression: ...
def as_expression(value: Array) -> Expression:
    """*value* with a bare variable as its linear expression."""
    return value.to_linexpr() if isinstance(value, Variable) else value


@dataclass(frozen=True)
class Context:
    """
    Everything evaluating a node needs beyond the node.

    ``parameter`` reads a parameter by name, filled the way its
    ``missing:`` says, each resolved once and cached.
    ``solved`` is the fold's switch: a build leaves it false
    and a variable enters an expression as its linopy term; a fold sets it
    true and a variable enters as its solved values, so a named expression
    reads off the primal. ``filled`` is the comparison's switch: on, a
    parameter enters as its filled values; a condition turns it off so a
    missing row reaches the comparison as a hole, read as false there. ``rows``
    narrows the finiteness check to the rows a constraint builds, and
    ``subject`` names what is being built in its refusals.
    """

    model: Model
    program: ms.Program
    coords: Mapping[str, pd.Index]
    relations: Mapping[str, xr.DataArray]
    parameter: Callable[[str], Parameter]
    solved: bool = False
    filled: bool = True
    rows: xr.DataArray | None = None
    subject: str = "the spec"
