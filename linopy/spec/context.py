"""
What evaluating a node reads beyond the node, and what it evaluates to.

A :class:`Context` holds the model, the program, the coordinates, the
relations and ``parameter``, which gives a parameter by name as an
:class:`~linopy.spec.attach.Parameter`: its values filled the way its
``missing:`` says beside where the data has a row. What a node evaluates
to is a :data:`Value`: a number, an array or a linopy term.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast, overload

import pandas as pd
import xarray as xr
from mathspec import program as ms

from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.model import Model
from linopy.spec.attach import Parameter, Table
from linopy.variables import Variable

Term = Variable | LinearExpression | QuadraticExpression
Array = xr.DataArray | Term
Value = float | Array
Expression = xr.DataArray | LinearExpression | QuadraticExpression
_Kept = TypeVar("_Kept")


@overload
def as_expression(value: Term) -> LinearExpression | QuadraticExpression: ...
@overload
def as_expression(value: Array) -> Expression: ...
def as_expression(value: Array) -> Expression:
    """*value* with a bare variable as its linear expression."""
    return value.to_linexpr() if isinstance(value, Variable) else value


@dataclass
class Memo:
    """
    What a build or a fold has evaluated: a variable, a parameter or a named
    expression by its node and ``filled``, a condition by its node.

    ``masked`` counts the refusals a ``rows`` narrowing let pass: a value
    made under one is not kept, so a value read back is refused wherever it
    would be when evaluated afresh.
    """

    values: dict[Hashable, Any] = field(default_factory=dict)
    masked: int = 0

    def kept(self, key: Hashable, make: Callable[[], _Kept]) -> _Kept:
        """The value kept at *key*, else made by *make* and kept unless a refusal was masked making it."""
        if key in self.values:
            return cast(_Kept, self.values[key])
        masked = self.masked
        value = make()
        if self.masked == masked:
            self.values[key] = value
        return value


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
    ``subject`` names what is being built in its refusals. ``memo`` is
    shared by every context replaced from one.
    """

    model: Model
    program: ms.Program
    coords: Mapping[str, pd.Index]
    relations: Mapping[str, Table]
    parameter: Callable[[str], Parameter]
    solved: bool = False
    filled: bool = True
    rows: xr.DataArray | None = None
    subject: str = "the spec"
    memo: Memo = field(default_factory=Memo, compare=False, repr=False)
