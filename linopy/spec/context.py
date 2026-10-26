"""
What evaluating a node reads beyond the node, and what it evaluates to.

A :class:`Context` holds the model, the program, the coordinates, the
relations and ``parameter``, which gives a parameter by name as an
:class:`~linopy.spec.attach.Parameter`: its filled values beside where the
data has a row. What a node evaluates to is a :data:`Value`: a number, an
array or a linopy term.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

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


@dataclass(frozen=True)
class Context:
    """
    Everything evaluating a node needs beyond the node.

    ``parameter`` reads a parameter by name; a build passes a cached
    closure over :meth:`Attached.read`, so each is resolved once.
    ``name`` is the spec's name, the stamp everything a build adds to the
    model carries. ``solved`` is the fold's switch: a build leaves it false
    and a variable enters an expression as its linopy term; a fold sets it
    true and a variable enters as its solved values, so a named expression
    reads off the primal. ``filled`` is the comparison's switch: on, a
    parameter enters as its filled values; a condition turns it off so a
    missing row reaches the comparison and reads as false there. ``rows``
    narrows the finiteness check to the rows a constraint builds, and
    ``subject`` names what is being built in its refusals.
    """

    model: Model
    program: ms.Program
    coords: Mapping[str, pd.Index]
    relations: Mapping[str, xr.DataArray]
    parameter: Callable[[str], Parameter]
    name: str
    solved: bool = field(default=False)
    filled: bool = field(default=True)
    rows: xr.DataArray | None = field(default=None)
    subject: str = field(default="the spec")

    @property
    def unsolved(self) -> Context:
        """The same context with the fold's switch off, so a variable enters as its linopy term."""
        return replace(self, solved=False)

    @property
    def unfilled(self) -> Context:
        """The same context with the comparison's switch off, so a missing row reaches the comparison."""
        return replace(self, filled=False)

    def within(self, rows: xr.DataArray, subject: str) -> Context:
        """The same context narrowed to the rows a constraint builds, naming it as *subject*."""
        return replace(self, rows=rows, subject=subject)
