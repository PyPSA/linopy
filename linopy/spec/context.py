"""
The data an evaluation reads, and what a node evaluates to.

A :class:`Context` is everything evaluating a node needs beyond the node:
the model, the program, the coordinates, the relations and the parameters,
every parameter resolved once by :class:`Parameters`. What a node evaluates
to is a :data:`Value`: a number, an array or a linopy term.

A parameter arrives filled the way its ``missing:`` says, beside the mask of
the rows the data supplied: an ``absent`` row is NaN and travels as a mask on
any term it meets, a ``neutral`` one is 0, a literal one is the literal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace

import pandas as pd
import xarray as xr
from mathspec import program as ms

from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.model import Model
from linopy.variables import Variable

Term = Variable | LinearExpression | QuadraticExpression
Array = xr.DataArray | Term
Value = float | Array
Resolve = Callable[[str], tuple[xr.DataArray, xr.DataArray]]


class Parameters:
    """Every parameter of a program by name, each resolved from *resolve* on first read and then held, with the rows it has."""

    def __init__(self, program: ms.Program, resolve: Resolve) -> None:
        self._program = program
        self._resolve = resolve
        self._arrays: dict[str, tuple[xr.DataArray, xr.DataArray]] = {}

    def _read(self, name: str) -> tuple[xr.DataArray, xr.DataArray]:
        if name not in self._arrays:
            self._arrays[name] = self._resolve(name)
        return self._arrays[name]

    def __getitem__(self, name: str) -> xr.DataArray:
        """The parameter filled the way its ``missing:`` says."""
        return self._read(name)[0]

    def present(self, name: str) -> xr.DataArray:
        """Where the data has a row for the parameter."""
        return self._read(name)[1]

    def __contains__(self, name: str) -> bool:
        return name in self._program.parameters


@dataclass(frozen=True)
class Context:
    """
    Everything evaluating a node needs beyond the node.

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
    parameters: Parameters
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


def variable_term(variable: Variable, missing: str) -> Term:
    """The variable as it enters a built expression, carrying its declared ``missing:``."""
    return variable.fillna(0) if missing == "neutral" else variable


def solution(variable: Variable, missing: str) -> xr.DataArray:
    """The solved variable as it enters a fold, carrying its declared ``missing:``."""
    return variable.solution.fillna(0) if missing == "neutral" else variable.solution
