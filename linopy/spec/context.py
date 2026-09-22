"""
The data an evaluation reads, and what a node evaluates to.

A :class:`Context` is everything evaluating a node needs beyond the node:
the model, the program, the coordinates, the relations and the parameters,
every parameter resolved once by :class:`Parameters`. What a node evaluates
to is a :data:`Value`: a number, an array or a linopy term.

Absence is positional: one missing parameter row is a zero in a coefficient,
a refusal in ``bounds:`` and false in a ``where`` operand, so there is no
single fill applied once and each position states its own answer. The
convention underneath is linopy v1's, which a spec-built model requires.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace

import pandas as pd
import xarray as xr
from math_spec import program as ms

from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.model import Model
from linopy.spec import curves
from linopy.variables import Variable

Term = Variable | LinearExpression | QuadraticExpression
Array = xr.DataArray | Term
Value = float | Array
Resolve = Callable[[str], xr.DataArray]


class Parameters(Mapping[str, xr.DataArray]):
    """
    Every parameter of a program by name, each resolved on first read and then held.

    A declared parameter comes from *resolve*; one a ``piecewise:`` expansion
    emitted is derived from the block's own breakpoints the way its
    derivation says, so a caller never supplies it.
    """

    def __init__(self, program: ms.Program, resolve: Resolve) -> None:
        self._program = program
        self._resolve = resolve
        self._arrays: dict[str, xr.DataArray] = {}

    def __getitem__(self, name: str) -> xr.DataArray:
        if name not in self._arrays:
            derivation = self._program.parameters[name].derivation
            self._arrays[name] = (
                self._resolve(name)
                if derivation is None
                else curves.derive(derivation, self, self._program)
            )
        return self._arrays[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._program.parameters)

    def __len__(self) -> int:
        return len(self._program.parameters)


@dataclass(frozen=True)
class Context:
    """
    Everything evaluating a node needs beyond the node.

    ``name`` is the spec's name, the stamp everything a build adds to the
    model carries. ``solved`` is the fold's switch: a build leaves it false
    and a variable enters an expression as its linopy term; a fold sets it
    true and a variable enters as its solved values, so a named expression
    reads off the primal.
    """

    model: Model
    program: ms.Program
    coords: Mapping[str, pd.Index]
    relations: Mapping[str, xr.DataArray]
    parameters: Mapping[str, xr.DataArray]
    name: str
    solved: bool = field(default=False)

    @property
    def unsolved(self) -> Context:
        """The same context with the fold's switch off, so a variable enters as its linopy term."""
        return replace(self, solved=False)


def variable_term(variable: Variable, absence: str) -> Term:
    """The variable as it enters a built expression, carrying its declared ``absence:``."""
    return variable.fillna(0) if absence == "zero" else variable


def solution(variable: Variable, absence: str) -> xr.DataArray:
    """The solved variable as it enters a fold, carrying its declared ``absence:``."""
    return variable.solution.fillna(0) if absence == "zero" else variable.solution


def coefficient(parameter: xr.DataArray) -> xr.DataArray:
    """A parameter in a coefficient position, its uncovered slots at zero."""
    return parameter.fillna(0.0)
