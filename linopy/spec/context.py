"""The data an evaluation reads: parameters resolved once, and the model, coordinates and lookups beside them."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field

import pandas as pd
import xarray as xr
from math_spec import program as ms

from linopy.model import Model
from linopy.spec import curves

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
            derivation = self._program.parameter(name).derivation
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

    ``solved`` is the fold's switch: a build leaves it false and a variable
    enters an expression as its linopy term; a fold sets it true and a
    variable enters as its solved values, so a named expression reads off the
    primal.
    """

    model: Model
    program: ms.Program
    coords: Mapping[str, pd.Index]
    lookups: Mapping[str, Mapping[str, xr.DataArray]]
    parameters: Parameters
    solved: bool = field(default=False)
