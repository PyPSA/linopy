"""
Every parameter of a program by name, resolved once.

A declared parameter is resolved from the caller's data and aligned by the
binder; one a ``piecewise:`` expansion emitted is derived from the block's own
breakpoints. Which of the two a name is, is the declaration's answer, and this
module is where it is asked.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

import xarray as xr
from math_spec import program as ms

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
