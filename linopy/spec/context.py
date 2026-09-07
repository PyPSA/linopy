"""The data an evaluation reads: the parameters, and the model, coordinates and lookups beside them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import pandas as pd
import xarray as xr
from math_spec import program as ms

from linopy.model import Model


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
    parameters: Mapping[str, xr.DataArray]
    solved: bool = field(default=False)

    @property
    def unsolved(self) -> Context:
        """The same context with the fold's switch off, so a variable enters as its linopy term."""
        return replace(self, solved=False)

    def lookup(self, name: str, over: str) -> xr.DataArray:
        """The lookup *name* as an array over *over*, NaN where a label is unmapped."""
        return self.lookups[over][name]
