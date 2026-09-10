"""The data an evaluation reads: the parameters, and the model, coordinates and lookups beside them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import pandas as pd
import xarray as xr
from math_spec import program as ms

from linopy.model import Model
from linopy.variables import Variable

Views = dict[str, tuple[xr.Dataset, Variable]]


@dataclass(frozen=True)
class Context:
    """
    Everything evaluating a node needs beyond the node.

    ``solved`` is the fold's switch: a build leaves it false and a variable
    enters an expression as its linopy term; a fold sets it true and a
    variable enters as its solved values, so a named expression reads off the
    primal. ``names`` maps a bound spec variable to the model variable it
    reads; a variable the spec introduced is absent and keeps its own name.
    ``views`` caches each bound variable reindexed onto the master
    coordinates, keyed by spec name and good for as long as the model
    variable's data is the one it was made from.
    """

    model: Model
    program: ms.Program
    coords: Mapping[str, pd.Index]
    lookups: Mapping[str, Mapping[str, xr.DataArray]]
    parameters: Mapping[str, xr.DataArray]
    solved: bool = field(default=False)
    names: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    views: Views = field(default_factory=dict)

    @property
    def unsolved(self) -> Context:
        """The same context with the fold's switch off, so a variable enters as its linopy term."""
        return replace(self, solved=False)

    def variable(self, name: str) -> Variable:
        """
        The model variable the spec variable *name* stands for, on the master coordinates.

        A bound variable spanning fewer labels than the master is a reindexed
        view, absent where it has none; it is a copy, so nothing written on it
        reaches ``model.variables``.
        """
        if name not in self.names:
            return self.model.variables[name]
        owned = self.model.variables[self.names[name]]
        cached = self.views.get(name)
        if cached is None or cached[0] is not owned.data:
            cached = (owned.data, _onto(owned, self.coords))
            self.views[name] = cached
        return cached[1]

    def lookup(self, name: str, over: str) -> xr.DataArray:
        """The lookup *name* as an array over *over*, NaN where a label is unmapped."""
        return self.lookups[over][name]


def _onto(variable: Variable, coords: Mapping[str, pd.Index]) -> Variable:
    """*variable* reindexed onto *coords* along every dimension it does not already span whole."""
    partial = {
        str(d): coords[str(d)]
        for d in variable.dims
        if not variable.indexes[d].equals(coords[str(d)])
    }
    return variable.reindex(partial) if partial else variable
