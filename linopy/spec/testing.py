"""
Synthetic data for a spec, for tests and benchmarks.

A spec says what data it takes, which is enough to make some up: the shape is
the declaration's, only the values are invented. What comes out builds and
solves, and says nothing about a real system.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from math_spec import program as ms

_START = "2030-01-01"


def synthetic_sources(program: ms.Program, n: int = 3) -> dict[str, Any]:
    """
    Dense data for every declaration of *program*, *n* labels per dimension.

    Labels are numbered after their dimension, parameters are a linear ramp,
    and each lookup cycles through the labels it maps into.
    """
    sources: dict[str, Any] = {
        dim: _labels(dim, decl.dtype, n) for dim, decl in program.dimensions.items()
    }
    for over, lookup in program.lookups:
        into = (
            sources[lookup.target]
            if lookup.target is not None
            else _labels(lookup.name, lookup.dtype, n)
        )
        sources[lookup.name] = pd.Series(
            [into[i % len(into)] for i in range(n)], index=sources[over]
        )
    for name, parameter in program.parameters.items():
        if parameter.derivation is None:
            sources[name] = _parameter(name, parameter, sources, n)
    return sources


def _labels(name: str, dtype: str | None, n: int) -> pd.Index:
    """*n* labels of the declared dtype, named after what they label."""
    if dtype == "int":
        return pd.Index(range(n), name=name)
    if dtype == "datetime":
        return pd.date_range(_START, periods=n, freq="h", name=name)
    return pd.Index([f"{name}{i}" for i in range(n)], name=name)


def _parameter(
    name: str, declared: ms.ParameterDeclaration, sources: dict[str, Any], n: int
) -> Any:
    dims = declared.dims
    shape = [n] * len(dims)
    if declared.dtype == "bool":
        values: Any = np.ones(shape, dtype=bool)
    elif declared.dtype == "int":
        values = np.ones(shape, dtype=int)
    elif declared.dtype == "str":
        values = np.full(shape, "a", dtype=object)
    elif dims:
        values = np.broadcast_to(1.0 + np.arange(n), shape).copy()
    else:
        values = np.array(1.0)
    if not dims:
        return values.item()
    return xr.DataArray(values, coords={d: sources[d] for d in dims}, dims=list(dims))
