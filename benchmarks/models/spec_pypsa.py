"""
Model built from math-spec's ``pypsa.yaml`` example (requires math-spec).

The subject is :meth:`linopy.Model.from_spec`: lowering a spec of PyPSA's full
statement, binding synthetic data to it and building every variable and
constraint it declares. The example lives outside the wheel, so its directory
comes from ``MATH_SPEC_EXAMPLES`` and the case skips without it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr

from benchmarks.registry import BUILD, FROM_NETCDF, TO_NETCDF, BenchSpec, register

if TYPE_CHECKING:
    import linopy

SIZES = (5, 40)  # labels per dimension; 40 is ~20k variables

EXAMPLES = os.environ.get("MATH_SPEC_EXAMPLES")
EXAMPLE = Path(EXAMPLES, "pypsa.yaml") if EXAMPLES else None


def synthetic_sources(program: Any, n: int) -> dict[str, Any]:
    """``n`` labels per dimension, a linear ramp per parameter, cyclic lookups."""
    sources: dict[str, Any] = {}
    for dim, decl in program.dimensions.items():
        if decl.dtype == "int":
            sources[dim] = pd.Index(range(n), name=dim)
        elif decl.dtype == "datetime":
            sources[dim] = pd.date_range("2030-01-01", periods=n, freq="h", name=dim)
        else:
            sources[dim] = pd.Index([f"{dim}{i}" for i in range(n)], name=dim)
    for over, lookup in program.lookups:
        into = sources[lookup.target] if lookup.target else range(n)
        sources[lookup.name] = pd.Series(
            [into[i % n] for i in range(n)], index=sources[over]
        )
    ramp = 1.0 + np.arange(n)
    for name, parameter in program.parameters.items():
        if parameter.derivation is not None:
            continue
        dims = parameter.dims
        shape = [n] * len(dims)
        if parameter.dtype == "bool":
            values: Any = np.ones(shape, dtype=bool)
        elif parameter.dtype == "int":
            values = np.ones(shape, dtype=int)
        elif parameter.dtype == "str":
            values = np.full(shape, "a", dtype=object)
        else:
            values = np.broadcast_to(ramp, shape).copy() if dims else np.array(1.0)
        sources[name] = (
            values.item()
            if not dims
            else xr.DataArray(
                values, coords={d: sources[d] for d in dims}, dims=list(dims)
            )
        )
    return sources


def build_spec_pypsa(n: int) -> linopy.Model:
    """Lower ``pypsa.yaml`` and build it with ``n`` labels per dimension."""
    import pytest

    if EXAMPLE is None or not EXAMPLE.exists():
        pytest.skip("set MATH_SPEC_EXAMPLES to a math-spec examples directory")
    import math_spec

    import linopy

    path = str(EXAMPLE)
    sources = synthetic_sources(math_spec.to_program(path), n)
    with linopy.options as options:
        options["semantics"] = "v1"  # a spec-built model is v1 only
        return linopy.Model.from_spec(path, sources)


SPEC = register(
    BenchSpec(
        name="spec_pypsa",
        build=build_spec_pypsa,
        sweep=SIZES,
        phases=frozenset({BUILD, TO_NETCDF, FROM_NETCDF}),
        requires=("math_spec",),
    )
)
