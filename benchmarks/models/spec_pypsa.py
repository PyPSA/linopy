"""
Model built from math-spec's ``pypsa.yaml`` example (requires math-spec).

The subject is :meth:`linopy.Model.from_spec`: lowering a spec of PyPSA's full
statement, binding synthetic data to it and building every variable and
constraint it declares. The example lives outside the wheel, so its directory
comes from ``MATH_SPEC_EXAMPLES`` and the case skips without it. A sweep
value is the number of labels per dimension; 40 of them is about 20k
variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.registry import BUILD, FROM_NETCDF, TO_NETCDF, BenchSpec, register

if TYPE_CHECKING:
    import linopy

SIZES = (5, 40)

EXAMPLES = os.environ.get("MATH_SPEC_EXAMPLES")
EXAMPLE = Path(EXAMPLES, "pypsa.yaml") if EXAMPLES else None


def build_spec_pypsa(n: int) -> linopy.Model:
    """Lower ``pypsa.yaml`` and build it with ``n`` labels per dimension."""
    import pytest

    if EXAMPLE is None or not EXAMPLE.exists():
        pytest.skip("set MATH_SPEC_EXAMPLES to a math-spec examples directory")
    import math_spec

    import linopy
    from linopy.spec.testing import synthetic_sources

    path = str(EXAMPLE)
    sources = synthetic_sources(math_spec.to_program(path), n)
    with linopy.options as options:
        options["semantics"] = "v1"
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
