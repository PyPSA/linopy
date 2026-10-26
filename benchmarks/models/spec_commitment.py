"""
Models built from mathspec specs (requires mathspec).

The subject is :meth:`linopy.Model.from_spec`: lowering a spec, attaching
synthetic data to it and building every variable and constraint it declares.
Two specs: mathspec's ``commitment.yaml`` example, which lives outside the
wheel so its directory comes from ``MATHSPEC_EXAMPLES`` and the case skips
without it, and an inline relation case shaped like ``operators/sum_by.yaml``,
so the grouped sum through a relation is measured. A sweep value is the
number of labels per dimension.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.registry import BUILD, FROM_NETCDF, TO_NETCDF, BenchSpec, register

if TYPE_CHECKING:
    import linopy

SIZES = (5, 40)

EXAMPLES = os.environ.get("MATHSPEC_EXAMPLES")
EXAMPLE = Path(EXAMPLES, "commitment.yaml") if EXAMPLES else None

SUM_BY = """
dimensions:
  snapshot: { dtype: int }
  generator: { dtype: str }
  bus: { dtype: str }
relations:
  gen_bus: { key: generator, values: bus }
parameters:
  limit: { dims: [snapshot, bus] }
variables:
  p:
    dims: [snapshot, generator]
    bounds: { lower: 0 }
constraints:
  bus_total:
    dims: [snapshot, bus]
    expression: sum(p, over=generator, by=gen_bus[bus]) <= limit
objective: { sense: minimize, expression: sum(p) }
"""


def _build(spec: str, n: int) -> linopy.Model:
    import linopy
    from linopy.spec import lower, synthetic_sources

    return linopy.Model.from_spec(spec, synthetic_sources(lower(spec), n))


def build_spec_commitment(n: int) -> linopy.Model:
    """Lower ``commitment.yaml`` and build it with ``n`` labels per dimension."""
    import pytest

    if EXAMPLE is None or not EXAMPLE.exists():
        pytest.skip("set MATHSPEC_EXAMPLES to a mathspec examples directory")
    return _build(str(EXAMPLE), n)


def build_spec_sum_by(n: int) -> linopy.Model:
    """Build the grouped-sum spec with ``n`` labels per dimension."""
    return _build(SUM_BY, n)


COMMITMENT = register(
    BenchSpec(
        name="spec_commitment",
        build=build_spec_commitment,
        sweep=SIZES,
        phases=frozenset({BUILD, TO_NETCDF, FROM_NETCDF}),
        requires=("mathspec",),
    )
)

SUM_BY_SPEC = register(
    BenchSpec(
        name="spec_sum_by",
        build=build_spec_sum_by,
        sweep=SIZES,
        phases=frozenset({BUILD, TO_NETCDF, FROM_NETCDF}),
        requires=("mathspec",),
    )
)
