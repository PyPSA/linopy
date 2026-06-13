"""
Benchmarks for solver handoff (model -> native solver instance).

Times each ``linopy.io.to_<solver>`` wrapper. These wrappers delegate to the
same direct-API build path as the new stateful Solver API
(``Solver.from_name(name, model, io_api="direct")``), so the numbers serve
double duty: regression tracking for the wrappers, *and* for the underlying
``Solver._build_direct`` paths. They've also been available for many releases
— using them keeps the suite runnable on older linopy versions.

The actual ``Solver.solve()`` runtime (i.e. solver-side algorithm time) is
intentionally not benchmarked.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import pytest

from benchmarks.conftest import build_model
from benchmarks.phases import SOLVER_HANDOFFS
from benchmarks.registry import iter_params, spec_param_id
from linopy.solvers import available_solvers

if TYPE_CHECKING:
    from benchmarks.registry import BenchSpec

# One case per (available solver wrapper) × (spec, value) it applies to.
_PARAMS = [
    (name, wrapper, spec, n)
    for name, tag, wrapper in SOLVER_HANDOFFS
    for spec, n in iter_params(tag)
]
_IDS = [f"{name}-{spec_param_id(s.name, s.axis, v)}" for name, _w, s, v in _PARAMS]


@pytest.mark.parametrize(("name", "wrapper", "spec", "n"), _PARAMS, ids=_IDS)
def test_to_solver(
    benchmark: Callable[..., object],
    name: str,
    wrapper: Callable[..., object],
    spec: BenchSpec,
    n: int,
) -> None:
    if name not in available_solvers:
        pytest.skip(f"{name} not installed")
    m = build_model(spec, n)
    benchmark(lambda: wrapper(m))
