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

import pytest

from benchmarks.conftest import run_case
from benchmarks.phases import PhaseCase, phase_cases

_CASES = list(phase_cases("to_solver"))


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_to_solver(
    benchmark: Callable[..., object],
    case: PhaseCase,
    request: pytest.FixtureRequest,
) -> None:
    run_case(benchmark, case, request)
