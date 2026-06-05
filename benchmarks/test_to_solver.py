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

from benchmarks.conftest import maybe_skip
from benchmarks.phases import SOLVER_HANDOFFS
from benchmarks.registry import ModelSpec, iter_params
from benchmarks.snapshot import spec_param_id
from linopy.solvers import available_solvers


def _make_params() -> list[object]:
    out: list[object] = []
    for solver_name, phase, wrapper in SOLVER_HANDOFFS:
        for spec, size in iter_params(phase):
            out.append(
                pytest.param(
                    solver_name,
                    wrapper,
                    spec,
                    size,
                    id=f"{solver_name}-{spec_param_id(spec.name, spec.axis, size)}",
                )
            )
    return out


@pytest.mark.parametrize("solver_name,wrapper,spec,size", _make_params())
def test_to_solver(
    benchmark: Callable[..., object],
    solver_name: str,
    wrapper: Callable[..., object],
    spec: ModelSpec,
    size: int,
    request: pytest.FixtureRequest,
) -> None:
    if solver_name not in available_solvers:
        pytest.skip(f"{solver_name} not installed")
    maybe_skip(request, spec, size)
    model = spec.build(size)
    benchmark(wrapper, model)
