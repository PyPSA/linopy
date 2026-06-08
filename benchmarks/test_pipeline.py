"""
End-to-end pipeline benchmark: build → matrices → LP write in one region.

Opt-in (deselected unless ``--pipeline``): it re-runs the per-phase work and,
unlike the individual phase benchmarks, *includes the model build* — so it
captures the end-to-end cost/peak a real build-then-export session hits, which
can't be recovered by summing the marginal per-phase numbers. The memory side
measures the same thing via ``... --metric memory --phase pipeline``.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from benchmarks.conftest import run_case
from benchmarks.phases import PIPELINE, PhaseCase, phase_cases

_CASES = list(phase_cases(PIPELINE))


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_pipeline(
    benchmark: Callable[..., object],
    case: PhaseCase,
    request: pytest.FixtureRequest,
) -> None:
    run_case(benchmark, case, request)
