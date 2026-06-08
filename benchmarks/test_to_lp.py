"""Benchmarks for LP file writing speed."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from benchmarks.conftest import run_case
from benchmarks.phases import PhaseCase, phase_cases
from benchmarks.registry import TO_LP

_CASES = list(phase_cases(TO_LP))


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_to_lp(
    benchmark: Callable[..., object],
    case: PhaseCase,
    request: pytest.FixtureRequest,
) -> None:
    run_case(benchmark, case, request)
