"""Benchmarks for matrix generation (model -> sparse matrices)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from benchmarks.conftest import run_case
from benchmarks.phases import PhaseCase, phase_cases
from benchmarks.registry import MATRICES

_CASES = list(phase_cases(MATRICES))


@pytest.mark.parametrize("case", _CASES, ids=[c.id for c in _CASES])
def test_matrices(
    benchmark: Callable[..., object],
    case: PhaseCase,
    request: pytest.FixtureRequest,
) -> None:
    run_case(benchmark, case, request)
