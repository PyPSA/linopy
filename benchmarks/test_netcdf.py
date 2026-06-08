"""
Benchmarks for the netCDF persistence round-trip.

We track ``to_netcdf`` and ``read_netcdf`` separately because the cost split
matters in practice: distributed workflows tend to do many reads of a single
written artifact.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from benchmarks.conftest import run_case
from benchmarks.phases import PhaseCase, phase_cases
from benchmarks.registry import FROM_NETCDF, TO_NETCDF

_WRITE_CASES = list(phase_cases(TO_NETCDF))
_READ_CASES = list(phase_cases(FROM_NETCDF))


@pytest.mark.parametrize("case", _WRITE_CASES, ids=[c.id for c in _WRITE_CASES])
def test_to_netcdf(
    benchmark: Callable[..., object],
    case: PhaseCase,
    request: pytest.FixtureRequest,
) -> None:
    run_case(benchmark, case, request)


@pytest.mark.parametrize("case", _READ_CASES, ids=[c.id for c in _READ_CASES])
def test_from_netcdf(
    benchmark: Callable[..., object],
    case: PhaseCase,
    request: pytest.FixtureRequest,
) -> None:
    run_case(benchmark, case, request)
