"""
Benchmarks for the netCDF persistence round-trip.

We track ``to_netcdf`` and ``read_netcdf`` separately because the cost split
matters in practice: distributed workflows tend to do many reads of a single
written artifact.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from benchmarks.conftest import maybe_skip
from benchmarks.phases import read_netcdf, write_netcdf
from benchmarks.registry import (
    FROM_NETCDF,
    TO_NETCDF,
    ModelSpec,
    iter_params,
    param_ids,
)

_WRITE_PARAMS = iter_params(TO_NETCDF)
_READ_PARAMS = iter_params(FROM_NETCDF)


@pytest.mark.parametrize("spec,size", _WRITE_PARAMS, ids=param_ids(_WRITE_PARAMS))
def test_to_netcdf(
    benchmark: Callable[..., object],
    spec: ModelSpec,
    size: int,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    maybe_skip(request, spec, size)
    m = spec.build(size)
    out = tmp_path / "model.nc"
    benchmark(write_netcdf, m, out)


@pytest.mark.parametrize("spec,size", _READ_PARAMS, ids=param_ids(_READ_PARAMS))
def test_from_netcdf(
    benchmark: Callable[..., object],
    spec: ModelSpec,
    size: int,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    maybe_skip(request, spec, size)
    m = spec.build(size)
    out = tmp_path / "model.nc"
    write_netcdf(m, out)
    benchmark(read_netcdf, out)
