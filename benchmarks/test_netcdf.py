"""
Benchmarks for the netCDF persistence round-trip.

We track ``to_netcdf`` and ``read_netcdf`` separately because the cost split
matters in practice: distributed workflows tend to do many reads of a single
written artifact.
"""

from __future__ import annotations

import pytest

from benchmarks.conftest import maybe_skip
from benchmarks.registry import NETCDF, iter_params, param_ids
from linopy import read_netcdf

_PARAMS = iter_params(NETCDF)


@pytest.mark.parametrize("spec,size", _PARAMS, ids=param_ids(_PARAMS))
def test_netcdf_write(benchmark, spec, size, request, tmp_path):
    maybe_skip(request, spec, size)
    m = spec.build(size)
    out = tmp_path / "model.nc"
    benchmark(m.to_netcdf, out)


@pytest.mark.parametrize("spec,size", _PARAMS, ids=param_ids(_PARAMS))
def test_netcdf_read(benchmark, spec, size, request, tmp_path):
    maybe_skip(request, spec, size)
    m = spec.build(size)
    out = tmp_path / "model.nc"
    m.to_netcdf(out)
    benchmark(read_netcdf, out)
