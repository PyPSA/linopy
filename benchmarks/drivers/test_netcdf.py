"""
Benchmarks for the netCDF persistence round-trip.

We track ``to_netcdf`` and ``read_netcdf`` separately because the cost split
matters in practice: distributed workflows tend to do many reads of a single
written artifact.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from benchmarks.conftest import build_model, cases
from benchmarks.phases import read_netcdf, write_netcdf
from benchmarks.registry import FROM_NETCDF, TO_NETCDF

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.registry import BenchSpec


@cases(TO_NETCDF)
def test_to_netcdf(
    benchmark: Callable[..., object], spec: BenchSpec, n: int, tmp_path: Path
) -> None:
    m = build_model(spec, n)
    benchmark(lambda: write_netcdf(m, tmp_path / "model.nc"))


@cases(FROM_NETCDF)
def test_from_netcdf(
    benchmark: Callable[..., object], spec: BenchSpec, n: int, tmp_path: Path
) -> None:
    m = build_model(spec, n)
    path = tmp_path / "model.nc"
    write_netcdf(m, path)  # setup — untimed
    benchmark(lambda: read_netcdf(path))
