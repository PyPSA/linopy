"""Benchmarks for model construction speed."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from benchmarks.conftest import cases, require
from benchmarks.registry import BUILD

if TYPE_CHECKING:
    from benchmarks.registry import BenchSpec


@cases(BUILD)
def test_build(benchmark: Callable[..., object], spec: BenchSpec, n: int) -> None:
    require(spec)
    benchmark(lambda: spec.build(n))
