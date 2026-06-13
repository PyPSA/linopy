"""Benchmarks for matrix generation (model -> sparse matrices)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from benchmarks.conftest import build_model, cases
from benchmarks.phases import touch_matrices
from benchmarks.registry import MATRICES

if TYPE_CHECKING:
    from benchmarks.registry import BenchSpec


@cases(MATRICES)
def test_matrices(benchmark: Callable[..., object], spec: BenchSpec, n: int) -> None:
    m = build_model(spec, n)
    benchmark(lambda: touch_matrices(m))
