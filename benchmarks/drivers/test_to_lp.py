"""Benchmarks for LP file writing speed."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from benchmarks.conftest import build_model, cases
from benchmarks.phases import write_lp
from benchmarks.registry import TO_LP

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.registry import BenchSpec


@cases(TO_LP)
def test_to_lp(
    benchmark: Callable[..., object], spec: BenchSpec, n: int, tmp_path: Path
) -> None:
    m = build_model(spec, n)
    path = tmp_path / "model.lp"
    benchmark(lambda: write_lp(m, path))
