"""
End-to-end pipeline benchmark: build → matrices → LP write in one region.

Opt-in (deselected unless ``--pipeline``): it re-runs the per-phase work and,
unlike the individual phase benchmarks, *includes the model build* — so it
captures the end-to-end cost/peak a real build-then-export session hits, which
can't be recovered by summing the marginal per-phase numbers. Parametrized over
the ``to_lp`` specs (it ends in an LP write).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from benchmarks.conftest import cases, require
from benchmarks.phases import touch_matrices, write_lp
from benchmarks.registry import TO_LP

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarks.registry import BenchSpec


@cases(TO_LP)
def test_pipeline(
    benchmark: Callable[..., object], spec: BenchSpec, n: int, tmp_path: Path
) -> None:
    require(spec)
    path = tmp_path / "model.lp"

    def pipeline() -> None:
        m = spec.build(n)
        touch_matrices(m)
        write_lp(m, path)

    benchmark(pipeline)
