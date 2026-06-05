"""Benchmarks for LP file writing speed."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from benchmarks.conftest import maybe_skip
from benchmarks.phases import write_lp
from benchmarks.registry import LP_WRITE, ModelSpec, iter_params, param_ids

_PARAMS = iter_params(LP_WRITE)


@pytest.mark.parametrize("spec,size", _PARAMS, ids=param_ids(_PARAMS))
def test_lp_write(
    benchmark: Callable[..., object],
    spec: ModelSpec,
    size: int,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    maybe_skip(request, spec, size)
    m = spec.build(size)
    lp_file = tmp_path / "model.lp"
    benchmark(write_lp, m, lp_file)
