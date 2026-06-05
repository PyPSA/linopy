"""Benchmarks for model construction speed."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from benchmarks.conftest import maybe_skip
from benchmarks.registry import BUILD, ModelSpec, iter_params, param_ids

_PARAMS = iter_params(BUILD)


@pytest.mark.parametrize("spec,size", _PARAMS, ids=param_ids(_PARAMS))
def test_build(
    benchmark: Callable[..., object],
    spec: ModelSpec,
    size: int,
    request: pytest.FixtureRequest,
) -> None:
    maybe_skip(request, spec, size)
    benchmark(spec.build, size)
