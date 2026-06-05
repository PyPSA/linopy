"""Benchmarks for matrix generation (model -> sparse matrices)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from benchmarks.conftest import maybe_skip
from benchmarks.phases import touch_matrices
from benchmarks.registry import MATRICES, ModelSpec, iter_params, param_ids

_PARAMS = iter_params(MATRICES)


@pytest.mark.parametrize("spec,size", _PARAMS, ids=param_ids(_PARAMS))
def test_matrices(
    benchmark: Callable[..., object],
    spec: ModelSpec,
    size: int,
    request: pytest.FixtureRequest,
) -> None:
    maybe_skip(request, spec, size)
    m = spec.build(size)
    benchmark(touch_matrices, m)
