"""Benchmarks for model construction speed."""

from __future__ import annotations

import pytest

from benchmarks.conftest import maybe_skip
from benchmarks.registry import BUILD, iter_params, param_ids

_PARAMS = iter_params(BUILD)


@pytest.mark.parametrize("spec,size", _PARAMS, ids=param_ids(_PARAMS))
def test_build(benchmark, spec, size, request):
    maybe_skip(request, spec, size)
    benchmark(spec.build, size)
