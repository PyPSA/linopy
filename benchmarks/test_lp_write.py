"""Benchmarks for LP file writing speed."""

from __future__ import annotations

import pytest

from benchmarks.conftest import maybe_skip
from benchmarks.registry import LP_WRITE, iter_params, param_ids

_PARAMS = iter_params(LP_WRITE)


@pytest.mark.parametrize("spec,size", _PARAMS, ids=param_ids(_PARAMS))
def test_lp_write(benchmark, spec, size, request, tmp_path):
    maybe_skip(request, spec, size)
    m = spec.build(size)
    lp_file = tmp_path / "model.lp"
    benchmark(m.to_file, lp_file, progress=False)
