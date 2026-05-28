"""Benchmarks for matrix generation (model -> sparse matrices)."""

from __future__ import annotations

import pytest

from benchmarks.conftest import maybe_skip
from benchmarks.registry import MATRICES, iter_params, param_ids

_PARAMS = iter_params(MATRICES)


def _access_matrices(m):
    """Touch every matrix property to force computation."""
    matrices = m.matrices
    _ = matrices.A
    _ = matrices.b
    _ = matrices.c
    _ = matrices.lb
    _ = matrices.ub
    _ = matrices.sense
    _ = matrices.vlabels
    _ = matrices.clabels
    if m.is_quadratic:
        _ = matrices.Q  # exercise the QP path when present


@pytest.mark.parametrize("spec,size", _PARAMS, ids=param_ids(_PARAMS))
def test_matrices(benchmark, spec, size, request):
    maybe_skip(request, spec, size)
    m = spec.build(size)
    benchmark(_access_matrices, m)
