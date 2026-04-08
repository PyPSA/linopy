"""Benchmarks for matrix generation (model -> sparse matrices)."""

from __future__ import annotations

import pytest

from benchmarks.conftest import skip_if_quick
from benchmarks.models import (
    BASIC_SIZES,
    EXPR_SIZES,
    SPARSE_SIZES,
    build_basic,
    build_expression_arithmetic,
    build_sparse_network,
)


def _access_matrices(m):
    """Access all matrix properties to force computation."""
    m.matrices.clean_cached_properties()
    _ = m.matrices.A
    _ = m.matrices.b
    _ = m.matrices.c
    _ = m.matrices.lb
    _ = m.matrices.ub
    _ = m.matrices.sense
    _ = m.matrices.vlabels
    _ = m.matrices.clabels


@pytest.mark.parametrize("n", BASIC_SIZES, ids=[f"n={n}" for n in BASIC_SIZES])
def test_matrices_basic(benchmark, n, request):
    skip_if_quick(request, "basic", n)
    m = build_basic(n)
    benchmark(_access_matrices, m)


@pytest.mark.parametrize("n", EXPR_SIZES, ids=[f"n={n}" for n in EXPR_SIZES])
def test_matrices_expression_arithmetic(benchmark, n, request):
    skip_if_quick(request, "expression_arithmetic", n)
    m = build_expression_arithmetic(n)
    benchmark(_access_matrices, m)


@pytest.mark.parametrize("n", SPARSE_SIZES, ids=[f"n={n}" for n in SPARSE_SIZES])
def test_matrices_sparse_network(benchmark, n, request):
    skip_if_quick(request, "sparse_network", n)
    m = build_sparse_network(n)
    benchmark(_access_matrices, m)
