"""Benchmarks for model construction speed."""

from __future__ import annotations

import pytest

from benchmarks.conftest import skip_if_quick
from benchmarks.models import (
    BASIC_SIZES,
    EXPR_SIZES,
    KNAPSACK_SIZES,
    SPARSE_SIZES,
    build_basic,
    build_expression_arithmetic,
    build_knapsack,
    build_sparse_network,
)
from benchmarks.models.pypsa_scigrid import SIZES as PYPSA_SIZES


@pytest.mark.parametrize("n", BASIC_SIZES, ids=[f"n={n}" for n in BASIC_SIZES])
def test_build_basic(benchmark, n, request):
    skip_if_quick(request, "basic", n)
    benchmark(build_basic, n)


@pytest.mark.parametrize("n", KNAPSACK_SIZES, ids=[f"n={n}" for n in KNAPSACK_SIZES])
def test_build_knapsack(benchmark, n, request):
    skip_if_quick(request, "knapsack", n)
    benchmark(build_knapsack, n)


@pytest.mark.parametrize("n", EXPR_SIZES, ids=[f"n={n}" for n in EXPR_SIZES])
def test_build_expression_arithmetic(benchmark, n, request):
    skip_if_quick(request, "expression_arithmetic", n)
    benchmark(build_expression_arithmetic, n)


@pytest.mark.parametrize("n", SPARSE_SIZES, ids=[f"n={n}" for n in SPARSE_SIZES])
def test_build_sparse_network(benchmark, n, request):
    skip_if_quick(request, "sparse_network", n)
    benchmark(build_sparse_network, n)


@pytest.mark.parametrize(
    "snapshots", PYPSA_SIZES, ids=[f"snapshots={s}" for s in PYPSA_SIZES]
)
def test_build_pypsa_scigrid(benchmark, snapshots, request):
    pytest.importorskip("pypsa")
    skip_if_quick(request, "pypsa_scigrid", snapshots)
    from benchmarks.models.pypsa_scigrid import build_pypsa_scigrid

    benchmark(build_pypsa_scigrid, snapshots)
