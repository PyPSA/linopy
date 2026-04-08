"""Benchmarks for LP file writing speed."""

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
def test_lp_write_basic(benchmark, n, request, tmp_path):
    skip_if_quick(request, "basic", n)
    m = build_basic(n)
    lp_file = tmp_path / "model.lp"
    benchmark(m.to_file, lp_file, progress=False)


@pytest.mark.parametrize("n", KNAPSACK_SIZES, ids=[f"n={n}" for n in KNAPSACK_SIZES])
def test_lp_write_knapsack(benchmark, n, request, tmp_path):
    skip_if_quick(request, "knapsack", n)
    m = build_knapsack(n)
    lp_file = tmp_path / "model.lp"
    benchmark(m.to_file, lp_file, progress=False)


@pytest.mark.parametrize("n", EXPR_SIZES, ids=[f"n={n}" for n in EXPR_SIZES])
def test_lp_write_expression_arithmetic(benchmark, n, request, tmp_path):
    skip_if_quick(request, "expression_arithmetic", n)
    m = build_expression_arithmetic(n)
    lp_file = tmp_path / "model.lp"
    benchmark(m.to_file, lp_file, progress=False)


@pytest.mark.parametrize("n", SPARSE_SIZES, ids=[f"n={n}" for n in SPARSE_SIZES])
def test_lp_write_sparse_network(benchmark, n, request, tmp_path):
    skip_if_quick(request, "sparse_network", n)
    m = build_sparse_network(n)
    lp_file = tmp_path / "model.lp"
    benchmark(m.to_file, lp_file, progress=False)


@pytest.mark.parametrize(
    "snapshots", PYPSA_SIZES, ids=[f"snapshots={s}" for s in PYPSA_SIZES]
)
def test_lp_write_pypsa_scigrid(benchmark, snapshots, request, tmp_path):
    pytest.importorskip("pypsa")
    skip_if_quick(request, "pypsa_scigrid", snapshots)
    from benchmarks.models.pypsa_scigrid import build_pypsa_scigrid

    m = build_pypsa_scigrid(snapshots)
    lp_file = tmp_path / "model.lp"
    benchmark(m.to_file, lp_file, progress=False)
