"""Benchmark configuration and shared fixtures."""

from __future__ import annotations

import pytest

QUICK_THRESHOLD = {
    "basic": 100,
    "knapsack": 10_000,
    "pypsa_scigrid": 50,
    "expression_arithmetic": 100,
    "sparse_network": 100,
}


def pytest_addoption(parser):
    parser.addoption(
        "--quick",
        action="store_true",
        default=False,
        help="Use smaller problem sizes for quick benchmarking",
    )


def skip_if_quick(request, model: str, size: int):
    """Skip large sizes when --quick is passed."""
    if request.config.getoption("--quick"):
        threshold = QUICK_THRESHOLD.get(model, float("inf"))
        if size > threshold:
            pytest.skip(f"--quick: skipping {model} size {size}")
