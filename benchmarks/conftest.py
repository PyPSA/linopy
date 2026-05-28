"""Benchmark configuration and shared fixtures."""

from __future__ import annotations

import pytest

from benchmarks.registry import ModelSpec


def pytest_addoption(parser):
    parser.addoption(
        "--quick",
        action="store_true",
        default=False,
        help="Use smaller problem sizes for quick benchmarking",
    )


def maybe_skip(request: pytest.FixtureRequest, spec: ModelSpec, size: int) -> None:
    """
    Apply ``--quick`` size cap and ``spec.requires`` importorskips.

    Centralised so every phase test stays a one-liner.
    """
    for mod in spec.requires:
        pytest.importorskip(mod)
    if request.config.getoption("--quick") and size > spec.quick_threshold:
        pytest.skip(f"--quick: skipping {spec.name} size {size}")
