"""Benchmark configuration and shared fixtures."""

from __future__ import annotations

import pytest

from benchmarks.registry import BenchSpec


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--quick",
        action="store_true",
        default=False,
        help="Use smaller problem sizes for quick benchmarking (CI smoke).",
    )
    parser.addoption(
        "--long",
        action="store_true",
        default=False,
        help=(
            "Include the slowest sizes (above each spec's long_threshold). "
            "Default runs skip them."
        ),
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    Drop PyPSA end-to-end tests under ``--quick``.

    The PyPSA carbon-management network is ~30s by itself; CodSpeed under
    cachegrind would make it minutes. ``--quick`` is for sub-30s sweeps,
    so the end-to-end module doesn't belong there.
    """
    if not config.getoption("--quick"):
        return
    skip = pytest.mark.skip(reason="--quick: pypsa end-to-end skipped")
    for item in items:
        if "test_pypsa_carbon_management" in item.nodeid:
            item.add_marker(skip)


def maybe_skip(request: pytest.FixtureRequest, spec: BenchSpec, size: int) -> None:
    """
    Apply size-tier skips and ``spec.requires`` importorskips.

    Tiers (most restrictive first):

    - ``--quick``                 → skip ``size > quick_threshold``
    - default (no flag)           → skip ``size > long_threshold``
    - ``--long``                  → no size cap

    If both ``--quick`` and ``--long`` are passed, ``--quick`` wins (the more
    restrictive mode is honoured).
    """
    for mod in spec.requires:
        pytest.importorskip(mod)

    quick = request.config.getoption("--quick")
    long_ = request.config.getoption("--long")

    if quick:
        if size > spec.quick_threshold:
            pytest.skip(f"--quick: skipping {spec.name} {spec.axis}={size}")
    elif not long_:
        if size > spec.long_threshold:
            pytest.skip(
                f"long sweep needs --long: skipping {spec.name} {spec.axis}={size}"
            )
