"""Benchmark configuration and shared fixtures."""

from __future__ import annotations

import pytest

from benchmarks.registry import BenchSpec

# Test modules the CodSpeed instruments measure (edit to change coverage).
# Covers construction, both solver-IO paths (lp_write = file, solver_handoff =
# direct in-memory), and the matrix build. test_netcdf is excluded — disk I/O is
# slow and noisy under walltime; it still runs under ``benchmarks smoke``.
CODSPEED_MODULES = (
    "test_build",
    "test_matrices",
    "test_lp_write",
    "test_solver_handoff",
)


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
    ``--quick`` drops the PyPSA end-to-end test (~30s; minutes under cachegrind).
    ``--codspeed`` narrows the run to ``CODSPEED_MODULES``.
    """
    if config.getoption("--quick"):
        skip = pytest.mark.skip(reason="--quick: pypsa end-to-end skipped")
        for item in items:
            if "test_pypsa_carbon_management" in item.nodeid:
                item.add_marker(skip)

    if getattr(config.option, "codspeed", False):
        deselected = [i for i in items if i.path.stem not in CODSPEED_MODULES]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = [i for i in items if i.path.stem in CODSPEED_MODULES]


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
