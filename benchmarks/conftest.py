"""Benchmark configuration and shared fixtures."""

from __future__ import annotations

import pytest

from benchmarks.registry import BenchSpec

# Test modules the CodSpeed instruments measure (edit to change coverage).
# build + the two export paths: to_lp (LP text) and to_solver (direct handoff,
# which also exercises matrix-gen). matrices is dropped — a subset of to_solver;
# netcdf excluded — disk I/O, noisy. All still run under ``benchmarks smoke``.
CODSPEED_MODULES = (
    "test_build",
    "test_to_lp",
    "test_to_solver",
)

# Cachegrind (simulation) is ~hundreds× native and under-weights sparse/native
# work, so the simulation job trims to the cheap phases; to_solver is carried
# per-PR by the memory instrument and on master by walltime instead.
CODSPEED_SIMULATION_MODULES = ("test_build", "test_to_lp")

# Only the cachegrind (simulation) job trims; memory/walltime use the full set
# (the default). Chosen via our own ``--codspeed-set`` option — no
# pytest-codspeed internals, no silent mode-sniffing.
CODSPEED_SETS = {
    "full": CODSPEED_MODULES,
    "simulation": CODSPEED_SIMULATION_MODULES,
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--quick",
        action="store_true",
        default=False,
        help="Use smaller problem sizes for quick benchmarking (CI smoke).",
    )
    parser.addoption(
        "--codspeed-set",
        choices=sorted(CODSPEED_SETS),
        default="full",
        help=(
            "Which CodSpeed module subset to run (default 'full'; 'simulation' "
            "trims the expensive cachegrind phases). Only takes effect under "
            "--codspeed."
        ),
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
    ``--codspeed`` narrows the run to the ``--codspeed-set`` instrument subset.
    """
    if config.getoption("--quick"):
        skip = pytest.mark.skip(reason="--quick: pypsa end-to-end skipped")
        for item in items:
            if "test_pypsa_carbon_management" in item.nodeid:
                item.add_marker(skip)

    if getattr(config.option, "codspeed", False):
        modules = CODSPEED_SETS[config.getoption("--codspeed-set")]
        deselected = [i for i in items if i.path.stem not in modules]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = [i for i in items if i.path.stem in modules]


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
