"""Benchmark configuration and shared fixtures."""

from __future__ import annotations

import pytest

from benchmarks.registry import BenchSpec, skip_reason

# Test modules the CodSpeed instruments measure (edit to change coverage).
# build + the two export paths: to_lp (LP text) and to_solver (direct handoff,
# which also exercises matrix-gen). matrices is dropped — a subset of to_solver;
# netcdf excluded — disk I/O, noisy. All still run under ``benchmarks smoke``.
CODSPEED_MODULES = (
    "test_build",
    "test_to_lp",
    "test_to_solver",
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
    parser.addoption(
        "--size",
        action="append",
        type=int,
        default=[],
        metavar="N",
        help=(
            "Run only these model sizes (repeatable). Overrides --quick/--long "
            "for models, leaving patterns on the prevailing tier."
        ),
    )
    parser.addoption(
        "--severity",
        action="append",
        type=int,
        default=[],
        metavar="S",
        help=(
            "Run only these pattern severities (repeatable). Overrides "
            "--quick/--long for patterns, leaving models on the prevailing tier."
        ),
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    ``--quick`` drops the PyPSA end-to-end test (~30s; minutes under cachegrind).
    ``--codspeed`` narrows the run to ``CODSPEED_MODULES`` (drops netcdf/matrices).
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
    Apply size selection and ``spec.requires`` importorskips.

    Selection (most specific first):

    - ``--size N`` / ``--severity S`` → run only the listed values for that
      axis (models read ``--size``, patterns ``--severity``); overrides tiers.
    - ``--quick``                     → only ``spec.quick_subset``
    - default (no flag)               → skip ``size > long_threshold``
    - ``--long``                      → no size cap

    A manual axis flag wins over ``--quick``/``--long``; ``--quick`` in turn
    wins over ``--long`` (the more restrictive mode is honoured).
    """
    for mod in spec.requires:
        pytest.importorskip(mod)

    reason = skip_reason(
        spec,
        size,
        quick=request.config.getoption("--quick"),
        long=request.config.getoption("--long"),
        sizes=tuple(request.config.getoption("--size")),
        severities=tuple(request.config.getoption("--severity")),
    )
    if reason:
        pytest.skip(reason)
