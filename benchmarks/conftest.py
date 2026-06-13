"""Benchmark configuration and shared fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from benchmarks.phases import PhaseCase

# Test modules the CodSpeed instruments measure (edit to change coverage).
# build + the two export paths: to_lp (LP text) and to_solver (direct handoff,
# which also exercises matrix-gen). matrices is dropped — a subset of to_solver;
# netcdf excluded — disk I/O, noisy. All still run under the smoke job.
CODSPEED_MODULES = (
    "test_build",
    "test_to_lp",
    "test_to_solver",
)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--pipeline",
        action="store_true",
        default=False,
        help=(
            "Include the opt-in end-to-end pipeline benchmark (build → matrices "
            "→ lp in one measured region). Off by default — it re-runs the "
            "per-phase work and includes the build."
        ),
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """
    ``test_pipeline`` (end-to-end) is opt-in — deselected unless ``--pipeline``.
    ``--codspeed`` narrows the run to ``CODSPEED_MODULES`` (drops netcdf/matrices).
    """
    if not config.getoption("--pipeline"):
        dropped = [i for i in items if i.path.stem == "test_pipeline"]
        if dropped:
            config.hook.pytest_deselected(items=dropped)
            items[:] = [i for i in items if i.path.stem != "test_pipeline"]

    if getattr(config.option, "codspeed", False):
        deselected = [i for i in items if i.path.stem not in CODSPEED_MODULES]
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = [i for i in items if i.path.stem in CODSPEED_MODULES]


def run_case(benchmark: Callable[..., object], case: PhaseCase) -> None:
    """
    Shared pytest-benchmark driver for one :class:`PhaseCase`: honour its
    ``skip`` and ``requires``, then run its action under ``benchmark``.
    """
    if case.skip:
        pytest.skip(case.skip)
    for mod in case.spec.requires:
        pytest.importorskip(mod)
    with case.run() as action:
        benchmark(action)
