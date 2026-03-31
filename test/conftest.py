"""Pytest configuration and fixtures."""

from __future__ import annotations

import os
import warnings
from collections.abc import Generator
from typing import TYPE_CHECKING

import pandas as pd
import pytest

if TYPE_CHECKING:
    from linopy import Model, Variable


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options."""
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run tests that require GPU hardware",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers and behavior."""
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU hardware")
    config.addinivalue_line(
        "markers", "legacy_only: test runs only under the legacy arithmetic convention"
    )
    config.addinivalue_line(
        "markers", "v1_only: test runs only under the v1 arithmetic convention"
    )

    # Set environment variable so test modules can check if GPU tests are enabled
    # This is needed because parametrize happens at import time
    if config.getoption("--run-gpu", default=False):
        os.environ["LINOPY_RUN_GPU_TESTS"] = "1"
    else:
        os.environ.pop("LINOPY_RUN_GPU_TESTS", None)


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Automatically skip GPU tests unless --run-gpu is passed."""
    if config.getoption("--run-gpu"):
        return

    skip_gpu = pytest.mark.skip(reason="need --run-gpu option to run GPU tests")
    for item in items:
        # Check if this is a parametrized test with a GPU solver
        if hasattr(item, "callspec") and "solver" in item.callspec.params:
            solver = item.callspec.params["solver"]
            # Import here to avoid circular dependency
            from linopy.solver_capabilities import (
                SolverFeature,
                solver_supports,
            )

            if solver_supports(solver, SolverFeature.GPU_ACCELERATION):
                item.add_marker(skip_gpu)
                item.add_marker(pytest.mark.gpu)


@pytest.fixture(autouse=True, params=["legacy", "v1"])
def convention(request: pytest.FixtureRequest) -> Generator[str, None, None]:
    """
    Run every test under both arithmetic conventions by default.

    Tests marked ``@pytest.mark.legacy_only`` or ``@pytest.mark.v1_only``
    are automatically skipped for the other convention.

    Under "legacy", LinopyDeprecationWarning is suppressed so that the test
    output stays clean.  Dedicated tests in test_convention.py verify that
    these warnings are actually emitted.
    """
    import linopy
    from linopy.config import LinopyDeprecationWarning

    item = request.node
    if item.get_closest_marker("legacy_only") and request.param != "legacy":
        pytest.skip("legacy-only test")
    if item.get_closest_marker("v1_only") and request.param != "v1":
        pytest.skip("v1-only test")

    old = linopy.options["arithmetic_convention"]
    linopy.options["arithmetic_convention"] = request.param
    if request.param == "legacy":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinopyDeprecationWarning)
            yield request.param
    else:
        yield request.param
    linopy.options["arithmetic_convention"] = old


@pytest.fixture
def m() -> Model:
    from linopy import Model

    m = Model()
    m.add_variables(pd.Series([0, 0]), 1, name="x")
    m.add_variables(4, pd.Series([8, 10]), name="y")
    m.add_variables(0, pd.DataFrame([[1, 2], [3, 4], [5, 6]]).T, name="z")
    m.add_variables(coords=[pd.RangeIndex(20, name="dim_2")], name="v")
    idx = pd.MultiIndex.from_product([[1, 2], ["a", "b"]], names=("level1", "level2"))
    idx.name = "dim_3"
    m.add_variables(coords=[idx], name="u")
    return m


@pytest.fixture
def x(m: Model) -> Variable:
    return m.variables["x"]


@pytest.fixture
def y(m: Model) -> Variable:
    return m.variables["y"]


@pytest.fixture
def z(m: Model) -> Variable:
    return m.variables["z"]


@pytest.fixture
def v(m: Model) -> Variable:
    return m.variables["v"]


@pytest.fixture
def u(m: Model) -> Variable:
    return m.variables["u"]
