"""Pytest configuration and fixtures."""

import os

import pytest


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
