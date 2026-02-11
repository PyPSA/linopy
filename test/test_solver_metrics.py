#!/usr/bin/env python3
"""
Tests for the SolverMetrics feature.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from linopy import Model, available_solvers
from linopy.constants import Result, Solution, SolverMetrics, Status
from linopy.solver_capabilities import SolverFeature, get_available_solvers_with_feature

# ---------------------------------------------------------------------------
# SolverMetrics dataclass tests
# ---------------------------------------------------------------------------


def test_solver_metrics_defaults() -> None:
    m = SolverMetrics()
    assert m.solver_name is None
    assert m.solve_time is None
    assert m.objective_value is None
    assert m.best_bound is None
    assert m.mip_gap is None


def test_solver_metrics_partial() -> None:
    m = SolverMetrics(solver_name="highs", solve_time=1.5)
    assert m.solver_name == "highs"
    assert m.solve_time == 1.5
    assert m.objective_value is None


def test_solver_metrics_repr_only_non_none() -> None:
    m = SolverMetrics(solver_name="gurobi", solve_time=2.3)
    r = repr(m)
    assert "solver_name='gurobi'" in r
    assert "solve_time=2.3" in r
    assert "objective_value" not in r
    assert "best_bound" not in r


def test_solver_metrics_repr_empty() -> None:
    m = SolverMetrics()
    assert repr(m) == "SolverMetrics()"


# ---------------------------------------------------------------------------
# Result backward compatibility tests
# ---------------------------------------------------------------------------


def test_result_without_metrics() -> None:
    """Result without metrics should still work (backward compatible)."""
    status = Status.from_termination_condition("optimal")
    result = Result(status=status, solution=Solution())
    assert result.metrics is None
    # repr should not crash
    repr(result)


def test_result_with_metrics() -> None:
    status = Status.from_termination_condition("optimal")
    metrics = SolverMetrics(solver_name="test", solve_time=1.0)
    result = Result(status=status, solution=Solution(), metrics=metrics)
    assert result.metrics is not None
    assert result.metrics.solver_name == "test"
    r = repr(result)
    assert "Solver metrics:" in r


# ---------------------------------------------------------------------------
# Model integration tests
# ---------------------------------------------------------------------------


def test_model_metrics_none_before_solve() -> None:
    m = Model()
    assert m.solver_metrics is None


def test_model_metrics_populated_after_mock_solve() -> None:
    m = Model()
    x = m.add_variables(
        lower=xr.DataArray(np.zeros(5), dims=["i"]),
        upper=xr.DataArray(np.ones(5), dims=["i"]),
        name="x",
    )
    m.add_objective(x.sum())
    m.solve(mock_solve=True)
    assert m.solver_metrics is not None
    assert m.solver_metrics.solver_name == "mock"
    assert m.solver_metrics.objective_value == 0.0


def test_model_metrics_reset() -> None:
    m = Model()
    x = m.add_variables(
        lower=xr.DataArray(np.zeros(5), dims=["i"]),
        upper=xr.DataArray(np.ones(5), dims=["i"]),
        name="x",
    )
    m.add_objective(x.sum())
    m.solve(mock_solve=True)
    assert m.solver_metrics is not None
    m.reset_solution()
    assert m.solver_metrics is None


# ---------------------------------------------------------------------------
# Solver-specific integration tests (parametrized over available solvers)
# ---------------------------------------------------------------------------

direct_solvers = get_available_solvers_with_feature(
    SolverFeature.DIRECT_API, available_solvers
)
file_io_solvers = get_available_solvers_with_feature(
    SolverFeature.READ_MODEL_FROM_FILE, available_solvers
)


def _make_simple_model() -> Model:
    m = Model()
    x = m.add_variables(
        lower=xr.DataArray(np.zeros(3), dims=["i"]),
        upper=xr.DataArray(np.ones(3), dims=["i"]),
        name="x",
    )
    m.add_constraints(x.sum() >= 1, name="con")
    m.add_objective(x.sum())
    return m


@pytest.mark.parametrize("solver", direct_solvers)
def test_solver_metrics_direct(solver: str) -> None:
    m = _make_simple_model()
    m.solve(solver_name=solver, io_api="direct")
    metrics = m.solver_metrics
    assert metrics is not None
    assert metrics.solver_name == solver
    assert metrics.objective_value is not None
    assert metrics.objective_value == pytest.approx(1.0)
    # Direct API solvers should generally report solve_time
    if solver in ("gurobi", "highs"):
        assert metrics.solve_time is not None
        assert metrics.solve_time >= 0


@pytest.mark.parametrize("solver", file_io_solvers)
def test_solver_metrics_file_io(solver: str) -> None:
    m = _make_simple_model()
    m.solve(solver_name=solver, io_api="lp")
    metrics = m.solver_metrics
    assert metrics is not None
    assert metrics.solver_name == solver
    assert metrics.objective_value is not None
    assert metrics.objective_value == pytest.approx(1.0)
