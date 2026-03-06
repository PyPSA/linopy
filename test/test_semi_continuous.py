"""Tests for semi-continuous variable support."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from linopy import Model, available_solvers


def test_add_semi_continuous_variable() -> None:
    """Semi-continuous variable is created with correct attributes."""
    m = Model()
    x = m.add_variables(lower=1, upper=10, name="x", semi_continuous=True)
    assert x.attrs["semi_continuous"] is True
    assert not x.attrs["binary"]
    assert not x.attrs["integer"]


def test_semi_continuous_mutual_exclusivity() -> None:
    """Semi-continuous cannot be combined with binary or integer."""
    m = Model()
    with pytest.raises(ValueError, match="only be one of"):
        m.add_variables(lower=1, upper=10, binary=True, semi_continuous=True)
    with pytest.raises(ValueError, match="only be one of"):
        m.add_variables(lower=1, upper=10, integer=True, semi_continuous=True)


def test_semi_continuous_requires_positive_lb() -> None:
    """Semi-continuous variables require a positive lower bound."""
    m = Model()
    with pytest.raises(ValueError, match="positive scalar lower bound"):
        m.add_variables(lower=-1, upper=10, semi_continuous=True)
    with pytest.raises(ValueError, match="positive scalar lower bound"):
        m.add_variables(lower=0, upper=10, semi_continuous=True)


def test_semi_continuous_collection_property() -> None:
    """Variables.semi_continuous filters correctly."""
    m = Model()
    m.add_variables(lower=1, upper=10, name="x", semi_continuous=True)
    m.add_variables(lower=0, upper=5, name="y")
    m.add_variables(name="z", binary=True)

    assert list(m.variables.semi_continuous) == ["x"]
    assert "x" not in m.variables.continuous
    assert "y" in m.variables.continuous
    assert "z" not in m.variables.continuous


def test_semi_continuous_repr() -> None:
    """Semi-continuous annotation appears in repr."""
    m = Model()
    m.add_variables(lower=1, upper=10, name="x", semi_continuous=True)
    r = repr(m.variables)
    assert "semi-continuous" in r


def test_semi_continuous_vtypes() -> None:
    """Matrices vtypes returns 'S' for semi-continuous variables."""
    m = Model()
    m.add_variables(lower=1, upper=10, name="x", semi_continuous=True)
    m.add_variables(lower=0, upper=5, name="y")
    m.add_variables(name="z", binary=True)
    # Add a dummy constraint and objective so the model is valid
    m.add_constraints(m.variables["y"] >= 0, name="dummy")
    m.add_objective(m.variables["y"])

    vtypes = m.matrices.vtypes
    # x is semi-continuous -> "S", y is continuous -> "C", z is binary -> "B"
    assert "S" in vtypes
    assert "C" in vtypes
    assert "B" in vtypes


def test_semi_continuous_lp_file(tmp_path: Path) -> None:
    """LP file contains semi-continuous section."""
    m = Model()
    m.add_variables(lower=1, upper=10, name="x", semi_continuous=True)
    m.add_variables(lower=0, upper=5, name="y")
    m.add_constraints(m.variables["y"] >= 0, name="dummy")
    m.add_objective(m.variables["y"])

    fn = tmp_path / "test.lp"
    m.to_file(fn)
    content = fn.read_text()
    assert "semi-continuous" in content


def test_semi_continuous_with_coords() -> None:
    """Semi-continuous variables work with multi-dimensional coords."""
    m = Model()
    idx = pd.RangeIndex(5, name="i")
    x = m.add_variables(lower=2, upper=20, coords=[idx], name="x", semi_continuous=True)
    assert x.attrs["semi_continuous"] is True
    assert list(m.variables.semi_continuous) == ["x"]


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobi not installed")
def test_semi_continuous_solve_gurobi() -> None:
    """
    Semi-continuous variable solves correctly with Gurobi.

    Maximize x subject to x <= 0.5, x semi-continuous in [1, 10].
    Since x can be 0 or in [1, 10], and x <= 0.5 prevents [1, 10],
    the optimal x should be 0.
    """
    m = Model()
    x = m.add_variables(lower=1, upper=10, name="x", semi_continuous=True)
    m.add_constraints(x <= 0.5, name="ub")
    m.add_objective(x, sense="max")
    m.solve(solver_name="gurobi")
    assert m.objective.value is not None
    assert np.isclose(m.objective.value, 0, atol=1e-6)


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobi not installed")
def test_semi_continuous_solve_gurobi_active() -> None:
    """
    Semi-continuous variable takes value in [lb, ub] when beneficial.

    Maximize x subject to x <= 5, x semi-continuous in [1, 10].
    Optimal x should be 5.
    """
    m = Model()
    x = m.add_variables(lower=1, upper=10, name="x", semi_continuous=True)
    m.add_constraints(x <= 5, name="ub")
    m.add_objective(x, sense="max")
    m.solve(solver_name="gurobi")
    assert m.objective.value is not None
    assert np.isclose(m.objective.value, 5, atol=1e-6)


def test_unsupported_solver_raises() -> None:
    """Solvers without semi-continuous support raise ValueError."""
    m = Model()
    m.add_variables(lower=1, upper=10, name="x", semi_continuous=True)
    m.add_constraints(m.variables["x"] <= 5, name="ub")
    m.add_objective(m.variables["x"])

    for solver in ["glpk", "highs", "mosek", "mindopt"]:
        if solver in available_solvers:
            with pytest.raises(ValueError, match="does not support semi-continuous"):
                m.solve(solver_name=solver)
