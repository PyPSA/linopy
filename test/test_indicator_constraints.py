"""Tests for indicator constraint support."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from linopy import Model, available_solvers


def test_add_indicator_constraints_basic() -> None:
    """Indicator constraint is created with correct fields."""
    m = Model()
    b = m.add_variables(name="b", binary=True)
    x = m.add_variables(lower=0, upper=10, name="x")
    m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")

    assert "ic0" in m.indicator_constraints
    ic = m.indicator_constraints["ic0"]
    assert "coeffs" in ic
    assert "vars" in ic
    assert "sign" in ic
    assert "rhs" in ic
    assert "binary_var" in ic
    assert "binary_val" in ic
    assert "labels" in ic


def test_add_indicator_constraints_from_constraint() -> None:
    """Indicator constraint accepts a Constraint object as lhs."""
    m = Model()
    b = m.add_variables(name="b", binary=True)
    x = m.add_variables(lower=0, upper=10, name="x")
    con = x <= 5
    m.add_indicator_constraints(b, 1, con, name="ic0")
    assert "ic0" in m.indicator_constraints


def test_indicator_constraints_validation_non_binary() -> None:
    """Non-binary variable raises ValueError."""
    m = Model()
    y = m.add_variables(lower=0, upper=1, name="y")
    x = m.add_variables(lower=0, upper=10, name="x")
    with pytest.raises(ValueError, match="must be binary"):
        m.add_indicator_constraints(y, 1, x, "<=", 5)


def test_indicator_constraints_validation_bad_value() -> None:
    """Invalid binary_val raises ValueError."""
    m = Model()
    b = m.add_variables(name="b", binary=True)
    x = m.add_variables(lower=0, upper=10, name="x")
    with pytest.raises(ValueError, match="must be 0 or 1"):
        m.add_indicator_constraints(b, 2, x, "<=", 5)


def test_indicator_constraints_duplicate_name() -> None:
    """Duplicate name raises ValueError."""
    m = Model()
    b = m.add_variables(name="b", binary=True)
    x = m.add_variables(lower=0, upper=10, name="x")
    m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")
    with pytest.raises(ValueError, match="already assigned"):
        m.add_indicator_constraints(b, 0, x, ">=", 0, name="ic0")


def test_indicator_constraints_not_in_regular_constraints() -> None:
    """Indicator constraints are separate from regular constraints."""
    m = Model()
    b = m.add_variables(name="b", binary=True)
    x = m.add_variables(lower=0, upper=10, name="x")
    m.add_constraints(x >= 0, name="regular")
    m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")

    assert "regular" in m.constraints
    assert "ic0" not in m.constraints
    assert "ic0" in m.indicator_constraints


def test_remove_indicator_constraints() -> None:
    """Indicator constraints can be removed."""
    m = Model()
    b = m.add_variables(name="b", binary=True)
    x = m.add_variables(lower=0, upper=10, name="x")
    m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")
    assert "ic0" in m.indicator_constraints
    m.remove_indicator_constraints("ic0")
    assert "ic0" not in m.indicator_constraints


def test_indicator_constraints_lp_file(tmp_path: Path) -> None:
    """LP file contains general constraints section."""
    m = Model()
    b = m.add_variables(name="b", binary=True)
    x = m.add_variables(lower=0, upper=10, name="x")
    m.add_constraints(x >= 0, name="dummy")
    m.add_objective(x)
    m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")

    fn = tmp_path / "test.lp"
    m.to_file(fn)
    content = fn.read_text()
    assert "= 1 ->" in content
    assert "ic0:" in content


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobi not installed")
def test_indicator_constraints_solve_gurobi_active() -> None:
    """
    Indicator constraint enforced when binary is at trigger value.

    Maximize x subject to:
        b = 1 (fixed)
        (b == 1) => (x <= 5)
        x in [0, 10]
    Since b=1 and the indicator enforces x<=5, optimal x=5.
    """
    m = Model()
    b = m.add_variables(name="b", binary=True)
    x = m.add_variables(lower=0, upper=10, name="x")
    # Force b = 1
    m.add_constraints(b >= 1, name="fix_b")
    m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")
    m.add_objective(x, sense="max")
    m.solve(solver_name="gurobi")
    assert m.objective.value is not None
    assert np.isclose(m.objective.value, 5, atol=1e-6)


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobi not installed")
def test_indicator_constraints_solve_gurobi_inactive() -> None:
    """
    Indicator constraint NOT enforced when binary differs from trigger.

    Maximize x subject to:
        b = 1 (fixed)
        (b == 0) => (x <= 5)
        x in [0, 10]
    Since b=1 but trigger is 0, the constraint is inactive. Optimal x=10.
    """
    m = Model()
    b = m.add_variables(name="b", binary=True)
    x = m.add_variables(lower=0, upper=10, name="x")
    # Force b = 1
    m.add_constraints(b >= 1, name="fix_b")
    m.add_indicator_constraints(b, 0, x, "<=", 5, name="ic0")
    m.add_objective(x, sense="max")
    m.solve(solver_name="gurobi")
    assert m.objective.value is not None
    assert np.isclose(m.objective.value, 10, atol=1e-6)


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobi not installed")
def test_indicator_constraints_solve_gurobi_multiple() -> None:
    """
    Multiple indicator constraints work together.

    b1=1, b2=1 (forced), x in [0, 20].
    (b1 == 1) => (x <= 10)
    (b2 == 1) => (x <= 5)
    Maximize x.
    Both constraints active, so x <= min(10, 5) = 5. Optimal x=5.
    """
    m = Model()
    b1 = m.add_variables(name="b1", binary=True)
    b2 = m.add_variables(name="b2", binary=True)
    x = m.add_variables(lower=0, upper=20, name="x")
    m.add_constraints(b1 >= 1, name="fix_b1")
    m.add_constraints(b2 >= 1, name="fix_b2")
    m.add_indicator_constraints(b1, 1, x, "<=", 10, name="ic1")
    m.add_indicator_constraints(b2, 1, x, "<=", 5, name="ic2")
    m.add_objective(x, sense="max")
    m.solve(solver_name="gurobi")
    assert m.objective.value is not None
    assert np.isclose(m.objective.value, 5, atol=1e-6)


def test_unsupported_solver_raises() -> None:
    """Solvers without indicator support raise ValueError."""
    m = Model()
    b = m.add_variables(name="b", binary=True)
    x = m.add_variables(lower=0, upper=10, name="x")
    m.add_constraints(x >= 0, name="dummy")
    m.add_objective(x)
    m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")

    for solver in ["glpk", "highs", "mosek", "mindopt"]:
        if solver in available_solvers:
            with pytest.raises(
                ValueError, match="does not support indicator constraints"
            ):
                m.solve(solver_name=solver)


def test_indicator_constraints_with_coords() -> None:
    """Indicator constraints work with multi-dimensional coords."""
    m = Model()
    idx = pd.RangeIndex(3, name="i")
    b = m.add_variables(coords=[idx], name="b", binary=True)
    x = m.add_variables(coords=[idx], lower=0, upper=10, name="x")
    m.add_indicator_constraints(b, 1, x, "<=", 5, name="ic0")
    ic = m.indicator_constraints["ic0"]
    # Should have 3 indicator constraints (one per index)
    assert ic.labels.size == 3
