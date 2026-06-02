"""Tests for linopy/dual.py - LP dualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model
from linopy.dual import dualize
from linopy.solvers import licensed_solvers

_lp_solver = next((s for s in ("highs", "glpk", "scip") if s in licensed_solvers), None)
needs_solver = pytest.mark.skipif(_lp_solver is None, reason="No LP solver available")


# Structural tests (no solver required)
def test_dualize_empty_model():
    m = Model()
    m_dual = dualize(m)
    assert len(m_dual.variables) == 0
    assert len(m_dual.constraints) == 0


def test_dual_variables_named_after_primal_constraints():
    m = Model()
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(3)], name="x")
    m.add_constraints(x >= 1.0, name="lb_con")
    m.add_objective(2.0 * x)

    m_dual = dualize(m)
    assert "lb_con" in m_dual.variables
    assert "x-bound-lower" in m_dual.variables


def test_dual_feasibility_constraints_named_after_primal_variables():
    m = Model()
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(3)], name="x")
    m.add_constraints(x >= 1.0, name="lb_con")
    m.add_objective(2.0 * x)

    m_dual = dualize(m)
    assert "x" in m_dual.constraints


def test_dual_objective_sense_flipped_min():
    m = Model()
    x = m.add_variables(lower=0, name="x")
    m.add_constraints(x >= 1, name="c")
    m.add_objective(x)  # min

    m_dual = dualize(m)
    assert m_dual.objective.sense == "max"


def test_dual_objective_sense_flipped_max():
    m = Model()
    x = m.add_variables(upper=10, name="x")
    m.add_constraints(x <= 5, name="c")
    m.add_objective(x, sense="max")

    m_dual = dualize(m)
    assert m_dual.objective.sense == "min"


def test_dual_sign_conventions_min():
    """For a min primal: <= -> dual <= 0, >= -> dual >= 0, = -> dual free."""
    m = Model()
    x = m.add_variables(lower=-np.inf, upper=np.inf, name="x")
    m.add_constraints(x == 5, name="eq")
    m.add_constraints(x <= 10, name="leq")
    m.add_constraints(x >= -10, name="geq")
    m.add_objective(x)

    m_dual = dualize(m)
    assert m_dual.variables["eq"].lower.item() == -np.inf
    assert m_dual.variables["eq"].upper.item() == np.inf
    assert m_dual.variables["leq"].upper.item() == 0
    assert m_dual.variables["geq"].lower.item() == 0


def test_dual_sign_conventions_max():
    """For a max primal: <= -> dual >= 0, >= -> dual <= 0."""
    m = Model()
    x = m.add_variables(lower=-np.inf, upper=np.inf, name="x")
    m.add_constraints(x <= 10, name="leq")
    m.add_constraints(x >= -10, name="geq")
    m.add_objective(x, sense="max")

    m_dual = dualize(m)
    assert m_dual.variables["leq"].lower.item() == 0
    assert m_dual.variables["geq"].upper.item() == 0


def test_dual_feasibility_rhs_equals_objective_coefficients():
    """The dual feasibility constraint RHS must equal the primal objective coefficient."""
    m = Model()
    x = m.add_variables(
        lower=-np.inf, upper=np.inf, coords=[pd.RangeIndex(4)], name="x"
    )
    m.add_constraints(x == np.array([1.0, 2.0, 3.0, 4.0]), name="eq")
    c = np.array([10.0, 20.0, 30.0, 40.0])
    m.add_objective(c * x)

    m_dual = dualize(m)
    np.testing.assert_allclose(m_dual.constraints["x"].rhs.values, c)


def test_dual_multi_constraint_per_variable():
    """Each primal variable appearing in k constraints produces k terms in its dual feas constraint."""
    m = Model()
    n = 3
    x = m.add_variables(
        lower=-np.inf, upper=np.inf, coords=[pd.RangeIndex(n)], name="x"
    )
    # Two equality constraints, both using x
    m.add_constraints(x == 1.0, name="c1")
    m.add_constraints(2.0 * x == 2.0, name="c2")
    m.add_objective(5.0 * x)

    m_dual = dualize(m)
    # dual feas constraint for x: lambda_c1 + 2*lambda_c2 = 5
    con_x = m_dual.constraints["x"]
    # Should have 2 terms (one from c1, one from c2)
    n_terms = (con_x.vars != -1).sum(dim="_term").max().item()
    assert n_terms == 2


def test_dual_with_masked_variable():
    """Partially masked variable: active elements produce a constraint, masked ones do not."""
    m = Model()
    mask = xr.DataArray([True, False, True], dims=["dim_0"])
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(3)], name="x", mask=mask)
    m.add_constraints(x >= 1.0, name="c", mask=mask)
    m.add_objective(x)

    m_dual = dualize(m)
    assert "x" in m_dual.constraints
    assert (
        m_dual.constraints["x"].labels != -1
    ).sum().item() == 2  # only 2 active rows


def test_dual_free_unconstrained_variable_no_error():
    """A free variable with no constraint connections is silently skipped."""
    m = Model()
    m.add_variables(
        lower=-np.inf, upper=np.inf, name="x"
    )  # no bounds → no bound constraints
    y = m.add_variables(lower=-np.inf, upper=np.inf, name="y")
    m.add_constraints(y == 5, name="eq")
    m.add_objective(y)  # x has no connections at all

    m_dual = dualize(m)  # must not raise
    assert "y" in m_dual.constraints
    assert "x" not in m_dual.constraints  # no constraint connections → silently skipped


# Numerical tests (require solver)
def _solve(model, **kwargs):
    model.solve(solver_name=_lp_solver, io_api="lp", **kwargs)
    return model.objective.value


@needs_solver
def test_strong_duality_simple():
    """Strong duality: primal obj == dual obj at optimality."""
    m = Model()
    x = m.add_variables(lower=0, name="x")
    y = m.add_variables(lower=0, name="y")
    m.add_constraints(x + y >= 3, name="c1")
    m.add_constraints(x + 2 * y >= 4, name="c2")
    m.add_objective(5 * x + 4 * y)

    primal_obj = _solve(m)
    dual_obj = _solve(m.dualize())
    assert abs(primal_obj - dual_obj) < 1e-5


@needs_solver
def test_strong_duality_array_variable():
    """Strong duality with multi-dimensional variables."""
    rng = np.random.default_rng(0)
    n = 6
    A = rng.random((4, n)) - 0.5
    b = rng.random(4) + 0.5
    c_obj = rng.random(n) + 0.1

    m = Model()
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(n)], name="x")
    for i in range(4):
        m.add_constraints(
            sum(float(A[i, j]) * x[j] for j in range(n)) <= float(b[i]),
            name=f"c{i}",
        )
    m.add_objective(sum(float(c_obj[j]) * x[j] for j in range(n)))

    primal_obj = _solve(m)
    dual_obj = _solve(m.dualize())
    assert abs(primal_obj - dual_obj) < 1e-4


@needs_solver
def test_strong_duality_mixed_constraint_types():
    """Strong duality with =, <=, >= constraints."""
    m = Model()
    x = m.add_variables(lower=-np.inf, upper=np.inf, name="x")
    y = m.add_variables(lower=-np.inf, upper=np.inf, name="y")
    m.add_constraints(x + y == 5, name="eq")
    m.add_constraints(x - y <= 2, name="leq")
    m.add_constraints(x >= 0, name="geq")
    m.add_objective(2 * x + 3 * y)

    primal_obj = _solve(m)
    dual_obj = _solve(m.dualize())
    assert abs(primal_obj - dual_obj) < 1e-5


@needs_solver
def test_strong_duality_maximization():
    """Strong duality for a maximization primal."""
    m = Model()
    x = m.add_variables(lower=0, name="x")
    y = m.add_variables(lower=0, name="y")
    m.add_constraints(x + y <= 10, name="c1")
    m.add_constraints(x <= 6, name="c2")
    m.add_constraints(y <= 8, name="c3")
    m.add_objective(3 * x + 5 * y, sense="max")

    primal_obj = _solve(m)
    dual_obj = _solve(m.dualize())
    assert abs(primal_obj - dual_obj) < 1e-5
