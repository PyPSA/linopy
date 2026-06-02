"""Tests for linopy/dual.py - LP dualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model
from linopy.dual import (
    _build_label_to_flat_index_lookup,
    _lookup_flat_indices,
    _term_slots_for_sorted_flat_indices,
    dualize,
)
from linopy.solvers import licensed_solvers

_lp_solver = next((s for s in ("highs", "glpk", "scip") if s in licensed_solvers), None)
needs_solver = pytest.mark.skipif(_lp_solver is None, reason="No LP solver available")


# Structural tests for important internal functions
def test_build_label_to_flat_index_lookup():
    """Flat labels are mapped to their positions in the flattened label array."""
    labels = np.array([10, -1, 12, 99], dtype=np.int64)

    lookup = _build_label_to_flat_index_lookup(labels)

    assert lookup[10] == 0
    assert lookup[12] == 2
    assert lookup[99] == 3
    assert lookup[11] == -1


def test_build_label_to_flat_index_lookup_all_masked():
    """An all-masked label array produces an empty lookup."""
    labels = np.array([-1, -1], dtype=np.int64)

    lookup = _build_label_to_flat_index_lookup(labels)

    assert lookup.dtype == np.int64
    assert len(lookup) == 0


def test_lookup_flat_indices_bounds_safe():
    """Out-of-range and masked labels safely map to -1."""
    lookup = np.array([-1, -1, 5, -1, 7], dtype=np.int64)
    labels = np.array([-1, 0, 2, 4, 99], dtype=np.int64)

    flat_indices = _lookup_flat_indices(labels, lookup)

    np.testing.assert_array_equal(
        flat_indices,
        np.array([-1, -1, 5, 7, -1], dtype=np.int64),
    )


def test_term_slots_for_sorted_flat_indices():
    """Repeated sorted flat indices are assigned increasing term slots."""
    sorted_flat_indices = np.array([2, 2, 2, 5, 5, 9], dtype=np.int64)

    slots = _term_slots_for_sorted_flat_indices(sorted_flat_indices)

    np.testing.assert_array_equal(
        slots,
        np.array([0, 1, 2, 0, 1, 0], dtype=np.int64),
    )


# Structural tests (no solver required)
def test_dualize_empty_model():
    """Dualizing an empty model returns an empty dual model."""
    m = Model()
    m_dual = dualize(m)
    assert len(m_dual.variables) == 0
    assert len(m_dual.constraints) == 0


def test_variable_bounds_lifted_to_dual_variables():
    """Finite variable bounds are converted into dual variables."""
    m = Model()
    x = m.add_variables(lower=1, upper=3, name="x")
    m.add_objective(x)

    m_dual = dualize(m)

    assert "x-bound-lower" in m_dual.variables
    assert "x-bound-upper" in m_dual.variables


def test_dual_variables_named_after_primal_constraints():
    """Dual variables use the names of their corresponding primal constraints."""
    m = Model()
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(3)], name="x")
    m.add_constraints(x >= 1.0, name="lb_con")
    m.add_objective(2.0 * x)

    m_dual = dualize(m)
    assert "lb_con" in m_dual.variables
    assert "x-bound-lower" in m_dual.variables


def test_dual_feasibility_constraints_named_after_primal_variables():
    """Dual-feasibility constraints use the names of the primal variables."""
    m = Model()
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(3)], name="x")
    m.add_constraints(x >= 1.0, name="lb_con")
    m.add_objective(2.0 * x)

    m_dual = dualize(m)
    assert "x" in m_dual.constraints


def test_dual_objective_sense_flipped_min():
    """A minimization primal produces a maximization dual."""
    m = Model()
    x = m.add_variables(lower=0, name="x")
    m.add_constraints(x >= 1, name="c")
    m.add_objective(x)  # min

    m_dual = dualize(m)
    assert m_dual.objective.sense == "max"


def test_dual_objective_sense_flipped_max():
    """A maximization primal produces a minimization dual."""
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
    """The dual feasibility RHS equals the primal objective coefficient."""
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
    """A variable appearing in k constraints gets k dual-feasibility terms."""
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
    """Partially masked variables only produce constraints for unmasked elements."""
    m = Model()
    mask = xr.DataArray([True, False, True], dims=["dim_0"])
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(3)], name="x", mask=mask)
    m.add_constraints(x >= 1.0, name="c", mask=mask)
    m.add_objective(x)

    m_dual = dualize(m)
    assert "x" in m_dual.constraints
    assert (
        m_dual.constraints["x"].labels != -1
    ).sum().item() == 2  # only 2 unmasked elements


def test_dual_free_unconstrained_zero_cost_variable_no_error():
    """A disconnected zero-cost free variable is skipped."""
    m = Model()
    m.add_variables(
        lower=-np.inf, upper=np.inf, name="x"
    )  # no bounds → no bound constraints
    y = m.add_variables(lower=-np.inf, upper=np.inf, name="y")
    m.add_constraints(y == 5, name="eq")
    m.add_objective(y)  # x has no connections at all

    m_dual = dualize(m)  # must not raise
    assert "y" in m_dual.constraints
    assert "x" not in m_dual.constraints  # no constraint connections


def test_dual_free_unconstrained_variable_with_objective_warns(caplog):
    """A disconnected variable with nonzero objective coefficient is reported."""
    m = Model()
    x = m.add_variables(lower=-np.inf, upper=np.inf, name="x")
    y = m.add_variables(lower=-np.inf, upper=np.inf, name="y")

    m.add_constraints(y == 5, name="eq")
    m.add_objective(x + y)

    with caplog.at_level("WARNING"):
        m_dual = dualize(m)

    assert "y" in m_dual.constraints
    assert "x" not in m_dual.constraints
    assert any(
        "corresponding dual-feasibility condition is infeasible" in rec.message
        for rec in caplog.records
    )


def test_dual_multi_constraint_per_variable_coefficients():
    """Dual-feasibility terms keep the correct A-matrix coefficients."""
    m = Model()
    x = m.add_variables(lower=-np.inf, upper=np.inf, name="x")

    m.add_constraints(x == 1.0, name="c1")
    m.add_constraints(2.0 * x == 2.0, name="c2")
    m.add_objective(5.0 * x)

    m_dual = dualize(m)
    coeffs = m_dual.constraints["x"].coeffs.values.ravel()
    coeffs = coeffs[m_dual.constraints["x"].vars.values.ravel() != -1]

    np.testing.assert_allclose(np.sort(coeffs), np.array([1.0, 2.0]))


# Numerical tests (require solver)
def _solve(model, **kwargs):
    """Solve a model with the available LP solver and return its objective value."""
    assert _lp_solver is not None
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
    """Strong duality with array variables."""
    rng = np.random.default_rng(0)
    n = 6

    A = rng.random((4, n)) + 0.1
    b = rng.random(4) + 1.0
    c_obj = rng.random(n) + 0.1

    m = Model()
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(n)], name="x")

    for i in range(4):
        m.add_constraints(
            sum(float(A[i, j]) * x[j] for j in range(n)) <= float(b[i]),
            name=f"c{i}",
        )

    # Bounded because x >= 0 and A x <= b with A > 0.
    m.add_objective(sum(float(c_obj[j]) * x[j] for j in range(n)), sense="max")

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
