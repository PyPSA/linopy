"""Tests for SOS constraint reformulation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from linopy import Model, available_solvers
from linopy.constants import SOS_TYPE_ATTR
from linopy.sos_reformulation import (
    compute_big_m_values,
    reformulate_sos1,
    reformulate_sos2,
    reformulate_sos_constraints,
    undo_sos_reformulation,
)


class TestValidateBounds:
    """Tests for bound validation in compute_big_m_values."""

    def test_finite_bounds_pass(self) -> None:
        """Finite non-negative bounds should pass validation."""
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        compute_big_m_values(x)  # Should not raise

    def test_infinite_upper_bounds_raise(self) -> None:
        """Infinite upper bounds should raise ValueError."""
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=np.inf, coords=[idx], name="x")
        with pytest.raises(ValueError, match="infinite upper bounds"):
            compute_big_m_values(x)

    def test_negative_lower_bounds_raise(self) -> None:
        """Negative lower bounds should raise ValueError."""
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=-1, upper=1, coords=[idx], name="x")
        with pytest.raises(ValueError, match="negative lower bounds"):
            compute_big_m_values(x)

    def test_mixed_negative_lower_bounds_raise(self) -> None:
        """Mixed finite/negative lower bounds should raise ValueError."""
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(
            lower=np.array([0, -1, 0]),
            upper=np.array([1, 1, 1]),
            coords=[idx],
            name="x",
        )
        with pytest.raises(ValueError, match="negative lower bounds"):
            compute_big_m_values(x)


class TestComputeBigM:
    """Tests for compute_big_m_values."""

    def test_positive_bounds(self) -> None:
        """Test Big-M computation with positive bounds."""
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=10, coords=[idx], name="x")
        M = compute_big_m_values(x)
        assert np.allclose(M.values, [10, 10, 10])

    def test_varying_bounds(self) -> None:
        """Test Big-M computation with varying upper bounds."""
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(
            lower=np.array([0, 0, 0]),
            upper=np.array([1, 2, 3]),
            coords=[idx],
            name="x",
        )
        M = compute_big_m_values(x)
        assert np.allclose(M.values, [1, 2, 3])

    def test_custom_big_m_scalar(self) -> None:
        """Test Big-M uses tighter of custom value and bounds."""
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=100, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i", big_m=10)
        M = compute_big_m_values(x)
        # M = min(10, 100) = 10 (custom is tighter)
        assert np.allclose(M.values, [10, 10, 10])

    def test_custom_big_m_allows_infinite_bounds(self) -> None:
        """Test that custom big_m allows variables with infinite bounds."""
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=np.inf, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i", big_m=10)
        # Should not raise - custom big_m makes result finite
        M = compute_big_m_values(x)
        assert np.allclose(M.values, [10, 10, 10])


class TestSOS1Reformulation:
    """Tests for SOS1 reformulation."""

    def test_basic_sos1(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")

        reformulate_sos1(m, x, "_test_")
        m.remove_sos_constraints(x)

        # Check auxiliary variables and constraints were added
        assert "_test_x_y" in m.variables
        assert "_test_x_upper" in m.constraints
        assert "_test_x_card" in m.constraints

        # Binary variable should have same dimensions
        y = m.variables["_test_x_y"]
        assert y.dims == x.dims
        assert y.sizes == x.sizes

    def test_sos1_multidimensional(self) -> None:
        m = Model()
        idx_i = pd.Index([0, 1, 2], name="i")
        idx_j = pd.Index([0, 1], name="j")
        x = m.add_variables(lower=0, upper=1, coords=[idx_i, idx_j], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")

        reformulate_sos1(m, x, "_test_")
        m.remove_sos_constraints(x)

        # Binary variable should have same dimensions
        y = m.variables["_test_x_y"]
        assert set(y.dims) == {"i", "j"}

        # Cardinality constraint should have reduced dimensions (summed over i)
        card_con = m.constraints["_test_x_card"]
        assert "j" in card_con.dims


class TestSOS2Reformulation:
    """Tests for SOS2 reformulation."""

    def test_basic_sos2(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=2, sos_dim="i")

        reformulate_sos2(m, x, "_test_")
        m.remove_sos_constraints(x)

        # Check auxiliary variables and constraints were added
        assert "_test_x_z" in m.variables
        assert "_test_x_upper_first" in m.constraints
        assert "_test_x_upper_last" in m.constraints
        assert "_test_x_card" in m.constraints

        # Segment indicators should have n-1 elements
        z = m.variables["_test_x_z"]
        assert z.sizes["i"] == 2  # n-1 = 3-1 = 2

    def test_sos2_trivial_single_element(self) -> None:
        m = Model()
        idx = pd.Index([0], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=2, sos_dim="i")

        reformulate_sos2(m, x, "_test_")

        assert "_test_x_z" not in m.variables

    def test_sos2_two_elements(self) -> None:
        m = Model()
        idx = pd.Index([0, 1], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=2, sos_dim="i")

        reformulate_sos2(m, x, "_test_")
        m.remove_sos_constraints(x)

        # Should have 1 segment indicator
        z = m.variables["_test_x_z"]
        assert z.sizes["i"] == 1

    def test_sos2_with_middle_constraints(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2, 3], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=2, sos_dim="i")

        reformulate_sos2(m, x, "_test_")
        m.remove_sos_constraints(x)

        assert "_test_x_upper_first" in m.constraints
        assert "_test_x_upper_mid" in m.constraints
        assert "_test_x_upper_last" in m.constraints

    def test_sos2_multidimensional(self) -> None:
        m = Model()
        idx_i = pd.Index([0, 1, 2], name="i")
        idx_j = pd.Index([0, 1], name="j")
        x = m.add_variables(lower=0, upper=1, coords=[idx_i, idx_j], name="x")
        m.add_sos_constraints(x, sos_type=2, sos_dim="i")

        reformulate_sos2(m, x, "_test_")
        m.remove_sos_constraints(x)

        # Segment indicator should have (n-1) elements in i dimension, same j dimension
        z = m.variables["_test_x_z"]
        assert set(z.dims) == {"i", "j"}
        assert z.sizes["i"] == 2  # n-1 = 3-1 = 2
        assert z.sizes["j"] == 2

        # Cardinality constraint should have j dimension preserved
        card_con = m.constraints["_test_x_card"]
        assert "j" in card_con.dims


class TestReformulateAllSOS:
    """Tests for reformulate_all_sos."""

    def test_reformulate_single_sos1(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")

        result = reformulate_sos_constraints(m)

        assert result.reformulated == ["x"]
        assert len(list(m.variables.sos)) == 0

    def test_reformulate_multiple_sos(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        y = m.add_variables(lower=0, upper=2, coords=[idx], name="y")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")
        m.add_sos_constraints(y, sos_type=2, sos_dim="i")

        result = reformulate_sos_constraints(m)

        assert set(result.reformulated) == {"x", "y"}
        assert len(list(m.variables.sos)) == 0

    def test_reformulate_removes_sos_attrs_for_single_element(self) -> None:
        m = Model()
        idx = pd.Index([0], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")

        result = reformulate_sos_constraints(m)

        assert result.reformulated == ["x"]
        assert len(list(m.variables.sos)) == 0
        assert len(result.added_variables) == 0
        assert len(result.added_constraints) == 0

    def test_reformulate_removes_sos_attrs_for_zero_bounds(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=0, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")

        result = reformulate_sos_constraints(m)

        assert result.reformulated == ["x"]
        assert len(list(m.variables.sos)) == 0
        assert len(result.added_variables) == 0
        assert len(result.added_constraints) == 0

    def test_reformulate_raises_on_infinite_bounds(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=np.inf, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")

        with pytest.raises(ValueError, match="infinite"):
            reformulate_sos_constraints(m)

    def test_reformulate_raises_on_negative_lower_bounds(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=-1, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")

        with pytest.raises(ValueError, match="negative lower bounds"):
            reformulate_sos_constraints(m)


class TestModelReformulateSOS:
    """Tests for Model.reformulate_sos_constraints method."""

    def test_reformulate_inplace(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")

        result = m.reformulate_sos_constraints()

        assert result.reformulated == ["x"]
        assert len(list(m.variables.sos)) == 0
        assert "_sos_reform_x_y" in m.variables


@pytest.mark.skipif("highs" not in available_solvers, reason="HiGHS not installed")
class TestSolveWithReformulation:
    """Tests for solving with SOS reformulation."""

    def test_sos1_maximize_with_highs(self) -> None:
        """Test SOS1 maximize problem with HiGHS using reformulation."""
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")
        m.add_objective(x * np.array([1, 2, 3]), sense="max")

        m.solve(solver_name="highs", reformulate_sos=True)

        # Should maximize by choosing x[2] = 1
        assert np.isclose(x.solution.values[2], 1, atol=1e-5)
        assert np.isclose(x.solution.values[0], 0, atol=1e-5)
        assert np.isclose(x.solution.values[1], 0, atol=1e-5)
        assert m.objective.value is not None
        assert np.isclose(m.objective.value, 3, atol=1e-5)

    def test_sos1_minimize_with_highs(self) -> None:
        """Test SOS1 minimize problem with HiGHS using reformulation."""
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")
        m.add_objective(x * np.array([3, 2, 1]), sense="min")

        m.solve(solver_name="highs", reformulate_sos=True)

        # Should minimize to 0 by setting all x = 0
        assert m.objective.value is not None
        assert np.isclose(m.objective.value, 0, atol=1e-5)

    def test_sos2_maximize_with_highs(self) -> None:
        """Test SOS2 maximize problem with HiGHS using reformulation."""
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=2, sos_dim="i")
        m.add_objective(x * np.array([1, 2, 3]), sense="max")

        m.solve(solver_name="highs", reformulate_sos=True)

        # SOS2 allows two adjacent non-zeros, so x[1] and x[2] can both be 1
        # Maximum is 2 + 3 = 5
        assert m.objective.value is not None
        assert np.isclose(m.objective.value, 5, atol=1e-5)
        # Check that at most two adjacent variables are non-zero
        nonzero_count = (np.abs(x.solution.values) > 1e-5).sum()
        assert nonzero_count <= 2

    def test_sos2_different_coefficients(self) -> None:
        """Test SOS2 with different coefficients."""
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=2, sos_dim="i")
        m.add_objective(x * np.array([2, 1, 3]), sense="max")

        m.solve(solver_name="highs", reformulate_sos=True)

        # Best is x[1]=1 and x[2]=1 giving 1+3=4
        # or x[0]=1 and x[1]=1 giving 2+1=3
        assert m.objective.value is not None
        assert np.isclose(m.objective.value, 4, atol=1e-5)

    def test_reformulate_sos_false_raises_error(self) -> None:
        """Test that HiGHS without reformulate_sos raises error."""
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")
        m.add_objective(x.sum(), sense="max")

        with pytest.raises(ValueError, match="does not support SOS"):
            m.solve(solver_name="highs", reformulate_sos=False)

    def test_multidimensional_sos1_with_highs(self) -> None:
        """Test multi-dimensional SOS1 with HiGHS."""
        m = Model()
        idx_i = pd.Index([0, 1, 2], name="i")
        idx_j = pd.Index([0, 1], name="j")
        x = m.add_variables(lower=0, upper=1, coords=[idx_i, idx_j], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")
        m.add_objective(x.sum(), sense="max")

        m.solve(solver_name="highs", reformulate_sos=True)

        # For each j, at most one x[i, j] can be non-zero
        # Maximum is achieved by one non-zero per j column: 2 total
        assert m.objective.value is not None
        assert np.isclose(m.objective.value, 2, atol=1e-5)

        # Check SOS1 is satisfied for each j
        for j in idx_j:
            nonzero_count = (np.abs(x.solution.sel(j=j).values) > 1e-5).sum()
            assert nonzero_count <= 1

    def test_multidimensional_sos2_with_highs(self) -> None:
        """Test multi-dimensional SOS2 with HiGHS."""
        m = Model()
        idx_i = pd.Index([0, 1, 2], name="i")
        idx_j = pd.Index([0, 1], name="j")
        x = m.add_variables(lower=0, upper=1, coords=[idx_i, idx_j], name="x")
        m.add_sos_constraints(x, sos_type=2, sos_dim="i")
        m.add_objective(x.sum(), sense="max")

        m.solve(solver_name="highs", reformulate_sos=True)

        # For each j, at most two adjacent x[i, j] can be non-zero
        # Maximum is achieved by two adjacent non-zeros per j column: 4 total
        assert m.objective.value is not None
        assert np.isclose(m.objective.value, 4, atol=1e-5)

        # Check SOS2 is satisfied for each j
        for j in idx_j:
            sol_j = x.solution.sel(j=j).values
            nonzero_indices = np.where(np.abs(sol_j) > 1e-5)[0]
            # At most 2 non-zeros
            assert len(nonzero_indices) <= 2
            # If 2 non-zeros, they must be adjacent
            if len(nonzero_indices) == 2:
                assert abs(nonzero_indices[1] - nonzero_indices[0]) == 1


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobi not installed")
class TestEquivalenceWithGurobi:
    """Tests comparing reformulated solutions with native Gurobi SOS."""

    def test_sos1_equivalence(self) -> None:
        """Test that reformulated SOS1 gives same result as native Gurobi."""
        gurobipy = pytest.importorskip("gurobipy")

        # Native Gurobi solution
        m1 = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x1 = m1.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m1.add_sos_constraints(x1, sos_type=1, sos_dim="i")
        m1.add_objective(x1 * np.array([1, 2, 3]), sense="max")

        try:
            m1.solve(solver_name="gurobi")
        except gurobipy.GurobiError as exc:
            pytest.skip(f"Gurobi environment unavailable: {exc}")

        # Reformulated solution with HiGHS
        m2 = Model()
        x2 = m2.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m2.add_sos_constraints(x2, sos_type=1, sos_dim="i")
        m2.add_objective(x2 * np.array([1, 2, 3]), sense="max")

        if "highs" in available_solvers:
            m2.solve(solver_name="highs", reformulate_sos=True)
            assert m1.objective.value is not None
            assert m2.objective.value is not None
            assert np.isclose(m1.objective.value, m2.objective.value, atol=1e-5)

    def test_sos2_equivalence(self) -> None:
        """Test that reformulated SOS2 gives same result as native Gurobi."""
        gurobipy = pytest.importorskip("gurobipy")

        # Native Gurobi solution
        m1 = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x1 = m1.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m1.add_sos_constraints(x1, sos_type=2, sos_dim="i")
        m1.add_objective(x1 * np.array([1, 2, 3]), sense="max")

        try:
            m1.solve(solver_name="gurobi")
        except gurobipy.GurobiError as exc:
            pytest.skip(f"Gurobi environment unavailable: {exc}")

        # Reformulated solution with HiGHS
        m2 = Model()
        x2 = m2.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m2.add_sos_constraints(x2, sos_type=2, sos_dim="i")
        m2.add_objective(x2 * np.array([1, 2, 3]), sense="max")

        if "highs" in available_solvers:
            m2.solve(solver_name="highs", reformulate_sos=True)
            assert m1.objective.value is not None
            assert m2.objective.value is not None
            assert np.isclose(m1.objective.value, m2.objective.value, atol=1e-5)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_preserves_non_sos_variables(self) -> None:
        """Test that non-SOS variables are preserved."""
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_variables(lower=0, upper=2, coords=[idx], name="y")  # No SOS
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")

        reformulate_sos_constraints(m)

        # y should be unchanged
        assert "y" in m.variables
        assert SOS_TYPE_ATTR not in m.variables["y"].attrs

    def test_custom_prefix(self) -> None:
        """Test custom prefix for reformulation."""
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")

        reformulate_sos_constraints(m, prefix="_custom_")

        assert "_custom_x_y" in m.variables
        assert "_custom_x_upper" in m.constraints
        assert "_custom_x_card" in m.constraints

    def test_constraints_with_sos_variables(self) -> None:
        """Test that existing constraints with SOS variables work after reformulation."""
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        y = m.add_variables(lower=0, upper=10, name="y")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")

        # Add constraint involving SOS variable
        m.add_constraints(x.sum() <= y, name="linking")

        # Reformulate
        reformulate_sos_constraints(m)

        # Original constraint should still exist
        assert "linking" in m.constraints

    def test_float_coordinates(self) -> None:
        """Test SOS with float coordinates (common for piecewise linear)."""
        m = Model()
        breakpoints = pd.Index([0.0, 0.5, 1.0], name="bp")
        lambdas = m.add_variables(lower=0, upper=1, coords=[breakpoints], name="lambda")
        m.add_sos_constraints(lambdas, sos_type=2, sos_dim="bp")

        reformulate_sos_constraints(m)

        # Should work with float coordinates
        assert "_sos_reform_lambda_z" in m.variables
        z = m.variables["_sos_reform_lambda_z"]
        # Segment indicators have n-1 = 2 elements
        assert z.sizes["bp"] == 2

    def test_custom_big_m_removed_on_remove_sos(self) -> None:
        """Test that custom big_m attribute is removed with SOS constraint."""
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=100, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i", big_m=10)

        assert "big_m_upper" in x.attrs

        m.remove_sos_constraints(x)

        assert "big_m_upper" not in x.attrs


@pytest.mark.skipif("highs" not in available_solvers, reason="HiGHS not installed")
class TestCustomBigM:
    """Tests for custom Big-M functionality."""

    def test_solve_with_custom_big_m(self) -> None:
        """Test solving with custom big_m value."""
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        # Large bounds but tight effective constraint
        x = m.add_variables(lower=0, upper=1000, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i", big_m=1)
        m.add_objective(x * np.array([1, 2, 3]), sense="max")

        m.solve(solver_name="highs", reformulate_sos=True)

        # With big_m=1, maximum should be 3 (x[2]=1)
        assert m.objective.value is not None
        assert np.isclose(m.objective.value, 3, atol=1e-5)

    def test_solve_with_infinite_bounds_and_custom_big_m(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=np.inf, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i", big_m=5)
        m.add_objective(x * np.array([1, 2, 3]), sense="max")

        m.solve(solver_name="highs", reformulate_sos=True)

        assert m.objective.value is not None
        assert np.isclose(m.objective.value, 15, atol=1e-5)

    def test_solve_does_not_mutate_model(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")
        m.add_objective(x * np.array([1, 2, 3]), sense="max")

        vars_before = set(m.variables)
        cons_before = set(m.constraints)
        sos_before = list(m.variables.sos)

        m.solve(solver_name="highs", reformulate_sos=True)

        assert set(m.variables) == vars_before
        assert set(m.constraints) == cons_before
        assert list(m.variables.sos) == sos_before

    def test_solve_twice_with_reformulate_sos(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")
        m.add_objective(x * np.array([1, 2, 3]), sense="max")

        m.solve(solver_name="highs", reformulate_sos=True)
        obj1 = m.objective.value

        m.solve(solver_name="highs", reformulate_sos=True)
        obj2 = m.objective.value

        assert obj1 is not None and obj2 is not None
        assert np.isclose(obj1, obj2, atol=1e-5)


@pytest.mark.skipif("highs" not in available_solvers, reason="HiGHS not installed")
class TestNoSosConstraints:
    def test_reformulate_sos_true_with_no_sos(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_objective(x.sum(), sense="max")

        m.solve(solver_name="highs", reformulate_sos=True)

        assert m.objective.value is not None
        assert np.isclose(m.objective.value, 3, atol=1e-5)


class TestPartialFailure:
    def test_partial_failure_rolls_back(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        y = m.add_variables(lower=-1, upper=1, coords=[idx], name="y")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")
        m.add_sos_constraints(y, sos_type=1, sos_dim="i")

        vars_before = set(m.variables)
        cons_before = set(m.constraints)
        sos_before = list(m.variables.sos)

        with pytest.raises(ValueError, match="negative lower bounds"):
            reformulate_sos_constraints(m)

        assert set(m.variables) == vars_before
        assert set(m.constraints) == cons_before
        assert list(m.variables.sos) == sos_before


class TestMixedBounds:
    def test_mixed_finite_infinite_with_big_m(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(
            lower=np.array([0, 0, 0]),
            upper=np.array([5, np.inf, 10]),
            coords=[idx],
            name="x",
        )
        m.add_sos_constraints(x, sos_type=1, sos_dim="i", big_m=8)
        M = compute_big_m_values(x)
        assert np.allclose(M.values, [5, 8, 8])

    def test_mixed_finite_infinite_without_big_m_raises(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(
            lower=np.array([0, 0, 0]),
            upper=np.array([5, np.inf, 10]),
            coords=[idx],
            name="x",
        )
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")
        with pytest.raises(ValueError, match="infinite upper bounds"):
            compute_big_m_values(x)


class TestBigMValidation:
    def test_big_m_zero_raises(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        with pytest.raises(ValueError, match="big_m must be positive"):
            m.add_sos_constraints(x, sos_type=1, sos_dim="i", big_m=0)

    def test_big_m_negative_raises(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        with pytest.raises(ValueError, match="big_m must be positive"):
            m.add_sos_constraints(x, sos_type=1, sos_dim="i", big_m=-5)


class TestUndoReformulation:
    def test_undo_restores_sos_attrs(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")

        result = reformulate_sos_constraints(m)

        assert len(list(m.variables.sos)) == 0
        assert "_sos_reform_x_y" in m.variables

        undo_sos_reformulation(m, result)

        assert list(m.variables.sos) == ["x"]
        assert "_sos_reform_x_y" not in m.variables
        assert "_sos_reform_x_upper" not in m.constraints
        assert "_sos_reform_x_card" not in m.constraints

    def test_double_reformulate_is_noop(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")

        m.reformulate_sos_constraints()

        result2 = m.reformulate_sos_constraints()
        assert result2.reformulated == []

    def test_undo_restores_skipped_single_element(self) -> None:
        m = Model()
        idx = pd.Index([0], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")

        result = reformulate_sos_constraints(m)

        assert len(list(m.variables.sos)) == 0

        undo_sos_reformulation(m, result)

        assert list(m.variables.sos) == ["x"]

    def test_undo_restores_skipped_zero_bounds(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=0, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")

        result = reformulate_sos_constraints(m)

        assert len(list(m.variables.sos)) == 0

        undo_sos_reformulation(m, result)

        assert list(m.variables.sos) == ["x"]


@pytest.mark.skipif("highs" not in available_solvers, reason="HiGHS not installed")
class TestUnsortedCoords:
    def test_sos2_unsorted_coords_matches_sorted(self) -> None:
        coeffs = np.array([1, 2, 3])

        m_sorted = Model()
        idx_sorted = pd.Index([1, 2, 3], name="i")
        x_sorted = m_sorted.add_variables(
            lower=0, upper=1, coords=[idx_sorted], name="x"
        )
        m_sorted.add_sos_constraints(x_sorted, sos_type=2, sos_dim="i")
        m_sorted.add_objective(x_sorted * coeffs, sense="max")
        m_sorted.solve(solver_name="highs", reformulate_sos=True)

        m_unsorted = Model()
        idx_unsorted = pd.Index([3, 1, 2], name="i")
        x_unsorted = m_unsorted.add_variables(
            lower=0, upper=1, coords=[idx_unsorted], name="x"
        )
        m_unsorted.add_sos_constraints(x_unsorted, sos_type=2, sos_dim="i")
        m_unsorted.add_objective(x_unsorted * coeffs, sense="max")
        m_unsorted.solve(solver_name="highs", reformulate_sos=True)

        assert m_sorted.objective.value is not None
        assert m_unsorted.objective.value is not None
        assert np.isclose(
            m_sorted.objective.value, m_unsorted.objective.value, atol=1e-5
        )

    def test_sos1_unsorted_coords(self) -> None:
        m = Model()
        idx = pd.Index([3, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")
        m.add_objective(x * np.array([1, 2, 3]), sense="max")
        m.solve(solver_name="highs", reformulate_sos=True)

        assert m.objective.value is not None
        assert np.isclose(m.objective.value, 3, atol=1e-5)


@pytest.mark.skipif("highs" not in available_solvers, reason="HiGHS not installed")
class TestAutoReformulation:
    """Tests for reformulate_sos='auto' functionality."""

    @pytest.fixture()
    def sos1_model(self) -> tuple[Model, ...]:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")
        m.add_objective(x * np.array([1, 2, 3]), sense="max")
        return m, x

    def test_auto_reformulates_when_solver_lacks_sos(self, sos1_model) -> None:
        m, x = sos1_model
        m.solve(solver_name="highs", reformulate_sos="auto")

        assert np.isclose(x.solution.values[2], 1, atol=1e-5)
        assert np.isclose(x.solution.values[0], 0, atol=1e-5)
        assert np.isclose(x.solution.values[1], 0, atol=1e-5)
        assert m.objective.value is not None
        assert np.isclose(m.objective.value, 3, atol=1e-5)

    def test_auto_with_sos2(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2, 3], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=2, sos_dim="i")
        m.add_objective(x * np.array([10, 1, 1, 10]), sense="max")

        m.solve(solver_name="highs", reformulate_sos="auto")

        assert m.objective.value is not None
        nonzero_indices = np.where(np.abs(x.solution.values) > 1e-5)[0]
        assert len(nonzero_indices) <= 2
        if len(nonzero_indices) == 2:
            assert abs(nonzero_indices[1] - nonzero_indices[0]) == 1
        assert not np.isclose(m.objective.value, 20, atol=1e-5)

    def test_auto_emits_info_no_warning(self, sos1_model, caplog) -> None:
        m, _ = sos1_model

        with caplog.at_level(logging.INFO):
            m.solve(solver_name="highs", reformulate_sos="auto")

        assert any("Reformulating SOS" in msg for msg in caplog.messages)
        assert not any("supports SOS natively" in msg for msg in caplog.messages)

    @pytest.mark.skipif(
        "gurobi" not in available_solvers, reason="Gurobi not installed"
    )
    def test_auto_passes_through_native_sos_without_reformulation(self) -> None:
        import gurobipy

        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")
        m.add_objective(x * np.array([1, 2, 3]), sense="max")

        try:
            m.solve(solver_name="gurobi", reformulate_sos="auto")
        except gurobipy.GurobiError as exc:
            pytest.skip(f"Gurobi environment unavailable: {exc}")

        assert m.objective.value is not None
        assert np.isclose(m.objective.value, 3, atol=1e-5)
        assert np.isclose(x.solution.values[2], 1, atol=1e-5)
        assert np.isclose(x.solution.values[0], 0, atol=1e-5)
        assert np.isclose(x.solution.values[1], 0, atol=1e-5)

    def test_auto_multidimensional_sos1(self) -> None:
        m = Model()
        idx_i = pd.Index([0, 1, 2], name="i")
        idx_j = pd.Index([0, 1], name="j")
        x = m.add_variables(lower=0, upper=1, coords=[idx_i, idx_j], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")
        m.add_objective(x.sum(), sense="max")

        m.solve(solver_name="highs", reformulate_sos="auto")

        assert m.objective.value is not None
        assert np.isclose(m.objective.value, 2, atol=1e-5)
        for j in idx_j:
            nonzero_count = (np.abs(x.solution.sel(j=j).values) > 1e-5).sum()
            assert nonzero_count <= 1

    def test_auto_noop_without_sos(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_objective(x.sum(), sense="max")

        m.solve(solver_name="highs", reformulate_sos="auto")

        assert m.objective.value is not None
        assert np.isclose(m.objective.value, 3, atol=1e-5)

    def test_invalid_reformulate_sos_value(self) -> None:
        m = Model()
        idx = pd.Index([0, 1, 2], name="i")
        x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
        m.add_sos_constraints(x, sos_type=1, sos_dim="i")
        m.add_objective(x.sum(), sense="max")

        with pytest.raises(ValueError, match="Invalid value for reformulate_sos"):
            m.solve(solver_name="highs", reformulate_sos="invalid")
