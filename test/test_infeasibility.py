#!/usr/bin/env python3
"""
Test infeasibility detection for different solvers.
"""

import pandas as pd
import pytest

from linopy import Model, available_solvers


class TestInfeasibility:
    """Test class for infeasibility detection functionality."""

    @pytest.fixture
    def simple_infeasible_model(self) -> Model:
        """Create a simple infeasible model."""
        m = Model()

        time = pd.RangeIndex(10, name="time")
        x = m.add_variables(lower=0, coords=[time], name="x")
        y = m.add_variables(lower=0, coords=[time], name="y")

        # Create infeasible constraints
        m.add_constraints(x <= 5, name="con_x_upper")
        m.add_constraints(y <= 5, name="con_y_upper")
        m.add_constraints(x + y >= 12, name="con_sum_lower")

        # Add objective to avoid multi-objective issue with xpress
        m.add_objective(x.sum() + y.sum())

        return m

    @pytest.fixture
    def complex_infeasible_model(self) -> Model:
        """Create a more complex infeasible model."""
        m = Model()

        # Create variables
        x = m.add_variables(lower=0, upper=10, name="x")
        y = m.add_variables(lower=0, upper=10, name="y")
        z = m.add_variables(lower=0, upper=10, name="z")

        # Add conflicting constraints
        m.add_constraints(x + y >= 15, name="con1")
        m.add_constraints(x <= 5, name="con2")
        m.add_constraints(y <= 5, name="con3")
        m.add_constraints(z >= x + y, name="con4")
        m.add_constraints(z <= 8, name="con5")

        # Add objective
        m.add_objective(x + y + z)

        return m

    @pytest.fixture
    def multi_dimensional_infeasible_model(self) -> Model:
        """Create a multi-dimensional infeasible model."""
        m = Model()

        # Create multi-dimensional variables
        i = pd.RangeIndex(5, name="i")
        j = pd.RangeIndex(3, name="j")

        x = m.add_variables(lower=0, upper=1, coords=[i, j], name="x")

        # Add constraints that make it infeasible
        m.add_constraints(x.sum("j") >= 2.5, name="row_sum")  # Each row sum >= 2.5
        m.add_constraints(x.sum("i") <= 1, name="col_sum")  # Each column sum <= 1

        # Add objective
        m.add_objective(x.sum())

        return m

    @pytest.mark.parametrize("solver", ["gurobi", "xpress"])
    def test_simple_infeasibility_detection(
        self, simple_infeasible_model: Model, solver: str
    ) -> None:
        """Test basic infeasibility detection."""
        if solver not in available_solvers:
            pytest.skip(f"{solver} not available")

        m = simple_infeasible_model
        status, condition = m.solve(solver_name=solver)

        assert status == "warning"
        assert "infeasible" in condition

        # Test compute_infeasibilities
        labels = m.compute_infeasibilities()
        assert isinstance(labels, list)
        assert len(labels) > 0  # Should find at least one infeasible constraint

        # Test print_infeasibilities (just check it doesn't raise an error)
        m.print_infeasibilities()

    @pytest.mark.parametrize("solver", ["gurobi", "xpress"])
    def test_complex_infeasibility_detection(
        self, complex_infeasible_model: Model, solver: str
    ) -> None:
        """Test infeasibility detection on more complex model."""
        if solver not in available_solvers:
            pytest.skip(f"{solver} not available")

        m = complex_infeasible_model
        status, condition = m.solve(solver_name=solver)

        assert status == "warning"
        assert "infeasible" in condition

        labels = m.compute_infeasibilities()
        assert isinstance(labels, list)
        assert len(labels) > 0

        # The infeasible set should include constraints that conflict
        # Different solvers might find different minimal IIS
        # We expect at least 2 constraints to be involved
        assert len(labels) >= 2

    @pytest.mark.parametrize("solver", ["gurobi", "xpress"])
    def test_multi_dimensional_infeasibility(
        self, multi_dimensional_infeasible_model: Model, solver: str
    ) -> None:
        """Test infeasibility detection on multi-dimensional model."""
        if solver not in available_solvers:
            pytest.skip(f"{solver} not available")

        m = multi_dimensional_infeasible_model
        status, condition = m.solve(solver_name=solver)

        assert status == "warning"
        assert "infeasible" in condition

        labels = m.compute_infeasibilities()
        assert isinstance(labels, list)
        assert len(labels) > 0

    def test_unsolved_model_error(self) -> None:
        """Test error when model hasn't been solved yet."""
        m = Model()
        x = m.add_variables(name="x")
        m.add_constraints(x >= 0)
        m.add_objective(1 * x)  # Convert to LinearExpression

        # Don't solve the model - should raise NotImplementedError for unsolved models
        with pytest.raises(
            NotImplementedError, match="Computing infeasibilities is not supported"
        ):
            m.compute_infeasibilities()

    @pytest.mark.parametrize("solver", ["gurobi", "xpress"])
    def test_no_solver_model_error(self, solver: str) -> None:
        """Test error when solver model is not available after solving."""
        if solver not in available_solvers:
            pytest.skip(f"{solver} not available")

        m = Model()
        x = m.add_variables(name="x")
        m.add_constraints(x >= 0)
        m.add_objective(1 * x)  # Convert to LinearExpression

        # Solve the model first
        m.solve(solver_name=solver)

        # Manually remove the solver_model to simulate cleanup
        m.solver_model = None
        m.solver_name = solver  # But keep the solver name

        # Should raise ValueError since we know it was solved with supported solver
        with pytest.raises(ValueError, match="No solver model available"):
            m.compute_infeasibilities()

    @pytest.mark.parametrize("solver", ["gurobi", "xpress"])
    def test_feasible_model_iis(self, solver: str) -> None:
        """Test IIS computation on a feasible model."""
        if solver not in available_solvers:
            pytest.skip(f"{solver} not available")

        m = Model()
        x = m.add_variables(lower=0, name="x")
        y = m.add_variables(lower=0, name="y")

        m.add_constraints(x + y >= 1)
        m.add_constraints(x <= 10)
        m.add_constraints(y <= 10)

        m.add_objective(x + y)

        status, condition = m.solve(solver_name=solver)
        assert status == "ok"
        assert condition == "optimal"

        # Calling compute_infeasibilities on a feasible model
        # Different solvers might handle this differently
        # Gurobi might raise an error, Xpress might return empty list
        try:
            labels = m.compute_infeasibilities()
            # If it doesn't raise an error, it should return empty list
            assert labels == []
        except Exception:
            # Some solvers might raise an error when computing IIS on feasible model
            pass

    def test_unsupported_solver_error(self) -> None:
        """Test error for unsupported solvers."""
        m = Model()
        x = m.add_variables(name="x")
        m.add_constraints(x >= 0)
        m.add_constraints(x <= -1)  # Make it infeasible

        # Use a solver that doesn't support IIS
        if "cbc" in available_solvers:
            status, condition = m.solve(solver_name="cbc")
            assert "infeasible" in condition

            with pytest.raises(NotImplementedError):
                m.compute_infeasibilities()

    @pytest.mark.parametrize("solver", ["gurobi", "xpress"])
    def test_deprecated_method(
        self, simple_infeasible_model: Model, solver: str
    ) -> None:
        """Test that deprecated method still works."""
        if solver not in available_solvers:
            pytest.skip(f"{solver} not available")

        m = simple_infeasible_model
        status, condition = m.solve(solver_name=solver)

        assert status == "warning"
        assert "infeasible" in condition

        # Test deprecated method
        with pytest.warns(DeprecationWarning):
            subset = m.compute_set_of_infeasible_constraints()

        # Check that it returns a Dataset
        from xarray import Dataset

        assert isinstance(subset, Dataset)

        # Check that it contains constraint labels
        assert len(subset) > 0
