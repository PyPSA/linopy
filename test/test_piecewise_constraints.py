"""Tests for piecewise linear constraints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model, available_solvers
from linopy.constants import (
    PWL_CONVEX_SUFFIX,
    PWL_DELTA_SUFFIX,
    PWL_FILL_SUFFIX,
    PWL_LAMBDA_SUFFIX,
    PWL_LINK_SUFFIX,
)


class TestBasicSingleVariable:
    """Tests for single variable piecewise constraints."""

    def test_basic_single_variable(self) -> None:
        """Test basic piecewise constraint with a single variable."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [0, 10, 50, 100], dims=["bp"], coords={"bp": [0, 1, 2, 3]}
        )

        m.add_piecewise_constraints(x, breakpoints, dim="bp")

        # Check lambda variables were created
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables

        # Check constraints were created
        assert f"pwl0{PWL_CONVEX_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_LINK_SUFFIX}" in m.constraints

        # Check SOS2 constraint was added
        lambda_var = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        assert lambda_var.attrs.get("sos_type") == 2
        assert lambda_var.attrs.get("sos_dim") == "bp"

    def test_single_variable_with_coords(self) -> None:
        """Test piecewise constraint with a variable that has coordinates."""
        m = Model()
        generators = pd.Index(["gen1", "gen2"], name="generator")
        x = m.add_variables(coords=[generators], name="x")

        bp_coords = [0, 1, 2]
        breakpoints = xr.DataArray(
            [[0, 50, 100], [0, 30, 80]],
            dims=["generator", "bp"],
            coords={"generator": generators, "bp": bp_coords},
        )

        m.add_piecewise_constraints(x, breakpoints, dim="bp")

        # Lambda should have both generator and bp dimensions
        lambda_var = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        assert "generator" in lambda_var.dims
        assert "bp" in lambda_var.dims


class TestDictOfVariables:
    """Tests for dict of variables (multiple linked variables)."""

    def test_dict_of_variables(self) -> None:
        """Test piecewise constraint with multiple linked variables."""
        m = Model()
        power = m.add_variables(name="power")
        efficiency = m.add_variables(name="efficiency")

        breakpoints = xr.DataArray(
            [[0, 50, 100], [0.8, 0.95, 0.9]],
            dims=["var", "bp"],
            coords={"var": ["power", "efficiency"], "bp": [0, 1, 2]},
        )

        m.add_piecewise_constraints(
            {"power": power, "efficiency": efficiency},
            breakpoints,
            link_dim="var",
            dim="bp",
        )

        # Check single linking constraint was created for all variables
        assert f"pwl0{PWL_LINK_SUFFIX}" in m.constraints

    def test_dict_with_coordinates(self) -> None:
        """Test dict of variables with additional coordinates."""
        m = Model()
        generators = pd.Index(["gen1", "gen2"], name="generator")
        power = m.add_variables(coords=[generators], name="power")
        efficiency = m.add_variables(coords=[generators], name="efficiency")

        breakpoints = xr.DataArray(
            [[[0, 50, 100], [0.8, 0.95, 0.9]], [[0, 30, 80], [0.75, 0.9, 0.85]]],
            dims=["generator", "var", "bp"],
            coords={
                "generator": generators,
                "var": ["power", "efficiency"],
                "bp": [0, 1, 2],
            },
        )

        m.add_piecewise_constraints(
            {"power": power, "efficiency": efficiency},
            breakpoints,
            link_dim="var",
            dim="bp",
        )

        # Lambda should have generator and bp dimensions (not var)
        lambda_var = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        assert "generator" in lambda_var.dims
        assert "bp" in lambda_var.dims
        assert "var" not in lambda_var.dims


class TestAutoDetectLinkDim:
    """Tests for auto-detection of link_dim."""

    def test_auto_detect_link_dim(self) -> None:
        """Test that link_dim is auto-detected from breakpoints."""
        m = Model()
        power = m.add_variables(name="power")
        efficiency = m.add_variables(name="efficiency")

        breakpoints = xr.DataArray(
            [[0, 50, 100], [0.8, 0.95, 0.9]],
            dims=["var", "bp"],
            coords={"var": ["power", "efficiency"], "bp": [0, 1, 2]},
        )

        # Should auto-detect link_dim="var"
        m.add_piecewise_constraints(
            {"power": power, "efficiency": efficiency},
            breakpoints,
            dim="bp",
        )

        assert f"pwl0{PWL_LINK_SUFFIX}" in m.constraints

    def test_auto_detect_fails_with_no_match(self) -> None:
        """Test that auto-detection fails when no dimension matches keys."""
        m = Model()
        power = m.add_variables(name="power")
        efficiency = m.add_variables(name="efficiency")

        # Dimension 'wrong' doesn't match variable keys
        breakpoints = xr.DataArray(
            [[0, 50, 100], [0.8, 0.95, 0.9]],
            dims=["wrong", "bp"],
            coords={"wrong": ["a", "b"], "bp": [0, 1, 2]},
        )

        with pytest.raises(ValueError, match="Could not auto-detect link_dim"):
            m.add_piecewise_constraints(
                {"power": power, "efficiency": efficiency},
                breakpoints,
                dim="bp",
            )


class TestMasking:
    """Tests for masking functionality."""

    def test_nan_masking(self) -> None:
        """Test that NaN values in breakpoints create masked constraints."""
        m = Model()
        x = m.add_variables(name="x")

        # Third breakpoint is NaN
        breakpoints = xr.DataArray(
            [0, 10, np.nan, 100],
            dims=["bp"],
            coords={"bp": [0, 1, 2, 3]},
        )

        m.add_piecewise_constraints(x, breakpoints, dim="bp")

        # Lambda for NaN breakpoint should be masked
        lambda_var = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        # Check that at least some labels are valid
        assert (lambda_var.labels != -1).any()

    def test_explicit_mask(self) -> None:
        """Test user-provided mask."""
        m = Model()
        generators = pd.Index(["gen1", "gen2"], name="generator")
        x = m.add_variables(coords=[generators], name="x")

        breakpoints = xr.DataArray(
            [[0, 50, 100], [0, 30, 80]],
            dims=["generator", "bp"],
            coords={"generator": generators, "bp": [0, 1, 2]},
        )

        # Mask out gen2
        mask = xr.DataArray(
            [[True, True, True], [False, False, False]],
            dims=["generator", "bp"],
            coords={"generator": generators, "bp": [0, 1, 2]},
        )

        m.add_piecewise_constraints(x, breakpoints, dim="bp", mask=mask)

        # Should still create variables and constraints
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables

    def test_skip_nan_check(self) -> None:
        """Test skip_nan_check parameter for performance."""
        m = Model()
        x = m.add_variables(name="x")

        # Breakpoints with no NaNs
        breakpoints = xr.DataArray([0, 10, 50], dims=["bp"], coords={"bp": [0, 1, 2]})

        # Should work with skip_nan_check=True
        m.add_piecewise_constraints(x, breakpoints, dim="bp", skip_nan_check=True)

        # All lambda variables should be valid (no masking)
        lambda_var = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        assert (lambda_var.labels != -1).all()


class TestMultiDimensional:
    """Tests for multi-dimensional piecewise constraints."""

    def test_multi_dimensional(self) -> None:
        """Test piecewise constraint with multiple loop dimensions."""
        m = Model()
        generators = pd.Index(["gen1", "gen2"], name="generator")
        timesteps = pd.Index([0, 1, 2], name="time")
        x = m.add_variables(coords=[generators, timesteps], name="x")

        rng = np.random.default_rng(42)
        breakpoints = xr.DataArray(
            rng.random((2, 3, 4)) * 100,
            dims=["generator", "time", "bp"],
            coords={"generator": generators, "time": timesteps, "bp": [0, 1, 2, 3]},
        )

        m.add_piecewise_constraints(x, breakpoints, dim="bp")

        # Lambda should have all dimensions
        lambda_var = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        assert "generator" in lambda_var.dims
        assert "time" in lambda_var.dims
        assert "bp" in lambda_var.dims


class TestValidationErrors:
    """Tests for input validation."""

    def test_invalid_vars_type(self) -> None:
        """Test error when expr is not Variable, LinearExpression, or dict."""
        m = Model()

        breakpoints = xr.DataArray([0, 10, 50], dims=["bp"], coords={"bp": [0, 1, 2]})

        with pytest.raises(
            ValueError, match="must be a Variable, LinearExpression, or dict"
        ):
            m.add_piecewise_constraints("invalid", breakpoints, dim="bp")  # type: ignore

    def test_missing_dim(self) -> None:
        """Test error when breakpoints don't have the required dim."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray([0, 10, 50], dims=["wrong"])

        with pytest.raises(ValueError, match="must have dimension"):
            m.add_piecewise_constraints(x, breakpoints, dim="bp")

    def test_non_numeric_dim(self) -> None:
        """Test error when dim coordinates are not numeric."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [0, 10, 50],
            dims=["bp"],
            coords={"bp": ["a", "b", "c"]},  # Non-numeric
        )

        with pytest.raises(ValueError, match="numeric coordinates"):
            m.add_piecewise_constraints(x, breakpoints, dim="bp")

    def test_expression_support(self) -> None:
        """Test that LinearExpression is supported as input."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")

        breakpoints = xr.DataArray([0, 10, 50], dims=["bp"], coords={"bp": [0, 1, 2]})

        # Should work with a LinearExpression
        m.add_piecewise_constraints(x + y, breakpoints, dim="bp")

        # Check constraints were created
        assert f"pwl0{PWL_LINK_SUFFIX}" in m.constraints

    def test_link_dim_not_in_breakpoints(self) -> None:
        """Test error when link_dim is not in breakpoints."""
        m = Model()
        power = m.add_variables(name="power")
        efficiency = m.add_variables(name="efficiency")

        breakpoints = xr.DataArray([0, 50, 100], dims=["bp"], coords={"bp": [0, 1, 2]})

        with pytest.raises(ValueError, match="not found in breakpoints dimensions"):
            m.add_piecewise_constraints(
                {"power": power, "efficiency": efficiency},
                breakpoints,
                link_dim="var",
                dim="bp",
            )

    def test_link_dim_coords_mismatch(self) -> None:
        """Test error when link_dim coords don't match dict keys."""
        m = Model()
        power = m.add_variables(name="power")
        efficiency = m.add_variables(name="efficiency")

        breakpoints = xr.DataArray(
            [[0, 50, 100], [0.8, 0.95, 0.9]],
            dims=["var", "bp"],
            coords={"var": ["wrong1", "wrong2"], "bp": [0, 1, 2]},
        )

        with pytest.raises(ValueError, match="don't match expression keys"):
            m.add_piecewise_constraints(
                {"power": power, "efficiency": efficiency},
                breakpoints,
                link_dim="var",
                dim="bp",
            )


class TestNameGeneration:
    """Tests for automatic name generation."""

    def test_auto_name_generation(self) -> None:
        """Test that names are auto-generated correctly."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")

        bp1 = xr.DataArray([0, 10, 50], dims=["bp"], coords={"bp": [0, 1, 2]})
        bp2 = xr.DataArray([0, 20, 80], dims=["bp"], coords={"bp": [0, 1, 2]})

        m.add_piecewise_constraints(x, bp1, dim="bp")
        m.add_piecewise_constraints(y, bp2, dim="bp")

        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl1{PWL_LAMBDA_SUFFIX}" in m.variables

    def test_custom_name(self) -> None:
        """Test using a custom name."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray([0, 10, 50], dims=["bp"], coords={"bp": [0, 1, 2]})

        m.add_piecewise_constraints(x, breakpoints, dim="bp", name="my_pwl")

        assert f"my_pwl{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"my_pwl{PWL_CONVEX_SUFFIX}" in m.constraints
        assert f"my_pwl{PWL_LINK_SUFFIX}" in m.constraints


class TestLPFileOutput:
    """Tests for LP file output with piecewise constraints."""

    def test_piecewise_written_to_lp(self, tmp_path: Path) -> None:
        """Test that piecewise constraints are properly written to LP file."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [0.0, 10.0, 50.0],
            dims=["bp"],
            coords={"bp": [0, 1, 2]},
        )

        m.add_piecewise_constraints(x, breakpoints, dim="bp")

        # Add a simple objective to make it a valid LP
        m.add_objective(x)

        fn = tmp_path / "pwl.lp"
        m.to_file(fn, io_api="lp")
        content = fn.read_text()

        # Should contain SOS2 section
        assert "\nsos\n" in content.lower()
        assert "s2" in content.lower()


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobi not installed")
class TestSolverIntegration:
    """Integration tests with Gurobi solver."""

    def test_solve_single_variable(self) -> None:
        """Test solving a model with piecewise constraint."""
        gurobipy = pytest.importorskip("gurobipy")

        m = Model()
        # Variable that should be between 0 and 100
        x = m.add_variables(lower=0, upper=100, name="x")

        # Piecewise linear cost function: cost = f(x)
        # f(0) = 0, f(50) = 10, f(100) = 50
        cost = m.add_variables(name="cost")

        breakpoints = xr.DataArray(
            [[0, 50, 100], [0, 10, 50]],
            dims=["var", "bp"],
            coords={"var": ["x", "cost"], "bp": [0, 1, 2]},
        )

        m.add_piecewise_constraints(
            {"x": x, "cost": cost}, breakpoints, link_dim="var", dim="bp"
        )

        # Minimize cost, but need x >= 50 to make it interesting
        m.add_constraints(x >= 50, name="x_min")
        m.add_objective(cost)

        try:
            status, cond = m.solve(solver_name="gurobi", io_api="direct")
        except gurobipy.GurobiError as exc:
            pytest.skip(f"Gurobi environment unavailable: {exc}")

        assert status == "ok"
        # At x=50, cost should be 10
        assert np.isclose(x.solution.values, 50, atol=1e-5)
        assert np.isclose(cost.solution.values, 10, atol=1e-5)

    def test_solve_efficiency_curve(self) -> None:
        """Test solving with a realistic efficiency curve."""
        gurobipy = pytest.importorskip("gurobipy")

        m = Model()
        power = m.add_variables(lower=0, upper=100, name="power")
        efficiency = m.add_variables(name="efficiency")

        # Efficiency curve: starts low, peaks, then decreases
        # power:      0    25    50    75   100
        # efficiency: 0.7  0.85  0.95  0.9  0.8
        breakpoints = xr.DataArray(
            [[0, 25, 50, 75, 100], [0.7, 0.85, 0.95, 0.9, 0.8]],
            dims=["var", "bp"],
            coords={"var": ["power", "efficiency"], "bp": [0, 1, 2, 3, 4]},
        )

        m.add_piecewise_constraints(
            {"power": power, "efficiency": efficiency},
            breakpoints,
            link_dim="var",
            dim="bp",
        )

        # Maximize efficiency
        m.add_objective(efficiency, sense="max")

        try:
            status, cond = m.solve(solver_name="gurobi", io_api="direct")
        except gurobipy.GurobiError as exc:
            pytest.skip(f"Gurobi environment unavailable: {exc}")

        assert status == "ok"
        # Maximum efficiency is at power=50
        assert np.isclose(power.solution.values, 50, atol=1e-5)
        assert np.isclose(efficiency.solution.values, 0.95, atol=1e-5)

    def test_solve_multi_generator(self) -> None:
        """Test with multiple generators each with different curves."""
        gurobipy = pytest.importorskip("gurobipy")

        m = Model()
        generators = pd.Index(["gen1", "gen2"], name="generator")
        power = m.add_variables(lower=0, upper=100, coords=[generators], name="power")
        cost = m.add_variables(coords=[generators], name="cost")

        # Different cost curves for each generator
        # gen1: cheaper at low power, expensive at high
        # gen2: more expensive at low power, cheaper at high
        breakpoints = xr.DataArray(
            [
                [[0, 50, 100], [0, 5, 30]],  # gen1: power, cost
                [[0, 50, 100], [0, 15, 20]],  # gen2: power, cost
            ],
            dims=["generator", "var", "bp"],
            coords={
                "generator": generators,
                "var": ["power", "cost"],
                "bp": [0, 1, 2],
            },
        )

        m.add_piecewise_constraints(
            {"power": power, "cost": cost}, breakpoints, link_dim="var", dim="bp"
        )

        # Need total power of 120
        m.add_constraints(power.sum() >= 120, name="demand")

        # Minimize total cost
        m.add_objective(cost.sum())

        try:
            status, cond = m.solve(solver_name="gurobi", io_api="direct")
        except gurobipy.GurobiError as exc:
            pytest.skip(f"Gurobi environment unavailable: {exc}")

        assert status == "ok"
        # gen1 should provide ~50 (cheap up to 50), gen2 provides rest
        total_power = power.solution.sum().values
        assert np.isclose(total_power, 120, atol=1e-5)


class TestIncrementalFormulation:
    """Tests for the incremental (delta) piecewise formulation."""

    def test_single_variable_incremental(self) -> None:
        """Test incremental formulation with a single variable."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [0, 10, 50, 100], dims=["bp"], coords={"bp": [0, 1, 2, 3]}
        )

        m.add_piecewise_constraints(x, breakpoints, dim="bp", method="incremental")

        # Check delta variables created
        assert f"pwl0{PWL_DELTA_SUFFIX}" in m.variables
        # 3 segments → 3 delta vars
        delta_var = m.variables[f"pwl0{PWL_DELTA_SUFFIX}"]
        assert "bp_seg" in delta_var.dims
        assert len(delta_var.coords["bp_seg"]) == 3

        # Check filling-order constraints (2 for 3 segments)
        assert f"pwl0{PWL_FILL_SUFFIX}_0" in m.constraints
        assert f"pwl0{PWL_FILL_SUFFIX}_1" in m.constraints

        # Check link constraint
        assert f"pwl0{PWL_LINK_SUFFIX}" in m.constraints

        # No SOS2 or lambda variables
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" not in m.variables

    def test_two_breakpoints_incremental(self) -> None:
        """Test incremental with only 2 breakpoints (1 segment, no fill constraints)."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray([0, 100], dims=["bp"], coords={"bp": [0, 1]})

        m.add_piecewise_constraints(x, breakpoints, dim="bp", method="incremental")

        # 1 segment → 1 delta var, no filling constraints
        delta_var = m.variables[f"pwl0{PWL_DELTA_SUFFIX}"]
        assert len(delta_var.coords["bp_seg"]) == 1

        # Link constraint should exist
        assert f"pwl0{PWL_LINK_SUFFIX}" in m.constraints

    def test_dict_incremental(self) -> None:
        """Test incremental formulation with dict of variables."""
        m = Model()
        power = m.add_variables(name="power")
        cost = m.add_variables(name="cost")

        # Both power and cost breakpoints are strictly increasing
        breakpoints = xr.DataArray(
            [[0, 50, 100], [0, 10, 50]],
            dims=["var", "bp"],
            coords={"var": ["power", "cost"], "bp": [0, 1, 2]},
        )

        m.add_piecewise_constraints(
            {"power": power, "cost": cost},
            breakpoints,
            link_dim="var",
            dim="bp",
            method="incremental",
        )

        assert f"pwl0{PWL_DELTA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_LINK_SUFFIX}" in m.constraints

    def test_non_monotonic_raises(self) -> None:
        """Test that non-monotonic breakpoints raise ValueError for incremental."""
        m = Model()
        x = m.add_variables(name="x")

        # Not monotonic: 0, 50, 30
        breakpoints = xr.DataArray([0, 50, 30], dims=["bp"], coords={"bp": [0, 1, 2]})

        with pytest.raises(ValueError, match="strictly monotonic"):
            m.add_piecewise_constraints(x, breakpoints, dim="bp", method="incremental")

    def test_decreasing_monotonic_works(self) -> None:
        """Test that strictly decreasing breakpoints work for incremental."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [100, 50, 10, 0], dims=["bp"], coords={"bp": [0, 1, 2, 3]}
        )

        m.add_piecewise_constraints(x, breakpoints, dim="bp", method="incremental")
        assert f"pwl0{PWL_DELTA_SUFFIX}" in m.variables

    def test_auto_selects_incremental(self) -> None:
        """Test method='auto' selects incremental for monotonic breakpoints."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [0, 10, 50, 100], dims=["bp"], coords={"bp": [0, 1, 2, 3]}
        )

        m.add_piecewise_constraints(x, breakpoints, dim="bp", method="auto")

        # Should use incremental (delta vars, no lambda)
        assert f"pwl0{PWL_DELTA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" not in m.variables

    def test_auto_selects_sos2(self) -> None:
        """Test method='auto' falls back to sos2 for non-monotonic breakpoints."""
        m = Model()
        x = m.add_variables(name="x")

        # Non-monotonic across the full array (dict case would have link_dim)
        # For single expr, breakpoints along dim are [0, 50, 30]
        breakpoints = xr.DataArray([0, 50, 30], dims=["bp"], coords={"bp": [0, 1, 2]})

        m.add_piecewise_constraints(x, breakpoints, dim="bp", method="auto")

        # Should use sos2 (lambda vars, no delta)
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_DELTA_SUFFIX}" not in m.variables

    def test_invalid_method_raises(self) -> None:
        """Test that an invalid method raises ValueError."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray([0, 10, 50], dims=["bp"], coords={"bp": [0, 1, 2]})

        with pytest.raises(ValueError, match="method must be"):
            m.add_piecewise_constraints(x, breakpoints, dim="bp", method="invalid")

    def test_incremental_with_coords(self) -> None:
        """Test incremental formulation with extra coordinates."""
        m = Model()
        generators = pd.Index(["gen1", "gen2"], name="generator")
        x = m.add_variables(coords=[generators], name="x")

        breakpoints = xr.DataArray(
            [[0, 50, 100], [0, 30, 80]],
            dims=["generator", "bp"],
            coords={"generator": generators, "bp": [0, 1, 2]},
        )

        m.add_piecewise_constraints(x, breakpoints, dim="bp", method="incremental")

        delta_var = m.variables[f"pwl0{PWL_DELTA_SUFFIX}"]
        assert "generator" in delta_var.dims
        assert "bp_seg" in delta_var.dims


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobi not installed")
class TestIncrementalSolverIntegration:
    """Integration tests for incremental formulation with Gurobi."""

    def test_solve_incremental_single(self) -> None:
        """Test solving with incremental formulation."""
        gurobipy = pytest.importorskip("gurobipy")

        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        cost = m.add_variables(name="cost")

        # Monotonic breakpoints for both x and cost
        breakpoints = xr.DataArray(
            [[0, 50, 100], [0, 10, 50]],
            dims=["var", "bp"],
            coords={"var": ["x", "cost"], "bp": [0, 1, 2]},
        )

        m.add_piecewise_constraints(
            {"x": x, "cost": cost},
            breakpoints,
            link_dim="var",
            dim="bp",
            method="incremental",
        )

        m.add_constraints(x >= 50, name="x_min")
        m.add_objective(cost)

        try:
            status, cond = m.solve(solver_name="gurobi", io_api="direct")
        except gurobipy.GurobiError as exc:
            pytest.skip(f"Gurobi environment unavailable: {exc}")

        assert status == "ok"
        assert np.isclose(x.solution.values, 50, atol=1e-5)
        assert np.isclose(cost.solution.values, 10, atol=1e-5)
