"""Tests for piecewise linear constraints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model, available_solvers
from linopy.constants import (
    PWL_BINARY_SUFFIX,
    PWL_CONVEX_SUFFIX,
    PWL_DELTA_SUFFIX,
    PWL_FILL_SUFFIX,
    PWL_LAMBDA_SUFFIX,
    PWL_LINK_SUFFIX,
    PWL_SELECT_SUFFIX,
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

        # Check filling-order constraint (single vectorized constraint)
        assert f"pwl0{PWL_FILL_SUFFIX}" in m.constraints

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

    def test_opposite_directions_in_dict(self) -> None:
        """Test that dict with opposite monotonic directions works."""
        m = Model()
        power = m.add_variables(name="power")
        eff = m.add_variables(name="eff")

        # power increasing, efficiency decreasing
        breakpoints = xr.DataArray(
            [[0, 50, 100], [0.95, 0.9, 0.8]],
            dims=["var", "bp"],
            coords={"var": ["power", "eff"], "bp": [0, 1, 2]},
        )

        m.add_piecewise_constraints(
            {"power": power, "eff": eff},
            breakpoints,
            link_dim="var",
            dim="bp",
            method="incremental",
        )

        assert f"pwl0{PWL_DELTA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_LINK_SUFFIX}" in m.constraints

    def test_nan_breakpoints_monotonic(self) -> None:
        """Test that NaN breakpoints don't break monotonicity check."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [0, 10, np.nan, 100], dims=["bp"], coords={"bp": [0, 1, 2, 3]}
        )

        # Should detect as monotonic despite NaN
        m.add_piecewise_constraints(x, breakpoints, dim="bp", method="auto")
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


# ===== Disjunctive Piecewise Linear Constraint Tests =====


class TestDisjunctiveBasicSingleVariable:
    """Tests for single variable disjunctive piecewise constraints."""

    def test_two_equal_segments(self) -> None:
        """Test with two equal-length segments."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [[0, 10], [50, 100]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1]},
        )

        m.add_disjunctive_piecewise_constraints(x, breakpoints)

        # Binary variables created
        assert f"pwl0{PWL_BINARY_SUFFIX}" in m.variables
        # Selection constraint
        assert f"pwl0{PWL_SELECT_SUFFIX}" in m.constraints
        # Lambda variables
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        # Convexity constraint
        assert f"pwl0{PWL_CONVEX_SUFFIX}" in m.constraints
        # Link constraint
        assert f"pwl0{PWL_LINK_SUFFIX}" in m.constraints
        # SOS2 on lambda
        lambda_var = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        assert lambda_var.attrs.get("sos_type") == 2
        assert lambda_var.attrs.get("sos_dim") == "breakpoint"

    def test_uneven_segments_with_nan(self) -> None:
        """Test segments of different lengths with NaN padding."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [[0, 5, 10], [50, 100, np.nan]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1, 2]},
        )

        m.add_disjunctive_piecewise_constraints(x, breakpoints)

        # Lambda for NaN breakpoint should be masked
        lambda_var = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        assert "segment" in lambda_var.dims
        assert "breakpoint" in lambda_var.dims

    def test_single_breakpoint_segment(self) -> None:
        """Test with a segment that has only one valid breakpoint (point segment)."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [[0, 10], [42, np.nan]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1]},
        )

        m.add_disjunctive_piecewise_constraints(x, breakpoints)
        assert f"pwl0{PWL_BINARY_SUFFIX}" in m.variables

    def test_single_variable_with_coords(self) -> None:
        """Test coordinates are preserved on binary and lambda variables."""
        m = Model()
        generators = pd.Index(["gen1", "gen2"], name="generator")
        x = m.add_variables(coords=[generators], name="x")

        breakpoints = xr.DataArray(
            [
                [[0, 10], [50, 100]],
                [[0, 20], [60, 90]],
            ],
            dims=["generator", "segment", "breakpoint"],
            coords={
                "generator": generators,
                "segment": [0, 1],
                "breakpoint": [0, 1],
            },
        )

        m.add_disjunctive_piecewise_constraints(x, breakpoints)

        binary_var = m.variables[f"pwl0{PWL_BINARY_SUFFIX}"]
        lambda_var = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]

        # Both should preserve generator coordinates
        assert list(binary_var.coords["generator"].values) == ["gen1", "gen2"]
        assert list(lambda_var.coords["generator"].values) == ["gen1", "gen2"]

        # Binary has (generator, segment), lambda has (generator, segment, breakpoint)
        assert set(binary_var.dims) == {"generator", "segment"}
        assert set(lambda_var.dims) == {"generator", "segment", "breakpoint"}

    def test_return_value_is_selection_constraint(self) -> None:
        """Test the return value is the selection constraint."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [[0, 10], [50, 100]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1]},
        )

        result = m.add_disjunctive_piecewise_constraints(x, breakpoints)

        # Return value should be the selection constraint
        assert result is not None
        select_name = f"pwl0{PWL_SELECT_SUFFIX}"
        assert select_name in m.constraints


class TestDisjunctiveDictOfVariables:
    """Tests for dict of variables with disjunctive constraints."""

    def test_dict_with_two_segments(self) -> None:
        """Test dict of variables with two segments."""
        m = Model()
        power = m.add_variables(name="power")
        cost = m.add_variables(name="cost")

        breakpoints = xr.DataArray(
            [[[0, 50], [0, 10]], [[80, 100], [20, 50]]],
            dims=["segment", "var", "breakpoint"],
            coords={
                "segment": [0, 1],
                "var": ["power", "cost"],
                "breakpoint": [0, 1],
            },
        )

        m.add_disjunctive_piecewise_constraints(
            {"power": power, "cost": cost},
            breakpoints,
            link_dim="var",
        )

        assert f"pwl0{PWL_BINARY_SUFFIX}" in m.variables
        assert f"pwl0{PWL_LINK_SUFFIX}" in m.constraints

    def test_auto_detect_link_dim_with_segment_dim(self) -> None:
        """Test auto-detection of link_dim when segment_dim is also present."""
        m = Model()
        power = m.add_variables(name="power")
        cost = m.add_variables(name="cost")

        breakpoints = xr.DataArray(
            [[[0, 50], [0, 10]], [[80, 100], [20, 50]]],
            dims=["segment", "var", "breakpoint"],
            coords={
                "segment": [0, 1],
                "var": ["power", "cost"],
                "breakpoint": [0, 1],
            },
        )

        # Should auto-detect link_dim="var" (not segment)
        m.add_disjunctive_piecewise_constraints(
            {"power": power, "cost": cost},
            breakpoints,
        )

        assert f"pwl0{PWL_LINK_SUFFIX}" in m.constraints


class TestDisjunctiveExtraDimensions:
    """Tests for extra dimensions on disjunctive constraints."""

    def test_extra_generator_dimension(self) -> None:
        """Test with an extra generator dimension."""
        m = Model()
        generators = pd.Index(["gen1", "gen2"], name="generator")
        x = m.add_variables(coords=[generators], name="x")

        breakpoints = xr.DataArray(
            [
                [[0, 10], [50, 100]],
                [[0, 20], [60, 90]],
            ],
            dims=["generator", "segment", "breakpoint"],
            coords={
                "generator": generators,
                "segment": [0, 1],
                "breakpoint": [0, 1],
            },
        )

        m.add_disjunctive_piecewise_constraints(x, breakpoints)

        # Binary and lambda should have generator dimension
        binary_var = m.variables[f"pwl0{PWL_BINARY_SUFFIX}"]
        lambda_var = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        assert "generator" in binary_var.dims
        assert "generator" in lambda_var.dims
        assert "segment" in binary_var.dims
        assert "segment" in lambda_var.dims

    def test_multi_dimensional_generator_time(self) -> None:
        """Test variable with generator + time coords, verify all dims present."""
        m = Model()
        generators = pd.Index(["gen1", "gen2"], name="generator")
        timesteps = pd.Index([0, 1, 2], name="time")
        x = m.add_variables(coords=[generators, timesteps], name="x")

        rng = np.random.default_rng(42)
        bp_data = rng.random((2, 3, 2, 2)) * 100
        # Sort breakpoints within each segment
        bp_data = np.sort(bp_data, axis=-1)

        breakpoints = xr.DataArray(
            bp_data,
            dims=["generator", "time", "segment", "breakpoint"],
            coords={
                "generator": generators,
                "time": timesteps,
                "segment": [0, 1],
                "breakpoint": [0, 1],
            },
        )

        m.add_disjunctive_piecewise_constraints(x, breakpoints)

        binary_var = m.variables[f"pwl0{PWL_BINARY_SUFFIX}"]
        lambda_var = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]

        # All extra dims should be present
        for dim_name in ["generator", "time", "segment"]:
            assert dim_name in binary_var.dims
        for dim_name in ["generator", "time", "segment", "breakpoint"]:
            assert dim_name in lambda_var.dims

    def test_dict_with_additional_coords(self) -> None:
        """Test dict of variables with extra generator dim, binary/lambda exclude link_dim."""
        m = Model()
        generators = pd.Index(["gen1", "gen2"], name="generator")
        power = m.add_variables(coords=[generators], name="power")
        cost = m.add_variables(coords=[generators], name="cost")

        breakpoints = xr.DataArray(
            [
                [[[0, 50], [0, 10]], [[80, 100], [20, 30]]],
                [[[0, 40], [0, 8]], [[70, 90], [15, 25]]],
            ],
            dims=["generator", "segment", "var", "breakpoint"],
            coords={
                "generator": generators,
                "segment": [0, 1],
                "var": ["power", "cost"],
                "breakpoint": [0, 1],
            },
        )

        m.add_disjunctive_piecewise_constraints(
            {"power": power, "cost": cost},
            breakpoints,
            link_dim="var",
        )

        binary_var = m.variables[f"pwl0{PWL_BINARY_SUFFIX}"]
        lambda_var = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]

        # link_dim (var) should NOT be in binary or lambda dims
        assert "var" not in binary_var.dims
        assert "var" not in lambda_var.dims

        # generator should be present
        assert "generator" in binary_var.dims
        assert "generator" in lambda_var.dims


class TestDisjunctiveMasking:
    """Tests for masking functionality in disjunctive constraints."""

    def test_nan_masking_labels(self) -> None:
        """Test NaN breakpoints mask lambda labels to -1."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [[0, 5, 10], [50, 100, np.nan]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1, 2]},
        )

        m.add_disjunctive_piecewise_constraints(x, breakpoints)

        lambda_var = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        # Segment 0: all 3 breakpoints valid (labels != -1)
        seg0_labels = lambda_var.labels.sel(segment=0)
        assert (seg0_labels != -1).all()
        # Segment 1: breakpoint 2 is NaN → masked (label == -1)
        seg1_bp2_label = lambda_var.labels.sel(segment=1, breakpoint=2)
        assert int(seg1_bp2_label) == -1

        # Binary: both segments have at least one valid breakpoint
        binary_var = m.variables[f"pwl0{PWL_BINARY_SUFFIX}"]
        assert (binary_var.labels != -1).all()

    def test_nan_masking_partial_segment(self) -> None:
        """Test partial NaN — lambda masked but segment binary still valid."""
        m = Model()
        x = m.add_variables(name="x")

        # Segment 0 has 3 valid breakpoints, segment 1 has 2 valid + 1 NaN
        breakpoints = xr.DataArray(
            [[0, 5, 10], [50, 100, np.nan]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1, 2]},
        )

        m.add_disjunctive_piecewise_constraints(x, breakpoints)

        lambda_var = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        binary_var = m.variables[f"pwl0{PWL_BINARY_SUFFIX}"]

        # Segment 1 binary is still valid (has 2 valid breakpoints)
        assert int(binary_var.labels.sel(segment=1)) != -1

        # Segment 1 valid lambdas (breakpoint 0, 1) should be valid
        assert int(lambda_var.labels.sel(segment=1, breakpoint=0)) != -1
        assert int(lambda_var.labels.sel(segment=1, breakpoint=1)) != -1

    def test_explicit_mask(self) -> None:
        """Test user-provided mask disables specific entries."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [[0, 10], [50, 100]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1]},
        )

        # Mask out entire segment 1
        mask = xr.DataArray(
            [[True, True], [False, False]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1]},
        )

        m.add_disjunctive_piecewise_constraints(x, breakpoints, mask=mask)

        lambda_var = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        binary_var = m.variables[f"pwl0{PWL_BINARY_SUFFIX}"]

        # Segment 0 lambdas should be valid
        assert (lambda_var.labels.sel(segment=0) != -1).all()
        # Segment 1 lambdas should be masked
        assert (lambda_var.labels.sel(segment=1) == -1).all()
        # Segment 1 binary should be masked (no valid breakpoints)
        assert int(binary_var.labels.sel(segment=1)) == -1

    def test_skip_nan_check(self) -> None:
        """Test skip_nan_check=True treats all breakpoints as valid."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [[0, 5, 10], [50, 100, np.nan]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1, 2]},
        )

        m.add_disjunctive_piecewise_constraints(x, breakpoints, skip_nan_check=True)

        lambda_var = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        # All labels should be valid (no masking)
        assert (lambda_var.labels != -1).all()


class TestDisjunctiveValidationErrors:
    """Tests for validation errors in disjunctive constraints."""

    def test_missing_dim(self) -> None:
        """Test error when breakpoints don't have dim."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [[0, 10], [50, 100]],
            dims=["segment", "wrong"],
            coords={"segment": [0, 1], "wrong": [0, 1]},
        )

        with pytest.raises(ValueError, match="must have dimension"):
            m.add_disjunctive_piecewise_constraints(x, breakpoints, dim="breakpoint")

    def test_missing_segment_dim(self) -> None:
        """Test error when breakpoints don't have segment_dim."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [0, 10, 50],
            dims=["breakpoint"],
            coords={"breakpoint": [0, 1, 2]},
        )

        with pytest.raises(ValueError, match="must have dimension"):
            m.add_disjunctive_piecewise_constraints(x, breakpoints)

    def test_same_dim_segment_dim(self) -> None:
        """Test error when dim == segment_dim."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [[0, 10], [50, 100]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1]},
        )

        with pytest.raises(ValueError, match="must be different"):
            m.add_disjunctive_piecewise_constraints(
                x, breakpoints, dim="segment", segment_dim="segment"
            )

    def test_non_numeric_coords(self) -> None:
        """Test error when dim coordinates are not numeric."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [[0, 10], [50, 100]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": ["a", "b"]},
        )

        with pytest.raises(ValueError, match="numeric coordinates"):
            m.add_disjunctive_piecewise_constraints(x, breakpoints)

    def test_invalid_expr(self) -> None:
        """Test error when expr is invalid type."""
        m = Model()

        breakpoints = xr.DataArray(
            [[0, 10], [50, 100]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1]},
        )

        with pytest.raises(
            ValueError, match="must be a Variable, LinearExpression, or dict"
        ):
            m.add_disjunctive_piecewise_constraints("invalid", breakpoints)  # type: ignore

    def test_expression_support(self) -> None:
        """Test that LinearExpression (x + y) works as input."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")

        breakpoints = xr.DataArray(
            [[0, 10], [50, 100]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1]},
        )

        m.add_disjunctive_piecewise_constraints(x + y, breakpoints)

        assert f"pwl0{PWL_BINARY_SUFFIX}" in m.variables
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_LINK_SUFFIX}" in m.constraints

    def test_link_dim_not_in_breakpoints(self) -> None:
        """Test error when explicit link_dim not in breakpoints."""
        m = Model()
        power = m.add_variables(name="power")
        cost = m.add_variables(name="cost")

        breakpoints = xr.DataArray(
            [[0, 50], [80, 100]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1]},
        )

        with pytest.raises(ValueError, match="not found in breakpoints dimensions"):
            m.add_disjunctive_piecewise_constraints(
                {"power": power, "cost": cost},
                breakpoints,
                link_dim="var",
            )

    def test_link_dim_coords_mismatch(self) -> None:
        """Test error when link_dim coords don't match dict keys."""
        m = Model()
        power = m.add_variables(name="power")
        cost = m.add_variables(name="cost")

        breakpoints = xr.DataArray(
            [[[0, 50], [0, 10]], [[80, 100], [20, 30]]],
            dims=["segment", "var", "breakpoint"],
            coords={
                "segment": [0, 1],
                "var": ["wrong1", "wrong2"],
                "breakpoint": [0, 1],
            },
        )

        with pytest.raises(ValueError, match="don't match expression keys"):
            m.add_disjunctive_piecewise_constraints(
                {"power": power, "cost": cost},
                breakpoints,
                link_dim="var",
            )


class TestDisjunctiveNameGeneration:
    """Tests for name generation in disjunctive constraints."""

    def test_shared_counter_with_continuous(self) -> None:
        """Test that disjunctive and continuous PWL share the counter."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")

        bp_continuous = xr.DataArray([0, 10, 50], dims=["bp"], coords={"bp": [0, 1, 2]})
        m.add_piecewise_constraints(x, bp_continuous, dim="bp")

        bp_disjunctive = xr.DataArray(
            [[0, 10], [50, 100]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1]},
        )
        m.add_disjunctive_piecewise_constraints(y, bp_disjunctive)

        # First is pwl0, second is pwl1
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl1{PWL_BINARY_SUFFIX}" in m.variables

    def test_custom_name(self) -> None:
        """Test custom name for disjunctive constraints."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [[0, 10], [50, 100]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1]},
        )

        m.add_disjunctive_piecewise_constraints(x, breakpoints, name="my_dpwl")

        assert f"my_dpwl{PWL_BINARY_SUFFIX}" in m.variables
        assert f"my_dpwl{PWL_SELECT_SUFFIX}" in m.constraints
        assert f"my_dpwl{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"my_dpwl{PWL_CONVEX_SUFFIX}" in m.constraints
        assert f"my_dpwl{PWL_LINK_SUFFIX}" in m.constraints


class TestDisjunctiveLPFileOutput:
    """Tests for LP file output with disjunctive piecewise constraints."""

    def test_lp_contains_sos2_and_binary(self, tmp_path: Path) -> None:
        """Test LP file contains SOS2 section and binary variables."""
        m = Model()
        x = m.add_variables(name="x")

        breakpoints = xr.DataArray(
            [[0.0, 10.0], [50.0, 100.0]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1]},
        )

        m.add_disjunctive_piecewise_constraints(x, breakpoints)
        m.add_objective(x)

        fn = tmp_path / "dpwl.lp"
        m.to_file(fn, io_api="lp")
        content = fn.read_text()

        # Should contain SOS2 section
        assert "\nsos\n" in content.lower()
        assert "s2" in content.lower()

        # Should contain binary section
        assert "binary" in content.lower() or "binaries" in content.lower()


class TestDisjunctiveMultiBreakpointSegments:
    """Tests for segments with multiple breakpoints (unique to disjunctive formulation)."""

    def test_three_breakpoints_per_segment(self) -> None:
        """Test segments with 3 breakpoints each — verify lambda shape."""
        m = Model()
        x = m.add_variables(name="x")

        # 2 segments, each with 3 breakpoints
        breakpoints = xr.DataArray(
            [[0, 5, 10], [50, 75, 100]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1, 2]},
        )

        m.add_disjunctive_piecewise_constraints(x, breakpoints)

        lambda_var = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        # Lambda should have shape (2 segments, 3 breakpoints)
        assert lambda_var.labels.sizes["segment"] == 2
        assert lambda_var.labels.sizes["breakpoint"] == 3
        # All labels valid (no NaN)
        assert (lambda_var.labels != -1).all()

    def test_mixed_segment_lengths_nan_padding(self) -> None:
        """Test one segment with 4 breakpoints, another with 2 (NaN-padded)."""
        m = Model()
        x = m.add_variables(name="x")

        # Segment 0: 4 valid breakpoints
        # Segment 1: 2 valid breakpoints + 2 NaN
        breakpoints = xr.DataArray(
            [[0, 5, 10, 15], [50, 100, np.nan, np.nan]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1, 2, 3]},
        )

        m.add_disjunctive_piecewise_constraints(x, breakpoints)

        lambda_var = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        binary_var = m.variables[f"pwl0{PWL_BINARY_SUFFIX}"]

        # Lambda shape: (2 segments, 4 breakpoints)
        assert lambda_var.labels.sizes["segment"] == 2
        assert lambda_var.labels.sizes["breakpoint"] == 4

        # Segment 0: all 4 lambdas valid
        assert (lambda_var.labels.sel(segment=0) != -1).all()

        # Segment 1: first 2 valid, last 2 masked
        assert (lambda_var.labels.sel(segment=1, breakpoint=0) != -1).item()
        assert (lambda_var.labels.sel(segment=1, breakpoint=1) != -1).item()
        assert (lambda_var.labels.sel(segment=1, breakpoint=2) == -1).item()
        assert (lambda_var.labels.sel(segment=1, breakpoint=3) == -1).item()

        # Both segment binaries valid (both have at least one valid breakpoint)
        assert (binary_var.labels != -1).all()


_disjunctive_solvers = [s for s in ["gurobi", "highs"] if s in available_solvers]


@pytest.mark.skipif(
    len(_disjunctive_solvers) == 0,
    reason="No supported solver (gurobi/highs) installed",
)
class TestDisjunctiveSolverIntegration:
    """Integration tests for disjunctive piecewise constraints."""

    @pytest.fixture(params=_disjunctive_solvers)
    def solver_name(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def _solve(self, m: Model, solver_name: str) -> tuple:
        """Solve model, skipping if solver environment is unavailable."""
        try:
            if solver_name == "gurobi":
                gurobipy = pytest.importorskip("gurobipy")
                try:
                    return m.solve(solver_name="gurobi", io_api="direct")
                except gurobipy.GurobiError as exc:
                    pytest.skip(f"Gurobi environment unavailable: {exc}")
            else:
                return m.solve(solver_name=solver_name)
        except Exception as exc:
            pytest.skip(f"Solver {solver_name} unavailable: {exc}")

    def test_minimize_picks_low_segment(self, solver_name: str) -> None:
        """Test minimizing x picks the lower segment."""
        m = Model()
        x = m.add_variables(name="x")

        # Two segments: [0, 10] and [50, 100]
        breakpoints = xr.DataArray(
            [[0.0, 10.0], [50.0, 100.0]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1]},
        )

        m.add_disjunctive_piecewise_constraints(x, breakpoints)
        m.add_objective(x)

        status, cond = self._solve(m, solver_name)

        assert status == "ok"
        # Should pick x=0 (minimum of low segment)
        assert np.isclose(x.solution.values, 0.0, atol=1e-5)

    def test_maximize_picks_high_segment(self, solver_name: str) -> None:
        """Test maximizing x picks the upper segment."""
        m = Model()
        x = m.add_variables(name="x")

        # Two segments: [0, 10] and [50, 100]
        breakpoints = xr.DataArray(
            [[0.0, 10.0], [50.0, 100.0]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1]},
        )

        m.add_disjunctive_piecewise_constraints(x, breakpoints)
        m.add_objective(x, sense="max")

        status, cond = self._solve(m, solver_name)

        assert status == "ok"
        # Should pick x=100 (maximum of high segment)
        assert np.isclose(x.solution.values, 100.0, atol=1e-5)

    def test_dict_case_solver(self, solver_name: str) -> None:
        """Test disjunctive with dict of variables and solver."""
        m = Model()
        power = m.add_variables(name="power")
        cost = m.add_variables(name="cost")

        # Two operating regions:
        # Region 0: power [0,50], cost [0,10]
        # Region 1: power [80,100], cost [20,30]
        breakpoints = xr.DataArray(
            [[[0.0, 50.0], [0.0, 10.0]], [[80.0, 100.0], [20.0, 30.0]]],
            dims=["segment", "var", "breakpoint"],
            coords={
                "segment": [0, 1],
                "var": ["power", "cost"],
                "breakpoint": [0, 1],
            },
        )

        m.add_disjunctive_piecewise_constraints(
            {"power": power, "cost": cost},
            breakpoints,
            link_dim="var",
        )

        # Minimize cost
        m.add_objective(cost)

        status, cond = self._solve(m, solver_name)

        assert status == "ok"
        # Should pick region 0, minimum cost = 0
        assert np.isclose(cost.solution.values, 0.0, atol=1e-5)
        assert np.isclose(power.solution.values, 0.0, atol=1e-5)

    def test_three_segments_min(self, solver_name: str) -> None:
        """Test 3 segments, minimize picks lowest."""
        m = Model()
        x = m.add_variables(name="x")

        # Three segments: [0, 10], [30, 50], [80, 100]
        breakpoints = xr.DataArray(
            [[0.0, 10.0], [30.0, 50.0], [80.0, 100.0]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1, 2], "breakpoint": [0, 1]},
        )

        m.add_disjunctive_piecewise_constraints(x, breakpoints)
        m.add_objective(x)

        status, cond = self._solve(m, solver_name)

        assert status == "ok"
        assert np.isclose(x.solution.values, 0.0, atol=1e-5)

    def test_constrained_mid_segment(self, solver_name: str) -> None:
        """Test constraint forcing x into middle of a segment, verify interpolation."""
        m = Model()
        x = m.add_variables(name="x")

        # Two segments: [0, 10] and [50, 100]
        breakpoints = xr.DataArray(
            [[0.0, 10.0], [50.0, 100.0]],
            dims=["segment", "breakpoint"],
            coords={"segment": [0, 1], "breakpoint": [0, 1]},
        )

        m.add_disjunctive_piecewise_constraints(x, breakpoints)

        # Force x >= 60, so must be in segment 1
        m.add_constraints(x >= 60, name="x_lower")
        m.add_objective(x)

        status, cond = self._solve(m, solver_name)

        assert status == "ok"
        # Minimum in segment 1 with x >= 60 → x = 60
        assert np.isclose(x.solution.values, 60.0, atol=1e-5)

    def test_multi_breakpoint_segment_solver(self, solver_name: str) -> None:
        """Test segment with 3 breakpoints, verify correct interpolated value."""
        m = Model()
        power = m.add_variables(name="power")
        cost = m.add_variables(name="cost")

        # Both segments have 3 breakpoints (no NaN padding needed)
        # Segment 0: 3-breakpoint curve (power [0,50,100], cost [0,10,50])
        # Segment 1: 3-breakpoint curve (power [200,250,300], cost [80,90,100])
        breakpoints = xr.DataArray(
            [
                [[0.0, 50.0, 100.0], [0.0, 10.0, 50.0]],
                [[200.0, 250.0, 300.0], [80.0, 90.0, 100.0]],
            ],
            dims=["segment", "var", "breakpoint"],
            coords={
                "segment": [0, 1],
                "var": ["power", "cost"],
                "breakpoint": [0, 1, 2],
            },
        )

        m.add_disjunctive_piecewise_constraints(
            {"power": power, "cost": cost},
            breakpoints,
            link_dim="var",
        )

        # Constraint: power >= 50, minimize cost → picks segment 0, power=50, cost=10
        m.add_constraints(power >= 50, name="power_min")
        m.add_constraints(power <= 150, name="power_max")
        m.add_objective(cost)

        status, cond = self._solve(m, solver_name)

        assert status == "ok"
        assert np.isclose(power.solution.values, 50.0, atol=1e-5)
        assert np.isclose(cost.solution.values, 10.0, atol=1e-5)

    def test_multi_generator_solver(self, solver_name: str) -> None:
        """Test multiple generators with different disjunctive segments."""
        m = Model()
        generators = pd.Index(["gen1", "gen2"], name="generator")
        power = m.add_variables(lower=0, coords=[generators], name="power")
        cost = m.add_variables(coords=[generators], name="cost")

        # gen1: two operating regions
        #   Region 0: power [0,50], cost [0,15]
        #   Region 1: power [80,100], cost [30,50]
        # gen2: two operating regions
        #   Region 0: power [0,60], cost [0,10]
        #   Region 1: power [70,100], cost [12,40]
        breakpoints = xr.DataArray(
            [
                [[[0.0, 50.0], [0.0, 15.0]], [[80.0, 100.0], [30.0, 50.0]]],
                [[[0.0, 60.0], [0.0, 10.0]], [[70.0, 100.0], [12.0, 40.0]]],
            ],
            dims=["generator", "segment", "var", "breakpoint"],
            coords={
                "generator": generators,
                "segment": [0, 1],
                "var": ["power", "cost"],
                "breakpoint": [0, 1],
            },
        )

        m.add_disjunctive_piecewise_constraints(
            {"power": power, "cost": cost},
            breakpoints,
            link_dim="var",
        )

        # Total power demand >= 100
        m.add_constraints(power.sum() >= 100, name="demand")
        m.add_objective(cost.sum())

        status, cond = self._solve(m, solver_name)

        assert status == "ok"
        total_power = power.solution.sum().values
        assert total_power >= 100 - 1e-5
