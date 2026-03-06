"""Tests for the new piecewise linear constraints API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import (
    Model,
    available_solvers,
    breakpoints,
    piecewise,
    segments,
    slopes_to_points,
)
from linopy.constants import (
    PWL_AUX_SUFFIX,
    PWL_BINARY_SUFFIX,
    PWL_CONVEX_SUFFIX,
    PWL_DELTA_SUFFIX,
    PWL_FILL_SUFFIX,
    PWL_LAMBDA_SUFFIX,
    PWL_LP_SUFFIX,
    PWL_SELECT_SUFFIX,
    PWL_X_LINK_SUFFIX,
    PWL_Y_LINK_SUFFIX,
)
from linopy.piecewise import PiecewiseConstraintDescriptor, PiecewiseExpression
from linopy.solver_capabilities import SolverFeature, get_available_solvers_with_feature

_sos2_solvers = get_available_solvers_with_feature(
    SolverFeature.SOS_CONSTRAINTS, available_solvers
)
_any_solvers = [
    s for s in ["highs", "gurobi", "glpk", "cplex"] if s in available_solvers
]


# ===========================================================================
# slopes_to_points
# ===========================================================================


class TestSlopesToPoints:
    def test_basic(self) -> None:
        assert slopes_to_points([0, 1, 2], [1, 2], 0) == [0, 1, 3]

    def test_negative_slopes(self) -> None:
        result = slopes_to_points([0, 10, 20], [-0.5, -1.0], 10)
        assert result == [10, 5, -5]

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="len\\(slopes\\)"):
            slopes_to_points([0, 1, 2], [1], 0)


# ===========================================================================
# breakpoints() factory
# ===========================================================================


class TestBreakpointsFactory:
    def test_list(self) -> None:
        bp = breakpoints([0, 50, 100])
        assert bp.dims == ("breakpoint",)
        assert list(bp.values) == [0.0, 50.0, 100.0]

    def test_dict(self) -> None:
        bp = breakpoints({"gen1": [0, 50, 100], "gen2": [0, 30]}, dim="generator")
        assert set(bp.dims) == {"generator", "breakpoint"}
        assert bp.sizes["breakpoint"] == 3
        assert np.isnan(bp.sel(generator="gen2", breakpoint=2))

    def test_dict_without_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="'dim' is required"):
            breakpoints({"a": [0, 50], "b": [0, 30]})

    def test_slopes_list(self) -> None:
        bp = breakpoints(slopes=[1, 2], x_points=[0, 1, 2], y0=0)
        expected = breakpoints([0, 1, 3])
        xr.testing.assert_equal(bp, expected)

    def test_slopes_dict(self) -> None:
        bp = breakpoints(
            slopes={"a": [1, 0.5], "b": [2, 1]},
            x_points={"a": [0, 10, 50], "b": [0, 20, 80]},
            y0={"a": 0, "b": 10},
            dim="gen",
        )
        assert set(bp.dims) == {"gen", "breakpoint"}
        # a: [0, 10, 30], b: [10, 50, 110]
        np.testing.assert_allclose(bp.sel(gen="a").values, [0, 10, 30])
        np.testing.assert_allclose(bp.sel(gen="b").values, [10, 50, 110])

    def test_slopes_dict_shared_xpoints(self) -> None:
        bp = breakpoints(
            slopes={"a": [1, 2], "b": [3, 4]},
            x_points=[0, 1, 2],
            y0={"a": 0, "b": 0},
            dim="gen",
        )
        np.testing.assert_allclose(bp.sel(gen="a").values, [0, 1, 3])
        np.testing.assert_allclose(bp.sel(gen="b").values, [0, 3, 7])

    def test_slopes_dict_shared_y0(self) -> None:
        bp = breakpoints(
            slopes={"a": [1, 2], "b": [3, 4]},
            x_points={"a": [0, 1, 2], "b": [0, 1, 2]},
            y0=5.0,
            dim="gen",
        )
        np.testing.assert_allclose(bp.sel(gen="a").values, [5, 6, 8])

    def test_values_and_slopes_raises(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            breakpoints([0, 1], slopes=[1], x_points=[0, 1], y0=0)

    def test_slopes_without_xpoints_raises(self) -> None:
        with pytest.raises(ValueError, match="requires both"):
            breakpoints(slopes=[1], y0=0)

    def test_slopes_without_y0_raises(self) -> None:
        with pytest.raises(ValueError, match="requires both"):
            breakpoints(slopes=[1], x_points=[0, 1])

    def test_xpoints_with_values_raises(self) -> None:
        with pytest.raises(ValueError, match="forbidden"):
            breakpoints([0, 1], x_points=[0, 1])

    def test_y0_with_values_raises(self) -> None:
        with pytest.raises(ValueError, match="forbidden"):
            breakpoints([0, 1], y0=5)


# ===========================================================================
# segments() factory
# ===========================================================================


class TestSegmentsFactory:
    def test_list(self) -> None:
        bp = segments([[0, 10], [50, 100]])
        assert set(bp.dims) == {"segment", "breakpoint"}
        assert bp.sizes["segment"] == 2
        assert bp.sizes["breakpoint"] == 2

    def test_dict(self) -> None:
        bp = segments(
            {"a": [[0, 10], [50, 100]], "b": [[0, 20], [60, 90]]},
            dim="gen",
        )
        assert "gen" in bp.dims
        assert "segment" in bp.dims
        assert "breakpoint" in bp.dims

    def test_ragged(self) -> None:
        bp = segments([[0, 5, 10], [50, 100]])
        assert bp.sizes["breakpoint"] == 3
        assert np.isnan(bp.sel(segment=1, breakpoint=2))

    def test_dict_without_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="'dim' is required"):
            segments({"a": [[0, 10]], "b": [[50, 100]]})


# ===========================================================================
# piecewise() and operator overloading
# ===========================================================================


class TestPiecewiseFunction:
    def test_returns_expression(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        pw = piecewise(x, x_points=[0, 10, 50], y_points=[5, 2, 20])
        assert isinstance(pw, PiecewiseExpression)

    def test_eq_returns_descriptor(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        desc = piecewise(x, [0, 10, 50], [5, 2, 20]) == y
        assert isinstance(desc, PiecewiseConstraintDescriptor)
        assert desc.sign == "=="

    def test_ge_returns_le_descriptor(self) -> None:
        """Pw >= y means y <= pw"""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        desc = piecewise(x, [0, 10, 50], [5, 2, 20]) >= y
        assert isinstance(desc, PiecewiseConstraintDescriptor)
        assert desc.sign == "<="

    def test_le_returns_ge_descriptor(self) -> None:
        """Pw <= y means y >= pw"""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        desc = piecewise(x, [0, 10, 50], [5, 2, 20]) <= y
        assert isinstance(desc, PiecewiseConstraintDescriptor)
        assert desc.sign == ">="

    def test_mismatched_sizes_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        with pytest.raises(ValueError, match="same size"):
            piecewise(x, [0, 10, 50, 100], [5, 2, 20])

    def test_wrong_dim_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        xp = xr.DataArray([0, 10, 50], dims=["wrong"])
        yp = xr.DataArray([5, 2, 20], dims=["wrong"])
        with pytest.raises(ValueError, match="must have dimension.*breakpoint"):
            piecewise(x, xp, yp)

    def test_segment_dim_mismatch_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        xp = segments([[0, 10], [50, 100]])
        yp = xr.DataArray([0, 5], dims=["breakpoint"])
        with pytest.raises(ValueError, match="segment.*dimension.*both must"):
            piecewise(x, xp, yp)

    def test_detects_disjunctive(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        pw = piecewise(x, segments([[0, 10], [50, 100]]), segments([[0, 5], [20, 80]]))
        assert pw.disjunctive is True

    def test_detects_continuous(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        pw = piecewise(x, [0, 10, 50], [5, 2, 20])
        assert pw.disjunctive is False


# ===========================================================================
# Continuous piecewise – equality
# ===========================================================================


class TestContinuousEquality:
    def test_sos2(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(x, [0, 10, 50, 100], [5, 2, 20, 80]) == y,
            method="sos2",
        )
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_CONVEX_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_X_LINK_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_Y_LINK_SUFFIX}" in m.constraints
        lam = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        assert lam.attrs.get("sos_type") == 2

    def test_auto_selects_incremental_for_monotonic(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(x, [0, 10, 50, 100], [5, 2, 20, 80]) == y,
        )
        assert f"pwl0{PWL_DELTA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" not in m.variables

    def test_auto_selects_sos2_for_nonmonotonic(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 30, 100], [5, 20, 15, 80]) == y,
        )
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_DELTA_SUFFIX}" not in m.variables

    def test_multi_dimensional(self) -> None:
        m = Model()
        gens = pd.Index(["gen_a", "gen_b"], name="generator")
        x = m.add_variables(coords=[gens], name="x")
        y = m.add_variables(coords=[gens], name="y")
        m.add_piecewise_constraints(
            piecewise(
                x,
                breakpoints(
                    {"gen_a": [0, 10, 50], "gen_b": [0, 20, 80]}, dim="generator"
                ),
                breakpoints(
                    {"gen_a": [0, 5, 30], "gen_b": [0, 8, 50]}, dim="generator"
                ),
            )
            == y,
        )
        delta = m.variables[f"pwl0{PWL_DELTA_SUFFIX}"]
        assert "generator" in delta.dims

    def test_with_slopes(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(
                x,
                [0, 10, 50, 100],
                breakpoints(slopes=[-0.3, 0.45, 1.2], x_points=[0, 10, 50, 100], y0=5),
            )
            == y,
        )
        assert f"pwl0{PWL_DELTA_SUFFIX}" in m.variables


# ===========================================================================
# Continuous piecewise – inequality
# ===========================================================================


class TestContinuousInequality:
    def test_concave_le_uses_lp(self) -> None:
        """Y <= concave f(x) → LP tangent lines"""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        # Concave: slopes 0.8, 0.4 (decreasing)
        # pw >= y means y <= pw (sign="<=")
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 100], [0, 40, 60]) >= y,
        )
        assert f"pwl0{PWL_LP_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" not in m.variables
        assert f"pwl0{PWL_AUX_SUFFIX}" not in m.variables

    def test_convex_le_uses_sos2_aux(self) -> None:
        """Y <= convex f(x) → SOS2 + aux"""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        # Convex: slopes 0.2, 1.0 (increasing)
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 100], [0, 10, 60]) >= y,
        )
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_AUX_SUFFIX}" in m.variables

    def test_convex_ge_uses_lp(self) -> None:
        """Y >= convex f(x) → LP tangent lines"""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        # Convex: slopes 0.2, 1.0 (increasing)
        # pw <= y means y >= pw (sign=">=")
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 100], [0, 10, 60]) <= y,
        )
        assert f"pwl0{PWL_LP_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" not in m.variables
        assert f"pwl0{PWL_AUX_SUFFIX}" not in m.variables

    def test_concave_ge_uses_sos2_aux(self) -> None:
        """Y >= concave f(x) → SOS2 + aux"""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        # Concave: slopes 0.8, 0.4 (decreasing)
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 100], [0, 40, 60]) <= y,
        )
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_AUX_SUFFIX}" in m.variables

    def test_mixed_uses_sos2(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        # Mixed: slopes 0.5, 0.3, 0.9 (down then up)
        m.add_piecewise_constraints(
            piecewise(x, [0, 30, 60, 100], [0, 15, 24, 60]) >= y,
        )
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_AUX_SUFFIX}" in m.variables

    def test_method_lp_wrong_convexity_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        # Convex function + y <= pw + method="lp" should fail
        with pytest.raises(ValueError, match="convex"):
            m.add_piecewise_constraints(
                piecewise(x, [0, 50, 100], [0, 10, 60]) >= y,
                method="lp",
            )

    def test_method_lp_equality_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        with pytest.raises(ValueError, match="equality"):
            m.add_piecewise_constraints(
                piecewise(x, [0, 50, 100], [0, 40, 60]) == y,
                method="lp",
            )


# ===========================================================================
# Incremental formulation
# ===========================================================================


class TestIncremental:
    def test_creates_delta_vars(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(x, [0, 10, 50, 100], [5, 2, 20, 80]) == y,
            method="incremental",
        )
        assert f"pwl0{PWL_DELTA_SUFFIX}" in m.variables
        delta = m.variables[f"pwl0{PWL_DELTA_SUFFIX}"]
        assert delta.labels.sizes["breakpoint_seg"] == 3
        assert f"pwl0{PWL_FILL_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" not in m.variables

    def test_nonmonotonic_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        with pytest.raises(ValueError, match="strictly monotonic"):
            m.add_piecewise_constraints(
                piecewise(x, [0, 50, 30, 100], [5, 20, 15, 80]) == y,
                method="incremental",
            )

    def test_two_breakpoints_no_fill(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(x, [0, 100], [5, 80]) == y,
            method="incremental",
        )
        delta = m.variables[f"pwl0{PWL_DELTA_SUFFIX}"]
        assert delta.labels.sizes["breakpoint_seg"] == 1
        assert f"pwl0{PWL_X_LINK_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_Y_LINK_SUFFIX}" in m.constraints

    def test_decreasing_monotonic(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(x, [100, 50, 10, 0], [80, 20, 2, 5]) == y,
            method="incremental",
        )
        assert f"pwl0{PWL_DELTA_SUFFIX}" in m.variables


# ===========================================================================
# Disjunctive piecewise
# ===========================================================================


class TestDisjunctive:
    def test_equality_creates_binary(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(x, segments([[0, 10], [50, 100]]), segments([[0, 5], [20, 80]]))
            == y,
        )
        assert f"pwl0{PWL_BINARY_SUFFIX}" in m.variables
        assert f"pwl0{PWL_SELECT_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_CONVEX_SUFFIX}" in m.constraints
        lam = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        assert lam.attrs.get("sos_type") == 2

    def test_inequality_creates_aux(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(x, segments([[0, 10], [50, 100]]), segments([[0, 5], [20, 80]]))
            >= y,
        )
        assert f"pwl0{PWL_AUX_SUFFIX}" in m.variables
        assert f"pwl0{PWL_BINARY_SUFFIX}" in m.variables
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables

    def test_method_lp_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        with pytest.raises(ValueError, match="disjunctive"):
            m.add_piecewise_constraints(
                piecewise(
                    x, segments([[0, 10], [50, 100]]), segments([[0, 5], [20, 80]])
                )
                >= y,
                method="lp",
            )

    def test_method_incremental_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        with pytest.raises(ValueError, match="disjunctive"):
            m.add_piecewise_constraints(
                piecewise(
                    x, segments([[0, 10], [50, 100]]), segments([[0, 5], [20, 80]])
                )
                == y,
                method="incremental",
            )

    def test_multi_dimensional(self) -> None:
        m = Model()
        gens = pd.Index(["gen_a", "gen_b"], name="generator")
        x = m.add_variables(coords=[gens], name="x")
        y = m.add_variables(coords=[gens], name="y")
        m.add_piecewise_constraints(
            piecewise(
                x,
                segments(
                    {"gen_a": [[0, 10], [50, 100]], "gen_b": [[0, 20], [60, 90]]},
                    dim="generator",
                ),
                segments(
                    {"gen_a": [[0, 5], [20, 80]], "gen_b": [[0, 8], [30, 70]]},
                    dim="generator",
                ),
            )
            == y,
        )
        binary = m.variables[f"pwl0{PWL_BINARY_SUFFIX}"]
        lam = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        assert "generator" in binary.dims
        assert "generator" in lam.dims


# ===========================================================================
# Validation
# ===========================================================================


class TestValidation:
    def test_non_descriptor_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        with pytest.raises(TypeError, match="PiecewiseConstraintDescriptor"):
            m.add_piecewise_constraints(x)  # type: ignore

    def test_invalid_method_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        with pytest.raises(ValueError, match="method must be"):
            m.add_piecewise_constraints(
                piecewise(x, [0, 10, 50], [5, 2, 20]) == y,
                method="invalid",  # type: ignore
            )


# ===========================================================================
# Name generation
# ===========================================================================


class TestNameGeneration:
    def test_auto_name(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        z = m.add_variables(name="z")
        m.add_piecewise_constraints(piecewise(x, [0, 10, 50], [5, 2, 20]) == y)
        m.add_piecewise_constraints(piecewise(x, [0, 20, 80], [10, 15, 50]) == z)
        assert f"pwl0{PWL_DELTA_SUFFIX}" in m.variables
        assert f"pwl1{PWL_DELTA_SUFFIX}" in m.variables

    def test_custom_name(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(x, [0, 10, 50], [5, 2, 20]) == y,
            name="my_pwl",
        )
        assert f"my_pwl{PWL_DELTA_SUFFIX}" in m.variables
        assert f"my_pwl{PWL_X_LINK_SUFFIX}" in m.constraints
        assert f"my_pwl{PWL_Y_LINK_SUFFIX}" in m.constraints


# ===========================================================================
# Broadcasting
# ===========================================================================


class TestBroadcasting:
    def test_broadcast_over_extra_dims(self) -> None:
        m = Model()
        gens = pd.Index(["gen_a", "gen_b"], name="generator")
        times = pd.Index([0, 1, 2], name="time")
        x = m.add_variables(coords=[gens, times], name="x")
        y = m.add_variables(coords=[gens, times], name="y")
        # Points only have generator dim → broadcast over time
        m.add_piecewise_constraints(
            piecewise(
                x,
                breakpoints(
                    {"gen_a": [0, 10, 50], "gen_b": [0, 20, 80]}, dim="generator"
                ),
                breakpoints(
                    {"gen_a": [0, 5, 30], "gen_b": [0, 8, 50]}, dim="generator"
                ),
            )
            == y,
        )
        delta = m.variables[f"pwl0{PWL_DELTA_SUFFIX}"]
        assert "generator" in delta.dims
        assert "time" in delta.dims


# ===========================================================================
# NaN masking
# ===========================================================================


class TestNaNMasking:
    def test_nan_masks_lambda_labels(self) -> None:
        """NaN in y_points produces masked labels in SOS2 formulation."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        x_pts = xr.DataArray([0, 10, 50, np.nan], dims=["breakpoint"])
        y_pts = xr.DataArray([0, 5, 20, np.nan], dims=["breakpoint"])
        m.add_piecewise_constraints(
            piecewise(x, x_pts, y_pts) == y,
            method="sos2",
        )
        lam = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        # First 3 should be valid, last masked
        assert (lam.labels.isel(breakpoint=slice(None, 3)) != -1).all()
        assert int(lam.labels.isel(breakpoint=3)) == -1

    def test_skip_nan_check_no_masking(self) -> None:
        """skip_nan_check=True treats all breakpoints as valid."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        x_pts = xr.DataArray([0, 10, 50, np.nan], dims=["breakpoint"])
        y_pts = xr.DataArray([0, 5, 20, np.nan], dims=["breakpoint"])
        m.add_piecewise_constraints(
            piecewise(x, x_pts, y_pts) == y,
            method="sos2",
            skip_nan_check=True,
        )
        lam = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        assert (lam.labels != -1).all()


# ===========================================================================
# Convexity detection edge cases
# ===========================================================================


class TestConvexityDetection:
    def test_linear_uses_lp_both_directions(self) -> None:
        """Linear function uses LP for both <= and >= inequalities."""
        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        y1 = m.add_variables(name="y1")
        y2 = m.add_variables(name="y2")
        # y1 >= f(x) → LP
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 100], [0, 25, 50]) <= y1,
        )
        assert f"pwl0{PWL_LP_SUFFIX}" in m.constraints
        # y2 <= f(x) → also LP (linear is both convex and concave)
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 100], [0, 25, 50]) >= y2,
        )
        assert f"pwl1{PWL_LP_SUFFIX}" in m.constraints

    def test_single_segment_uses_lp(self) -> None:
        """A single segment (2 breakpoints) is linear; uses LP."""
        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(x, [0, 100], [0, 50]) <= y,
        )
        assert f"pwl0{PWL_LP_SUFFIX}" in m.constraints

    def test_mixed_convexity_uses_sos2(self) -> None:
        """Mixed convexity should fall back to SOS2 for inequalities."""
        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        y = m.add_variables(name="y")
        # Mixed: slope goes up then down → neither convex nor concave
        # y <= f(x) → piecewise >= y → sign="<=" internally
        m.add_piecewise_constraints(
            piecewise(x, [0, 30, 60, 100], [0, 40, 30, 50]) >= y,
        )
        assert f"pwl0{PWL_AUX_SUFFIX}" in m.variables
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables


# ===========================================================================
# LP file output
# ===========================================================================


class TestLPFileOutput:
    def test_sos2_equality(self, tmp_path: Path) -> None:
        m = Model()
        x = m.add_variables(name="x", lower=0, upper=100)
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(x, [0.0, 10.0, 50.0, 100.0], [5.0, 2.0, 20.0, 80.0]) == y,
            method="sos2",
        )
        m.add_objective(y)
        fn = tmp_path / "pwl_eq.lp"
        m.to_file(fn, io_api="lp")
        content = fn.read_text().lower()
        assert "sos" in content
        assert "s2" in content

    def test_lp_formulation_no_sos2(self, tmp_path: Path) -> None:
        m = Model()
        x = m.add_variables(name="x", lower=0, upper=100)
        y = m.add_variables(name="y")
        # Concave: pw >= y uses LP
        m.add_piecewise_constraints(
            piecewise(x, [0.0, 50.0, 100.0], [0.0, 40.0, 60.0]) >= y,
        )
        m.add_objective(y)
        fn = tmp_path / "pwl_lp.lp"
        m.to_file(fn, io_api="lp")
        content = fn.read_text().lower()
        assert "s2" not in content

    def test_disjunctive_sos2_and_binary(self, tmp_path: Path) -> None:
        m = Model()
        x = m.add_variables(name="x", lower=0, upper=100)
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(
                x,
                segments([[0.0, 10.0], [50.0, 100.0]]),
                segments([[0.0, 5.0], [20.0, 80.0]]),
            )
            == y,
        )
        m.add_objective(y)
        fn = tmp_path / "pwl_disj.lp"
        m.to_file(fn, io_api="lp")
        content = fn.read_text().lower()
        assert "s2" in content
        assert "binary" in content or "binaries" in content


# ===========================================================================
# Solver integration – SOS2 capable
# ===========================================================================


@pytest.mark.skipif(len(_sos2_solvers) == 0, reason="No solver with SOS2 support")
class TestSolverSOS2:
    @pytest.fixture(params=_sos2_solvers)
    def solver_name(self, request: pytest.FixtureRequest) -> str:
        return request.param  # type: ignore

    def test_equality_minimize_cost(self, solver_name: str) -> None:
        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        cost = m.add_variables(name="cost")
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 100], [0, 10, 50]) == cost,
        )
        m.add_constraints(x >= 50, name="x_min")
        m.add_objective(cost)
        status, _ = m.solve(solver_name=solver_name)
        assert status == "ok"
        np.testing.assert_allclose(x.solution.values, 50, atol=1e-4)
        np.testing.assert_allclose(cost.solution.values, 10, atol=1e-4)

    def test_equality_maximize_efficiency(self, solver_name: str) -> None:
        m = Model()
        power = m.add_variables(lower=0, upper=100, name="power")
        eff = m.add_variables(name="eff")
        m.add_piecewise_constraints(
            piecewise(power, [0, 25, 50, 75, 100], [0.7, 0.85, 0.95, 0.9, 0.8]) == eff,
        )
        m.add_objective(eff, sense="max")
        status, _ = m.solve(solver_name=solver_name)
        assert status == "ok"
        np.testing.assert_allclose(power.solution.values, 50, atol=1e-4)
        np.testing.assert_allclose(eff.solution.values, 0.95, atol=1e-4)

    def test_disjunctive_solve(self, solver_name: str) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(
                x,
                segments([[0.0, 10.0], [50.0, 100.0]]),
                segments([[0.0, 5.0], [20.0, 80.0]]),
            )
            == y,
        )
        m.add_constraints(x >= 60, name="x_min")
        m.add_objective(y)
        status, _ = m.solve(solver_name=solver_name)
        assert status == "ok"
        # x=60 on second segment: y = 20 + (80-20)/(100-50)*(60-50) = 32
        np.testing.assert_allclose(float(x.solution.values), 60, atol=1e-4)
        np.testing.assert_allclose(float(y.solution.values), 32, atol=1e-4)


# ===========================================================================
# Solver integration – LP formulation (any solver)
# ===========================================================================


@pytest.mark.skipif(len(_any_solvers) == 0, reason="No solver available")
class TestSolverLP:
    @pytest.fixture(params=_any_solvers)
    def solver_name(self, request: pytest.FixtureRequest) -> str:
        return request.param  # type: ignore

    def test_concave_le(self, solver_name: str) -> None:
        """Y <= concave f(x), maximize y"""
        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        y = m.add_variables(name="y")
        # Concave: [0,0],[50,40],[100,60]
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 100], [0, 40, 60]) >= y,
        )
        m.add_constraints(x <= 75, name="x_max")
        m.add_objective(y, sense="max")
        status, _ = m.solve(solver_name=solver_name)
        assert status == "ok"
        # At x=75: y = 40 + 0.4*(75-50) = 50
        np.testing.assert_allclose(float(x.solution.values), 75, atol=1e-4)
        np.testing.assert_allclose(float(y.solution.values), 50, atol=1e-4)

    def test_convex_ge(self, solver_name: str) -> None:
        """Y >= convex f(x), minimize y"""
        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        y = m.add_variables(name="y")
        # Convex: [0,0],[50,10],[100,60]
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 100], [0, 10, 60]) <= y,
        )
        m.add_constraints(x >= 25, name="x_min")
        m.add_objective(y)
        status, _ = m.solve(solver_name=solver_name)
        assert status == "ok"
        # At x=25: y = 0.2*25 = 5
        np.testing.assert_allclose(float(x.solution.values), 25, atol=1e-4)
        np.testing.assert_allclose(float(y.solution.values), 5, atol=1e-4)

    def test_slopes_equivalence(self, solver_name: str) -> None:
        """Same model with y_points vs slopes produces identical solutions."""
        # Model 1: direct y_points
        m1 = Model()
        x1 = m1.add_variables(lower=0, upper=100, name="x")
        y1 = m1.add_variables(name="y")
        m1.add_piecewise_constraints(
            piecewise(x1, [0, 50, 100], [0, 40, 60]) >= y1,
        )
        m1.add_constraints(x1 <= 75, name="x_max")
        m1.add_objective(y1, sense="max")
        s1, _ = m1.solve(solver_name=solver_name)

        # Model 2: slopes
        m2 = Model()
        x2 = m2.add_variables(lower=0, upper=100, name="x")
        y2 = m2.add_variables(name="y")
        m2.add_piecewise_constraints(
            piecewise(
                x2,
                [0, 50, 100],
                breakpoints(slopes=[0.8, 0.4], x_points=[0, 50, 100], y0=0),
            )
            >= y2,
        )
        m2.add_constraints(x2 <= 75, name="x_max")
        m2.add_objective(y2, sense="max")
        s2, _ = m2.solve(solver_name=solver_name)

        assert s1 == "ok"
        assert s2 == "ok"
        np.testing.assert_allclose(
            float(y1.solution.values), float(y2.solution.values), atol=1e-4
        )
