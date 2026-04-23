"""Tests for the new piecewise linear constraints API."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import (
    Model,
    available_solvers,
    breakpoints,
    segments,
    slopes_to_points,
    tangent_lines,
)
from linopy.constants import (
    BREAKPOINT_DIM,
    LP_SEG_DIM,
    PWL_ACTIVE_BOUND_SUFFIX,
    PWL_BINARY_ORDER_SUFFIX,
    PWL_CHORD_SUFFIX,
    PWL_CONVEX_SUFFIX,
    PWL_DELTA_BOUND_SUFFIX,
    PWL_DELTA_SUFFIX,
    PWL_DOMAIN_HI_SUFFIX,
    PWL_DOMAIN_LO_SUFFIX,
    PWL_FILL_ORDER_SUFFIX,
    PWL_LAMBDA_SUFFIX,
    PWL_LINK_SUFFIX,
    PWL_ORDER_BINARY_SUFFIX,
    PWL_OUTPUT_LINK_SUFFIX,
    PWL_SEGMENT_BINARY_SUFFIX,
    PWL_SELECT_SUFFIX,
    SEGMENT_DIM,
)
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
        assert bp.dims == (BREAKPOINT_DIM,)
        assert list(bp.values) == [0.0, 50.0, 100.0]

    def test_dict(self) -> None:
        bp = breakpoints({"gen1": [0, 50, 100], "gen2": [0, 30]}, dim="generator")
        assert set(bp.dims) == {"generator", BREAKPOINT_DIM}
        assert bp.sizes[BREAKPOINT_DIM] == 3
        assert np.isnan(bp.sel(generator="gen2").sel({BREAKPOINT_DIM: 2}))

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
        assert set(bp.dims) == {"gen", BREAKPOINT_DIM}
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

    # --- pandas and xarray inputs ---

    def test_series(self) -> None:
        bp = breakpoints(pd.Series([0, 50, 100]))
        assert bp.dims == (BREAKPOINT_DIM,)
        assert list(bp.values) == [0.0, 50.0, 100.0]

    def test_dataframe(self) -> None:
        df = pd.DataFrame(
            {"gen1": [0, 50, 100], "gen2": [0, 30, np.nan]}
        ).T  # rows=entities, cols=breakpoints
        bp = breakpoints(df, dim="generator")
        assert set(bp.dims) == {"generator", BREAKPOINT_DIM}
        assert bp.sizes[BREAKPOINT_DIM] == 3
        np.testing.assert_allclose(bp.sel(generator="gen1").values, [0, 50, 100])
        assert np.isnan(bp.sel(generator="gen2").values[2])

    def test_dataframe_without_dim_raises(self) -> None:
        df = pd.DataFrame({"a": [0, 50], "b": [0, 30]}).T
        with pytest.raises(ValueError, match="'dim' is required"):
            breakpoints(df)

    def test_dataarray_passthrough(self) -> None:
        da = xr.DataArray(
            [0, 50, 100],
            dims=[BREAKPOINT_DIM],
            coords={BREAKPOINT_DIM: np.arange(3)},
        )
        bp = breakpoints(da)
        xr.testing.assert_equal(bp, da)

    def test_dataarray_missing_dim_raises(self) -> None:
        da = xr.DataArray([0, 50, 100], dims=["foo"])
        with pytest.raises(ValueError, match="must have a"):
            breakpoints(da)

    def test_slopes_series(self) -> None:
        bp = breakpoints(
            slopes=pd.Series([1, 2]),
            x_points=pd.Series([0, 1, 2]),
            y0=0,
        )
        expected = breakpoints([0, 1, 3])
        xr.testing.assert_equal(bp, expected)

    def test_slopes_dataarray(self) -> None:
        slopes_da = xr.DataArray(
            [[1, 2], [3, 4]],
            dims=["gen", BREAKPOINT_DIM],
            coords={"gen": ["a", "b"], BREAKPOINT_DIM: [0, 1]},
        )
        xp_da = xr.DataArray(
            [[0, 1, 2], [0, 1, 2]],
            dims=["gen", BREAKPOINT_DIM],
            coords={"gen": ["a", "b"], BREAKPOINT_DIM: [0, 1, 2]},
        )
        y0_da = xr.DataArray([0, 5], dims=["gen"], coords={"gen": ["a", "b"]})
        bp = breakpoints(slopes=slopes_da, x_points=xp_da, y0=y0_da, dim="gen")
        np.testing.assert_allclose(bp.sel(gen="a").values, [0, 1, 3])
        np.testing.assert_allclose(bp.sel(gen="b").values, [5, 8, 12])

    def test_slopes_dataframe(self) -> None:
        slopes_df = pd.DataFrame({"a": [1, 0.5], "b": [2, 1]}).T
        xp_df = pd.DataFrame({"a": [0, 10, 50], "b": [0, 20, 80]}).T
        y0_series = pd.Series({"a": 0, "b": 10})
        bp = breakpoints(slopes=slopes_df, x_points=xp_df, y0=y0_series, dim="gen")
        np.testing.assert_allclose(bp.sel(gen="a").values, [0, 10, 30])
        np.testing.assert_allclose(bp.sel(gen="b").values, [10, 50, 110])


# ===========================================================================
# segments() factory
# ===========================================================================


class TestSegmentsFactory:
    def test_list(self) -> None:
        bp = segments([[0, 10], [50, 100]])
        assert set(bp.dims) == {SEGMENT_DIM, BREAKPOINT_DIM}
        assert bp.sizes[SEGMENT_DIM] == 2
        assert bp.sizes[BREAKPOINT_DIM] == 2

    def test_dict(self) -> None:
        bp = segments(
            {"a": [[0, 10], [50, 100]], "b": [[0, 20], [60, 90]]},
            dim="gen",
        )
        assert "gen" in bp.dims
        assert SEGMENT_DIM in bp.dims
        assert BREAKPOINT_DIM in bp.dims

    def test_ragged(self) -> None:
        bp = segments([[0, 5, 10], [50, 100]])
        assert bp.sizes[BREAKPOINT_DIM] == 3
        assert np.isnan(bp.sel({SEGMENT_DIM: 1, BREAKPOINT_DIM: 2}))

    def test_dict_without_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="'dim' is required"):
            segments({"a": [[0, 10]], "b": [[50, 100]]})

    def test_dataframe(self) -> None:
        df = pd.DataFrame([[0, 10], [50, 100]])  # rows=segments, cols=breakpoints
        bp = segments(df)
        assert set(bp.dims) == {SEGMENT_DIM, BREAKPOINT_DIM}
        assert bp.sizes[SEGMENT_DIM] == 2
        assert bp.sizes[BREAKPOINT_DIM] == 2
        np.testing.assert_allclose(bp.sel({SEGMENT_DIM: 0}).values, [0, 10])
        np.testing.assert_allclose(bp.sel({SEGMENT_DIM: 1}).values, [50, 100])

    def test_dataarray_passthrough(self) -> None:
        da = xr.DataArray(
            [[0, 10], [50, 100]],
            dims=[SEGMENT_DIM, BREAKPOINT_DIM],
            coords={SEGMENT_DIM: [0, 1], BREAKPOINT_DIM: [0, 1]},
        )
        bp = segments(da)
        xr.testing.assert_equal(bp, da)

    def test_dataarray_missing_dim_raises(self) -> None:
        da_no_seg = xr.DataArray(
            [[0, 10], [50, 100]],
            dims=["foo", BREAKPOINT_DIM],
        )
        with pytest.raises(ValueError, match="must have both"):
            segments(da_no_seg)

        da_no_bp = xr.DataArray(
            [[0, 10], [50, 100]],
            dims=[SEGMENT_DIM, "bar"],
        )
        with pytest.raises(ValueError, match="must have both"):
            segments(da_no_bp)


# ===========================================================================
# Continuous piecewise -- equality
# ===========================================================================


class TestContinuousEquality:
    def test_sos2(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_formulation(
            (x, [0, 10, 50, 100]),
            (y, [5, 2, 20, 80]),
            method="sos2",
        )
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_CONVEX_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_LINK_SUFFIX}" in m.constraints
        # N-var path uses a single stacked link constraint (no separate y_link)
        lam = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        assert lam.attrs.get("sos_type") == 2

    def test_auto_selects_incremental_for_monotonic(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        # Both breakpoint sequences must be monotonic for incremental
        m.add_piecewise_formulation(
            (x, [0, 10, 50, 100]),
            (y, [0, 5, 20, 80]),
        )
        assert f"pwl0{PWL_DELTA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" not in m.variables

    def test_auto_nonmonotonic_falls_back_to_sos2(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        # Non-monotonic y-breakpoints force SOS2
        m.add_piecewise_formulation(
            (x, [0, 50, 30, 100]),
            (y, [5, 20, 15, 80]),
        )
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_DELTA_SUFFIX}" not in m.variables

    def test_multi_dimensional(self) -> None:
        m = Model()
        gens = pd.Index(["gen_a", "gen_b"], name="generator")
        x = m.add_variables(coords=[gens], name="x")
        y = m.add_variables(coords=[gens], name="y")
        m.add_piecewise_formulation(
            (
                x,
                breakpoints(
                    {"gen_a": [0, 10, 50], "gen_b": [0, 20, 80]}, dim="generator"
                ),
            ),
            (
                y,
                breakpoints(
                    {"gen_a": [0, 5, 30], "gen_b": [0, 8, 50]}, dim="generator"
                ),
            ),
        )
        delta = m.variables[f"pwl0{PWL_DELTA_SUFFIX}"]
        assert "generator" in delta.dims

    def test_with_slopes(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        # slopes=[-0.3, 0.45, 1.2] with y0=5 -> y_points=[5, 2, 20, 80]
        # Non-monotonic y-breakpoints, so auto selects SOS2
        m.add_piecewise_formulation(
            (x, [0, 10, 50, 100]),
            (y, breakpoints(slopes=[-0.3, 0.45, 1.2], x_points=[0, 10, 50, 100], y0=5)),
        )
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables


# ===========================================================================
# Piecewise Envelope
# ===========================================================================


class TestTangentLines:
    def test_basic_variable(self) -> None:
        """Envelope from a Variable produces a LinearExpression with seg dim."""
        m = Model()
        x = m.add_variables(name="x", lower=0, upper=100)
        env = tangent_lines(x, [0, 50, 100], [0, 40, 60])
        assert LP_SEG_DIM in env.dims

    def test_basic_linexpr(self) -> None:
        """Envelope from a LinearExpression works too."""
        m = Model()
        x = m.add_variables(name="x", lower=0, upper=100)
        env = tangent_lines(1 * x, [0, 50, 100], [0, 40, 60])
        assert LP_SEG_DIM in env.dims

    def test_segment_count(self) -> None:
        """Number of segments = number of breakpoints - 1."""
        m = Model()
        x = m.add_variables(name="x")
        env = tangent_lines(x, [0, 50, 100], [0, 40, 60])
        assert env.sizes[LP_SEG_DIM] == 2

    def test_invalid_x_type_raises(self) -> None:
        with pytest.raises(TypeError, match="must be a Variable or LinearExpression"):
            tangent_lines(42, [0, 50, 100], [0, 40, 60])  # type: ignore

    def test_concave_le_constraint(self) -> None:
        """Using envelope with <= constraint creates regular constraints."""
        m = Model()
        x = m.add_variables(name="x", lower=0, upper=100)
        y = m.add_variables(name="y")
        env = tangent_lines(x, [0, 50, 100], [0, 40, 60])
        m.add_constraints(y <= env, name="pwl")
        assert "pwl" in m.constraints

    def test_convex_ge_constraint(self) -> None:
        """Using envelope with >= constraint creates regular constraints."""
        m = Model()
        x = m.add_variables(name="x", lower=0, upper=100)
        y = m.add_variables(name="y")
        env = tangent_lines(x, [0, 50, 100], [0, 10, 60])
        m.add_constraints(y >= env, name="pwl")
        assert "pwl" in m.constraints

    def test_dataarray_breakpoints(self) -> None:
        """Envelope accepts DataArray breakpoints."""
        m = Model()
        x = m.add_variables(name="x")
        x_pts = xr.DataArray([0, 50, 100], dims=[BREAKPOINT_DIM])
        y_pts = xr.DataArray([0, 40, 60], dims=[BREAKPOINT_DIM])
        env = tangent_lines(x, x_pts, y_pts)
        assert LP_SEG_DIM in env.dims


# ===========================================================================
# Incremental formulation
# ===========================================================================


class TestIncremental:
    def test_creates_delta_vars(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_formulation(
            (x, [0, 10, 50, 100]),
            (y, [5, 10, 20, 80]),
            method="incremental",
        )
        assert f"pwl0{PWL_DELTA_SUFFIX}" in m.variables
        delta = m.variables[f"pwl0{PWL_DELTA_SUFFIX}"]
        assert delta.labels.sizes[LP_SEG_DIM] == 3
        assert f"pwl0{PWL_FILL_ORDER_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" not in m.variables

    def test_nonmonotonic_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        with pytest.raises(ValueError, match="strictly monotonic"):
            m.add_piecewise_formulation(
                (x, [0, 50, 30, 100]),
                (y, [5, 20, 15, 80]),
                method="incremental",
            )

    def test_sos2_nonmonotonic_succeeds(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_formulation(
            (x, [0, 50, 30, 100]),
            (y, [5, 20, 15, 80]),
            method="sos2",
        )
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_DELTA_SUFFIX}" not in m.variables

    def test_two_breakpoints_no_fill(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_formulation(
            (x, [0, 100]),
            (y, [5, 80]),
            method="incremental",
        )
        delta = m.variables[f"pwl0{PWL_DELTA_SUFFIX}"]
        assert delta.labels.sizes[LP_SEG_DIM] == 1
        assert f"pwl0{PWL_LINK_SUFFIX}" in m.constraints
        # N-var path uses a single stacked link constraint (no separate y_link)

    def test_creates_binary_indicator_vars(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_formulation(
            (x, [0, 10, 50, 100]),
            (y, [5, 10, 20, 80]),
            method="incremental",
        )
        assert f"pwl0{PWL_ORDER_BINARY_SUFFIX}" in m.variables
        binary = m.variables[f"pwl0{PWL_ORDER_BINARY_SUFFIX}"]
        assert binary.labels.sizes[LP_SEG_DIM] == 3
        assert f"pwl0{PWL_DELTA_BOUND_SUFFIX}" in m.constraints

    def test_creates_order_constraints(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_formulation(
            (x, [0, 10, 50, 100]),
            (y, [5, 10, 20, 80]),
            method="incremental",
        )
        assert f"pwl0{PWL_BINARY_ORDER_SUFFIX}" in m.constraints

    def test_two_breakpoints_no_order_constraint(self) -> None:
        """With only one segment, there's no order constraint needed."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_formulation(
            (x, [0, 100]),
            (y, [5, 80]),
            method="incremental",
        )
        assert f"pwl0{PWL_ORDER_BINARY_SUFFIX}" in m.variables
        assert f"pwl0{PWL_DELTA_BOUND_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_BINARY_ORDER_SUFFIX}" not in m.constraints

    def test_decreasing_monotonic(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_formulation(
            (x, [100, 50, 10, 0]),
            (y, [80, 20, 5, 2]),
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
        m.add_piecewise_formulation(
            (x, segments([[0, 10], [50, 100]])),
            (y, segments([[0, 5], [20, 80]])),
        )
        assert f"pwl0{PWL_SEGMENT_BINARY_SUFFIX}" in m.variables
        assert f"pwl0{PWL_SELECT_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_CONVEX_SUFFIX}" in m.constraints
        lam = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        assert lam.attrs.get("sos_type") == 2

    def test_method_incremental_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        with pytest.raises(ValueError, match="disjunctive"):
            m.add_piecewise_formulation(
                (x, segments([[0, 10], [50, 100]])),
                (y, segments([[0, 5], [20, 80]])),
                method="incremental",
            )

    def test_multi_dimensional(self) -> None:
        m = Model()
        gens = pd.Index(["gen_a", "gen_b"], name="generator")
        x = m.add_variables(coords=[gens], name="x")
        y = m.add_variables(coords=[gens], name="y")
        m.add_piecewise_formulation(
            (
                x,
                segments(
                    {"gen_a": [[0, 10], [50, 100]], "gen_b": [[0, 20], [60, 90]]},
                    dim="generator",
                ),
            ),
            (
                y,
                segments(
                    {"gen_a": [[0, 5], [20, 80]], "gen_b": [[0, 8], [30, 70]]},
                    dim="generator",
                ),
            ),
        )
        binary = m.variables[f"pwl0{PWL_SEGMENT_BINARY_SUFFIX}"]
        lam = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        assert "generator" in binary.dims
        assert "generator" in lam.dims

    def test_three_variables(self) -> None:
        """Disjunctive with 3 variables creates single link constraint."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        z = m.add_variables(name="z")
        m.add_piecewise_formulation(
            (x, segments([[0, 10], [50, 100]])),
            (y, segments([[0, 5], [20, 80]])),
            (z, segments([[0, 3], [15, 60]])),
        )
        assert f"pwl0{PWL_SEGMENT_BINARY_SUFFIX}" in m.variables
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        # Single link constraint with _pwl_var dimension
        link = m.constraints[f"pwl0{PWL_LINK_SUFFIX}"]
        assert "_pwl_var" in [str(d) for d in link.dims]

    @pytest.mark.skipif(not _sos2_solvers, reason="no SOS2-capable solver available")
    def test_sign_le_respected_by_solver(self) -> None:
        """
        Disjunctive + sign='<=' must actually bound the solved output
        (not just structurally wire up the output link).
        """
        m = Model()
        x = m.add_variables(lower=0, upper=30, name="x")
        y = m.add_variables(lower=0, upper=40, name="y")
        # Two segments forming a concave profile: (0,0)→(10,20), (10,20)→(20,30)
        m.add_piecewise_formulation(
            (y, segments([[0.0, 20.0], [20.0, 30.0]])),
            (x, segments([[0.0, 10.0], [10.0, 20.0]])),
            sign="<=",
        )
        m.add_constraints(x == 15)
        m.add_objective(-y)  # maximise y
        m.solve()
        # f(15) = 20 + (30-20)*0.5 = 25
        assert m.solution["y"].item() == pytest.approx(25.0, abs=1e-3)


# ===========================================================================
# Validation
# ===========================================================================


class TestValidation:
    def test_wrong_arg_types_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        with pytest.raises(TypeError, match="at least 2"):
            m.add_piecewise_formulation((x, [0, 10, 50]))

    def test_invalid_method_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        with pytest.raises(ValueError, match="method must be"):
            m.add_piecewise_formulation(
                (x, [0, 10, 50]),
                (y, [5, 10, 20]),
                method="invalid",  # type: ignore
            )

    def test_mismatched_breakpoint_sizes_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        with pytest.raises(ValueError, match="same size"):
            m.add_piecewise_formulation(
                (x, [0, 10, 50]),
                (y, [5, 10]),
            )

    def test_non_tuple_arg_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        with pytest.raises(TypeError, match="tuple"):
            m.add_piecewise_formulation(x, [0, 10, 50])  # type: ignore


# ===========================================================================
# Name generation
# ===========================================================================


class TestNameGeneration:
    def test_auto_name(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        z = m.add_variables(name="z")
        m.add_piecewise_formulation((x, [0, 10, 50]), (y, [5, 10, 20]))
        m.add_piecewise_formulation((x, [0, 20, 80]), (z, [10, 15, 50]))
        assert f"pwl0{PWL_DELTA_SUFFIX}" in m.variables
        assert f"pwl1{PWL_DELTA_SUFFIX}" in m.variables

    def test_custom_name(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_formulation(
            (x, [0, 10, 50]),
            (y, [5, 10, 20]),
            name="my_pwl",
        )
        assert f"my_pwl{PWL_DELTA_SUFFIX}" in m.variables
        assert f"my_pwl{PWL_LINK_SUFFIX}" in m.constraints
        # N-var path uses a single stacked link constraint (no separate y_link)


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
        # Points only have generator dim -> broadcast over time
        m.add_piecewise_formulation(
            (
                x,
                breakpoints(
                    {"gen_a": [0, 10, 50], "gen_b": [0, 20, 80]}, dim="generator"
                ),
            ),
            (
                y,
                breakpoints(
                    {"gen_a": [0, 5, 30], "gen_b": [0, 8, 50]}, dim="generator"
                ),
            ),
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
        x_pts = xr.DataArray([0, 10, 50, np.nan], dims=[BREAKPOINT_DIM])
        y_pts = xr.DataArray([0, 5, 20, np.nan], dims=[BREAKPOINT_DIM])
        m.add_piecewise_formulation(
            (x, x_pts),
            (y, y_pts),
            method="sos2",
        )
        lam = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        # First 3 should be valid, last masked
        assert (lam.labels.isel({BREAKPOINT_DIM: slice(None, 3)}) != -1).all()
        assert int(lam.labels.isel({BREAKPOINT_DIM: 3})) == -1

    def test_sos2_interior_nan_raises(self) -> None:
        """SOS2 with interior NaN breakpoints raises ValueError."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        x_pts = xr.DataArray([0, np.nan, 50, 100], dims=[BREAKPOINT_DIM])
        y_pts = xr.DataArray([0, np.nan, 20, 40], dims=[BREAKPOINT_DIM])
        with pytest.raises(ValueError, match="non-trailing NaN"):
            m.add_piecewise_formulation(
                (x, x_pts),
                (y, y_pts),
                method="sos2",
            )


# ===========================================================================
# LP file output
# ===========================================================================


class TestLPFileOutput:
    def test_sos2_equality(self, tmp_path: Path) -> None:
        m = Model()
        x = m.add_variables(name="x", lower=0, upper=100)
        y = m.add_variables(name="y")
        m.add_piecewise_formulation(
            (x, [0.0, 10.0, 50.0, 100.0]),
            (y, [5.0, 2.0, 20.0, 80.0]),
            method="sos2",
        )
        m.add_objective(y)
        fn = tmp_path / "pwl_eq.lp"
        m.to_file(fn, io_api="lp")
        content = fn.read_text().lower()
        assert "sos" in content
        assert "s2" in content

    def test_disjunctive_sos2_and_binary(self, tmp_path: Path) -> None:
        m = Model()
        x = m.add_variables(name="x", lower=0, upper=100)
        y = m.add_variables(name="y")
        m.add_piecewise_formulation(
            (x, segments([[0.0, 10.0], [50.0, 100.0]])),
            (y, segments([[0.0, 5.0], [20.0, 80.0]])),
        )
        m.add_objective(y)
        fn = tmp_path / "pwl_disj.lp"
        m.to_file(fn, io_api="lp")
        content = fn.read_text().lower()
        assert "s2" in content
        assert "binary" in content or "binaries" in content


# ===========================================================================
# Solver integration -- SOS2 capable
# ===========================================================================


@pytest.mark.skipif(len(_sos2_solvers) == 0, reason="No solver with SOS2 support")
class TestSolverSOS2:
    @pytest.fixture(params=_sos2_solvers)
    def solver_name(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_equality_minimize_cost(self, solver_name: str) -> None:
        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        cost = m.add_variables(name="cost")
        m.add_piecewise_formulation(
            (x, [0, 50, 100]),
            (cost, [0, 10, 50]),
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
        m.add_piecewise_formulation(
            (power, [0, 25, 50, 75, 100]),
            (eff, [0.7, 0.85, 0.95, 0.9, 0.8]),
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
        m.add_piecewise_formulation(
            (x, segments([[0.0, 10.0], [50.0, 100.0]])),
            (y, segments([[0.0, 5.0], [20.0, 80.0]])),
        )
        m.add_constraints(x >= 60, name="x_min")
        m.add_objective(y)
        status, _ = m.solve(solver_name=solver_name)
        assert status == "ok"
        # x=60 on second segment: y = 20 + (80-20)/(100-50)*(60-50) = 32
        np.testing.assert_allclose(float(x.solution.values), 60, atol=1e-4)
        np.testing.assert_allclose(float(y.solution.values), 32, atol=1e-4)


# ===========================================================================
# Solver integration -- Envelope (any solver)
# ===========================================================================


@pytest.mark.skipif(len(_any_solvers) == 0, reason="No solver available")
class TestSolverTangentLines:
    @pytest.fixture(params=_any_solvers)
    def solver_name(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_concave_le(self, solver_name: str) -> None:
        """Y <= concave f(x), maximize y"""
        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        y = m.add_variables(name="y")
        # Concave: [0,0],[50,40],[100,60]
        env = tangent_lines(x, [0, 50, 100], [0, 40, 60])
        m.add_constraints(y <= env, name="pwl")
        m.add_constraints(x <= 75, name="x_max")
        m.add_constraints(x >= 0, name="x_lo")
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
        env = tangent_lines(x, [0, 50, 100], [0, 10, 60])
        m.add_constraints(y >= env, name="pwl")
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
        env1 = tangent_lines(x1, [0, 50, 100], [0, 40, 60])
        m1.add_constraints(y1 <= env1, name="pwl")
        m1.add_constraints(x1 <= 75, name="x_max")
        m1.add_constraints(x1 >= 0, name="x_lo")
        m1.add_objective(y1, sense="max")
        s1, _ = m1.solve(solver_name=solver_name)

        # Model 2: slopes
        m2 = Model()
        x2 = m2.add_variables(lower=0, upper=100, name="x")
        y2 = m2.add_variables(name="y")
        env2 = tangent_lines(
            x2,
            [0, 50, 100],
            breakpoints(slopes=[0.8, 0.4], x_points=[0, 50, 100], y0=0),
        )
        m2.add_constraints(y2 <= env2, name="pwl")
        m2.add_constraints(x2 <= 75, name="x_max")
        m2.add_constraints(x2 >= 0, name="x_lo")
        m2.add_objective(y2, sense="max")
        s2, _ = m2.solve(solver_name=solver_name)

        assert s1 == "ok"
        assert s2 == "ok"
        np.testing.assert_allclose(
            float(y1.solution.values), float(y2.solution.values), atol=1e-4
        )


# ===========================================================================
# Active parameter (commitment binary)
# ===========================================================================


class TestActiveParameter:
    """Tests for the ``active`` parameter in piecewise constraints."""

    def test_incremental_creates_active_bound(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        u = m.add_variables(binary=True, name="u")
        m.add_piecewise_formulation(
            (x, [0, 10, 50, 100]),
            (y, [5, 10, 20, 80]),
            active=u,
            method="incremental",
        )
        assert f"pwl0{PWL_ACTIVE_BOUND_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_DELTA_SUFFIX}" in m.variables

    def test_active_none_is_default(self) -> None:
        """Without active, formulation is identical to before."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_formulation(
            (x, [0, 10, 50]),
            (y, [0, 5, 30]),
            method="incremental",
        )
        assert f"pwl0{PWL_ACTIVE_BOUND_SUFFIX}" not in m.constraints

    def test_active_with_linear_expression(self) -> None:
        """Active can be a LinearExpression, not just a Variable."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        u = m.add_variables(binary=True, name="u")
        m.add_piecewise_formulation(
            (x, [0, 50, 100]),
            (y, [0, 10, 50]),
            active=1 * u,
            method="incremental",
        )
        assert f"pwl0{PWL_ACTIVE_BOUND_SUFFIX}" in m.constraints


# ===========================================================================
# Solver integration -- active parameter
# ===========================================================================


@pytest.mark.skipif(len(_any_solvers) == 0, reason="No solver available")
class TestSolverActive:
    @pytest.fixture(params=_any_solvers)
    def solver_name(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_incremental_active_on(self, solver_name: str) -> None:
        """When u=1 (forced on), normal PWL domain is active."""
        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        y = m.add_variables(name="y")
        u = m.add_variables(binary=True, name="u")
        m.add_piecewise_formulation(
            (x, [0, 50, 100]),
            (y, [0, 10, 50]),
            active=u,
            method="incremental",
        )
        m.add_constraints(u >= 1, name="force_on")
        m.add_constraints(x >= 50, name="x_min")
        m.add_objective(y)
        status, _ = m.solve(solver_name=solver_name)
        assert status == "ok"
        np.testing.assert_allclose(float(x.solution.values), 50, atol=1e-4)
        np.testing.assert_allclose(float(y.solution.values), 10, atol=1e-4)

    def test_incremental_active_off(self, solver_name: str) -> None:
        """When u=0 (forced off), x and y must be zero."""
        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        y = m.add_variables(name="y")
        u = m.add_variables(binary=True, name="u")
        m.add_piecewise_formulation(
            (x, [0, 50, 100]),
            (y, [0, 10, 50]),
            active=u,
            method="incremental",
        )
        m.add_constraints(u <= 0, name="force_off")
        m.add_objective(y, sense="max")
        status, _ = m.solve(solver_name=solver_name)
        assert status == "ok"
        np.testing.assert_allclose(float(x.solution.values), 0, atol=1e-4)
        np.testing.assert_allclose(float(y.solution.values), 0, atol=1e-4)

    def test_incremental_nonzero_base_active_off(self, solver_name: str) -> None:
        """
        Non-zero base (x0=20, y0=5) with u=0 must still force zero.

        Tests the x0*u / y0*u base term multiplication -- would fail if
        base terms aren't multiplied by active.
        """
        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        y = m.add_variables(name="y")
        u = m.add_variables(binary=True, name="u")
        m.add_piecewise_formulation(
            (x, [20, 60, 100]),
            (y, [5, 20, 50]),
            active=u,
            method="incremental",
        )
        m.add_constraints(u <= 0, name="force_off")
        m.add_objective(y, sense="max")
        status, _ = m.solve(solver_name=solver_name)
        assert status == "ok"
        np.testing.assert_allclose(float(x.solution.values), 0, atol=1e-4)
        np.testing.assert_allclose(float(y.solution.values), 0, atol=1e-4)

    def test_unit_commitment_pattern(self, solver_name: str) -> None:
        """Solver decides to commit: verifies correct fuel at operating point."""
        m = Model()
        p_min, p_max = 20.0, 100.0
        fuel_at_pmin, fuel_at_pmax = 10.0, 60.0

        power = m.add_variables(lower=0, upper=p_max, name="power")
        fuel = m.add_variables(name="fuel")
        u = m.add_variables(binary=True, name="commit")

        m.add_piecewise_formulation(
            (power, [p_min, p_max]),
            (fuel, [fuel_at_pmin, fuel_at_pmax]),
            active=u,
            method="incremental",
        )
        m.add_constraints(power >= 50, name="demand")
        m.add_objective(fuel + 5 * u)

        status, _ = m.solve(solver_name=solver_name)
        assert status == "ok"
        np.testing.assert_allclose(float(u.solution.values), 1, atol=1e-4)
        np.testing.assert_allclose(float(power.solution.values), 50, atol=1e-4)
        # fuel = 10 + (60-10)/(100-20) * (50-20) = 28.75
        np.testing.assert_allclose(float(fuel.solution.values), 28.75, atol=1e-4)

    def test_multi_dimensional_solver(self, solver_name: str) -> None:
        """Per-entity on/off: gen_a on at x=50, gen_b off at x=0."""
        m = Model()
        gens = pd.Index(["a", "b"], name="gen")
        x = m.add_variables(lower=0, upper=100, coords=[gens], name="x")
        y = m.add_variables(coords=[gens], name="y")
        u = m.add_variables(binary=True, coords=[gens], name="u")
        m.add_piecewise_formulation(
            (x, [0, 50, 100]),
            (y, [0, 10, 50]),
            active=u,
            method="incremental",
        )
        m.add_constraints(u.sel(gen="a") >= 1, name="a_on")
        m.add_constraints(u.sel(gen="b") <= 0, name="b_off")
        m.add_constraints(x.sel(gen="a") >= 50, name="a_min")
        m.add_objective(y.sum())
        status, _ = m.solve(solver_name=solver_name)
        assert status == "ok"
        np.testing.assert_allclose(float(x.solution.sel(gen="a")), 50, atol=1e-4)
        np.testing.assert_allclose(float(y.solution.sel(gen="a")), 10, atol=1e-4)
        np.testing.assert_allclose(float(x.solution.sel(gen="b")), 0, atol=1e-4)
        np.testing.assert_allclose(float(y.solution.sel(gen="b")), 0, atol=1e-4)


@pytest.mark.skipif(len(_sos2_solvers) == 0, reason="No SOS2-capable solver")
class TestSolverActiveSOS2:
    @pytest.fixture(params=_sos2_solvers)
    def solver_name(self, request: pytest.FixtureRequest) -> str:
        return request.param

    def test_sos2_active_off(self, solver_name: str) -> None:
        """SOS2: u=0 forces sum(lambda)=0, collapsing x=0, y=0."""
        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        y = m.add_variables(name="y")
        u = m.add_variables(binary=True, name="u")
        m.add_piecewise_formulation(
            (x, [0, 50, 100]),
            (y, [0, 10, 50]),
            active=u,
            method="sos2",
        )
        m.add_constraints(u <= 0, name="force_off")
        m.add_objective(y, sense="max")
        status, _ = m.solve(solver_name=solver_name)
        assert status == "ok"
        np.testing.assert_allclose(float(x.solution.values), 0, atol=1e-4)
        np.testing.assert_allclose(float(y.solution.values), 0, atol=1e-4)

    def test_disjunctive_active_off(self, solver_name: str) -> None:
        """Disjunctive: u=0 forces sum(z_k)=0, collapsing x=0, y=0."""
        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        y = m.add_variables(name="y")
        u = m.add_variables(binary=True, name="u")
        m.add_piecewise_formulation(
            (x, segments([[0.0, 10.0], [50.0, 100.0]])),
            (y, segments([[0.0, 5.0], [20.0, 80.0]])),
            active=u,
        )
        m.add_constraints(u <= 0, name="force_off")
        m.add_objective(y, sense="max")
        status, _ = m.solve(solver_name=solver_name)
        assert status == "ok"
        np.testing.assert_allclose(float(x.solution.values), 0, atol=1e-4)
        np.testing.assert_allclose(float(y.solution.values), 0, atol=1e-4)


# ===========================================================================
# N-variable path
# ===========================================================================


class TestNVariable:
    """Tests for the N-variable tuple-based piecewise constraint API."""

    def test_sos2_creates_lambda_and_link(self) -> None:
        m = Model()
        power = m.add_variables(lower=0, upper=100, name="power")
        fuel = m.add_variables(name="fuel")
        m.add_piecewise_formulation(
            (power, [0.0, 50.0, 100.0]),
            (fuel, [0.0, 20.0, 60.0]),
            method="sos2",
        )
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_CONVEX_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_LINK_SUFFIX}" in m.constraints

    def test_incremental_creates_delta(self) -> None:
        m = Model()
        power = m.add_variables(lower=0, upper=100, name="power")
        fuel = m.add_variables(name="fuel")
        m.add_piecewise_formulation(
            (power, [0.0, 50.0, 100.0]),
            (fuel, [0.0, 20.0, 60.0]),
            method="incremental",
        )
        assert f"pwl0{PWL_DELTA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_LINK_SUFFIX}" in m.constraints

    def test_auto_selects_method(self) -> None:
        m = Model()
        power = m.add_variables(lower=0, upper=100, name="power")
        fuel = m.add_variables(name="fuel")
        m.add_piecewise_formulation(
            (power, [0.0, 50.0, 100.0]),
            (fuel, [0.0, 20.0, 60.0]),
        )
        # Auto should select incremental for monotonic breakpoints
        assert f"pwl0{PWL_DELTA_SUFFIX}" in m.variables

    def test_single_pair_raises(self) -> None:
        m = Model()
        power = m.add_variables(name="power")
        with pytest.raises(TypeError, match="at least 2"):
            m.add_piecewise_formulation(
                (power, [0.0, 50.0, 100.0]),
            )

    def test_three_variables(self) -> None:
        m = Model()
        power = m.add_variables(lower=0, upper=100, name="power")
        fuel = m.add_variables(name="fuel")
        heat = m.add_variables(name="heat")
        m.add_piecewise_formulation(
            (power, [0.0, 50.0, 100.0]),
            (fuel, [0.0, 20.0, 60.0]),
            (heat, [0.0, 30.0, 80.0]),
            method="sos2",
        )
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_LINK_SUFFIX}" in m.constraints
        # link constraint should have _pwl_var dimension
        link = m.constraints[f"pwl0{PWL_LINK_SUFFIX}"]
        assert "_pwl_var" in link.labels.dims

    def test_custom_name(self) -> None:
        m = Model()
        power = m.add_variables(lower=0, upper=100, name="power")
        fuel = m.add_variables(name="fuel")
        m.add_piecewise_formulation(
            (power, [0.0, 50.0, 100.0]),
            (fuel, [0.0, 20.0, 60.0]),
            name="chp",
        )
        assert f"chp{PWL_DELTA_SUFFIX}" in m.variables


# ===========================================================================
# Additional validation and edge-case coverage
# ===========================================================================


class TestValidationEdgeCases:
    def test_non_1d_sequence_raises(self) -> None:
        """breakpoints() with a 2D nested list raises ValueError."""
        with pytest.raises(ValueError, match="1D sequence"):
            breakpoints([[1, 2], [3, 4]])

    def test_breakpoints_no_values_no_slopes_raises(self) -> None:
        """breakpoints() with neither values nor slopes raises."""
        with pytest.raises(ValueError, match="Must pass either"):
            breakpoints()

    def test_slopes_1d_non_scalar_y0_raises(self) -> None:
        """1D slopes with dict y0 raises TypeError."""
        with pytest.raises(TypeError, match="scalar float"):
            breakpoints(slopes=[1, 2], x_points=[0, 10, 20], y0={"a": 0})

    def test_slopes_bad_y0_type_raises(self) -> None:
        """Slopes with unsupported y0 type raises TypeError."""
        with pytest.raises(TypeError, match="y0"):
            breakpoints(
                slopes={"a": [1, 2], "b": [3, 4]},
                x_points={"a": [0, 10, 20], "b": [0, 10, 20]},
                y0="bad",
                dim="entity",
            )

    def test_slopes_dataarray_y0(self) -> None:
        """Slopes mode with DataArray y0 works."""
        y0_da = xr.DataArray([0, 5], dims=["gen"], coords={"gen": ["a", "b"]})
        bp = breakpoints(
            slopes={"a": [1, 2], "b": [3, 4]},
            x_points={"a": [0, 10, 20], "b": [0, 10, 20]},
            y0=y0_da,
            dim="gen",
        )
        assert BREAKPOINT_DIM in bp.dims
        assert "gen" in bp.dims

    def test_non_numeric_breakpoint_coords_raises(self) -> None:
        """SOS2 with string breakpoint coords raises ValueError."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        x_pts = xr.DataArray(
            [0, 10, 50],
            dims=[BREAKPOINT_DIM],
            coords={BREAKPOINT_DIM: ["a", "b", "c"]},
        )
        y_pts = xr.DataArray(
            [0, 5, 20],
            dims=[BREAKPOINT_DIM],
            coords={BREAKPOINT_DIM: ["a", "b", "c"]},
        )
        with pytest.raises(ValueError, match="numeric coordinates"):
            m.add_piecewise_formulation(
                (x, x_pts),
                (y, y_pts),
                method="sos2",
            )

    def test_missing_breakpoint_dim_on_second_arg_raises(self) -> None:
        """Second breakpoint array missing breakpoint dim raises."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        good = xr.DataArray([0, 10, 50], dims=[BREAKPOINT_DIM])
        bad = xr.DataArray([0, 5, 20], dims=["wrong"])
        with pytest.raises(ValueError, match="missing"):
            m.add_piecewise_formulation((x, good), (y, bad))

    def test_segment_dim_mismatch_raises(self) -> None:
        """Segment dim on only one breakpoint array raises."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        x_pts = segments([[0, 10], [50, 100]])
        y_pts = breakpoints([0, 5])  # same breakpoint count but no segment dim
        with pytest.raises(ValueError, match="segment dimension"):
            m.add_piecewise_formulation((x, x_pts), (y, y_pts))

    def test_disjunctive_three_pairs(self) -> None:
        """Disjunctive with 3 pairs works (N-variable)."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        z = m.add_variables(name="z")
        seg = segments([[0, 10], [50, 100]])
        m.add_piecewise_formulation(
            (x, seg),
            (y, seg),
            (z, seg),
        )
        assert f"pwl0{PWL_SEGMENT_BINARY_SUFFIX}" in m.variables
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_LINK_SUFFIX}" in m.constraints

    def test_disjunctive_interior_nan_raises(self) -> None:
        """Disjunctive with interior NaN raises ValueError."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        # 3 breakpoints per segment, NaN in the middle of segment 0
        x_pts = xr.DataArray(
            [[0, np.nan, 10], [50, 75, 100]],
            dims=[SEGMENT_DIM, BREAKPOINT_DIM],
        )
        y_pts = xr.DataArray(
            [[0, np.nan, 5], [20, 50, 80]],
            dims=[SEGMENT_DIM, BREAKPOINT_DIM],
        )
        with pytest.raises(ValueError, match="non-trailing NaN"):
            m.add_piecewise_formulation((x, x_pts), (y, y_pts))

    def test_expression_name_fallback(self) -> None:
        """LinExpr (not Variable) gets numeric name in link coords."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        # Non-monotonic so auto picks SOS2 (which creates lambda vars)
        m.add_piecewise_formulation(
            (1.0 * x, [0, 50, 10]),
            (1.0 * y, [0, 20, 5]),
            method="sos2",
        )
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables

    def test_incremental_with_nan_mask(self) -> None:
        """Incremental method with trailing NaN creates masked delta vars."""
        m = Model()
        gens = pd.Index(["a", "b"], name="gen")
        x = m.add_variables(coords=[gens], name="x")
        y = m.add_variables(coords=[gens], name="y")
        x_pts = breakpoints({"a": [0, 10, 50], "b": [0, 20]}, dim="gen")
        y_pts = breakpoints({"a": [0, 5, 20], "b": [0, 8]}, dim="gen")
        m.add_piecewise_formulation(
            (x, x_pts),
            (y, y_pts),
            method="incremental",
        )
        delta = m.variables[f"pwl0{PWL_DELTA_SUFFIX}"]
        assert delta.labels.shape[0] == 2  # 2 generators

    def test_scalar_coord_dropped(self) -> None:
        """Scalar coords on breakpoints are dropped before stacking."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        bp = breakpoints([0, 10, 50])
        bp_with_scalar = bp.assign_coords(extra=42)
        m.add_piecewise_formulation(
            (x, bp_with_scalar),
            (y, [0, 5, 20]),
            method="sos2",
        )
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables


# ===========================================================================
# Sign parameter (inequality bounds)
# ===========================================================================


class TestSignParameter:
    """Tests for sign="<=" / ">=" with the first-tuple convention."""

    def test_default_is_equality(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_formulation((x, [0, 10, 50]), (y, [0, 5, 20]))
        # no output_link for equality — single stacked link only
        assert f"pwl0{PWL_OUTPUT_LINK_SUFFIX}" not in m.constraints

    def test_invalid_sign_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        with pytest.raises(ValueError, match="sign must be"):
            m.add_piecewise_formulation((x, [0, 10]), (y, [0, 5]), sign="!")  # type: ignore

    def test_lp_with_equality_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        with pytest.raises(ValueError, match="method='lp'"):
            m.add_piecewise_formulation((x, [0, 10, 50]), (y, [0, 5, 20]), method="lp")

    def test_auto_picks_lp_for_concave_le(self) -> None:
        """Concave curve + sign='<=' + auto → LP tangent lines (no aux vars)."""
        m = Model()
        power = m.add_variables(lower=0, upper=30, name="power")
        fuel = m.add_variables(lower=0, upper=40, name="fuel")
        # Concave: slopes 2, 1, 0.5
        m.add_piecewise_formulation(
            (fuel, [0, 20, 30, 35]),
            (power, [0, 10, 20, 30]),
            sign="<=",
        )
        assert f"pwl0{PWL_CHORD_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_DOMAIN_LO_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_DOMAIN_HI_SUFFIX}" in m.constraints
        # No SOS2 lambdas for LP
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" not in m.variables

    def test_auto_picks_lp_for_convex_ge(self) -> None:
        """Convex curve + sign='>=' + auto → LP tangent lines."""
        m = Model()
        x = m.add_variables(lower=0, upper=30, name="x")
        y = m.add_variables(lower=0, upper=100, name="y")
        # Convex: slopes 1, 2, 3
        m.add_piecewise_formulation(
            (y, [0, 10, 30, 60]),
            (x, [0, 10, 20, 30]),
            sign=">=",
        )
        assert f"pwl0{PWL_CHORD_SUFFIX}" in m.constraints

    def test_auto_falls_back_to_sos2_for_nonmonotonic(self) -> None:
        """Non-monotonic x + sign='<=' + auto → SOS2 with signed output link."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        # Non-monotonic x
        m.add_piecewise_formulation(
            (y, [0, 5, 2, 20]),
            (x, [0, 10, 5, 50]),
            sign="<=",
        )
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_OUTPUT_LINK_SUFFIX}" in m.constraints

    def test_auto_concave_ge_falls_back_from_lp(self) -> None:
        """Concave + sign='>=' is LP-loose → auto must not pick LP."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        f = m.add_piecewise_formulation(
            (y, [0, 20, 30, 35]),  # concave
            (x, [0, 10, 20, 30]),
            sign=">=",
        )
        assert f.method != "lp"  # fallback (sos2 or incremental)

    def test_auto_convex_le_falls_back_from_lp(self) -> None:
        """Convex + sign='<=' is LP-loose → auto must not pick LP."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        f = m.add_piecewise_formulation(
            (y, [0, 10, 30, 60]),  # convex
            (x, [0, 10, 20, 30]),
            sign="<=",
        )
        assert f.method != "lp"

    def test_lp_concave_ge_raises(self) -> None:
        """Explicit LP + sign='>=' on concave curve is loose → raise."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        with pytest.raises(ValueError, match="convex"):
            m.add_piecewise_formulation(
                (y, [0, 20, 30, 35]),  # concave
                (x, [0, 10, 20, 30]),
                sign=">=",
                method="lp",
            )

    def test_lp_nonmatching_convexity_raises(self) -> None:
        """Explicit LP with sign='<=' on a convex curve → error."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        # Convex curve, sign='<=' mismatch
        with pytest.raises(ValueError, match="concave"):
            m.add_piecewise_formulation(
                (y, [0, 10, 30, 60]),  # convex
                (x, [0, 10, 20, 30]),
                sign="<=",
                method="lp",
            )

    def test_sos2_sign_le_has_output_link(self) -> None:
        """Explicit SOS2 with sign='<=' gets a signed output link."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_formulation(
            (y, [0, 20, 30, 35]),
            (x, [0, 10, 20, 30]),
            sign="<=",
            method="sos2",
        )
        link = m.constraints[f"pwl0{PWL_OUTPUT_LINK_SUFFIX}"]
        assert (link.sign == "<=").all().item()

    def test_incremental_sign_le(self) -> None:
        """Incremental method honours sign on output link."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_formulation(
            (y, [0, 20, 30, 35]),
            (x, [0, 10, 20, 30]),
            sign="<=",
            method="incremental",
        )
        assert f"pwl0{PWL_DELTA_SUFFIX}" in m.variables
        link = m.constraints[f"pwl0{PWL_OUTPUT_LINK_SUFFIX}"]
        assert (link.sign == "<=").all().item()

    def test_nvar_inequality_bounds_first_tuple(self) -> None:
        """N-variable: first tuple is bounded, others on curve."""
        m = Model()
        fuel = m.add_variables(name="fuel")
        power = m.add_variables(name="power")
        heat = m.add_variables(name="heat")
        m.add_piecewise_formulation(
            (fuel, [0, 40, 85, 160]),  # bounded
            (power, [0, 30, 60, 100]),  # input ==
            (heat, [0, 25, 55, 95]),  # input ==
            sign="<=",
            method="sos2",
        )
        # inputs stacked, output signed
        link = m.constraints[f"pwl0{PWL_LINK_SUFFIX}"]
        output_link = m.constraints[f"pwl0{PWL_OUTPUT_LINK_SUFFIX}"]
        assert "_pwl_var" in link.labels.dims  # stacked inputs
        assert "_pwl_var" not in output_link.labels.dims  # single output
        assert (output_link.sign == "<=").all().item()

    def test_lp_consistency_with_sos2(self) -> None:
        """LP and SOS2 give the same fuel at a fixed power (within domain)."""
        x_pts = [0, 10, 20, 30]
        y_pts = [0, 20, 30, 35]  # concave

        solutions = {}
        for method in ["lp", "sos2", "incremental"]:
            m = Model()
            power = m.add_variables(lower=0, upper=30, name="power")
            fuel = m.add_variables(lower=0, upper=40, name="fuel")
            m.add_piecewise_formulation(
                (fuel, y_pts),
                (power, x_pts),
                sign="<=",
                method=method,
            )
            m.add_constraints(power == 15)
            m.add_objective(-fuel)  # maximize fuel
            m.solve()
            solutions[method] = float(m.solution["fuel"])

        # all methods should max out at f(15) = 25
        for method, val in solutions.items():
            assert abs(val - 25.0) < 1e-4, f"{method}: got {val}"

    def test_convexity_invariant_to_x_direction(self) -> None:
        """Decreasing x must classify the same curve identically to ascending x."""
        m_asc = Model()
        xa = m_asc.add_variables(name="x")
        ya = m_asc.add_variables(name="y")
        f_asc = m_asc.add_piecewise_formulation(
            (ya, [0, 20, 30, 35]),
            (xa, [0, 10, 20, 30]),
            sign=">=",
        )
        m_desc = Model()
        xd = m_desc.add_variables(name="x")
        yd = m_desc.add_variables(name="y")
        f_desc = m_desc.add_piecewise_formulation(
            (yd, [35, 30, 20, 0]),
            (xd, [30, 20, 10, 0]),
            sign=">=",
        )
        assert f_asc.convexity == f_desc.convexity == "concave"
        # concave + >= must fall back from LP
        assert f_asc.method != "lp"
        assert f_desc.method != "lp"

    def test_lp_per_entity_nan_padding(self) -> None:
        """
        Per-entity NaN-padded breakpoints with method='lp': padded
        segments must be masked out so they don't create spurious
        ``y ≤ 0`` constraints (bug-2 regression).
        """
        from linopy.piecewise import breakpoints

        bp_y = pd.DataFrame([[0, 20, 30, 35], [0, 10, 15, np.nan]], index=["a", "b"])
        bp_x = pd.DataFrame([[0, 10, 20, 30], [0, 5, 15, np.nan]], index=["a", "b"])
        results: dict[str, float] = {}
        for method in ["lp", "sos2"]:
            m = Model()
            coord = pd.Index(["a", "b"], name="entity")
            x = m.add_variables(lower=0, upper=20, coords=[coord], name="x")
            y = m.add_variables(lower=0, upper=40, coords=[coord], name="y")
            m.add_piecewise_formulation(
                (y, breakpoints(bp_y, dim="entity")),
                (x, breakpoints(bp_x, dim="entity")),
                sign="<=",
                method=method,
            )
            m.add_constraints(x.sel(entity="b") == 10)
            m.add_objective(-y.sel(entity="b"))
            m.solve()
            results[method] = float(m.solution.sel({"entity": "b"})["y"])
        # f_b(10) on chord (5,10)→(15,15) is 12.5
        assert abs(results["lp"] - 12.5) < 1e-3
        assert abs(results["sos2"] - results["lp"]) < 1e-3

    def test_lp_rejects_decreasing_x_concave_ge(self) -> None:
        """
        Explicit LP on a concave curve with sign='>=' must raise, even
        when x is specified in decreasing order (bug-1 regression).
        """
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        with pytest.raises(ValueError, match="convex"):
            m.add_piecewise_formulation(
                (y, [35, 30, 20, 0]),  # same concave curve
                (x, [30, 20, 10, 0]),  # decreasing x
                sign=">=",
                method="lp",
            )

    @pytest.mark.skipif(not _sos2_solvers, reason="no SOS2-capable solver available")
    @pytest.mark.parametrize("method", ["sos2", "incremental"])
    def test_active_off_with_sign_le_leaves_lower_open(self, method: str) -> None:
        """
        Documents the asymmetry between sign='==' and sign='<=' under
        active=0: equality forces y=0, but '<=' only bounds y ≤ 0 — the
        lower side still comes from the variable's own bounds.  Verified
        uniform across sos2 and incremental.  A future change to add the
        complementary bound automatically should flip this test.
        """
        m = Model()
        x = m.add_variables(lower=-100, upper=100, name="x")
        y = m.add_variables(lower=-100, upper=100, name="y")
        active = m.add_variables(binary=True, name="active")
        m.add_piecewise_formulation(
            (y, [0, 20, 30, 35]),
            (x, [0, 10, 20, 30]),
            sign="<=",
            method=method,
            active=active,
        )
        m.add_constraints(active == 0)
        m.add_objective(y)  # minimize y
        m.solve()
        # y hits its own lower bound (not 0) — matches docstring note.
        assert m.solution["y"].item() == pytest.approx(-100.0, abs=1e-6)
        # Input x is still pinned to 0 by the equality input link.
        assert m.solution["x"].item() == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.skipif(not _sos2_solvers, reason="no SOS2-capable solver available")
    def test_active_off_with_sign_le_and_lower_zero_pins_output(self) -> None:
        """
        Docstring recipe: with ``y.lower = 0`` (the common case for
        fuel/cost/heat outputs), the sign='<=' + active=0 asymmetry
        disappears — the variable bound combined with y ≤ 0 forces
        y = 0 automatically.
        """
        m = Model()
        x = m.add_variables(lower=0, upper=30, name="x")
        y = m.add_variables(lower=0, upper=100, name="y")  # the recipe
        active = m.add_variables(binary=True, name="active")
        m.add_piecewise_formulation(
            (y, [0, 20, 30, 35]),
            (x, [0, 10, 20, 30]),
            sign="<=",
            method="sos2",
            active=active,
        )
        m.add_constraints(active == 0)
        m.add_objective(y, sense="max")  # try to push y up
        m.solve()
        assert m.solution["y"].item() == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.skipif(not _sos2_solvers, reason="no SOS2-capable solver available")
    def test_active_off_with_sign_le_disjunctive(self) -> None:
        """Same asymmetry applies to the disjunctive (segments) path."""
        m = Model()
        x = m.add_variables(lower=-100, upper=100, name="x")
        y = m.add_variables(lower=-100, upper=100, name="y")
        active = m.add_variables(binary=True, name="active")
        m.add_piecewise_formulation(
            (y, segments([[0.0, 20.0], [20.0, 35.0]])),
            (x, segments([[0.0, 10.0], [10.0, 30.0]])),
            sign="<=",
            active=active,
        )
        m.add_constraints(active == 0)
        m.add_objective(y)
        m.solve()
        assert m.solution["y"].item() == pytest.approx(-100.0, abs=1e-6)
        assert m.solution["x"].item() == pytest.approx(0.0, abs=1e-6)

    def test_lp_active_explicit_raises(self) -> None:
        """
        method='lp' + active is ValueError (silently ignoring active
        would produce a wrong model).
        """
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        u = m.add_variables(binary=True, name="u")
        with pytest.raises(ValueError, match="active"):
            m.add_piecewise_formulation(
                (y, [0, 20, 30, 35]),
                (x, [0, 10, 20, 30]),
                sign="<=",
                method="lp",
                active=u,
            )

    def test_lp_accepts_linear_curve(self) -> None:
        """
        A linear curve is both convex and concave per detection, so
        LP must accept it with either sign and build the formulation.
        """
        for sign in ["<=", ">="]:
            m = Model()
            x = m.add_variables(lower=0, upper=30, name="x")
            y = m.add_variables(lower=0, upper=60, name="y")
            f = m.add_piecewise_formulation(
                (y, [0, 10, 20, 30]),  # linear (all slopes = 1)
                (x, [0, 10, 20, 30]),
                sign=sign,
                method="lp",
            )
            assert f.method == "lp"
            assert f.convexity == "linear"

    def test_auto_logs_when_lp_is_skipped(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        method='auto' on a non-LP-eligible case emits an INFO log
        explaining why LP was passed over.
        """
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        with caplog.at_level(logging.INFO, logger="linopy.piecewise"):
            m.add_piecewise_formulation(
                (y, [0, 20, 30, 35]),  # concave + sign='>=' → LP skipped
                (x, [0, 10, 20, 30]),
                sign=">=",
            )
        assert "LP not applicable" in caplog.text

    @pytest.mark.skipif(not _any_solvers, reason="no solver available")
    def test_lp_domain_bound_infeasible_when_x_out_of_range(self) -> None:
        """
        LP's x ∈ [x_min, x_max] domain bound bites — forcing x beyond
        the breakpoint range must make the model infeasible.
        """
        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        y = m.add_variables(lower=0, upper=100, name="y")
        m.add_piecewise_formulation(
            (y, [0, 20, 30, 35]),
            (x, [0, 10, 20, 30]),  # x_max = 30
            sign="<=",
            method="lp",
        )
        m.add_constraints(x >= 50)
        m.add_objective(-y)
        status, _ = m.solve()
        assert status != "ok"

    @pytest.mark.skipif(not _any_solvers, reason="no solver available")
    def test_lp_matches_sos2_on_multi_dim_variables(self) -> None:
        """
        LP with an entity dimension beyond BREAKPOINT_DIM must match
        the SOS2 solution per entity.
        """
        entities = pd.Index(["a", "b"], name="entity")
        bp_x = pd.DataFrame([[0, 10, 20, 30], [0, 10, 20, 30]], index=["a", "b"])
        bp_y = pd.DataFrame([[0, 20, 30, 35], [0, 15, 25, 30]], index=["a", "b"])
        ys: dict[str, xr.DataArray] = {}
        for method in ["lp", "sos2"]:
            m = Model()
            x = m.add_variables(lower=0, upper=30, coords=[entities], name="x")
            y = m.add_variables(lower=0, upper=40, coords=[entities], name="y")
            m.add_piecewise_formulation(
                (y, breakpoints(bp_y, dim="entity")),
                (x, breakpoints(bp_x, dim="entity")),
                sign="<=",
                method=method,
            )
            m.add_constraints(x.sel(entity="a") == 15)
            m.add_constraints(x.sel(entity="b") == 5)
            m.add_objective(-y.sum())
            m.solve()
            ys[method] = y.solution
        for entity in ["a", "b"]:
            assert float(ys["lp"].sel(entity=entity)) == pytest.approx(
                float(ys["sos2"].sel(entity=entity)), abs=1e-3
            )

    @pytest.mark.skipif(not _any_solvers, reason="no solver available")
    def test_lp_consistency_with_sos2_both_directions(self) -> None:
        """
        Extends test_lp_consistency_with_sos2 to also probe the
        minimisation side of y ≤ f(x).
        """
        x_pts = [0, 10, 20, 30]
        y_pts = [0, 20, 30, 35]  # concave
        for obj_sign in [-1.0, +1.0]:
            sols: dict[str, float] = {}
            for method in ["lp", "sos2"]:
                m = Model()
                p = m.add_variables(lower=0, upper=30, name="p")
                f = m.add_variables(lower=0, upper=50, name="f")
                m.add_piecewise_formulation(
                    (f, y_pts), (p, x_pts), sign="<=", method=method
                )
                m.add_constraints(p == 15)
                m.add_objective(obj_sign * f)
                m.solve()
                sols[method] = float(m.solution["f"])
            assert sols["lp"] == pytest.approx(sols["sos2"], abs=1e-3)


def _bp(values: list[float]) -> xr.DataArray:
    """Small helper: plain 1-D breakpoint DataArray for convexity tests."""
    return breakpoints(values)


class TestDetectConvexity:
    """Direct unit tests for the _detect_convexity classifier."""

    def test_convex(self) -> None:
        from linopy.piecewise import _detect_convexity

        x = _bp([0, 1, 2, 3])
        y = _bp([0, 1, 4, 9])  # y = x^2
        assert _detect_convexity(x, y) == "convex"

    def test_concave(self) -> None:
        from linopy.piecewise import _detect_convexity

        x = _bp([0, 1, 2, 3])
        y = _bp([0, 1, 1.5, 1.75])  # diminishing returns
        assert _detect_convexity(x, y) == "concave"

    def test_linear_exact(self) -> None:
        from linopy.piecewise import _detect_convexity

        x = _bp([0, 1, 2, 3])
        y = _bp([0, 2, 4, 6])
        assert _detect_convexity(x, y) == "linear"

    def test_linear_within_tol(self) -> None:
        from linopy.piecewise import _detect_convexity

        # Tiny slope wobble within 1e-10 tolerance
        x = _bp([0, 1, 2, 3])
        y = _bp([0, 2.0, 4.0 + 1e-12, 6.0 + 2e-12])
        assert _detect_convexity(x, y) == "linear"

    def test_mixed(self) -> None:
        from linopy.piecewise import _detect_convexity

        x = _bp([0, 1, 2, 3, 4])
        y = _bp([0, 1, 4, 5, 4])  # convex then concave
        assert _detect_convexity(x, y) == "mixed"

    def test_too_few_points_returns_linear(self) -> None:
        from linopy.piecewise import _detect_convexity

        # Only two points — no second difference to examine
        x = _bp([0, 1])
        y = _bp([0, 2])
        assert _detect_convexity(x, y) == "linear"

    def test_decreasing_x_matches_ascending(self) -> None:
        """Reversing the breakpoint order must not change the label."""
        from linopy.piecewise import _detect_convexity

        # convex
        assert _detect_convexity(_bp([0, 1, 2, 3]), _bp([0, 1, 4, 9])) == "convex"
        assert _detect_convexity(_bp([3, 2, 1, 0]), _bp([9, 4, 1, 0])) == "convex"
        # concave
        assert (
            _detect_convexity(_bp([0, 10, 20, 30]), _bp([0, 20, 30, 35])) == "concave"
        )
        assert (
            _detect_convexity(_bp([30, 20, 10, 0]), _bp([35, 30, 20, 0])) == "concave"
        )

    def test_trailing_nan_ignored(self) -> None:
        from linopy.piecewise import _detect_convexity

        # Concave curve with a trailing NaN padding
        x = _bp([0.0, 1.0, 2.0, np.nan])
        y = _bp([0.0, 1.0, 1.5, np.nan])
        assert _detect_convexity(x, y) == "concave"

    def test_multi_entity_same_shape(self) -> None:
        from linopy.piecewise import _detect_convexity

        # Both rows convex
        bp_x = pd.DataFrame([[0, 1, 2, 3], [0, 1, 2, 3]], index=["a", "b"])
        bp_y = pd.DataFrame([[0, 1, 4, 9], [0, 2, 8, 18]], index=["a", "b"])
        assert (
            _detect_convexity(
                breakpoints(bp_x, dim="entity"),
                breakpoints(bp_y, dim="entity"),
            )
            == "convex"
        )

    def test_multi_entity_mixed_direction(self) -> None:
        """Same concave curve, one entity ascending, one descending."""
        from linopy.piecewise import _detect_convexity

        bp_x = pd.DataFrame([[0, 10, 20, 30], [30, 20, 10, 0]], index=["a", "b"])
        bp_y = pd.DataFrame([[0, 20, 30, 35], [35, 30, 20, 0]], index=["a", "b"])
        assert (
            _detect_convexity(
                breakpoints(bp_x, dim="entity"),
                breakpoints(bp_y, dim="entity"),
            )
            == "concave"
        )

    def test_multi_entity_mixed_curvatures(self) -> None:
        """One convex, one concave across entities → mixed."""
        from linopy.piecewise import _detect_convexity

        bp_x = pd.DataFrame([[0, 1, 2, 3], [0, 1, 2, 3]], index=["a", "b"])
        bp_y = pd.DataFrame([[0, 1, 4, 9], [0, 1, 1.5, 1.75]], index=["a", "b"])
        assert (
            _detect_convexity(
                breakpoints(bp_x, dim="entity"),
                breakpoints(bp_y, dim="entity"),
            )
            == "mixed"
        )


# ===========================================================================
# netCDF round-trip
# ===========================================================================


class TestPiecewiseNetCDFRoundtrip:
    def test_formulation_survives_netcdf(self, tmp_path: Path) -> None:
        from linopy import read_netcdf
        from linopy.piecewise import PiecewiseFormulation

        m = Model()
        y = m.add_variables(name="y")
        x = m.add_variables(lower=0, upper=30, name="x")
        f = m.add_piecewise_formulation(
            (y, [0, 20, 30, 35]),
            (x, [0, 10, 20, 30]),
            name="pwl",
        )
        assert f.convexity == "concave"

        path = tmp_path / "model.nc"
        m.to_netcdf(path)
        f2 = read_netcdf(path)._piecewise_formulations["pwl"]

        # Compare every slot except the back-reference to the model, so this
        # test auto-catches any future field that IO forgets to persist.
        fields = [s for s in PiecewiseFormulation.__slots__ if s != "_model"]
        before = {s: getattr(f, s) for s in fields}
        after = {s: getattr(f2, s) for s in fields}
        assert before == after
