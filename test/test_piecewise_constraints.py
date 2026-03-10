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
    BREAKPOINT_DIM,
    LP_SEG_DIM,
    PWL_ACTIVE_BOUND_SUFFIX,
    PWL_AUX_SUFFIX,
    PWL_BINARY_SUFFIX,
    PWL_CONVEX_SUFFIX,
    PWL_DELTA_SUFFIX,
    PWL_FILL_SUFFIX,
    PWL_INC_BINARY_SUFFIX,
    PWL_INC_LINK_SUFFIX,
    PWL_INC_ORDER_SUFFIX,
    PWL_LAMBDA_SUFFIX,
    PWL_LP_DOMAIN_SUFFIX,
    PWL_LP_SUFFIX,
    PWL_SELECT_SUFFIX,
    PWL_X_LINK_SUFFIX,
    PWL_Y_LINK_SUFFIX,
    SEGMENT_DIM,
)
from linopy.piecewise import (
    PiecewiseConstraintDescriptor,
    PiecewiseExpression,
)
from linopy.solver_capabilities import SolverFeature, get_available_solvers_with_feature

_sos2_solvers = get_available_solvers_with_feature(
    SolverFeature.SOS_CONSTRAINTS, available_solvers
)
_any_solvers = [
    s for s in ["highs", "gurobi", "glpk", "cplex"] if s in available_solvers
]


@pytest.fixture(autouse=True)
def _legacy_only(legacy_convention: None) -> None:
    """Piecewise implementation not yet adapted for v1 convention."""


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
# piecewise() and operator overloading
# ===========================================================================


class TestPiecewiseFunction:
    def test_returns_expression(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        pw = piecewise(x, x_points=[0, 10, 50], y_points=[5, 2, 20])
        assert isinstance(pw, PiecewiseExpression)

    def test_series_inputs(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        pw = piecewise(x, pd.Series([0, 10, 50]), pd.Series([5, 2, 20]))
        assert isinstance(pw, PiecewiseExpression)

    def test_tuple_inputs(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        pw = piecewise(x, (0, 10, 50), (5, 2, 20))
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

    @pytest.mark.parametrize(
        ("operator", "expected_sign"),
        [("==", "=="), ("<=", "<="), (">=", ">=")],
    )
    def test_rhs_piecewise_returns_descriptor(
        self, operator: str, expected_sign: str
    ) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        pw = piecewise(x, [0, 10, 50], [5, 2, 20])

        if operator == "==":
            desc = y == pw
        elif operator == "<=":
            desc = y <= pw
        else:
            desc = y >= pw

        assert isinstance(desc, PiecewiseConstraintDescriptor)
        assert desc.sign == expected_sign
        assert desc.piecewise_func is pw

    @pytest.mark.parametrize(
        ("operator", "expected_sign"),
        [("==", "=="), ("<=", "<="), (">=", ">=")],
    )
    def test_rhs_piecewise_linear_expression_returns_descriptor(
        self, operator: str, expected_sign: str
    ) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        z = m.add_variables(name="z")
        lhs = 2 * y + z
        pw = piecewise(x, [0, 10, 50], [5, 2, 20])

        if operator == "==":
            desc = lhs == pw
        elif operator == "<=":
            desc = lhs <= pw
        else:
            desc = lhs >= pw

        assert isinstance(desc, PiecewiseConstraintDescriptor)
        assert desc.sign == expected_sign
        assert desc.lhs is lhs
        assert desc.piecewise_func is pw

    def test_rhs_piecewise_add_constraint(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(y == piecewise(x, [0, 10, 50], [5, 2, 20]))
        assert len(m.constraints) > 0

    def test_mismatched_sizes_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        with pytest.raises(ValueError, match="same size"):
            piecewise(x, [0, 10, 50, 100], [5, 2, 20])

    def test_missing_breakpoint_dim_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        xp = xr.DataArray([0, 10, 50], dims=["knot"])
        yp = xr.DataArray([5, 2, 20], dims=["knot"])
        with pytest.raises(ValueError, match="must have a breakpoint dimension"):
            piecewise(x, xp, yp)

    def test_missing_breakpoint_dim_x_only_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        xp = xr.DataArray([0, 10, 50], dims=["knot"])
        yp = xr.DataArray([5, 2, 20], dims=[BREAKPOINT_DIM])
        with pytest.raises(
            ValueError, match="x_points is missing the breakpoint dimension"
        ):
            piecewise(x, xp, yp)

    def test_missing_breakpoint_dim_y_only_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        xp = xr.DataArray([0, 10, 50], dims=[BREAKPOINT_DIM])
        yp = xr.DataArray([5, 2, 20], dims=["knot"])
        with pytest.raises(
            ValueError, match="y_points is missing the breakpoint dimension"
        ):
            piecewise(x, xp, yp)

    def test_segment_dim_mismatch_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        xp = segments([[0, 10], [50, 100]])
        yp = xr.DataArray([0, 5], dims=[BREAKPOINT_DIM])
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

    def test_auto_nonmonotonic_falls_back_to_sos2(self) -> None:
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

    def test_method_lp_decreasing_breakpoints_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        with pytest.raises(ValueError, match="strictly increasing x_points"):
            m.add_piecewise_constraints(
                piecewise(x, [100, 50, 0], [60, 10, 0]) <= y,
                method="lp",
            )

    def test_auto_inequality_decreasing_breakpoints_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        with pytest.raises(ValueError, match="strictly increasing x_points"):
            m.add_piecewise_constraints(
                piecewise(x, [100, 50, 0], [60, 10, 0]) <= y,
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
        assert delta.labels.sizes[LP_SEG_DIM] == 3
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

    def test_sos2_nonmonotonic_succeeds(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 30, 100], [5, 20, 15, 80]) == y,
            method="sos2",
        )
        assert f"pwl0{PWL_LAMBDA_SUFFIX}" in m.variables
        assert f"pwl0{PWL_DELTA_SUFFIX}" not in m.variables

    def test_two_breakpoints_no_fill(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(x, [0, 100], [5, 80]) == y,
            method="incremental",
        )
        delta = m.variables[f"pwl0{PWL_DELTA_SUFFIX}"]
        assert delta.labels.sizes[LP_SEG_DIM] == 1
        assert f"pwl0{PWL_X_LINK_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_Y_LINK_SUFFIX}" in m.constraints

    def test_creates_binary_indicator_vars(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(x, [0, 10, 50, 100], [5, 2, 20, 80]) == y,
            method="incremental",
        )
        assert f"pwl0{PWL_INC_BINARY_SUFFIX}" in m.variables
        binary = m.variables[f"pwl0{PWL_INC_BINARY_SUFFIX}"]
        assert binary.labels.sizes[LP_SEG_DIM] == 3
        assert f"pwl0{PWL_INC_LINK_SUFFIX}" in m.constraints

    def test_creates_order_constraints(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(x, [0, 10, 50, 100], [5, 2, 20, 80]) == y,
            method="incremental",
        )
        assert f"pwl0{PWL_INC_ORDER_SUFFIX}" in m.constraints

    def test_two_breakpoints_no_order_constraint(self) -> None:
        """With only one segment, there's no order constraint needed."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(x, [0, 100], [5, 80]) == y,
            method="incremental",
        )
        assert f"pwl0{PWL_INC_BINARY_SUFFIX}" in m.variables
        assert f"pwl0{PWL_INC_LINK_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_INC_ORDER_SUFFIX}" not in m.constraints

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
        x_pts = xr.DataArray([0, 10, 50, np.nan], dims=[BREAKPOINT_DIM])
        y_pts = xr.DataArray([0, 5, 20, np.nan], dims=[BREAKPOINT_DIM])
        m.add_piecewise_constraints(
            piecewise(x, x_pts, y_pts) == y,
            method="sos2",
        )
        lam = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        # First 3 should be valid, last masked
        assert (lam.labels.isel({BREAKPOINT_DIM: slice(None, 3)}) != -1).all()
        assert int(lam.labels.isel({BREAKPOINT_DIM: 3})) == -1

    def test_skip_nan_check_with_nan_raises(self) -> None:
        """skip_nan_check=True with NaN breakpoints raises ValueError."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        x_pts = xr.DataArray([0, 10, 50, np.nan], dims=[BREAKPOINT_DIM])
        y_pts = xr.DataArray([0, 5, 20, np.nan], dims=[BREAKPOINT_DIM])
        with pytest.raises(ValueError, match="skip_nan_check=True but breakpoints"):
            m.add_piecewise_constraints(
                piecewise(x, x_pts, y_pts) == y,
                method="sos2",
                skip_nan_check=True,
            )

    def test_skip_nan_check_without_nan(self) -> None:
        """skip_nan_check=True without NaN works fine (no mask computed)."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        x_pts = xr.DataArray([0, 10, 50, 100], dims=[BREAKPOINT_DIM])
        y_pts = xr.DataArray([0, 5, 20, 40], dims=[BREAKPOINT_DIM])
        m.add_piecewise_constraints(
            piecewise(x, x_pts, y_pts) == y,
            method="sos2",
            skip_nan_check=True,
        )
        lam = m.variables[f"pwl0{PWL_LAMBDA_SUFFIX}"]
        assert (lam.labels != -1).all()

    def test_sos2_interior_nan_raises(self) -> None:
        """SOS2 with interior NaN breakpoints raises ValueError."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        x_pts = xr.DataArray([0, np.nan, 50, 100], dims=[BREAKPOINT_DIM])
        y_pts = xr.DataArray([0, np.nan, 20, 40], dims=[BREAKPOINT_DIM])
        with pytest.raises(ValueError, match="non-trailing NaN"):
            m.add_piecewise_constraints(
                piecewise(x, x_pts, y_pts) == y,
                method="sos2",
            )


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
        return request.param

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
        return request.param

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


class TestLPDomainConstraints:
    """Tests for LP domain bound constraints."""

    def test_lp_domain_constraints_created(self) -> None:
        """LP method creates domain bound constraints."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        # Concave: slopes decreasing → y <= pw uses LP
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 100], [0, 40, 60]) >= y,
        )
        assert f"pwl0{PWL_LP_DOMAIN_SUFFIX}_lo" in m.constraints
        assert f"pwl0{PWL_LP_DOMAIN_SUFFIX}_hi" in m.constraints

    def test_lp_domain_constraints_multidim(self) -> None:
        """Domain constraints have entity dimension for per-entity breakpoints."""
        m = Model()
        x = m.add_variables(coords=[pd.Index(["a", "b"], name="entity")], name="x")
        y = m.add_variables(coords=[pd.Index(["a", "b"], name="entity")], name="y")
        x_pts = breakpoints({"a": [0, 50, 100], "b": [10, 60, 110]}, dim="entity")
        y_pts = breakpoints({"a": [0, 40, 60], "b": [5, 35, 55]}, dim="entity")
        m.add_piecewise_constraints(
            piecewise(x, x_pts, y_pts) >= y,
        )
        lo_name = f"pwl0{PWL_LP_DOMAIN_SUFFIX}_lo"
        hi_name = f"pwl0{PWL_LP_DOMAIN_SUFFIX}_hi"
        assert lo_name in m.constraints
        assert hi_name in m.constraints
        # Domain constraints should have the entity dimension
        assert "entity" in m.constraints[lo_name].labels.dims
        assert "entity" in m.constraints[hi_name].labels.dims


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
        m.add_piecewise_constraints(
            piecewise(x, [0, 10, 50, 100], [5, 2, 20, 80], active=u) == y,
            method="incremental",
        )
        assert f"pwl0{PWL_ACTIVE_BOUND_SUFFIX}" in m.constraints
        assert f"pwl0{PWL_DELTA_SUFFIX}" in m.variables

    def test_active_none_is_default(self) -> None:
        """Without active, formulation is identical to before."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        m.add_piecewise_constraints(
            piecewise(x, [0, 10, 50], [0, 5, 30]) == y,
            method="incremental",
        )
        assert f"pwl0{PWL_ACTIVE_BOUND_SUFFIX}" not in m.constraints

    def test_active_with_lp_method_raises(self) -> None:
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        u = m.add_variables(binary=True, name="u")
        with pytest.raises(ValueError, match="not supported with method='lp'"):
            m.add_piecewise_constraints(
                piecewise(x, [0, 50, 100], [0, 40, 60], active=u) >= y,
                method="lp",
            )

    def test_active_with_auto_lp_raises(self) -> None:
        """Auto selects LP for concave >=, but active is incompatible."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        u = m.add_variables(binary=True, name="u")
        with pytest.raises(ValueError, match="not supported with method='lp'"):
            m.add_piecewise_constraints(
                piecewise(x, [0, 50, 100], [0, 40, 60], active=u) >= y,
            )

    def test_incremental_inequality_with_active(self) -> None:
        """Inequality + active creates aux variable and active bound."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        u = m.add_variables(binary=True, name="u")
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 100], [0, 10, 50], active=u) >= y,
            method="incremental",
        )
        assert f"pwl0{PWL_AUX_SUFFIX}" in m.variables
        assert f"pwl0{PWL_ACTIVE_BOUND_SUFFIX}" in m.constraints
        assert "pwl0_ineq" in m.constraints

    def test_active_with_linear_expression(self) -> None:
        """Active can be a LinearExpression, not just a Variable."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")
        u = m.add_variables(binary=True, name="u")
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 100], [0, 10, 50], active=1 * u) == y,
            method="incremental",
        )
        assert f"pwl0{PWL_ACTIVE_BOUND_SUFFIX}" in m.constraints


# ===========================================================================
# Solver integration – active parameter
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
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 100], [0, 10, 50], active=u) == y,
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
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 100], [0, 10, 50], active=u) == y,
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
        Non-zero base (x₀=20, y₀=5) with u=0 must still force zero.

        Tests the x₀*u / y₀*u base term multiplication — would fail if
        base terms aren't multiplied by active.
        """
        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        y = m.add_variables(name="y")
        u = m.add_variables(binary=True, name="u")
        m.add_piecewise_constraints(
            piecewise(x, [20, 60, 100], [5, 20, 50], active=u) == y,
            method="incremental",
        )
        m.add_constraints(u <= 0, name="force_off")
        m.add_objective(y, sense="max")
        status, _ = m.solve(solver_name=solver_name)
        assert status == "ok"
        np.testing.assert_allclose(float(x.solution.values), 0, atol=1e-4)
        np.testing.assert_allclose(float(y.solution.values), 0, atol=1e-4)

    def test_incremental_inequality_active_off(self, solver_name: str) -> None:
        """Inequality with active=0: aux variable is 0, so y <= 0."""
        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        y = m.add_variables(lower=0, name="y")
        u = m.add_variables(binary=True, name="u")
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 100], [0, 10, 50], active=u) >= y,
            method="incremental",
        )
        m.add_constraints(u <= 0, name="force_off")
        m.add_objective(y, sense="max")
        status, _ = m.solve(solver_name=solver_name)
        assert status == "ok"
        np.testing.assert_allclose(float(y.solution.values), 0, atol=1e-4)

    def test_unit_commitment_pattern(self, solver_name: str) -> None:
        """Solver decides to commit: verifies correct fuel at operating point."""
        m = Model()
        p_min, p_max = 20.0, 100.0
        fuel_at_pmin, fuel_at_pmax = 10.0, 60.0

        power = m.add_variables(lower=0, upper=p_max, name="power")
        fuel = m.add_variables(name="fuel")
        u = m.add_variables(binary=True, name="commit")

        m.add_piecewise_constraints(
            piecewise(power, [p_min, p_max], [fuel_at_pmin, fuel_at_pmax], active=u)
            == fuel,
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
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 100], [0, 10, 50], active=u) == y,
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
        """SOS2: u=0 forces Σλ=0, collapsing x=0, y=0."""
        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        y = m.add_variables(name="y")
        u = m.add_variables(binary=True, name="u")
        m.add_piecewise_constraints(
            piecewise(x, [0, 50, 100], [0, 10, 50], active=u) == y,
            method="sos2",
        )
        m.add_constraints(u <= 0, name="force_off")
        m.add_objective(y, sense="max")
        status, _ = m.solve(solver_name=solver_name)
        assert status == "ok"
        np.testing.assert_allclose(float(x.solution.values), 0, atol=1e-4)
        np.testing.assert_allclose(float(y.solution.values), 0, atol=1e-4)

    def test_disjunctive_active_off(self, solver_name: str) -> None:
        """Disjunctive: u=0 forces Σz_k=0, collapsing x=0, y=0."""
        m = Model()
        x = m.add_variables(lower=0, upper=100, name="x")
        y = m.add_variables(name="y")
        u = m.add_variables(binary=True, name="u")
        m.add_piecewise_constraints(
            piecewise(
                x,
                segments([[0.0, 10.0], [50.0, 100.0]]),
                segments([[0.0, 5.0], [20.0, 80.0]]),
                active=u,
            )
            == y,
        )
        m.add_constraints(u <= 0, name="force_off")
        m.add_objective(y, sense="max")
        status, _ = m.solve(solver_name=solver_name)
        assert status == "ok"
        np.testing.assert_allclose(float(x.solution.values), 0, atol=1e-4)
        np.testing.assert_allclose(float(y.solution.values), 0, atol=1e-4)
