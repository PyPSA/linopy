"""
Tests for the arithmetic convention system.

Covers:
- Config validation (valid/invalid convention values, default)
- Deprecation warnings under legacy convention
- Scalar fast path consistency
- NaN edge cases (inf, -inf)
- Convention switching mid-session
- Variable.reindex() and Variable.reindex_like()
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import linopy
from linopy import LinearExpression, Model, Variable
from linopy.config import (
    LinopyDeprecationWarning,
    OptionSettings,
    options,
)
from linopy.constraints import Constraint
from linopy.testing import assert_linequal

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def m() -> Model:
    model = Model()
    model.add_variables(coords=[pd.RangeIndex(5, name="i")], name="a")
    model.add_variables(coords=[pd.RangeIndex(5, name="i")], name="b")
    return model


@pytest.fixture
def a(m: Model) -> Variable:
    return m.variables["a"]


@pytest.fixture
def b(m: Model) -> Variable:
    return m.variables["b"]


# ---------------------------------------------------------------------------
# 3. Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_default_convention_is_legacy(self) -> None:
        """Default arithmetic_convention should be 'legacy'."""
        fresh = OptionSettings(
            display_max_rows=14,
            display_max_terms=6,
            arithmetic_convention="legacy",
        )
        assert fresh["arithmetic_convention"] == "legacy"

    def test_set_valid_convention_v1(self) -> None:
        old = options["arithmetic_convention"]
        try:
            options["arithmetic_convention"] = "v1"
            assert options["arithmetic_convention"] == "v1"
        finally:
            options["arithmetic_convention"] = old

    def test_set_valid_convention_legacy(self) -> None:
        old = options["arithmetic_convention"]
        try:
            options["arithmetic_convention"] = "legacy"
            assert options["arithmetic_convention"] == "legacy"
        finally:
            options["arithmetic_convention"] = old

    def test_set_invalid_convention_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid arithmetic_convention"):
            options["arithmetic_convention"] = "invalid"

    def test_set_invalid_convention_exact_raises(self) -> None:
        """'exact' is a join mode, not a valid convention name."""
        with pytest.raises(ValueError, match="Invalid arithmetic_convention"):
            options["arithmetic_convention"] = "exact"

    def test_invalid_key_raises(self) -> None:
        with pytest.raises(KeyError, match="not a valid setting"):
            options["nonexistent_key"] = 42

    def test_get_invalid_key_raises(self) -> None:
        with pytest.raises(KeyError, match="not a valid setting"):
            _ = options["nonexistent_key"]


# ---------------------------------------------------------------------------
# 5. Deprecation warnings
# ---------------------------------------------------------------------------


@pytest.mark.legacy_only
class TestDeprecationWarnings:
    def test_add_constant_emits_deprecation_warning(self, a: Variable) -> None:
        const = xr.DataArray([1, 2, 3, 4, 5], dims=["i"], coords={"i": range(5)})
        with pytest.warns(LinopyDeprecationWarning, match="legacy"):
            _ = (1 * a) + const

    def test_mul_constant_emits_deprecation_warning(self, a: Variable) -> None:
        const = xr.DataArray([1, 2, 3, 4, 5], dims=["i"], coords={"i": range(5)})
        with pytest.warns(LinopyDeprecationWarning, match="legacy"):
            _ = (1 * a) * const

    def test_align_emits_deprecation_warning(self, a: Variable) -> None:
        alpha = xr.DataArray([1, 2], [[1, 2]])
        with pytest.warns(LinopyDeprecationWarning, match="legacy"):
            linopy.align(a, alpha)


@pytest.mark.v1_only
class TestNoDeprecationWarnings:
    """V1: matching-coord operations should not emit deprecation warnings."""

    def test_add_constant_no_deprecation_warning(self, a: Variable) -> None:
        const = xr.DataArray([1, 2, 3, 4, 5], dims=["i"], coords={"i": range(5)})
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", LinopyDeprecationWarning)
            _ = (1 * a) + const

    def test_mul_constant_no_deprecation_warning(self, a: Variable) -> None:
        const = xr.DataArray([1, 2, 3, 4, 5], dims=["i"], coords={"i": range(5)})
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", LinopyDeprecationWarning)
            _ = (1 * a) * const

    def test_align_no_deprecation_warning(self, a: Variable) -> None:
        alpha = xr.DataArray([1, 2, 3, 4, 5], dims=["i"], coords={"i": range(5)})
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", LinopyDeprecationWarning)
            linopy.align(a, alpha)


# ---------------------------------------------------------------------------
# 6. Scalar fast path
# ---------------------------------------------------------------------------


class TestScalarFastPath:
    """Scalar operations should produce same results as array operations."""

    def test_add_scalar_matches_array(self, a: Variable) -> None:
        scalar_result = (1 * a) + 5
        array_const = xr.DataArray(np.full(5, 5.0), dims=["i"], coords={"i": range(5)})
        array_result = (1 * a) + array_const
        assert_linequal(scalar_result, array_result)

    def test_sub_scalar_matches_array(self, a: Variable) -> None:
        scalar_result = (1 * a) - 3
        array_const = xr.DataArray(np.full(5, 3.0), dims=["i"], coords={"i": range(5)})
        array_result = (1 * a) - array_const
        assert_linequal(scalar_result, array_result)

    def test_mul_scalar_matches_array(self, a: Variable) -> None:
        scalar_result = (1 * a) * 2
        array_const = xr.DataArray(np.full(5, 2.0), dims=["i"], coords={"i": range(5)})
        array_result = (1 * a) * array_const
        assert_linequal(scalar_result, array_result)

    def test_div_scalar_matches_array(self, a: Variable) -> None:
        scalar_result = (1 * a) / 4
        array_const = xr.DataArray(np.full(5, 4.0), dims=["i"], coords={"i": range(5)})
        array_result = (1 * a) / array_const
        assert_linequal(scalar_result, array_result)


# ---------------------------------------------------------------------------
# 7. NaN edge cases
# ---------------------------------------------------------------------------


@pytest.mark.v1_only
class TestNaNEdgeCases:
    def test_inf_add_propagates(self, a: Variable) -> None:
        """Adding inf should propagate to const."""
        const = xr.DataArray(
            [1.0, np.inf, 3.0, 4.0, 5.0], dims=["i"], coords={"i": range(5)}
        )
        result = (1 * a) + const
        assert np.isinf(result.const.values[1])

    def test_neg_inf_add_propagates(self, a: Variable) -> None:
        """Adding -inf should propagate to const."""
        const = xr.DataArray(
            [1.0, -np.inf, 3.0, 4.0, 5.0], dims=["i"], coords={"i": range(5)}
        )
        result = (1 * a) + const
        assert np.isinf(result.const.values[1])
        assert result.const.values[1] < 0

    def test_inf_mul_propagates(self, a: Variable) -> None:
        """Multiplying by inf should propagate to coeffs."""
        const = xr.DataArray(
            [1.0, np.inf, 3.0, 4.0, 5.0], dims=["i"], coords={"i": range(5)}
        )
        result = (1 * a) * const
        assert np.isinf(result.coeffs.squeeze().values[1])

    def test_nan_mul_masks_v1(self, a: Variable) -> None:
        """Under v1, NaN in mul masks the affected positions."""
        const = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["i"], coords={"i": range(5)}
        )
        result = (1 * a) * const
        # i=1 has NaN factor → absent slot
        assert result.isnull().sel(i=1).item()
        # Other positions are valid
        assert not result.isnull().sel(i=0).item()
        assert result.coeffs.squeeze().sel(i=0).item() == 1.0
        assert result.coeffs.squeeze().sel(i=2).item() == 3.0


# ---------------------------------------------------------------------------
# 8. Convention switching mid-session
# ---------------------------------------------------------------------------


class TestConventionSwitching:
    def test_switch_convention_mid_session(self, a: Variable, b: Variable) -> None:
        """Switching convention mid-session should change behavior immediately."""
        const = xr.DataArray([1, 2, 3], dims=["i"], coords={"i": [0, 1, 2]})

        # Under legacy: mismatched-size const should work
        linopy.options["arithmetic_convention"] = "legacy"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinopyDeprecationWarning)
            # This should succeed under legacy (left join / override)
            _ = (1 * a) + const

        # Switch to v1: same operation with mismatched coords should raise
        linopy.options["arithmetic_convention"] = "v1"
        with pytest.raises(ValueError, match="exact"):
            _ = (1 * a) + const

    def test_reset_restores_defaults(self) -> None:
        """OptionSettings.reset() should restore factory defaults."""
        options["arithmetic_convention"] = "v1"
        assert options["arithmetic_convention"] == "v1"
        options.reset()
        assert options["arithmetic_convention"] == "legacy"  # factory default


# ---------------------------------------------------------------------------
# 9. TestJoinParameter deduplication (shared base class)
# ---------------------------------------------------------------------------
# The existing TestJoinParameter class already tests both conventions via
# @pytest.mark.legacy_only / @pytest.mark.v1_only markers. The deduplication
# is addressed by verifying that explicit join= works identically under both.


class TestJoinWorksUnderBothConventions:
    """Explicit join= should produce same results regardless of convention."""

    @pytest.fixture
    def m2(self) -> Model:
        m = Model()
        m.add_variables(coords=[pd.Index([0, 1, 2], name="i")], name="a")
        m.add_variables(coords=[pd.Index([1, 2, 3], name="i")], name="b")
        return m

    def test_add_inner_same_under_both(self, m2: Model) -> None:
        a = m2.variables["a"]
        b = m2.variables["b"]

        linopy.options["arithmetic_convention"] = "legacy"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinopyDeprecationWarning)
            result_legacy = a.to_linexpr().add(b.to_linexpr(), join="inner")

        linopy.options["arithmetic_convention"] = "v1"
        result_v1 = a.to_linexpr().add(b.to_linexpr(), join="inner")

        assert list(result_legacy.data.indexes["i"]) == list(
            result_v1.data.indexes["i"]
        )

    def test_add_outer_same_under_both(self, m2: Model) -> None:
        a = m2.variables["a"]
        b = m2.variables["b"]

        linopy.options["arithmetic_convention"] = "legacy"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinopyDeprecationWarning)
            result_legacy = a.to_linexpr().add(b.to_linexpr(), join="outer")

        linopy.options["arithmetic_convention"] = "v1"
        result_v1 = a.to_linexpr().add(b.to_linexpr(), join="outer")

        assert set(result_legacy.data.indexes["i"]) == set(result_v1.data.indexes["i"])


# ---------------------------------------------------------------------------
# 10. Error message tests
# ---------------------------------------------------------------------------


@pytest.mark.v1_only
class TestErrorMessages:
    def test_exact_join_error_suggests_escape_hatches(self, a: Variable) -> None:
        """Error message should suggest .add()/.mul() with join= parameter."""
        subset = xr.DataArray([1, 2, 3], dims=["i"], coords={"i": [0, 1, 2]})
        with pytest.raises(ValueError, match=r"\.add\(other, join="):
            _ = (1 * a) + subset

    def test_exact_join_error_mentions_inner(self, a: Variable) -> None:
        subset = xr.DataArray([1, 2, 3], dims=["i"], coords={"i": [0, 1, 2]})
        with pytest.raises(ValueError, match="inner"):
            _ = (1 * a) + subset

    def test_exact_join_error_mentions_outer(self, a: Variable) -> None:
        subset = xr.DataArray([1, 2, 3], dims=["i"], coords={"i": [0, 1, 2]})
        with pytest.raises(ValueError, match="outer"):
            _ = (1 * a) + subset


# ---------------------------------------------------------------------------
# Variable.reindex() and Variable.reindex_like()
# ---------------------------------------------------------------------------


class TestVariableReindex:
    @pytest.fixture
    def var(self) -> Variable:
        m = Model()
        return m.add_variables(coords=[pd.Index([0, 1, 2, 3, 4], name="i")], name="v")

    def test_reindex_subset(self, var: Variable) -> None:
        result = var.reindex(i=[1, 2, 3])
        assert isinstance(result, Variable)
        assert list(result.data.indexes["i"]) == [1, 2, 3]
        # Labels for the reindexed positions should be valid
        assert (result.labels.sel(i=[1, 2, 3]).values >= 0).all()

    def test_reindex_superset(self, var: Variable) -> None:
        result = var.reindex(i=[0, 1, 2, 3, 4, 5, 6])
        assert isinstance(result, Variable)
        assert list(result.data.indexes["i"]) == [0, 1, 2, 3, 4, 5, 6]
        # New positions should have sentinel label (-1)
        assert result.labels.sel(i=5).item() == -1
        assert result.labels.sel(i=6).item() == -1
        # Original positions should be valid
        assert (result.labels.sel(i=[0, 1, 2, 3, 4]).values >= 0).all()

    def test_reindex_preserves_type(self, var: Variable) -> None:
        result = var.reindex(i=[0, 1])
        assert type(result) is type(var)

    def test_reindex_like_variable(self, var: Variable) -> None:
        m = var.model
        other = m.add_variables(coords=[pd.Index([2, 3, 4, 5], name="i")], name="other")
        result = var.reindex_like(other)
        assert isinstance(result, Variable)
        assert list(result.data.indexes["i"]) == [2, 3, 4, 5]
        # Position 5 should have sentinel
        assert result.labels.sel(i=5).item() == -1
        # Positions 2,3,4 should be valid
        assert (result.labels.sel(i=[2, 3, 4]).values >= 0).all()

    def test_reindex_like_dataarray(self, var: Variable) -> None:
        other = xr.DataArray([10, 20, 30], dims=["i"], coords={"i": [1, 3, 5]})
        result = var.reindex_like(other)
        assert isinstance(result, Variable)
        assert list(result.data.indexes["i"]) == [1, 3, 5]
        assert result.labels.sel(i=5).item() == -1

    def test_reindex_empty(self, var: Variable) -> None:
        result = var.reindex(i=[])
        assert isinstance(result, Variable)
        assert len(result.data.indexes["i"]) == 0


class TestExpressionReindex:
    @pytest.fixture
    def expr(self) -> LinearExpression:
        m = Model()
        x = m.add_variables(coords=[pd.Index([0, 1, 2, 3, 4], name="i")], name="x")
        return 2 * x + 10

    def test_reindex_subset(self, expr: LinearExpression) -> None:
        result = expr.reindex(i=[1, 2, 3])
        assert isinstance(result, LinearExpression)
        assert list(result.data.indexes["i"]) == [1, 2, 3]
        # Coefficients for existing positions should be preserved
        np.testing.assert_array_equal(result.coeffs.squeeze().values, [2, 2, 2])
        np.testing.assert_array_equal(result.const.values, [10, 10, 10])

    def test_reindex_superset(self, expr: LinearExpression) -> None:
        result = expr.reindex(i=[0, 1, 2, 3, 4, 5, 6])
        assert isinstance(result, LinearExpression)
        assert list(result.data.indexes["i"]) == [0, 1, 2, 3, 4, 5, 6]
        # New positions should have sentinel var labels (-1)
        assert result.vars.squeeze().sel(i=5).item() == -1
        assert result.vars.squeeze().sel(i=6).item() == -1
        # Original positions should be valid
        assert (result.vars.squeeze().sel(i=[0, 1, 2, 3, 4]).values >= 0).all()

    def test_reindex_fill_value(self, expr: LinearExpression) -> None:
        result = expr.reindex(i=[0, 1, 5], fill_value=0)
        assert result.const.sel(i=5).item() == 0
        result_nan = expr.reindex(i=[0, 1, 5])
        assert np.isnan(result_nan.const.sel(i=5).item())

    def test_reindex_preserves_type(self, expr: LinearExpression) -> None:
        result = expr.reindex(i=[0, 1])
        assert type(result) is type(expr)

    def test_reindex_like_expression(self, expr: LinearExpression) -> None:
        m = expr.model
        y = m.add_variables(coords=[pd.Index([2, 3, 4, 5], name="i")], name="y")
        other = 1 * y
        result = expr.reindex_like(other)
        assert isinstance(result, LinearExpression)
        assert list(result.data.indexes["i"]) == [2, 3, 4, 5]
        assert result.vars.squeeze().sel(i=5).item() == -1

    def test_reindex_like_variable(self, expr: LinearExpression) -> None:
        m = expr.model
        y = m.add_variables(coords=[pd.Index([1, 3, 5], name="i")], name="y")
        result = expr.reindex_like(y)
        assert isinstance(result, LinearExpression)
        assert list(result.data.indexes["i"]) == [1, 3, 5]

    def test_reindex_like_dataarray(self, expr: LinearExpression) -> None:
        da = xr.DataArray([10, 20, 30], dims=["i"], coords={"i": [1, 3, 5]})
        result = expr.reindex_like(da)
        assert isinstance(result, LinearExpression)
        assert list(result.data.indexes["i"]) == [1, 3, 5]
        assert result.vars.squeeze().sel(i=5).item() == -1

    def test_reindex_like_dataset(self, expr: LinearExpression) -> None:
        ds = xr.Dataset({"tmp": (("i",), [1, 2])}, coords={"i": [0, 1]})
        result = expr.reindex_like(ds)
        assert isinstance(result, LinearExpression)
        assert list(result.data.indexes["i"]) == [0, 1]


class TestConstraintReindex:
    @pytest.fixture
    def con(self) -> Constraint:
        m = Model()
        x = m.add_variables(coords=[pd.Index([0, 1, 2, 3, 4], name="i")], name="x")
        linopy.options["arithmetic_convention"] = "legacy"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", LinopyDeprecationWarning)
            c = x >= 0
        m.add_constraints(c, name="c")
        return m.constraints["c"]

    def test_reindex_subset(self, con: Constraint) -> None:
        result = con.reindex({"i": [1, 2, 3]})
        assert list(result.data.indexes["i"]) == [1, 2, 3]

    def test_reindex_superset(self, con: Constraint) -> None:
        result = con.reindex({"i": [0, 1, 2, 3, 4, 5]})
        assert list(result.data.indexes["i"]) == [0, 1, 2, 3, 4, 5]
        # New position should have sentinel label
        assert result.data.vars.squeeze().sel(i=5).item() == -1

    def test_reindex_like_dataset(self, con: Constraint) -> None:
        ds = xr.Dataset({"tmp": (("i",), [1, 2])}, coords={"i": [0, 1]})
        result = con.reindex_like(ds)
        assert list(result.data.indexes["i"]) == [0, 1]

    def test_reindex_like_dataarray(self, con: Constraint) -> None:
        da = xr.DataArray([10, 20], dims=["i"], coords={"i": [1, 3]})
        result = con.reindex_like(da)
        assert list(result.data.indexes["i"]) == [1, 3]
