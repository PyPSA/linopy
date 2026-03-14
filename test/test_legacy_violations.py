"""
Legacy convention violations.

Documents concrete bugs and surprising behaviors in the legacy arithmetic
convention.  Each test class corresponds to a reported issue or PR and
contains paired legacy_only / v1_only tests showing:

- **legacy_only** — what legacy actually does (the wrong/surprising behavior)
- **v1_only** — what v1 does instead (correct behavior, usually ValueError)

This file serves as a living catalog of *why* the v1 convention exists.

Related issues / PRs
====================

Positional alignment (override join):
    #586  — Constraint RHS matched by position, not label
    #550  — Silent data corruption with reordered coordinates
    #257  — .loc[] reorder undone by override

Subset / superset alignment (left join):
    #572  — Non-associative arithmetic with constants
    #569  — Variable vs Expression inconsistency
    #571  — Multiplication with subset constant differs between paths

User NaN silently swallowed:
    #620  — NaN in user data filled with neutral elements

Absent-slot NaN propagation:
    #620  — Multiplication doesn't propagate absence in legacy
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model
from linopy.variables import Variable

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def m() -> Model:
    return Model()


@pytest.fixture
def time() -> pd.RangeIndex:
    return pd.RangeIndex(5, name="time")


@pytest.fixture
def x(m: Model, time: pd.RangeIndex) -> Variable:
    return m.add_variables(lower=0, coords=[time], name="x")


@pytest.fixture
def y(m: Model, time: pd.RangeIndex) -> Variable:
    return m.add_variables(lower=0, coords=[time], name="y")


# ============================================================
# 1. Positional alignment: same shape, different labels (#586, #550)
# ============================================================


class TestPositionalAlignment:
    """
    Legacy uses override (positional) join when operands have matching sizes.
    This silently pairs values by array position, ignoring coordinate labels.

    Issues: #586, #550, #257
    """

    @pytest.mark.legacy_only
    def test_add_same_size_different_labels_silent(self, m: Model) -> None:
        """
        #550: Two variables with same shape but different labels get
        silently paired by position. Labels from the left operand win.
        """
        time_a = pd.Index([0, 1, 2, 3, 4], name="time")
        time_b = pd.Index([5, 6, 7, 8, 9], name="time")
        a = m.add_variables(lower=0, coords=[time_a], name="a")
        b = m.add_variables(lower=0, coords=[time_b], name="b")

        result = a + b
        # Legacy: silently pairs a[0] with b[5], a[1] with b[6], etc.
        # The result has a's labels (0-4), but b's variable IDs are from 5-9
        assert list(result.coords["time"].values) == [0, 1, 2, 3, 4]
        # b's variables are present despite the labels saying time=0..4
        b_var_ids = b.labels.values
        result_var_ids = result.vars.values[:, 1]  # second term is b
        np.testing.assert_array_equal(result_var_ids, b_var_ids)

    @pytest.mark.v1_only
    def test_add_same_size_different_labels_raises(self, m: Model) -> None:
        """v1: Mismatched labels raise ValueError with helpful message."""
        time_a = pd.Index([0, 1, 2, 3, 4], name="time")
        time_b = pd.Index([5, 6, 7, 8, 9], name="time")
        a = m.add_variables(lower=0, coords=[time_a], name="a")
        b = m.add_variables(lower=0, coords=[time_b], name="b")

        with pytest.raises(ValueError, match="Coordinate mismatch"):
            a + b

    @pytest.mark.legacy_only
    def test_mul_reordered_labels_silent(self, m: Model) -> None:
        """
        #550: Multiplying by a constant with reordered labels of the same
        size silently uses positional alignment, producing wrong results.
        """
        idx = pd.Index(["costs", "penalty"], name="effect")
        v = m.add_variables(lower=0, coords=[idx], name="v")
        # Reversed order — same labels, different positions
        factors = xr.DataArray(
            [2.0, 1.0],
            dims=["effect"],
            coords={"effect": pd.Index(["penalty", "costs"], name="effect")},
        )

        result = v * factors
        # Legacy: positional match → v["costs"] * 2.0, v["penalty"] * 1.0
        # But the user meant: v["costs"] * 1.0, v["penalty"] * 2.0
        assert result.coeffs.sel(effect="costs").item() == 2.0  # WRONG
        assert result.coeffs.sel(effect="penalty").item() == 1.0  # WRONG

    @pytest.mark.v1_only
    def test_mul_reordered_labels_raises(self, m: Model) -> None:
        """v1: Reordered labels on same dim raise ValueError."""
        idx = pd.Index(["costs", "penalty"], name="effect")
        v = m.add_variables(lower=0, coords=[idx], name="v")
        factors = xr.DataArray(
            [2.0, 1.0],
            dims=["effect"],
            coords={"effect": pd.Index(["penalty", "costs"], name="effect")},
        )

        with pytest.raises(ValueError, match="exact"):
            v * factors

    @pytest.mark.legacy_only
    def test_add_reordered_labels_positional(self, m: Model) -> None:
        """
        Same labels in different order: legacy silently uses positional
        alignment on addition too, producing wrong constant values.
        """
        idx_a = pd.Index(["A1", "A5", "A11", "A100"], name="item")
        x = m.add_variables(lower=0, coords=[idx_a], name="x")

        # Same labels, different order, same size → override join
        rhs = xr.DataArray(
            [100.0, 1.0, 5.0, 11.0],
            dims=["item"],
            coords={"item": pd.Index(["A100", "A1", "A5", "A11"], name="item")},
        )
        result = x + rhs
        # Legacy: positional match → A1 gets 100.0, A100 gets 11.0
        assert result.const.sel(item="A1").item() == 100.0  # WRONG
        assert result.const.sel(item="A100").item() == 11.0  # WRONG

    @pytest.mark.v1_only
    def test_add_reordered_labels_raises(self, m: Model) -> None:
        """v1: Reordered labels raise ValueError."""
        idx_a = pd.Index(["A1", "A5", "A11", "A100"], name="item")
        x = m.add_variables(lower=0, coords=[idx_a], name="x")

        rhs = xr.DataArray(
            [100.0, 1.0, 5.0, 11.0],
            dims=["item"],
            coords={"item": pd.Index(["A100", "A1", "A5", "A11"], name="item")},
        )
        with pytest.raises(ValueError, match="exact"):
            x + rhs


# ============================================================
# 2. Subset constant breaks associativity (#572)
# ============================================================


class TestSubsetConstantAssociativity:
    """
    Legacy uses left-join when a constant has different-sized coordinates.
    This drops coordinates that might be needed by a later operation,
    breaking associativity: (a + c) + b != a + (c + b).

    Issue: #572 (review by @FBumann)
    """

    @pytest.mark.legacy_only
    def test_add_order_matters(self, m: Model) -> None:
        """
        Adding a subset constant first vs last gives different results
        because left-join drops the constant's extra coordinates.
        """
        time3 = pd.RangeIndex(3, name="time")
        time5 = pd.RangeIndex(5, name="time")
        a = m.add_variables(lower=0, coords=[time3], name="a")
        b = m.add_variables(lower=0, coords=[time5], name="b")
        factor = xr.DataArray(
            [10.0, 20.0, 30.0, 40.0, 50.0],
            dims=["time"],
            coords={"time": time5},
        )

        # a + factor + b: factor left-joined to a's coords (0,1,2),
        # then merged with b (0..4). factor at time=3,4 is lost.
        r1 = a + factor + b
        # a + b + factor: a+b merged first (outer → 0..4),
        # then factor left-joined to (0..4). factor at time=3,4 preserved.
        r2 = a + b + factor

        # At time=3,4 the constant should be 40,50 but r1 loses them
        assert r1.const.sel(time=3).item() == 0.0  # WRONG: lost
        assert r2.const.sel(time=3).item() == 40.0  # correct

    @pytest.mark.v1_only
    def test_subset_constant_raises(self, m: Model) -> None:
        """v1: Subset constant on a shared dim raises ValueError."""
        time3 = pd.RangeIndex(3, name="time")
        a = m.add_variables(lower=0, coords=[time3], name="a")
        factor = xr.DataArray(
            [10.0, 20.0, 30.0, 40.0, 50.0],
            dims=["time"],
            coords={"time": pd.RangeIndex(5, name="time")},
        )

        with pytest.raises(ValueError, match="exact"):
            a + factor


# ============================================================
# 3. User NaN silently swallowed (#620)
# ============================================================


class TestUserNaNSwallowed:
    """
    Legacy silently fills NaN in user-supplied constants with neutral
    elements: 0 for addition, 0 for multiplication (zeroes out variable),
    1 for division (leaves variable unchanged). The fill values are
    inconsistent and hide data bugs.

    Issue: #620
    """

    @pytest.fixture
    def nan_data(self, time: pd.RangeIndex) -> xr.DataArray:
        vals = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        return xr.DataArray(vals, dims=["time"], coords={"time": time})

    @pytest.mark.legacy_only
    def test_add_nan_silently_filled_with_zero(
        self, x: Variable, nan_data: xr.DataArray
    ) -> None:
        """NaN in addend becomes 0 — user's missing data silently ignored."""
        result = x + nan_data
        assert not np.isnan(result.const.values).any()
        assert result.const.sel(time=1).item() == 0.0  # was NaN → 0

    @pytest.mark.legacy_only
    def test_mul_nan_silently_zeroes_variable(
        self, x: Variable, nan_data: xr.DataArray
    ) -> None:
        """NaN in multiplier becomes 0 — variable silently zeroed out."""
        result = x * nan_data
        assert not np.isnan(result.coeffs.squeeze().values).any()
        assert result.coeffs.squeeze().sel(time=1).item() == 0.0

    @pytest.mark.legacy_only
    def test_div_nan_silently_becomes_one(
        self, x: Variable, nan_data: xr.DataArray
    ) -> None:
        """
        NaN in divisor becomes 1 — variable silently left unchanged.
        Note: inconsistent with mul which fills with 0.
        """
        # Avoid division by zero at time=0 by using nan_data + 1
        divisor = nan_data.copy()
        divisor[0] = 2.0  # avoid 1/0
        result = x / divisor
        assert not np.isnan(result.coeffs.squeeze().values).any()
        # time=1 had NaN → filled with 1 → coefficient unchanged
        assert result.coeffs.squeeze().sel(time=1).item() == 1.0

    @pytest.mark.v1_only
    def test_add_nan_raises(self, x: Variable, nan_data: xr.DataArray) -> None:
        """v1: NaN in user data raises ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            x + nan_data

    @pytest.mark.v1_only
    def test_mul_nan_raises(self, x: Variable, nan_data: xr.DataArray) -> None:
        with pytest.raises(ValueError, match="NaN"):
            x * nan_data

    @pytest.mark.v1_only
    def test_div_nan_raises(self, x: Variable, nan_data: xr.DataArray) -> None:
        with pytest.raises(ValueError, match="NaN"):
            x / nan_data

    @pytest.mark.legacy_only
    def test_nan_fill_inconsistency(self, x: Variable, nan_data: xr.DataArray) -> None:
        """
        Legacy fills NaN with DIFFERENT values per operation:
        add→0, mul→0, div→1. This is internally inconsistent.
        """
        add_result = x + nan_data
        mul_result = x * nan_data
        divisor = nan_data.copy()
        divisor[0] = 2.0
        div_result = x / divisor

        nan_pos = 1  # time=1 has NaN in input
        add_fill = add_result.const.sel(time=nan_pos).item()
        mul_fill = mul_result.coeffs.squeeze().sel(time=nan_pos).item()
        div_fill = div_result.coeffs.squeeze().sel(time=nan_pos).item()

        assert add_fill == 0.0  # additive "identity"
        assert mul_fill == 0.0  # kills the variable
        assert div_fill == 1.0  # leaves variable unchanged
        # mul fills with 0 (destructive) but div fills with 1 (preserving)
        # — no consistent principle


# ============================================================
# 4. Variable vs Expression inconsistency (#569, #571)
# ============================================================


class TestVariableExpressionInconsistency:
    """
    Variable and Expression code paths produce different results for the
    same mathematical operation. x * c and (1*x) * c should be identical.

    Issues: #569, #571
    """

    @pytest.mark.legacy_only
    def test_mul_subset_var_vs_expr_same_result(self, m: Model) -> None:
        """
        Legacy: after the fix in #572, both paths produce the same result
        (fill with 0). Before #572, the expression path crashed.
        """
        coords = pd.RangeIndex(5, name="i")
        x = m.add_variables(lower=0, coords=[coords], name="x")
        subset = xr.DataArray([10.0, 30.0], dims=["i"], coords={"i": [1, 3]})

        var_result = x * subset
        expr_result = (1 * x) * subset

        # Both should produce identical coefficients
        np.testing.assert_array_equal(
            var_result.coeffs.squeeze().values,
            expr_result.coeffs.squeeze().values,
        )

    @pytest.mark.v1_only
    def test_mul_subset_both_raise(self, m: Model) -> None:
        """v1: Both paths raise the same error."""
        coords = pd.RangeIndex(5, name="i")
        x = m.add_variables(lower=0, coords=[coords], name="x")
        subset = xr.DataArray([10.0, 30.0], dims=["i"], coords={"i": [1, 3]})

        with pytest.raises(ValueError, match="exact"):
            x * subset
        with pytest.raises(ValueError, match="exact"):
            (1 * x) * subset


# ============================================================
# 5. Absent slot NaN not propagated in legacy (#620)
# ============================================================


class TestAbsentSlotPropagation:
    """
    Legacy does not mark absent variable slots as NaN in to_linexpr(),
    so multiplication/division cannot distinguish 'absent' from 'zero'.

    Issue: #620
    """

    @pytest.mark.legacy_only
    def test_absent_times_scalar_becomes_zero(self, x: Variable) -> None:
        """
        Legacy: absent slot * 3 becomes coeffs=3, const=0 (a valid
        zero-contribution term). The absence is lost.
        """
        xs = x.shift(time=1)  # time=0 is absent
        result = xs * 3
        # Legacy treats absent as zero → coeffs=3 * 0 = wait, actually
        # labels=-1 but coeffs=3 (label -1 is unused but coeff not NaN)
        assert not result.isnull().values[0]  # NOT absent — this is wrong

    @pytest.mark.v1_only
    def test_absent_times_scalar_stays_absent(self, x: Variable) -> None:
        """v1: absent slot * 3 stays absent (NaN propagates)."""
        xs = x.shift(time=1)
        result = xs * 3
        assert result.isnull().values[0]  # correctly absent
        assert not result.isnull().values[1]  # valid slot unaffected

    @pytest.mark.legacy_only
    def test_absent_indistinguishable_from_zero(self, x: Variable) -> None:
        """
        Legacy cannot tell apart an absent variable from a zero variable.
        Both produce isnull()=False after multiplication.
        """
        xs = x.shift(time=1)  # time=0 is absent
        result_absent = xs * 3
        result_zero = x * 0  # genuinely zero

        # Both look non-null under legacy — information lost
        assert not result_absent.isnull().values[0]
        assert not result_zero.isnull().values[0]

    @pytest.mark.v1_only
    def test_absent_distinguishable_from_zero(self, x: Variable) -> None:
        """v1: absent and zero are distinct."""
        xs = x.shift(time=1)
        result_absent = xs * 3
        result_zero = x * 0

        assert result_absent.isnull().values[0]  # absent
        assert not result_zero.isnull().values[0]  # zero but present

    @pytest.mark.legacy_only
    def test_fillna_noop_on_absent_variable(self, x: Variable) -> None:
        """
        Legacy: fillna(42) on a shifted variable does nothing because
        to_linexpr() doesn't produce NaN to fill.
        """
        xs = x.shift(time=1)
        result = xs.fillna(42)
        # The absent slot at time=0 has const=0 (not NaN), so fillna
        # has nothing to replace
        assert result.const.values[0] == 0.0  # should be 42

    @pytest.mark.v1_only
    def test_fillna_works_on_absent_variable(self, x: Variable) -> None:
        """v1: fillna(42) correctly fills the absent slot."""
        xs = x.shift(time=1)
        result = xs.fillna(42)
        assert result.const.values[0] == 42.0
        assert result.const.values[1] == 0.0  # valid slot unchanged
