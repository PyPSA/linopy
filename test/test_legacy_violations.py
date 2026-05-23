"""
Legacy convention violations and v1 fixes.

Pairs ``@pytest.mark.legacy`` tests that document the surprising legacy
behaviour against ``@pytest.mark.v1`` tests that pin the v1 fix. Each
class corresponds to a section of ``arithmetics-design/convention.md``
and to one or more linopy bug reports.

Slice A — constant operand path (§5, §8, §9):
    §8  Shared dimensions must match exactly  → #708 / #586 / #550
    §5  User-supplied NaN raises              → #713 / PyPSA #1683
    §9  Non-shared dimensions broadcast       → (positive regression guard)

Slice B — expression-OP-expression / variable-OP-variable (§8 via `merge`):
    §8  Shared dimensions must match exactly  → #708 / #570 (expr+expr branch)

Slice C — absence propagation (§3, §6, §7):
    §6  Absence propagates through every operator → #712 (absent-as-zero)
    §3  isnull() is the unifying predicate        → #711
    §7  fillna()/.where() resolve absence         → (positive coverage)

Slice D — Variable.reindex / .reindex_like (§4 absence creation):
    §4  Reindexing extends coords and marks new slots absent
"""

from __future__ import annotations

import warnings
from collections.abc import Generator

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model
from linopy.config import LinopySemanticsWarning


@pytest.fixture
def m() -> Model:
    return Model()


@pytest.fixture
def time() -> pd.RangeIndex:
    return pd.RangeIndex(5, name="time")


@pytest.fixture
def x(m: Model, time: pd.RangeIndex):
    return m.add_variables(lower=0, coords=[time], name="x")


@pytest.fixture
def unsilenced() -> Generator[None, None, None]:
    """Drop the autouse fixture's LinopySemanticsWarning filter for one test."""
    with warnings.catch_warnings():
        warnings.simplefilter("always", LinopySemanticsWarning)
        yield


# =====================================================================
# §8 — Shared dimensions must match exactly (constant operand)
# =====================================================================


class TestExactAlignmentConstant:
    @pytest.mark.v1
    def test_add_same_size_different_labels_raises(
        self, x, time: pd.RangeIndex
    ) -> None:
        """
        #708 / #550 — same shape, different labels: legacy aligns by
        position; v1 raises.
        """
        other = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            dims=["time"],
            coords={"time": pd.Index([10, 11, 12, 13, 14], name="time")},
        )
        with pytest.raises(ValueError, match="exact"):
            x + other

    @pytest.mark.v1
    def test_mul_same_size_different_labels_raises(self, x) -> None:
        other = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            dims=["time"],
            coords={"time": pd.Index([10, 11, 12, 13, 14], name="time")},
        )
        with pytest.raises(ValueError, match="exact"):
            x * other

    @pytest.mark.v1
    def test_div_same_size_different_labels_raises(self, x) -> None:
        other = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            dims=["time"],
            coords={"time": pd.Index([10, 11, 12, 13, 14], name="time")},
        )
        with pytest.raises(ValueError, match="exact"):
            x / other

    @pytest.mark.v1
    def test_add_subset_constant_raises(self, x, time: pd.RangeIndex) -> None:
        """
        #711 / #708 — constant covers only some of the variable's
        coords. Legacy left-joins (silently drops the gap); v1 raises.
        """
        subset = xr.DataArray(
            [10.0, 20.0], dims=["time"], coords={"time": pd.Index([1, 3], name="time")}
        )
        with pytest.raises(ValueError, match="exact"):
            x + subset

    @pytest.mark.v1
    def test_mul_subset_constant_raises(self, x) -> None:
        subset = xr.DataArray(
            [10.0, 20.0], dims=["time"], coords={"time": pd.Index([1, 3], name="time")}
        )
        with pytest.raises(ValueError, match="exact"):
            x * subset

    @pytest.mark.legacy
    def test_add_same_size_different_labels_silent(self, x) -> None:
        """Document the legacy behaviour: silent positional alignment."""
        other = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            dims=["time"],
            coords={"time": pd.Index([10, 11, 12, 13, 14], name="time")},
        )
        # Legacy keeps left coords; the user's intended pairing by label is lost.
        result = x + other
        assert list(result.coords["time"].values) == [0, 1, 2, 3, 4]
        assert result.const.values.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]

    @pytest.mark.legacy
    def test_add_subset_constant_silent(self, x) -> None:
        """Document the legacy behaviour: silent left-join (gaps → 0)."""
        subset = xr.DataArray(
            [10.0, 20.0], dims=["time"], coords={"time": pd.Index([1, 3], name="time")}
        )
        result = x + subset
        # Legacy reindex_like fills the missing positions with 0 (additive fill).
        assert result.const.sel(time=0).item() == 0.0
        assert result.const.sel(time=1).item() == 10.0
        assert result.const.sel(time=3).item() == 20.0


class TestBroadcastNonSharedDim:
    """
    §9 — a dimension that exists only in one operand broadcasts freely.
    Runs under both semantics: this is unchanged behaviour.
    """

    def test_add_broadcast_introduces_new_dim(self, x) -> None:
        bcast = xr.DataArray(
            [10.0, 20.0], dims=["scenario"], coords={"scenario": [0, 1]}
        )
        result = x + bcast
        assert set(result.const.dims) == {"time", "scenario"}
        assert result.const.sizes == {"time": 5, "scenario": 2}

    def test_mul_broadcast_introduces_new_dim(self, x) -> None:
        bcast = xr.DataArray([2.0, 3.0], dims=["scenario"], coords={"scenario": [0, 1]})
        result = x * bcast
        assert set(result.coeffs.dims) == {"time", "scenario", "_term"}


# =====================================================================
# §5 — User-supplied NaN raises (covers #713 and PyPSA #1683)
# =====================================================================


class TestUserNaNRaises:
    @pytest.mark.v1
    def test_add_nan_dataarray_raises(self, x, time: pd.RangeIndex) -> None:
        nan_data = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        with pytest.raises(ValueError, match="NaN"):
            x + nan_data

    @pytest.mark.v1
    def test_mul_nan_dataarray_raises(self, x, time: pd.RangeIndex) -> None:
        nan_data = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        with pytest.raises(ValueError, match="NaN"):
            x * nan_data

    @pytest.mark.v1
    def test_div_nan_dataarray_raises(self, x, time: pd.RangeIndex) -> None:
        nan_data = xr.DataArray(
            [2.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        with pytest.raises(ValueError, match="NaN"):
            x / nan_data

    @pytest.mark.v1
    def test_add_nan_scalar_raises(self, x) -> None:
        with pytest.raises(ValueError, match="NaN"):
            x + float("nan")

    @pytest.mark.v1
    def test_mul_nan_scalar_raises(self, x) -> None:
        with pytest.raises(ValueError, match="NaN"):
            x * float("nan")

    @pytest.mark.v1
    def test_pypsa_1683_inf_times_zero_raises(self, x, time: pd.RangeIndex) -> None:
        """
        PyPSA #1683 — ``min_pu * nominal_fix`` with ``p_nom=inf`` and
        ``p_min_pu=0`` yields a NaN bound. v1 surfaces this at construction,
        not as a downstream solve failure.
        """
        nominal_fix = xr.DataArray(
            [np.inf, np.inf, np.inf, np.inf, np.inf],
            dims=["time"],
            coords={"time": time},
        )
        min_pu = xr.DataArray(
            [1.0, 0.0, 1.0, 1.0, 1.0], dims=["time"], coords={"time": time}
        )
        bound = min_pu * nominal_fix  # 0 * inf = NaN at time=1
        assert np.isnan(bound.values[1])
        with pytest.raises(ValueError, match="NaN"):
            x * bound

    @pytest.mark.legacy
    def test_add_nan_dataarray_silently_fills_with_zero(
        self, x, time: pd.RangeIndex
    ) -> None:
        """Document legacy: NaN in addend silently becomes 0 (#713)."""
        nan_data = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        result = x + nan_data
        assert result.const.sel(time=1).item() == 0.0  # NaN → 0

    @pytest.mark.legacy
    def test_mul_nan_dataarray_silently_fills_with_zero(
        self, x, time: pd.RangeIndex
    ) -> None:
        """
        Document legacy: NaN in multiplier silently becomes 0 — variable
        zeroed out at that slot (#713).
        """
        nan_data = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        result = x * nan_data
        assert result.coeffs.squeeze().sel(time=1).item() == 0.0


# =====================================================================
# Legacy emits LinopySemanticsWarning where v1 would diverge
# =====================================================================


class TestLegacyWarning:
    """
    One representative case per divergence class — not a tautology
    check; verifies the rollout signal users will actually see.
    """

    @pytest.mark.legacy
    def test_warn_on_mismatched_coords(self, x, unsilenced) -> None:
        other = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            dims=["time"],
            coords={"time": pd.Index([10, 11, 12, 13, 14], name="time")},
        )
        with pytest.warns(LinopySemanticsWarning):
            x + other

    @pytest.mark.legacy
    def test_warn_on_subset_constant(self, x, unsilenced) -> None:
        subset = xr.DataArray(
            [10.0, 20.0], dims=["time"], coords={"time": pd.Index([1, 3], name="time")}
        )
        with pytest.warns(LinopySemanticsWarning):
            x + subset

    @pytest.mark.legacy
    def test_warn_on_nan_in_user_constant(
        self, x, time: pd.RangeIndex, unsilenced
    ) -> None:
        nan_data = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        with pytest.warns(LinopySemanticsWarning):
            x + nan_data


# =====================================================================
# §8 — Shared dimensions must match exactly (expr+expr / var+var, merge path)
# =====================================================================


class TestExactAlignmentMerge:
    @pytest.fixture
    def x_other(self, m: Model):
        # Same shape, different labels — legacy uses positional override.
        return m.add_variables(
            lower=0,
            coords=[pd.Index([10, 11, 12, 13, 14], name="time")],
            name="x_other",
        )

    @pytest.fixture
    def x_subset(self, m: Model):
        # Subset coords on the same dim — legacy outer-joins (and pads).
        return m.add_variables(
            lower=0,
            coords=[pd.Index([1, 3], name="time")],
            name="x_subset",
        )

    @pytest.mark.v1
    def test_var_plus_var_different_labels_raises(self, x, x_other) -> None:
        with pytest.raises(ValueError, match="Coordinate mismatch"):
            x + x_other

    @pytest.mark.v1
    def test_expr_plus_expr_different_labels_raises(self, x, x_other) -> None:
        with pytest.raises(ValueError, match="Coordinate mismatch"):
            (1 * x) + (1 * x_other)

    @pytest.mark.v1
    def test_var_plus_var_subset_raises(self, x, x_subset) -> None:
        with pytest.raises(ValueError, match="Coordinate mismatch"):
            x + x_subset

    @pytest.mark.v1
    def test_var_minus_var_different_labels_raises(self, x, x_other) -> None:
        with pytest.raises(ValueError, match="Coordinate mismatch"):
            x - x_other

    @pytest.mark.v1
    def test_var_plus_var_same_coords_works(
        self, m: Model, time: pd.RangeIndex
    ) -> None:
        """Same coords on a shared dim is fine — regression guard."""
        a = m.add_variables(lower=0, coords=[time], name="a")
        b = m.add_variables(lower=0, coords=[time], name="b")
        result = a + b
        assert result.sizes["time"] == 5

    @pytest.mark.v1
    def test_var_plus_var_broadcast_non_shared_dim_works(
        self, m: Model, time: pd.RangeIndex
    ) -> None:
        """§9 regression guard for the merge path: non-shared dims broadcast."""
        a = m.add_variables(lower=0, coords=[time], name="a")
        b = m.add_variables(
            lower=0, coords=[pd.Index([0, 1], name="scenario")], name="b"
        )
        result = a + b
        assert set(result.coord_dims) == {"time", "scenario"}

    @pytest.mark.legacy
    def test_var_plus_var_different_labels_silent(self, x, x_other) -> None:
        """
        Document legacy: same-shape var+var aligns by position via
        override; the right-hand labels are silently dropped.
        """
        result = x + x_other
        # Left wins via override → time coords are x's [0..4], even though
        # x_other was time=[10..14]. The two terms are paired by position.
        assert list(result.coords["time"].values) == [0, 1, 2, 3, 4]

    @pytest.mark.legacy
    def test_warn_on_var_plus_var_different_labels(
        self, x, x_other, unsilenced
    ) -> None:
        with pytest.warns(LinopySemanticsWarning):
            x + x_other


# =====================================================================
# §6 — Absence propagates through every operator
# §3 — isnull() reports absent slots (covers #712 absent-as-zero)
# =====================================================================


class TestAbsencePropagation:
    @pytest.fixture
    def xs(self, x):
        # x.shift(time=1) → absent at time=0, present elsewhere.
        return x.shift(time=1)

    @pytest.mark.v1
    def test_to_linexpr_marks_absent_with_nan_const(self, xs) -> None:
        """
        Variable.to_linexpr() encodes absence as NaN const + NaN
        coeff + vars=-1, so §6 has something to propagate.
        """
        expr = xs.to_linexpr()
        assert np.isnan(expr.const.values[0])
        assert np.isnan(expr.coeffs.values[0, 0])
        assert int(expr.vars.values[0, 0]) == -1
        assert not np.isnan(expr.const.values[1:]).any()

    @pytest.mark.v1
    def test_isnull_reports_absent_slot(self, xs) -> None:
        """§3: isnull() reports the absent slot on a LinearExpression."""
        expr = xs.to_linexpr()
        assert bool(expr.isnull().values[0])
        assert not bool(expr.isnull().values[1:].any())

    @pytest.mark.v1
    def test_mul_scalar_preserves_absence(self, xs) -> None:
        """#712 — ``shifted * 3`` stays absent (not coeff=3, const=0)."""
        result = xs * 3
        assert np.isnan(result.const.values[0])
        assert np.isnan(result.coeffs.values[0, 0])
        assert bool(result.isnull().values[0])

    @pytest.mark.v1
    def test_add_scalar_preserves_absence(self, xs) -> None:
        """`shifted + 5` is absent at the shifted slot, not const=5."""
        result = xs + 5
        assert np.isnan(result.const.values[0])
        assert result.const.values[1:].tolist() == [5.0, 5.0, 5.0, 5.0]

    @pytest.mark.v1
    def test_sub_scalar_preserves_absence(self, xs) -> None:
        result = xs - 5
        assert np.isnan(result.const.values[0])
        assert result.const.values[1:].tolist() == [-5.0, -5.0, -5.0, -5.0]

    @pytest.mark.v1
    def test_div_scalar_preserves_absence(self, xs) -> None:
        result = xs / 2
        assert np.isnan(result.const.values[0])
        assert np.isnan(result.coeffs.values[0, 0])

    @pytest.mark.v1
    def test_add_present_variable_propagates_absence(self, xs, x) -> None:
        """`x + xs` is absent wherever xs is, even though x is fine there."""
        result = xs + x
        assert np.isnan(result.const.values[0])
        assert bool(result.isnull().values[0])
        assert not bool(result.isnull().values[1:].any())

    @pytest.mark.v1
    def test_absent_distinguishable_from_zero(self, x, xs) -> None:
        """
        #712 — under v1, ``x.shift(time=1) * 3`` and ``x * 0`` are
        distinct: the first is absent, the second is a present zero.
        """
        absent = xs * 3
        zero = x * 0
        assert bool(absent.isnull().values[0])
        assert not bool(zero.isnull().values[0])

    @pytest.mark.legacy
    def test_legacy_collapses_absent_to_zero(self, xs) -> None:
        """
        Document the #712 bug: legacy treats absent as 0 after `* 3`.

        The term ends up as ``coeffs=3 * vars=-1 + const=0`` — a
        ``coeff*sentinel`` term that evaluates to 0 at the solver layer.
        There is no NaN signal anywhere, so ``isnull()`` returns False and
        downstream code can't tell ``xs * 3`` apart from ``x * 0``.
        """
        result = xs * 3
        assert not np.isnan(result.const.values[0])
        assert not np.isnan(result.coeffs.values[0, 0])
        assert not bool(result.isnull().values[0])


class TestFillnaResolves:
    """§7 — fillna()/.where() are how the caller resolves an absent slot."""

    @pytest.fixture
    def xs(self, x):
        return x.shift(time=1)

    @pytest.mark.v1
    def test_expr_fillna_replaces_absent_const(self, xs) -> None:
        result = xs.to_linexpr().fillna(42)
        assert result.const.values[0] == 42.0
        assert result.const.values[1:].tolist() == [0.0, 0.0, 0.0, 0.0]
        assert not bool(result.isnull().values.any())

    @pytest.mark.v1
    def test_variable_fillna_numeric_returns_expression(self, xs) -> None:
        """
        A constant fill is not a variable, so the return type is a
        LinearExpression.
        """
        from linopy import LinearExpression

        result = xs.fillna(42)
        assert isinstance(result, LinearExpression)
        assert result.const.values[0] == 42.0

    @pytest.mark.v1
    def test_variable_fillna_zero_revives_slot_as_present_zero(self, xs) -> None:
        result = xs.fillna(0)
        assert not bool(result.isnull().values[0])
        assert result.const.values[0] == 0.0

    @pytest.mark.v1
    def test_outer_fillna_then_add_collapses_to_just_added(
        self, m: Model, time: pd.RangeIndex
    ) -> None:
        """
        Interpretation A — once `(x + y.shift())` is absent at slot 0,
        ``.fillna(0)`` revives the slot as the constant 0 (dead terms
        stay dead), and a subsequent ``+ x`` re-introduces only ``x[0]``.
        Compare ``x + y.shift().fillna(0) + x`` which would double-count
        ``x`` at slot 0 — the placement of fillna is load-bearing.
        """
        x = m.add_variables(lower=0, coords=[time], name="x")
        y = m.add_variables(lower=0, coords=[time], name="y")
        expr = (x + y.shift(time=1)).fillna(0) + x

        # At slot 0 the only live term is 1·x[0]; const is 0 → result == x[0].
        coeffs0 = expr.coeffs.values[0]
        vars0 = expr.vars.values[0]
        live = ~np.isnan(coeffs0)
        assert int(live.sum()) == 1
        assert float(coeffs0[live][0]) == 1.0
        assert int(vars0[live][0]) == int(x.labels.values[0])
        assert float(expr.const.values[0]) == 0.0

        # At slots 1+ all three terms are live (x[i] + y[i-1] + x[i]) — the
        # outer ``+ x`` is genuinely additive where y.shift was present.
        assert int((~np.isnan(expr.coeffs.values[1])).sum()) == 3

    @pytest.mark.v1
    def test_masked_variable_constraint_via_fillna(self) -> None:
        """
        v1 counterpart of ``test_masked_variable_model`` — under §6 the
        constraint ``x + y >= 10`` drops at the masked y slots, so the
        caller must say ``y.fillna(0)`` to keep ``x >= 10`` there.
        """
        m = Model()
        lower = pd.Series(0, range(10))
        x = m.add_variables(lower, name="x")
        mask = pd.Series([True] * 8 + [False, False])
        y = m.add_variables(lower, name="y", mask=mask)
        m.add_constraints(x + y.fillna(0), ">=", 10)
        m.add_constraints(y, ">=", 0)
        m.add_objective(2 * x + y)

        # The constraint x + y.fillna(0) >= 10 binds at every slot.
        rhs = m.constraints["con0"].rhs.values
        assert not np.isnan(rhs).any()


# =====================================================================
# §4 — Variable.reindex / .reindex_like create absence
# =====================================================================


class TestVariableReindex:
    """
    Reindexing past the original coords marks the new positions
    absent (labels=-1, lower/upper=NaN); §4 lists this as one of the
    named mechanisms for creating absence. Runs under both semantics:
    this is a new API that didn't exist on master.
    """

    def test_reindex_extends_with_absent(self, x, time: pd.RangeIndex) -> None:
        extended = pd.RangeIndex(8, name="time")
        result = x.reindex(time=extended)
        assert result.sizes["time"] == 8
        # Original slots 0..4 are preserved
        assert int(result.labels.values[0]) == int(x.labels.values[0])
        # New slots 5..7 are absent
        assert (result.labels.values[5:] == -1).all()
        assert np.isnan(result.lower.values[5:]).all()
        assert np.isnan(result.upper.values[5:]).all()

    def test_reindex_subset_drops_coords(self, x) -> None:
        """
        Reindex to a strict subset shrinks the variable (no absence
        introduced — those slots are just gone).
        """
        result = x.reindex(time=pd.RangeIndex(3, name="time"))
        assert result.sizes["time"] == 3
        assert not (result.labels.values == -1).any()

    def test_reindex_like_extends_with_absent(self, m: Model, x) -> None:
        wider = m.add_variables(
            lower=0, coords=[pd.RangeIndex(7, name="time")], name="wider"
        )
        result = x.reindex_like(wider)
        assert result.sizes["time"] == 7
        assert (result.labels.values[5:] == -1).all()

    @pytest.mark.v1
    def test_reindexed_variable_propagates_absence_in_arithmetic(
        self, x, time: pd.RangeIndex
    ) -> None:
        """
        §4 + §6 hand-off: a reindex-introduced absence flows through
        the next operator and is visible via isnull().
        """
        wider = x.reindex(time=pd.RangeIndex(7, name="time"))
        expr = wider * 3
        assert bool(expr.isnull().values[5:].all())
        assert not bool(expr.isnull().values[:5].any())
