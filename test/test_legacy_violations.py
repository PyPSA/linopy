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

Slice E — named-method join= + constraint RHS (§10, §12):
    §10 .add/.sub/.mul/.div/.le/.ge/.eq accept explicit join=
    §12 NaN in constraint RHS raises (v1)              → PyPSA #1683
    §12 Coord mismatch in RHS raises (v1)              → #707

Slice G — reductions skip absent slots (§13):
    §13 sum / groupby.sum skip absent, sum of none is the zero expression

Slice F — auxiliary-coordinate conflicts (§11):
    §11 Non-dim coord conflict raises (v1)             → #295
    §11 Non-conflicting aux coords propagate through arithmetic
"""

from __future__ import annotations

import operator
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


_OPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
}


class TestExactAlignmentConstant:
    @pytest.mark.v1
    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_same_size_different_labels_raises(self, x, op) -> None:
        """
        #708 / #550 — same shape, different labels: legacy aligns by
        position; v1 raises. Holds for every binary operator.
        """
        other = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            dims=["time"],
            coords={"time": pd.Index([10, 11, 12, 13, 14], name="time")},
        )
        with pytest.raises(ValueError, match="exact"):
            _OPS[op](x, other)

    @pytest.mark.v1
    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_subset_constant_raises(self, x, op) -> None:
        """
        #711 / #708 — constant covers only some of the variable's
        coords. Legacy left-joins (silently drops the gap); v1 raises.
        """
        subset = xr.DataArray(
            [10.0, 20.0], dims=["time"], coords={"time": pd.Index([1, 3], name="time")}
        )
        with pytest.raises(ValueError, match="exact"):
            _OPS[op](x, subset)

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
    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_nan_dataarray_raises(self, x, time: pd.RangeIndex, op) -> None:
        # Use [2, NaN, 3, 4, 5] so div doesn't trip on a 0 divisor at slot 0.
        nan_data = xr.DataArray(
            [2.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        with pytest.raises(ValueError, match="NaN"):
            _OPS[op](x, nan_data)

    @pytest.mark.v1
    @pytest.mark.parametrize("op", ["add", "sub", "mul"])
    def test_nan_scalar_raises(self, x, op) -> None:
        # Skip div: ``x / nan`` raises *before* our check (TypeError on
        # the unary negation in ``__div__``); the scalar-NaN scenario for
        # div is the same code path as for mul.
        with pytest.raises(ValueError, match="NaN"):
            _OPS[op](x, float("nan"))

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
    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_scalar_op_preserves_absence(self, xs, op) -> None:
        """
        #712 — `shifted OP scalar` stays absent at the shifted slot.
        Holds for every binary operator: const and coeffs both NaN.
        """
        result = _OPS[op](xs, 3)
        assert np.isnan(result.const.values[0])
        assert np.isnan(result.coeffs.values[0, 0])
        assert bool(result.isnull().values[0])
        # And the present slots carry the expected per-op value.
        expected_const = {"add": 3.0, "sub": -3.0, "mul": 0.0, "div": 0.0}[op]
        assert (result.const.values[1:] == expected_const).all()

    @pytest.mark.v1
    def test_add_present_variable_propagates_absence(self, xs, x) -> None:
        """`x + xs` is absent wherever xs is, even though x is fine there."""
        result = xs + x
        assert np.isnan(result.const.values[0])
        assert bool(result.isnull().values[0])
        assert not bool(result.isnull().values[1:].any())

    @pytest.mark.v1
    def test_merge_absorbs_dead_terms_at_absent_slot(self, xs, x) -> None:
        """
        §1/§2 storage invariant — ``const.isnull()`` at a slot implies
        every term at that slot has ``coeffs = NaN`` and ``vars = -1``.
        ``xs + x`` merges xs's absent slot with x's live term; the live
        term must be absorbed, not silently kept alongside a NaN const.
        Regression guard for ``_absorb_absence`` (commit 4d87a05).
        """
        result = xs + x
        assert np.isnan(result.coeffs.values[0]).all()
        assert (result.vars.values[0] == -1).all()

    @pytest.mark.v1
    def test_merge_absorbs_dead_terms_multi_operand(
        self, m: Model, time: pd.RangeIndex
    ) -> None:
        """
        Same invariant on a 3-operand merge: a regression that absorbs
        only on the binary path would still leave one live term at the
        absent slot here.
        """
        x = m.add_variables(lower=0, coords=[time], name="x")
        y = m.add_variables(lower=0, coords=[time], name="y")
        xs = x.shift(time=1)
        result = (1 * x) + (1 * y) + xs
        assert np.isnan(result.coeffs.values[0]).all()
        assert (result.vars.values[0] == -1).all()
        # And the present rows still carry all three live terms.
        assert (~np.isnan(result.coeffs.values[1:])).all()

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


# =====================================================================
# §10 — named-method join= argument (opt-in alignment)
# =====================================================================


class TestNamedMethodJoin:
    """
    Under v1 the bare operators raise on coord mismatch (§8). The
    named methods let the caller opt in to a specific join mode.
    """

    @pytest.fixture
    def subset(self, time: pd.RangeIndex) -> xr.DataArray:
        return xr.DataArray(
            [10.0, 30.0], dims=["time"], coords={"time": pd.Index([1, 3], name="time")}
        )

    @pytest.mark.v1
    def test_add_join_inner_intersects(self, x, subset) -> None:
        """`.add(other, join="inner")` picks the intersection of coords."""
        result = x.add(subset, join="inner")
        assert list(result.coords["time"].values) == [1, 3]

    @pytest.mark.v1
    def test_add_join_outer_fills(self, x, subset) -> None:
        """`.add(other, join="outer")` unions coords (gaps are filled)."""
        result = x.add(subset, join="outer")
        assert list(result.coords["time"].values) == [0, 1, 2, 3, 4]

    @pytest.mark.v1
    def test_mul_join_inner(self, x, subset) -> None:
        result = x.mul(subset, join="inner")
        assert list(result.coords["time"].values) == [1, 3]

    @pytest.mark.v1
    def test_le_join_inner_on_subset_rhs(self, x, subset) -> None:
        """`.le(rhs, join="inner")` lets a subset RHS through cleanly."""
        result = x.le(subset, join="inner")
        assert list(result.coords["time"].values) == [1, 3]

    @pytest.mark.v1
    def test_bare_op_still_raises_on_mismatch(self, x, subset) -> None:
        """`x + subset` (no `join=`) still raises — opt-in is required."""
        with pytest.raises(ValueError, match="exact"):
            x + subset


# =====================================================================
# §12 — constraints follow the same rules
# =====================================================================


class TestConstraintRHS:
    @pytest.mark.v1
    def test_subset_rhs_raises(self, x) -> None:
        subset = xr.DataArray(
            [10.0, 20.0],
            dims=["time"],
            coords={"time": pd.Index([1, 3], name="time")},
        )
        with pytest.raises(ValueError, match="exact"):
            x <= subset

    @pytest.mark.v1
    def test_nan_rhs_raises(self, x, time: pd.RangeIndex) -> None:
        """
        §5/§12 — a NaN in a user-supplied RHS raises, never silently
        becomes "no constraint" the way legacy auto_mask treats it.
        """
        nan_rhs = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        with pytest.raises(ValueError, match="NaN"):
            x <= nan_rhs

    @pytest.mark.v1
    def test_pypsa_1683_nan_rhs_raises(self, x, time: pd.RangeIndex) -> None:
        """
        PyPSA #1683 on the constraint side — ``min_pu * nominal_fix``
        with ``p_nom=inf`` and ``p_min_pu=0`` yields NaN at the bad slot;
        v1 raises at construction instead of silently passing NaN to
        the solver.
        """
        nominal = xr.DataArray([np.inf] * 5, dims=["time"], coords={"time": time})
        min_pu = xr.DataArray(
            [1.0, 0.0, 1.0, 1.0, 1.0], dims=["time"], coords={"time": time}
        )
        bound = min_pu * nominal  # 0*inf = NaN at time=1
        with pytest.raises(ValueError, match="NaN"):
            x >= bound

    @pytest.mark.v1
    def test_absence_propagates_to_rhs_drops_constraint(
        self, x, time: pd.RangeIndex
    ) -> None:
        """
        §6 → §12: a constraint over an absent LHS slot yields NaN RHS,
        which downstream auto-mask interprets as "no constraint here".
        """
        xs = x.shift(time=1)
        # xs is absent at time=0; the constraint's RHS at that slot
        # should be NaN (no constraint), not 10.
        constraint = xs >= 10
        rhs = constraint.rhs.values
        assert np.isnan(rhs[0])
        assert (rhs[1:] == 10).all()

    @pytest.mark.v1
    def test_subset_rhs_eq_raises(self, x) -> None:
        """§12 — equality comparison aligns by §8 like ``<=``/``>=``."""
        subset = xr.DataArray(
            [10.0, 20.0],
            dims=["time"],
            coords={"time": pd.Index([1, 3], name="time")},
        )
        with pytest.raises(ValueError, match="exact"):
            x == subset

    @pytest.mark.v1
    def test_nan_rhs_eq_raises(self, x, time: pd.RangeIndex) -> None:
        """§5/§12 — a NaN in an equality RHS raises like ``<=`` does."""
        nan_rhs = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        with pytest.raises(ValueError, match="NaN"):
            x == nan_rhs

    @pytest.mark.v1
    def test_absence_propagates_to_rhs_eq_drops_constraint(self, x) -> None:
        """§6 → §12 on equality — absent LHS slot drops the constraint."""
        xs = x.shift(time=1)
        constraint = xs == 10
        rhs = constraint.rhs.values
        assert np.isnan(rhs[0])
        assert (rhs[1:] == 10).all()

    @pytest.mark.legacy
    def test_nan_rhs_silently_treated_as_unconstrained(
        self, x, time: pd.RangeIndex
    ) -> None:
        """
        Document the legacy auto_mask path: a NaN RHS is silently
        kept as NaN and the constraint at that row is later dropped.
        """
        nan_rhs = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        constraint = x <= nan_rhs
        assert np.isnan(constraint.rhs.values[1])

    @pytest.mark.legacy
    def test_warn_on_nan_rhs(self, x, time: pd.RangeIndex, unsilenced) -> None:
        nan_rhs = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        with pytest.warns(LinopySemanticsWarning):
            x <= nan_rhs


# =====================================================================
# §13 — reductions skip absent slots (not propagate)
# =====================================================================


class TestReductionsSkipAbsent:
    """
    Per §13, ``sum`` / ``groupby.sum`` skip absent slots rather than
    propagating them — the only asymmetry against §6's binary-operator
    rule. The expected behaviour falls out of xarray's ``skipna=True``
    default; these tests pin it under v1 so future changes don't drift.

    Scope: §13 also names ``mean``, ``resample``, and ``coarsen``, but
    those are not yet exposed on ``LinearExpression`` (see #703). The
    spec text is the rule they will follow when implemented; tests
    belong with the implementation PR.
    """

    @pytest.fixture
    def xs(self, x):
        return x.shift(time=1)

    @pytest.mark.v1
    def test_sum_over_dim_skips_absent(self, xs) -> None:
        """
        ``(xs + 5).sum('time')`` skips the absent slot at t=0 and
        sums the four present 5s → 20.
        """
        result = (xs + 5).sum("time")
        assert float(result.const) == 20.0

    @pytest.mark.v1
    def test_sum_no_dim_skips_absent(self, xs) -> None:
        result = (xs + 5).sum()
        assert float(result.const) == 20.0

    @pytest.mark.v1
    def test_sum_of_all_absent_is_zero(self, x) -> None:
        """§13 — "the sum of none is the zero expression.""" ""
        all_absent = x.shift(time=10).to_linexpr()
        assert bool(all_absent.isnull().all().item())
        result = all_absent.sum("time")
        assert float(result.const) == 0.0

    @pytest.mark.v1
    def test_groupby_sum_skips_absent(self, xs) -> None:
        """Each group's sum drops absent members, just like ``.sum``."""
        groups = xr.DataArray(
            [0, 0, 1, 1, 1], dims=["time"], coords={"time": xs.coords["time"]}
        )
        result = (xs + 5).groupby(groups).sum()
        # group 0: [NaN, 5] → 5; group 1: [5, 5, 5] → 15
        assert result.const.values.tolist() == [5.0, 15.0]


# =====================================================================
# §11 — auxiliary (non-dim) coordinate conflicts raise (covers #295)
# =====================================================================


class TestAuxCoordConflict:
    """
    Per §11, an auxiliary (non-dim) coord that two operands carry
    with disagreeing values must raise — xarray silently drops the
    conflict in arithmetic, which is the #295 bug.
    """

    @pytest.fixture
    def A(self) -> pd.Index:
        return pd.Index([1, 2, 3], name="A")

    @pytest.mark.v1
    def test_expr_plus_dataarray_aux_conflict_raises(self, m: Model, A) -> None:
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        const = xr.DataArray(
            [10.0, 20.0, 30.0],
            dims=["A"],
            coords={"A": A, "B": ("A", [400, 400, 500])},
        )
        with pytest.raises(ValueError, match="Auxiliary coordinate"):
            v + const

    @pytest.mark.v1
    def test_var_plus_var_aux_conflict_raises(self, m: Model, A) -> None:
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        w = m.add_variables(lower=0, coords=[A], name="w").assign_coords(
            B=("A", [400, 400, 500])
        )
        with pytest.raises(ValueError, match="Auxiliary coordinate"):
            v + w

    @pytest.mark.v1
    def test_mul_constant_aux_conflict_raises(self, m: Model, A) -> None:
        """Same rule on the multiplication path — not just ``+``."""
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        const = xr.DataArray(
            [2.0, 3.0, 4.0],
            dims=["A"],
            coords={"A": A, "B": ("A", [400, 400, 500])},
        )
        with pytest.raises(ValueError, match="Auxiliary coordinate"):
            v * const

    @pytest.mark.v1
    def test_constraint_aux_conflict_raises(self, m: Model, A) -> None:
        """§11 reaches constraint construction via the same machinery."""
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        const = xr.DataArray(
            [10.0, 20.0, 30.0],
            dims=["A"],
            coords={"A": A, "B": ("A", [400, 400, 500])},
        )
        with pytest.raises(ValueError, match="Auxiliary coordinate"):
            v == const

    @pytest.mark.v1
    def test_scalar_isel_aux_conflict_raises(self, m: Model, A) -> None:
        """
        Scalar isels leave the indexed dim as a non-dim coord whose
        value differs between operands picked at different positions.
        """
        v = m.add_variables(lower=0, coords=[A], name="v")
        a0 = (1 * v).isel({"A": 0})  # scalar A=1
        a1 = (1 * v).isel({"A": 1})  # scalar A=2
        with pytest.raises(ValueError, match="Auxiliary coordinate"):
            a0 + a1

    @pytest.mark.v1
    def test_isel_with_drop_true_avoids_conflict(self, m: Model, A) -> None:
        """
        The §11 escape hatch the convention recommends: drop the
        leftover scalar coord with ``isel(..., drop=True)``.
        """
        v = m.add_variables(lower=0, coords=[A], name="v")
        a0 = (1 * v).isel({"A": 0}, drop=True)
        a1 = (1 * v).isel({"A": 1}, drop=True)
        result = a0 + a1  # no aux coord → no conflict
        assert "A" not in result.coords

    @pytest.mark.legacy
    def test_aux_conflict_silently_keeps_left(self, m: Model, A) -> None:
        """
        Document legacy: a conflict is silently resolved by keeping
        the left operand's aux coord — the right operand's [400,400,500]
        disappears with no signal to the caller.
        """
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        const = xr.DataArray(
            [10.0, 20.0, 30.0],
            dims=["A"],
            coords={"A": A, "B": ("A", [400, 400, 500])},
        )
        result = v + const
        assert result.coords["B"].values.tolist() == [311, 311, 322]

    @pytest.mark.legacy
    def test_warn_on_aux_conflict(self, m: Model, A, unsilenced) -> None:
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        const = xr.DataArray(
            [10.0, 20.0, 30.0],
            dims=["A"],
            coords={"A": A, "B": ("A", [400, 400, 500])},
        )
        with pytest.warns(LinopySemanticsWarning):
            v + const


class TestAuxCoordPropagation:
    """
    Non-conflicting aux coords must propagate through arithmetic and
    into constraints — the positive half of §11.
    """

    @pytest.fixture
    def A(self) -> pd.Index:
        return pd.Index([1, 2, 3], name="A")

    def test_aux_coord_survives_scalar_mul(self, m: Model, A) -> None:
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        assert "B" in (3 * v).coords

    def test_aux_coord_survives_scalar_add(self, m: Model, A) -> None:
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        assert "B" in (v + 5).coords

    def test_aux_coord_propagates_through_var_plus_var(self, m: Model, A) -> None:
        B = ("A", [311, 311, 322])
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(B=B)
        w = m.add_variables(lower=0, coords=[A], name="w").assign_coords(B=B)
        result = v + w
        assert "B" in result.coords
        assert result.coords["B"].values.tolist() == [311, 311, 322]

    def test_aux_coord_propagates_into_constraint(self, m: Model, A) -> None:
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        c = v <= 10
        assert "B" in c.coords

    def test_aux_coord_only_on_dataarray_propagates(self, m: Model, A) -> None:
        """
        ``x * a`` where ``a`` carries an aux coord and ``x`` doesn't —
        the coord propagates through every binary operator and into the
        constraint. Hits the `_align_constant` path (var-OP-DataArray)
        distinct from the `merge` path tested below.
        """
        x = m.add_variables(lower=0, coords=[A], name="x")
        a = xr.DataArray(
            [2.0, 3.0, 4.0], dims=["A"], coords={"A": A, "B": ("A", [10, 20, 30])}
        )
        for expr in (x * a, x + a, x / a):
            assert "B" in expr.coords
            assert expr.coords["B"].values.tolist() == [10, 20, 30]
        # And into the constraint
        c = x <= a
        assert "B" in c.coords

    def test_aux_coord_only_on_one_side_propagates(self, m: Model, A) -> None:
        """Var+var counterpart of the above — hits the `merge` path."""
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        w = m.add_variables(lower=0, coords=[A], name="w")  # no B
        result = v + w
        assert "B" in result.coords
