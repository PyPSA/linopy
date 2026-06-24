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

Slice H — unlabeled-operand pairing (coordinate-alignment intro):
    Unlabeled operands (numpy / list / polars) pair with the linopy
    operand's dims by size; ambiguity or no-match raises (v1)   → #736

Slice H — object scope (convention preamble):
    Non-linopy operands behave exactly like constant-only expressions
"""

from __future__ import annotations

import contextlib
import operator
import warnings
from collections.abc import Generator
from typing import Any, cast

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from linopy import LinearExpression, Model, QuadraticExpression
from linopy.config import LinopySemanticsWarning
from linopy.expressions import merge
from linopy.testing import assert_conequal, assert_linequal, assert_quadequal
from linopy.variables import Variable


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
    def test_same_size_different_labels_raises(self, x: Variable, op: str) -> None:
        """
        #708 / #550 — same shape, different labels: legacy aligns by
        position; v1 raises. Holds for every binary operator.
        """
        other = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            dims=["time"],
            coords={"time": pd.Index([10, 11, 12, 13, 14], name="time")},
        )
        with pytest.raises(ValueError, match="Coordinate mismatch on shared dimension"):
            _OPS[op](x, other)

    @pytest.mark.v1
    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_subset_constant_raises(self, x: Variable, op: str) -> None:
        """
        #711 / #708 — constant covers only some of the variable's
        coords. Legacy left-joins (silently drops the gap); v1 raises.
        """
        subset = xr.DataArray(
            [10.0, 20.0], dims=["time"], coords={"time": pd.Index([1, 3], name="time")}
        )
        with pytest.raises(ValueError, match="Coordinate mismatch on shared dimension"):
            _OPS[op](x, subset)

    @pytest.mark.legacy
    def test_add_same_size_different_labels_silent(self, x: Variable) -> None:
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
    def test_add_subset_constant_silent(self, x: Variable) -> None:
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

    def test_add_broadcast_introduces_new_dim(self, x: Variable) -> None:
        bcast = xr.DataArray(
            [10.0, 20.0], dims=["scenario"], coords={"scenario": [0, 1]}
        )
        result = x + bcast
        assert set(result.const.dims) == {"time", "scenario"}
        assert result.const.sizes == {"time": 5, "scenario": 2}

    def test_mul_broadcast_introduces_new_dim(self, x: Variable) -> None:
        bcast = xr.DataArray([2.0, 3.0], dims=["scenario"], coords={"scenario": [0, 1]})
        result = x * bcast
        assert set(result.coeffs.dims) == {"time", "scenario", "_term"}


# =====================================================================
# Coordinate-alignment intro — unlabeled operands pair by size (#736)
# =====================================================================


# The three unlabeled types must behave identically; each constructor takes a
# value sequence so the same cases parametrize over numpy / list / polars.
UNLABELED_1D = [
    pytest.param(lambda v: np.asarray(v, dtype=float), id="numpy"),
    pytest.param(lambda v: [float(x) for x in v], id="list"),
    pytest.param(lambda v: pl.Series([float(x) for x in v]), id="polars"),
]


class TestUnlabeledPairing:
    """
    Unlabeled operands (numpy arrays, lists, polars Series) carry no labels,
    so they pair with the linopy operand's dims *by size*. Ambiguity (two
    dims share the size, or the array is square) or no size match raises
    under v1; legacy pairs with the leading dims positionally and warns when
    the v1 pairing would differ or reject.
    """

    @pytest.fixture
    def xy(self) -> Variable:
        # dims of distinct sizes so a 1-d operand pairs unambiguously
        m = Model()
        return m.add_variables(
            coords=[pd.RangeIndex(3, name="a"), pd.RangeIndex(4, name="b")], name="xy"
        )

    @pytest.fixture
    def square(self) -> Variable:
        # both dims size 4 → a 1-d length-4 operand is ambiguous
        m = Model()
        return m.add_variables(
            coords=[pd.RangeIndex(4, name="p"), pd.RangeIndex(4, name="q")], name="sq"
        )

    @pytest.fixture
    def wide(self) -> Variable:
        # four dims of distinct sizes — a lower-rank operand pairs a subset
        m = Model()
        return m.add_variables(
            coords=[
                pd.RangeIndex(3, name="a"),
                pd.RangeIndex(4, name="b"),
                pd.RangeIndex(5, name="c"),
                pd.RangeIndex(6, name="d"),
            ],
            name="wide",
        )

    # -- 1-d operands -----------------------------------------------------

    @pytest.mark.v1
    @pytest.mark.parametrize("make", UNLABELED_1D)
    def test_v1_pairs_by_size(self, xy: Variable, make: Any) -> None:
        # length-4 array pairs with dim "b" (size 4), not the leading "a" (3)
        result = (1 * xy) + make(range(4))
        assert set(result.const.dims) == {"a", "b"}
        assert result.const.sizes == {"a": 3, "b": 4}

    @pytest.mark.v1
    @pytest.mark.parametrize("make", UNLABELED_1D)
    def test_v1_ambiguous_square_raises(self, square: Variable, make: Any) -> None:
        with pytest.raises(ValueError, match=r"sizes alone cannot decide"):
            (1 * square) + make(range(4))

    @pytest.mark.v1
    @pytest.mark.parametrize("make", UNLABELED_1D)
    def test_v1_no_size_match_raises(self, xy: Variable, make: Any) -> None:
        with pytest.raises(ValueError, match=r"no unambiguous dimension match"):
            (1 * xy) + make(range(7))

    def test_dataarray_wrapping_resolves_ambiguity(self, square: Variable) -> None:
        # the documented escape hatch: name the axis with a DataArray
        result = (1 * square) + xr.DataArray(np.arange(4.0), dims=["p"])
        assert set(result.const.dims) == {"p", "q"}

    # -- multi-dim operands (numpy only — list / polars are 1-d) ----------

    @pytest.mark.v1
    def test_v1_size_order_independent(self, xy: Variable) -> None:
        # a 2-d (4, 3) operand pairs (b, a) by size regardless of axis order.
        result = (1 * xy) + np.ones((4, 3))
        assert result.const.sizes == {"a": 3, "b": 4}

    @pytest.mark.v1
    def test_v1_multidim_square_ambiguous_raises(self, square: Variable) -> None:
        # a 2-d (4, 4) operand against dims (p: 4, q: 4) — sizes cannot tell
        # (p, q) from (q, p). Exercises the multi-axis-same-size branch the 1-d
        # cases never reach.
        with pytest.raises(ValueError, match=r"sizes alone cannot decide"):
            (1 * square) + np.ones((4, 4))

    @pytest.mark.v1
    def test_v1_multidim_no_size_match_raises(self, xy: Variable) -> None:
        # a 2-d (5, 3) operand against (a: 3, b: 4) — the length-5 axis matches
        # no dim.
        with pytest.raises(ValueError, match=r"no unambiguous dimension match"):
            (1 * xy) + np.ones((5, 3))

    @pytest.mark.v1
    def test_v1_lower_rank_operand_pairs_subset_and_broadcasts(
        self, wide: Variable
    ) -> None:
        # a 2-d (4, 5) operand against four dims pairs its axes with (b, c) by
        # size and broadcasts over the unpaired (a, d).
        result = (1 * wide) + np.ones((4, 5))
        assert result.const.sizes == {"a": 3, "b": 4, "c": 5, "d": 6}

    @pytest.mark.v1
    def test_v1_ambiguous_axis_within_higher_rank_raises(self) -> None:
        # a 2-d (4, 5) operand where the length-4 axis matches two dims of the
        # operand (a: 4, b: 4) — ambiguous even though the length-5 axis is
        # unique.
        m = Model()
        y = m.add_variables(
            coords=[
                pd.RangeIndex(4, name="a"),
                pd.RangeIndex(4, name="b"),
                pd.RangeIndex(5, name="c"),
            ],
            name="y",
        )
        with pytest.raises(ValueError, match=r"sizes alone cannot decide"):
            (1 * y) + np.ones((4, 5))

    # -- matmul -----------------------------------------------------------

    @pytest.mark.v1
    @pytest.mark.parametrize("make", UNLABELED_1D)
    def test_v1_matmul_pairs_by_size(self, xy: Variable, make: Any) -> None:
        # matmul contracts the paired dim: length-4 array pairs with "b"
        result = (1 * xy) @ make(range(4))
        assert set(result.coord_dims) == {"a"}

    # -- construction (add_variables bounds) ------------------------------

    @pytest.mark.v1
    def test_v1_add_variables_bound_pairs_by_size(self) -> None:
        # The rule is the same for construction inputs: a bare-numpy bound
        # pairs with the matching dim by size, not positionally (where the
        # length-5 array would hit the leading dim "a" and conflict).
        m = Model()
        x = m.add_variables(
            coords=[pd.RangeIndex(4, name="a"), pd.RangeIndex(5, name="time")],
            lower=np.arange(5.0),
            name="x",
        )
        assert dict(x.lower.sizes) == {"a": 4, "time": 5}
        assert (x.lower.isel(a=0).values == np.arange(5.0)).all()

    @pytest.mark.v1
    def test_v1_add_variables_ambiguous_bound_raises(self) -> None:
        m = Model()
        with pytest.raises(ValueError, match=r"sizes alone cannot decide"):
            m.add_variables(
                coords=[pd.RangeIndex(4, name="p"), pd.RangeIndex(4, name="q")],
                lower=np.arange(4.0),
                name="x",
            )

    @pytest.mark.legacy
    def test_legacy_no_divergence_is_silent(
        self, xy: Variable, unsilenced: None
    ) -> None:
        # length-3 array: legacy pairs positionally with the leading dim "a";
        # v1 would pair it with "a" too (only "a" is size 3), so there is no
        # divergence and no warning. Pins the silent case.
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinopySemanticsWarning)
            result = (1 * xy) + np.arange(3.0)
        assert result.const.sizes == {"a": 3, "b": 4}
        assert (result.const.isel(b=0).values == np.arange(3.0)).all()

    @pytest.mark.legacy
    def test_legacy_warns_when_v1_would_differ(
        self, xy: Variable, unsilenced: None
    ) -> None:
        # length-4 array: legacy pairs positionally with "a" (size 3) → error,
        # but the warning fires first explaining the v1 divergence.
        # legacy positional pairing assigns the len-4 array to "a" (size 3),
        # which then conflicts — assert it's that shape error, nothing incidental.
        with pytest.warns(LinopySemanticsWarning, match=r"pairs by size instead"):
            with contextlib.suppress(ValueError):
                (1 * xy) + np.arange(4.0)

    @pytest.mark.legacy
    def test_legacy_ambiguous_pairs_positionally_with_warning(
        self, square: Variable, unsilenced: None
    ) -> None:
        # The square (p:4, q:4) case where v1 *raises* — legacy must instead
        # pair positionally with the leading dim and warn, never raise. This is
        # the biggest legacy/v1 divergence and the strongest no-regression guard.
        with pytest.warns(LinopySemanticsWarning, match=r"this raises"):
            result = (1 * square) + np.arange(4.0)
        assert result.const.sizes == {"p": 4, "q": 4}
        # paired with the leading dim "p": the array varies along p, broadcast over q
        assert (result.const.isel(q=0).values == np.arange(4.0)).all()

    @pytest.mark.legacy
    def test_legacy_add_variables_bound_positional(self, unsilenced: None) -> None:
        # Regression guard for the unification: legacy add_variables still
        # assigns an unlabeled bound positionally (here unambiguous — only "a"
        # is size 5 — so it is also silent), producing the pre-#736 result.
        m = Model()
        with warnings.catch_warnings():
            warnings.simplefilter("error", LinopySemanticsWarning)
            x = m.add_variables(
                coords=[pd.RangeIndex(5, name="a"), pd.RangeIndex(3, name="b")],
                lower=np.arange(5.0),
                name="x",
            )
        assert dict(x.lower.sizes) == {"a": 5, "b": 3}
        assert (x.lower.isel(b=0).values == np.arange(5.0)).all()


# =====================================================================
# §5 — User-supplied NaN raises (covers #713 and PyPSA #1683)
# =====================================================================


class TestUserNaNRaises:
    @pytest.mark.v1
    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_nan_dataarray_raises(
        self, x: Variable, time: pd.RangeIndex, op: str
    ) -> None:
        # Use [2, NaN, 3, 4, 5] so div doesn't trip on a 0 divisor at slot 0.
        nan_data = xr.DataArray(
            [2.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        with pytest.raises(ValueError, match="NaN"):
            _OPS[op](x, nan_data)

    @pytest.mark.v1
    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_nan_scalar_raises(self, x: Variable, op: str) -> None:
        with pytest.raises(ValueError, match="NaN"):
            _OPS[op](x, float("nan"))

    @pytest.mark.v1
    def test_pypsa_1683_inf_times_zero_raises(
        self, x: Variable, time: pd.RangeIndex
    ) -> None:
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

    @pytest.mark.v1
    def test_to_linexpr_coefficient_nan_raises(
        self, x: Variable, time: pd.RangeIndex
    ) -> None:
        """
        §5 on the direct ``Variable.to_linexpr(coefficient)`` entry —
        callers can construct an expression bypassing the operator
        overloads. NaN in the explicit coefficient is still user data
        and must raise (otherwise the NaN flows into ``coeffs`` and
        §6 silently propagates absence from what was actually a
        data error).
        """
        nan_coeff = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        with pytest.raises(ValueError, match="NaN"):
            x.to_linexpr(nan_coeff)

    @pytest.mark.legacy
    def test_add_nan_dataarray_silently_fills_with_zero(
        self, x: Variable, time: pd.RangeIndex
    ) -> None:
        """Document legacy: NaN in addend silently becomes 0 (#713)."""
        nan_data = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        result = x + nan_data
        assert result.const.sel(time=1).item() == 0.0  # NaN → 0

    @pytest.mark.legacy
    def test_mul_nan_dataarray_silently_fills_with_zero(
        self, x: Variable, time: pd.RangeIndex
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


def _one_legacy_warning(*ops) -> str:  # type: ignore[no-untyped-def]
    """
    Run ``ops`` (a series of callables) under fresh warning capture
    and return the first ``LinopySemanticsWarning``'s text. Test
    helper — keeps each test focused on the message, not the plumbing.
    """
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        for op in ops:
            op()
    legacy = [w for w in ws if issubclass(w.category, LinopySemanticsWarning)]
    assert legacy, "expected at least one LinopySemanticsWarning"
    return str(legacy[0].message)


# Common tail shared by every legacy warning — separated so each
# expected-message test can focus on the part that's specific to the
# rule, without 4 lines of boilerplate per test.
_OPT_IN_HINT = (
    "\n  Opt in:    linopy.options['semantics'] = 'v1'"
    "\n  Silence:   warnings.filterwarnings('ignore', "
    "category=LinopySemanticsWarning)"
)


class TestLegacyWarning:
    """
    Asserts the *full text* of each legacy warning. The point: the
    test reads like a spec — a reviewer judges the message's helpfulness
    by reading the test, and any change to a message surfaces as a diff
    in the test. Goal #2 (actionable warnings) lives or dies here.
    """

    @pytest.mark.legacy
    def test_coord_mismatch_const_operand_same_size(
        self, x: Variable, unsilenced: None
    ) -> None:
        other = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            dims=["time"],
            coords={"time": pd.Index([10, 11, 12, 13, 14], name="time")},
        )
        msg = _one_legacy_warning(lambda: x + other)
        assert msg == (
            "Coordinate mismatch in this operator's constant operand "
            "silently aligned by legacy (positional when sizes match, "
            "otherwise left-join). Under v1 this raises ValueError."
            "\n  Dim:       'time': left=[0, 1, 2, 3, 4], "
            "right=[10, 11, 12, 13, 14]"
            "\n  Resolve:   `.sel(...)` / `.reindex(...)` to align"
            "\n             `.assign_coords(...)` to relabel one side"
            "\n             `linopy.align(...)` to pre-align several operands"
            "\n             or pass an explicit `join=` argument." + _OPT_IN_HINT
        )

    @pytest.mark.legacy
    def test_coord_mismatch_const_operand_subset(
        self, x: Variable, unsilenced: None
    ) -> None:
        subset = xr.DataArray(
            [10.0, 20.0], dims=["time"], coords={"time": pd.Index([1, 3], name="time")}
        )
        msg = _one_legacy_warning(lambda: x + subset)
        assert msg == (
            "Coordinate mismatch in this operator's constant operand "
            "silently aligned by legacy (positional when sizes match, "
            "otherwise left-join). Under v1 this raises ValueError."
            "\n  Dim:       'time': left=[0, 1, 2, 3, 4], right=[1, 3]"
            "\n  Resolve:   `.sel(...)` / `.reindex(...)` to align"
            "\n             `.assign_coords(...)` to relabel one side"
            "\n             `linopy.align(...)` to pre-align several operands"
            "\n             or pass an explicit `join=` argument." + _OPT_IN_HINT
        )

    @pytest.mark.legacy
    def test_coord_mismatch_merge_path(
        self, m: Model, time: pd.RangeIndex, unsilenced: None
    ) -> None:
        x_local = m.add_variables(lower=0, coords=[time], name="x_local")
        other = m.add_variables(
            lower=0, coords=[pd.Index([10, 11, 12, 13, 14], name="time")], name="other"
        )
        msg = _one_legacy_warning(lambda: x_local + other)
        assert msg == (
            "Coordinate mismatch in merge along dim '_term' silently "
            "aligned by legacy (positional when sizes match, otherwise "
            "left-join). Under v1 this raises ValueError."
            "\n  Dim:       'time': left=[0, 1, 2, 3, 4], "
            "right=[10, 11, 12, 13, 14]"
            "\n  Resolve:   `.sel(...)` / `.reindex(...)` to align"
            "\n             `.assign_coords(...)` to relabel one side"
            "\n             `linopy.align(...)` to pre-align several operands"
            "\n             or pass an explicit `join=` argument." + _OPT_IN_HINT
        )

    @pytest.mark.legacy
    def test_nan_addend(
        self, x: Variable, time: pd.RangeIndex, unsilenced: None
    ) -> None:
        nan_data = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        msg = _one_legacy_warning(lambda: x + nan_data)
        assert msg == (
            "NaN in the constant operand was silently treated as 0 by "
            "legacy (additive identity). Under v1 this raises ValueError."
            "\n  Resolve:   `.fillna(value)` (data error)"
            "\n             or `mask=` / `.where(cond)` / `.reindex(...)` "
            "on the variable (intended absence)." + _OPT_IN_HINT
        )

    @pytest.mark.legacy
    def test_nan_multiplier(
        self, x: Variable, time: pd.RangeIndex, unsilenced: None
    ) -> None:
        nan_factor = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        msg = _one_legacy_warning(lambda: x * nan_factor)
        assert msg == (
            "NaN in the multiplicative factor was silently treated as 0 "
            "by legacy (so the variable was zeroed out at that slot). "
            "Under v1 this raises ValueError."
            "\n  Resolve:   `.fillna(value)` (data error)"
            "\n             or `mask=` / `.where(cond)` / `.reindex(...)` "
            "on the variable (intended absence)." + _OPT_IN_HINT
        )

    @pytest.mark.legacy
    def test_nan_divisor(
        self, x: Variable, time: pd.RangeIndex, unsilenced: None
    ) -> None:
        nan_divisor = xr.DataArray(
            [2.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        msg = _one_legacy_warning(lambda: x / nan_divisor)
        assert msg == (
            "NaN in the divisor was silently treated as 1 by legacy (a "
            "different fill from `+`/`*` which use 0). Under v1 this "
            "raises ValueError."
            "\n  Resolve:   `.fillna(value)` (data error)"
            "\n             or `mask=` / `.where(cond)` / `.reindex(...)` "
            "on the variable (intended absence)." + _OPT_IN_HINT
        )

    @pytest.mark.legacy
    def test_aux_conflict(self, m: Model, unsilenced: None) -> None:
        A = pd.Index([1, 2, 3], name="A")
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        const = xr.DataArray(
            [10.0, 20.0, 30.0],
            dims=["A"],
            coords={"A": A, "B": ("A", [400, 400, 500])},
        )
        msg = _one_legacy_warning(lambda: v + const)
        assert msg == (
            "Auxiliary coordinate 'B' was conflicting across operands "
            "and silently dropped by legacy (xarray's default). Under v1 "
            "this raises ValueError."
            "\n  Values:    left=[311, 311, 322], right=[400, 400, 500]"
            "\n  Resolve:   `.drop_vars('B')`"
            "\n             `.assign_coords(B=...)` to relabel one side"
            "\n             or `.isel(..., drop=True)` if a scalar isel "
            "introduced it." + _OPT_IN_HINT
        )

    @pytest.mark.legacy
    def test_nan_constraint_rhs(
        self, x: Variable, time: pd.RangeIndex, unsilenced: None
    ) -> None:
        nan_rhs = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        msg = _one_legacy_warning(lambda: x <= nan_rhs)
        assert msg == (
            "NaN in the constraint RHS was silently kept as 'no "
            "constraint at this row' by legacy auto-mask. Under v1 this "
            "raises ValueError."
            "\n  Resolve:   `mask=` on the variable for explicit per-row "
            "masking"
            "\n             or `.fillna(value)` if the NaN was a data error."
            + _OPT_IN_HINT
        )

    @pytest.mark.legacy
    def test_masked_variable_in_arithmetic(
        self, time: pd.RangeIndex, unsilenced: None
    ) -> None:
        """
        The masked-variable warning is the one that catches the
        ``2 * x + y`` (no fillna) divergence — no other site fires for
        this case. Message names the variable and the fillna(0) fix.
        """
        m = Model()
        x = m.add_variables(lower=0, coords=[time], name="x")
        mask = xr.DataArray(
            [True, True, True, True, False], dims=["time"], coords={"time": time}
        )
        m.add_variables(lower=0, coords=[time], name="y", mask=mask)
        y = m.variables["y"]
        msg = _one_legacy_warning(lambda: 2 * x + y)
        assert msg == (
            "Variable 'y' has absent slots (from `mask=` / `.where()` / "
            "`.shift()` / `.reindex()`). Under legacy each absent slot "
            "contributes 0 to the resulting expression's terms (so `x + "
            "y >= 10` reduces to `x >= 10` there). Under v1 the absence "
            "propagates through arithmetic instead (`x + y` becomes "
            "absent at the slot and the constraint drops)."
            "\n  Resolve:   wrap with `y.fillna(0)` for the legacy "
            "behaviour under v1"
            "\n             (no fix needed if you only use the variable "
            "in a constraint LHS alone — `y >= 0` drops the same way in "
            "both)." + _OPT_IN_HINT
        )

    @pytest.mark.legacy
    def test_warning_stacklevel_points_to_user_call(
        self, time: pd.RangeIndex, unsilenced: None
    ) -> None:
        """
        The warning's source frame must be the user's call site, not
        a linopy internal — IDE jump-to-source on the warning depends
        on it, and the rollout-warning is useless if it points at
        ``linopy/expressions.py`` instead of the user's source.
        """
        m = Model()
        x = m.add_variables(lower=0, coords=[time], name="x")
        mask = xr.DataArray(
            [True, True, True, True, False], dims=["time"], coords={"time": time}
        )
        y = m.add_variables(lower=0, coords=[time], name="y", mask=mask)
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            _ = 2 * x + y  # this is the user's call site
        relevant = [w for w in ws if issubclass(w.category, LinopySemanticsWarning)]
        assert relevant, "expected the masked-variable warning to fire"
        assert relevant[0].filename == __file__, (
            f"warning frame is {relevant[0].filename!r}, "
            "should be the user's source file"
        )


# =====================================================================
# §8 — Shared dimensions must match exactly (expr+expr / var+var, merge path)
# =====================================================================


class TestExactAlignmentMerge:
    @pytest.fixture
    def x_other(self, m: Model) -> Variable:
        # Same shape, different labels — legacy uses positional override.
        return m.add_variables(
            lower=0,
            coords=[pd.Index([10, 11, 12, 13, 14], name="time")],
            name="x_other",
        )

    @pytest.fixture
    def x_subset(self, m: Model) -> Variable:
        # Subset coords on the same dim — legacy outer-joins (and pads).
        return m.add_variables(
            lower=0,
            coords=[pd.Index([1, 3], name="time")],
            name="x_subset",
        )

    @pytest.mark.v1
    def test_var_plus_var_different_labels_raises(
        self, x: Variable, x_other: Variable
    ) -> None:
        with pytest.raises(ValueError, match="Coordinate mismatch"):
            x + x_other

    @pytest.mark.v1
    def test_expr_plus_expr_different_labels_raises(
        self, x: Variable, x_other: Variable
    ) -> None:
        with pytest.raises(ValueError, match="Coordinate mismatch"):
            (1 * x) + (1 * x_other)

    @pytest.mark.v1
    def test_var_plus_var_subset_raises(self, x: Variable, x_subset: Variable) -> None:
        with pytest.raises(ValueError, match="Coordinate mismatch"):
            x + x_subset

    @pytest.mark.v1
    def test_var_minus_var_different_labels_raises(
        self, x: Variable, x_other: Variable
    ) -> None:
        with pytest.raises(ValueError, match="Coordinate mismatch"):
            x - x_other

    def test_var_plus_var_same_coords_works(
        self, m: Model, time: pd.RangeIndex
    ) -> None:
        """Same coords on a shared dim is fine — regression guard."""
        a = m.add_variables(lower=0, coords=[time], name="a")
        b = m.add_variables(lower=0, coords=[time], name="b")
        result = a + b
        assert result.sizes["time"] == 5

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

    @pytest.mark.v1
    def test_var_plus_var_reordered_labels_align(self, m: Model) -> None:
        a = m.add_variables(coords=[pd.Index(["costs", "penalty"], name="e")], name="a")
        b = m.add_variables(coords=[pd.Index(["penalty", "costs"], name="e")], name="b")
        result = (1 * a) + (1 * b)
        assert list(result.indexes["e"]) == ["costs", "penalty"]
        # The index alone is identical under legacy — assert the variable
        # *pairing*: §8 sums a["costs"] with b["costs"], NOT b's leading
        # (positional) entry b["penalty"].
        costs = {int(v) for v in result.vars.sel(e="costs").values}
        assert costs == {int(a.labels.sel(e="costs")), int(b.labels.sel(e="costs"))}

    @pytest.mark.v1
    def test_reordered_constants_pair_by_label_not_position(self, m: Model) -> None:
        ea = pd.Index(["costs", "penalty"], name="e")
        eb = pd.Index(["penalty", "costs"], name="e")
        a = m.add_variables(coords=[ea], name="a") + pd.Series([100.0, 200.0], index=ea)
        b = m.add_variables(coords=[eb], name="b") + pd.Series([1.0, 2.0], index=eb)
        result = a + b
        assert float(result.const.sel(e="costs")) == 102.0
        assert float(result.const.sel(e="penalty")) == 201.0

    @pytest.mark.v1
    def test_multi_operand_merge_reordered_pairs_by_label(self, m: Model) -> None:
        ea = pd.Index(["x", "y", "z"], name="e")
        er = pd.Index(["z", "y", "x"], name="e")
        a = m.add_variables(coords=[ea], name="a") + pd.Series(
            [1.0, 2.0, 3.0], index=ea
        )
        b = m.add_variables(coords=[er], name="b") + pd.Series(
            [10.0, 20.0, 30.0], index=er
        )
        c = m.add_variables(coords=[ea], name="c") + pd.Series(
            [100, 200, 300.0], index=ea
        )
        result = merge([a, b, c], cls=type(a))
        assert float(result.const.sel(e="x")) == 131.0
        assert float(result.const.sel(e="z")) == 313.0

    @pytest.mark.v1
    def test_quadratic_merge_reordered_aligns(self, m: Model) -> None:
        ea = pd.Index(["x", "y", "z"], name="e")
        er = pd.Index(["z", "y", "x"], name="e")
        x = m.add_variables(coords=[ea], name="x")
        y = m.add_variables(coords=[er], name="y")
        result = (x * x) + (y * y)
        assert list(result.indexes["e"]) == ["x", "y", "z"]
        # by-label: the "x" slot holds x["x"]**2 and y["x"]**2, so the only
        # variable labels there are x["x"] and y["x"] — not y's leading y["z"].
        labels = {int(v) for v in result.vars.sel(e="x").values.ravel() if v >= 0}
        assert labels == {int(x.labels.sel(e="x")), int(y.labels.sel(e="x"))}

    @pytest.mark.v1
    def test_reordered_multiindex_aligns_by_tuple(self, m: Model) -> None:
        mi1 = pd.MultiIndex.from_tuples(
            [(1, "a"), (1, "b"), (2, "a")], names=["p", "s"]
        )
        mi2 = pd.MultiIndex.from_tuples(
            [(2, "a"), (1, "b"), (1, "a")], names=["p", "s"]
        )
        mi1.name = "snap"
        mi2.name = "snap"
        x = m.add_variables(coords=[mi1], name="x") + pd.Series(
            [1.0, 2.0, 3.0], index=mi1
        )
        y = m.add_variables(coords=[mi2], name="y") + pd.Series(
            [10.0, 20.0, 30.0], index=mi2
        )
        result = x + y
        got = dict(zip(map(tuple, result.indexes["snap"]), result.const.values))
        assert got == {(1, "a"): 31.0, (1, "b"): 22.0, (2, "a"): 13.0}

    @pytest.mark.legacy
    def test_reordered_merge_positional_legacy(self, m: Model) -> None:
        ea = pd.Index(["costs", "penalty"], name="e")
        eb = pd.Index(["penalty", "costs"], name="e")
        a = m.add_variables(coords=[ea], name="a") + pd.Series([100.0, 200.0], index=ea)
        b = m.add_variables(coords=[eb], name="b") + pd.Series([1.0, 2.0], index=eb)
        result = a + b
        assert float(result.const.sel(e="costs")) == 101.0
        assert float(result.const.sel(e="penalty")) == 202.0

    @pytest.mark.legacy
    def test_reordered_merge_warns_legacy(self, m: Model, unsilenced: None) -> None:
        ea = pd.Index(["costs", "penalty"], name="e")
        eb = pd.Index(["penalty", "costs"], name="e")
        a = m.add_variables(coords=[ea], name="a")
        b = m.add_variables(coords=[eb], name="b")
        with pytest.warns(LinopySemanticsWarning) as record:
            (1 * a) + (1 * b)
        msg = next(
            str(w.message) for w in record if w.category is LinopySemanticsWarning
        )
        assert msg == (
            "Coordinate order mismatch in merge along dim '_term' aligned "
            "positionally by legacy. Under v1 the same labels in a different "
            "order align by label (a reindex), giving a different result."
            "\n  Dim:       'e': left=['costs', 'penalty'], "
            "right=['penalty', 'costs']"
            "\n  Resolve:   `.sel(...)` / `.reindex(...)` to align"
            "\n             `.assign_coords(...)` to relabel one side"
            "\n             or pass an explicit `join=` argument."
            "\n  Opt in:    linopy.options['semantics'] = 'v1'"
            "\n  Silence:   warnings.filterwarnings('ignore', "
            "category=LinopySemanticsWarning)"
        )

    @pytest.mark.legacy
    def test_var_plus_var_reordered_pairs_positionally_legacy(self, m: Model) -> None:
        """
        Legacy preserves master's #550 behaviour: var+var with reordered
        labels pairs by position, so the "costs" slot wrongly sums
        a["costs"] with b's leading entry b["penalty"]. The index still
        reads as the left operand's order, which is what hides the
        mispairing — so assert the variable pairing, not just the index.
        """
        a = m.add_variables(coords=[pd.Index(["costs", "penalty"], name="e")], name="a")
        b = m.add_variables(coords=[pd.Index(["penalty", "costs"], name="e")], name="b")
        result = (1 * a) + (1 * b)
        assert list(result.indexes["e"]) == ["costs", "penalty"]
        costs = {int(v) for v in result.vars.sel(e="costs").values}
        assert costs == {int(a.labels.sel(e="costs")), int(b.labels.sel(e="penalty"))}

    @pytest.mark.legacy
    def test_quadratic_merge_reordered_pairs_positionally_legacy(
        self, m: Model
    ) -> None:
        """
        Legacy pairs the quadratic merge by position too: the "x" slot
        pairs x["x"]**2 with y's leading y["z"]**2, not y["x"]**2.
        """
        ea = pd.Index(["x", "y", "z"], name="e")
        er = pd.Index(["z", "y", "x"], name="e")
        x = m.add_variables(coords=[ea], name="x")
        y = m.add_variables(coords=[er], name="y")
        result = (x * x) + (y * y)
        assert list(result.indexes["e"]) == ["x", "y", "z"]
        labels = {int(v) for v in result.vars.sel(e="x").values.ravel() if v >= 0}
        assert labels == {int(x.labels.sel(e="x")), int(y.labels.sel(e="z"))}

    @pytest.mark.v1
    def test_different_multiindex_raises_dim_mismatch(self, m: Model) -> None:
        mi1 = pd.MultiIndex.from_tuples(
            [(1, "a"), (1, "b"), (2, "a")], names=["p", "s"]
        )
        mi2 = pd.MultiIndex.from_tuples(
            [(1, "a"), (1, "b"), (3, "c")], names=["p", "s"]
        )
        mi1.name = "snap"
        mi2.name = "snap"
        x = m.add_variables(coords=[mi1], name="x")
        y = m.add_variables(coords=[mi2], name="y")
        with pytest.raises(ValueError, match="shared dimension 'snap'"):
            (1 * x) + (1 * y)

    @pytest.mark.legacy
    def test_var_plus_var_different_labels_silent(
        self, x: Variable, x_other: Variable
    ) -> None:
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
        self, x: Variable, x_other: Variable, unsilenced: None
    ) -> None:
        with pytest.warns(LinopySemanticsWarning, match="merge along dim"):
            x + x_other


# =====================================================================
# §6 — Absence propagates through every operator
# §3 — isnull() reports absent slots (covers #712 absent-as-zero)
# =====================================================================


class TestAbsencePropagation:
    @pytest.fixture
    def xs(self, x: Variable) -> Variable:
        # x.shift(time=1) → absent at time=0, present elsewhere.
        return x.shift(time=1)

    @pytest.mark.v1
    def test_to_linexpr_marks_absent_with_nan_const(self, xs: Variable) -> None:
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
    def test_isnull_reports_absent_slot(self, xs: Variable) -> None:
        """§3: isnull() reports the absent slot on a LinearExpression."""
        expr = xs.to_linexpr()
        assert bool(expr.isnull().values[0])
        assert not bool(expr.isnull().values[1:].any())

    @pytest.mark.v1
    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_scalar_op_preserves_absence(self, xs: Variable, op: str) -> None:
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
    def test_add_present_variable_propagates_absence(
        self, xs: Variable, x: Variable
    ) -> None:
        """`x + xs` is absent wherever xs is, even though x is fine there."""
        result = xs + x
        assert np.isnan(result.const.values[0])
        assert bool(result.isnull().values[0])
        assert not bool(result.isnull().values[1:].any())

    @pytest.mark.v1
    def test_merge_absorbs_dead_terms_at_absent_slot(
        self, xs: Variable, x: Variable
    ) -> None:
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
    def test_absent_distinguishable_from_zero(self, x: Variable, xs: Variable) -> None:
        """
        #712 — under v1, ``x.shift(time=1) * 3`` and ``x * 0`` are
        distinct: the first is absent, the second is a present zero.
        """
        absent = xs * 3
        zero = x * 0
        assert bool(absent.isnull().values[0])
        assert not bool(zero.isnull().values[0])

    @pytest.mark.v1
    @pytest.mark.parametrize(
        "build",
        [
            "var_mul_var",
            "var_pow_2",
            "expr_mul_var",
            "expr_mul_expr",
            "quad_plus_linexpr",
            "quad_times_scalar",
        ],
    )
    def test_quadratic_absence_propagates(
        self, xs: Variable, x: Variable, build: str
    ) -> None:
        """
        §6 on the quadratic build paths — every entry point that ends
        in a QuadraticExpression must keep an absent factor absent.

        Regression for ``prod(skipna=True)`` on the FACTOR_DIM branch
        and the cross-term ``self.const * other.reset_const()`` path,
        plus the downstream operators on the resulting quadratic.
        """
        builders = {
            "var_mul_var": lambda: xs * x,
            "var_pow_2": lambda: xs**2,
            "expr_mul_var": lambda: (1 * xs) * x,
            "expr_mul_expr": lambda: (1 * xs) * (1 * x),
            "quad_plus_linexpr": lambda: (xs * x) + (2 * x),
            "quad_times_scalar": lambda: (xs * x) * 3,
        }
        quad = builders[build]()
        # absent slot stays absent in the resulting quadratic
        assert bool(quad.isnull().values[0])
        # and §1/§2: every term at the absent slot has coeffs NaN and vars -1.
        assert np.isnan(quad.coeffs.values[0]).all()
        assert (quad.vars.values[0] == -1).all()
        # And the present slots stay present (cross-term storage may carry
        # vars=-1 as the "no second factor" sentinel inside a term — that's
        # not absence, so check the slot-level isnull predicate, not vars).
        assert not bool(quad.isnull().values[1:].any())

    @pytest.mark.legacy
    def test_legacy_collapses_absent_to_zero(self, xs: Variable) -> None:
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

    @pytest.mark.legacy
    def test_legacy_to_linexpr_fills_absent_with_zero(self, xs: Variable) -> None:
        """
        Legacy counterpart of the NaN encoding: ``to_linexpr()`` stores
        the absent slot as a present zero (``const=0``, ``coeff=1`` over
        the ``vars=-1`` sentinel), so ``isnull()`` is blind to it.
        """
        expr = xs.to_linexpr()
        assert expr.const.values[0] == 0.0
        assert not np.isnan(expr.coeffs.values[0, 0])
        assert not bool(expr.isnull().values.any())

    @pytest.mark.legacy
    @pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
    def test_legacy_scalar_op_fills_absent(self, xs: Variable, op: str) -> None:
        """
        Legacy fills the absent slot with 0 before the op, so the shifted
        slot carries the same value as every present slot (vs v1's NaN).
        """
        result = _OPS[op](xs, 3)
        assert not bool(result.isnull().values[0])
        expected = {"add": 3.0, "sub": -3.0, "mul": 0.0, "div": 0.0}[op]
        assert (result.const.values == expected).all()

    @pytest.mark.legacy
    def test_legacy_add_present_variable_keeps_live_term(
        self, xs: Variable, x: Variable
    ) -> None:
        """
        ``xs + x``: legacy keeps ``x[0]`` live at the absent slot (the
        merge does not absorb it), so the slot is present, not NaN.
        """
        result = xs + x
        assert not bool(result.isnull().values[0])
        # x[0] survives as a live term alongside the xs=-1 sentinel.
        assert int(x.labels.values[0]) in result.vars.values[0].tolist()

    @pytest.mark.legacy
    def test_legacy_merge_keeps_live_terms_multi_operand(
        self, m: Model, time: pd.RangeIndex
    ) -> None:
        """3-operand merge: legacy keeps x[0] and y[0] live at the absent slot."""
        x = m.add_variables(lower=0, coords=[time], name="x")
        y = m.add_variables(lower=0, coords=[time], name="y")
        xs = x.shift(time=1)
        result = (1 * x) + (1 * y) + xs
        assert not bool(result.isnull().values[0])
        live = result.vars.values[0][~np.isnan(result.coeffs.values[0])].tolist()
        assert int(x.labels.values[0]) in live
        assert int(y.labels.values[0]) in live

    @pytest.mark.legacy
    @pytest.mark.parametrize(
        "build",
        [
            "var_mul_var",
            "var_pow_2",
            "expr_mul_var",
            "expr_mul_expr",
            "quad_plus_linexpr",
            "quad_times_scalar",
        ],
    )
    def test_legacy_quadratic_collapses_absent(
        self, xs: Variable, x: Variable, build: str
    ) -> None:
        """
        Every quadratic build path collapses the absent factor to a
        present zero under legacy — no NaN signal anywhere (vs v1, which
        keeps the slot absent).
        """
        builders = {
            "var_mul_var": lambda: xs * x,
            "var_pow_2": lambda: xs**2,
            "expr_mul_var": lambda: (1 * xs) * x,
            "expr_mul_expr": lambda: (1 * xs) * (1 * x),
            "quad_plus_linexpr": lambda: (xs * x) + (2 * x),
            "quad_times_scalar": lambda: (xs * x) * 3,
        }
        quad = builders[build]()
        assert not bool(quad.isnull().values[0])
        assert not np.isnan(quad.coeffs.values[0]).any()


class TestFillnaResolves:
    """§7 — fillna()/.where() are how the caller resolves an absent slot."""

    @pytest.fixture
    def xs(self, x: Variable) -> Variable:
        return x.shift(time=1)

    @pytest.mark.v1
    def test_expr_fillna_replaces_absent_const(self, xs: Variable) -> None:
        result = xs.to_linexpr().fillna(42)
        assert result.const.values[0] == 42.0
        assert result.const.values[1:].tolist() == [0.0, 0.0, 0.0, 0.0]
        assert not bool(result.isnull().values.any())

    @pytest.mark.v1
    def test_variable_fillna_numeric_returns_expression(self, xs: Variable) -> None:
        """
        A constant fill is not a variable, so the return type is a
        LinearExpression.
        """
        from linopy import LinearExpression

        result = xs.fillna(42)
        assert isinstance(result, LinearExpression)
        assert result.const.values[0] == 42.0

    @pytest.mark.v1
    def test_variable_fillna_zero_revives_slot_as_present_zero(
        self, xs: Variable
    ) -> None:
        from linopy import LinearExpression

        result = xs.fillna(0)
        assert isinstance(result, LinearExpression)  # numeric fill → expression
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

    @pytest.mark.legacy
    def test_legacy_expr_fillna_is_noop(self, xs: Variable) -> None:
        """
        Legacy has already filled the absent slot with 0, so there is no
        NaN for ``fillna(42)`` to replace — the 42 never lands (vs v1,
        which puts 42 at the formerly-absent slot).
        """
        result = xs.to_linexpr().fillna(42)
        assert result.const.values.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]
        assert not bool(result.isnull().values.any())

    @pytest.mark.legacy
    def test_legacy_variable_fillna_numeric_is_noop(self, xs: Variable) -> None:
        """Same no-op on the Variable path: the fill value is ignored."""
        from linopy import LinearExpression

        result = xs.fillna(42)
        assert isinstance(result, LinearExpression)
        assert result.const.values[0] == 0.0

    @pytest.mark.legacy
    def test_legacy_outer_fillna_then_add_double_counts(
        self, m: Model, time: pd.RangeIndex
    ) -> None:
        """
        Legacy never made slot 0 absent, so ``(x + y.shift()).fillna(0)``
        already carries x[0] (and the y sentinel); the outer ``+ x`` then
        adds a *second* x[0] — three live terms at slot 0, double-counting
        x (vs v1's single live x[0]).
        """
        x = m.add_variables(lower=0, coords=[time], name="x")
        y = m.add_variables(lower=0, coords=[time], name="y")
        expr = (x + y.shift(time=1)).fillna(0) + x

        coeffs0 = expr.coeffs.values[0]
        vars0 = expr.vars.values[0]
        live = ~np.isnan(coeffs0)
        assert int(live.sum()) == 3
        # x[0] appears twice — the double-count legacy can't avoid.
        assert vars0[live].tolist().count(int(x.labels.values[0])) == 2
        assert float(expr.const.values[0]) == 0.0


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

    def test_reindex_extends_with_absent(
        self, x: Variable, time: pd.RangeIndex
    ) -> None:
        extended = pd.RangeIndex(8, name="time")
        result = x.reindex(time=extended)
        assert result.sizes["time"] == 8
        # Original slots 0..4 are preserved
        assert int(result.labels.values[0]) == int(x.labels.values[0])
        # New slots 5..7 are absent
        assert (result.labels.values[5:] == -1).all()
        assert np.isnan(result.lower.values[5:]).all()
        assert np.isnan(result.upper.values[5:]).all()

    def test_reindex_subset_drops_coords(self, x: Variable) -> None:
        """
        Reindex to a strict subset shrinks the variable (no absence
        introduced — those slots are just gone).
        """
        result = x.reindex(time=pd.RangeIndex(3, name="time"))
        assert result.sizes["time"] == 3
        assert not (result.labels.values == -1).any()

    def test_reindex_like_extends_with_absent(self, m: Model, x: Variable) -> None:
        wider = m.add_variables(
            lower=0, coords=[pd.RangeIndex(7, name="time")], name="wider"
        )
        result = x.reindex_like(wider)
        assert result.sizes["time"] == 7
        assert (result.labels.values[5:] == -1).all()

    @pytest.mark.v1
    def test_reindexed_variable_propagates_absence_in_arithmetic(
        self, x: Variable, time: pd.RangeIndex
    ) -> None:
        """
        §4 + §6 hand-off: a reindex-introduced absence flows through
        the next operator and is visible via isnull().
        """
        wider = x.reindex(time=pd.RangeIndex(7, name="time"))
        expr = wider * 3
        assert bool(expr.isnull().values[5:].all())
        assert not bool(expr.isnull().values[:5].any())

    @pytest.mark.legacy
    def test_legacy_reindexed_variable_fills_absent_in_arithmetic(
        self, x: Variable, time: pd.RangeIndex
    ) -> None:
        """
        Legacy collapses the reindex-introduced absence to 0, so the
        extended slots are present zeros after ``* 3`` (vs v1, which
        keeps them absent and visible via isnull()).
        """
        wider = x.reindex(time=pd.RangeIndex(7, name="time"))
        expr = wider * 3
        assert not bool(expr.isnull().values.any())
        assert expr.const.values.tolist() == [0.0] * 7

    def test_where_creates_absence(self, x: Variable) -> None:
        """§4 — ``.where(cond)`` marks slots absent in place."""
        cond = xr.DataArray(
            [True, True, False, False, False],
            dims=["time"],
            coords={"time": x.coords["time"]},
        )
        masked = x.where(cond)
        assert (masked.labels.values[2:] == -1).all()
        assert not (masked.labels.values[:2] == -1).any()

    def test_unstack_creates_absence_at_missing_combinations(self, m: Model) -> None:
        """
        §4 — ``.unstack`` of a non-rectangular MultiIndex leaves the
        missing combinations as absent slots.
        """
        # Three (region, year) observations that don't form a full grid:
        # (DE, 2030) and (DE, 2040) exist but (FR, 2030) only — so
        # unstacking (FR, 2040) becomes absent.
        idx = pd.MultiIndex.from_tuples(
            [("DE", 2030), ("DE", 2040), ("FR", 2030)],
            names=("region", "year"),
        )
        # Since #732, sequence-form MultiIndex coords entries must be named.
        idx.name = "dim_0"
        v = m.add_variables(coords=[idx], name="v")
        unstacked = v.unstack("dim_0")
        assert unstacked.sizes == {"region": 2, "year": 2}
        # (FR, 2040) missing → absent
        assert int(unstacked.labels.sel(region="FR", year=2040).item()) == -1
        # The three present cells stay present
        assert int(unstacked.labels.sel(region="DE", year=2030).item()) != -1

    @pytest.mark.parametrize("method", ["roll", "sel", "isel"])
    def test_data_preserving_methods_do_not_create_absence(
        self, x: Variable, method: str
    ) -> None:
        """
        §4 negative — operations that *move or select* existing data
        never introduce absent slots. Pins the spec's contrast against
        the absence-creating mechanisms.
        """
        results = {
            "roll": lambda: x.roll(time=2),
            "sel": lambda: x.sel(time=[0, 2, 4]),
            "isel": lambda: x.isel(time=[0, 2, 4]),
        }
        result = results[method]()
        assert not (result.labels.values == -1).any()


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

    def test_add_join_inner_intersects(self, x: Variable, subset: xr.DataArray) -> None:
        """`.add(other, join="inner")` picks the intersection of coords."""
        result = x.add(subset, join="inner")
        assert list(result.coords["time"].values) == [1, 3]

    def test_add_join_outer_fills(self, x: Variable, subset: xr.DataArray) -> None:
        """`.add(other, join="outer")` unions coords (gaps are filled)."""
        result = x.add(subset, join="outer")
        assert list(result.coords["time"].values) == [0, 1, 2, 3, 4]

    def test_mul_join_inner(self, x: Variable, subset: xr.DataArray) -> None:
        result = x.mul(subset, join="inner")
        assert list(result.coords["time"].values) == [1, 3]

    @pytest.mark.v1
    def test_le_join_inner_on_subset_rhs(
        self, x: Variable, subset: xr.DataArray
    ) -> None:
        """`.le(rhs, join="inner")` lets a subset RHS through cleanly."""
        result = x.le(subset, join="inner")
        assert list(result.coords["time"].values) == [1, 3]

    @pytest.mark.legacy
    def test_legacy_le_join_inner_keeps_all_coords(
        self, x: Variable, subset: xr.DataArray
    ) -> None:
        """
        On the constraint path legacy ignores ``join="inner"`` and keeps
        all left coords, leaving the unmatched RHS slots NaN (vs v1's
        clean intersection to [1, 3]).
        """
        result = x.le(subset, join="inner")
        assert list(result.coords["time"].values) == [0, 1, 2, 3, 4]
        rhs = result.rhs.values
        assert np.isnan(rhs[[0, 2, 4]]).all()
        assert rhs[[1, 3]].tolist() == [10.0, 30.0]

    @pytest.mark.v1
    def test_bare_op_still_raises_on_mismatch(
        self, x: Variable, subset: xr.DataArray
    ) -> None:
        """`x + subset` (no `join=`) still raises — opt-in is required."""
        with pytest.raises(ValueError, match="Coordinate mismatch on shared dimension"):
            x + subset

    def test_add_join_override_aligns_positionally(self, x: Variable) -> None:
        """
        ``join="override"`` is the explicit-positional mode — the right
        operand's labels are dropped and the left's are reused. The mode
        is opt-in precisely because it can silently mis-pair if the user
        didn't mean it.
        """
        relabelled = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            dims=["time"],
            coords={"time": pd.Index([10, 11, 12, 13, 14], name="time")},
        )
        result = x.add(relabelled, join="override")
        # Override keeps the left operand's labels — and silently re-uses
        # the right's values at those positions.
        assert list(result.coords["time"].values) == [0, 1, 2, 3, 4]
        assert result.const.values.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_add_join_override_size_mismatch_raises(self, x: Variable) -> None:
        """
        §10 / ``override`` documentation says "positional alignment, made
        explicit". Positional pairing is only well-defined when the
        shared-dim sizes match; with mismatched sizes ``override`` would
        silently mis-pair (or raise opaquely from xarray) instead of
        producing a clear error. Regression for the dropped legacy
        ``other.sizes == self.const.sizes`` gate.
        """
        shorter = xr.DataArray(
            [10.0, 20.0, 30.0],
            dims=["time"],
            coords={"time": pd.Index([0, 1, 2], name="time")},
        )
        with pytest.raises(ValueError, match="join='override' requires matching"):
            x.add(shorter, join="override")

    def test_reindex_like_resolves_mismatch_before_bare_op(self, x: Variable) -> None:
        """
        §10 names ``.reindex(...)`` / ``.reindex_like(...)`` as
        canonical resolutions — pre-aligning lets the bare operator
        accept the once-mismatched operand without ``join=``.
        """
        other = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            dims=["time"],
            coords={"time": pd.Index([10, 11, 12, 13, 14], name="time")},
        )
        aligned = other.reindex_like(x.labels, fill_value=0)
        result = x + aligned  # bare + succeeds because coords now match
        assert list(result.coords["time"].values) == [0, 1, 2, 3, 4]

    def test_assign_coords_resolves_mismatch_before_bare_op(self, x: Variable) -> None:
        """
        ``.assign_coords(...)`` is the explicit-positional escape —
        relabels one side outright so the bare operator's exact-join
        check passes.
        """
        other = xr.DataArray(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            dims=["time"],
            coords={"time": pd.Index([10, 11, 12, 13, 14], name="time")},
        )
        relabelled = other.assign_coords(time=x.coords["time"])
        result = x + relabelled  # bare + succeeds after relabel
        assert list(result.coords["time"].values) == [0, 1, 2, 3, 4]


# =====================================================================
# §12 — constraints follow the same rules
# =====================================================================


_SIGNS = {
    "le": operator.le,
    "ge": operator.ge,
    "eq": operator.eq,
}


class TestConstraintRHS:
    @pytest.mark.v1
    @pytest.mark.parametrize("sign", ["le", "ge", "eq"])
    def test_subset_rhs_raises(self, x: Variable, sign: str) -> None:
        """§12 — all three comparison signs align by §8 the same way."""
        subset = xr.DataArray(
            [10.0, 20.0],
            dims=["time"],
            coords={"time": pd.Index([1, 3], name="time")},
        )
        with pytest.raises(ValueError, match="Coordinate mismatch on shared dimension"):
            _SIGNS[sign](x, subset)

    @pytest.mark.v1
    @pytest.mark.parametrize("sign", ["le", "ge", "eq"])
    def test_nan_rhs_raises(self, x: Variable, time: pd.RangeIndex, sign: str) -> None:
        """
        §5/§12 — a NaN in a user-supplied RHS raises for every sign,
        never silently becomes "no constraint" the way legacy auto_mask
        treats it.
        """
        nan_rhs = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        with pytest.raises(ValueError, match="NaN"):
            _SIGNS[sign](x, nan_rhs)

    @pytest.mark.v1
    @pytest.mark.parametrize("sign", ["le", "ge", "eq"])
    def test_absence_propagates_to_rhs_drops_constraint(
        self, x: Variable, sign: str
    ) -> None:
        """
        §6 → §12 for every sign: a constraint over an absent LHS slot
        yields NaN RHS, which downstream auto-mask interprets as "no
        constraint here".
        """
        xs = x.shift(time=1)
        # xs is absent at time=0; the constraint's RHS at that slot
        # should be NaN (no constraint), not 10.
        constraint = _SIGNS[sign](xs, 10)
        rhs = constraint.rhs.values
        assert np.isnan(rhs[0])
        assert (rhs[1:] == 10).all()

    @pytest.mark.legacy
    @pytest.mark.parametrize("sign", ["le", "ge", "eq"])
    def test_legacy_absence_keeps_rhs_at_absent_slot(
        self, x: Variable, sign: str
    ) -> None:
        """
        Legacy fills the absent LHS slot with 0, so the RHS stays 10
        everywhere and the constraint is emitted at that slot too (vs
        v1, where the NaN RHS drops it).
        """
        xs = x.shift(time=1)
        constraint = _SIGNS[sign](xs, 10)
        assert (constraint.rhs.values == 10).all()

    @pytest.mark.v1
    def test_pypsa_1683_nan_rhs_raises(self, x: Variable, time: pd.RangeIndex) -> None:
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

    @pytest.mark.legacy
    def test_nan_rhs_silently_treated_as_unconstrained(
        self, x: Variable, time: pd.RangeIndex
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
    def test_warn_on_nan_rhs(
        self, x: Variable, time: pd.RangeIndex, unsilenced: None
    ) -> None:
        nan_rhs = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        with pytest.warns(LinopySemanticsWarning, match="no constraint at this row"):
            x <= nan_rhs

    @pytest.mark.legacy
    def test_warn_on_coord_mismatch_rhs_distinguishes_from_nan(
        self, x: Variable, unsilenced: None
    ) -> None:
        """
        A subset RHS has no user NaN — legacy's ``reindex_like`` is what
        introduces the NaN at the unmatched positions. The warning should
        diagnose the *coord mismatch* (fix: ``.sel`` / ``.reindex``), not
        the NaN-RHS auto-mask (fix: ``mask=`` / ``.fillna``). Regression
        for the conflated warn text where both causes used the same
        ``_legacy_nan_rhs_constraint_message``.
        """
        subset = xr.DataArray(
            [10.0, 20.0],
            dims=["time"],
            coords={"time": pd.Index([1, 3], name="time")},
        )
        with pytest.warns(
            LinopySemanticsWarning, match="Coordinate mismatch in constraint RHS"
        ):
            x <= subset

    @pytest.mark.legacy
    def test_both_warnings_fire_when_rhs_has_user_nan_and_mismatch(
        self, x: Variable, unsilenced: None
    ) -> None:
        """
        Independent causes — when the RHS is both subset (mismatch) and
        carries a user NaN, both fix-hints should surface so the caller
        sees each problem with its own resolution.
        """
        both = xr.DataArray(
            [10.0, np.nan],
            dims=["time"],
            coords={"time": pd.Index([1, 3], name="time")},
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", LinopySemanticsWarning)
            x <= both
        messages = [str(w.message) for w in caught]
        assert any("Coordinate mismatch in constraint RHS" in m for m in messages)
        assert any("no constraint at this row" in m for m in messages)


# =====================================================================
# §13 — reductions skip absent slots (not propagate)
# =====================================================================


class TestReductionsSkipAbsent:
    """
    Per §13, ``sum`` / ``groupby.sum`` skip absent slots rather than
    propagating them — the only asymmetry against §6's binary-operator
    rule. The expected behaviour falls out of xarray's ``skipna=True``
    default; these tests pin it under v1 so future changes don't drift.

    Each v1 test is paired with its ``legacy`` counterpart so the
    divergence is pinned on both sides: legacy fills the absent slot
    with 0, so ``xs + 5`` stores ``5`` there too and that extra term is
    counted by the sum (→ 25), whereas v1 skips it (→ 20).

    Scope: §13 also names ``mean``, ``resample``, and ``coarsen``, but
    those are not yet exposed on ``LinearExpression`` (see #703). The
    spec text is the rule they will follow when implemented; tests
    belong with the implementation PR.
    """

    @pytest.fixture
    def xs(self, x: Variable) -> Variable:
        return x.shift(time=1)

    @pytest.mark.v1
    def test_sum_over_dim_skips_absent(self, xs: Variable) -> None:
        """
        ``(xs + 5).sum('time')`` skips the absent slot at t=0 and
        sums the four present 5s → 20.
        """
        result = (xs + 5).sum("time")
        assert float(result.const) == 20.0

    @pytest.mark.v1
    def test_sum_no_dim_skips_absent(self, xs: Variable) -> None:
        result = (xs + 5).sum()
        assert float(result.const) == 20.0

    @pytest.mark.v1
    def test_sum_of_all_absent_is_zero(self, x: Variable) -> None:
        """§13 — "the sum of none is the zero expression.""" ""
        all_absent = x.shift(time=10).to_linexpr()
        assert bool(all_absent.isnull().all().item())
        result = all_absent.sum("time")
        assert float(result.const) == 0.0

    @pytest.mark.v1
    def test_groupby_sum_skips_absent(self, xs: Variable) -> None:
        """Each group's sum drops absent members, just like ``.sum``."""
        groups = xr.DataArray(
            [0, 0, 1, 1, 1], dims=["time"], coords={"time": xs.coords["time"]}
        )
        result = (xs + 5).groupby(groups).sum()
        # group 0: [NaN, 5] → 5; group 1: [5, 5, 5] → 15
        assert result.const.values.tolist() == [5.0, 15.0]

    @pytest.mark.legacy
    def test_sum_over_dim_fills_absent(self, xs: Variable) -> None:
        """
        Legacy fills the absent slot at t=0 with 0, so ``xs + 5`` stores
        ``5`` there and all five 5s are summed → 25 (vs v1's 20).
        """
        result = (xs + 5).sum("time")
        assert float(result.const) == 25.0

    @pytest.mark.legacy
    def test_sum_no_dim_fills_absent(self, xs: Variable) -> None:
        result = (xs + 5).sum()
        assert float(result.const) == 25.0

    @pytest.mark.legacy
    def test_groupby_sum_fills_absent(self, xs: Variable) -> None:
        """Legacy fills the absent member with 0, so group 0 sums [5, 5] → 10."""
        groups = xr.DataArray(
            [0, 0, 1, 1, 1], dims=["time"], coords={"time": xs.coords["time"]}
        )
        result = (xs + 5).groupby(groups).sum()
        # group 0: [5, 5] → 10 (vs v1's 5); group 1: [5, 5, 5] → 15
        assert result.const.values.tolist() == [10.0, 15.0]


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
    def test_expr_plus_dataarray_aux_conflict_raises(
        self, m: Model, A: pd.Index
    ) -> None:
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
    def test_var_plus_var_aux_conflict_raises(self, m: Model, A: pd.Index) -> None:
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        w = m.add_variables(lower=0, coords=[A], name="w").assign_coords(
            B=("A", [400, 400, 500])
        )
        with pytest.raises(ValueError, match="Auxiliary coordinate"):
            v + w

    @pytest.mark.v1
    def test_aux_conflict_survives_reordered_dim(self, m: Model) -> None:
        v = m.add_variables(
            lower=0, coords=[pd.Index(["x", "y", "z"], name="A")], name="v"
        ).assign_coords(B=("A", [1, 2, 3]))
        w = m.add_variables(
            lower=0, coords=[pd.Index(["z", "y", "x"], name="A")], name="w"
        ).assign_coords(B=("A", [1, 2, 3]))
        with pytest.raises(ValueError, match="Auxiliary coordinate"):
            v + w

    @pytest.mark.v1
    def test_mul_constant_aux_conflict_raises(self, m: Model, A: pd.Index) -> None:
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
    def test_constraint_aux_conflict_raises(self, m: Model, A: pd.Index) -> None:
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
    def test_scalar_isel_aux_conflict_raises(self, m: Model, A: pd.Index) -> None:
        """
        Scalar isels leave the indexed dim as a non-dim coord whose
        value differs between operands picked at different positions.
        """
        v = m.add_variables(lower=0, coords=[A], name="v")
        a0 = (1 * v).isel({"A": 0})  # scalar A=1
        a1 = (1 * v).isel({"A": 1})  # scalar A=2
        with pytest.raises(ValueError, match="Auxiliary coordinate"):
            a0 + a1

    def test_isel_with_drop_true_avoids_conflict(self, m: Model, A: pd.Index) -> None:
        """
        The §11 escape hatch the convention recommends: drop the
        leftover scalar coord with ``isel(..., drop=True)``.
        """
        v = m.add_variables(lower=0, coords=[A], name="v")
        a0 = (1 * v).isel({"A": 0}, drop=True)
        a1 = (1 * v).isel({"A": 1}, drop=True)
        result = a0 + a1  # no aux coord → no conflict
        assert "A" not in result.coords

    def test_assign_coords_resolves_conflict(self, m: Model, A: pd.Index) -> None:
        """
        §11's third escape hatch: relabel one side with
        ``.assign_coords`` so the coord values agree across operands.
        """
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        const = xr.DataArray(
            [10.0, 20.0, 30.0],
            dims=["A"],
            coords={"A": A, "B": ("A", [400, 400, 500])},
        )
        relabelled = const.assign_coords(B=v.coords["B"])
        result = v + relabelled
        assert result.coords["B"].values.tolist() == [311, 311, 322]

    @pytest.mark.v1
    def test_multi_operand_merge_aux_conflict_raises(
        self, m: Model, A: pd.Index
    ) -> None:
        """
        The merge-path check inspects all operands, not just two —
        a 3-way ``sum(...)`` where the third disagrees still raises.
        """
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        w = m.add_variables(lower=0, coords=[A], name="w").assign_coords(
            B=("A", [311, 311, 322])
        )
        u = m.add_variables(lower=0, coords=[A], name="u").assign_coords(
            B=("A", [999, 999, 999])
        )
        with pytest.raises(ValueError, match="Auxiliary coordinate"):
            v + w + u

    @pytest.mark.v1
    def test_aux_conflict_raises_under_explicit_join_constant(
        self, m: Model, A: pd.Index
    ) -> None:
        """
        §11 is independent of §8 — an explicit ``join=`` must not
        silence the aux-coord raise. Regression for the
        ``if join is None:`` gating bug where ``.add(const, join="override")``
        would silently drop the conflicting coord.
        """
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        const = xr.DataArray(
            [10.0, 20.0, 30.0],
            dims=["A"],
            coords={"A": A, "B": ("A", [400, 400, 500])},
        )
        for join in ("override", "inner", "outer", "left", "right", "exact"):
            with pytest.raises(ValueError, match="Auxiliary coordinate"):
                v.add(const, join=join)

    @pytest.mark.v1
    def test_aux_conflict_raises_under_explicit_join_merge(
        self, m: Model, A: pd.Index
    ) -> None:
        """
        Same rule on the merge path: ``linopy.merge([v, w], join="override")``
        with a conflicting aux coord must raise.
        """
        import linopy

        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        w = m.add_variables(lower=0, coords=[A], name="w").assign_coords(
            B=("A", [400, 400, 500])
        )
        for join in ("override", "inner", "outer", "left", "right", "exact"):
            with pytest.raises(ValueError, match="Auxiliary coordinate"):
                linopy.merge([1 * v, 1 * w], join=join)

    @pytest.mark.legacy
    def test_aux_conflict_silently_keeps_left(self, m: Model, A: pd.Index) -> None:
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
    def test_warn_on_aux_conflict(
        self, m: Model, A: pd.Index, unsilenced: None
    ) -> None:
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        const = xr.DataArray(
            [10.0, 20.0, 30.0],
            dims=["A"],
            coords={"A": A, "B": ("A", [400, 400, 500])},
        )
        with pytest.warns(LinopySemanticsWarning, match=r"(?s)'B'.*silently dropped"):
            v + const


class TestAuxCoordPropagation:
    """
    Non-conflicting aux coords must propagate through arithmetic and
    into constraints — the positive half of §11.
    """

    @pytest.fixture
    def A(self) -> pd.Index:
        return pd.Index([1, 2, 3], name="A")

    def test_aux_coord_survives_scalar_mul(self, m: Model, A: pd.Index) -> None:
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        assert "B" in (3 * v).coords

    def test_aux_coord_survives_scalar_add(self, m: Model, A: pd.Index) -> None:
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        assert "B" in (v + 5).coords

    def test_aux_coord_propagates_through_var_plus_var(
        self, m: Model, A: pd.Index
    ) -> None:
        B = ("A", [311, 311, 322])
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(B=B)
        w = m.add_variables(lower=0, coords=[A], name="w").assign_coords(B=B)
        result = v + w
        assert "B" in result.coords
        assert result.coords["B"].values.tolist() == [311, 311, 322]

    def test_aux_coord_propagates_into_constraint(self, m: Model, A: pd.Index) -> None:
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        c = v <= 10
        assert "B" in c.coords

    def test_aux_coord_only_on_dataarray_propagates(
        self, m: Model, A: pd.Index
    ) -> None:
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

    def test_aux_coord_only_on_one_side_propagates(self, m: Model, A: pd.Index) -> None:
        """Var+var counterpart of the above — hits the `merge` path."""
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        w = m.add_variables(lower=0, coords=[A], name="w")  # no B
        result = v + w
        assert "B" in result.coords

    def test_aux_coord_object_dtype_with_nan_compares_equal(
        self, m: Model, A: pd.Index
    ) -> None:
        """
        Aux coords with object dtype can embed NaN placeholders (e.g.
        ragged category labels). Two operands with identical NaN
        placement must compare equal — `np.array_equal` alone treats
        NaN as self-unequal on object dtype, so the §11 raise would
        false-positive without the pandas-equals fallback.
        """
        B = np.array([311, np.nan, 322], dtype=object)
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(B=("A", B))
        w = m.add_variables(lower=0, coords=[A], name="w").assign_coords(B=("A", B))
        # Same B on both sides, NaN at the same slot — should propagate, not raise.
        result = v + w
        assert "B" in result.coords


# =====================================================================
# Error-message content (raise self-description)
# =====================================================================


class TestErrorMessageContent:
    """
    The three v1 raises must be self-describing: name the dim or
    coord and show the disagreeing values so the user can act on the
    message without re-running with extra prints. Substring assertions
    elsewhere don't cover this — these tests pin the rich content.
    """

    @pytest.fixture
    def A(self) -> pd.Index:
        return pd.Index([1, 2, 3], name="A")

    @pytest.mark.v1
    def test_user_nan_message_separates_intents(self, x: Variable) -> None:
        """
        The §5 raise must not collapse `data error` and `absence` into a
        single suggestion — they need different fixes.
        """
        with pytest.raises(ValueError) as exc:
            x + float("nan")
        msg = str(exc.value)
        assert "data error" in msg and ".fillna(value)" in msg
        assert "absent" in msg and "mask=" in msg
        assert ".reindex" in msg or ".where(cond)" in msg

    @pytest.mark.v1
    def test_shared_dim_message_names_dim_and_values(
        self, m: Model, time: pd.RangeIndex
    ) -> None:
        """
        The merge-path §8 raise must name the offending dim and show
        both sides' labels — otherwise the user can't tell which dim
        out of many.
        """
        other = m.add_variables(
            lower=0, coords=[pd.Index([10, 11, 12, 13, 14], name="time")], name="other"
        )
        x_local = m.add_variables(lower=0, coords=[time], name="x_local")
        with pytest.raises(ValueError) as exc:
            x_local + other
        msg = str(exc.value)
        assert "'time'" in msg
        assert "[0, 1, 2, 3, 4]" in msg
        assert "[10, 11, 12, 13, 14]" in msg

    @pytest.mark.v1
    def test_aux_conflict_message_names_coord_and_values(
        self, m: Model, A: pd.Index
    ) -> None:
        """
        The §11 raise must name the conflicting coord and show both
        sides' values — and mention `.assign_coords` as a fix, not only
        `.drop_vars` and `isel(drop=True)`.
        """
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        const = xr.DataArray(
            [10.0, 20.0, 30.0],
            dims=["A"],
            coords={"A": A, "B": ("A", [400, 400, 500])},
        )
        with pytest.raises(ValueError) as exc:
            v + const
        msg = str(exc.value)
        assert "'B'" in msg
        assert "conflicting values" in msg  # value-mismatch failure mode
        assert "[311, 311, 322]" in msg
        assert "[400, 400, 500]" in msg
        # All three resolution paths from §11 should be listed.
        assert ".drop_vars" in msg
        assert ".assign_coords" in msg
        assert ".isel" in msg

    @pytest.mark.v1
    def test_aux_conflict_message_distinguishes_shape_vs_value(
        self, m: Model, A: pd.Index
    ) -> None:
        """
        Shape mismatch and value disagreement are different failure
        modes — surface that in the message text so the caller can
        diagnose without re-reading both arrays.
        """
        # scalar-isel leaves a 0-d aux coord on one side; the full vector
        # on the other has a different shape, not a different value.
        v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
            B=("A", [311, 311, 322])
        )
        scalar_side = (1 * v).isel({"A": 0})  # B becomes a 0-d scalar coord
        full_side = 1 * v
        with pytest.raises(ValueError, match="differing shapes") as exc:
            scalar_side + full_side
        msg = str(exc.value)
        assert "'B'" in msg
        assert "shape" in msg


# =====================================================================
# Rough edges — catches NaN that slips past the operator-level check
# =====================================================================


class TestUserNaNEdgeCases:
    """
    Regression guards for three NaN-entry routes that were untested.
    The first two are already caught upstream (at the operator that
    constructs the expression); the third needed an ``add_objective``
    boundary check because a hand-built expression with NaN const
    skips the operator path entirely.
    """

    @pytest.mark.v1
    def test_nan_in_expression_used_in_objective_raises(
        self, m: Model, time: pd.RangeIndex
    ) -> None:
        """
        ``add_objective((x * nan_costs).sum())`` raises at the ``*``
        before the objective even sees the expression — guard against
        a regression that lets NaN-cost objectives slip through.
        """
        x = m.add_variables(lower=0, coords=[time], name="x")
        nan_costs = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        with pytest.raises(ValueError, match="NaN"):
            m.add_objective((x * nan_costs).sum())

    @pytest.mark.v1
    def test_nan_in_constraint_lhs_raises(self, m: Model, time: pd.RangeIndex) -> None:
        """
        ``(x + nan_da) <= 5`` raises at the ``+`` on the LHS — the
        RHS path is tested elsewhere; this guards the symmetric LHS
        case.
        """
        x = m.add_variables(lower=0, coords=[time], name="x")
        nan_da = xr.DataArray(
            [1.0, np.nan, 3.0, 4.0, 5.0], dims=["time"], coords={"time": time}
        )
        with pytest.raises(ValueError, match="NaN"):
            (x + nan_da) <= 5


# =====================================================================
# Object scope — non-linopy operands behave like constant expressions
# =====================================================================


class TestObjectScope:
    """
    Per the convention's object-scope statement, behaviour is
    object-agnostic: ``x OP arr`` builds exactly what ``x OP arr_expr``
    builds, where ``arr_expr`` is the constant-only LinearExpression
    holding ``arr``'s values and coordinates — whatever type ``arr``
    enters as, in either operand position.
    """

    _VALUES = [1.0, 2.0, 3.0, 4.0, 5.0]

    @pytest.fixture
    def da(self, time: pd.RangeIndex) -> xr.DataArray:
        return xr.DataArray(self._VALUES, dims=["time"], coords={"time": time})

    def raw_and_wrapped(
        self, kind: str, m: Model, time: pd.RangeIndex, da: xr.DataArray
    ) -> tuple[Any, Any]:
        """Return a raw constant of the given kind and its constant-expression twin."""
        from linopy import LinearExpression

        if kind == "dataarray":
            return da, LinearExpression(da, m)
        if kind == "series":
            return pd.Series(self._VALUES, index=time), LinearExpression(da, m)
        if kind == "scalar":
            return 7.5, LinearExpression(7.5, m)
        raise AssertionError(kind)

    @pytest.mark.parametrize("op", ["add", "sub", "mul", "radd", "rsub", "rmul"])
    @pytest.mark.parametrize("kind", ["dataarray", "series", "scalar"])
    def test_op_matches_const_expr_op(
        self,
        m: Model,
        x: Variable,
        time: pd.RangeIndex,
        da: xr.DataArray,
        kind: str,
        op: str,
    ) -> None:
        raw, wrapped = self.raw_and_wrapped(kind, m, time, da)
        forward = op in ("add", "sub", "mul")
        opfunc = _OPS[op.removeprefix("r")]
        if forward:
            assert_linequal(opfunc(x, raw), opfunc(x, wrapped))
        else:
            assert_linequal(opfunc(raw, x), opfunc(wrapped, x))

    @pytest.mark.parametrize("kind", ["dataarray", "series", "scalar"])
    def test_distributive_law_mixed_types(
        self,
        m: Model,
        x: Variable,
        time: pd.RangeIndex,
        da: xr.DataArray,
        kind: str,
    ) -> None:
        """``(x + y) * arr`` distributes the same whether ``arr`` is raw or wrapped."""
        raw, wrapped = self.raw_and_wrapped(kind, m, time, da)
        y = m.add_variables(lower=0, coords=[time], name="y")
        assert_linequal((x + y) * raw, x * raw + y * raw)
        assert_linequal((x + y) * raw, (x + y) * wrapped)

    @pytest.mark.parametrize("kind", ["dataarray", "series", "scalar"])
    def test_associative_law_mixed_types(
        self,
        m: Model,
        x: Variable,
        time: pd.RangeIndex,
        da: xr.DataArray,
        kind: str,
    ) -> None:
        """``(x + arr) + y`` and ``x + (arr + y)`` agree for raw constants."""
        raw, _ = self.raw_and_wrapped(kind, m, time, da)
        y = m.add_variables(lower=0, coords=[time], name="y")
        assert_linequal((x + raw) + y, x + (raw + y))

    @pytest.mark.v1
    def test_coord_mismatch_raises_on_either_route(self, m: Model, x: Variable) -> None:
        """§8 fires identically whether the mismatched constant is raw or wrapped."""
        from linopy import LinearExpression

        mismatched = xr.DataArray(
            self._VALUES,
            dims=["time"],
            coords={"time": pd.Index([10, 11, 12, 13, 14], name="time")},
        )
        with pytest.raises(ValueError, match="Coordinate mismatch on shared dimension"):
            x + mismatched
        with pytest.raises(ValueError, match="Coordinate mismatch on shared dimension"):
            x + LinearExpression(mismatched, m)

    def test_division_by_const_expr_is_type_error(
        self, m: Model, x: Variable, da: xr.DataArray
    ) -> None:
        """
        The one type-decided footnote: a constant can be a divisor, an
        expression cannot — even one holding only constants.
        """
        from linopy import LinearExpression

        x / da  # works: dividing by a constant
        with pytest.raises(TypeError):
            x / LinearExpression(da, m)


# =====================================================================
# Cross-cutting guard: operations that MUST agree under both semantics
# =====================================================================
#
# The autouse ``semantics`` fixture runs each test under a single mode, so
# a per-op test that under-asserts (e.g. checks only ``.indexes``) can pass
# under both modes while the actual result silently diverges — that is how
# the reordered-merge mispairing slipped through review. This guard builds
# each mode-invariant operation under *both* semantics and compares them
# with linopy.testing's strict structural helpers (which align by coords
# and compare vars/coeffs/const), so a regression that makes one of these
# paths semantics-dependent fails loudly. Genuinely divergent operations
# belong in the per-section classes above, not here.


def _build_under_both(builder: Any) -> tuple[Any, Any]:
    """Build ``builder()`` under legacy then v1; return ``(legacy, v1)``."""
    from linopy.config import options

    saved = options["semantics"]
    out = {}
    try:
        for sem in ("legacy", "v1"):
            options["semantics"] = sem
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", LinopySemanticsWarning)
                out[sem] = builder()
    finally:
        options["semantics"] = saved
    return out["legacy"], out["v1"]


def _eq_time() -> pd.RangeIndex:
    return pd.RangeIndex(5, name="time")


def _eq_da() -> xr.DataArray:
    return xr.DataArray(np.arange(1.0, 6.0), dims=["time"], coords={"time": _eq_time()})


def _eq_subset() -> xr.DataArray:
    return xr.DataArray(
        [10.0, 30.0], dims=["time"], coords={"time": pd.Index([1, 3], name="time")}
    )


def _op_merge_same_coords() -> LinearExpression:
    m = Model()
    return m.add_variables(lower=0, coords=[_eq_time()], name="a") + m.add_variables(
        lower=0, coords=[_eq_time()], name="b"
    )


def _op_merge_broadcast() -> LinearExpression:
    m = Model()
    a = m.add_variables(lower=0, coords=[_eq_time()], name="a")
    b = m.add_variables(lower=0, coords=[pd.Index([0, 1], name="scenario")], name="b")
    return a + b


def _op_quadratic_same_coords() -> QuadraticExpression:
    m = Model()
    x = m.add_variables(coords=[_eq_time()], name="x")
    y = m.add_variables(coords=[_eq_time()], name="y")
    return cast(QuadraticExpression, (x * x) + (y * y))


def _op_add_join_inner() -> LinearExpression:
    return cast(
        LinearExpression,
        Model()
        .add_variables(coords=[_eq_time()], name="x")
        .add(_eq_subset(), join="inner"),
    )


def _op_add_join_outer() -> LinearExpression:
    return cast(
        LinearExpression,
        Model()
        .add_variables(coords=[_eq_time()], name="x")
        .add(_eq_subset(), join="outer"),
    )


def _op_add_join_override() -> LinearExpression:
    relabelled = xr.DataArray(
        [1.0, 2.0, 3.0, 4.0, 5.0],
        dims=["time"],
        coords={"time": pd.Index([10, 11, 12, 13, 14], name="time")},
    )
    return cast(
        LinearExpression,
        Model()
        .add_variables(coords=[_eq_time()], name="x")
        .add(relabelled, join="override"),
    )


def _op_associative() -> LinearExpression:
    m = Model()
    x = m.add_variables(coords=[_eq_time()], name="x")
    y = m.add_variables(lower=0, coords=[_eq_time()], name="y")
    return (x + _eq_da()) + y


def _op_distributive() -> LinearExpression:
    m = Model()
    x = m.add_variables(coords=[_eq_time()], name="x")
    y = m.add_variables(lower=0, coords=[_eq_time()], name="y")
    return (x + y) * _eq_da()


def _op_raw_matches_wrapped() -> LinearExpression:
    m = Model()
    x = m.add_variables(coords=[_eq_time()], name="x")
    return x + LinearExpression(_eq_da(), m)


def _op_aux_coord_resolved() -> LinearExpression:
    m = Model()
    A = pd.Index([1, 2, 3], name="A")
    v = m.add_variables(lower=0, coords=[A], name="v").assign_coords(
        B=("A", [311, 311, 322])
    )
    const = xr.DataArray(
        [10.0, 20.0, 30.0], dims=["A"], coords={"A": A, "B": ("A", [400, 400, 500])}
    )
    return v + const.assign_coords(B=v.coords["B"])


def _op_clean_constraint() -> Any:
    m = Model()
    x = m.add_variables(coords=[_eq_time()], name="x")
    return x <= _eq_da()


_EQUIVALENT_OPS = [
    pytest.param(_op_merge_same_coords, assert_linequal, id="merge_same_coords"),
    pytest.param(_op_merge_broadcast, assert_linequal, id="merge_broadcast"),
    pytest.param(_op_quadratic_same_coords, assert_quadequal, id="quadratic_same"),
    pytest.param(_op_add_join_inner, assert_linequal, id="add_join_inner"),
    pytest.param(_op_add_join_outer, assert_linequal, id="add_join_outer"),
    pytest.param(_op_add_join_override, assert_linequal, id="add_join_override"),
    pytest.param(_op_associative, assert_linequal, id="associative_law"),
    pytest.param(_op_distributive, assert_linequal, id="distributive_law"),
    pytest.param(_op_raw_matches_wrapped, assert_linequal, id="raw_matches_wrapped"),
    pytest.param(_op_aux_coord_resolved, assert_linequal, id="aux_coord_resolved"),
    pytest.param(_op_clean_constraint, assert_conequal, id="clean_constraint"),
]


@pytest.mark.parametrize("builder, comparator", _EQUIVALENT_OPS)
def test_semantics_invariant_ops_agree(builder: Any, comparator: Any) -> None:
    """Mode-invariant operations must be byte-identical under both semantics."""
    legacy, v1 = _build_under_both(builder)
    comparator(legacy, v1)
