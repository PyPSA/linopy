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
