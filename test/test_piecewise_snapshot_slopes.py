"""
Public multidimensional slope integration contracts.

Failure list: snapshot/name coordinates are lost; shared x or y0 fails to
broadcast; leading alignment shifts a piece; ragged padding integrates a
missing piece; malformed counts, interior holes or infinity go unnoticed.
Existing tests cover one entity dimension, not snapshot/name arrays. These
independent arithmetic expectations require no production test seam.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Slopes
from linopy.constants import BREAKPOINT_DIM


def arrays() -> tuple[xr.DataArray, xr.DataArray]:
    coords = {
        "snapshot": pd.Index([2, 7]),
        "name": pd.Index(["a", "b"], dtype="string"),
        BREAKPOINT_DIM: [0, 1, 2],
    }
    x = xr.DataArray(
        [[[0, 1, 3], [0, 2, 4]], [[0, 2, 3], [0, 1, np.nan]]],
        dims=["snapshot", "name", BREAKPOINT_DIM],
        coords=coords,
    )
    slopes = xr.DataArray(
        [[[2, 4], [3, 5]], [[6, 8], [7, np.nan]]],
        dims=x.dims,
        coords={**coords, BREAKPOINT_DIM: [0, 1]},
    )
    return x, slopes


@pytest.mark.parametrize("leading", [False, True])
def test_snapshot_name_slopes_preserve_coordinates_and_padding(leading: bool) -> None:
    x, slopes = arrays()
    if leading:
        first = xr.full_like(
            slopes.isel({BREAKPOINT_DIM: 0}, drop=True), np.nan
        ).expand_dims({BREAKPOINT_DIM: [0]})
        slopes = xr.concat(
            [first, slopes.assign_coords({BREAKPOINT_DIM: [1, 2]})], dim=BREAKPOINT_DIM
        )
    y = Slopes(slopes, y0=1, align="leading" if leading else "pieces").to_breakpoints(x)
    expected = xr.DataArray(
        [[[1, 3, 11], [1, 7, 17]], [[1, 13, 21], [1, 8, np.nan]]],
        dims=x.dims,
        coords=x.coords,
    )
    xr.testing.assert_equal(y.transpose(*x.dims), expected)


def test_static_x_and_per_snapshot_y0_broadcast() -> None:
    x, slopes = arrays()
    slopes = slopes.fillna(9)
    static_x = xr.DataArray([0, 1, 2], dims=[BREAKPOINT_DIM])
    y0 = xr.DataArray([10, 20], dims=["snapshot"], coords={"snapshot": x.snapshot})
    y = Slopes(slopes, y0=y0).to_breakpoints(static_x)
    expected = xr.DataArray(
        [[[10, 12, 16], [10, 13, 18]], [[20, 26, 34], [20, 27, 36]]],
        dims=x.dims,
        coords=x.coords,
    )
    xr.testing.assert_equal(y.transpose(*x.dims), expected)


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("count", "count"),
        ("interior", "trailing"),
        ("infinite", "finite"),
        ("leading", "first slope"),
    ],
)
def test_invalid_snapshot_slopes_raise(invalid: str, message: str) -> None:
    x, slopes = arrays()
    align: Literal["pieces", "leading"] = "pieces"
    if invalid == "count":
        slopes[0, 0, 1] = np.nan
    elif invalid == "interior":
        x[0, 0, 1] = np.nan
    elif invalid == "infinite":
        slopes[0, 0, 0] = np.inf
    else:
        align = "leading"
    with pytest.raises(ValueError, match=message):
        Slopes(slopes, align=align).to_breakpoints(x)


def test_empty_slopes_with_one_x_point_keeps_initial_value() -> None:
    x, slopes = arrays()
    points = x.isel({BREAKPOINT_DIM: slice(0, 1)})
    empty_slopes = slopes.isel({BREAKPOINT_DIM: slice(0, 0)})
    result = Slopes(empty_slopes, y0=3).to_breakpoints(points)
    xr.testing.assert_equal(result, xr.full_like(points, 3))


def test_empty_x_points_raise_owned_error() -> None:
    x, slopes = arrays()
    with pytest.raises(ValueError, match="at least one x_point"):
        Slopes(slopes).to_breakpoints(x.isel({BREAKPOINT_DIM: slice(0, 0)}))


def test_mismatched_entity_coordinates_are_rejected() -> None:
    x, slopes = arrays()
    with pytest.raises(ValueError, match="align.*exact"):
        Slopes(slopes).to_breakpoints(x.assign_coords(snapshot=[2, 8]))


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("unknown-axis", "entity dimensions"),
        ("missing-coordinate", "align.*exact"),
        ("nan", "finite"),
        ("infinite", "finite"),
    ],
)
def test_invalid_initial_values_are_rejected(invalid: str, message: str) -> None:
    x, slopes = arrays()
    y0: xr.DataArray | float
    if invalid == "unknown-axis":
        y0 = xr.DataArray([1], dims=["unrelated"])
    elif invalid == "missing-coordinate":
        y0 = xr.DataArray([1], dims=["snapshot"], coords={"snapshot": [2]})
    elif invalid == "nan":
        y0 = np.nan
    else:
        y0 = np.inf
    with pytest.raises(ValueError, match=message):
        Slopes(slopes, y0=y0).to_breakpoints(x)
