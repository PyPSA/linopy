#!/usr/bin/env python3
"""
Created on Mon Jun 19 12:11:03 2023

@author: fabian
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from linopy import Model
from linopy.common import (
    assign_multiindex_safe,
    best_int,
    coords_from_dataset,
    coords_to_dataset_vars,
    get_dims_with_index_levels,
    is_constant,
    iterate_slices,
    maybe_group_terms_polars,
)


def test_best_int() -> None:
    # Test for int8
    assert best_int(127) == np.int8
    # Test for int16
    assert best_int(128) == np.int16
    assert best_int(32767) == np.int16
    # Test for int32
    assert best_int(32768) == np.int32
    assert best_int(2147483647) == np.int32
    # Test for int64
    assert best_int(2147483648) == np.int64
    assert best_int(9223372036854775807) == np.int64

    # Test for value too large
    with pytest.raises(
        ValueError, match=r"Value 9223372036854775808 is too large for int64."
    ):
        best_int(9223372036854775808)


def test_assign_multiindex_safe() -> None:
    # Create a multi-indexed dataset
    index = pd.MultiIndex.from_product([["A", "B"], [1, 2]], names=["letter", "number"])
    data = xr.DataArray([1, 2, 3, 4], dims=["index"], coords={"index": index})
    ds = xr.Dataset({"value": data})

    # This would now warn about the index deletion of single index level
    # ds["humidity"] = data

    # Case 1: Assigning a single DataArray
    result = assign_multiindex_safe(ds, humidity=data)
    assert "humidity" in result
    assert "value" in result
    assert result["humidity"].equals(data)

    # Case 2: Assigning a Dataset
    result = assign_multiindex_safe(ds, **xr.Dataset({"humidity": data}))  # type: ignore
    assert "humidity" in result
    assert "value" in result
    assert result["humidity"].equals(data)

    # Case 3: Assigning multiple DataArrays
    result = assign_multiindex_safe(ds, humidity=data, pressure=data)
    assert "humidity" in result
    assert "pressure" in result
    assert "value" in result
    assert result["humidity"].equals(data)
    assert result["pressure"].equals(data)


def test_coords_dataset_vars_roundtrip_multiindex() -> None:
    """MultiIndex and plain coords survive serialization to Dataset vars and back."""
    mi = pd.MultiIndex.from_product(
        [[2020, 2030], ["t1", "t2"]], names=("period", "timestep")
    )
    mi.name = "snapshot"
    plain = pd.Index([1, 2, 3], name="simple")

    ds = xr.Dataset(coords_to_dataset_vars([mi, plain]))
    restored = coords_from_dataset(ds, ["snapshot", "simple"])

    assert isinstance(restored[0], pd.MultiIndex)
    assert restored[0].equals(mi)
    assert list(restored[0].names) == ["period", "timestep"]
    assert restored[0].name == "snapshot"
    assert restored[1].equals(plain)
    assert restored[1].name == "simple"


def test_iterate_slices_basic() -> None:
    ds = xr.Dataset(
        {"var": (("x", "y"), np.random.rand(10, 10))},  # noqa: NPY002
        coords={"x": np.arange(10), "y": np.arange(10)},
    )
    slices = list(iterate_slices(ds, slice_size=20))
    assert len(slices) == 5
    for s in slices:
        assert isinstance(s, xr.Dataset)
        assert set(s.dims) == set(ds.dims)


def test_iterate_slices_with_exclude_dims() -> None:
    ds = xr.Dataset(
        {"var": (("x", "y"), np.random.rand(10, 20))},  # noqa: NPY002
        coords={"x": np.arange(10), "y": np.arange(20)},
    )
    slices = list(iterate_slices(ds, slice_size=20, slice_dims=["x"]))
    assert len(slices) == 10
    for s in slices:
        assert isinstance(s, xr.Dataset)
        assert set(s.dims) == set(ds.dims)


def test_iterate_slices_large_max_size() -> None:
    ds = xr.Dataset(
        {"var": (("x", "y"), np.random.rand(10, 10))},  # noqa: NPY002
        coords={"x": np.arange(10), "y": np.arange(10)},
    )
    slices = list(iterate_slices(ds, slice_size=200))
    assert len(slices) == 1
    for s in slices:
        assert isinstance(s, xr.Dataset)
        assert set(s.dims) == set(ds.dims)


def test_iterate_slices_small_max_size() -> None:
    ds = xr.Dataset(
        {"var": (("x", "y"), np.random.rand(10, 20))},  # noqa: NPY002
        coords={"x": np.arange(10), "y": np.arange(20)},
    )
    slices = list(iterate_slices(ds, slice_size=8, slice_dims=["x"]))
    assert (
        len(slices) == 10
    )  # goes to the smallest slice possible which is 1 for the x dimension
    for s in slices:
        assert isinstance(s, xr.Dataset)
        assert set(s.dims) == set(ds.dims)


def test_iterate_slices_slice_size_none() -> None:
    ds = xr.Dataset(
        {"var": (("x", "y"), np.random.rand(10, 10))},  # noqa: NPY002
        coords={"x": np.arange(10), "y": np.arange(10)},
    )
    slices = list(iterate_slices(ds, slice_size=None))
    assert len(slices) == 1
    for s in slices:
        assert ds.equals(s)


def test_iterate_slices_includes_last_slice() -> None:
    ds = xr.Dataset(
        {"var": (("x"), np.random.rand(10))},  # noqa: NPY002
        coords={"x": np.arange(10)},
    )
    slices = list(iterate_slices(ds, slice_size=3, slice_dims=["x"]))
    assert len(slices) == 4  # 10 slices for dimension 'x' with size 10
    total_elements = sum(s.sizes["x"] for s in slices)
    assert total_elements == ds.sizes["x"]  # Ensure all elements are included
    for s in slices:
        assert isinstance(s, xr.Dataset)
        assert set(s.dims) == set(ds.dims)


def test_iterate_slices_empty_slice_dims() -> None:
    ds = xr.Dataset(
        {"var": (("x", "y"), np.random.rand(10, 10))},  # noqa: NPY002
        coords={"x": np.arange(10), "y": np.arange(10)},
    )
    slices = list(iterate_slices(ds, slice_size=50, slice_dims=[]))
    assert len(slices) == 1
    for s in slices:
        assert ds.equals(s)


def test_iterate_slices_invalid_slice_dims() -> None:
    ds = xr.Dataset(
        {"var": (("x", "y"), np.random.rand(10, 10))},  # noqa: NPY002
        coords={"x": np.arange(10), "y": np.arange(10)},
    )
    with pytest.raises(ValueError):
        list(iterate_slices(ds, slice_size=50, slice_dims=["z"]))


def test_iterate_slices_empty_dataset() -> None:
    ds = xr.Dataset(
        {"var": (("x", "y"), np.array([]).reshape(0, 0))}, coords={"x": [], "y": []}
    )
    slices = list(iterate_slices(ds, slice_size=10, slice_dims=["x"]))
    assert len(slices) == 1
    assert ds.equals(slices[0])


def test_iterate_slices_single_element() -> None:
    ds = xr.Dataset({"var": (("x", "y"), np.array([[1]]))}, coords={"x": [0], "y": [0]})
    slices = list(iterate_slices(ds, slice_size=1, slice_dims=["x"]))
    assert len(slices) == 1
    assert ds.equals(slices[0])


def test_get_dims_with_index_levels() -> None:
    # Create test data

    # Case 1: Simple dataset with regular dimensions
    ds1 = xr.Dataset(
        {"temp": (("time", "lat"), np.random.rand(3, 2))},  # noqa: NPY002
        coords={"time": pd.date_range("2024-01-01", periods=3), "lat": [0, 1]},
    )

    # Case 2: Dataset with a multi-index dimension
    stations_index = pd.MultiIndex.from_product(
        [["USA", "Canada"], ["NYC", "Toronto"]], names=["country", "city"]
    )
    stations_coords = xr.Coordinates.from_pandas_multiindex(stations_index, "station")
    ds2 = xr.Dataset(
        {"temp": (("time", "station"), np.random.rand(3, 4))},  # noqa: NPY002
        coords={"time": pd.date_range("2024-01-01", periods=3), **stations_coords},
    )

    # Case 3: Dataset with unnamed multi-index levels
    unnamed_stations_index = pd.MultiIndex.from_product(
        [["USA", "Canada"], ["NYC", "Toronto"]]
    )
    unnamed_stations_coords = xr.Coordinates.from_pandas_multiindex(
        unnamed_stations_index, "station"
    )
    ds3 = xr.Dataset(
        {"temp": (("time", "station"), np.random.rand(3, 4))},  # noqa: NPY002
        coords={
            "time": pd.date_range("2024-01-01", periods=3),
            **unnamed_stations_coords,
        },
    )

    # Case 4: Dataset with multiple multi-indexed dimensions
    locations_index = pd.MultiIndex.from_product(
        [["North", "South"], ["A", "B"]], names=["region", "site"]
    )
    locations_coords = xr.Coordinates.from_pandas_multiindex(
        locations_index, "location"
    )

    ds4 = xr.Dataset(
        {"temp": (("time", "station", "location"), np.random.rand(2, 4, 4))},  # noqa: NPY002
        coords={
            "time": pd.date_range("2024-01-01", periods=2),
            **stations_coords,
            **locations_coords,
        },
    )

    # Run tests

    # Test case 1: Regular dimensions
    assert get_dims_with_index_levels(ds1) == ["time", "lat"]

    # Test case 2: Named multi-index
    assert get_dims_with_index_levels(ds2) == ["time", "station (country, city)"]

    # Test case 3: Unnamed multi-index
    assert get_dims_with_index_levels(ds3) == [
        "time",
        "station (station_level_0, station_level_1)",
    ]

    # Test case 4: Multiple multi-indices
    expected = ["time", "station (country, city)", "location (region, site)"]
    assert get_dims_with_index_levels(ds4) == expected

    # Test case 5: Empty dataset
    ds5 = xr.Dataset()
    assert get_dims_with_index_levels(ds5) == []


def test_is_constant() -> None:
    model = Model()
    index = pd.Index(range(10), name="t")
    a = model.add_variables(name="a", coords=[index])
    b = a.sel(t=1)
    c = a * 2
    d = a * a

    non_constant = [a, b, c, d]
    for nc in non_constant:
        assert not is_constant(nc)

    constant_values = [
        5,
        3.14,
        np.int32(7),
        np.float64(2.71),
        pd.Series([1, 2, 3]),
        np.array([4, 5, 6]),
        xr.DataArray([k for k in range(10)], coords=[index]),
    ]
    for cv in constant_values:
        assert is_constant(cv)


def test_maybe_group_terms_polars_no_duplicates() -> None:
    """Fast path: distinct (labels, vars) pairs skip group_by."""
    df = pl.DataFrame({"labels": [0, 0], "vars": [1, 2], "coeffs": [3.0, 4.0]})
    result = maybe_group_terms_polars(df)
    assert result.shape == (2, 3)
    assert result.columns == ["labels", "vars", "coeffs"]
    assert result["coeffs"].to_list() == [3.0, 4.0]


def test_maybe_group_terms_polars_with_duplicates() -> None:
    """Slow path: duplicate (labels, vars) pairs trigger group_by."""
    df = pl.DataFrame({"labels": [0, 0], "vars": [1, 1], "coeffs": [3.0, 4.0]})
    result = maybe_group_terms_polars(df)
    assert result.shape == (1, 3)
    assert result["coeffs"].to_list() == [7.0]
