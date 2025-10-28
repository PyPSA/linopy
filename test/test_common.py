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
from test_linear_expression import m, u, x  # noqa: F401
from xarray import DataArray
from xarray.testing.assertions import assert_equal

from linopy import LinearExpression, Variable
from linopy.common import (
    align,
    as_dataarray,
    assign_multiindex_safe,
    best_int,
    get_dims_with_index_levels,
    iterate_slices,
)
from linopy.testing import assert_linequal, assert_varequal


def test_as_dataarray_with_series_dims_default() -> None:
    target_dim = "dim_0"
    target_index = [0, 1, 2]
    s = pd.Series([1, 2, 3])
    da = as_dataarray(s)
    assert isinstance(da, DataArray)
    assert da.dims == (target_dim,)
    assert list(da.coords[target_dim].values) == target_index


def test_as_dataarray_with_series_dims_set() -> None:
    target_dim = "dim1"
    target_index = ["a", "b", "c"]
    s = pd.Series([1, 2, 3], index=target_index)
    dims = [target_dim]
    da = as_dataarray(s, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == (target_dim,)
    assert list(da.coords[target_dim].values) == target_index


def test_as_dataarray_with_series_dims_given() -> None:
    target_dim = "dim1"
    target_index = ["a", "b", "c"]
    index = pd.Index(target_index, name=target_dim)
    s = pd.Series([1, 2, 3], index=index)
    dims: list[str] = []
    da = as_dataarray(s, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == (target_dim,)
    assert list(da.coords[target_dim].values) == target_index


def test_as_dataarray_with_series_dims_priority() -> None:
    """The dimension name from the pandas object should have priority."""
    target_dim = "dim1"
    target_index = ["a", "b", "c"]
    index = pd.Index(target_index, name=target_dim)
    s = pd.Series([1, 2, 3], index=index)
    dims = ["other"]
    da = as_dataarray(s, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == (target_dim,)
    assert list(da.coords[target_dim].values) == target_index


def test_as_dataarray_with_series_dims_subset() -> None:
    target_dim = "dim_0"
    target_index = ["a", "b", "c"]
    s = pd.Series([1, 2, 3], index=target_index)
    dims: list[str] = []
    da = as_dataarray(s, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == (target_dim,)
    assert list(da.coords[target_dim].values) == target_index


def test_as_dataarray_with_series_dims_superset() -> None:
    target_dim = "dim_a"
    target_index = ["a", "b", "c"]
    s = pd.Series([1, 2, 3], index=target_index)
    dims = [target_dim, "other"]
    da = as_dataarray(s, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == (target_dim,)
    assert list(da.coords[target_dim].values) == target_index


def test_as_dataarray_with_series_override_coords() -> None:
    target_dim = "dim_0"
    target_index = ["a", "b", "c"]
    s = pd.Series([1, 2, 3], index=target_index)
    with pytest.warns(UserWarning):
        da = as_dataarray(s, coords=[[1, 2, 3]])
    assert isinstance(da, DataArray)
    assert da.dims == (target_dim,)
    assert list(da.coords[target_dim].values) == target_index


def test_as_dataarray_with_series_aligned_coords() -> None:
    """This should not give out a warning even though coords are given."""
    target_dim = "dim_0"
    target_index = ["a", "b", "c"]
    s = pd.Series([1, 2, 3], index=target_index)
    da = as_dataarray(s, coords=[target_index])
    assert isinstance(da, DataArray)
    assert da.dims == (target_dim,)
    assert list(da.coords[target_dim].values) == target_index

    da = as_dataarray(s, coords={target_dim: target_index})
    assert isinstance(da, DataArray)
    assert da.dims == (target_dim,)
    assert list(da.coords[target_dim].values) == target_index


def test_as_dataarray_with_pl_series_dims_default() -> None:
    target_dim = "dim_0"
    target_index = [0, 1, 2]
    s = pl.Series([1, 2, 3])
    da = as_dataarray(s)
    assert isinstance(da, DataArray)
    assert da.dims == (target_dim,)
    assert list(da.coords[target_dim].values) == target_index


def test_as_dataarray_dataframe_dims_default() -> None:
    target_dims = ("dim_0", "dim_1")
    target_index = [0, 1]
    target_columns = ["A", "B"]
    df = pd.DataFrame([[1, 2], [3, 4]], index=target_index, columns=target_columns)
    da = as_dataarray(df)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    assert list(da.coords[target_dims[0]].values) == target_index
    assert list(da.coords[target_dims[1]].values) == target_columns


def test_as_dataarray_dataframe_dims_set() -> None:
    target_dims = ("dim1", "dim2")
    target_index = ["a", "b"]
    target_columns = ["A", "B"]
    df = pd.DataFrame([[1, 2], [3, 4]], index=target_index, columns=target_columns)
    da = as_dataarray(df, dims=target_dims)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    assert list(da.coords[target_dims[0]].values) == target_index
    assert list(da.coords[target_dims[1]].values) == target_columns


def test_as_dataarray_dataframe_dims_given() -> None:
    target_dims = ("dim1", "dim2")
    target_index = ["a", "b"]
    target_columns = ["A", "B"]
    index = pd.Index(target_index, name=target_dims[0])
    columns = pd.Index(target_columns, name=target_dims[1])
    df = pd.DataFrame([[1, 2], [3, 4]], index=index, columns=columns)
    dims: list[str] = []
    da = as_dataarray(df, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    assert list(da.coords[target_dims[0]].values) == target_index
    assert list(da.coords[target_dims[1]].values) == target_columns


def test_as_dataarray_dataframe_dims_priority() -> None:
    """The dimension name from the pandas object should have priority."""
    target_dims = ("dim1", "dim2")
    target_index = ["a", "b"]
    target_columns = ["A", "B"]
    index = pd.Index(target_index, name=target_dims[0])
    columns = pd.Index(target_columns, name=target_dims[1])
    df = pd.DataFrame([[1, 2], [3, 4]], index=index, columns=columns)
    dims = ["other"]
    da = as_dataarray(df, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    assert list(da.coords[target_dims[0]].values) == target_index
    assert list(da.coords[target_dims[1]].values) == target_columns


def test_as_dataarray_dataframe_dims_subset() -> None:
    target_dims = ("dim_0", "dim_1")
    target_index = ["a", "b"]
    target_columns = ["A", "B"]
    df = pd.DataFrame([[1, 2], [3, 4]], index=target_index, columns=target_columns)
    dims: list[str] = []
    da = as_dataarray(df, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    assert list(da.coords[target_dims[0]].values) == target_index
    assert list(da.coords[target_dims[1]].values) == target_columns


def test_as_dataarray_dataframe_dims_superset() -> None:
    target_dims = ("dim_a", "dim_b")
    target_index = ["a", "b"]
    target_columns = ["A", "B"]
    df = pd.DataFrame([[1, 2], [3, 4]], index=target_index, columns=target_columns)
    dims = [*target_dims, "other"]
    da = as_dataarray(df, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    assert list(da.coords[target_dims[0]].values) == target_index
    assert list(da.coords[target_dims[1]].values) == target_columns


def test_as_dataarray_dataframe_override_coords() -> None:
    target_dims = ("dim_0", "dim_1")
    target_index = ["a", "b"]
    target_columns = ["A", "B"]
    df = pd.DataFrame([[1, 2], [3, 4]], index=target_index, columns=target_columns)
    with pytest.warns(UserWarning):
        da = as_dataarray(df, coords=[[1, 2], [2, 3]])
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    assert list(da.coords[target_dims[0]].values) == target_index
    assert list(da.coords[target_dims[1]].values) == target_columns


def test_as_dataarray_dataframe_aligned_coords() -> None:
    """This should not give out a warning even though coords are given."""
    target_dims = ("dim_0", "dim_1")
    target_index = ["a", "b"]
    target_columns = ["A", "B"]
    df = pd.DataFrame([[1, 2], [3, 4]], index=target_index, columns=target_columns)
    da = as_dataarray(df, coords=[target_index, target_columns])
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    assert list(da.coords[target_dims[0]].values) == target_index
    assert list(da.coords[target_dims[1]].values) == target_columns

    coords = dict(zip(target_dims, [target_index, target_columns]))
    da = as_dataarray(df, coords=coords)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    assert list(da.coords[target_dims[0]].values) == target_index
    assert list(da.coords[target_dims[1]].values) == target_columns


def test_as_dataarray_with_ndarray_no_coords_no_dims() -> None:
    target_dims = ("dim_0", "dim_1")
    target_coords = [[0, 1], [0, 1]]
    arr = np.array([[1, 2], [3, 4]])
    da = as_dataarray(arr)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    for i, dim in enumerate(target_dims):
        assert list(da.coords[dim]) == target_coords[i]


def test_as_dataarray_with_ndarray_coords_list_no_dims() -> None:
    target_dims = ("dim_0", "dim_1")
    target_coords = [["a", "b"], ["A", "B"]]
    arr = np.array([[1, 2], [3, 4]])
    da = as_dataarray(arr, coords=target_coords)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    for i, dim in enumerate(target_dims):
        assert list(da.coords[dim]) == target_coords[i]


def test_as_dataarray_with_ndarray_coords_indexes_no_dims() -> None:
    target_dims = ("dim1", "dim2")
    target_coords = [
        pd.Index(["a", "b"], name="dim1"),
        pd.Index(["A", "B"], name="dim2"),
    ]
    arr = np.array([[1, 2], [3, 4]])
    da = as_dataarray(arr, coords=target_coords)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    for i, dim in enumerate(target_dims):
        assert list(da.coords[dim]) == list(target_coords[i])


def test_as_dataarray_with_ndarray_coords_dict_set_no_dims() -> None:
    """If no dims are given and coords are a dict, the keys of the dict should be used as dims."""
    target_dims = ("dim_0", "dim_2")
    target_coords = {"dim_0": ["a", "b"], "dim_2": ["A", "B"]}
    arr = np.array([[1, 2], [3, 4]])
    da = as_dataarray(arr, coords=target_coords)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    for dim in target_dims:
        assert list(da.coords[dim]) == target_coords[dim]


def test_as_dataarray_with_ndarray_coords_list_dims() -> None:
    target_dims = ("dim1", "dim2")
    target_coords = [["a", "b"], ["A", "B"]]
    arr = np.array([[1, 2], [3, 4]])
    da = as_dataarray(arr, coords=target_coords, dims=target_dims)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    for i, dim in enumerate(target_dims):
        assert list(da.coords[dim]) == target_coords[i]


def test_as_dataarray_with_ndarray_coords_list_dims_superset() -> None:
    target_dims = ("dim1", "dim2")
    target_coords = [["a", "b"], ["A", "B"]]
    arr = np.array([[1, 2], [3, 4]])
    dims = [*target_dims, "dim3"]
    da = as_dataarray(arr, coords=target_coords, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    for i, dim in enumerate(target_dims):
        assert list(da.coords[dim]) == target_coords[i]


def test_as_dataarray_with_ndarray_coords_list_dims_subset() -> None:
    target_dims = ("dim0", "dim_1")
    target_coords = [["a", "b"], ["A", "B"]]
    arr = np.array([[1, 2], [3, 4]])
    dims = ["dim0"]
    da = as_dataarray(arr, coords=target_coords, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    for i, dim in enumerate(target_dims):
        assert list(da.coords[dim]) == target_coords[i]


def test_as_dataarray_with_ndarray_coords_indexes_dims_aligned() -> None:
    target_dims = ("dim1", "dim2")
    target_coords = [
        pd.Index(["a", "b"], name="dim1"),
        pd.Index(["A", "B"], name="dim2"),
    ]
    arr = np.array([[1, 2], [3, 4]])
    da = as_dataarray(arr, coords=target_coords, dims=target_dims)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    for i, dim in enumerate(target_dims):
        assert list(da.coords[dim]) == list(target_coords[i])


def test_as_dataarray_with_ndarray_coords_indexes_dims_not_aligned() -> None:
    target_dims = ("dim3", "dim4")
    target_coords = [
        pd.Index(["a", "b"], name="dim1"),
        pd.Index(["A", "B"], name="dim2"),
    ]
    arr = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        as_dataarray(arr, coords=target_coords, dims=target_dims)


def test_as_dataarray_with_ndarray_coords_dict_dims_aligned() -> None:
    target_dims = ("dim_0", "dim_1")
    target_coords = {"dim_0": ["a", "b"], "dim_1": ["A", "B"]}
    arr = np.array([[1, 2], [3, 4]])
    da = as_dataarray(arr, coords=target_coords, dims=target_dims)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    for dim in target_dims:
        assert list(da.coords[dim]) == target_coords[dim]


def test_as_dataarray_with_ndarray_coords_dict_set_dims_not_aligned() -> None:
    target_dims = ("dim_0", "dim_1")
    target_coords = {"dim_0": ["a", "b"], "dim_2": ["A", "B"]}
    arr = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        as_dataarray(arr, coords=target_coords, dims=target_dims)


def test_as_dataarray_with_number() -> None:
    num = 1
    da = as_dataarray(num, dims=["dim1"], coords=[["a"]])
    assert isinstance(da, DataArray)
    assert da.dims == ("dim1",)
    assert list(da.coords["dim1"].values) == ["a"]


def test_as_dataarray_with_np_number() -> None:
    num = np.float64(1)
    da = as_dataarray(num, dims=["dim1"], coords=[["a"]])
    assert isinstance(da, DataArray)
    assert da.dims == ("dim1",)
    assert list(da.coords["dim1"].values) == ["a"]


def test_as_dataarray_with_number_default_dims_coords() -> None:
    num = 1
    da = as_dataarray(num)
    assert isinstance(da, DataArray)
    assert da.dims == ()
    assert da.coords == {}


def test_as_dataarray_with_number_and_coords() -> None:
    num = 1
    da = as_dataarray(num, coords=[pd.RangeIndex(10, name="a")])
    assert isinstance(da, DataArray)
    assert da.dims == ("a",)
    assert list(da.coords["a"].values) == list(range(10))


def test_as_dataarray_with_dataarray() -> None:
    da_in = DataArray(
        data=[[1, 2], [3, 4]],
        dims=["dim1", "dim2"],
        coords={"dim1": ["a", "b"], "dim2": ["A", "B"]},
    )
    da_out = as_dataarray(da_in, dims=["dim1", "dim2"], coords=[["a", "b"], ["A", "B"]])
    assert isinstance(da_out, DataArray)
    assert da_out.dims == da_in.dims
    assert list(da_out.coords["dim1"].values) == list(da_in.coords["dim1"].values)
    assert list(da_out.coords["dim2"].values) == list(da_in.coords["dim2"].values)


def test_as_dataarray_with_dataarray_default_dims_coords() -> None:
    da_in = DataArray(
        data=[[1, 2], [3, 4]],
        dims=["dim1", "dim2"],
        coords={"dim1": ["a", "b"], "dim2": ["A", "B"]},
    )
    da_out = as_dataarray(da_in)
    assert isinstance(da_out, DataArray)
    assert da_out.dims == da_in.dims
    assert list(da_out.coords["dim1"].values) == list(da_in.coords["dim1"].values)
    assert list(da_out.coords["dim2"].values) == list(da_in.coords["dim2"].values)


def test_as_dataarray_with_unsupported_type() -> None:
    with pytest.raises(TypeError):
        as_dataarray(lambda x: 1, dims=["dim1"], coords=[["a"]])


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


def test_align(x: Variable, u: Variable) -> None:  # noqa: F811
    alpha = xr.DataArray([1, 2], [[1, 2]])
    beta = xr.DataArray(
        [1, 2, 3],
        [
            (
                "dim_3",
                pd.MultiIndex.from_tuples(
                    [(1, "b"), (2, "b"), (1, "c")], names=["level1", "level2"]
                ),
            )
        ],
    )

    # inner join
    x_obs, alpha_obs = align(x, alpha)
    assert isinstance(x_obs, Variable)
    assert x_obs.shape == alpha_obs.shape == (1,)
    assert_varequal(x_obs, x.loc[[1]])

    # left-join
    x_obs, alpha_obs = align(x, alpha, join="left")
    assert x_obs.shape == alpha_obs.shape == (2,)
    assert isinstance(x_obs, Variable)
    assert_varequal(x_obs, x)
    assert_equal(alpha_obs, DataArray([np.nan, 1], [[0, 1]]))

    # multiindex
    beta_obs, u_obs = align(beta, u)
    assert u_obs.shape == beta_obs.shape == (2,)
    assert isinstance(u_obs, Variable)
    assert_varequal(u_obs, u.loc[[(1, "b"), (2, "b")]])
    assert_equal(beta_obs, beta.loc[[(1, "b"), (2, "b")]])

    # with linear expression
    expr = 20 * x
    x_obs, expr_obs, alpha_obs = align(x, expr, alpha)
    assert x_obs.shape == alpha_obs.shape == (1,)
    assert expr_obs.shape == (1, 1)  # _term dim
    assert isinstance(expr_obs, LinearExpression)
    assert_linequal(expr_obs, expr.loc[[1]])
