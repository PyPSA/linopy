#!/usr/bin/env python3
"""
Created on Mon Jun 19 12:11:03 2023

@author: fabian
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray import DataArray

from linopy.common import as_dataarray, assign_multiindex_safe, best_int


def test_as_dataarray_with_series_dims_default():
    target_dim = "dim_0"
    target_index = [0, 1, 2]
    s = pd.Series([1, 2, 3])
    da = as_dataarray(s)
    assert isinstance(da, DataArray)
    assert da.dims == (target_dim,)
    assert list(da.coords[target_dim].values) == target_index


def test_as_dataarray_with_series_dims_set():
    target_dim = "dim1"
    target_index = ["a", "b", "c"]
    s = pd.Series([1, 2, 3], index=target_index)
    dims = [target_dim]
    da = as_dataarray(s, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == (target_dim,)
    assert list(da.coords[target_dim].values) == target_index


def test_as_dataarray_with_series_dims_given():
    target_dim = "dim1"
    target_index = ["a", "b", "c"]
    index = pd.Index(target_index, name=target_dim)
    s = pd.Series([1, 2, 3], index=index)
    dims = []
    da = as_dataarray(s, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == (target_dim,)
    assert list(da.coords[target_dim].values) == target_index


def test_as_dataarray_with_series_dims_priority():
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


def test_as_dataarray_with_series_dims_subset():
    target_dim = "dim_0"
    target_index = ["a", "b", "c"]
    s = pd.Series([1, 2, 3], index=target_index)
    dims = []
    da = as_dataarray(s, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == (target_dim,)
    assert list(da.coords[target_dim].values) == target_index


def test_as_dataarray_with_series_dims_superset():
    target_dim = "dim_a"
    target_index = ["a", "b", "c"]
    s = pd.Series([1, 2, 3], index=target_index)
    dims = [target_dim, "other"]
    da = as_dataarray(s, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == (target_dim,)
    assert list(da.coords[target_dim].values) == target_index


def test_as_dataarray_with_series_override_coords():
    target_dim = "dim_0"
    target_index = ["a", "b", "c"]
    s = pd.Series([1, 2, 3], index=target_index)
    with pytest.warns(UserWarning):
        da = as_dataarray(s, coords=[[1, 2, 3]])
    assert isinstance(da, DataArray)
    assert da.dims == (target_dim,)
    assert list(da.coords[target_dim].values) == target_index


def test_as_dataarray_with_series_aligned_coords():
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


def test_as_dataarray_dataframe_dims_default():
    target_dims = ("dim_0", "dim_1")
    target_index = [0, 1]
    target_columns = ["A", "B"]
    df = pd.DataFrame([[1, 2], [3, 4]], index=target_index, columns=target_columns)
    da = as_dataarray(df)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    assert list(da.coords[target_dims[0]].values) == target_index
    assert list(da.coords[target_dims[1]].values) == target_columns


def test_as_dataarray_dataframe_dims_set():
    target_dims = ("dim1", "dim2")
    target_index = ["a", "b"]
    target_columns = ["A", "B"]
    df = pd.DataFrame([[1, 2], [3, 4]], index=target_index, columns=target_columns)
    da = as_dataarray(df, dims=target_dims)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    assert list(da.coords[target_dims[0]].values) == target_index
    assert list(da.coords[target_dims[1]].values) == target_columns


def test_as_dataarray_dataframe_dims_given():
    target_dims = ("dim1", "dim2")
    target_index = ["a", "b"]
    target_columns = ["A", "B"]
    index = pd.Index(target_index, name=target_dims[0])
    columns = pd.Index(target_columns, name=target_dims[1])
    df = pd.DataFrame([[1, 2], [3, 4]], index=index, columns=columns)
    dims = []
    da = as_dataarray(df, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    assert list(da.coords[target_dims[0]].values) == target_index
    assert list(da.coords[target_dims[1]].values) == target_columns


def test_as_dataarray_dataframe_dims_priority():
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


def test_as_dataarray_dataframe_dims_subset():
    target_dims = ("dim_0", "dim_1")
    target_index = ["a", "b"]
    target_columns = ["A", "B"]
    df = pd.DataFrame([[1, 2], [3, 4]], index=target_index, columns=target_columns)
    dims = []
    da = as_dataarray(df, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    assert list(da.coords[target_dims[0]].values) == target_index
    assert list(da.coords[target_dims[1]].values) == target_columns


def test_as_dataarray_dataframe_dims_superset():
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


def test_as_dataarray_dataframe_override_coords():
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


def test_as_dataarray_dataframe_aligned_coords():
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


def test_as_dataarray_with_ndarray_no_coords_no_dims():
    target_dims = ("dim_0", "dim_1")
    target_coords = [[0, 1], [0, 1]]
    arr = np.array([[1, 2], [3, 4]])
    da = as_dataarray(arr)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    for i, dim in enumerate(target_dims):
        assert list(da.coords[dim]) == target_coords[i]


def test_as_dataarray_with_ndarray_coords_list_no_dims():
    target_dims = ("dim_0", "dim_1")
    target_coords = [["a", "b"], ["A", "B"]]
    arr = np.array([[1, 2], [3, 4]])
    da = as_dataarray(arr, coords=target_coords)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    for i, dim in enumerate(target_dims):
        assert list(da.coords[dim]) == target_coords[i]


def test_as_dataarray_with_ndarray_coords_indexes_no_dims():
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


def test_as_dataarray_with_ndarray_coords_dict_set_no_dims():
    """If no dims are given and coords are a dict, the keys of the dict should be used as dims."""
    target_dims = ("dim_0", "dim_2")
    target_coords = {"dim_0": ["a", "b"], "dim_2": ["A", "B"]}
    arr = np.array([[1, 2], [3, 4]])
    da = as_dataarray(arr, coords=target_coords)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    for dim in target_dims:
        assert list(da.coords[dim]) == target_coords[dim]


def test_as_dataarray_with_ndarray_coords_list_dims():
    target_dims = ("dim1", "dim2")
    target_coords = [["a", "b"], ["A", "B"]]
    arr = np.array([[1, 2], [3, 4]])
    da = as_dataarray(arr, coords=target_coords, dims=target_dims)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    for i, dim in enumerate(target_dims):
        assert list(da.coords[dim]) == target_coords[i]


def test_as_dataarray_with_ndarray_coords_list_dims_superset():
    target_dims = ("dim1", "dim2")
    target_coords = [["a", "b"], ["A", "B"]]
    arr = np.array([[1, 2], [3, 4]])
    dims = [*target_dims, "dim3"]
    da = as_dataarray(arr, coords=target_coords, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    for i, dim in enumerate(target_dims):
        assert list(da.coords[dim]) == target_coords[i]


def test_as_dataarray_with_ndarray_coords_list_dims_subset():
    target_dims = ("dim0", "dim_1")
    target_coords = [["a", "b"], ["A", "B"]]
    arr = np.array([[1, 2], [3, 4]])
    dims = ["dim0"]
    da = as_dataarray(arr, coords=target_coords, dims=dims)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    for i, dim in enumerate(target_dims):
        assert list(da.coords[dim]) == target_coords[i]


def test_as_dataarray_with_ndarray_coords_indexes_dims_aligned():
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


def test_as_dataarray_with_ndarray_coords_indexes_dims_not_aligned():
    target_dims = ("dim3", "dim4")
    target_coords = [
        pd.Index(["a", "b"], name="dim1"),
        pd.Index(["A", "B"], name="dim2"),
    ]
    arr = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        as_dataarray(arr, coords=target_coords, dims=target_dims)


def test_as_dataarray_with_ndarray_coords_dict_dims_aligned():
    target_dims = ("dim_0", "dim_1")
    target_coords = {"dim_0": ["a", "b"], "dim_1": ["A", "B"]}
    arr = np.array([[1, 2], [3, 4]])
    da = as_dataarray(arr, coords=target_coords, dims=target_dims)
    assert isinstance(da, DataArray)
    assert da.dims == target_dims
    for dim in target_dims:
        assert list(da.coords[dim]) == target_coords[dim]


def test_as_dataarray_with_ndarray_coords_dict_set_dims_not_aligned():
    target_dims = ("dim_0", "dim_1")
    target_coords = {"dim_0": ["a", "b"], "dim_2": ["A", "B"]}
    arr = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        as_dataarray(arr, coords=target_coords, dims=target_dims)


def test_as_dataarray_with_number():
    num = 1
    da = as_dataarray(num, dims=["dim1"], coords=[["a"]])
    assert isinstance(da, DataArray)
    assert da.dims == ("dim1",)
    assert list(da.coords["dim1"].values) == ["a"]


def test_as_dataarray_with_number_default_dims_coords():
    num = 1
    da = as_dataarray(num)
    assert isinstance(da, DataArray)
    assert da.dims == ()
    assert da.coords == {}


def test_as_dataarray_with_number_and_coords():
    num = 1
    da = as_dataarray(num, coords=[pd.RangeIndex(10, name="a")])
    assert isinstance(da, DataArray)
    assert da.dims == ("a",)
    assert list(da.coords["a"].values) == list(range(10))


def test_as_dataarray_with_dataarray():
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


def test_as_dataarray_with_dataarray_default_dims_coords():
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


def test_as_dataarray_with_unsupported_type():
    with pytest.raises(TypeError):
        as_dataarray(lambda x: 1, dims=["dim1"], coords=[["a"]])


def test_best_int():
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


def test_assign_multiindex_safe():
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
