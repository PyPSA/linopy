#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 12:11:03 2023

@author: fabian
"""

import numpy as np
import pandas as pd
import pytest
from xarray import DataArray

from linopy.common import as_dataarray


def test_as_dataarray_with_series():
    s = pd.Series([1, 2, 3], index=["a", "b", "c"])
    da = as_dataarray(s, dims=["dim1"])
    assert isinstance(da, DataArray)
    assert da.dims == ("dim1",)
    assert list(da.coords["dim1"].values) == ["a", "b", "c"]


def test_as_dataarray_with_series_override_coords():
    s = pd.Series([1, 2, 3], index=["a", "b", "c"])
    da = as_dataarray(s, dims=["dim1"], coords=[[1, 2, 3]])
    assert isinstance(da, DataArray)
    assert da.dims == ("dim1",)
    assert list(da.coords["dim1"].values) == [1, 2, 3]


def test_as_dataarray_with_series_default_dims_coords():
    s = pd.Series([1, 2, 3])
    da = as_dataarray(s)
    assert isinstance(da, DataArray)
    assert da.dims == ("dim_0",)
    assert list(da.coords["dim_0"].values) == list(s.index)


def test_as_dataarray_with_dataframe():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["a", "b"])
    da = as_dataarray(df, dims=["dim1", "dim2"])
    assert isinstance(da, DataArray)
    assert da.dims == ("dim1", "dim2")
    assert list(da.coords["dim1"].values) == ["a", "b"]
    assert list(da.coords["dim2"].values) == ["A", "B"]


def test_as_dataarray_with_dataframe_override_coords():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["a", "b"])
    da = as_dataarray(df, dims=["dim1", "dim2"], coords=[[1, 2], [2, 3]])
    assert isinstance(da, DataArray)
    assert da.dims == ("dim1", "dim2")
    assert list(da.coords["dim1"].values) == [1, 2]
    assert list(da.coords["dim2"].values) == [2, 3]


def test_as_dataarray_with_dataframe_default_dims_coords():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    da = as_dataarray(df)
    assert isinstance(da, DataArray)
    assert da.dims == ("dim_0", "dim_1")
    assert list(da.coords["dim_0"].values) == [0, 1]
    assert list(da.coords["dim_1"].values) == ["A", "B"]


def test_as_dataarray_with_ndarray():
    arr = np.array([[1, 2], [3, 4]])
    da = as_dataarray(arr, dims=["dim1", "dim2"], coords=[["a", "b"], ["A", "B"]])
    assert isinstance(da, DataArray)
    assert da.dims == ("dim1", "dim2")
    assert list(da.coords["dim1"].values) == ["a", "b"]
    assert list(da.coords["dim2"].values) == ["A", "B"]


def test_as_dataarray_with_ndarray_more_dims_than_given():
    arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    da = as_dataarray(arr, dims=["dim0"], coords=[["a", "b"]])
    assert isinstance(da, DataArray)
    assert da.dims == ("dim0", "dim_1", "dim_2")
    assert list(da.coords["dim0"].values) == ["a", "b"]
    assert list(da.coords["dim_1"].values) == list(range(arr.shape[1]))
    assert list(da.coords["dim_2"].values) == list(range(arr.shape[2]))


def test_as_dataarray_with_ndarray_default_dims_coords():
    arr = np.array([[1, 2], [3, 4]])
    da = as_dataarray(arr)
    assert isinstance(da, DataArray)
    assert da.dims == ("dim_0", "dim_1")
    assert list(da.coords["dim_0"].values) == list(range(arr.shape[0]))
    assert list(da.coords["dim_1"].values) == list(range(arr.shape[1]))


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
