#!/usr/bin/env python3
"""
This module aims at testing the correct assignment of variable to the model.
"""

import dask
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model

# Test model functions

target_shape = (10, 10)


def test_variable_assignment_default():
    m = Model()
    m.add_variables(name="x")
    assert m.variables.lower.x.item() == -np.inf
    assert m.variables.upper.x.item() == np.inf


def test_variable_assignment_with_scalars():
    m = Model()
    m.add_variables(-5, 10, name="x")
    assert "x" in m.variables.labels


def test_variable_assignment():
    m = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    assert x.shape == target_shape


def test_variable_assignment_broadcasted():
    m = Model()
    # setting only one dimension, the other has to be broadcasted
    lower = xr.DataArray(np.zeros(10), coords=[range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    assert x.shape == target_shape


def test_variable_assignment_no_coords():
    # setting bounds without explicit coords
    m = Model()
    lower = xr.DataArray(np.zeros(10))
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    assert x.shape == target_shape


def test_variable_assignment_pd_index():
    # setting bounds with pandas index
    m = Model()
    lower = xr.DataArray(np.zeros(10), coords=[pd.Index(range(10))])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    assert x.shape == target_shape


def test_variable_assignment_by_coords():
    # setting bounds with scalar and coords
    m = Model()
    lower = 0
    upper = 1
    coords = [pd.Index(range(10)), pd.Index(range(10))]
    x = m.add_variables(lower, upper, coords=coords, name="x")
    assert x.shape == target_shape


def test_variable_assignment_with_dataframes():
    # setting bounds with pd.DataFrames
    m = Model()
    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.DataFrame(np.ones((10, 10)))
    x = m.add_variables(lower, upper, name="x")
    assert x.shape == target_shape


def test_variable_assignment_with_dataframe_and_series():
    # setting bounds with one pd.DataFrame and one pd.Series
    m = Model()
    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.Series(np.ones(10))
    x = m.add_variables(lower, upper, name="x")
    assert x.shape == target_shape


def test_variable_assignment_chunked():
    # setting bounds with one pd.DataFrame and one pd.Series
    m = Model(chunk=5)
    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.Series(np.ones(10))
    x = m.add_variables(lower, upper, name="x")
    assert x.shape == target_shape
    assert isinstance(m.variables.labels.x.data, dask.array.core.Array)


def test_variable_assignment_different_shapes():
    # setting bounds with different shapes
    m = Model()
    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.Series(np.ones(10))
    x = m.add_variables(lower, upper, name="x")
    assert x.shape == target_shape


def test_variable_assignment_without_coords():
    # setting bounds without explicit coords
    m = Model()
    lower = np.zeros((10, 10))
    upper = np.ones(10)
    x = m.add_variables(lower, upper, name="x")
    assert x.shape == target_shape


def test_variable_assignment_without_coords_and_dims_names():
    # setting bounds without explicit coords
    m = Model()
    lower = np.zeros((10, 10))
    upper = np.ones((10, 10))
    x = m.add_variables(lower, upper, name="x", dims=["i", "j"])
    assert x.shape == target_shape
    assert x.dims == ("i", "j")


def test_variable_assignment_without_coords_in_bounds():
    # setting bounds without explicit coords
    m = Model()
    lower = xr.DataArray(np.zeros((10, 10)), dims=["i", "j"])
    upper = xr.DataArray(np.ones((10, 10)), dims=["i", "j"])
    x = m.add_variables(lower, upper, name="x")
    assert x.shape == target_shape
    assert x.dims == ("i", "j")


def test_variable_assignment_without_coords_pandas_types():
    # setting bounds without explicit coords
    m = Model()
    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.DataFrame(np.ones((10, 10)))
    x = m.add_variables(lower, upper, name="x", dims=["i", "j"])
    assert x.shape == target_shape
    assert x.dims == ("i", "j")


def test_variable_assignment_without_coords_mixed_types():
    # setting bounds without explicit coords
    m = Model()
    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = 1
    x = m.add_variables(lower, upper, name="x")
    assert x.shape == target_shape

    m = Model()
    lower = xr.DataArray(np.zeros((10, 10)), dims=["i", "j"])
    upper = 1
    x = m.add_variables(lower, upper, name="x")
    assert x.shape == target_shape
    assert x.dims == ("i", "j")


def test_variable_assignment_different_coords():
    # set a variable with different set of coordinates
    # since v0.1 new coordinates are reindexed to the old ones
    m = Model()
    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.Series(np.ones(10))
    m.add_variables(lower, upper, name="x")

    lower = pd.DataFrame(np.zeros((20, 10)))
    upper = pd.Series(np.ones(20))
    m.add_variables(lower, upper, name="y")

    with pytest.warns(UserWarning):
        assert m.variables.labels.y.shape == (20, 10)
        # x should now be aligned to new coords and contain 100 nans
        assert m.variables.labels.x.shape == (20, 10)
        assert (m.variables.labels.x != -1).sum() == 100


def test_variable_assignment_with_broadcast():
    # setting with scalar and list
    m = Model()
    m.add_variables(lower=0, upper=[1, 2])

    with pytest.raises(ValueError):
        m.add_variables(lower=0, upper=[1, 2], dims=["i"])


def test_variable_assignment_repeated():
    # repeated variable assignment is forbidden
    m = Model()
    m.add_variables(name="x")
    with pytest.raises(ValueError):
        m.add_variables(name="x")


def test_variable_assigment_masked():
    m = Model()

    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.Series(np.ones(10))
    mask = pd.Series([True] * 5 + [False] * 5)
    m.add_variables(lower, upper, mask=mask)
    assert m.variables.labels.var0[-1, -1].item() == -1


def test_variable_assignment_binary():
    m = Model()

    coords = [pd.Index(range(10)), pd.Index(range(10))]
    m.add_variables(coords=coords, binary=True)

    assert m.variables.labels.var0.shape == target_shape


def test_variable_assignment_binary_with_error():
    m = Model()

    coords = [pd.Index(range(10)), pd.Index(range(10))]
    with pytest.raises(ValueError):
        m.add_variables(lower=-2, coords=coords, binary=True)


def test_variable_assignment_integer():
    m = Model()

    coords = [pd.Index(range(10)), pd.Index(range(10))]
    m.add_variables(coords=coords, integer=True)

    assert m.variables.labels.var0.shape == target_shape


def test_variable_assignment_binary_and_integer_invalid():
    m = Model()

    coords = [pd.Index(range(10)), pd.Index(range(10))]
    with pytest.raises(ValueError):
        m.add_variables(coords=coords, binary=True, integer=True)
