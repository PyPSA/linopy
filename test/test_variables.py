#!/usr/bin/env python3
"""
This module aims at testing the correct behavior of the Variables class.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xarray.core.indexes
import xarray.core.utils

import linopy
from linopy import Model
from linopy.testing import assert_varequal
from linopy.variables import ScalarVariable


@pytest.fixture
def m() -> Model:
    m = Model()
    m.add_variables(coords=[pd.RangeIndex(10, name="first")], name="x")
    m.add_variables(coords=[pd.Index([1, 2, 3], name="second")], name="y")
    m.add_variables(0, 10, name="z")
    return m


def test_variables_repr(m: Model) -> None:
    m.variables.__repr__()


def test_variables_inherited_properties(m: Model) -> None:
    assert isinstance(m.variables.attrs, dict)
    assert isinstance(m.variables.coords, xr.Coordinates)
    assert isinstance(m.variables.indexes, xarray.core.indexes.Indexes)
    assert isinstance(m.variables.sizes, xarray.core.utils.Frozen)


def test_variables_getattr_formatted() -> None:
    m = Model()
    m.add_variables(name="y-0")
    assert_varequal(m.variables.y_0, m.variables["y-0"])


def test_variables_assignment_with_merge() -> None:
    """
    Test the merger of a variables with same dimension name but with different
    lengths.

    New coordinates are aligned to the existing ones. Thus this should
    raise a warning.
    """
    m = Model()

    upper = pd.Series(np.ones(10))
    var0 = m.add_variables(upper)

    upper = pd.Series(np.ones(12))
    var1 = m.add_variables(upper)

    with pytest.warns(UserWarning):
        assert m.variables.labels.var0[-1].item() == -1

    assert_varequal(var0, m.variables.var0)
    assert_varequal(var1, m.variables.var1)


def test_variables_assignment_with_reindex(m: Model) -> None:
    shuffled_coords = [pd.Index([2, 1, 3, 4, 6, 5, 7, 9, 8, 0], name="first")]
    m.add_variables(coords=shuffled_coords, name="a")

    with pytest.warns(UserWarning):
        m.variables.labels

        for dtype in m.variables.labels.dtypes.values():
            assert np.issubdtype(dtype, np.integer)

        for dtype in m.variables.lower.dtypes.values():
            assert np.issubdtype(dtype, np.floating)

        for dtype in m.variables.upper.dtypes.values():
            assert np.issubdtype(dtype, np.floating)


def test_scalar_variables_name_counter() -> None:
    m = Model()
    m.add_variables()
    m.add_variables()
    assert "var0" in m.variables
    assert "var1" in m.variables


def test_variables_binaries(m: Model) -> None:
    assert isinstance(m.binaries, linopy.variables.Variables)


def test_variables_integers(m: Model) -> None:
    assert isinstance(m.integers, linopy.variables.Variables)


def test_variables_nvars(m: Model) -> None:
    assert m.variables.nvars == 14

    idx = pd.RangeIndex(10, name="first")
    mask = pd.Series([True] * 5 + [False] * 5, idx)
    m.add_variables(coords=[idx], mask=mask)
    assert m.variables.nvars == 19


def test_variables_get_name_by_label(m: Model) -> None:
    assert m.variables.get_name_by_label(4) == "x"
    assert m.variables.get_name_by_label(12) == "y"

    with pytest.raises(ValueError):
        m.variables.get_name_by_label(30)

    with pytest.raises(ValueError):
        m.variables.get_name_by_label("anystring")  # type: ignore


def test_scalar_variable(m: Model) -> None:
    x = ScalarVariable(label=0, model=m)
    assert isinstance(x, ScalarVariable)
    assert x.__rmul__(x) is NotImplemented  # type: ignore
