#!/usr/bin/env python3
"""
Created on Mon Feb 28 15:47:32 2022.

@author: fabian
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model

NAN_BOUND = "NaN found in a user-supplied variable bound"
COORDS = [pd.RangeIndex(3, name="i")]


@pytest.mark.parametrize("bound", ["lower", "upper"])
@pytest.mark.parametrize(
    "value", [np.nan, xr.DataArray([0.0, np.nan, 2.0], coords=COORDS)]
)
def test_nan_in_variable_bound_raises(bound: str, value: float | xr.DataArray) -> None:
    m = Model()
    with pytest.raises(ValueError, match=NAN_BOUND):
        m.add_variables(coords=COORDS, name="x", **{bound: value})
    assert "x" not in m.variables


@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("bound", ["lower", "upper"])
def test_nan_in_variable_bound_under_mask(bound: str, binary: bool) -> None:
    m = Model()
    value = xr.DataArray([0.0, np.nan, 1.0], coords=COORDS)
    mask = xr.DataArray([True, False, True], coords=COORDS)
    x = m.add_variables(
        coords=COORDS, name="x", mask=mask, binary=binary, **{bound: value}
    )
    assert x.mask.values.tolist() == [True, False, True]


@pytest.mark.parametrize("bound", ["lower", "upper"])
def test_nan_in_variable_bound_update_raises(bound: str) -> None:
    m = Model()
    mask = xr.DataArray([True, False, True], coords=COORDS)
    x = m.add_variables(0, 5, coords=COORDS, name="x", mask=mask)
    x.update(**{bound: xr.DataArray([1.0, np.nan, 2.0], coords=COORDS)})
    with pytest.raises(ValueError, match=NAN_BOUND):
        x.update(**{bound: xr.DataArray([np.nan, 1.0, 2.0], coords=COORDS)})


def test_nan_in_constraint_sign() -> None:
    m: Model = Model()

    x = m.add_variables(name="x")
    y = m.add_variables(name="y")

    with pytest.raises(ValueError):
        m.add_constraints(2 * x + 6 * y, np.nan, 10)


# TODO: this requires a strict propagation of the NaN values across expressions
# def test_nan_in_constraint_rhs():
#     m = Model()

#     x = m.add_variables(name="x")
#     y = m.add_variables(name="y")

#     m.add_constraints(2 * x + 6 * y, GREATER_EQUAL, np.nan)
#     m.add_constraints(4 * x + 2 * y, GREATER_EQUAL, 3)

#     m.add_objective(2 * y + x)

# with pytest.raises(ValueError):
#     m.solve()


# TODO: this requires a strict propagation of the NaN values across expressions
# def test_nan_in_objective():
#     m = Model()

#     x = m.add_variables(name="x")
#     y = m.add_variables(name="y")

#     m.add_constraints(2 * x + 6 * y, GREATER_EQUAL, np.nan)
#     m.add_constraints(4 * x + 2 * y, GREATER_EQUAL, 3)

#     m.add_objective(np.nan * y + x)

# with pytest.raises(ValueError):
#     m.solve()
