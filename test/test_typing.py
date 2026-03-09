import pytest
import xarray as xr

import linopy


def test_operations_with_data_arrays_are_typed_correctly() -> None:
    m = linopy.Model()

    s: xr.DataArray = xr.DataArray(5.0)

    v: linopy.Variable = m.add_variables(lower=0.0, name="v")
    e: linopy.LinearExpression = v * 1.0
    q = v * v

    _ = s * v
    _ = v * s
    _ = v + s

    _ = s * e
    _ = e * s
    _ = e + s

    _ = s * q
    _ = q * s
    _ = q + s


def test_constant_with_extra_dims_raises() -> None:
    m = linopy.Model()

    a: xr.DataArray = xr.DataArray([1, 2, 3])

    v: linopy.Variable = m.add_variables(lower=0.0, name="v")
    e: linopy.LinearExpression = v * 1.0
    q = v * v

    with pytest.raises(ValueError, match="not present"):
        a * v
    with pytest.raises(ValueError, match="not present"):
        a * e
    with pytest.raises(ValueError, match="not present"):
        a * q
