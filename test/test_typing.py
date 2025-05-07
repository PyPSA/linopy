import xarray as xr

import linopy


def test_operations_with_data_arrays_are_typed_correctly() -> None:
    m = linopy.Model()

    a: xr.DataArray = xr.DataArray([1, 2, 3])

    v: linopy.Variable = m.add_variables(lower=0.0, name="v")
    e: linopy.LinearExpression = v * 1.0
    q = v * v

    _ = a * v
    _ = v * a
    _ = v + a

    _ = a * e
    _ = e * a
    _ = e + a

    _ = a * q
    _ = q * a
    _ = q + a
