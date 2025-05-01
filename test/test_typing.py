import xarray as xr
from mypy import api

import linopy


def test_operations_with_data_arrays_are_typed_correctly() -> None:
    m = linopy.Model()

    a: xr.DataArray = xr.DataArray([1, 2, 3])

    v: linopy.Variable = m.add_variables(lower=0.0, name="v")
    e: linopy.LinearExpression = v * 1.0
    q = v * v
    assert isinstance(q, linopy.QuadraticExpression)

    _ = a * v
    _ = v * a
    _ = v + a

    _ = a * e
    _ = e * a
    _ = e + a

    _ = a * q
    _ = q * a
    _ = q + a

    # Get the path of this file
    file_path = __file__
    result = api.run([file_path])
    assert result[2] == 0, "Mypy returned issues: " + result[0]
