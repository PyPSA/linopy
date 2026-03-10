import xarray as xr

import linopy


def test_operations_with_data_arrays_are_typed_correctly(convention: str) -> None:
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


def test_constant_with_extra_dims_broadcasts() -> None:
    """Only valid under v1 convention (legacy uses outer join which also works)."""
    m = linopy.Model()

    a: xr.DataArray = xr.DataArray([1, 2, 3])

    v: linopy.Variable = m.add_variables(lower=0.0, name="v")
    e: linopy.LinearExpression = v * 1.0
    q = v * v

    # Constants can introduce new dimensions (broadcasting)
    result_v = a * v
    assert "dim_0" in result_v.dims

    result_e = a * e
    assert "dim_0" in result_e.dims

    # QuadraticExpression also allows constant broadcasting
    result_q = a * q
    assert isinstance(result_q, linopy.expressions.QuadraticExpression)
