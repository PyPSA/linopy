#!/usr/bin/env python3
"""
Created on Thu Mar 18 09:03:35 2021.

@author: fabian
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from linopy import LESS_EQUAL, Model, available_solvers, read_netcdf
from linopy.io import signed_number
from linopy.testing import assert_model_equal


@pytest.fixture
def model() -> Model:
    m = Model()

    x = m.add_variables(4, pd.Series([8, 10]), name="x")
    y = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4]]), name="y")

    m.add_constraints(x + y, LESS_EQUAL, 10)

    m.add_objective(2 * x + 3 * y)

    m.parameters["param"] = xr.DataArray([1, 2, 3, 4], dims=["x"])

    return m


@pytest.fixture
def model_with_dash_names() -> Model:
    m = Model()

    x = m.add_variables(4, pd.Series([8, 10]), name="x-var")
    x = m.add_variables(4, pd.Series([8, 10]), name="x-var-2")
    y = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4]]), name="y-var")

    m.add_constraints(x + y, LESS_EQUAL, 10, name="constraint-1")

    m.add_objective(2 * x + 3 * y)

    return m


@pytest.fixture
def model_with_multiindex() -> Model:
    m = Model()

    index = pd.MultiIndex.from_tuples(
        [(1, "a"), (1, "b"), (2, "a"), (2, "b")], names=["first", "second"]
    )
    x = m.add_variables(4, pd.Series([8, 10, 12, 14], index=index), name="x-var")
    y = m.add_variables(
        0, pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], index=index), name="y-var"
    )

    m.add_constraints(x + y, LESS_EQUAL, 10, name="constraint-1")

    m.add_objective(2 * x + 3 * y)

    return m


def test_model_to_netcdf(model: Model, tmp_path: Path) -> None:
    m = model
    fn = tmp_path / "test.nc"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    assert_model_equal(m, p)


def test_model_to_netcdf_with_sense(model: Model, tmp_path: Path) -> None:
    m = model
    m.objective.sense = "max"
    fn = tmp_path / "test.nc"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    assert_model_equal(m, p)


def test_model_to_netcdf_with_dash_names(
    model_with_dash_names: Model, tmp_path: Path
) -> None:
    m = model_with_dash_names
    fn = tmp_path / "test.nc"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    assert_model_equal(m, p)


def test_model_to_netcdf_with_status_and_condition(
    model_with_dash_names: Model, tmp_path: Path
) -> None:
    m = model_with_dash_names
    fn = tmp_path / "test.nc"
    m._status = "ok"
    m._termination_condition = "optimal"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    assert_model_equal(m, p)


def test_pickle_model(model_with_dash_names: Model, tmp_path: Path) -> None:
    m = model_with_dash_names
    fn = tmp_path / "test.nc"
    m._status = "ok"
    m._termination_condition = "optimal"

    with open(fn, "wb") as f:
        pickle.dump(m, f)

    with open(fn, "rb") as f:
        p = pickle.load(f)

    assert_model_equal(m, p)


# skip it xarray version is 2024.01.0 due to issue https://github.com/pydata/xarray/issues/8628
@pytest.mark.skipif(
    xr.__version__ in ["2024.1.0", "2024.1.1"],
    reason="xarray version 2024.1.0 has a bug with MultiIndex deserialize",
)
def test_model_to_netcdf_with_multiindex(
    model_with_multiindex: Model, tmp_path: Path
) -> None:
    m = model_with_multiindex
    fn = tmp_path / "test.nc"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    assert_model_equal(m, p)


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_file_lp(model: Model, tmp_path: Path) -> None:
    import gurobipy

    fn = tmp_path / "test.lp"
    model.to_file(fn)

    gurobipy.read(str(fn))


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_file_lp_explicit_coordinate_names(model: Model, tmp_path: Path) -> None:
    import gurobipy

    fn = tmp_path / "test.lp"
    model.to_file(fn, io_api="lp", explicit_coordinate_names=True)

    gurobipy.read(str(fn))


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_file_lp_None(model: Model) -> None:
    import gurobipy

    fn: str | None = None
    model.to_file(fn)

    fn_path = model.get_problem_file()
    gurobipy.read(str(fn_path))


@pytest.mark.skipif(
    not {"gurobi", "highs"}.issubset(available_solvers),
    reason="Gurobipy of highspy not installed",
)
def test_to_file_mps(model: Model, tmp_path: Path) -> None:
    import gurobipy

    fn = tmp_path / "test.mps"
    model.to_file(fn)

    gurobipy.read(str(fn))


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_file_invalid(model: Model, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        fn = tmp_path / "test.failedtype"
        model.to_file(fn)


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_gurobipy(model: Model) -> None:
    model.to_gurobipy()


@pytest.mark.skipif("highs" not in available_solvers, reason="Highspy not installed")
def test_to_highspy(model: Model) -> None:
    model.to_highspy()


def test_to_blocks(tmp_path: Path) -> None:
    m: Model = Model()

    lower: pd.Series = pd.Series(range(20))
    upper: pd.Series = pd.Series(range(30, 50))
    x = m.add_variables(lower, upper)
    y = m.add_variables(lower, upper)

    m.add_constraints(x + y, LESS_EQUAL, 10)

    m.add_objective(2 * x + 3 * y)

    m.blocks = xr.DataArray([1] * 10 + [2] * 10)

    with pytest.raises(NotImplementedError):
        m.to_block_files(tmp_path)


class TestSignedNumberExpr:
    """Test the signed_number helper function for LP file formatting."""

    def test_positive_numbers(self) -> None:
        """Positive numbers should get a '+' prefix."""
        df = pl.DataFrame({"value": [1.0, 2.5, 100.0]})
        result = df.select(pl.concat_str(signed_number(pl.col("value"))))
        values = result.to_series().to_list()
        assert values == ["+1.0", "+2.5", "+100.0"]

    def test_negative_numbers(self) -> None:
        """Negative numbers should not get a '+' prefix (already have '-')."""
        df = pl.DataFrame({"value": [-1.0, -2.5, -100.0]})
        result = df.select(pl.concat_str(signed_number(pl.col("value"))))
        values = result.to_series().to_list()
        assert values == ["-1.0", "-2.5", "-100.0"]

    def test_positive_zero(self) -> None:
        """Positive zero should get a '+' prefix."""
        df = pl.DataFrame({"value": [0.0]})
        result = df.select(pl.concat_str(signed_number(pl.col("value"))))
        values = result.to_series().to_list()
        assert values == ["+0.0"]

    def test_negative_zero(self) -> None:
        """Negative zero is normalized to +0.0 - this is the bug fix."""
        # Create negative zero using numpy
        neg_zero = np.float64(-0.0)
        df = pl.DataFrame({"value": [neg_zero]})
        result = df.select(pl.concat_str(signed_number(pl.col("value"))))
        values = result.to_series().to_list()
        # The key assertion: should NOT be "+-0.0", -0.0 is normalized to +0.0
        assert values == ["+0.0"]
        assert "+-" not in values[0]

    def test_mixed_values_including_negative_zero(self) -> None:
        """Test a mix of positive, negative, and zero values."""
        neg_zero = np.float64(-0.0)
        df = pl.DataFrame({"value": [1.0, -1.0, 0.0, neg_zero, 2.5, -2.5]})
        result = df.select(pl.concat_str(signed_number(pl.col("value"))))
        values = result.to_series().to_list()
        # -0.0 is normalized to +0.0
        assert values == ["+1.0", "-1.0", "+0.0", "+0.0", "+2.5", "-2.5"]
        # No value should contain "+-"
        for v in values:
            assert "+-" not in v


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_file_lp_with_negative_zero_bounds(tmp_path: Path) -> None:
    """
    Test that LP files with negative zero bounds are valid.

    This is a regression test for the bug where -0.0 bounds would produce
    invalid LP file syntax like "+-0.0 <= x1 <= +0.0".

    See: https://github.com/PyPSA/linopy/issues/XXX
    """
    import gurobipy

    m = Model()

    # Create bounds that could produce -0.0
    # Using numpy to ensure we can create actual negative zeros
    lower = pd.Series([np.float64(-0.0), np.float64(0.0), np.float64(-0.0)])
    upper = pd.Series([np.float64(0.0), np.float64(-0.0), np.float64(1.0)])

    m.add_variables(lower, upper, name="x")
    m.add_objective(m.variables["x"].sum())

    fn = tmp_path / "test_neg_zero.lp"
    m.to_file(fn)

    # Read the LP file content and verify no "+-" appears
    with open(fn) as f:
        content = f.read()
    assert "+-" not in content, f"Found invalid '+-' in LP file: {content}"

    # Verify Gurobi can read it without errors
    gurobipy.read(str(fn))


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_file_lp_with_negative_zero_coefficients(tmp_path: Path) -> None:
    """
    Test that LP files with negative zero coefficients are valid.

    Coefficients can also potentially be -0.0 due to floating point arithmetic.
    """
    import gurobipy

    m = Model()

    x = m.add_variables(name="x", lower=0, upper=10)
    y = m.add_variables(name="y", lower=0, upper=10)

    # Create an expression where coefficients could become -0.0
    # through arithmetic operations
    coeff = np.float64(-0.0)
    expr = coeff * x + 1 * y

    m.add_constraints(expr <= 5)
    m.add_objective(x + y)

    fn = tmp_path / "test_neg_zero_coeffs.lp"
    m.to_file(fn)

    # Read the LP file content and verify no "+-" appears
    with open(fn) as f:
        content = f.read()
    assert "+-" not in content, f"Found invalid '+-' in LP file: {content}"

    # Verify Gurobi can read it without errors
    gurobipy.read(str(fn))
