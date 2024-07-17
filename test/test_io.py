#!/usr/bin/env python3
"""
Created on Thu Mar 18 09:03:35 2021.

@author: fabian
"""

import pandas as pd
import pytest
import xarray as xr

from linopy import LESS_EQUAL, Model, available_solvers, read_netcdf
from linopy.testing import assert_model_equal


@pytest.fixture
def model():
    m = Model()

    x = m.add_variables(4, pd.Series([8, 10]), name="x")
    y = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4]]), name="y")

    m.add_constraints(x + y, LESS_EQUAL, 10)

    m.add_objective(2 * x + 3 * y)

    m.parameters["param"] = xr.DataArray([1, 2, 3, 4], dims=["x"])

    return m


@pytest.fixture
def model_with_dash_names():
    m = Model()

    x = m.add_variables(4, pd.Series([8, 10]), name="x-var")
    x = m.add_variables(4, pd.Series([8, 10]), name="x-var-2")
    y = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4]]), name="y-var")

    m.add_constraints(x + y, LESS_EQUAL, 10, name="constraint-1")

    m.add_objective(2 * x + 3 * y)

    return m


@pytest.fixture
def model_with_multiindex():
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


def test_model_to_netcdf(model, tmp_path):
    m = model
    fn = tmp_path / "test.nc"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    assert_model_equal(m, p)


def test_model_to_netcdf_with_sense(model, tmp_path):
    m = model
    m.objective.sense = "max"
    fn = tmp_path / "test.nc"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    assert_model_equal(m, p)


def test_model_to_netcdf_with_dash_names(model_with_dash_names, tmp_path):
    m = model_with_dash_names
    fn = tmp_path / "test.nc"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    assert_model_equal(m, p)


def test_model_to_netcdf_with_status_and_condition(model_with_dash_names, tmp_path):
    m = model_with_dash_names
    fn = tmp_path / "test.nc"
    m._status = "ok"
    m._termination_condition = "optimal"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    assert_model_equal(m, p)


# skip it xarray version is 2024.01.0 due to issue https://github.com/pydata/xarray/issues/8628
@pytest.mark.skipif(
    xr.__version__ in ["2024.1.0", "2024.1.1"],
    reason="xarray version 2024.1.0 has a bug with MultiIndex deserialize",
)
def test_model_to_netcdf_with_multiindex(model_with_multiindex, tmp_path):
    m = model_with_multiindex
    fn = tmp_path / "test.nc"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    assert_model_equal(m, p)


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_file_lp(model, tmp_path):
    import gurobipy

    fn = tmp_path / "test.lp"
    model.to_file(fn)

    gurobipy.read(str(fn))


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_file_lp_None(model):
    import gurobipy

    fn = None
    model.to_file(fn)

    fn = model.get_problem_file()
    gurobipy.read(str(fn))


@pytest.mark.skipif(
    not {"gurobi", "highs"}.issubset(available_solvers),
    reason="Gurobipy of highspy not installed",
)
def test_to_file_mps(model, tmp_path):
    import gurobipy

    fn = tmp_path / "test.mps"
    model.to_file(fn)

    gurobipy.read(str(fn))


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_file_invalid(model, tmp_path):
    with pytest.raises(ValueError):
        fn = tmp_path / "test.failedtype"
        model.to_file(fn)


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_gurobipy(model):
    model.to_gurobipy()


@pytest.mark.skipif("highs" not in available_solvers, reason="Highspy not installed")
def test_to_highspy(model):
    model.to_highspy()


def test_to_blocks(tmp_path):
    # This is currently broken and due to time-constraints not possible to fix
    m = Model()

    lower = pd.Series(range(20))
    upper = pd.Series(range(30, 50))
    x = m.add_variables(lower, upper)
    y = m.add_variables(lower, upper)

    m.add_constraints(x + y, LESS_EQUAL, 10)

    m.add_objective(2 * x + 3 * y)

    m.blocks = xr.DataArray([1] * 10 + [2] * 10)

    with pytest.raises(NotImplementedError):
        m.to_block_files(tmp_path)
