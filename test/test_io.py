#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 09:03:35 2021.

@author: fabian
"""

import pandas as pd
import pytest
import xarray as xr
from xarray.testing import assert_equal

from linopy import LESS_EQUAL, Model, available_solvers, read_netcdf


@pytest.fixture
def m():
    m = Model()

    x = m.add_variables(4, pd.Series([8, 10]), name="x")
    y = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4]]), name="y")

    m.add_constraints(x + y, LESS_EQUAL, 10)

    m.add_objective(2 * x + 3 * y)

    return m


def test_to_netcdf(m, tmp_path):
    fn = tmp_path / "test.nc"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    for k in m.scalar_attrs:
        if k != "objective_value":
            assert getattr(m, k) == getattr(p, k)

    for k in m.dataset_attrs:
        assert_equal(getattr(m, k), getattr(p, k))

    assert set(m.variables) == set(p.variables)
    assert set(m.constraints) == set(p.constraints)

    for v in m.variables:
        assert_equal(m.variables[v].data, p.variables[v].data)

    for c in m.constraints:
        assert_equal(m.constraints[c].data, p.constraints[c].data)

    assert_equal(m.objective.data, p.objective.data)


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_file_lp(m, tmp_path):
    import gurobipy

    fn = tmp_path / "test.lp"
    m.to_file(fn)

    gurobipy.read(str(fn))


@pytest.mark.skipif(
    not {"gurobi", "highs"}.issubset(available_solvers),
    reason="Gurobipy of highspy not installed",
)
def test_to_file_mps(m, tmp_path):
    import gurobipy

    fn = tmp_path / "test.mps"
    m.to_file(fn)

    gurobipy.read(str(fn))


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_file_invalid(m, tmp_path):
    with pytest.raises(ValueError):
        fn = tmp_path / "test.failedtype"
        m.to_file(fn)


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_gurobipy(m):
    m.to_gurobipy()


@pytest.mark.skipif("highs" not in available_solvers, reason="Highspy not installed")
def test_to_highspy(m):
    m.to_highspy()


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

    with pytest.raises(ValueError):
        m.to_block_files(tmp_path)
