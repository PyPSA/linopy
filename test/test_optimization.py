#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 08:49:08 2021

@author: fabian
"""

import numpy as np
import pandas as pd
import pytest

from linopy import Model
from linopy.solvers import available_solvers


@pytest.fixture
def model():
    m = Model(chunk=None)

    x = m.add_variables(name="x")
    y = m.add_variables(name="y")

    m.add_constraints(2 * x + 6 * y, ">=", 10)
    m.add_constraints(4 * x + 2 * y, ">=", 3)

    m.add_objective(2 * y + x)
    return m


@pytest.fixture
def milp_model():
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(coords=x.coords, name="y", binary=True)

    m.add_constraints(x + y, ">=", 10)

    m.add_objective(2 * x + y)
    return m


@pytest.fixture
def masked_variable_model():
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x")
    mask = [True] * 8 + [False, False]
    y = m.add_variables(lower, name="y", mask=mask)

    m.add_constraints(x + y, ">=", 10)

    m.add_objective(2 * x + y)
    return m


@pytest.fixture
def masked_constraint_model():
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(lower, name="y")

    mask = [True] * 8 + [False, False]
    m.add_constraints(x + y, ">=", 10, mask=mask)
    # for the last two entries only the following constraint will be active
    m.add_constraints(x + y, ">=", 5)

    m.add_objective(2 * x + y)
    return m


@pytest.mark.parametrize("solver", available_solvers)
def test_default_setting(model, solver):
    status, condition = model.solve(solver)
    assert status == "ok"
    assert np.isclose(model.objective_value, 3.3)


@pytest.mark.parametrize("solver", available_solvers)
def test_set_files(tmp_path, model, solver):
    status, condition = model.solve(
        solver,
        problem_fn=tmp_path / "problem.lp",
        solution_fn=tmp_path / "solution.sol",
        log_fn=tmp_path / "logging.log",
        keep_files=False,
    )
    assert status == "ok"


@pytest.mark.parametrize("solver", available_solvers)
def test_set_files_and_keep_files(tmp_path, model, solver):
    status, condition = model.solve(
        solver,
        problem_fn=tmp_path / "problem.lp",
        solution_fn=tmp_path / "solution.sol",
        log_fn=tmp_path / "logging.log",
        keep_files=True,
    )
    assert status == "ok"


@pytest.mark.parametrize("solver", available_solvers)
def test_infeasible_model(model, solver):
    model.add_constraints([(1, "x")], "<=", 0)
    model.add_constraints([(1, "y")], "<=", 0)

    status, condition = model.solve(solver)
    assert status == "warning"
    assert "infeasible" in condition


@pytest.mark.parametrize("solver", available_solvers)
def test_milp_model(milp_model, solver):
    status, condition = milp_model.solve(solver)
    assert condition == "optimal"
    assert ((milp_model.solution.y == 1) | (milp_model.solution.y == 0)).all()


@pytest.mark.parametrize("solver", available_solvers)
def test_masked_variable_model(masked_variable_model, solver):
    masked_variable_model.solve(solver)
    assert masked_variable_model.solution.y[-2:].isnull().all()
    assert masked_variable_model.solution.y[:-2].notnull().all()
    assert masked_variable_model.solution.x.notnull().all()
    assert (masked_variable_model.solution.x[-2:] == 10).all()


@pytest.mark.parametrize("solver", available_solvers)
def test_masked_constraint_model(masked_constraint_model, solver):
    masked_constraint_model.solve(solver)
    assert (masked_constraint_model.solution.y[:-2] == 10).all()
    assert (masked_constraint_model.solution.y[-2:] == 5).all()


@pytest.mark.parametrize("solver", available_solvers)
def test_basis_and_warmstart(tmp_path, model, solver):
    basis_fn = tmp_path / "basis.bas"
    model.solve(solver, basis_fn=basis_fn)
    model.solve(solver, warmstart_fn=basis_fn)


# def init_model_large():
#     m = Model()
#     time = pd.Index(range(10), name="time")

#     x = m.add_variables(name="x", lower=0, coords=[time])
#     y = m.add_variables(name="y", lower=0, coords=[time])
#     factor = pd.Series(time, index=time)

#     m.add_constraints(3 * x + 7 * y, ">=", 10 * factor, name="Constraint1")
#     m.add_constraints(5 * x + 2 * y, ">=", 3 * factor, name="Constraint2")

#     shifted = (1 * x).shift(time=1)
#     lhs = (x - shifted).sel(time=time[1:])
#     m.add_constraints(lhs, "<=", 0.2, "Limited growth")

#     m.add_objective((x + 2 * y).sum())
#     m.solve()
