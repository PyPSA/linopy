#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 08:49:08 2021.

@author: fabian
"""

import numpy as np
import pandas as pd
import pytest
from xarray.testing import assert_equal

from linopy import Model
from linopy.solvers import available_solvers

params = [(name, "lp") for name in available_solvers]
if "gurobi" in available_solvers:
    params.append(("gurobi", "direct"))
if "highs" in available_solvers:
    params.append(("highs", "direct"))


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
def model_anonymous_constraint():
    m = Model(chunk=None)

    x = m.add_variables(name="x")
    y = m.add_variables(name="y")

    m.add_constraints(2 * x + 6 * y >= 10)
    m.add_constraints(4 * x + 2 * y >= 3)

    m.add_objective(2 * y + x)
    return m


@pytest.fixture
def model_chunked():
    m = Model(chunk="auto")

    x = m.add_variables(name="x")
    y = m.add_variables(name="y")

    m.add_constraints(2 * x + 6 * y, ">=", 10)
    m.add_constraints(4 * x + 2 * y, ">=", 3)

    m.add_objective(2 * y + x)
    return m


@pytest.fixture
def model_with_inf():
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(coords=[lower.index], name="x", binary=True)
    y = m.add_variables(lower, name="y")

    m.add_constraints(x + y, ">=", 10)
    m.add_constraints(1 * x, "<=", np.inf)

    m.objective = 2 * x + y

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
def milp_model_r():
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(coords=[lower.index], name="x", binary=True)
    y = m.add_variables(lower, name="y")

    m.add_constraints(x + y, ">=", 10)

    m.add_objective(2 * x + y)
    return m


@pytest.fixture
def modified_model():
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(coords=[lower.index], name="x", binary=True)
    y = m.add_variables(lower, name="y")

    c = m.add_constraints(x + y, ">=", 10)

    y.lower = 9
    c.lhs = 2 * x + y
    m.objective = 2 * x + y

    return m


@pytest.fixture
def masked_variable_model():
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x")
    mask = pd.Series([True] * 8 + [False, False])
    y = m.add_variables(lower, name="y", mask=mask)

    m.add_constraints(x + y, ">=", 10)

    m.add_constraints(y, ">=", 0)

    m.add_objective(2 * x + y)
    return m


@pytest.fixture
def masked_constraint_model():
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(lower, name="y")

    mask = pd.Series([True] * 8 + [False, False])
    m.add_constraints(x + y, ">=", 10, mask=mask)
    # for the last two entries only the following constraint will be active
    m.add_constraints(x + y, ">=", 5)

    m.add_objective(2 * x + y)
    return m


@pytest.mark.parametrize("solver,io_api", params)
def test_default_setting(model, solver, io_api):
    status, condition = model.solve(solver, io_api=io_api)
    assert status == "ok"
    assert np.isclose(model.objective_value, 3.3)


@pytest.mark.parametrize("solver,io_api", params)
def test_default_setting_sol_and_dual_accessor(model, solver, io_api):
    status, condition = model.solve(solver, io_api=io_api)
    assert status == "ok"
    x = model["x"]
    assert_equal(x.sol, model.solution["x"])
    c = model.constraints["con1"]
    assert_equal(c.dual, model.dual["con1"])


@pytest.mark.parametrize("solver,io_api", params)
def test_anonymous_constraint(model, model_anonymous_constraint, solver, io_api):
    status, condition = model_anonymous_constraint.solve(solver, io_api=io_api)
    assert status == "ok"
    assert np.isclose(model_anonymous_constraint.objective_value, 3.3)

    model.solve(solver, io_api=io_api)
    assert_equal(model.solution, model_anonymous_constraint.solution)


@pytest.mark.parametrize("solver,io_api", params)
def test_default_settings_chunked(model_chunked, solver, io_api):
    status, condition = model_chunked.solve(solver, io_api=io_api)
    assert status == "ok"
    assert np.isclose(model_chunked.objective_value, 3.3)


@pytest.mark.parametrize("solver,io_api", params)
def test_set_files(tmp_path, model, solver, io_api):
    status, condition = model.solve(
        solver,
        io_api=io_api,
        problem_fn=tmp_path / "problem.lp",
        solution_fn=tmp_path / "solution.sol",
        log_fn=tmp_path / "logging.log",
        keep_files=False,
    )
    assert status == "ok"


@pytest.mark.parametrize("solver,io_api", params)
def test_set_files_and_keep_files(tmp_path, model, solver, io_api):
    status, condition = model.solve(
        solver,
        problem_fn=tmp_path / "problem.lp",
        solution_fn=tmp_path / "solution.sol",
        log_fn=tmp_path / "logging.log",
        keep_files=True,
    )
    assert status == "ok"


@pytest.mark.parametrize("solver,io_api", params)
def test_infeasible_model(model, solver, io_api):
    model.add_constraints([(1, "x")], "<=", 0)
    model.add_constraints([(1, "y")], "<=", 0)

    status, condition = model.solve(solver, io_api=io_api)
    assert status == "warning"
    assert "infeasible" in condition

    if solver == "gurobi":
        model.compute_set_of_infeasible_constraints()
    else:
        with pytest.raises(NotImplementedError):
            model.compute_set_of_infeasible_constraints()


@pytest.mark.parametrize(
    "solver,io_api", [p for p in params if p[0] not in ["glpk", "cplex"]]
)
def test_model_with_inf(model_with_inf, solver, io_api):
    status, condition = model_with_inf.solve(solver, io_api=io_api)
    assert condition == "optimal"
    assert (model_with_inf.solution.x == 0).all()
    assert (model_with_inf.solution.y == 10).all()


@pytest.mark.parametrize("solver,io_api", params)
def test_milp_model(milp_model, solver, io_api):
    status, condition = milp_model.solve(solver, io_api=io_api)
    assert condition == "optimal"
    assert ((milp_model.solution.y == 1) | (milp_model.solution.y == 0)).all()


@pytest.mark.parametrize("solver,io_api", params)
def test_milp_model_r(milp_model_r, solver, io_api):
    status, condition = milp_model_r.solve(solver, io_api=io_api)
    assert condition == "optimal"
    assert ((milp_model_r.solution.x == 1) | (milp_model_r.solution.x == 0)).all()


@pytest.mark.parametrize("solver,io_api", params)
def test_modified_model(modified_model, solver, io_api):
    status, condition = modified_model.solve(solver, io_api=io_api)
    assert condition == "optimal"
    assert (modified_model.solution.x == 0).all()
    assert (modified_model.solution.y == 10).all()


@pytest.mark.parametrize("solver,io_api", params)
def test_masked_variable_model(masked_variable_model, solver, io_api):
    masked_variable_model.solve(solver, io_api=io_api)
    assert masked_variable_model.solution.y[-2:].isnull().all()
    assert masked_variable_model.solution.y[:-2].notnull().all()
    assert masked_variable_model.solution.x.notnull().all()
    assert (masked_variable_model.solution.x[-2:] == 10).all()


@pytest.mark.parametrize("solver,io_api", params)
def test_masked_constraint_model(masked_constraint_model, solver, io_api):
    masked_constraint_model.solve(solver, io_api=io_api)
    assert (masked_constraint_model.solution.y[:-2] == 10).all()
    assert (masked_constraint_model.solution.y[-2:] == 5).all()


@pytest.mark.parametrize("solver,io_api", params)
def test_basis_and_warmstart(tmp_path, model, solver, io_api):
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
