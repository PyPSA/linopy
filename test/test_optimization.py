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


def init_model():
    m = Model(chunk=None)

    x = m.add_variables(name="x")
    y = m.add_variables(name="y")

    m.add_constraints(2 * x + 6 * y, ">=", 10)
    m.add_constraints(4 * x + 2 * y, ">=", 3)

    m.add_objective(2 * y + x)
    return m


@pytest.mark.skipif("glpk" not in available_solvers, reason="Solver not available")
def test_glpk():
    m = init_model()
    m.solve("glpk")
    assert np.isclose(m.objective_value, 3.3)


@pytest.mark.skipif("cbc" not in available_solvers, reason="Solver not available")
def test_cbc():
    m = init_model()
    m.solve("cbc")
    assert np.isclose(m.objective_value, 3.3)


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Solver not available")
def test_gurobi():
    m = init_model()
    m.solve("gurobi")
    assert np.isclose(m.objective_value, 3.3)


@pytest.mark.skipif("cplex" not in available_solvers, reason="Solver not available")
def test_cplex():
    m = init_model()
    m.solve("cplex")
    assert np.isclose(m.objective_value, 3.3)


@pytest.mark.skipif("xpress" not in available_solvers, reason="Solver not available")
def test_xpress():
    m = init_model()
    m.solve("xpress")
    assert np.isclose(m.objective_value, 3.3)


@pytest.mark.skipif("cplex" not in available_solvers, reason="Solver not available")
def test_masked_variables_with_cplex():
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x")
    mask = [True] * 8 + [False, False]
    y = m.add_variables(lower, name="y", mask=mask)

    m.add_constraints(x + y, ">=", 10)

    m.add_objective(2 * x + y)

    m.solve("cbc")
    assert m.solution.y[-2:].isnull().all()
    assert m.solution.y[:-2].notnull().all()
    assert m.solution.x.notnull().all()
    assert (m.solution.x[-2:] == 10).all()


@pytest.mark.skipif("cplex" not in available_solvers, reason="Solver not available")
def test_masked_constraints_with_cplex():
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(lower, name="y")

    mask = [True] * 8 + [False, False]
    m.add_constraints(x + y, ">=", 10, mask=mask)
    # for the last two entries only the following constraint will be active
    m.add_constraints(x + y, ">=", 5)

    m.add_objective(2 * x + y)

    m.solve("cbc")
    assert (m.solution.y[:-2] == 10).all()
    assert (m.solution.y[-2:] == 5).all()


def init_model_large():
    m = Model()
    time = pd.Index(range(10), name="time")

    x = m.add_variables(name="x", lower=0, coords=[time])
    y = m.add_variables(name="y", lower=0, coords=[time])
    factor = pd.Series(time, index=time)

    m.add_constraints(3 * x + 7 * y, ">=", 10 * factor, name="Constraint1")
    m.add_constraints(5 * x + 2 * y, ">=", 3 * factor, name="Constraint2")

    shifted = (1 * x).shift(time=1)
    lhs = (x - shifted).sel(time=time[1:])
    m.add_constraints(lhs, "<=", 0.2, "Limited growth")

    m.add_objective((x + 2 * y).sum())
    m.solve()
