#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 08:49:08 2021

@author: fabian
"""

import numpy as np
from linopy import Model
from linopy.solvers import available_solvers
import pytest
import pandas as pd

def init_model():
    m = Model(chunk=None)

    x = m.add_variables('x')
    y = m.add_variables('y')

    m.add_constraints('Constraint1', 2*x + 6*y, '>=', 10)
    m.add_constraints('Constraint2', 4*x + 2*y, '>=', 3)

    m.add_objective(2*y + x)
    return m


@pytest.mark.skipif('glpk' not in available_solvers, reason='Solver not available')
def test_glpk():
    m = init_model()
    m.solve('glpk')
    assert np.isclose(m.objective_value, 3.3)

@pytest.mark.skipif('cbc' not in available_solvers, reason='Solver not available')
def test_cbc():
    m = init_model()
    m.solve('cbc')
    assert np.isclose(m.objective_value, 3.3)

@pytest.mark.skipif('gurobi' not in available_solvers, reason='Solver not available')
def test_gurobi():
    m = init_model()
    m.solve('gurobi')
    assert np.isclose(m.objective_value, 3.3)

@pytest.mark.skipif('cplex' not in available_solvers, reason='Solver not available')
def test_cplex():
    m = init_model()
    m.solve('cplex')
    assert np.isclose(m.objective_value, 3.3)

@pytest.mark.skipif('xpress' not in available_solvers, reason='Solver not available')
def test_xpress():
    m = init_model()
    m.solve('xpress')
    assert np.isclose(m.objective_value, 3.3)


def init_model_large():
    m = Model()
    time = pd.Index(range(10), name='time')

    x = m.add_variables(name='x', lower=0, coords=[time])
    y = m.add_variables(name='y', lower=0, coords=[time])
    factor = pd.Series(time, index=time)

    m.add_constraints('Constraint1', 3*x + 7*y, '>=', 10*factor)
    m.add_constraints('Constraint2', 5*x + 2*y, '>=', 3*factor)

    shifted = (1*x).shift(time=1)
    lhs = (x - shifted).sel(time=time[1:])
    m.add_constraints('Limited growth', lhs, '<=', 0.2)

    m.add_objective((x + 2*y).sum())
    m.solve()