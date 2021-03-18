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

def init_model():
    m = Model(chunk=None)

    m.add_variables('x')
    m.add_variables('y')

    lhs = (2, 'x'), (6, 'y')
    m.add_constraints('Constraint1', lhs, '>=', 10)

    lhs = (4, 'x'), (2, 'y')
    m.add_constraints('Constraint2', lhs, '>=', 3)

    obj = (2, 'y'), (1, 'x')
    m.add_objective(obj)
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
