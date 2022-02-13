#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 17:40:33 2021.

@author: fabian
"""


from numpy import arange
from pyomo.environ import ConcreteModel, Constraint, Objective, Set, Var
from pyomo.opt import SolverFactory

SOLVER = snakemake.wildcards.solver
N = int(snakemake.wildcards.N)

m = ConcreteModel()
m.i = Set(initialize=arange(N))
m.j = Set(initialize=arange(N))
m.x = Var(m.i, m.j, bounds=(None, None))
m.y = Var(m.i, m.j, bounds=(None, None))


def bound1(m, i, j):
    return m.x[(i, j)] - m.y[(i, j)] >= i


def bound2(m, i, j):
    return m.x[(i, j)] + m.y[(i, j)] >= 0


def objective(m):
    return sum(2 * m.x[(i, j)] + m.y[(i, j)] for i in m.i for j in m.j)


m.con1 = Constraint(m.i, m.j, rule=bound1)
m.con2 = Constraint(m.i, m.j, rule=bound2)
m.obj = Objective(rule=objective)

opt = SolverFactory(SOLVER)
results = opt.solve(m)

# Write the output
results.write(num=1)
