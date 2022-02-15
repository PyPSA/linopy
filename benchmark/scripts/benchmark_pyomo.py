#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 17:40:33 2021.

@author: fabian
"""


from common import profile
from numpy import arange
from pyomo.environ import ConcreteModel, Constraint, Objective, Set, Var
from pyomo.opt import SolverFactory


def model(n, solver, integerlabels):
    m = ConcreteModel()
    if integerlabels:
        m.i = Set(initialize=arange(n))
        m.j = Set(initialize=arange(n))
    else:
        m.i = Set(initialize=arange(n).astype(float))
        m.j = Set(initialize=arange(n).astype(str))

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

    opt = SolverFactory(solver)
    opt.solve(m)
    return


if __name__ == "__main__":
    solver = snakemake.wildcards.solver
    integerlabels = snakemake.params.integerlabels

    # dry run first
    model(2, solver, integerlabels)

    res = profile(snakemake.params.nrange, model, solver, integerlabels)
    res["API"] = "pyomo"
    res = res.rename_axis("N").reset_index()

    res.to_csv(snakemake.output[0])
