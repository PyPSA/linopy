#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 17:40:33 2021.

@author: fabian
"""


import pandas as pd
from common import profile
from numpy import arange, cos, sin

from linopy import Model

SOLVER = snakemake.wildcards.solver


def model(N):
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
    return


res = profile(snakemake.params.nrange, model)
res["API"] = "linopy"
res = res.rename_axis("N").reset_index()

res.to_csv(snakemake.output[0])
