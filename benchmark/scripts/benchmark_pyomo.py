#!/usr/bin/env python3
"""
Created on Fri Nov 19 17:40:33 2021.

@author: fabian
"""

from common import profile
from numpy import arange
from numpy.random import default_rng
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    Objective,
    Set,
    Var,
    maximize,
)
from pyomo.opt import SolverFactory

# Random seed for reproducibility
rng = default_rng(125)


def basic_model(n, solver):
    m = ConcreteModel()
    m.i = Set(initialize=arange(n))
    m.j = Set(initialize=arange(n))

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
    return m.obj()


def knapsack_model(n, solver):
    m = ConcreteModel()
    m.i = Set(initialize=arange(n))

    m.x = Var(m.i, domain=Binary)
    m.weight = rng.integers(1, 100, size=n)
    m.value = rng.integers(1, 100, size=n)

    def bound1(m):
        return sum(m.x[i] * m.weight[i] for i in m.i) <= 200

    def objective(m):
        return sum(m.x[i] * m.value[i] for i in m.i)

    m.con1 = Constraint(rule=bound1)
    m.obj = Objective(rule=objective, sense=maximize)

    opt = SolverFactory(solver)
    opt.solve(m)
    return m.obj()


if __name__ == "__main__":
    solver = snakemake.config["solver"]

    if snakemake.config["benchmark"] == "basic":
        model = basic_model
    elif snakemake.config["benchmark"] == "knapsack":
        model = knapsack_model

    # dry run first
    model(2, solver)

    res = profile(snakemake.params.nrange, model, solver)
    res["API"] = "pyomo"
    res = res.rename_axis("N").reset_index()

    res.to_csv(snakemake.output[0])
