#!/usr/bin/env python3
"""
Created on Fri Nov 19 17:40:33 2021.

@author: fabian
"""

from common import profile
from numpy import arange
from numpy.random import default_rng

from linopy import Model

# Random seed for reproducibility
rng = default_rng(125)


def basic_model(n, solver):
    m = Model()
    N, M = [arange(n), arange(n)]
    x = m.add_variables(coords=[N, M])
    y = m.add_variables(coords=[N, M])
    m.add_constraints(x - y >= N)
    m.add_constraints(x + y >= 0)
    m.add_objective(2 * x.sum() + y.sum())
    # m.to_file(f"linopy-model.lp")
    m.solve(solver)
    return m.objective.value


def knapsack_model(n, solver):
    m = Model()
    packages = m.add_variables(coords=[arange(n)], binary=True)
    weight = rng.integers(1, 100, size=n)
    value = rng.integers(1, 100, size=n)
    m.add_constraints((weight * packages).sum() <= 200)
    m.add_objective(-(value * packages).sum())  # use minus because of minimization
    m.solve(solver_name=solver)
    return -m.objective.value


if __name__ == "__main__":
    solver = snakemake.config["solver"]

    if snakemake.config["benchmark"] == "basic":
        model = basic_model
    elif snakemake.config["benchmark"] == "knapsack":
        model = knapsack_model

    # dry run first
    model(2, solver)

    res = profile(snakemake.params.nrange, model, solver)
    res["API"] = "linopy"
    res = res.rename_axis("N").reset_index()

    res.to_csv(snakemake.output[0])
