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


def model(n, solver, integerlabels):
    m = Model()
    if integerlabels:
        N, M = [arange(n), arange(n)]
    else:
        N, M = [arange(n).astype(float), arange(n).astype(str)]
    x = m.add_variables(coords=[N, M])
    y = m.add_variables(coords=[N, M])
    m.add_constraints(x - y >= N)
    m.add_constraints(x + y >= 0)
    m.add_objective((2 * x).sum() + y.sum())
    m.solve(solver)
    return


if __name__ == "__main__":
    solver = snakemake.wildcards.solver
    integerlabels = snakemake.params.integerlabels

    # dry run first
    model(2, solver, integerlabels)

    res = profile(snakemake.params.nrange, model, solver, integerlabels)
    res["API"] = "linopy"
    res = res.rename_axis("N").reset_index()

    res.to_csv(snakemake.output[0])
