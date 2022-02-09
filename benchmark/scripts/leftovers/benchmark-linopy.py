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
    m = Model()
    coords = [arange(N), arange(N)]
    x = m.add_variables(coords=coords)
    y = m.add_variables(coords=coords)
    m.add_constraints(x - y >= arange(N))
    m.add_constraints(x + y >= 0)
    m.add_objective((2 * x).sum() + y.sum())
    m.solve(SOLVER)
    return


res = profile(snakemake.params.nrange, model)
res["API"] = "linopy"
res = res.rename_axis("N").reset_index()

res.to_csv(snakemake.output[0])
