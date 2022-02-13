#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 17:40:33 2021.

@author: fabian
"""

from numpy import arange

from linopy import Model

SOLVER = snakemake.wildcards.solver
N = int(snakemake.wildcards.N)

m = Model()
coords = [arange(N), arange(N)]
x = m.add_variables(coords=coords)
y = m.add_variables(coords=coords)
m.add_constraints(x - y >= arange(N))
m.add_constraints(x + y >= 0)
m.add_objective((2 * x + y).sum())
m.solve(SOLVER)
