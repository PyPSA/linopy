#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 14:23:00 2022.

@author: fabian
"""

from linopy import Model

m = Model()
coords = [arange(N), arange(N)]
x = m.add_variables(coords=coords)
y = m.add_variables(coords=coords)
m.add_constraints(x - y >= arange(N))
m.add_constraints(x + y >= 0)
m.add_objective((2 * x).sum() + y.sum())
m.solve(SOLVER)
