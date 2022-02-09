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

SOLVER = "cbc"
N = 40


def model(N):
    m = Model()
    c = arange(N)
    x = m.add_variables(coords=[c, c])
    y = m.add_variables(coords=[c, c])
    m.add_constraints(x - y >= sin(c))
    m.add_constraints(x + y >= 0)
    m.add_objective((2 * x).sum() + y.sum())
    m.solve(SOLVER)
    return m


res = profile([N], model)
