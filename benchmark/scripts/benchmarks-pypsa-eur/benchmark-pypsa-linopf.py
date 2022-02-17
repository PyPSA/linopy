#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 16:20:51 2022.

@author: fabian
"""

import numpy as np
import pypsa
from memory_profiler import profile

PATH = "/home/fabian/vres/py/pypsa-eur/networks/elec_s_37.nc"
SOLVER_PARAMS = {
    "crossover": 0,
    "method": 2,
    "BarConvTol": 1.0e-3,
    "FeasibilityTol": 1.0e-3,
}


@profile
def solve():
    n = pypsa.Network(PATH)
    n.generators.p_nom_max.fillna(np.inf, inplace=True)

    m = n.lopf(solver_options=SOLVER_PARAMS, pyomo=False, solver_name="gurobi")


solve()
