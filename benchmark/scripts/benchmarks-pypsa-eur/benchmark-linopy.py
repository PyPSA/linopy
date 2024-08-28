#!/usr/bin/env python3
"""
Created on Tue Feb 15 16:20:51 2022.

@author: fabian
"""

import numpy as np
import pypsa
from common import NSNAPSHOTS, PATH, SOLVER, SOLVER_PARAMS
from memory_profiler import profile


@profile
def solve():
    n = pypsa.Network(PATH)
    n.generators.p_nom_max.fillna(np.inf, inplace=True)
    n.snapshots = n.snapshots[:NSNAPSHOTS]

    n.optimize(solver_name=SOLVER, **SOLVER_PARAMS)


solve()
