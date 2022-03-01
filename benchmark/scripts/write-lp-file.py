import gurobipy
import pandas as pd
from numpy import arange

from linopy import Model


def create_model(N):
    m = Model()
    coords = [arange(N), arange(N)]
    x = m.add_variables(coords=coords)
    y = m.add_variables(coords=coords)
    m.add_constraints(x - y >= arange(N))
    m.add_constraints(x + y >= 0)
    m.add_objective((2 * x).sum() + y.sum())
    return m


for fn in snakemake.output:
    N = int(fn.split("/")[-1][:-3])
    create_model(N).to_file(fn)
