import os

import gurobipy
import pandas as pd
from common import profile_shell
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


# def model(N):
#     fn = f'{snakemake.output.lp_files}/{N}.lp'
#     m = gurobipy.read(fn)
#     m.optimize()
#     return m


def model(N):
    return f"gurobi_cl {snakemake.output.lp_files}/{N}.lp"


os.mkdir(snakemake.output.lp_files)

for N in snakemake.params.nrange:
    fn = f"{snakemake.output.lp_files}/{N}.lp"
    print(fn)
    create_model(N).to_file(fn)

res = profile_shell(snakemake.params.nrange, model)
res["API"] = "Solver Process"
res = res.rename_axis("N").reset_index()

res.to_csv(snakemake.output.benchmark)
