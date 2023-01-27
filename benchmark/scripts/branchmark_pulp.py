from common import profile
from numpy import arange
from pulp import *


def model(n, solver, integerlabels):
    m = LpProblem("Model", LpMinimize)

    if integerlabels:
        m.i = list(range(n))
        m.j = list(range(n))
    else:
        m.i = list(range(n))
        m.j = list(range(n))

    x = LpVariable.dicts("x", (m.i, m.j), lowBound=None, upBound=None)
    y = LpVariable.dicts("y", (m.i, m.j), lowBound=None, upBound=None)

    for i in m.i:
        for j in m.j:
            m += x[i][j] - y[i][j] >= i
            m += x[i][j] + y[i][j] >= 0
    m += lpSum(2 * x[i][j] + y[i][j] for i in m.i for j in m.j)

    m.solve(solver)
    return


if __name__ == "__main__":
    solver = snakemake.wildcards.solver
    integerlabels = snakemake.params.integerlabels

    # dry run first
    model(2, solver, integerlabels)

    res = profile(snakemake.params.nrange, model, solver, integerlabels)
    res["API"] = "pulp"
    res = res.rename_axis("N").reset_index()

    res.to_csv(snakemake.output[0])
