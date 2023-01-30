import pulp
from common import profile
from numpy import arange


def model(n, solver):
    m = pulp.LpProblem("Model", pulp.LpMinimize)

    m.i = list(range(n))
    m.j = list(range(n))

    x = pulp.LpVariable.dicts("x", (m.i, m.j), lowBound=None, upBound=None)
    y = pulp.LpVariable.dicts("y", (m.i, m.j), lowBound=None, upBound=None)

    for i in m.i:
        for j in m.j:
            m += x[i][j] - y[i][j] >= i
            m += x[i][j] + y[i][j] >= 0
    m += pulp.lpSum(2 * x[i][j] + y[i][j] for i in m.i for j in m.j)

    solver = pulp.getSolver(solver.upper())
    m.solve(solver)
    return


if __name__ == "__main__":
    solver = snakemake.wildcards.solver

    # dry run first
    model(2, solver)

    res = profile(snakemake.params.nrange, model, solver)
    res["API"] = "pulp"
    res = res.rename_axis("N").reset_index()

    res.to_csv(snakemake.output[0])
