import gurobipy as gp
import numpy as np
from common import profile


def model(n, solver):
    # Create a new model
    m = gp.Model()

    # Create variables
    x = m.addMVar((n, n), lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="x")
    y = m.addMVar((n, n), lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name="y")

    m.addConstr(x - y >= np.arange(n))
    m.addConstr(x + y >= 0)

    # Create objective
    obj = gp.quicksum(gp.quicksum(2 * x + y))
    m.setObjective(obj, sense=gp.GRB.MINIMIZE)

    # Optimize the model
    m.optimize()

    return m


if __name__ == "__main__":
    solver = snakemake.wildcards.solver

    # dry run first
    model(2, solver)

    res = profile(snakemake.params.nrange, model, solver)
    res["API"] = "gurobipy"
    res = res.rename_axis("N").reset_index()

    res.to_csv(snakemake.output[0])
