import gurobipy as gp
import numpy as np
from common import profile
from numpy.random import default_rng

# Random seed for reproducibility
rng = default_rng(125)


def basic_model(n, solver):
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

    return m.ObjVal


def knapsack_model(n, solver):
    # Create a new model
    m = gp.Model()

    weight = rng.integers(1, 100, size=n)
    value = rng.integers(1, 100, size=n)

    # Create variables
    x = m.addMVar(n, vtype=gp.GRB.BINARY, name="x")

    # Create constraints
    m.addConstr(weight @ x <= 200)

    # Create objective
    obj = value @ x
    m.setObjective(obj, sense=gp.GRB.MAXIMIZE)

    # Optimize the model
    m.optimize()

    return m.ObjVal


if __name__ == "__main__":
    solver = snakemake.config["solver"]

    if snakemake.config["benchmark"] == "basic":
        model = basic_model
    elif snakemake.config["benchmark"] == "knapsack":
        model = knapsack_model

    # dry run first
    model(2, None)

    res = profile(snakemake.params.nrange, model, solver)
    res["API"] = "gurobipy"
    res = res.rename_axis("N").reset_index()

    res.to_csv(snakemake.output[0])
