import cvxpy as cp
import numpy as np
from common import profile
from numpy.random import default_rng

# Random seed for reproducibility
rng = default_rng(125)


def basic_model(n, solver):
    # Create variables
    x = cp.Variable((n, n))
    y = cp.Variable((n, n))

    constraints = [x - y >= np.repeat(np.arange(n)[:, np.newaxis], n, 1), x + y >= 0]

    # Create objective
    objective = cp.Minimize(2 * cp.sum(x) + cp.sum(y))

    # Optimize the model
    m = cp.Problem(objective, constraints)
    m.solve(solver=solver.upper())

    return m.value


def knapsack_model(n, solver):
    # Define the variables
    weight = rng.integers(1, 100, size=n)
    value = rng.integers(1, 100, size=n)

    x = cp.Variable(n, boolean=True)

    # Define the constraints
    constraints = [weight @ x <= 200]

    # Define the objective function
    objective = cp.Maximize(value @ x)

    # Optimize the model
    m = cp.Problem(objective, constraints)
    m.solve(solver=solver.upper())

    # return objective
    return m.value


if __name__ == "__main__":
    solver = snakemake.config["solver"]

    if snakemake.config["benchmark"] == "basic":
        model = basic_model
    elif snakemake.config["benchmark"] == "knapsack":
        model = knapsack_model

    # dry run first
    model(2, solver)

    res = profile(snakemake.params.nrange, model, solver)
    res["API"] = "cvxpy"
    res = res.rename_axis("N").reset_index()

    res.to_csv(snakemake.output[0])
