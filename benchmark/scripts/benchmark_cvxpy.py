import cvxpy as cp
import numpy as np
from common import profile


def model(n, solver):
    # Create variables
    x = cp.Variable((n, n))
    y = cp.Variable((n, n))

    constraints = [x - y >= np.repeat(np.arange(n)[:, np.newaxis], n, 1), x + y >= 0]

    # Create objective
    objective = cp.Minimize(2 * cp.sum(x) + cp.sum(y))

    # Optimize the model
    m = cp.Problem(objective, constraints)
    m.solve(solver=solver.upper())

    return m


if __name__ == "__main__":
    solver = snakemake.wildcards.solver

    # dry run first
    model(2, solver)

    res = profile(snakemake.params.nrange, model, solver)
    res["API"] = "cvxpy"
    res = res.rename_axis("N").reset_index()

    res.to_csv(snakemake.output[0])
