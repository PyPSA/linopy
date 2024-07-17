from common import profile
from numpy.random import default_rng
from ortools.linear_solver import pywraplp

# Random seed for reproducibility
rng = default_rng(125)


def model(n, solver):
    # Create a new linear solver
    solver = pywraplp.Solver("LinearExample", pywraplp.Solver.GUROBI_LINEAR_PROGRAMMING)

    # Create variables
    x = {}
    y = {}
    for i in range(n):
        for j in range(n):
            x[i, j] = solver.NumVar(lb=None, ub=None, name="x_%d_%d" % (i, j))
            y[i, j] = solver.NumVar(lb=None, ub=None, name="y_%d_%d" % (i, j))

    # Create constraints
    for i in range(n):
        for j in range(n):
            solver.Add(x[i, j] - y[i, j] >= i)
            solver.Add(x[i, j] + y[i, j] >= 0)

    # Create objective
    obj = solver.Objective()
    for i in range(n):
        for j in range(n):
            obj.Add(2 * x[i, j] + y[i, j])
    obj.SetMinimization()

    # Solve the model
    solver.Solve()

    return solver


def knapsack_model(n, solver):
    # Create a new linear solver
    solver = pywraplp.Solver("LinearExample", pywraplp.Solver.GUROBI_LINEAR_PROGRAMMING)

    weight = rng.integers(1, 100, size=n)
    value = rng.integers(1, 100, size=n)

    x = {i: solver.BoolVar("x_%d" % i) for i in range(n)}
    # Create constraints
    solver.Add(solver.Sum([weight[i] * x[i] for i in range(n)]) <= 200)

    # Create objective
    obj = solver.Objective()
    for i in range(n):
        obj.Add(value[i] * x[i])
    obj.SetMaximization()

    # Solve the model
    solver.Solve()

    return solver


if snakemake.config["benchmark"] == "basic":
    model = basic_model
elif snakemake.config["benchmark"] == "knapsack":
    model = knapsack_model


if __name__ == "__main__":
    solver = snakemake.config["solver"]

    # dry run first
    model(2, solver)

    res = profile(snakemake.params.nrange, model, solver)
    res["API"] = "ortools"
    res = res.rename_axis("N").reset_index()

    res.to_csv(snakemake.output[0])
