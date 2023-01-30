from common import profile
from ortools.linear_solver import pywraplp


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


if __name__ == "__main__":
    solver = snakemake.wildcards.solver

    # dry run first
    model(2, solver)

    res = profile(snakemake.params.nrange, model, solver)
    res["API"] = "ortools"
    res = res.rename_axis("N").reset_index()

    res.to_csv(snakemake.output[0])
