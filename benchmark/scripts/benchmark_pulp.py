import pulp
from common import profile
from numpy.random import default_rng

# Random seed for reproducibility
rng = default_rng(125)


def basic_model(n, solver):
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
    return pulp.value(m.objective)


def knapsack_model(n, solver):
    # Define the problem
    m = pulp.LpProblem("Knapsack Problem", pulp.LpMaximize)

    m.i = list(range(n))

    # Define the variables
    weight = rng.integers(1, 100, size=n)
    value = rng.integers(1, 100, size=n)

    x = pulp.LpVariable.dicts("x", (m.i,), lowBound=0, upBound=1, cat=pulp.LpInteger)

    # Define the constraints
    m += pulp.lpSum([weight[i] * x[i] for i in m.i]) <= 200

    # Define the objective function
    m += pulp.lpSum([value[i] * x[i] for i in m.i])

    # Solve the problem
    solver = pulp.getSolver(solver.upper())
    m.solve(solver)

    return pulp.value(m.objective)


if __name__ == "__main__":
    solver = snakemake.config["solver"]

    if snakemake.config["benchmark"] == "basic":
        model = basic_model
    elif snakemake.config["benchmark"] == "knapsack":
        model = knapsack_model

    # dry run first
    model(2, solver)

    res = profile(snakemake.params.nrange, model, solver)
    res["API"] = "pulp"
    res = res.rename_axis("N").reset_index()

    res.to_csv(snakemake.output[0])
