from numpy import arange
from numpy.random import default_rng

from linopy import Model

# Random seed for reproducibility
rng = default_rng(125)


if snakemake.config["benchmark"] == "basic":

    def create_model(n):
        m = Model()
        N, M = [arange(n), arange(n)]
        x = m.add_variables(coords=[N, M])
        y = m.add_variables(coords=[N, M])
        m.add_constraints(x - y >= N)
        m.add_constraints(x + y >= 0)
        m.add_objective((2 * x).sum() + y.sum())
        return m

elif snakemake.config["benchmark"] == "knapsack":

    def create_model(n):
        m = Model()
        packages = m.add_variables(coords=[arange(n)], binary=True)
        weight = rng.integers(1, 100, size=n)
        value = rng.integers(1, 100, size=n)
        m.add_constraints((weight * packages).sum() <= 200)
        m.add_objective(-(value * packages).sum())  # use minus because of minimization
        return m


for fn in snakemake.output:
    N = int(fn.split("/")[-1][:-3])
    create_model(N).to_file(fn)
