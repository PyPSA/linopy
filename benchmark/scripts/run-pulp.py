from benchmark_pulp import model

n = int(snakemake.wildcards.N)
solver = snakemake.wildcards.solver
integerlabels = snakemake.params.integerlabels
model(n, solver, integerlabels)
