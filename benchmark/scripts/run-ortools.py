from benchmark_ortools import basic_model, knapsack_model

if snakemake.config["benchmark"] == "basic":
    model = basic_model
elif snakemake.config["benchmark"] == "knapsack":
    model = knapsack_model

n = int(snakemake.wildcards.N)
solver = snakemake.config["solver"]
model(n, solver)
