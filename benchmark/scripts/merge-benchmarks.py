#!/usr/bin/env python3
"""
Created on Mon Feb 14 18:00:44 2022.

@author: fabian
"""

import pandas as pd

df = [pd.read_csv(fn, index_col=0) for fn in snakemake.input.benchmarks]
df = pd.concat(df, ignore_index=True)

if snakemake.config["benchmark"] == "basic":
    df["Number of Variables"] = df.N**2 * 2
    df["Number of Constraints"] = df.N**2 * 2
elif snakemake.config["benchmark"] == "knapsack":
    df["Number of Variables"] = df.N
    df["Number of Constraints"] = df.N

solver_memory = df.loc[df.API == "Solving Process", "Memory"].values
solver_time = df.loc[df.API == "Solving Process", "Time"].values

# Make a correction of the memory usage, some APIs use external processes for the solving process
api_with_external_process = {"pyomo"}
api_with_internal_process = set(snakemake.params.apis).difference(
    api_with_external_process
)


absolute = df.copy()
for api in api_with_external_process:
    absolute.loc[absolute.API == api, "Memory"] += solver_memory
absolute.to_csv(snakemake.output.absolute)

overhead = df.copy()
for api in snakemake.params.apis:
    overhead.loc[overhead.API == api, "Time"] -= solver_time
for api in api_with_internal_process:
    overhead.loc[overhead.API == api, "Memory"] -= solver_memory
overhead = overhead.query("API != 'Solving Process'")
overhead.to_csv(snakemake.output.overhead)
