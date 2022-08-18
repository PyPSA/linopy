#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 18:00:44 2022.

@author: fabian
"""

from pathlib import Path

import pandas as pd

df = [pd.read_csv(fn, index_col=0) for fn in snakemake.input.benchmarks]
df = pd.concat(df, ignore_index=True)

df_time = [pd.read_csv(fn, index_col=0) for fn in snakemake.input.benchmarks_time]
df_time = pd.concat(df_time, ignore_index=True)

# update time benchmarks
for api in df_time.API.unique():
    df.loc[df.API == api, "Time"] = df_time.loc[df.API == api, "Time"].values

# update solver times (wihout reading time)
input_dir = Path(snakemake.input[0].rsplit("/", 1)[0])
Ns = snakemake.params[0]
fns = [input_dir / "solver" / f"{N}-time.txt" for N in Ns]
if all(fn.exists() for fn in fns):
    for N, fn in zip(Ns, fns):
        df.loc[(df.N == N) & (df.API == "Solving Process"), "Time"] = float(
            fn.read_text()
        )

df["Number of Variables"] = df.N**2 * 2
df["Number of Constraints"] = df.N**2 * 2

solver_memory = df.loc[df.API == "Solving Process", "Memory"].values
solver_time = df.loc[df.API == "Solving Process", "Time"].values

absolute = df.copy()
# Make a correction of the memory usage:
# Pyomo uses external threads for the solving process, this is not counted by the snakemake
# memory tracking
absolute.loc[absolute.API == "pyomo", "Memory"] += solver_memory
absolute.to_csv(snakemake.output.absolute)

overhead = df.copy()
for api in ["jump", "pyomo", "linopy"]:
    overhead.loc[overhead.API == api, "Time"] -= solver_time
overhead.loc[overhead.API == "jump", "Memory"] -= solver_memory
overhead.loc[overhead.API == "linopy", "Memory"] -= solver_memory
overhead = overhead.query("API != 'Solving Process'")
overhead.to_csv(snakemake.output.overhead)
