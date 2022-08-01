#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 18:00:44 2022.

@author: fabian
"""

import pandas as pd

df = [pd.read_csv(fn, index_col=0) for fn in snakemake.input.benchmarks]
df = pd.concat(df, ignore_index=True)

df_time = [pd.read_csv(fn, index_col=0) for fn in snakemake.input.benchmarks_time]
df_time = pd.concat(df_time, ignore_index=True)

# update time benchmarks
for api in df_time.API.unique():
    df.loc[df.API == api, "Time"] = df_time.loc[df.API == api, "Time"].values


df["Number of Variables"] = df.N**2 * 2
df["Number of Constraints"] = df.N**2 * 2

solver_usage = df.loc[df.API == "Solving Process", "Memory"].values

absolute = df.copy()
# Make a correction of the memory usage:
# Pyomo uses external threads for the solving process, this is not counted by the snakemake
# memory tracking
absolute.loc[absolute.API == "pyomo", "Memory"] += solver_usage
absolute.to_csv(snakemake.output.absolute)

overhead = df.copy()
overhead.loc[overhead.API == "linopy", "Memory"] -= solver_usage
overhead.loc[overhead.API == "jump", "Memory"] -= solver_usage
overhead = overhead.query("API != 'Solving Process'")
overhead.to_csv(snakemake.output.overhead)
