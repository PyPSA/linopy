#!/usr/bin/env python3
"""
Created on Wed Feb  9 09:34:00 2022.

@author: fabian
"""

from pathlib import Path

import pandas as pd

dfs = [pd.read_csv(fn, sep="\t") for fn in snakemake.input.memory]
df = pd.concat(dfs, axis=0, ignore_index=True)
df["N"] = snakemake.params.nrange

df = df.rename(columns={"s": "Time", "max_rss": "Memory"})
df = df.replace("-", 0)
df["Memory"] = df["Memory"].astype(float)

if snakemake.params.api == "solver":
    df["API"] = "Solving Process"
else:
    df["API"] = snakemake.params.api

df = df[["N", "Time", "Memory", "API"]]

benchmark_time = snakemake.input.time
if benchmark_time is not None:
    if isinstance(benchmark_time, str):
        df_time = pd.read_csv(benchmark_time, index_col=0)
        df["Time"] = df_time.Time.values
        df["Objective"] = df_time.Objective.values
    else:
        # for solvers we need to read the time from the single output files
        df["Time"] = [float(Path(fn).read_text()) for fn in benchmark_time]


df.to_csv(snakemake.output.benchmark)
