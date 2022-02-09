#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 09:34:00 2022.

@author: fabian
"""

import pandas as pd

dfs = [pd.read_csv(fn, sep="\t") for fn in snakemake.input]
dfs = pd.concat(dfs, axis=0, ignore_index=True)
dfs["N"] = snakemake.params.nrange

dfs = dfs.rename(columns={"s": "Time [s]", "max_rss": "Memory Usage"})
dfs = dfs.replace("-", 0)
dfs["Memory Usage"] = dfs["Memory Usage"].astype(float)

if snakemake.wildcards.api == snakemake.wildcards.solver:
    dfs["API"] = "Solver Process"
else:
    dfs["API"] = snakemake.wildcards.api

cols = ["N", "Time [s]", "Memory Usage", "API"]
dfs[cols].to_csv(snakemake.output[0])
