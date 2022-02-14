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

dfs = dfs.rename(columns={"s": "Time", "max_rss": "Memory"})
dfs = dfs.replace("-", 0)
dfs["Memory"] = dfs["Memory"].astype(float)

if snakemake.wildcards.api == "solver":
    dfs["API"] = "Solving Process"
else:
    dfs["API"] = snakemake.wildcards.api

cols = ["N", "Time", "Memory", "API"]
dfs[cols].to_csv(snakemake.output[0])
