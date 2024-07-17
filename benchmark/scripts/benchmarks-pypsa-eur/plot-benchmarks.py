#!/usr/bin/env python3
"""
Created on Tue Feb 15 17:11:01 2022.

@author: fabian
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dfs = []
for api, path in zip(snakemake.input.keys(), snakemake.input):
    df = pd.read_csv(path, skiprows=1, header=None, sep=" ")

    df.columns = ["API", "Memory", "Time"]
    df.API = api
    df.Time -= df.Time[0]
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(data=df, y="Memory", x="Time", hue="API", style="API", ax=ax)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Memory Usage [MB]")
# ax.set_xlim()
fig.tight_layout()
fig.savefig(snakemake.output[0])
