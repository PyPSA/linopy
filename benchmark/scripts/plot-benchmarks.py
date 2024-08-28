#!/usr/bin/env python3
"""
Created on Wed Jan 26 23:37:38 2022.

@author: fabian
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = [pd.read_csv(fn) for fn in snakemake.input]
df = pd.concat(df, ignore_index=True)

df["# Variables"] = df.N**2 * 2

fig, ax = plt.subplots()
sns.lineplot(x="# Variables", y="Time [s]", hue="API", data=df, ax=ax)
fig.tight_layout()
fig.savefig(snakemake.output.time)

fig, ax = plt.subplots()
sns.lineplot(x="# Variables", y="Memory Usage", hue="API", data=df, ax=ax)
fig.tight_layout()
fig.savefig(snakemake.output.memory)
