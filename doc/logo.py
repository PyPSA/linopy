#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 12:08:47 2021.

@author: fabian
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

textparams = dict(font="Ubuntu", color="#6f6070")

fig, (ax, ax1) = plt.subplots(
    1, 2, gridspec_kw={"width_ratios": [2, 3]}, figsize=(10, 3)
)
# fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1,3]})
ax.axis("off")
ax.set_aspect("equal", adjustable="box")
N = 10
ax.set_xlim(0, N - 1)

c = np.array([2, 3])
x = np.array([2.5, 3])
carray = pd.DataFrame([[c @ np.array([x1, x2]) for x1 in range(N)] for x2 in range(N)])
ax.contourf(carray, levels=1000, cmap="Greens")
ax.fill_between(np.linspace(-1, 7, N), np.linspace(6, 0, N), alpha=0.3, color="orange")
ax.fill_between(np.linspace(1, 9, N), np.linspace(0, 4, N), alpha=0.2, color="red")


ax.scatter(4.6, 1.9, marker="8", color="white", zorder=8)

ax1.text(0, 0.4, "linopy", **textparams, size=170, ha="left", va="center")
ax1.axis("off")


fig.tight_layout()
fig.savefig("logo.png", bbox_inches="tight", transparent=True)
fig.savefig("logo.pdf", bbox_inches="tight", transparent=True)
