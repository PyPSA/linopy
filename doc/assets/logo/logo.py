#!/usr/bin/env python3
"""
Created on Sun Oct 17 12:08:47 2021.

@author: fabian
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

textcolor = "#6f6070"
textparams = dict(font="Ubuntu", color=textcolor)

fig, (ax, ax1) = plt.subplots(
    1, 2, gridspec_kw={"width_ratios": [2, 3]}, figsize=(10, 3)
)
# fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1,3]})
ax.axis("off")
ax.set_aspect("equal", adjustable="box")
N = 11
ax.set_xlim(0, N - 1)

c = np.array([0, 3])
x = np.array([2.5, 3])
carray = pd.DataFrame([[c @ np.array([x1, x2]) for x1 in range(N)] for x2 in range(N)])

# Define the diamond-shaped region of interest
midpoint_x = N / 2
midpoint_y = N / 2

x, y = np.meshgrid(np.arange(N), np.arange(N))
roi = (np.abs(x - midpoint_x) + np.abs(y - midpoint_y) <= midpoint_x) | (  # Top half
    np.abs(x - midpoint_x) + np.abs(y - midpoint_y) <= midpoint_y
)  # Bottom half (mirrored)

# Mask


# Mask the contour plot using the region of interest
masked_array = np.ma.masked_array(carray.values, ~roi)
# ax.contourf(masked_array, levels=1000, cmap="Greens")
# Draw boundaries around the masked contour shape
contour_lines = ax.contour(masked_array, levels=20, colors=textcolor)

# image = ax.imshow(masked_array, cmap="Greens")

# ax.contourf(carray, levels=500, cmap="Greens")
# ax.fill_between(np.linspace(-1, 7, N), np.linspace(4, 0, N), color="white")
# ax.fill_between(np.linspace(1, 9, N), np.linspace(0, 2, N), color='white')
# # ax.plot(np.linspace(-1, 7, N), np.linspace(6, 0, N), alpha=0.3, color=textcolor)
# # ax.plot(np.linspace(1, 9, N), np.linspace(0, 4, N), alpha=0.2, color=textcolor)


ax.scatter(5.54, 0.63, marker="8", color="orange", zorder=8)

ax1.text(0, 0.44, "linopy", **textparams, size=170, ha="left", va="center")
ax1.axis("off")


fig.tight_layout()
fig.savefig("logo.png", bbox_inches="tight", transparent=True)
fig.savefig("logo.pdf", bbox_inches="tight", transparent=True)
