#!/usr/bin/env python3
"""Plot polar-high vs linopy (lp/direct) build+IO time and peak RSS vs size.

Same machine, same HiGHS, same dense LP. PyPSA/linopy#740.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("tool_compare_results.csv")
df["mvars"] = df["n_vars"] / 1e6

style = {
    "linopy-lp": ("#c1432e", "o-", 'linopy io_api="lp"'),
    "linopy-direct": ("#2e7dc1", "o-", 'linopy io_api="direct"'),
    "polar": ("#2e9e5b", "s-", "polar-high (regular)"),
    "polar-sm": ("#9b5bbf", "s--", "polar-high (save_memory)"),
}
order = ["linopy-lp", "linopy-direct", "polar", "polar-sm"]

fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(12, 4.6))

for tool in order:
    g = df[df["tool"] == tool].sort_values("mvars")
    if g.empty:
        continue
    color, ls, label = style[tool]
    ax_t.plot(g["mvars"], g["total_s"], ls, color=color, label=label)
    ax_m.plot(g["mvars"], g["peak_rss_gb"], ls, color=color, label=label)

linopy_ver = df.loc[df["tool"] == "linopy-lp", "version"].iloc[0]
polar_ver = df.loc[df["tool"] == "polar", "version"].iloc[0]

ax_t.set(
    xlabel="variables (millions)",
    ylabel="build + IO time (s)",
    title="Build + IO time\n(HiGHS time-limited, modelling-layer cost only)",
)
ax_m.set(
    xlabel="variables (millions)",
    ylabel="peak process RSS (GB)",
    title="Peak memory",
)
for ax in (ax_t, ax_m):
    ax.grid(alpha=0.3)
    ax.legend()

fig.suptitle(
    f"polar-high {polar_ver} vs linopy {linopy_ver} — dense LP, same machine "
    "(macOS arm64, single thread, HiGHS 1.14.0)",
    fontsize=11,
)
fig.tight_layout()
fig.savefig("tool_compare.svg")
fig.savefig("tool_compare.png", dpi=150)
print("wrote tool_compare.svg / .png")
