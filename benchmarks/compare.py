"""Compare two benchmark result JSON files and produce a plot."""

from __future__ import annotations

import json
from pathlib import Path


def compare(old_path: str, new_path: str) -> None:
    """Load two result JSONs and produce a comparison PNG."""
    import matplotlib.pyplot as plt

    with open(old_path) as f:
        old = json.load(f)
    with open(new_path) as f:
        new = json.load(f)

    old_label = old.get("label", Path(old_path).stem)
    new_label = new.get("label", Path(new_path).stem)
    phase = old.get("phase", "unknown")
    model_name = old.get("model", "unknown")

    old_runs = old.get("runs", [])
    new_runs = new.get("runs", [])

    if not old_runs or not new_runs:
        print("No runs to compare.")
        return

    # Find the primary metric based on phase
    metric_keys = {
        "build": "build_time_median_s",
        "memory": "peak_memory_median_mb",
        "lp_write": "write_time_median_s",
    }
    metric = metric_keys.get(phase)
    if metric is None:
        # Try to auto-detect
        for key in old_runs[0]:
            if "median" in key:
                metric = key
                break
    if metric is None:
        print("Cannot determine metric to plot.")
        return

    c_old, c_new = "#1b9e77", "#d95f02"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Absolute values
    ax = axes[0]
    x_old = list(range(len(old_runs)))
    x_new = list(range(len(new_runs)))
    y_old = [r.get(metric, 0) for r in old_runs]
    y_new = [r.get(metric, 0) for r in new_runs]
    labels_old = [str(r.get("params", {})) for r in old_runs]

    ax.plot(x_old, y_old, "o-", color=c_old, label=old_label, linewidth=2, markersize=8)
    ax.plot(
        x_new, y_new, "s--", color=c_new, label=new_label, linewidth=2, markersize=8
    )
    ax.set_xticks(x_old)
    ax.set_xticklabels(labels_old, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(metric)
    ax.set_title(f"{model_name} / {phase}: {metric}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Ratio (new / old)
    ax = axes[1]
    n_compare = min(len(old_runs), len(new_runs))
    ratios = []
    for i in range(n_compare):
        vo = old_runs[i].get(metric, 0)
        vn = new_runs[i].get(metric, 0)
        ratios.append(vn / vo if vo > 0 else float("nan"))

    ax.bar(range(n_compare), ratios, color=c_new, alpha=0.7)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.set_xticks(range(n_compare))
    ax.set_xticklabels(labels_old[:n_compare], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"Ratio ({new_label} / {old_label})")
    ax.set_title("Relative performance")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Benchmark Comparison: {model_name} / {phase}", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()

    out_png = Path(old_path).parent / f"compare_{model_name}_{phase}.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")
    plt.close()
