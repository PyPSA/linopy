"""Compare benchmark results across branches and produce plots."""

from __future__ import annotations

import json
from pathlib import Path

# Primary metric per phase
METRIC_KEYS = {
    "build": "build_time_median_s",
    "memory": "peak_memory_median_mb",
    "lp_write": "write_time_median_s",
}

# IQR band keys per phase (lower, upper)
IQR_KEYS = {
    "build": ("build_time_q25_s", "build_time_q75_s"),
    "memory": None,
    "lp_write": ("write_time_q25_s", "write_time_q75_s"),
}

COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
MARKERS = ["o", "s", "D", "^", "v", "P"]


def _load(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    data.setdefault("label", Path(path).stem)
    return data


def _detect_metric(phase: str, runs: list[dict]) -> str | None:
    metric = METRIC_KEYS.get(phase)
    if metric and runs and metric in runs[0]:
        return metric
    # Fallback: first key containing "median"
    if runs:
        for key in runs[0]:
            if "median" in key:
                return key
    return None


def _size_label(params: dict) -> str:
    """Short human-readable label from params dict."""
    parts = [f"{k}={v}" for k, v in params.items()]
    return ", ".join(parts)


def _x_value(params: dict) -> float:
    """Extract a numeric x-axis value from params (use product of all values)."""
    vals = [v for v in params.values() if isinstance(v, int | float)]
    result = 1
    for v in vals:
        result *= v
    return float(result)


def compare(*paths: str) -> None:
    """
    Compare any number of result JSONs for the same model×phase.

    Produces a 2-panel plot:
      Left:  absolute metric vs model size, one line per branch
      Right: ratio vs first file (baseline), one line per subsequent branch

    Args:
        *paths: Two or more paths to benchmark JSON files.
    """
    if len(paths) < 2:
        print("Need at least 2 files to compare.")
        return

    import matplotlib.pyplot as plt

    datasets = [_load(p) for p in paths]
    phase = datasets[0].get("phase", "unknown")
    model_name = datasets[0].get("model", "unknown")

    # Validate all files are the same model×phase
    for d in datasets[1:]:
        if d.get("model") != model_name or d.get("phase") != phase:
            print(
                f"Warning: mixing model/phase — "
                f"expected {model_name}/{phase}, "
                f"got {d.get('model')}/{d.get('phase')}"
            )

    metric = _detect_metric(phase, datasets[0].get("runs", []))
    if metric is None:
        print("Cannot determine metric to plot.")
        return

    iqr = IQR_KEYS.get(phase)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Panel 1: Absolute metric vs size ---
    ax = axes[0]
    all_x_labels = []
    for i, data in enumerate(datasets):
        runs = data.get("runs", [])
        if not runs:
            continue
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        xs = list(range(len(runs)))
        ys = [r.get(metric, 0) for r in runs]

        if i == 0:
            all_x_labels = [_size_label(r.get("params", {})) for r in runs]

        ax.plot(
            xs,
            ys,
            marker=marker,
            color=color,
            linewidth=2,
            markersize=7,
            alpha=0.85,
            label=data["label"],
        )

        # IQR band if available
        if iqr and runs[0].get(iqr[0]) is not None:
            lo = [r.get(iqr[0], 0) for r in runs]
            hi = [r.get(iqr[1], 0) for r in runs]
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15)

    ax.set_xticks(range(len(all_x_labels)))
    ax.set_xticklabels(all_x_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(metric)
    ax.set_title(f"{model_name} / {phase}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Ratio vs baseline (first file) ---
    ax = axes[1]
    baseline_runs = datasets[0].get("runs", [])
    baseline_by_params = {
        json.dumps(r["params"], sort_keys=True): r for r in baseline_runs
    }

    for i, data in enumerate(datasets[1:], 1):
        runs = data.get("runs", [])
        if not runs:
            continue
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]

        xs, ys, annots = [], [], []
        for j, r in enumerate(runs):
            key = json.dumps(r["params"], sort_keys=True)
            base = baseline_by_params.get(key)
            if base is None:
                continue
            base_val = base.get(metric, 0)
            cur_val = r.get(metric, 0)
            ratio = cur_val / base_val if base_val > 0 else float("nan")
            xs.append(j)
            ys.append(ratio)
            annots.append(f"{ratio:.2f}")

        ax.plot(
            xs,
            ys,
            marker=marker,
            color=color,
            linewidth=2,
            markersize=7,
            alpha=0.85,
            label=data["label"],
        )
        for x, y, txt in zip(xs, ys, annots):
            ax.annotate(
                txt,
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color=color,
            )

    ax.axhline(1.0, color="k", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.set_xticks(range(len(all_x_labels)))
    ax.set_xticklabels(all_x_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"Ratio (vs {datasets[0]['label']})")
    ax.set_title("Relative to baseline")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Benchmark: {model_name} / {phase}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()

    out_png = Path(paths[0]).parent / f"compare_{model_name}_{phase}.png"
    plt.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")
    plt.close()
