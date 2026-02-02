"""Compare benchmark results across branches and produce plots."""

from __future__ import annotations

import json
from pathlib import Path

# Metric keys per phase: (median, q25, q75)
METRIC_KEYS: dict[str, tuple[str, str, str]] = {
    "build": ("build_time_median_s", "build_time_q25_s", "build_time_q75_s"),
    "memory": ("peak_memory_median_mb", "peak_memory_median_mb", "peak_memory_max_mb"),
    "lp_write": ("write_time_median_s", "write_time_q25_s", "write_time_q75_s"),
}

METRIC_UNITS: dict[str, str] = {
    "build": "Build time (ms)",
    "memory": "Peak memory (MB)",
    "lp_write": "Write time (ms)",
}

# Phases where raw values are seconds → display in ms
MS_PHASES = {"build", "lp_write"}

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
MARKERS = ["o", "s", "D", "^", "v", "P"]


def _load(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    data.setdefault("label", Path(path).stem)
    return data


def _extract(
    runs: list[dict], phase: str
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Extract nvars, median, lo, hi from runs. Convert to ms where needed."""
    keys = METRIC_KEYS.get(phase)
    if not keys or not runs:
        return [], [], [], []

    med_key, lo_key, hi_key = keys
    scale = 1000.0 if phase in MS_PHASES else 1.0

    nvars = [r["nvars"] for r in runs]
    med = [r[med_key] * scale for r in runs]
    lo = [r.get(lo_key, r[med_key]) * scale for r in runs]
    hi = [r.get(hi_key, r[med_key]) * scale for r in runs]
    return nvars, med, lo, hi


def _plot_errorbar(ax, nvars, med, lo, hi, **kwargs):
    yerr_lo = [m - l for m, l in zip(med, lo)]
    yerr_hi = [h - m for m, h in zip(med, hi)]
    ax.errorbar(nvars, med, yerr=[yerr_lo, yerr_hi], capsize=3, **kwargs)


def compare(*paths: str) -> None:
    """
    Compare any number of result JSONs for the same model x phase.

    Produces a 4-panel plot:
      Top-left:     Log-log overview with error bars
      Top-right:    Speedup ratio vs baseline with uncertainty bounds
      Bottom-left:  Small models (linear scale)
      Bottom-right: Large models (log scale)
    """
    if len(paths) < 2:
        print("Need at least 2 files to compare.")
        return

    import matplotlib.pyplot as plt

    datasets = [_load(p) for p in paths]
    phase = datasets[0].get("phase", "unknown")
    model_name = datasets[0].get("model", "unknown")
    ylabel = METRIC_UNITS.get(phase, phase)

    for d in datasets[1:]:
        if d.get("model") != model_name or d.get("phase") != phase:
            print(
                f"Warning: mixing model/phase — "
                f"expected {model_name}/{phase}, "
                f"got {d.get('model')}/{d.get('phase')}"
            )

    # Extract stats for each dataset
    all_stats = []
    for d in datasets:
        nvars, med, lo, hi = _extract(d.get("runs", []), phase)
        all_stats.append((d["label"], nvars, med, lo, hi))

    if not all_stats[0][1]:
        print("No data to plot.")
        return

    labels = [s[0] for s in all_stats]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Benchmark: {model_name} / {phase}\n{' vs '.join(labels)}",
        fontsize=14,
    )

    # --- Panel 1: All data, log-log ---
    ax = axes[0, 0]
    for i, (label, nvars, med, lo, hi) in enumerate(all_stats):
        color = COLORS[i % len(COLORS)]
        marker = MARKERS[i % len(MARKERS)]
        ls = "--" if i == 0 else "-"
        _plot_errorbar(
            ax,
            nvars,
            med,
            lo,
            hi,
            marker=marker,
            color=color,
            linestyle=ls,
            label=label,
            alpha=0.8,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of variables")
    ax.set_ylabel(ylabel)
    ax.set_title("Overview (log-log)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Speedup ratio with uncertainty bounds ---
    ax = axes[0, 1]
    base_label, base_nv, base_med, base_lo, base_hi = all_stats[0]
    for i, (label, nvars, med, lo, hi) in enumerate(all_stats[1:], 1):
        if len(nvars) != len(base_nv):
            continue
        color = COLORS[i % len(COLORS)]
        # Ratio: baseline / current (>1 means current is faster)
        ratio = [b / c if c > 0 else float("nan") for b, c in zip(base_med, med)]
        # Uncertainty: best = base_hi/lo_cur, worst = base_lo/hi_cur
        ratio_lo = [bl / ch if ch > 0 else float("nan") for bl, ch in zip(base_lo, hi)]
        ratio_hi = [bh / cl if cl > 0 else float("nan") for bh, cl in zip(base_hi, lo)]
        yerr_lo = [r - rl for r, rl in zip(ratio, ratio_lo)]
        yerr_hi = [rh - r for r, rh in zip(ratio, ratio_hi)]
        ax.errorbar(
            nvars,
            ratio,
            yerr=[yerr_lo, yerr_hi],
            marker=MARKERS[i % len(MARKERS)],
            color=color,
            capsize=3,
            label=label,
        )
        ax.fill_between(nvars, ratio_lo, ratio_hi, alpha=0.15, color=color)
        for x, r in zip(nvars, ratio):
            ax.annotate(
                f"{r:.2f}",
                (x, r),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color=color,
            )
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Number of variables")
    ax.set_ylabel(f"Speedup ({base_label} / other)")
    ax.set_title("Relative performance")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panels 3 & 4: Small vs large models ---
    cutoff = 25000

    for panel_idx, (title, filt, use_log) in enumerate(
        [
            (f"Small models (≤ {cutoff:,} vars)", lambda n: n <= cutoff, False),
            (f"Large models (> {cutoff:,} vars)", lambda n: n > cutoff, True),
        ]
    ):
        ax = axes[1, panel_idx]
        has_data = False
        for i, (label, nvars, med, lo, hi) in enumerate(all_stats):
            idx = [j for j, n in enumerate(nvars) if filt(n)]
            if not idx:
                continue
            has_data = True
            color = COLORS[i % len(COLORS)]
            marker = MARKERS[i % len(MARKERS)]
            ls = "--" if i == 0 else "-"
            _plot_errorbar(
                ax,
                [nvars[j] for j in idx],
                [med[j] for j in idx],
                [lo[j] for j in idx],
                [hi[j] for j in idx],
                marker=marker,
                color=color,
                linestyle=ls,
                label=label,
                alpha=0.8,
            )
        if use_log and has_data:
            ax.set_xscale("log")
        if not use_log:
            ax.set_ylim(bottom=0)
        ax.set_xlabel("Number of variables")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if not has_data:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="gray",
            )

    plt.tight_layout()
    out_png = Path(paths[0]).parent / f"compare_{model_name}_{phase}.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close()
