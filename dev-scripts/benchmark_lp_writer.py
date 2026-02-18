#!/usr/bin/env python3
"""
Benchmark script for LP file writing and model build performance.

Usage:
    # Benchmark LP write speed (default):
    python dev-scripts/benchmark_lp_writer.py --output results.json [--label "my branch"]

    # Benchmark model build speed:
    python dev-scripts/benchmark_lp_writer.py --phase build --output results.json

    # Benchmark memory usage of the built model:
    python dev-scripts/benchmark_lp_writer.py --phase memory --output results.json

    # Plot comparison of two result files:
    python dev-scripts/benchmark_lp_writer.py --plot master.json this_pr.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np
from numpy.random import default_rng

from linopy import Model

rng = default_rng(125)


def basic_model(n: int) -> Model:
    """Create a basic model with 2*n^2 variables and 2*n^2 constraints."""
    m = Model()
    N = np.arange(n)
    x = m.add_variables(coords=[N, N], name="x")
    y = m.add_variables(coords=[N, N], name="y")
    m.add_constraints(x - y >= N, name="c1")
    m.add_constraints(x + y >= 0, name="c2")
    m.add_objective((2 * x).sum() + y.sum())
    return m


def knapsack_model(n: int) -> Model:
    """Create a knapsack model with n binary variables and 1 constraint."""
    m = Model()
    packages = m.add_variables(coords=[np.arange(n)], binary=True)
    weight = rng.integers(1, 100, size=n)
    value = rng.integers(1, 100, size=n)
    m.add_constraints((weight * packages).sum() <= 200)
    m.add_objective(-(value * packages).sum())
    return m


def pypsa_model(snapshots: int | None = None) -> Model | None:
    """Create a model from the PyPSA SciGrid-DE example network."""
    try:
        import pandas as pd
        import pypsa
    except ImportError:
        return None
    n = pypsa.examples.scigrid_de()
    if snapshots is not None and snapshots > len(n.snapshots):
        orig = n.snapshots
        repeats = -(-snapshots // len(orig))
        new_index = pd.date_range(orig[0], periods=len(orig) * repeats, freq=orig.freq)
        new_index = new_index[:snapshots]
        n.set_snapshots(new_index)
    n.optimize.create_model()
    return n.model


# ---------------------------------------------------------------------------
# Memory measurement helpers
# ---------------------------------------------------------------------------


def model_nbytes(m: Model) -> dict[str, int]:
    """Return byte sizes of the model's variable and constraint datasets."""
    var_bytes = sum(
        v.nbytes
        for name in m.variables
        for v in m.variables[name].data.data_vars.values()
    )
    con_bytes = sum(
        v.nbytes
        for name in m.constraints
        for v in m.constraints[name].data.data_vars.values()
    )
    return {
        "var_bytes": var_bytes,
        "con_bytes": con_bytes,
        "total_bytes": var_bytes + con_bytes,
    }


def measure_build_memory(builder, *args, **kwargs) -> tuple[Model, int]:
    """Build a model while tracking peak memory allocation with tracemalloc."""
    tracemalloc.start()
    m = builder(*args, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return m, peak


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------


def benchmark_lp_write(
    label: str, m: Model, iterations: int = 10, io_api: str | None = None
) -> dict:
    """Benchmark LP file writing speed. Returns dict with results."""
    to_file_kwargs: dict = dict(progress=False)
    if io_api is not None:
        to_file_kwargs["io_api"] = io_api
    with tempfile.TemporaryDirectory() as tmpdir:
        m.to_file(Path(tmpdir) / "warmup.lp", **to_file_kwargs)
        times = []
        for i in range(iterations):
            fn = Path(tmpdir) / f"bench_{i}.lp"
            start = time.perf_counter()
            m.to_file(fn, **to_file_kwargs)
            times.append(time.perf_counter() - start)

    return _timing_result(label, m, times, phase="lp_write")


def benchmark_build(
    label: str, builder, builder_args: tuple, iterations: int = 10
) -> dict:
    """Benchmark model build speed. Returns dict with results."""
    # warmup
    builder(*builder_args)
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        m = builder(*builder_args)
        times.append(time.perf_counter() - start)

    return _timing_result(label, m, times, phase="build")


def benchmark_memory(label: str, builder, builder_args: tuple) -> dict:
    """Benchmark memory usage of the built model."""
    m, peak_alloc = measure_build_memory(builder, *builder_args)
    nb = model_nbytes(m)
    nvars = int(m.nvars)
    ncons = int(m.ncons)
    print(
        f"  {label:55s} ({nvars:>9,} vars, {ncons:>9,} cons): "
        f"datasets={nb['total_bytes'] / 1e6:7.2f} MB, peak_alloc={peak_alloc / 1e6:7.2f} MB"
    )
    return {
        "label": label,
        "nvars": nvars,
        "ncons": ncons,
        "phase": "memory",
        **nb,
        "peak_alloc_bytes": peak_alloc,
    }


def _timing_result(label: str, m: Model, times: list[float], phase: str) -> dict:
    avg = float(np.mean(times))
    med = float(np.median(times))
    q25 = float(np.percentile(times, 25))
    q75 = float(np.percentile(times, 75))
    nvars = int(m.nvars)
    ncons = int(m.ncons)
    print(
        f"  {label:55s} ({nvars:>9,} vars, {ncons:>9,} cons): "
        f"{med * 1000:7.1f}ms (IQR {q25 * 1000:.1f}-{q75 * 1000:.1f}ms)"
    )
    return {
        "label": label,
        "nvars": nvars,
        "ncons": ncons,
        "phase": phase,
        "mean_s": avg,
        "median_s": med,
        "q25_s": q25,
        "q75_s": q75,
        "times_s": times,
    }


# ---------------------------------------------------------------------------
# Size configurations
# ---------------------------------------------------------------------------

BASIC_SIZES = [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000]
PYPSA_SNAPS = [24, 50, 100, 200, 500, 1000]


def run_benchmarks(
    phase: str = "lp_write",
    io_api: str | None = None,
    iterations: int = 10,
    model_type: str = "basic",
) -> list[dict]:
    """
    Run benchmarks for a single model type across sizes.

    Parameters
    ----------
    phase : str
        "lp_write" (default) - benchmark LP file writing speed.
        "build" - benchmark model construction speed.
        "memory" - measure dataset nbytes and peak allocation.
    model_type : str
        "basic" (default) - N from 5 to 2000, giving 50 to 8M vars.
        "pypsa" - PyPSA SciGrid-DE with varying snapshot counts.
    """
    results = []

    if model_type == "basic":
        print(f"\nbasic_model (2 x N^2 vars, 2 x N^2 constraints) — phase={phase}:")
        for n in BASIC_SIZES:
            iters = iterations * 5 if n <= 100 else iterations
            if phase == "lp_write":
                r = benchmark_lp_write(
                    f"basic N={n}", basic_model(n), iters, io_api=io_api
                )
            elif phase == "build":
                r = benchmark_build(f"basic N={n}", basic_model, (n,), iters)
            elif phase == "memory":
                r = benchmark_memory(f"basic N={n}", basic_model, (n,))
            else:
                raise ValueError(f"Unknown phase: {phase!r}")
            r["model"] = "basic"
            r["param"] = n
            results.append(r)

    elif model_type == "pypsa":
        print(f"\nPyPSA SciGrid-DE — phase={phase}:")
        for snaps in PYPSA_SNAPS:
            if phase == "memory":
                m, peak = measure_build_memory(pypsa_model, snaps)
                if m is None:
                    print("  (skipped, pypsa not installed)")
                    break
                nb = model_nbytes(m)
                r = {
                    "label": f"pypsa {snaps} snaps",
                    "nvars": int(m.nvars),
                    "ncons": int(m.ncons),
                    "phase": "memory",
                    **nb,
                    "peak_alloc_bytes": peak,
                }
                print(
                    f"  pypsa {snaps} snaps ({m.nvars:>9,} vars, {m.ncons:>9,} cons): "
                    f"datasets={nb['total_bytes'] / 1e6:7.2f} MB, peak_alloc={peak / 1e6:7.2f} MB"
                )
            elif phase == "build":
                # For PyPSA, "build" means calling pypsa_model()
                pypsa_model(snaps)  # warmup
                times = []
                m = None
                for _ in range(iterations):
                    start = time.perf_counter()
                    m = pypsa_model(snaps)
                    times.append(time.perf_counter() - start)
                if m is None:
                    print("  (skipped, pypsa not installed)")
                    break
                r = _timing_result(f"pypsa {snaps} snaps", m, times, phase="build")
            else:
                m = pypsa_model(snapshots=snaps)
                if m is None:
                    print("  (skipped, pypsa not installed)")
                    break
                r = benchmark_lp_write(
                    f"pypsa {snaps} snaps", m, iterations, io_api=io_api
                )
            r["model"] = "pypsa"
            r["param"] = snaps
            results.append(r)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_comparison(file_old: str, file_new: str) -> None:
    """Create 4-panel comparison plot from two JSON result files."""
    import matplotlib.pyplot as plt

    with open(file_old) as f:
        data_old = json.load(f)
    with open(file_new) as f:
        data_new = json.load(f)

    label_old = data_old.get("label", Path(file_old).stem)
    label_new = data_new.get("label", Path(file_new).stem)
    phase = data_old["results"][0].get("phase", "lp_write")

    is_memory = phase == "memory"

    def get_stats(data):
        nv = [r["nvars"] for r in data["results"]]
        if is_memory:
            vals = [r["total_bytes"] / 1e6 for r in data["results"]]
            return nv, vals, vals, vals  # no spread for memory
        if "median_s" in data["results"][0]:
            med = [r["median_s"] * 1000 for r in data["results"]]
            lo = [r["q25_s"] * 1000 for r in data["results"]]
            hi = [r["q75_s"] * 1000 for r in data["results"]]
        else:
            med = [r["mean_s"] * 1000 for r in data["results"]]
            std = [r["std_s"] * 1000 for r in data["results"]]
            lo = [m - s for m, s in zip(med, std)]
            hi = [m + s for m, s in zip(med, std)]
        return nv, med, lo, hi

    nv_old, med_old, lo_old, hi_old = get_stats(data_old)
    nv_new, med_new, lo_new, hi_new = get_stats(data_new)

    y_label = "Memory (MB)" if is_memory else "Time (ms, median)"
    title_prefix = f"{phase.replace('_', ' ').title()} Performance"

    color_old, color_new = "#1f77b4", "#ff7f0e"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{title_prefix}: {label_old} vs {label_new}", fontsize=14)

    def plot_errorbar(ax, nv, med, lo, hi, **kwargs):
        yerr_lo = [m - l for m, l in zip(med, lo)]
        yerr_hi = [h - m for m, h in zip(med, hi)]
        ax.errorbar(nv, med, yerr=[yerr_lo, yerr_hi], capsize=3, **kwargs)

    # Panel 1: All data, log-log
    ax = axes[0, 0]
    plot_errorbar(
        ax,
        nv_old,
        med_old,
        lo_old,
        hi_old,
        marker="o",
        color=color_old,
        linestyle="--",
        label=label_old,
        alpha=0.8,
    )
    plot_errorbar(
        ax,
        nv_new,
        med_new,
        lo_new,
        hi_new,
        marker="s",
        color=color_new,
        linestyle="-",
        label=label_new,
        alpha=0.8,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of variables")
    ax.set_ylabel(y_label)
    ax.set_title(f"{title_prefix} vs problem size (log-log)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Ratio (old/new)
    ax = axes[0, 1]
    if len(nv_old) == len(nv_new):
        ratio = [o / n if n > 0 else 1 for o, n in zip(med_old, med_new)]
        ax.plot(nv_old, ratio, marker="o", color="#2ca02c")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Number of variables")
    ratio_label = "Reduction" if is_memory else "Speedup"
    ax.set_ylabel(f"{ratio_label} ({label_old} / {label_new})")
    ax.set_title(f"{ratio_label} vs problem size")
    ax.grid(True, alpha=0.3)

    # Panel 3: Small models
    ax = axes[1, 0]
    cutoff = 25000
    idx_old = [i for i, n in enumerate(nv_old) if n <= cutoff]
    idx_new = [i for i, n in enumerate(nv_new) if n <= cutoff]
    plot_errorbar(
        ax,
        [nv_old[i] for i in idx_old],
        [med_old[i] for i in idx_old],
        [lo_old[i] for i in idx_old],
        [hi_old[i] for i in idx_old],
        marker="o",
        color=color_old,
        linestyle="--",
        label=label_old,
        alpha=0.8,
    )
    plot_errorbar(
        ax,
        [nv_new[i] for i in idx_new],
        [med_new[i] for i in idx_new],
        [lo_new[i] for i in idx_new],
        [hi_new[i] for i in idx_new],
        marker="s",
        color=color_new,
        linestyle="-",
        label=label_new,
        alpha=0.8,
    )
    ax.set_xlabel("Number of variables")
    ax.set_ylabel(y_label)
    ax.set_ylim(bottom=0)
    ax.set_title(f"Small models (<= {cutoff:,} vars)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Large models
    ax = axes[1, 1]
    idx_old = [i for i, n in enumerate(nv_old) if n > cutoff]
    idx_new = [i for i, n in enumerate(nv_new) if n > cutoff]
    plot_errorbar(
        ax,
        [nv_old[i] for i in idx_old],
        [med_old[i] for i in idx_old],
        [lo_old[i] for i in idx_old],
        [hi_old[i] for i in idx_old],
        marker="o",
        color=color_old,
        linestyle="--",
        label=label_old,
        alpha=0.8,
    )
    plot_errorbar(
        ax,
        [nv_new[i] for i in idx_new],
        [med_new[i] for i in idx_new],
        [lo_new[i] for i in idx_new],
        [hi_new[i] for i in idx_new],
        marker="s",
        color=color_new,
        linestyle="-",
        label=label_new,
        alpha=0.8,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Number of variables")
    ax.set_ylabel(y_label)
    ax.set_title(f"Large models (> {cutoff:,} vars)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"dev-scripts/benchmark_{phase}_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Linopy benchmark (speed & memory)")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument("--label", default=None, help="Label for this run")
    parser.add_argument("--io-api", default=None, help="io_api to pass to to_file()")
    parser.add_argument(
        "--phase",
        default="lp_write",
        choices=["lp_write", "build", "memory"],
        help="What to benchmark: lp_write (default), build, or memory",
    )
    parser.add_argument(
        "--model",
        default="basic",
        choices=["basic", "pypsa"],
        help="Model type to benchmark (default: basic)",
    )
    parser.add_argument(
        "--plot",
        nargs=2,
        metavar=("OLD", "NEW"),
        help="Plot comparison from two JSON files",
    )
    args = parser.parse_args()

    if args.plot:
        plot_comparison(args.plot[0], args.plot[1])
        return

    iterations = 10
    label = args.label or "benchmark"
    print(
        f"Linopy benchmark — phase={args.phase}, model={args.model}, "
        f"iterations={iterations}, label={label!r}"
    )
    print("=" * 90)

    results = run_benchmarks(
        phase=args.phase,
        io_api=args.io_api,
        iterations=iterations,
        model_type=args.model,
    )

    output = {"label": label, "phase": args.phase, "results": results}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        print("\n(use --output FILE to save results for later plotting)")


if __name__ == "__main__":
    main()
