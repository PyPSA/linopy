#!/usr/bin/env python3
"""
Benchmark script for LP file writing performance.

Usage:
    # Run benchmark and save results to JSON:
    python dev-scripts/benchmark_lp_writer.py --output results.json [--label "my branch"]

    # Plot comparison of two result files:
    python dev-scripts/benchmark_lp_writer.py --plot master.json this_pr.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
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


def benchmark_model(
    label: str, m: Model, iterations: int = 10, io_api: str | None = None
) -> dict:
    """Benchmark LP file writing. Returns dict with results."""
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

    avg = float(np.mean(times))
    std = float(np.std(times))
    nvars = int(m.nvars)
    ncons = int(m.ncons)
    print(
        f"  {label:55s} ({nvars:>9,} vars, {ncons:>9,} cons): "
        f"{avg * 1000:7.1f}ms ± {std * 1000:5.1f}ms"
    )
    return {
        "label": label,
        "nvars": nvars,
        "ncons": ncons,
        "mean_s": avg,
        "std_s": std,
        "times_s": times,
    }


def run_benchmarks(
    io_api: str | None = None,
    iterations: int = 10,
    model_type: str = "basic",
) -> list[dict]:
    """
    Run benchmarks for a single model type across sizes.

    Parameters
    ----------
    model_type : str
        "basic" (default) — N from 5 to 1000, giving 50 to 2M vars.
        "pypsa" — PyPSA SciGrid-DE with varying snapshot counts.
    """
    results = []

    if model_type == "basic":
        print("\nbasic_model (2 x N^2 vars, 2 x N^2 constraints):")
        for n in [5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]:
            r = benchmark_model(
                f"basic N={n}", basic_model(n), iterations, io_api=io_api
            )
            r["model"] = "basic"
            r["param"] = n
            results.append(r)

    elif model_type == "pypsa":
        print("\nPyPSA SciGrid-DE (realistic power system model):")
        for snaps in [24, 50, 100, 200, 500, 1000]:
            m = pypsa_model(snapshots=snaps)
            if m is not None:
                r = benchmark_model(
                    f"pypsa {snaps} snaps", m, iterations, io_api=io_api
                )
                r["model"] = "pypsa"
                r["param"] = snaps
                results.append(r)
            else:
                print("  (skipped, pypsa not installed)")
                break
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    return results


def plot_comparison(file_old: str, file_new: str) -> None:
    """Create 4-panel comparison plot from two JSON result files."""
    import matplotlib.pyplot as plt

    with open(file_old) as f:
        data_old = json.load(f)
    with open(file_new) as f:
        data_new = json.load(f)

    label_old = data_old.get("label", Path(file_old).stem)
    label_new = data_new.get("label", Path(file_new).stem)

    nv_old = [r["nvars"] for r in data_old["results"]]
    ms_old = [r["mean_s"] * 1000 for r in data_old["results"]]
    std_old = [r["std_s"] * 1000 for r in data_old["results"]]
    nv_new = [r["nvars"] for r in data_new["results"]]
    ms_new = [r["mean_s"] * 1000 for r in data_new["results"]]
    std_new = [r["std_s"] * 1000 for r in data_new["results"]]

    color_old, color_new = "#1f77b4", "#ff7f0e"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"LP Write Performance: {label_old} vs {label_new}", fontsize=14)

    # Panel 1: All data, log-log
    ax = axes[0, 0]
    ax.errorbar(
        nv_old,
        ms_old,
        yerr=std_old,
        marker="o",
        color=color_old,
        linestyle="--",
        label=label_old,
        alpha=0.8,
        capsize=3,
    )
    ax.errorbar(
        nv_new,
        ms_new,
        yerr=std_new,
        marker="s",
        color=color_new,
        linestyle="-",
        label=label_new,
        alpha=0.8,
        capsize=3,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of variables")
    ax.set_ylabel("Write time (ms)")
    ax.set_title("IO time vs problem size (log-log)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Speedup ratio (old/new)
    ax = axes[0, 1]
    if len(nv_old) == len(nv_new):
        speedup = [o / n for o, n in zip(ms_old, ms_new)]
        ax.plot(nv_old, speedup, marker="o", color="#2ca02c")
        ax.fill_between(nv_old, 1.0, speedup, alpha=0.15, color="#2ca02c")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Number of variables")
    ax.set_ylabel(f"Speedup ({label_old} / {label_new})")
    ax.set_title("Speedup vs problem size")
    ax.grid(True, alpha=0.3)

    # Panel 3: Small models (nvars <= 25000)
    ax = axes[1, 0]
    cutoff = 25000
    idx_old = [i for i, n in enumerate(nv_old) if n <= cutoff]
    idx_new = [i for i, n in enumerate(nv_new) if n <= cutoff]
    ax.errorbar(
        [nv_old[i] for i in idx_old],
        [ms_old[i] for i in idx_old],
        yerr=[std_old[i] for i in idx_old],
        marker="o",
        color=color_old,
        linestyle="--",
        label=label_old,
        alpha=0.8,
        capsize=3,
    )
    ax.errorbar(
        [nv_new[i] for i in idx_new],
        [ms_new[i] for i in idx_new],
        yerr=[std_new[i] for i in idx_new],
        marker="s",
        color=color_new,
        linestyle="-",
        label=label_new,
        alpha=0.8,
        capsize=3,
    )
    ax.set_xlabel("Number of variables")
    ax.set_ylabel("Write time (ms)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Small models (≤ {cutoff:,} vars)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Large models (nvars > 25000)
    ax = axes[1, 1]
    idx_old = [i for i, n in enumerate(nv_old) if n > cutoff]
    idx_new = [i for i, n in enumerate(nv_new) if n > cutoff]
    ax.errorbar(
        [nv_old[i] for i in idx_old],
        [ms_old[i] for i in idx_old],
        yerr=[std_old[i] for i in idx_old],
        marker="o",
        color=color_old,
        linestyle="--",
        label=label_old,
        alpha=0.8,
        capsize=3,
    )
    ax.errorbar(
        [nv_new[i] for i in idx_new],
        [ms_new[i] for i in idx_new],
        yerr=[std_new[i] for i in idx_new],
        marker="s",
        color=color_new,
        linestyle="-",
        label=label_new,
        alpha=0.8,
        capsize=3,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Number of variables")
    ax.set_ylabel("Write time (ms)")
    ax.set_title(f"Large models (> {cutoff:,} vars)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "dev-scripts/benchmark_lp_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="LP write benchmark")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument("--label", default=None, help="Label for this run")
    parser.add_argument("--io-api", default=None, help="io_api to pass to to_file()")
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
    print(f"LP file writing benchmark ({iterations} iterations, label={label!r})")
    print("=" * 90)

    results = run_benchmarks(
        io_api=args.io_api, iterations=iterations, model_type=args.model
    )

    output = {"label": label, "results": results}
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        print("\n(use --output FILE to save results for later plotting)")


if __name__ == "__main__":
    main()
