#!/usr/bin/env python3
"""
Benchmark script for linopy matrix generation and solution-unpacking performance.

Covers the code paths optimised by PRs #616–#619:
  - #616  cached_property on MatrixAccessor (flat_vars / flat_cons)
  - #617  np.char.add for label string concatenation
  - #618  sparse matrix slicing in MatrixAccessor.A
  - #619  numpy solution unpacking in Model.solve

Usage
-----
    # Quick run (24 snapshots only):
    python benchmark/scripts/benchmark_matrix_gen.py --quick

    # Full matrix-generation sweep with JSON output:
    python benchmark/scripts/benchmark_matrix_gen.py -o results.json --label "after-PR-616"

    # Include solution-unpacking benchmark (requires HiGHS solver, #619):
    python benchmark/scripts/benchmark_matrix_gen.py --include-solve -o results.json

    # Compare two runs:
    python benchmark/scripts/benchmark_matrix_gen.py --compare before.json after.json
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def build_scigrid_network(n_snapshots: int):
    """Return a PyPSA Network (SciGrid-DE) with extended snapshots, without building the model."""
    import pypsa

    n = pypsa.examples.scigrid_de()
    orig_snapshots = n.snapshots
    orig_len = len(orig_snapshots)

    new_snapshots = pd.date_range(orig_snapshots[0], periods=n_snapshots, freq="h")
    n.set_snapshots(new_snapshots)

    for component_t in (n.generators_t, n.loads_t, n.storage_units_t):
        for attr in list(component_t):
            df = getattr(component_t, attr)
            if df is not None and not df.empty:
                tiles = int(np.ceil(n_snapshots / orig_len)) + 1
                tiled = np.tile(df.values, (tiles, 1))[:n_snapshots]
                setattr(
                    component_t,
                    attr,
                    pd.DataFrame(
                        tiled,
                        index=new_snapshots,
                        columns=df.columns,
                    ),
                )
    return n


def build_scigrid(n_snapshots: int):
    """Return a linopy Model from PyPSA SciGrid-DE with extended snapshots."""
    n = build_scigrid_network(n_snapshots)
    n.optimize.create_model(include_objective_constant=False)
    return n.model


def build_synthetic(n: int):
    """Return a linopy Model with 2×N² variables and 2×N² constraints."""
    from linopy import Model

    m = Model()
    N = np.arange(n)
    x = m.add_variables(coords=[N, N], name="x")
    y = m.add_variables(coords=[N, N], name="y")
    m.add_constraints(x - y >= N, name="lower")
    m.add_constraints(x + y >= 0, name="upper")
    m.add_objective(2 * x.sum() + y.sum())
    return m


# ---------------------------------------------------------------------------
# Benchmark phases
# ---------------------------------------------------------------------------


def time_phase(func, label: str, repeats: int = 3) -> dict:
    """Time a callable, return best-of-N result."""
    times = []
    for _ in range(repeats):
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        result = func()
        elapsed = time.perf_counter() - t0
        gc.enable()
        times.append(elapsed)
        del result
    return {
        "phase": label,
        "best_s": min(times),
        "median_s": sorted(times)[len(times) // 2],
        "times": times,
    }


def benchmark_model(model, repeats: int = 3) -> list[dict]:
    """Benchmark all matrix generation phases on a linopy Model."""
    results = []
    matrices = model.matrices

    # Phase 1: flat_vars (exercises #616 caching + #617 label vectorisation)
    def do_flat_vars():
        matrices.clean_cached_properties()
        return matrices.flat_vars

    results.append(time_phase(do_flat_vars, "flat_vars", repeats))

    # Phase 2: flat_cons (exercises #616 caching + #617 label vectorisation)
    def do_flat_cons():
        matrices.clean_cached_properties()
        return matrices.flat_cons

    results.append(time_phase(do_flat_cons, "flat_cons", repeats))

    # Phase 3: vlabels + clabels (vector creation from flat data)
    # Ensure flat data is cached first
    matrices.clean_cached_properties()
    _ = matrices.flat_vars
    _ = matrices.flat_cons

    def do_labels():
        return (matrices.vlabels, matrices.clabels)

    results.append(time_phase(do_labels, "vlabels+clabels", repeats))

    # Phase 4: A matrix (exercises #618 sparse slicing)
    def do_A():
        return matrices.A

    results.append(time_phase(do_A, "A_matrix", repeats))

    # Phase 5: full get_matrix_data pipeline (end-to-end)
    def do_full():
        matrices.clean_cached_properties()
        _ = matrices.vlabels
        _ = matrices.clabels
        _ = matrices.lb
        _ = matrices.ub
        _ = matrices.A
        _ = matrices.b
        _ = matrices.sense
        return True

    results.append(time_phase(do_full, "full_pipeline", repeats))

    return results


# ---------------------------------------------------------------------------
# Solution-unpacking benchmark (#619)
# ---------------------------------------------------------------------------


def benchmark_solution_unpack(n_snapshots: int, repeats: int = 3) -> list[dict]:
    """
    Benchmark the solution-assignment loop in Model.solve (PR #619).

    Strategy: solve once with HiGHS to get a real solution vector, then
    re-run only the assignment loop (sol[idx] → var.solution) repeatedly
    without re-solving, isolating the unpacking cost from solver time.
    """
    import xarray as xr

    n = build_scigrid_network(n_snapshots)
    n.optimize.create_model(include_objective_constant=False)
    model = n.model

    # Solve once to populate the raw solution
    status, _ = model.solve(solver_name="highs", io_api="direct")
    if status != "ok":
        print(f"  WARNING: solve failed ({status}), skipping solution-unpack benchmark")
        return []

    # Reconstruct the raw solution Series (as returned by the solver):
    # a float-indexed Series mapping variable label → solution value.
    nan = float("nan")
    parts = []
    for name, var in model.variables.items():
        if var.solution is None:
            continue
        labels = np.ravel(var.labels)
        values = np.ravel(var.solution.values)
        parts.append(pd.Series(values, index=labels.astype(float)))
    if not parts:
        print("  WARNING: no solution found on variables, was solve successful?")
        return []
    sol_series = pd.concat(parts).drop_duplicates()
    sol_series.loc[-1] = nan

    n_vars = sum(
        np.ravel(model.variables[name].labels).size for name in model.variables
    )
    results = []

    # ----- Old path (pandas label-based, pre-#619) -----
    def unpack_pandas():
        for name, var in model.variables.items():
            idx = np.ravel(var.labels).astype(float)
            try:
                vals = sol_series[idx].values.reshape(var.labels.shape)
            except KeyError:
                vals = sol_series.reindex(idx).values.reshape(var.labels.shape)
            var.solution = xr.DataArray(vals, var.coords)

    results.append(time_phase(unpack_pandas, "unpack_pandas (before)", repeats))

    # ----- New path (numpy dense array, #619) -----
    def unpack_numpy():
        sol_max_idx = int(max(sol_series.index.max(), 0))
        sol_arr = np.full(sol_max_idx + 1, nan)
        mask = sol_series.index >= 0
        valid = sol_series.index[mask].astype(int)
        sol_arr[valid] = sol_series.values[mask]
        for name, var in model.variables.items():
            idx = np.ravel(var.labels)
            safe_idx = np.clip(idx, 0, sol_max_idx)
            vals = sol_arr[safe_idx]
            vals[idx < 0] = nan
            var.solution = xr.DataArray(vals.reshape(var.labels.shape), var.coords)

    results.append(time_phase(unpack_numpy, "unpack_numpy (after)", repeats))

    for r in results:
        r.update(model_type="scigrid_solve", size=n_snapshots, n_vars=n_vars, n_cons=0)
    return results


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

QUICK_SNAPSHOTS = [24]
FULL_SNAPSHOTS = [24, 100, 200, 500]
SYNTHETIC_SIZES = [20, 50, 100, 200]


def run_benchmarks(
    model_type: str, quick: bool, repeats: int, include_solve: bool = False
) -> list[dict]:
    """Run benchmarks across problem sizes, return flat list of results."""
    all_results = []

    if model_type in ("scigrid", "all"):
        sizes = QUICK_SNAPSHOTS if quick else FULL_SNAPSHOTS
        for n_snap in sizes:
            print(f"\n{'=' * 60}")
            print(f"SciGrid-DE  {n_snap} snapshots")
            print(f"{'=' * 60}")
            model = build_scigrid(n_snap)
            n_vars = len(model.variables.flat)
            n_cons = len(model.constraints.flat)
            print(f"  {n_vars:,} variables, {n_cons:,} constraints")

            for r in benchmark_model(model, repeats):
                r.update(
                    model_type="scigrid", size=n_snap, n_vars=n_vars, n_cons=n_cons
                )
                all_results.append(r)
                print(
                    f"  {r['phase']:20s}  {r['best_s']:.4f}s  (median {r['median_s']:.4f}s)"
                )

            del model
            gc.collect()

    if include_solve:
        # Solution-unpacking benchmark for PR #619 (SciGrid-DE only, small sizes)
        solve_sizes = QUICK_SNAPSHOTS if quick else [24, 100]
        for n_snap in solve_sizes:
            print(f"\n{'=' * 60}")
            print(f"SciGrid-DE solve + unpack  {n_snap} snapshots  (#619)")
            print(f"{'=' * 60}")
            for r in benchmark_solution_unpack(n_snap, repeats):
                all_results.append(r)
                print(
                    f"  {r['phase']:30s}  {r['best_s']:.4f}s  (median {r['median_s']:.4f}s)"
                )
            gc.collect()

    if model_type in ("synthetic", "all"):
        sizes = [20, 50] if quick else SYNTHETIC_SIZES
        for n in sizes:
            print(f"\n{'=' * 60}")
            print(f"Synthetic  N={n}  ({2 * n * n} vars, {2 * n * n} cons)")
            print(f"{'=' * 60}")
            model = build_synthetic(n)
            n_vars = 2 * n * n
            n_cons = 2 * n * n

            for r in benchmark_model(model, repeats):
                r.update(model_type="synthetic", size=n, n_vars=n_vars, n_cons=n_cons)
                all_results.append(r)
                print(
                    f"  {r['phase']:20s}  {r['best_s']:.4f}s  (median {r['median_s']:.4f}s)"
                )

            del model
            gc.collect()

    return all_results


def format_comparison(before: list[dict], after: list[dict]) -> str:
    """Format a before/after comparison table."""
    df_b = pd.DataFrame(before).set_index(["model_type", "size", "phase"])
    df_a = pd.DataFrame(after).set_index(["model_type", "size", "phase"])
    merged = df_b[["best_s"]].join(
        df_a[["best_s"]], lsuffix="_before", rsuffix="_after"
    )
    merged["speedup"] = merged["best_s_before"] / merged["best_s_after"]
    lines = [
        f"{'Model':>10s} {'Size':>6s} {'Phase':>20s} {'Before':>8s} {'After':>8s} {'Speedup':>8s}",
        "-" * 70,
    ]
    for (mtype, size, phase), row in merged.iterrows():
        lines.append(
            f"{mtype:>10s} {size:>6} {phase:>20s} "
            f"{row['best_s_before']:>7.4f}s {row['best_s_after']:>7.4f}s "
            f"{row['speedup']:>7.2f}x"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark linopy matrix generation (PRs #616–#619)"
    )
    parser.add_argument(
        "--model",
        choices=["scigrid", "synthetic", "all"],
        default="all",
        help="Model type to benchmark (default: all)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick mode (smallest sizes only)"
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Timing repeats per phase (default: 3)"
    )
    parser.add_argument("-o", "--output", type=str, help="Save results to JSON file")
    parser.add_argument("--label", type=str, default="", help="Label for this run")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="Compare two JSON result files instead of running benchmarks",
    )
    parser.add_argument(
        "--include-solve",
        action="store_true",
        help="Also benchmark solution unpacking (PR #619); requires HiGHS solver",
    )
    args = parser.parse_args()

    if args.compare:
        before = json.loads(Path(args.compare[0]).read_text())["results"]
        after = json.loads(Path(args.compare[1]).read_text())["results"]
        print(format_comparison(before, after))
        return

    print("linopy matrix generation benchmark")
    print(
        f"Python {sys.version.split()[0]}, numpy {np.__version__}, "
        f"{platform.machine()}, {platform.system()}"
    )

    results = run_benchmarks(args.model, args.quick, args.repeats, args.include_solve)

    if args.output:
        out = {
            "label": args.label,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": f"{platform.system()} {platform.machine()}",
            "results": results,
        }
        Path(args.output).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nResults saved to {args.output}")

    # Summary table
    print(f"\n{'=' * 60}")
    print("Summary (best times)")
    print(f"{'=' * 60}")
    df = pd.DataFrame(results)
    for (mtype, size), group in df.groupby(["model_type", "size"]):
        n_vars = group.iloc[0]["n_vars"]
        n_cons = group.iloc[0]["n_cons"]
        print(f"\n  {mtype} size={size}  ({n_vars:,} vars, {n_cons:,} cons)")
        for _, row in group.iterrows():
            print(f"    {row['phase']:20s}  {row['best_s']:.4f}s")


if __name__ == "__main__":
    main()
