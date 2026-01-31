#!/usr/bin/env python3
"""
Benchmark LP file writing performance across a series of commits.

Creates git worktrees for each commit, installs linopy, runs benchmarks
in a subprocess, and prints a markdown comparison table.

Usage:
    python dev-scripts/benchmark_lp_writer_commits.py
    python dev-scripts/benchmark_lp_writer_commits.py --commits abc1234 def5678
    python dev-scripts/benchmark_lp_writer_commits.py --baseline master_org
"""

import argparse
import json
import subprocess
import sys
import tempfile
import textwrap

# Default commits: the perf/lp-write-speed branch history
DEFAULT_COMMITS = [
    "master_org",
    "ccb9cd2",
    "8524c29",
    "aab95f5",
    "7762659",
    "bdbb042",
    "44b115f",
    "9ac474b",
]

# The benchmark script that runs inside each worktree
BENCH_SCRIPT = textwrap.dedent("""\
    import json
    import sys
    import tempfile
    import time
    from pathlib import Path

    import numpy as np

    from linopy import Model

    WARMUP = 2
    ITERATIONS = 8


    def basic_model(n):
        m = Model()
        N = np.arange(n)
        x = m.add_variables(coords=[N, N], name="x")
        y = m.add_variables(coords=[N, N], name="y")
        m.add_constraints(x - y >= N, name="c1")
        m.add_constraints(x + y >= 0, name="c2")
        m.add_objective((2 * x).sum() + y.sum())
        return m


    def knapsack_model(n):
        from numpy.random import default_rng
        rng = default_rng(125)
        m = Model()
        packages = m.add_variables(coords=[np.arange(n)], binary=True)
        weight = rng.integers(1, 100, size=n)
        value = rng.integers(1, 100, size=n)
        m.add_constraints((weight * packages).sum() <= 200)
        m.add_objective(-(value * packages).sum())
        return m


    def pypsa_model():
        try:
            import pypsa
        except ImportError:
            return None
        n = pypsa.examples.scigrid_de()
        n.optimize.create_model()
        return n.model


    def pypsa_model_240h():
        try:
            import pypsa
            import pandas as pd
        except ImportError:
            return None
        n = pypsa.examples.scigrid_de()
        n.set_snapshots(pd.date_range('2011-01-01', periods=240, freq='h'))
        n.optimize.create_model()
        return n.model


    def bench(label, m):
        with tempfile.TemporaryDirectory() as tmpdir:
            for _ in range(WARMUP):
                m.to_file(Path(tmpdir) / "warmup.lp", progress=False)
            times = []
            for i in range(ITERATIONS):
                fn = Path(tmpdir) / f"bench_{i}.lp"
                start = time.perf_counter()
                m.to_file(fn, progress=False)
                times.append(time.perf_counter() - start)
        return {"label": label, "mean": float(np.mean(times)), "std": float(np.std(times)),
                "nvars": m.nvars, "ncons": m.ncons}


    results = []
    for n in [50, 100, 200, 500]:
        results.append(bench(f"basic_model(N={n})", basic_model(n)))

    for n in [1000, 10000, 100000]:
        results.append(bench(f"knapsack(N={n})", knapsack_model(n)))

    m = pypsa_model()
    if m is not None:
        results.append(bench("PyPSA scigrid-de 24h", m))

    m = pypsa_model_240h()
    if m is not None:
        results.append(bench("PyPSA scigrid-de 240h", m))

    json.dump(results, sys.stdout)
""")


def resolve_commit(ref: str) -> tuple[str, str]:
    """Return (short_sha, subject) for a git ref."""
    out = subprocess.run(
        ["git", "log", "-1", "--format=%h\t%s", ref],
        capture_output=True,
        text=True,
        check=True,
    )
    sha, subject = out.stdout.strip().split("\t", 1)
    return sha, subject


def run_benchmark_at_commit(ref: str) -> list[dict]:
    """Checkout commit in a worktree, install, run benchmark, return results."""
    sha, subject = resolve_commit(ref)
    print(f"\n{'=' * 70}", file=sys.stderr)
    print(f"Benchmarking: {sha} {subject}", file=sys.stderr)
    print(f"{'=' * 70}", file=sys.stderr)

    with tempfile.TemporaryDirectory() as worktree_dir:
        # Create worktree
        subprocess.run(
            ["git", "worktree", "add", "--detach", worktree_dir, ref],
            check=True,
            capture_output=True,
        )
        try:
            # Install in current environment (non-editable + force to ensure clean)
            print(f"  Installing linopy from {sha}...", file=sys.stderr)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    worktree_dir,
                    "-q",
                    "--no-deps",
                    "--force-reinstall",
                ],
                check=True,
                capture_output=True,
            )

            # Run benchmark in subprocess (fresh import, cwd=/ to avoid
            # importing linopy from the repo working directory)
            print("  Running benchmarks...", file=sys.stderr)
            result = subprocess.run(
                [sys.executable, "-c", BENCH_SCRIPT],
                capture_output=True,
                text=True,
                cwd="/",
            )
            if result.returncode != 0:
                print(f"  FAILED! stderr:\n{result.stderr}", file=sys.stderr)
                return []
            return json.loads(result.stdout)
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", worktree_dir],
                capture_output=True,
            )


def main():
    parser = argparse.ArgumentParser(description="Benchmark LP writer across commits")
    parser.add_argument(
        "--commits",
        nargs="+",
        default=DEFAULT_COMMITS,
        help="Git refs to benchmark (first is baseline)",
    )
    args = parser.parse_args()

    commits = args.commits

    # Collect results: {commit_ref: {label: {mean, std, ...}}}
    all_results: dict[str, dict[str, dict]] = {}
    commit_info: dict[str, tuple[str, str]] = {}  # ref -> (sha, subject)

    for ref in commits:
        sha, subject = resolve_commit(ref)
        commit_info[ref] = (sha, subject)
        results = run_benchmark_at_commit(ref)
        all_results[ref] = {r["label"]: r for r in results}

    # Reinstall current version
    print("\nReinstalling current worktree linopy...", file=sys.stderr)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            ".",
            "-q",
            "--no-deps",
            "--force-reinstall",
        ],
        capture_output=True,
    )

    # Get benchmark labels from first commit that has results
    labels = list(next(iter(all_results.values())).keys())

    # Print markdown table per benchmark
    baseline_ref = commits[0]
    print()
    for label in labels:
        baseline_data = all_results[baseline_ref].get(label)
        if not baseline_data:
            continue

        nvars = baseline_data["nvars"]
        ncons = baseline_data["ncons"]
        print(f"### {label} ({nvars:,} vars, {ncons:,} cons)\n")
        print("| Commit | Description | Time (ms) | Δ vs prev | Δ vs baseline |")
        print("|--------|-------------|-----------|-----------|---------------|")

        prev_mean = None
        baseline_mean = baseline_data["mean"]

        for ref in commits:
            sha, subject = commit_info[ref]
            data = all_results[ref].get(label)
            if not data:
                continue

            mean_ms = data["mean"] * 1000
            std_ms = data["std"] * 1000

            # Delta vs previous
            if prev_mean is not None:
                delta_prev = (data["mean"] - prev_mean) / prev_mean * 100
                delta_prev_str = f"{delta_prev:+.1f}%"
            else:
                delta_prev_str = "—"

            # Delta vs baseline
            delta_base = (data["mean"] - baseline_mean) / baseline_mean * 100
            delta_base_str = f"{delta_base:+.1f}%"

            print(
                f"| `{sha}` | {subject[:40]:40s} | "
                f"{mean_ms:7.1f} ± {std_ms:4.1f} | "
                f"{delta_prev_str:>9s} | {delta_base_str:>13s} |"
            )

            prev_mean = data["mean"]

        print()


if __name__ == "__main__":
    main()
