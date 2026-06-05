#!/usr/bin/env python3
"""Same-machine build-only benchmark: polar-high vs linopy (lp / direct).

Context: PyPSA/linopy#740. polar-high's published dense-LP numbers and
linopy's were measured on different hardware. FBumann asked for all tools
in one run on one machine. This script builds the *same* dense LP with each
tool, time-limits HiGHS to ~0 so the wall clock is dominated by the
modelling + IO path, and captures peak RSS per subprocess.

The four columns:
  linopy-lp      : linopy build -> .lp text file -> HiGHS reparse
  linopy-direct  : linopy build -> in-memory highspy load (no disk)
  polar          : polar-high (regular)   -> highspy
  polar-sm       : polar-high save_memory -> highspy (MPS roundtrip, no warm reuse)

All tools share the same env (one HiGHS), same machine, same N sweep.

Two roles in one file:
  worker : build + solve one (tool, N), print timing JSON to stdout.
  runner : spawn each worker under `/usr/bin/time -l` for peak RSS,
           collect results, write CSV + print a table.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time

# HiGHS time limit (seconds). Tiny but nonzero so the solve returns at the
# limit, isolating the build/IO cost. polar-high's headline used ~1e-6;
# build dominates either way. Held identical across all tools.
SOLVE_TIME_LIMIT = 1e-3

TOOLS = ("linopy-lp", "linopy-direct", "polar", "polar-sm")


# --------------------------------------------------------------------------- #
# Model builders. Both express the identical dense LP (2*n^2 vars, 2*n^2 rows):
#   min  sum_{i,j} (2 x[i,j] + y[i,j])
#   s.t. x - y >= i ; x + y >= 0 ; x,y >= 0
# --------------------------------------------------------------------------- #
def linopy_build(n: int):
    """linopy's own benchmark dense LP (benchmark_linopy.basic_model)."""
    from numpy import arange

    from linopy import Model

    m = Model()
    N, M = arange(n), arange(n)
    x = m.add_variables(coords=[N, M])
    y = m.add_variables(coords=[N, M])
    m.add_constraints(x - y >= N)
    m.add_constraints(x + y >= 0)
    m.add_objective(2 * x.sum() + y.sum())
    return m


def polar_build(n: int):
    """polar-high build, copied verbatim from polar-high/benchmark/models/polar.py."""
    import numpy as np
    import polars as pl

    from polar_high import Param, Problem, Sum

    p = Problem()
    i_arr = np.repeat(np.arange(1, n + 1, dtype=np.int64), n)
    j_arr = np.tile(np.arange(1, n + 1, dtype=np.int64), n)
    idx = pl.DataFrame({"i": i_arr, "j": j_arr})

    x = p.add_var("x", dims=("i", "j"), index=idx, lower=0.0)
    y = p.add_var("y", dims=("i", "j"), index=idx, lower=0.0)

    obj = x.to_expr() * 2.0 + y.to_expr() * 1.0
    p.set_objective(Sum(obj), sense="min")

    rhs_i = Param(
        ("i",),
        pl.DataFrame(
            {
                "i": np.arange(1, n + 1, dtype=np.int64),
                "value": np.arange(1, n + 1, dtype=np.float64),
            }
        ),
    )
    p.add_cstr(
        "c1",
        over=idx,
        sense=">=",
        lhs_terms={"x": x, "neg_y": -y.to_expr()},
        rhs_terms={"i": rhs_i},
    )
    p.add_cstr(
        "c2",
        over=idx,
        sense=">=",
        lhs_terms={"x": x, "y": y},
        rhs_terms={"zero": 0.0},
    )
    return p


def run_worker(n: int, tool: str) -> dict:
    if tool not in TOOLS:
        raise ValueError(f"unknown tool {tool!r}, expected one of {TOOLS}")

    version = None
    if tool.startswith("linopy"):
        import linopy

        version = linopy.__version__
        io_api = "lp" if tool == "linopy-lp" else "direct"

        t0 = time.perf_counter()
        m = linopy_build(n)
        t_build = time.perf_counter() - t0

        t1 = time.perf_counter()
        m.solve("highs", io_api=io_api, time_limit=SOLVE_TIME_LIMIT)
        t_solve = time.perf_counter() - t1
    else:
        import importlib.metadata as md

        version = md.version("polar-high")
        save_memory = tool == "polar-sm"

        t0 = time.perf_counter()
        p = polar_build(n)
        t_build = time.perf_counter() - t0

        t1 = time.perf_counter()
        p.solve(options={"time_limit": SOLVE_TIME_LIMIT}, save_memory=save_memory)
        t_solve = time.perf_counter() - t1

    return {
        "tool": tool,
        "N": n,
        "n_vars": 2 * n * n,
        "build_s": round(t_build, 3),
        "solve_s": round(t_solve, 3),
        "total_s": round(t_build + t_solve, 3),
        "version": version,
    }


_RSS_RE = re.compile(r"(\d+)\s+maximum resident set size")


def parse_peak_rss_bytes(time_stderr: str) -> int | None:
    """Parse `/usr/bin/time -l` (macOS) maximum resident set size, in bytes."""
    m = _RSS_RE.search(time_stderr)
    if m:
        return int(m.group(1))  # macOS reports bytes
    m = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", time_stderr)
    if m:
        return int(m.group(1)) * 1024  # GNU time -v fallback
    return None


def run_one(n: int, tool: str) -> dict:
    cmd = [
        "/usr/bin/time",
        "-l",
        sys.executable,
        __file__,
        "--worker",
        "--n",
        str(n),
        "--tool",
        tool,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if not out_lines:
        raise RuntimeError(
            f"worker produced no output (N={n}, tool={tool}).\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )
    result = json.loads(out_lines[-1])
    peak = parse_peak_rss_bytes(proc.stderr)
    result["peak_rss_gb"] = round(peak / 1e9, 3) if peak else None
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--n", type=int)
    p.add_argument("--tool", type=str)
    p.add_argument(
        "--nrange",
        type=str,
        default="500,1000,1500,2000,2500,3000",
        help="comma-separated N values for the runner sweep",
    )
    p.add_argument(
        "--tools",
        type=str,
        default=",".join(TOOLS),
        help="comma-separated tools to run",
    )
    p.add_argument("--out", type=str, default="tool_compare_results.csv")
    args = p.parse_args()

    if args.worker:
        print(json.dumps(run_worker(args.n, args.tool)))
        return

    nrange = [int(x) for x in args.nrange.split(",") if x.strip()]
    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    rows: list[dict] = []
    print(f"platform: {platform.platform()}  python: {platform.python_version()}")
    print(f"sweep N={nrange}  tools={tools}  time_limit={SOLVE_TIME_LIMIT}s\n")
    hdr = (
        f"{'tool':>14} {'N':>6} {'n_vars':>10} "
        f"{'build_s':>8} {'solve_s':>8} {'total_s':>8} {'peak_GB':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for n in nrange:
        for tool in tools:
            r = run_one(n, tool)
            rows.append(r)
            peak = r["peak_rss_gb"] if r["peak_rss_gb"] is not None else float("nan")
            print(
                f"{r['tool']:>14} {r['N']:>6} {r['n_vars']:>10} "
                f"{r['build_s']:>8} {r['solve_s']:>8} {r['total_s']:>8} {peak:>8}"
            )

    import csv

    cols = [
        "tool",
        "N",
        "n_vars",
        "build_s",
        "solve_s",
        "total_s",
        "peak_rss_gb",
        "version",
    ]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
