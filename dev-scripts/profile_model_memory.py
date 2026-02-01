"""
Reusable memory profiling script for linopy model building.

Run with scalene for line-level memory attribution:
    scalene run dev-scripts/profile_model_memory.py --preset medium

Run standalone for quick peak RSS + timing:
    python dev-scripts/profile_model_memory.py --shape 100 100 20 --sparsity 0.05 --n-expr 5
"""

import argparse
import json
import resource
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import xarray as xr

import linopy

PRESETS = {
    "small": {"shape": (100, 100, 20), "sparsity": 0.2, "n_expr": 5},
    "medium": {"shape": (200, 200, 50), "sparsity": 0.2, "n_expr": 5},
    "large": {"shape": (300, 300, 100), "sparsity": 0.2, "n_expr": 5},
}


def get_git_info():
    """Return current git branch and short SHA."""
    info = {}
    try:
        info["git_branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        info["git_sha"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        info["git_branch"] = "unknown"
        info["git_sha"] = "unknown"
    return info


def make_coords(shape):
    return [pd.RangeIndex(s, name=f"d{i}") for i, s in enumerate(shape)]


def make_sparse_coeffs(shape, sparsity, rng, dtype=np.float64):
    """Return dense array with (1 - sparsity) fraction set to zero, plus bool mask."""
    mask = rng.random(shape) < sparsity
    vals = rng.standard_normal(shape).astype(dtype)
    vals[~mask] = 0.0
    return vals, mask


def build_model(shape, sparsity, n_expr, seed=42):
    """Build a linopy model with sparse expressions and constraints."""
    rng = np.random.default_rng(seed)
    coords = make_coords(shape)
    dims = [c.name for c in coords]

    m = linopy.Model()

    variables = []
    coeff_arrays = []
    masks = []

    for i in range(n_expr):
        v = m.add_variables(lower=0, coords=coords, name=f"x{i}")
        variables.append(v)
        c, mask = make_sparse_coeffs(shape, sparsity, rng)
        coeff_arrays.append(c)
        masks.append(mask)

    # Build expressions and sum
    exprs = [coeff_arrays[i] * variables[i] for i in range(n_expr)]
    total = sum(exprs)

    # Combined mask
    combined_mask = np.zeros(shape, dtype=bool)
    for mask in masks:
        combined_mask |= mask
    combined_mask_da = xr.DataArray(
        combined_mask, dims=dims, coords={c.name: c for c in coords}
    )

    # Add constraints
    m.add_constraints(total <= 1, name="con", mask=combined_mask_da)

    return m


def get_peak_rss_mb():
    """Get peak RSS in MB via resource module."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS ru_maxrss is in bytes, on Linux it's in KB
    import sys

    if sys.platform == "darwin":
        return ru.ru_maxrss / 1024**2
    return ru.ru_maxrss / 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile linopy model building memory usage."
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        help="Dimensions of the problem, e.g. --shape 100 100 20",
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.05,
        help="Fraction of non-zero coefficients (default: 0.05)",
    )
    parser.add_argument(
        "--n-expr",
        type=int,
        default=5,
        help="Number of expressions to build (default: 5)",
    )
    parser.add_argument(
        "--preset",
        choices=PRESETS.keys(),
        help="Use a preset configuration (overrides --shape, --sparsity, --n-expr)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON results (default: stdout only)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.preset:
        config = PRESETS[args.preset]
        shape = config["shape"]
        sparsity = config["sparsity"]
        n_expr = config["n_expr"]
    elif args.shape:
        shape = tuple(args.shape)
        sparsity = args.sparsity
        n_expr = args.n_expr
    else:
        config = PRESETS["medium"]
        shape = config["shape"]
        sparsity = config["sparsity"]
        n_expr = config["n_expr"]

    total_elements = int(np.prod(shape))
    print(f"Config: shape={shape}  sparsity={sparsity}  n_expr={n_expr}")
    print(f"Total elements: {total_elements:,}")
    print("-" * 60)

    t0 = time.perf_counter()
    build_model(shape, sparsity, n_expr)
    elapsed = time.perf_counter() - t0
    peak_rss = get_peak_rss_mb()

    print(f"Peak RSS:  {peak_rss:.1f} MB")
    print(f"Elapsed:   {elapsed:.2f} s")

    git_info = get_git_info()
    result = {
        **git_info,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {"shape": list(shape), "sparsity": sparsity, "n_expr": n_expr},
        "peak_rss_mb": round(peak_rss, 1),
        "elapsed_s": round(elapsed, 2),
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults written to {args.output}")
    else:
        print(f"\nJSON: {json.dumps(result)}")


if __name__ == "__main__":
    main()
