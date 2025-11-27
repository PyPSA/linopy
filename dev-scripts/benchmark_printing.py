#!/usr/bin/env python3
"""
User-story benchmark for linopy printing functions.

This benchmark compares OLD (O(n) linear search) vs NEW (O(log n) binary search)
implementations from a user's perspective with realistic workflows.

User Stories:
-------------
1. "I want to inspect my model" -> print(model) / repr(model)
2. "I want to see all my variables" -> print(model.variables)
3. "I want to see all my constraints" -> print(model.constraints)
4. "I want to inspect a single variable array" -> print(model.variables["x"])
5. "I want to inspect a single constraint" -> print(model.constraints["con0"])
6. "I want to look up specific variable labels" -> variables.print_labels([...])
7. "I want to look up specific constraint labels" -> constraints.print_labels([...])
8. "I want to see an expression" -> print(x + y)
9. "I want to see the objective" -> print(model.objective)

Results are stored in an xarray Dataset for analysis.

Usage:
    python dev-scripts/benchmark_printing.py [--vars N] [--cons N] [--repeats N]
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import Any

import numpy as np
import xarray as xr

from linopy import Model
from linopy.common import get_label_position


def build_model(n_vars: int, n_cons: int, terms_per_con: int = 8) -> Model:
    """Build a model with specified number of variables and constraints."""
    rng = np.random.default_rng(42)
    m = Model()

    n_vars_per_dim = int(np.sqrt(n_vars)) + 1
    x = m.add_variables(
        lower=0, upper=100, name="x",
        coords=[range(n_vars_per_dim), range(n_vars_per_dim)],
    )
    y = m.add_variables(
        lower=-50, upper=50, name="y",
        coords=[range(n_vars_per_dim), range(n_vars_per_dim)],
    )

    for i in range(n_cons):
        var_indices = rng.integers(0, n_vars_per_dim, size=(terms_per_con, 2))
        coeffs = rng.uniform(-10, 10, size=terms_per_con)
        lhs = sum(
            coeffs[j] * (x if j % 2 == 0 else y).isel(
                dim_0=var_indices[j, 0], dim_1=var_indices[j, 1]
            )
            for j in range(terms_per_con)
        )
        m.add_constraints(lhs >= rng.uniform(-100, 100), name=f"con{i}")

    # Add an objective
    m.objective = (x.sum() + y.sum()).sum()

    return m


def time_function(func: Callable[[], Any], repeats: int, warmup: int = 2) -> Iterable[float]:
    """Time a function over multiple iterations."""
    for _ in range(warmup):
        func()
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        yield time.perf_counter() - start


def suppress_output(func: Callable[[], Any]) -> Callable[[], Any]:
    """Wrapper to suppress stdout output from a function."""
    def wrapper():
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            return func()
        finally:
            sys.stdout = old_stdout
    return wrapper


@contextmanager
def use_original_implementation():
    """
    Context manager to temporarily use the original O(n) get_label_position.

    Monkey-patches Variables and Constraints to use the original implementation.
    """
    import linopy.constraints as constraints_module
    import linopy.variables as variables_module

    # Store optimized methods
    optimized_var_method = variables_module.Variables.get_label_position
    optimized_con_method = constraints_module.Constraints.get_label_position

    # Replace with original O(n) implementation
    def original_var_get_label_position(self, values):
        return get_label_position(self, values)

    def original_con_get_label_position(self, values):
        return get_label_position(self, values)

    variables_module.Variables.get_label_position = original_var_get_label_position
    constraints_module.Constraints.get_label_position = original_con_get_label_position

    try:
        yield
    finally:
        # Restore optimized methods
        variables_module.Variables.get_label_position = optimized_var_method
        constraints_module.Constraints.get_label_position = optimized_con_method


def run_benchmarks(model: Model, repeats: int) -> xr.Dataset:
    """
    Run user-story benchmarks comparing original vs optimized.

    Returns an xarray Dataset with timing results for both implementations.
    """
    results = {}

    # Prepare test data
    rng = np.random.default_rng(123)
    n_label_lookups = 20
    var_labels = rng.integers(0, model._xCounter, size=n_label_lookups).tolist()
    con_labels = rng.integers(0, model._cCounter, size=n_label_lookups).tolist()

    x = model.variables["x"]
    first_con_name = list(model.constraints)[0]
    con = model.constraints[first_con_name]

    # Define user-story benchmark operations
    user_stories = {
        # Story 1: Inspect the full model
        "print_model": {
            "func": lambda: repr(model),
            "description": "print(model) - inspect full model",
            "story": "I want to inspect my model",
        },
        # Story 2: See all variables (container repr)
        "print_all_variables": {
            "func": lambda: repr(model.variables),
            "description": "print(model.variables) - list all variable arrays",
            "story": "I want to see all my variables",
        },
        # Story 3: See all constraints (container repr)
        "print_all_constraints": {
            "func": lambda: repr(model.constraints),
            "description": "print(model.constraints) - list all constraint arrays",
            "story": "I want to see all my constraints",
        },
        # Story 4: Inspect a single variable array
        "print_single_variable_array": {
            "func": lambda: repr(x),
            "description": "print(model.variables['x']) - inspect variable array",
            "story": "I want to inspect a single variable array",
        },
        # Story 5: Inspect a single constraint array
        "print_single_constraint_array": {
            "func": lambda: repr(con),
            "description": f"print(model.constraints['{first_con_name}']) - inspect constraint",
            "story": "I want to inspect a single constraint",
        },
        # Story 6: Look up specific variable labels
        "lookup_variable_labels": {
            "func": suppress_output(lambda: model.variables.print_labels(var_labels)),
            "description": f"variables.print_labels({n_label_lookups} labels)",
            "story": "I want to look up specific variable labels",
        },
        # Story 7: Look up specific constraint labels
        "lookup_constraint_labels": {
            "func": suppress_output(lambda: model.constraints.print_labels(con_labels)),
            "description": f"constraints.print_labels({n_label_lookups} labels)",
            "story": "I want to look up specific constraint labels",
        },
        # Story 8: See an expression
        "print_expression": {
            "func": lambda: repr(x.sum()),
            "description": "print(x.sum()) - inspect expression",
            "story": "I want to see an expression",
        },
        # Story 9: See the objective
        "print_objective": {
            "func": lambda: repr(model.objective),
            "description": "print(model.objective) - inspect objective",
            "story": "I want to see the objective",
        },
    }

    # Run benchmarks for both implementations
    for impl in ["original", "optimized"]:
        if impl == "original":
            ctx = use_original_implementation()
        else:
            ctx = contextmanager(lambda: (yield))()

        with ctx:
            for op_name, op_info in user_stories.items():
                times = np.fromiter(
                    time_function(op_info["func"], repeats), dtype=float
                )
                key = f"{op_name}_{impl}"
                results[key] = xr.DataArray(
                    times,
                    dims=["repeat"],
                    coords={"repeat": range(repeats)},
                    attrs={
                        "description": op_info["description"],
                        "story": op_info["story"],
                    },
                )

    # Create dataset
    ds = xr.Dataset(results)
    ds.attrs["n_variables"] = model._xCounter
    ds.attrs["n_constraints"] = model._cCounter
    ds.attrs["n_variable_arrays"] = len(list(model.variables))
    ds.attrs["n_constraint_arrays"] = len(list(model.constraints))
    ds.attrs["n_label_lookups"] = n_label_lookups

    return ds


def compute_summary(ds: xr.Dataset) -> xr.Dataset:
    """Compute summary statistics."""
    stats = ["median_ms", "mean_ms", "std_ms"]
    data = {}

    for var_name in ds.data_vars:
        times = ds[var_name].values * 1000  # Convert to ms
        data[var_name] = xr.DataArray(
            [np.median(times), np.mean(times), np.std(times)],
            dims=["stat"],
            coords={"stat": stats},
        )

    summary = xr.Dataset(data)
    summary.attrs = ds.attrs.copy()
    return summary


def print_results(ds: xr.Dataset, summary: xr.Dataset) -> None:
    """Print benchmark results comparing original vs optimized."""
    print("\n" + "=" * 90)
    print("USER-STORY BENCHMARK: Original vs Optimized")
    print("=" * 90)

    print(
        f"\nModel: {ds.attrs['n_variables']} variables, "
        f"{ds.attrs['n_constraints']} constraints"
    )
    print(
        f"       {ds.attrs['n_variable_arrays']} variable arrays, "
        f"{ds.attrs['n_constraint_arrays']} constraint arrays\n"
    )

    # Extract operation names
    all_vars = list(ds.data_vars)
    operations = sorted(set(v.rsplit("_", 1)[0] for v in all_vars if v.endswith("_original")))

    # Group by user story category
    categories = {
        "Model Inspection": ["print_model"],
        "Container Listing": ["print_all_variables", "print_all_constraints"],
        "Single Array Inspection": ["print_single_variable_array", "print_single_constraint_array"],
        "Label Lookup": ["lookup_variable_labels", "lookup_constraint_labels"],
        "Expression/Objective": ["print_expression", "print_objective"],
    }

    total_orig = 0.0
    total_opt = 0.0

    for category, ops in categories.items():
        print(f"\n{category}:")
        print("-" * 90)
        print(f"  {'User Story':<45s} {'Original':>12s} {'Optimized':>12s} {'Speedup':>10s}")
        print("-" * 90)

        for op in ops:
            if op not in operations:
                continue

            orig_key = f"{op}_original"
            opt_key = f"{op}_optimized"

            if orig_key in summary.data_vars and opt_key in summary.data_vars:
                orig_ms = float(summary[orig_key].sel(stat="median_ms"))
                opt_ms = float(summary[opt_key].sel(stat="median_ms"))
                speedup = orig_ms / opt_ms if opt_ms > 0 else float("inf")

                total_orig += orig_ms
                total_opt += opt_ms

                story = ds[orig_key].attrs.get("story", op)
                # Truncate long stories
                if len(story) > 43:
                    story = story[:40] + "..."
                print(f"  {story:<45s} {orig_ms:>10.2f}ms {opt_ms:>10.2f}ms {speedup:>9.1f}x")

    print("\n" + "=" * 90)
    total_speedup = total_orig / total_opt if total_opt > 0 else float("inf")
    print(f"  {'TOTAL':<45s} {total_orig:>10.2f}ms {total_opt:>10.2f}ms {total_speedup:>9.1f}x")

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"\n  The optimized implementation is {total_speedup:.1f}x faster overall.")
    print(f"  Total time reduced from {total_orig:.1f}ms to {total_opt:.1f}ms.")

    # Highlight biggest improvements
    print("\n  Biggest improvements:")
    improvements = []
    for op in operations:
        orig_key = f"{op}_original"
        opt_key = f"{op}_optimized"
        if orig_key in summary.data_vars and opt_key in summary.data_vars:
            orig_ms = float(summary[orig_key].sel(stat="median_ms"))
            opt_ms = float(summary[opt_key].sel(stat="median_ms"))
            speedup = orig_ms / opt_ms if opt_ms > 0 else float("inf")
            story = ds[orig_key].attrs.get("story", op)
            improvements.append((speedup, story, orig_ms, opt_ms))

    improvements.sort(reverse=True)
    for speedup, story, orig_ms, opt_ms in improvements[:3]:
        if speedup > 1.1:  # Only show meaningful improvements
            print(f"    - {story}: {speedup:.1f}x faster ({orig_ms:.1f}ms -> {opt_ms:.1f}ms)")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vars", type=int, default=1000, help="Number of variables")
    parser.add_argument("--cons", type=int, default=500, help="Number of constraints")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repetitions")
    parser.add_argument("--output", type=str, default=None, help="Output NetCDF file")
    args = parser.parse_args()

    print("Building model...")
    model = build_model(args.vars, args.cons)

    print("Running user-story benchmarks (original vs optimized)...")
    ds = run_benchmarks(model, args.repeats)
    summary = compute_summary(ds)

    print_results(ds, summary)

    if args.output:
        xr.merge([ds, summary]).to_netcdf(args.output)
        print(f"Results saved to {args.output}")

    return ds, summary


if __name__ == "__main__":
    main()
