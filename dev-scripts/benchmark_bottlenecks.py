#!/usr/bin/env python3
"""
Bottleneck analysis for linopy printing functions.

This script identifies performance bottlenecks by measuring individual methods:
- get_label_position (original O(n) vs optimized O(log n))
- .sel() calls on xarray objects
- String formatting operations

Results are stored in an xarray Dataset for analysis.

Usage:
    python dev-scripts/benchmark_bottlenecks.py [--vars N] [--cons N] [--repeats N]
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import xarray as xr

from linopy import Model
from linopy.common import (
    get_label_position,
    print_coord,
)


def build_model(n_vars: int, n_cons: int, terms_per_con: int = 5) -> Model:
    """Build a model with specified number of variables and constraints."""
    rng = np.random.default_rng(42)
    m = Model()

    n_vars_per_dim = int(np.sqrt(n_vars)) + 1
    x = m.add_variables(
        lower=0,
        upper=100,
        name="x",
        coords=[range(n_vars_per_dim), range(n_vars_per_dim)],
    )
    y = m.add_variables(
        lower=-50,
        upper=50,
        name="y",
        coords=[range(n_vars_per_dim), range(n_vars_per_dim)],
    )

    for i in range(n_cons):
        var_indices = rng.integers(0, n_vars_per_dim, size=(terms_per_con, 2))
        coeffs = rng.uniform(-10, 10, size=terms_per_con)
        lhs = sum(
            coeffs[j]
            * (x if j % 2 == 0 else y).isel(
                dim_0=var_indices[j, 0], dim_1=var_indices[j, 1]
            )
            for j in range(terms_per_con)
        )
        m.add_constraints(lhs >= rng.uniform(-100, 100), name=f"con{i}")

    return m


def time_function(
    func: Callable[[], Any], repeats: int, warmup: int = 2
) -> Iterable[float]:
    """Time a function over multiple iterations."""
    for _ in range(warmup):
        func()
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        yield time.perf_counter() - start


def run_bottleneck_analysis(model: Model, n_lookups: int, repeats: int) -> xr.Dataset:
    """
    Run bottleneck analysis on individual methods.

    Returns an xarray Dataset with timing results for each method.
    """
    rng = np.random.default_rng(123)
    variables = model.variables
    constraints = model.constraints

    max_var_label = model._xCounter
    max_con_label = model._cCounter
    var_labels = rng.integers(0, max_var_label, size=n_lookups)
    con_labels = rng.integers(0, max_con_label, size=n_lookups)

    # Pre-compute positions for breakdown tests
    con_positions = [constraints.get_label_position(int(l)) for l in con_labels]
    var_positions = [variables.get_label_position(int(l)) for l in var_labels]

    # Collect variable labels from constraints for nested lookup test
    nested_var_labels = []
    for name, coord in con_positions:
        con = constraints[name]
        vars_arr = con.vars.sel(coord).values.flatten()
        nested_var_labels.extend(vars_arr[vars_arr != -1].tolist())

    results = {}

    # 1. get_label_position - Original O(n) implementation
    def bench_original_var():
        return [get_label_position(variables, int(l)) for l in var_labels]

    times = np.fromiter(time_function(bench_original_var, repeats), dtype=float)
    results["get_label_position_vars_original"] = xr.DataArray(
        times, dims=["repeat"], attrs={"n_operations": n_lookups, "complexity": "O(n)"}
    )

    def bench_original_con():
        return [get_label_position(constraints, int(l)) for l in con_labels]

    times = np.fromiter(time_function(bench_original_con, repeats), dtype=float)
    results["get_label_position_cons_original"] = xr.DataArray(
        times, dims=["repeat"], attrs={"n_operations": n_lookups, "complexity": "O(n)"}
    )

    # 2. get_label_position - Optimized O(log n) implementation (built-in)
    def bench_optimized_var():
        return [variables.get_label_position(int(l)) for l in var_labels]

    times = np.fromiter(time_function(bench_optimized_var, repeats), dtype=float)
    results["get_label_position_vars_optimized"] = xr.DataArray(
        times,
        dims=["repeat"],
        attrs={"n_operations": n_lookups, "complexity": "O(log n)"},
    )

    def bench_optimized_con():
        return [constraints.get_label_position(int(l)) for l in con_labels]

    times = np.fromiter(time_function(bench_optimized_con, repeats), dtype=float)
    results["get_label_position_cons_optimized"] = xr.DataArray(
        times,
        dims=["repeat"],
        attrs={"n_operations": n_lookups, "complexity": "O(log n)"},
    )

    # 3. .sel() calls on constraints
    def bench_sel_calls():
        for name, coord in con_positions:
            con = constraints[name]
            _ = con.coeffs.sel(coord).values
            _ = con.vars.sel(coord).values
            _ = con.sign.sel(coord).item()
            _ = con.rhs.sel(coord).item()

    times = np.fromiter(time_function(bench_sel_calls, repeats), dtype=float)
    results["sel_calls_constraint"] = xr.DataArray(
        times,
        dims=["repeat"],
        attrs={
            "n_operations": n_lookups,
            "description": "4 sel() calls per constraint",
        },
    )

    # 4. .sel() calls on variables
    def bench_sel_var():
        for name, coord in var_positions:
            var = variables[name]
            _ = var.lower.sel(coord).item()
            _ = var.upper.sel(coord).item()

    times = np.fromiter(time_function(bench_sel_var, repeats), dtype=float)
    results["sel_calls_variable"] = xr.DataArray(
        times,
        dims=["repeat"],
        attrs={"n_operations": n_lookups, "description": "2 sel() calls per variable"},
    )

    # 5. Nested variable lookups (as in print_single_constraint)
    def bench_nested_var_lookup():
        return [variables.get_label_position(int(l)) for l in nested_var_labels]

    times = np.fromiter(time_function(bench_nested_var_lookup, repeats), dtype=float)
    results["nested_var_lookup"] = xr.DataArray(
        times,
        dims=["repeat"],
        attrs={
            "n_operations": len(nested_var_labels),
            "description": "Variable lookups from constraint terms",
        },
    )

    # 6. print_coord formatting
    coords_list = [{"dim_0": i, "dim_1": j} for i in range(10) for j in range(10)]

    def bench_print_coord():
        return [print_coord(c) for c in coords_list]

    times = np.fromiter(time_function(bench_print_coord, repeats), dtype=float)
    results["print_coord"] = xr.DataArray(
        times,
        dims=["repeat"],
        attrs={
            "n_operations": len(coords_list),
            "description": "Coordinate formatting",
        },
    )

    # 7. String formatting
    def bench_string_format():
        for name, coord in con_positions:
            _ = f"{name}{print_coord(coord)}: expr >= 0.0"

    times = np.fromiter(time_function(bench_string_format, repeats), dtype=float)
    results["string_formatting"] = xr.DataArray(
        times,
        dims=["repeat"],
        attrs={"n_operations": n_lookups, "description": "f-string formatting"},
    )

    # Create dataset
    ds = xr.Dataset(results)
    ds.attrs["n_variables"] = model._xCounter
    ds.attrs["n_constraints"] = model._cCounter
    ds.attrs["n_variable_arrays"] = len(list(variables))
    ds.attrs["n_constraint_arrays"] = len(list(constraints))
    ds.attrs["n_lookups"] = n_lookups
    ds.attrs["n_nested_var_labels"] = len(nested_var_labels)

    return ds


def compute_summary(ds: xr.Dataset) -> xr.Dataset:
    """Compute summary statistics."""
    stats = ["median_s", "mean_s", "std_s", "per_op_us"]
    data = {}

    for var_name in ds.data_vars:
        times = ds[var_name].values
        n_ops = ds[var_name].attrs.get("n_operations", 1)
        data[var_name] = xr.DataArray(
            [
                np.median(times),
                np.mean(times),
                np.std(times),
                (np.median(times) / n_ops) * 1e6,
            ],
            dims=["stat"],
            coords={"stat": stats},
        )

    summary = xr.Dataset(data)
    summary.attrs = ds.attrs.copy()
    return summary


def print_analysis(ds: xr.Dataset, summary: xr.Dataset) -> None:
    """Print bottleneck analysis results."""
    print("\n" + "=" * 80)
    print("BOTTLENECK ANALYSIS")
    print("=" * 80)

    print(
        f"\nModel: {ds.attrs['n_variables']} variables, {ds.attrs['n_constraints']} constraints"
    )
    print(
        f"       {ds.attrs['n_variable_arrays']} variable arrays, {ds.attrs['n_constraint_arrays']} constraint arrays"
    )
    print(f"       {ds.attrs['n_lookups']} lookups per benchmark\n")

    # get_label_position comparison
    print("get_label_position Performance:")
    print("-" * 80)
    print(f"  {'Method':<45s} {'Time':>12s} {'Per Op':>12s} {'Complexity':<10s}")
    print("-" * 80)

    for impl in ["original", "optimized"]:
        for target in ["vars", "cons"]:
            key = f"get_label_position_{target}_{impl}"
            if key in summary.data_vars:
                median_ms = float(summary[key].sel(stat="median_s")) * 1000
                per_op = float(summary[key].sel(stat="per_op_us"))
                complexity = ds[key].attrs.get("complexity", "")
                print(
                    f"  {key:<45s} {median_ms:>10.2f}ms {per_op:>10.2f}µs {complexity:<10s}"
                )

    # Speedup calculation
    print("\nSpeedup (Optimized vs Original):")
    print("-" * 80)
    for target in ["vars", "cons"]:
        orig_key = f"get_label_position_{target}_original"
        opt_key = f"get_label_position_{target}_optimized"
        if orig_key in summary.data_vars and opt_key in summary.data_vars:
            orig = float(summary[orig_key].sel(stat="median_s"))
            opt = float(summary[opt_key].sel(stat="median_s"))
            speedup = orig / opt if opt > 0 else float("inf")
            print(f"  {target}: {speedup:.1f}x faster")

    # Other operations
    print("\nOther Operations:")
    print("-" * 80)
    print(f"  {'Operation':<45s} {'Time':>12s} {'Per Op':>12s}")
    print("-" * 80)

    other_ops = [
        "sel_calls_constraint",
        "sel_calls_variable",
        "nested_var_lookup",
        "print_coord",
        "string_formatting",
    ]
    for key in other_ops:
        if key in summary.data_vars:
            median_ms = float(summary[key].sel(stat="median_s")) * 1000
            per_op = float(summary[key].sel(stat="per_op_us"))
            print(f"  {key:<45s} {median_ms:>10.2f}ms {per_op:>10.2f}µs")

    # Bottleneck identification
    print("\n" + "=" * 80)
    print("BOTTLENECK IDENTIFICATION")
    print("=" * 80)

    # Calculate relative contributions (using original implementation)
    orig_con = float(summary["get_label_position_cons_original"].sel(stat="median_s"))
    sel_con = float(summary["sel_calls_constraint"].sel(stat="median_s"))
    nested = float(summary["nested_var_lookup"].sel(stat="median_s"))
    fmt = float(summary["string_formatting"].sel(stat="median_s"))

    total = orig_con + sel_con + nested + fmt
    print("\nprint_single_constraint breakdown (with original get_label_position):")
    print(
        f"  Constraint lookup:     {orig_con * 1000:>8.2f}ms ({orig_con / total * 100:>5.1f}%)"
    )
    print(
        f"  .sel() calls:          {sel_con * 1000:>8.2f}ms ({sel_con / total * 100:>5.1f}%)"
    )
    print(
        f"  Nested var lookups:    {nested * 1000:>8.2f}ms ({nested / total * 100:>5.1f}%)"
    )
    print(f"  String formatting:     {fmt * 1000:>8.2f}ms ({fmt / total * 100:>5.1f}%)")
    print(f"  Total:                 {total * 1000:>8.2f}ms")

    # With optimized implementation
    opt_con = float(summary["get_label_position_cons_optimized"].sel(stat="median_s"))
    total_opt = opt_con + sel_con + nested + fmt
    print("\nWith optimized get_label_position:")
    print(
        f"  Constraint lookup:     {opt_con * 1000:>8.2f}ms ({opt_con / total_opt * 100:>5.1f}%)"
    )
    print(
        f"  .sel() calls:          {sel_con * 1000:>8.2f}ms ({sel_con / total_opt * 100:>5.1f}%)"
    )
    print(
        f"  Nested var lookups:    {nested * 1000:>8.2f}ms ({nested / total_opt * 100:>5.1f}%)"
    )
    print(
        f"  String formatting:     {fmt * 1000:>8.2f}ms ({fmt / total_opt * 100:>5.1f}%)"
    )
    print(f"  Total:                 {total_opt * 1000:>8.2f}ms")
    print(f"\n  Overall speedup: {total / total_opt:.1f}x")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vars", type=int, default=1000, help="Number of variables")
    parser.add_argument("--cons", type=int, default=500, help="Number of constraints")
    parser.add_argument("--lookups", type=int, default=100, help="Number of lookups")
    parser.add_argument("--repeats", type=int, default=5, help="Number of repetitions")
    parser.add_argument("--output", type=str, default=None, help="Output NetCDF file")
    args = parser.parse_args()

    print("Building model...")
    model = build_model(args.vars, args.cons)

    print("Running bottleneck analysis...")
    ds = run_bottleneck_analysis(model, args.lookups, args.repeats)
    summary = compute_summary(ds)

    print_analysis(ds, summary)

    if args.output:
        xr.merge([ds, summary]).to_netcdf(args.output)
        print(f"Results saved to {args.output}")

    return ds, summary


if __name__ == "__main__":
    main()
