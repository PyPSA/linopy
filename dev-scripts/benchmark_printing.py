#!/usr/bin/env python3
"""
Benchmark for printing/formatting functions in linopy.

This benchmark measures the performance of:
1. get_label_position - Looking up variable/constraint names and coordinates from labels
2. print_single_variable - Formatting a single variable with bounds
3. print_single_expression - Formatting a linear expression
4. print_single_constraint - Formatting a complete constraint
5. print_coord - Formatting coordinate dictionaries

The optimized O(log n) binary search implementation is now built into the Variables
and Constraints classes with automatic cache invalidation.

Results are stored in an xarray Dataset with dimensions:
- function: The function being benchmarked
- repeat: Individual timing measurements

Usage:
    python dev-scripts/benchmark_printing.py [--vars N] [--cons N] [--terms N] [--repeats N]
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
    print_single_constraint,
    print_single_expression,
    print_single_variable,
)


def build_model(n_vars: int, n_cons: int, terms_per_con: int) -> Model:
    """
    Build a model with specified number of variables and constraints.

    Parameters
    ----------
    n_vars : int
        Number of variables (split across two variable arrays for realism).
    n_cons : int
        Number of constraints.
    terms_per_con : int
        Number of terms per constraint.

    Returns
    -------
    Model
        A linopy model with variables and constraints.
    """
    rng = np.random.default_rng(42)
    m = Model()

    # Create variables with 2D coordinates (more realistic)
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

    # Create constraints with random coefficients
    for i in range(n_cons):
        # Pick random variables for this constraint
        var_indices = rng.integers(0, n_vars_per_dim, size=(terms_per_con, 2))
        coeffs = rng.uniform(-10, 10, size=terms_per_con)

        # Build expression
        lhs = sum(
            coeffs[j]
            * (x if j % 2 == 0 else y).isel(
                dim_0=var_indices[j, 0], dim_1=var_indices[j, 1]
            )
            for j in range(terms_per_con)
        )
        rhs = rng.uniform(-100, 100)
        m.add_constraints(lhs >= rhs, name=f"con{i}")

    return m


def time_function(
    func: Callable[[], Any], repeats: int, warmup: int = 2
) -> Iterable[float]:
    """Time a function over multiple iterations."""
    # Warmup
    for _ in range(warmup):
        func()

    # Timed runs
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        yield end - start


def verify_correctness(model: Model, n_samples: int = 100) -> dict[str, bool]:
    """
    Verify that the optimized implementation produces correct results.

    This tests:
    1. Single lookups return valid (name, coord) tuples
    2. Batch lookups (1D and 2D arrays) work correctly
    3. Cache invalidation works when adding new variables/constraints

    Parameters
    ----------
    model : Model
        The model to test against.
    n_samples : int
        Number of random samples to test.

    Returns
    -------
    dict
        Dictionary with verification results for each test.
    """
    rng = np.random.default_rng(456)
    results = {}

    # Test variable label position
    max_var_label = model._xCounter
    var_labels = rng.integers(0, max_var_label, size=n_samples)

    # Test single lookups
    var_results = []
    for label in var_labels:
        name, coord = model.variables.get_label_position(int(label))
        # Verify the result is valid
        var = model.variables[name]
        actual_label = int(var.labels.sel(coord).values)
        var_results.append(actual_label == label)
    results["var_single_lookup"] = all(var_results)

    # Test constraint label position
    max_con_label = model._cCounter
    con_labels = rng.integers(0, max_con_label, size=n_samples)

    con_results = []
    for label in con_labels:
        name, coord = model.constraints.get_label_position(int(label))
        con = model.constraints[name]
        actual_label = int(con.labels.sel(coord).values)
        con_results.append(actual_label == label)
    results["con_single_lookup"] = all(con_results)

    # Test batch lookups (1D array)
    batch_results = model.variables.get_label_position(var_labels[:50])
    batch_valid = []
    for i, (name, coord) in enumerate(batch_results):
        var = model.variables[name]
        actual_label = int(var.labels.sel(coord).values)
        batch_valid.append(actual_label == var_labels[i])
    results["batch_lookup_1d"] = all(batch_valid)

    # Test 2D array lookups
    labels_2d = var_labels[:20].reshape(4, 5)
    batch_2d_results = model.variables.get_label_position(labels_2d)
    batch_2d_valid = []
    for col_idx, col_results in enumerate(batch_2d_results):
        for row_idx, (name, coord) in enumerate(col_results):
            var = model.variables[name]
            actual_label = int(var.labels.sel(coord).values)
            expected_label = labels_2d[row_idx, col_idx]
            batch_2d_valid.append(actual_label == expected_label)
    results["batch_lookup_2d"] = all(batch_2d_valid)

    # Test cache invalidation by adding a new variable
    old_max = model._xCounter
    _ = model.add_variables(lower=0, upper=10, name="test_cache_var", coords=[range(5)])

    # The new variable should be findable
    new_label = old_max  # First label of new variable
    name, coord = model.variables.get_label_position(new_label)
    results["cache_invalidation_var"] = name == "test_cache_var"

    # Test cache invalidation by adding a new constraint
    x = model.variables["x"]
    old_con_max = model._cCounter
    model.add_constraints(x.isel(dim_0=0, dim_1=0) >= 0, name="test_cache_con")
    new_con_label = old_con_max
    name, coord = model.constraints.get_label_position(new_con_label)
    results["cache_invalidation_con"] = name == "test_cache_con"

    return results


def compare_with_original(model: Model, n_samples: int = 50) -> dict[str, bool]:
    """
    Compare optimized implementation with original O(n) implementation.

    Parameters
    ----------
    model : Model
        The model to test against.
    n_samples : int
        Number of random samples to test.

    Returns
    -------
    dict
        Dictionary with comparison results.
    """
    rng = np.random.default_rng(789)
    results = {}

    # Test variables
    max_var_label = model._xCounter
    var_labels = rng.integers(0, max_var_label, size=n_samples)

    # Get results from original implementation
    original_var = [get_label_position(model.variables, int(l)) for l in var_labels]
    # Get results from optimized implementation (built into class)
    optimized_var = [model.variables.get_label_position(int(l)) for l in var_labels]

    results["var_matches_original"] = original_var == optimized_var

    # Test constraints
    max_con_label = model._cCounter
    con_labels = rng.integers(0, max_con_label, size=n_samples)

    original_con = [get_label_position(model.constraints, int(l)) for l in con_labels]
    optimized_con = [model.constraints.get_label_position(int(l)) for l in con_labels]

    results["con_matches_original"] = original_con == optimized_con

    return results


def run_benchmarks(
    model: Model, n_lookups: int, terms_per_expr: int, repeats: int
) -> xr.Dataset:
    """
    Run all benchmarks and return results as an xarray Dataset.

    Parameters
    ----------
    model : Model
        The linopy model to benchmark against.
    n_lookups : int
        Number of lookups/prints per benchmark.
    terms_per_expr : int
        Number of terms per expression.
    repeats : int
        Number of timing repetitions.

    Returns
    -------
    xr.Dataset
        Dataset with benchmark results.
    """
    rng = np.random.default_rng(123)
    variables = model.variables
    constraints = model.constraints

    # Prepare test data
    max_var_label = model._xCounter
    max_con_label = model._cCounter
    var_labels = rng.integers(0, max_var_label, size=n_lookups)
    con_labels = rng.integers(0, max_con_label, size=n_lookups)

    # Generate random expressions for print_single_expression
    expressions = []
    for _ in range(n_lookups):
        n_terms = rng.integers(1, terms_per_expr + 1)
        coeffs = rng.uniform(-10, 10, size=n_terms)
        vars_arr = rng.integers(0, max_var_label, size=n_terms)
        const = rng.uniform(-100, 100)
        expressions.append((coeffs, vars_arr, const))

    # Generate random coordinates for print_coord
    coords_list = []
    for _ in range(n_lookups * 10):
        n_dims = rng.integers(1, 5)
        coord = {f"dim_{i}": rng.integers(0, 100) for i in range(n_dims)}
        coords_list.append(coord)

    results = {}

    # Benchmark get_label_position for variables (optimized - built-in)
    def bench_var_lookup():
        return [variables.get_label_position(int(l)) for l in var_labels]

    times = np.fromiter(time_function(bench_var_lookup, repeats), dtype=float)
    results["get_label_position_vars"] = xr.DataArray(
        times, dims=["repeat"], coords={"repeat": range(repeats)}
    )
    results["get_label_position_vars"].attrs["n_operations"] = n_lookups

    # Benchmark get_label_position for constraints (optimized - built-in)
    def bench_con_lookup():
        return [constraints.get_label_position(int(l)) for l in con_labels]

    times = np.fromiter(time_function(bench_con_lookup, repeats), dtype=float)
    results["get_label_position_cons"] = xr.DataArray(
        times, dims=["repeat"], coords={"repeat": range(repeats)}
    )
    results["get_label_position_cons"].attrs["n_operations"] = n_lookups

    # Benchmark original O(n) implementation for comparison
    def bench_con_lookup_original():
        return [get_label_position(constraints, int(l)) for l in con_labels]

    times = np.fromiter(time_function(bench_con_lookup_original, repeats), dtype=float)
    results["get_label_position_cons_original"] = xr.DataArray(
        times, dims=["repeat"], coords={"repeat": range(repeats)}
    )
    results["get_label_position_cons_original"].attrs["n_operations"] = n_lookups

    # Benchmark print_single_variable
    def bench_print_var():
        return [print_single_variable(model, int(l)) for l in var_labels]

    times = np.fromiter(time_function(bench_print_var, repeats), dtype=float)
    results["print_single_variable"] = xr.DataArray(
        times, dims=["repeat"], coords={"repeat": range(repeats)}
    )
    results["print_single_variable"].attrs["n_operations"] = n_lookups

    # Benchmark print_single_constraint
    def bench_print_con():
        return [print_single_constraint(model, int(l)) for l in con_labels]

    times = np.fromiter(time_function(bench_print_con, repeats), dtype=float)
    results["print_single_constraint"] = xr.DataArray(
        times, dims=["repeat"], coords={"repeat": range(repeats)}
    )
    results["print_single_constraint"].attrs["n_operations"] = n_lookups

    # Benchmark print_single_expression
    def bench_print_expr():
        return [
            print_single_expression(c, v, const, model)
            for c, v, const in expressions
        ]

    times = np.fromiter(time_function(bench_print_expr, repeats), dtype=float)
    results["print_single_expression"] = xr.DataArray(
        times, dims=["repeat"], coords={"repeat": range(repeats)}
    )
    results["print_single_expression"].attrs["n_operations"] = n_lookups

    # Benchmark print_coord
    def bench_print_coord():
        return [print_coord(c) for c in coords_list]

    times = np.fromiter(time_function(bench_print_coord, repeats), dtype=float)
    results["print_coord"] = xr.DataArray(
        times, dims=["repeat"], coords={"repeat": range(repeats)}
    )
    results["print_coord"].attrs["n_operations"] = n_lookups * 10

    # Create dataset
    ds = xr.Dataset(results)

    # Add metadata
    ds.attrs["n_variables"] = model._xCounter
    ds.attrs["n_constraints"] = model._cCounter
    ds.attrs["n_variable_arrays"] = len(list(variables))
    ds.attrs["n_constraint_arrays"] = len(list(constraints))
    ds.attrs["n_lookups"] = n_lookups
    ds.attrs["terms_per_expr"] = terms_per_expr
    ds.attrs["description"] = "Benchmark timings in seconds"

    return ds


def compute_summary(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute summary statistics from benchmark results.

    Returns a Dataset with statistics for each function.
    """
    stats = ["median_s", "mean_s", "std_s", "per_op_us"]
    data = {}

    for func in ds.data_vars:
        times = ds[func].values
        n_ops = ds[func].attrs.get("n_operations", 1)

        row_data = [
            np.median(times),
            np.mean(times),
            np.std(times),
            (np.median(times) / n_ops) * 1_000_000,
        ]

        data[func] = xr.DataArray(row_data, dims=["stat"], coords={"stat": stats})

    summary_ds = xr.Dataset(data)
    summary_ds.attrs = ds.attrs.copy()
    summary_ds.attrs["stat_units"] = {
        "median_s": "seconds",
        "mean_s": "seconds",
        "std_s": "seconds",
        "per_op_us": "microseconds per operation",
    }

    return summary_ds


def print_summary(
    ds: xr.Dataset,
    summary: xr.Dataset,
    verification: dict,
    comparison: dict,
) -> None:
    """Print a concise summary of benchmark results."""
    print("\n" + "=" * 75)
    print("BENCHMARK RESULTS")
    print("=" * 75)

    print(
        f"\nModel: {ds.attrs['n_variables']} variables, "
        f"{ds.attrs['n_constraints']} constraints"
    )
    print(
        f"       {ds.attrs['n_variable_arrays']} variable arrays, "
        f"{ds.attrs['n_constraint_arrays']} constraint arrays"
    )
    print(
        f"       {ds.attrs['n_lookups']} lookups, "
        f"{ds.attrs['terms_per_expr']} terms/expr\n"
    )

    # Verification results
    print("Correctness Verification:")
    print("-" * 75)
    all_passed = all(verification.values()) and all(comparison.values())
    for func, passed in {**verification, **comparison}.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {func:40s}: {status}")
    print(
        f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}\n"
    )

    # Performance results
    print("Performance Results:")
    print("-" * 75)
    print(f"  {'Function':<40s} {'Time':>12s} {'Per Op':>12s}")
    print("-" * 75)

    for func in [
        "get_label_position_vars",
        "get_label_position_cons",
        "get_label_position_cons_original",
        "print_single_variable",
        "print_single_expression",
        "print_single_constraint",
        "print_coord",
    ]:
        if func in summary.data_vars:
            median_ms = float(summary[func].sel(stat="median_s")) * 1000
            per_op = float(summary[func].sel(stat="per_op_us"))
            print(f"  {func:<40s} {median_ms:>10.2f}ms {per_op:>10.2f}µs")

    # Speedup calculation
    print("\nSpeedup (Optimized vs Original):")
    print("-" * 75)
    if (
        "get_label_position_cons" in summary.data_vars
        and "get_label_position_cons_original" in summary.data_vars
    ):
        opt = float(summary["get_label_position_cons"].sel(stat="median_s"))
        orig = float(summary["get_label_position_cons_original"].sel(stat="median_s"))
        speedup = orig / opt if opt > 0 else float("inf")
        print(f"  get_label_position_cons: {speedup:.1f}x faster")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vars", type=int, default=1000, help="Number of variables")
    parser.add_argument("--cons", type=int, default=500, help="Number of constraints")
    parser.add_argument(
        "--terms", type=int, default=10, help="Terms per constraint/expression"
    )
    parser.add_argument(
        "--lookups", type=int, default=100, help="Number of lookups per benchmark"
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="Number of timing repetitions"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file for results (NetCDF)"
    )
    args = parser.parse_args()

    print("Building model...")
    model = build_model(args.vars, args.cons, args.terms)

    print("Verifying correctness...")
    verification = verify_correctness(model)

    print("Comparing with original implementation...")
    comparison = compare_with_original(model)

    print("Running benchmarks...")
    ds = run_benchmarks(model, args.lookups, args.terms, args.repeats)
    summary = compute_summary(ds)

    # Print summary
    print_summary(ds, summary, verification, comparison)

    # Save to file if requested
    if args.output:
        combined = xr.merge([ds, summary])
        combined.to_netcdf(args.output)
        print(f"Results saved to {args.output}")

    return ds, summary, verification, comparison


if __name__ == "__main__":
    main()
