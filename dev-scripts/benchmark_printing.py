#!/usr/bin/env python3
"""
Benchmark for printing/formatting functions in linopy.

This benchmark measures the performance of:
1. get_label_position - Looking up variable/constraint names and coordinates from labels
2. print_single_variable - Formatting a single variable with bounds
3. print_single_expression - Formatting a linear expression
4. print_single_constraint - Formatting a complete constraint
5. print_coord - Formatting coordinate dictionaries

The benchmark tests both original (O(n) linear search) and optimized (O(log n) binary
search) implementations, verifying correctness by comparing outputs.

Results are stored in an xarray Dataset with dimensions:
- function: The function being benchmarked
- implementation: "original" or "optimized"
- repeat: Individual timing measurements

Usage:
    python dev-scripts/benchmark_printing.py [--vars N] [--cons N] [--terms N] [--repeats N]
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from typing import Any

import numpy as np
import xarray as xr

from linopy import Model
from linopy.common import (
    LabelPositionIndex,
    get_label_position_optimized,
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


@contextmanager
def patch_get_label_position(use_optimized: bool = False):
    """
    Context manager to monkey-patch get_label_position in Variables and Constraints.

    Parameters
    ----------
    use_optimized : bool
        If True, use the optimized O(log n) implementation.
        If False, use the original O(n) implementation.
    """
    import linopy.constraints as constraints_module
    import linopy.variables as variables_module

    # Store original methods
    original_var_method = variables_module.Variables.get_label_position
    original_con_method = constraints_module.Constraints.get_label_position

    if use_optimized:
        # Create index caches
        _var_index_cache = {}
        _con_index_cache = {}

        def optimized_var_get_label_position(self, values):
            if id(self) not in _var_index_cache:
                _var_index_cache[id(self)] = LabelPositionIndex(self)
            return get_label_position_optimized(
                self, values, _var_index_cache[id(self)]
            )

        def optimized_con_get_label_position(self, values):
            if id(self) not in _con_index_cache:
                _con_index_cache[id(self)] = LabelPositionIndex(self)
            return get_label_position_optimized(
                self, values, _con_index_cache[id(self)]
            )

        variables_module.Variables.get_label_position = optimized_var_get_label_position
        constraints_module.Constraints.get_label_position = (
            optimized_con_get_label_position
        )

    try:
        yield
    finally:
        # Restore original methods
        variables_module.Variables.get_label_position = original_var_method
        constraints_module.Constraints.get_label_position = original_con_method


def verify_correctness(model: Model, n_samples: int = 100) -> dict[str, bool]:
    """
    Verify that optimized implementation produces same results as original.

    Parameters
    ----------
    model : Model
        The model to test against.
    n_samples : int
        Number of random samples to test.

    Returns
    -------
    dict
        Dictionary with verification results for each function.
    """
    rng = np.random.default_rng(456)
    results = {}

    # Test variable label position
    max_var_label = model._xCounter
    var_labels = rng.integers(0, max_var_label, size=n_samples)

    var_original = []
    with patch_get_label_position(use_optimized=False):
        for label in var_labels:
            var_original.append(model.variables.get_label_position(int(label)))

    var_optimized = []
    with patch_get_label_position(use_optimized=True):
        for label in var_labels:
            var_optimized.append(model.variables.get_label_position(int(label)))

    results["get_label_position_vars"] = var_original == var_optimized

    # Test constraint label position
    max_con_label = model._cCounter
    con_labels = rng.integers(0, max_con_label, size=n_samples)

    con_original = []
    with patch_get_label_position(use_optimized=False):
        for label in con_labels:
            con_original.append(model.constraints.get_label_position(int(label)))

    con_optimized = []
    with patch_get_label_position(use_optimized=True):
        for label in con_labels:
            con_optimized.append(model.constraints.get_label_position(int(label)))

    results["get_label_position_cons"] = con_original == con_optimized

    # Test print_single_variable
    var_print_original = []
    with patch_get_label_position(use_optimized=False):
        for label in var_labels[:20]:
            var_print_original.append(print_single_variable(model, int(label)))

    var_print_optimized = []
    with patch_get_label_position(use_optimized=True):
        for label in var_labels[:20]:
            var_print_optimized.append(print_single_variable(model, int(label)))

    results["print_single_variable"] = var_print_original == var_print_optimized

    # Test print_single_constraint
    con_print_original = []
    with patch_get_label_position(use_optimized=False):
        for label in con_labels[:20]:
            con_print_original.append(print_single_constraint(model, int(label)))

    con_print_optimized = []
    with patch_get_label_position(use_optimized=True):
        for label in con_labels[:20]:
            con_print_optimized.append(print_single_constraint(model, int(label)))

    results["print_single_constraint"] = con_print_original == con_print_optimized

    # Test batch lookups (1D array)
    batch_labels = var_labels[:50]
    with patch_get_label_position(use_optimized=False):
        batch_original = model.variables.get_label_position(batch_labels)
    with patch_get_label_position(use_optimized=True):
        batch_optimized = model.variables.get_label_position(batch_labels)

    results["batch_lookup_1d"] = batch_original == batch_optimized

    # Test 2D array lookups
    labels_2d = var_labels[:20].reshape(4, 5)
    with patch_get_label_position(use_optimized=False):
        batch_2d_original = model.variables.get_label_position(labels_2d)
    with patch_get_label_position(use_optimized=True):
        batch_2d_optimized = model.variables.get_label_position(labels_2d)

    results["batch_lookup_2d"] = batch_2d_original == batch_2d_optimized

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

    # Define benchmarks with both implementations
    implementations = ["original", "optimized"]
    results = {}

    for impl in implementations:
        use_opt = impl == "optimized"

        with patch_get_label_position(use_optimized=use_opt):
            # get_label_position for variables
            def bench_var_lookup():
                return [variables.get_label_position(int(l)) for l in var_labels]

            times = np.fromiter(time_function(bench_var_lookup, repeats), dtype=float)
            results[f"get_label_position_vars_{impl}"] = xr.DataArray(
                times, dims=["repeat"], coords={"repeat": range(repeats)}
            )
            results[f"get_label_position_vars_{impl}"].attrs["n_operations"] = n_lookups

            # get_label_position for constraints
            def bench_con_lookup():
                return [constraints.get_label_position(int(l)) for l in con_labels]

            times = np.fromiter(time_function(bench_con_lookup, repeats), dtype=float)
            results[f"get_label_position_cons_{impl}"] = xr.DataArray(
                times, dims=["repeat"], coords={"repeat": range(repeats)}
            )
            results[f"get_label_position_cons_{impl}"].attrs["n_operations"] = n_lookups

            # print_single_variable
            def bench_print_var():
                return [print_single_variable(model, int(l)) for l in var_labels]

            times = np.fromiter(time_function(bench_print_var, repeats), dtype=float)
            results[f"print_single_variable_{impl}"] = xr.DataArray(
                times, dims=["repeat"], coords={"repeat": range(repeats)}
            )
            results[f"print_single_variable_{impl}"].attrs["n_operations"] = n_lookups

            # print_single_constraint
            def bench_print_con():
                return [print_single_constraint(model, int(l)) for l in con_labels]

            times = np.fromiter(time_function(bench_print_con, repeats), dtype=float)
            results[f"print_single_constraint_{impl}"] = xr.DataArray(
                times, dims=["repeat"], coords={"repeat": range(repeats)}
            )
            results[f"print_single_constraint_{impl}"].attrs["n_operations"] = n_lookups

            # print_single_expression (uses variable lookups internally)
            def bench_print_expr():
                return [
                    print_single_expression(c, v, const, model)
                    for c, v, const in expressions
                ]

            times = np.fromiter(time_function(bench_print_expr, repeats), dtype=float)
            results[f"print_single_expression_{impl}"] = xr.DataArray(
                times, dims=["repeat"], coords={"repeat": range(repeats)}
            )
            results[f"print_single_expression_{impl}"].attrs["n_operations"] = n_lookups

    # print_coord (doesn't use get_label_position, no impl difference)
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

    Returns a Dataset with a 2D DataArray indexed by (function, stat).
    """
    # Extract unique function names (without _original/_optimized suffix)
    all_vars = list(ds.data_vars)
    functions = sorted(
        set(
            v.replace("_original", "").replace("_optimized", "")
            for v in all_vars
        )
    )
    implementations = ["original", "optimized"]
    stats = ["median_s", "mean_s", "std_s", "per_op_us"]

    # Build summary data
    data = {}
    for func in functions:
        for impl in implementations:
            key = f"{func}_{impl}"
            if key not in ds.data_vars:
                # Function doesn't have impl variants (e.g., print_coord)
                if func in ds.data_vars and impl == "original":
                    key = func
                else:
                    continue

            times = ds[key].values
            n_ops = ds[key].attrs.get("n_operations", 1)

            row_data = [
                np.median(times),
                np.mean(times),
                np.std(times),
                (np.median(times) / n_ops) * 1_000_000,
            ]

            data[f"{func}_{impl}"] = xr.DataArray(
                row_data, dims=["stat"], coords={"stat": stats}
            )

    summary_ds = xr.Dataset(data)
    summary_ds.attrs = ds.attrs.copy()
    summary_ds.attrs["stat_units"] = {
        "median_s": "seconds",
        "mean_s": "seconds",
        "std_s": "seconds",
        "per_op_us": "microseconds per operation",
    }

    return summary_ds


def print_summary(ds: xr.Dataset, summary: xr.Dataset, verification: dict) -> None:
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
    all_passed = all(verification.values())
    for func, passed in verification.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {func:40s}: {status}")
    print(f"\n  Overall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}\n")

    # Performance comparison
    print("Performance Comparison (Original vs Optimized):")
    print("-" * 75)
    print(f"  {'Function':<35s} {'Original':>12s} {'Optimized':>12s} {'Speedup':>10s}")
    print("-" * 75)

    functions = [
        "get_label_position_vars",
        "get_label_position_cons",
        "print_single_variable",
        "print_single_expression",
        "print_single_constraint",
    ]

    for func in functions:
        orig_key = f"{func}_original"
        opt_key = f"{func}_optimized"

        if orig_key in summary.data_vars and opt_key in summary.data_vars:
            orig_ms = float(summary[orig_key].sel(stat="median_s")) * 1000
            opt_ms = float(summary[opt_key].sel(stat="median_s")) * 1000
            speedup = orig_ms / opt_ms if opt_ms > 0 else float("inf")
            print(f"  {func:<35s} {orig_ms:>10.2f}ms {opt_ms:>10.2f}ms {speedup:>9.1f}x")

    # print_coord (no impl variants)
    if "print_coord_original" in summary.data_vars:
        pc_ms = float(summary["print_coord_original"].sel(stat="median_s")) * 1000
        print(f"  {'print_coord':<35s} {pc_ms:>10.2f}ms {'N/A':>12s} {'N/A':>10s}")

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

    print("Running benchmarks...")
    ds = run_benchmarks(model, args.lookups, args.terms, args.repeats)
    summary = compute_summary(ds)

    # Print summary
    print_summary(ds, summary, verification)

    # Save to file if requested
    if args.output:
        combined = xr.merge([ds, summary])
        combined.to_netcdf(args.output)
        print(f"Results saved to {args.output}")

    return ds, summary, verification


if __name__ == "__main__":
    main()
