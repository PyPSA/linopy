#!/usr/bin/env python3
"""
Benchmark for printing/formatting functions in linopy.

This benchmark measures the performance of:
1. get_label_position - Looking up variable/constraint names and coordinates from labels
2. print_single_variable - Formatting a single variable with bounds
3. print_single_expression - Formatting a linear expression
4. print_single_constraint - Formatting a complete constraint
5. print_coord - Formatting coordinate dictionaries

Results are stored in an xarray Dataset with dimensions:
- function: The function being benchmarked
- implementation: Original vs optimized (where applicable)
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
        Dataset with benchmark results. Dimensions:
        - repeat: Individual timing measurements
        Data variables include timing results for each function/step.
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

    # Prepare constraint breakdown data
    positions = [constraints.get_label_position(int(label)) for label in con_labels]
    all_var_labels = []
    extracted_data = []
    for name, coord in positions:
        con = constraints[name]
        vars_arr = con.vars.sel(coord).values.flatten()
        vars_arr = vars_arr[vars_arr != -1]
        all_var_labels.extend(vars_arr.tolist())
        extracted_data.append((con.coeffs.sel(coord).values, con.vars.sel(coord).values))

    # Build optimized lookup index
    ranges = []
    for name in constraints:
        con = constraints[name]
        start, stop = con.range
        ranges.append((start, stop, name))
    ranges.sort(key=lambda x: x[0])
    starts = np.array([r[0] for r in ranges])
    stops = np.array([r[1] for r in ranges])
    names = [r[2] for r in ranges]

    def optimized_find_single(value: int) -> tuple[str, dict] | tuple[None, None]:
        if value == -1:
            return None, None
        idx = np.searchsorted(starts, value, side="right") - 1
        if idx < 0 or idx >= len(starts) or value < starts[idx] or value >= stops[idx]:
            raise ValueError(f"Label {value} not found")
        name = names[idx]
        con = constraints[name]
        labels_arr = con.labels
        local_idx = value - starts[idx]
        index = np.unravel_index(local_idx, labels_arr.shape)
        coord = {
            dim: labels_arr.indexes[dim][i] for dim, i in zip(labels_arr.dims, index)
        }
        return name, coord

    # Define benchmark functions
    benchmarks = {
        # Main functions
        "get_label_position_vars": lambda: [
            variables.get_label_position(int(l)) for l in var_labels
        ],
        "get_label_position_cons": lambda: [
            constraints.get_label_position(int(l)) for l in con_labels
        ],
        "get_label_position_cons_optimized": lambda: [
            optimized_find_single(int(l)) for l in con_labels
        ],
        "print_single_variable": lambda: [
            print_single_variable(model, int(l)) for l in var_labels
        ],
        "print_single_expression": lambda: [
            print_single_expression(c, v, const, model) for c, v, const in expressions
        ],
        "print_single_constraint": lambda: [
            print_single_constraint(model, int(l)) for l in con_labels
        ],
        "print_coord": lambda: [print_coord(c) for c in coords_list],
        # Breakdown steps for print_single_constraint
        "breakdown_constraint_lookup": lambda: [
            constraints.get_label_position(int(l)) for l in con_labels
        ],
        "breakdown_sel_calls": lambda: [
            (
                constraints[name].coeffs.sel(coord).values,
                constraints[name].vars.sel(coord).values,
                constraints[name].sign.sel(coord).item(),
                constraints[name].rhs.sel(coord).item(),
            )
            for name, coord in positions
        ],
        "breakdown_variable_lookup": lambda: [
            variables.get_label_position(int(l)) for l in all_var_labels
        ],
        "breakdown_print_expression": lambda: [
            print_single_expression(c, v, 0, model) for c, v in extracted_data
        ],
        "breakdown_string_format": lambda: [
            f"{name}{print_coord(coord)}: expr sign 0.0" for name, coord in positions
        ],
        # __repr__ calls
        "repr_variable": lambda: repr(variables[list(variables)[0]]),
        "repr_constraint": lambda: repr(constraints[list(constraints)[0]]),
    }

    # Run benchmarks and collect results
    results = {}
    for name, func in benchmarks.items():
        times = np.fromiter(time_function(func, repeats), dtype=float)
        results[name] = xr.DataArray(
            times,
            dims=["repeat"],
            coords={"repeat": range(repeats)},
        )

    # Create dataset
    ds = xr.Dataset(results)

    # Add metadata as attributes
    ds.attrs["n_variables"] = model._xCounter
    ds.attrs["n_constraints"] = model._cCounter
    ds.attrs["n_variable_arrays"] = len(list(variables))
    ds.attrs["n_constraint_arrays"] = len(list(constraints))
    ds.attrs["n_lookups"] = n_lookups
    ds.attrs["terms_per_expr"] = terms_per_expr
    ds.attrs["n_var_labels_in_breakdown"] = len(all_var_labels)
    ds.attrs["description"] = "Benchmark timings in seconds"

    # Add per-variable attributes
    ds["get_label_position_vars"].attrs["n_operations"] = n_lookups
    ds["get_label_position_cons"].attrs["n_operations"] = n_lookups
    ds["get_label_position_cons_optimized"].attrs["n_operations"] = n_lookups
    ds["print_single_variable"].attrs["n_operations"] = n_lookups
    ds["print_single_expression"].attrs["n_operations"] = n_lookups
    ds["print_single_constraint"].attrs["n_operations"] = n_lookups
    ds["print_coord"].attrs["n_operations"] = n_lookups * 10
    ds["breakdown_constraint_lookup"].attrs["n_operations"] = n_lookups
    ds["breakdown_sel_calls"].attrs["n_operations"] = n_lookups
    ds["breakdown_variable_lookup"].attrs["n_operations"] = len(all_var_labels)
    ds["breakdown_print_expression"].attrs["n_operations"] = n_lookups
    ds["breakdown_string_format"].attrs["n_operations"] = n_lookups
    ds["repr_variable"].attrs["n_operations"] = 1
    ds["repr_constraint"].attrs["n_operations"] = 1

    return ds


def compute_summary(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute summary statistics from benchmark results.

    Parameters
    ----------
    ds : xr.Dataset
        Raw benchmark results from run_benchmarks.

    Returns
    -------
    xr.Dataset
        Summary statistics with dimensions:
        - function: Name of the benchmarked function
        - stat: Statistic type (median, mean, std, per_op_median)
    """
    functions = list(ds.data_vars)
    stats = ["median_s", "mean_s", "std_s", "per_op_us"]

    data = np.zeros((len(functions), len(stats)))

    for i, func in enumerate(functions):
        times = ds[func].values
        n_ops = ds[func].attrs.get("n_operations", 1)
        data[i, 0] = np.median(times)
        data[i, 1] = np.mean(times)
        data[i, 2] = np.std(times)
        data[i, 3] = (np.median(times) / n_ops) * 1_000_000  # Convert to microseconds

    summary = xr.DataArray(
        data,
        dims=["function", "stat"],
        coords={"function": functions, "stat": stats},
        name="timing_summary",
    )

    summary_ds = xr.Dataset({"timing_summary": summary})
    summary_ds.attrs = ds.attrs.copy()
    summary_ds.attrs["stat_units"] = {
        "median_s": "seconds",
        "mean_s": "seconds",
        "std_s": "seconds",
        "per_op_us": "microseconds per operation",
    }

    return summary_ds


def print_summary(ds: xr.Dataset, summary: xr.Dataset) -> None:
    """Print a concise summary of benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\nModel: {ds.attrs['n_variables']} variables, "
          f"{ds.attrs['n_constraints']} constraints")
    print(f"       {ds.attrs['n_variable_arrays']} variable arrays, "
          f"{ds.attrs['n_constraint_arrays']} constraint arrays")
    print(f"       {ds.attrs['n_lookups']} lookups, "
          f"{ds.attrs['terms_per_expr']} terms/expr\n")

    timing = summary["timing_summary"]

    # Main functions
    print("Main Functions:")
    print("-" * 70)
    main_funcs = [
        "get_label_position_vars",
        "get_label_position_cons",
        "print_single_variable",
        "print_single_expression",
        "print_single_constraint",
        "print_coord",
    ]
    for func in main_funcs:
        median_ms = float(timing.sel(function=func, stat="median_s")) * 1000
        per_op = float(timing.sel(function=func, stat="per_op_us"))
        print(f"  {func:35s}: {median_ms:8.2f} ms  ({per_op:8.2f} µs/op)")

    # Optimization comparison
    print("\nOptimization Comparison (get_label_position_cons):")
    print("-" * 70)
    orig = float(timing.sel(function="get_label_position_cons", stat="median_s"))
    opt = float(timing.sel(function="get_label_position_cons_optimized", stat="median_s"))
    speedup = orig / opt if opt > 0 else float("inf")
    print(f"  Original (linear):   {orig * 1000:8.2f} ms")
    print(f"  Optimized (binary):  {opt * 1000:8.2f} ms")
    print(f"  Speedup:             {speedup:8.1f}x")

    # Breakdown
    print("\nprint_single_constraint Breakdown:")
    print("-" * 70)
    total_time = float(timing.sel(function="print_single_constraint", stat="median_s"))
    breakdown_funcs = [
        ("breakdown_constraint_lookup", "Constraint label lookup"),
        ("breakdown_sel_calls", ".sel() calls"),
        ("breakdown_variable_lookup", "Variable label lookup"),
        ("breakdown_print_expression", "print_expression"),
        ("breakdown_string_format", "String formatting"),
    ]
    for func, label in breakdown_funcs:
        t = float(timing.sel(function=func, stat="median_s"))
        pct = (t / total_time) * 100 if total_time > 0 else 0
        print(f"  {label:30s}: {t * 1000:8.2f} ms  ({pct:5.1f}%)")

    # __repr__ calls
    print("\n__repr__ Calls:")
    print("-" * 70)
    for func in ["repr_variable", "repr_constraint"]:
        median_ms = float(timing.sel(function=func, stat="median_s")) * 1000
        print(f"  {func:35s}: {median_ms:8.2f} ms")

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

    print("Running benchmarks...")
    ds = run_benchmarks(model, args.lookups, args.terms, args.repeats)
    summary = compute_summary(ds)

    # Print summary
    print_summary(ds, summary)

    # Save to file if requested
    if args.output:
        # Combine raw and summary into one dataset
        combined = xr.merge([ds, summary])
        combined.to_netcdf(args.output)
        print(f"Results saved to {args.output}")

    # Return datasets for interactive use
    return ds, summary


if __name__ == "__main__":
    main()
