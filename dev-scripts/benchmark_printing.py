#!/usr/bin/env python3
"""
User-perspective benchmark for linopy printing functions.

This benchmark measures the performance of printing operations from a user's
perspective, focusing on realistic use cases:
- Printing variables (repr, str)
- Printing constraints (repr, str)
- Printing expressions
- Printing the model summary

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
from typing import Any

import numpy as np
import xarray as xr

from linopy import Model


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


def run_user_benchmarks(model: Model, repeats: int) -> xr.Dataset:
    """
    Run user-perspective benchmarks.

    Returns an xarray Dataset with timing results.
    """
    results = {}

    # 1. Variable repr (single variable array)
    x = model.variables["x"]

    def bench_var_repr():
        return repr(x)

    times = np.fromiter(time_function(bench_var_repr, repeats), dtype=float)
    results["variable_repr"] = xr.DataArray(
        times, dims=["repeat"],
        attrs={"description": "repr() of a single Variable array", "shape": str(x.shape)}
    )

    # 2. Variables container repr
    def bench_variables_repr():
        return repr(model.variables)

    times = np.fromiter(time_function(bench_variables_repr, repeats), dtype=float)
    results["variables_container_repr"] = xr.DataArray(
        times, dims=["repeat"],
        attrs={"description": "repr() of Variables container", "n_arrays": len(list(model.variables))}
    )

    # 3. Single constraint repr
    con = model.constraints[list(model.constraints)[0]]

    def bench_con_repr():
        return repr(con)

    times = np.fromiter(time_function(bench_con_repr, repeats), dtype=float)
    results["constraint_repr"] = xr.DataArray(
        times, dims=["repeat"],
        attrs={"description": "repr() of a single Constraint"}
    )

    # 4. Constraints container repr
    def bench_constraints_repr():
        return repr(model.constraints)

    times = np.fromiter(time_function(bench_constraints_repr, repeats), dtype=float)
    results["constraints_container_repr"] = xr.DataArray(
        times, dims=["repeat"],
        attrs={"description": "repr() of Constraints container", "n_arrays": len(list(model.constraints))}
    )

    # 5. Model repr
    def bench_model_repr():
        return repr(model)

    times = np.fromiter(time_function(bench_model_repr, repeats), dtype=float)
    results["model_repr"] = xr.DataArray(
        times, dims=["repeat"],
        attrs={"description": "repr() of Model"}
    )

    # 6. Expression repr (sum of variables)
    expr = x.sum()

    def bench_expr_repr():
        return repr(expr)

    times = np.fromiter(time_function(bench_expr_repr, repeats), dtype=float)
    results["expression_repr"] = xr.DataArray(
        times, dims=["repeat"],
        attrs={"description": "repr() of a summed expression"}
    )

    # 7. print_labels for variables (sample of labels)
    rng = np.random.default_rng(123)
    var_labels = rng.integers(0, model._xCounter, size=20).tolist()

    def bench_var_print_labels():
        model.variables.print_labels(var_labels)

    times = np.fromiter(time_function(suppress_output(bench_var_print_labels), repeats), dtype=float)
    results["variable_print_labels"] = xr.DataArray(
        times, dims=["repeat"],
        attrs={"description": "print_labels() for 20 variable labels", "n_labels": 20}
    )

    # 8. print_labels for constraints (sample of labels)
    con_labels = rng.integers(0, model._cCounter, size=20).tolist()

    def bench_con_print_labels():
        model.constraints.print_labels(con_labels)

    times = np.fromiter(time_function(suppress_output(bench_con_print_labels), repeats), dtype=float)
    results["constraint_print_labels"] = xr.DataArray(
        times, dims=["repeat"],
        attrs={"description": "print_labels() for 20 constraint labels", "n_labels": 20}
    )

    # 9. Sliced variable repr
    x_slice = x.isel(dim_0=slice(0, 5), dim_1=slice(0, 3))

    def bench_var_slice_repr():
        return repr(x_slice)

    times = np.fromiter(time_function(bench_var_slice_repr, repeats), dtype=float)
    results["variable_slice_repr"] = xr.DataArray(
        times, dims=["repeat"],
        attrs={"description": "repr() of sliced Variable (5x3)", "shape": str(x_slice.shape)}
    )

    # 10. Objective repr
    def bench_objective_repr():
        return repr(model.objective)

    times = np.fromiter(time_function(bench_objective_repr, repeats), dtype=float)
    results["objective_repr"] = xr.DataArray(
        times, dims=["repeat"],
        attrs={"description": "repr() of model objective"}
    )

    # Create dataset
    ds = xr.Dataset(results)
    ds.attrs["n_variables"] = model._xCounter
    ds.attrs["n_constraints"] = model._cCounter
    ds.attrs["n_variable_arrays"] = len(list(model.variables))
    ds.attrs["n_constraint_arrays"] = len(list(model.constraints))

    return ds


def compute_summary(ds: xr.Dataset) -> xr.Dataset:
    """Compute summary statistics."""
    stats = ["median_ms", "mean_ms", "std_ms"]
    data = {}

    for var_name in ds.data_vars:
        times = ds[var_name].values * 1000  # Convert to ms
        data[var_name] = xr.DataArray(
            [np.median(times), np.mean(times), np.std(times)],
            dims=["stat"], coords={"stat": stats}
        )

    summary = xr.Dataset(data)
    summary.attrs = ds.attrs.copy()
    return summary


def print_results(ds: xr.Dataset, summary: xr.Dataset) -> None:
    """Print benchmark results."""
    print("\n" + "=" * 70)
    print("USER-PERSPECTIVE BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\nModel: {ds.attrs['n_variables']} variables, {ds.attrs['n_constraints']} constraints")
    print(f"       {ds.attrs['n_variable_arrays']} variable arrays, {ds.attrs['n_constraint_arrays']} constraint arrays\n")

    print("Printing Operations:")
    print("-" * 70)
    print(f"  {'Operation':<40s} {'Median':>10s} {'Std':>10s}")
    print("-" * 70)

    # Group by category
    categories = {
        "Variable Operations": [
            "variable_repr", "variable_slice_repr", "variables_container_repr", "variable_print_labels"
        ],
        "Constraint Operations": [
            "constraint_repr", "constraints_container_repr", "constraint_print_labels"
        ],
        "Model Operations": [
            "model_repr", "expression_repr", "objective_repr"
        ],
    }

    for category, ops in categories.items():
        print(f"\n  {category}:")
        for op in ops:
            if op in summary.data_vars:
                median = float(summary[op].sel(stat="median_ms"))
                std = float(summary[op].sel(stat="std_ms"))
                desc = ds[op].attrs.get("description", op)
                print(f"    {desc:<38s} {median:>8.2f}ms {std:>8.2f}ms")

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

    print("Running user-perspective benchmarks...")
    ds = run_user_benchmarks(model, args.repeats)
    summary = compute_summary(ds)

    print_results(ds, summary)

    if args.output:
        xr.merge([ds, summary]).to_netcdf(args.output)
        print(f"Results saved to {args.output}")

    return ds, summary


if __name__ == "__main__":
    main()
