#!/usr/bin/env python3
"""
User-story benchmark for linopy printing functions.

This benchmark compares OLD (O(n) linear search) vs NEW (O(log n) binary search)
implementations from a user's perspective with realistic workflows.

User Stories:
-------------
1. "I want to inspect my model" -> print(model)
2. "I want to print all variables" -> {name: repr(v) for name, v in model.variables.items()}
3. "I want to print all constraints" -> {name: repr(c) for name, c in model.constraints.items()}
4. "I want to inspect a single variable array" -> print(model.variables["x"])
5. "I want to inspect a single constraint" -> print(model.constraints["con0"])
6. "I want to see an expression" -> print(x + y)
7. "I want to see the objective" -> print(model.objective)

The optimization primarily helps when there are many variable/constraint arrays,
since get_label_position searches through arrays to find which one contains a label.

Results are stored in an xarray Dataset for analysis.

Usage:
    python dev-scripts/benchmark_printing.py [--var-arrays N] [--con-arrays N] [--repeats N]
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
from linopy.common import _get_label_position_linear


def build_model(n_var_arrays: int, n_con_arrays: int, vars_per_array: int = 10) -> Model:
    """
    Build a model with specified number of variable and constraint arrays.

    This structure better demonstrates the optimization benefit since
    get_label_position searches through arrays (not individual variables).
    """
    rng = np.random.default_rng(42)
    m = Model()

    # Create many variable arrays (this is where the optimization helps!)
    var_names = []
    for i in range(n_var_arrays):
        m.add_variables(
            lower=0, upper=100, name=f"x{i}",
            coords=[range(vars_per_array)],
        )
        var_names.append(f"x{i}")

    # Create constraints that reference variables from different arrays
    for i in range(n_con_arrays):
        # Pick random variables from random arrays
        n_terms = min(8, n_var_arrays)
        array_indices = rng.integers(0, n_var_arrays, size=n_terms)
        var_indices = rng.integers(0, vars_per_array, size=n_terms)
        coeffs = rng.uniform(-10, 10, size=n_terms)

        lhs = sum(
            coeffs[j] * m.variables[f"x{array_indices[j]}"].isel(dim_0=var_indices[j])
            for j in range(n_terms)
        )
        m.add_constraints(lhs >= rng.uniform(-100, 100), name=f"con{i}")

    # Add an objective using all variables
    m.objective = sum(m.variables[name].sum() for name in var_names)

    return m


def time_function(func: Callable[[], Any], repeats: int, warmup: int = 2) -> Iterable[float]:
    """Time a function over multiple iterations."""
    for _ in range(warmup):
        func()
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        yield time.perf_counter() - start


@contextmanager
def use_original_implementation():
    """
    Context manager to temporarily use the original implementations.

    Monkey-patches:
    - Variables/Constraints.get_label_position to use O(n) linear search
    - print_single_variable to use .sel() instead of direct numpy indexing
    """
    import linopy.common as common_module
    import linopy.constraints as constraints_module
    import linopy.variables as variables_module

    # Store optimized methods
    optimized_var_method = variables_module.Variables.get_label_position
    optimized_con_method = constraints_module.Constraints.get_label_position
    optimized_print_single_variable = common_module.print_single_variable

    # Replace with original O(n) implementation
    def original_var_get_label_position(self, values):
        return _get_label_position_linear(self, values)

    def original_con_get_label_position(self, values):
        return _get_label_position_linear(self, values)

    # Original print_single_variable using .sel()
    def original_print_single_variable(model, label):
        from linopy.common import print_coord

        if label == -1:
            return "None"

        variables = model.variables
        name, coord = _get_label_position_linear(variables, label)

        # Original: use .sel() which is slower
        lower = variables[name].lower.sel(coord).item()
        upper = variables[name].upper.sel(coord).item()

        if variables[name].attrs["binary"]:
            bounds = " ∈ {0, 1}"
        elif variables[name].attrs["integer"]:
            bounds = f" ∈ Z ⋂ [{lower:.4g},...,{upper:.4g}]"
        else:
            bounds = f" ∈ [{lower:.4g}, {upper:.4g}]"

        return f"{name}{print_coord(coord)}{bounds}"

    # Also save the module-level import in variables.py
    optimized_var_print_single_variable = variables_module.print_single_variable

    variables_module.Variables.get_label_position = original_var_get_label_position
    constraints_module.Constraints.get_label_position = original_con_get_label_position
    common_module.print_single_variable = original_print_single_variable
    variables_module.print_single_variable = original_print_single_variable

    try:
        yield
    finally:
        # Restore optimized methods
        variables_module.Variables.get_label_position = optimized_var_method
        constraints_module.Constraints.get_label_position = optimized_con_method
        common_module.print_single_variable = optimized_print_single_variable
        variables_module.print_single_variable = optimized_var_print_single_variable


def document_linopy_model(model: Model) -> dict[str, Any]:
    """
    Convert all model variables and constraints to a structured string representation.
    This can take multiple seconds for large models.
    The output can be saved to a yaml file with readable formatting applied.

    This is a real-world use case that benefits from the get_label_position optimization.
    """
    documentation = {
        'objective': model.objective.__repr__(),
        'termination_condition': model.termination_condition,
        'status': model.status,
        'nvars': model.nvars,
        'ncons': model.ncons,
        'variables': {
            variable_name: variable.__repr__()
            for variable_name, variable in model.variables.items()
        },
        'constraints': {
            constraint_name: constraint.__repr__()
            for constraint_name, constraint in model.constraints.items()
        },
    }
    return documentation


def run_benchmarks(model: Model, repeats: int) -> xr.Dataset:
    """
    Run user-story benchmarks comparing original vs optimized.

    Returns an xarray Dataset with timing results for both implementations.
    """
    results = {}

    first_var_name = list(model.variables)[0]
    first_var = model.variables[first_var_name]
    first_con_name = list(model.constraints)[0]
    first_con = model.constraints[first_con_name]

    # Define user-story benchmark operations (PUBLIC API only)
    user_stories = {
        # Story 1: Document the full model (real-world use case!)
        "document_model": {
            "func": lambda: document_linopy_model(model),
            "description": "document_linopy_model(model)",
            "story": "I want to document my model",
        },
        # Story 2: Inspect the full model
        "print_model": {
            "func": lambda: repr(model),
            "description": "print(model)",
            "story": "I want to inspect my model",
        },
        # Story 3: Print ALL variables
        "print_all_variables": {
            "func": lambda: {
                name: repr(var) for name, var in model.variables.items()
            },
            "description": "{name: repr(v) for name, v in variables.items()}",
            "story": "I want to print all variables",
        },
        # Story 4: Print ALL constraints
        "print_all_constraints": {
            "func": lambda: {
                name: repr(con) for name, con in model.constraints.items()
            },
            "description": "{name: repr(c) for name, c in constraints.items()}",
            "story": "I want to print all constraints",
        },
        # Story 5: Inspect a single variable array
        "print_single_variable": {
            "func": lambda: repr(first_var),
            "description": f"print(model.variables['{first_var_name}'])",
            "story": "I want to inspect a single variable",
        },
        # Story 6: Inspect a single constraint array
        "print_single_constraint": {
            "func": lambda: repr(first_con),
            "description": f"print(model.constraints['{first_con_name}'])",
            "story": "I want to inspect a single constraint",
        },
        # Story 7: See an expression
        "print_expression": {
            "func": lambda: repr(first_var.sum()),
            "description": f"print({first_var_name}.sum())",
            "story": "I want to see an expression",
        },
        # Story 8: See the objective
        "print_objective": {
            "func": lambda: repr(model.objective),
            "description": "print(model.objective)",
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
        f"{ds.attrs['n_constraint_arrays']} constraint arrays"
    )
    print(
        "\nNote: The optimization helps when there are many arrays to search through."
    )
    print("      get_label_position finds which array contains a given label.\n")

    # Extract operation names
    all_vars = list(ds.data_vars)
    operations = sorted(set(v.rsplit("_", 1)[0] for v in all_vars if v.endswith("_original")))

    # Group by user story category
    categories = {
        "Document Model (real-world use case)": ["document_model"],
        "Model Inspection": ["print_model"],
        "Print All": ["print_all_variables", "print_all_constraints"],
        "Single Item Inspection": ["print_single_variable", "print_single_constraint"],
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--var-arrays", type=int, default=100,
        help="Number of variable arrays (default: 100)"
    )
    parser.add_argument(
        "--con-arrays", type=int, default=200,
        help="Number of constraint arrays (default: 200)"
    )
    parser.add_argument(
        "--vars-per-array", type=int, default=10,
        help="Variables per array (default: 10)"
    )
    parser.add_argument(
        "--repeats", type=int, default=5,
        help="Number of repetitions (default: 5)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output NetCDF file"
    )
    args = parser.parse_args()

    print("Building model...")
    print(f"  {args.var_arrays} variable arrays x {args.vars_per_array} vars each")
    print(f"  {args.con_arrays} constraint arrays")
    model = build_model(args.var_arrays, args.con_arrays, args.vars_per_array)

    print("\nRunning user-story benchmarks (original vs optimized)...")
    ds = run_benchmarks(model, args.repeats)
    summary = compute_summary(ds)

    print_results(ds, summary)

    if args.output:
        xr.merge([ds, summary]).to_netcdf(args.output)
        print(f"Results saved to {args.output}")

    return ds, summary


if __name__ == "__main__":
    main()
