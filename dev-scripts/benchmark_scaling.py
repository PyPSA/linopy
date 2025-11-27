#!/usr/bin/env python3
"""
Scaling benchmark for linopy printing optimizations.

Tests how performance scales with model size for:
1. Print all variables (user story 2)
2. Print all constraints (user story 3)
3. nvars property
4. ncons property

Results stored as xarray Dataset.

uv run python dev-scripts/benchmark_scaling.py --sizes 1 2 5 10 20 50 100 200 500 1000 --repeats 3 --plot dev-scripts/benchmark_scaling_plot.html
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
import xarray as xr

from linopy import Model
from linopy.common import _get_label_position_linear


def build_model(n_var_arrays: int, n_con_arrays: int, vars_per_array: int = 10) -> Model:
    """Build a model with specified number of variable and constraint arrays."""
    rng = np.random.default_rng(42)
    m = Model()

    for i in range(n_var_arrays):
        m.add_variables(
            lower=0, upper=100, name=f"x{i}",
            coords=[range(vars_per_array)],
        )

    for i in range(n_con_arrays):
        n_terms = min(8, n_var_arrays)
        array_indices = rng.integers(0, max(1, n_var_arrays), size=n_terms)
        var_indices = rng.integers(0, vars_per_array, size=n_terms)
        coeffs = rng.uniform(-10, 10, size=n_terms)

        if n_var_arrays > 0:
            lhs = sum(
                coeffs[j] * m.variables[f"x{array_indices[j]}"].isel(dim_0=var_indices[j])
                for j in range(n_terms)
            )
            m.add_constraints(lhs >= rng.uniform(-100, 100), name=f"con{i}")

    return m


def time_function(func: Callable[[], Any], repeats: int, warmup: int = 2) -> np.ndarray:
    """Time a function over multiple iterations."""
    for _ in range(warmup):
        func()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return np.array(times)


@contextmanager
def use_original_implementation():
    """Context manager to temporarily use the original O(n) implementations."""
    import linopy.common as common_module
    import linopy.constraints as constraints_module
    import linopy.variables as variables_module

    # Store optimized methods
    optimized_var_method = variables_module.Variables.get_label_position
    optimized_con_method = constraints_module.Constraints.get_label_position
    optimized_print_single_variable = common_module.print_single_variable
    optimized_var_print_single_variable = variables_module.print_single_variable

    # Store optimized nvars/ncons
    optimized_nvars = variables_module.Variables.nvars.fget
    optimized_ncons = constraints_module.Constraints.ncons.fget

    # Original O(n) get_label_position
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

        lower = variables[name].lower.sel(coord).item()
        upper = variables[name].upper.sel(coord).item()

        if variables[name].attrs["binary"]:
            bounds = " ∈ {0, 1}"
        elif variables[name].attrs["integer"]:
            bounds = f" ∈ Z ⋂ [{lower:.4g},...,{upper:.4g}]"
        else:
            bounds = f" ∈ [{lower:.4g}, {upper:.4g}]"

        return f"{name}{print_coord(coord)}{bounds}"

    # Original nvars using .flat.labels.unique()
    def original_nvars(self):
        return len(self.flat.labels.unique())

    # Original ncons using .flat.labels.unique()
    def original_ncons(self):
        return len(self.flat.labels.unique())

    # Apply original implementations
    variables_module.Variables.get_label_position = original_var_get_label_position
    constraints_module.Constraints.get_label_position = original_con_get_label_position
    common_module.print_single_variable = original_print_single_variable
    variables_module.print_single_variable = original_print_single_variable
    variables_module.Variables.nvars = property(original_nvars)
    constraints_module.Constraints.ncons = property(original_ncons)

    try:
        yield
    finally:
        # Restore optimized methods
        variables_module.Variables.get_label_position = optimized_var_method
        constraints_module.Constraints.get_label_position = optimized_con_method
        common_module.print_single_variable = optimized_print_single_variable
        variables_module.print_single_variable = optimized_var_print_single_variable
        variables_module.Variables.nvars = property(optimized_nvars)
        constraints_module.Constraints.ncons = property(optimized_ncons)


def run_scaling_benchmark(
    array_sizes: list[int],
    vars_per_array: int = 10,
    repeats: int = 5,
) -> xr.Dataset:
    """
    Run scaling benchmark across different model sizes.

    Parameters
    ----------
    array_sizes : list[int]
        List of array counts to test (used for both var and con arrays).
    vars_per_array : int
        Number of variables per array.
    repeats : int
        Number of timing repetitions.

    Returns
    -------
    xr.Dataset
        Dataset with timing results.
    """
    operations = {
        "print_all_variables": {
            "func": lambda m: {name: repr(var) for name, var in m.variables.items()},
            "description": "Print all variables",
        },
        "print_all_constraints": {
            "func": lambda m: {name: repr(con) for name, con in m.constraints.items()},
            "description": "Print all constraints",
        },
        "nvars": {
            "func": lambda m: m.variables.nvars,
            "description": "Get nvars property",
        },
        "ncons": {
            "func": lambda m: m.constraints.ncons,
            "description": "Get ncons property",
        },
    }

    results = {}

    for n_arrays in array_sizes:
        print(f"\nBuilding model with {n_arrays} arrays...")
        model = build_model(n_arrays, n_arrays, vars_per_array)
        print(f"  {model._xCounter} variables, {model._cCounter} constraints")

        for impl in ["original", "optimized"]:
            print(f"  Running {impl} benchmarks...")

            if impl == "original":
                ctx = use_original_implementation()
            else:
                ctx = contextmanager(lambda: (yield))()

            with ctx:
                for op_name, op_info in operations.items():
                    times = time_function(lambda: op_info["func"](model), repeats)
                    key = (op_name, impl, n_arrays)
                    results[key] = times

    # Build xarray Dataset
    data_vars = {}

    for op_name in operations:
        for impl in ["original", "optimized"]:
            var_name = f"{op_name}_{impl}"
            data = np.array([
                results[(op_name, impl, n)] for n in array_sizes
            ])
            data_vars[var_name] = xr.DataArray(
                data * 1000,  # Convert to milliseconds
                dims=["n_arrays", "repeat"],
                coords={
                    "n_arrays": array_sizes,
                    "repeat": range(repeats),
                },
                attrs={
                    "units": "ms",
                    "description": operations[op_name]["description"],
                    "implementation": impl,
                },
            )

    # Add speedup calculations (median)
    for op_name in operations:
        orig = data_vars[f"{op_name}_original"].median(dim="repeat")
        opt = data_vars[f"{op_name}_optimized"].median(dim="repeat")
        data_vars[f"{op_name}_speedup"] = orig / opt
        data_vars[f"{op_name}_speedup"].attrs = {
            "description": f"Speedup for {operations[op_name]['description']}",
        }

    ds = xr.Dataset(data_vars)
    ds.attrs["vars_per_array"] = vars_per_array
    ds.attrs["repeats"] = repeats

    return ds


OPERATIONS = ["print_all_variables", "print_all_constraints", "nvars", "ncons"]
OP_LABELS = {
    "print_all_variables": "Print all variables",
    "print_all_constraints": "Print all constraints",
    "nvars": "nvars property",
    "ncons": "ncons property",
}


def print_results(ds: xr.Dataset) -> None:
    """Print benchmark results in a formatted table."""
    n_arrays_values = ds.coords["n_arrays"].values

    for op in OPERATIONS:
        print(f"\n{'=' * 80}")
        print(f"{OP_LABELS[op]}")
        print("=" * 80)
        print(f"{'n_arrays':>10s} {'Original (ms)':>15s} {'Optimized (ms)':>15s} {'Speedup':>10s}")
        print("-" * 80)

        orig = ds[f"{op}_original"].median(dim="repeat")
        opt = ds[f"{op}_optimized"].median(dim="repeat")
        speedup = ds[f"{op}_speedup"]

        for i, n in enumerate(n_arrays_values):
            print(f"{n:>10d} {float(orig[i]):>15.2f} {float(opt[i]):>15.2f} {float(speedup[i]):>9.1f}x")

    print("\n" + "=" * 80)
    print("SUMMARY: Speedup at largest model size")
    print("=" * 80)
    max_idx = len(n_arrays_values) - 1
    for op in OPERATIONS:
        speedup = float(ds[f"{op}_speedup"][max_idx])
        print(f"  {OP_LABELS[op]}: {speedup:.1f}x")


def plot_results(ds: xr.Dataset, output_path: str = "benchmark_scaling_plot.html") -> None:
    """Create faceted plot comparing original vs optimized performance."""
    import pandas as pd
    import plotly.express as px

    rows = [
        {
            "operation": OP_LABELS[op],
            "implementation": impl.capitalize(),
            "n_arrays": int(n),
            "time_ms": float(ds[f"{op}_{impl}"].median(dim="repeat").sel(n_arrays=n)),
            "speedup": float(ds[f"{op}_speedup"].sel(n_arrays=n)),
        }
        for op in OPERATIONS
        for impl in ["original", "optimized"]
        for n in ds.coords["n_arrays"].values
    ]
    df = pd.DataFrame(rows)

    # Time comparison plot
    fig1 = px.line(
        df,
        x="n_arrays",
        y="time_ms",
        color="implementation",
        facet_col="operation",
        facet_col_wrap=2,
        markers=True,
        line_dash="implementation",
        line_dash_map={"Original": "dash", "Optimized": "solid"},
        color_discrete_map={"Original": "#EF553B", "Optimized": "#636EFA"},
        labels={"n_arrays": "Number of Arrays", "time_ms": "Time (ms)"},
        title="Linopy Printing Performance: Original vs Optimized",
    )
    fig1.update_layout(height=600, width=1000)
    fig1.update_yaxes(matches=None, showticklabels=True)
    fig1.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))

    # Speedup plot
    speedup_df = df[df["implementation"] == "Original"].copy()
    fig2 = px.line(
        speedup_df,
        x="n_arrays",
        y="speedup",
        color="operation",
        markers=True,
        labels={"n_arrays": "Number of Arrays", "speedup": "Speedup (x)", "operation": "Operation"},
        title="Speedup by Operation",
    )
    fig2.update_layout(height=400, width=1000)
    fig2.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="1x (no improvement)")

    # Combine into single HTML
    with open(output_path, "w") as f:
        f.write(fig1.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write(fig2.to_html(full_html=False, include_plotlyjs=False))

    print(f"\nPlot saved to {output_path}")


def main() -> xr.Dataset:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[10, 25, 50, 100, 200, 500],
        help="Array sizes to test (default: 10 25 50 100 200 500)"
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
    parser.add_argument(
        "--plot", type=str, default=None,
        help="Output HTML plot file"
    )
    args = parser.parse_args()

    print("Running scaling benchmark...")
    print(f"  Array sizes: {args.sizes}")
    print(f"  Vars per array: {args.vars_per_array}")
    print(f"  Repeats: {args.repeats}")

    ds = run_scaling_benchmark(
        array_sizes=args.sizes,
        vars_per_array=args.vars_per_array,
        repeats=args.repeats,
    )

    print_results(ds)

    if args.output:
        ds.to_netcdf(args.output)
        print(f"\nResults saved to {args.output}")

    if args.plot:
        plot_results(ds, args.plot)

    return ds


if __name__ == "__main__":
    ds = main()
