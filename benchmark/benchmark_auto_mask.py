#!/usr/bin/env python3
"""
Benchmark comparing manual masking vs auto_mask for models with NaN coefficients.

This creates a realistic scenario: a multi-period dispatch model where:
- Not all generators are available in all time periods (NaN in capacity bounds)
- Not all transmission lines exist between all regions (NaN in flow limits)
"""

import sys
from pathlib import Path

# Ensure we use the local linopy installation
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
from typing import Any

import numpy as np
import pandas as pd

from linopy import GREATER_EQUAL, Model


def create_nan_data(
    n_generators: int = 500,
    n_periods: int = 100,
    n_regions: int = 20,
    nan_fraction_gen: float = 0.3,  # 30% of generator-period combinations unavailable
    nan_fraction_lines: float = 0.7,  # 70% of region pairs have no direct line
    seed: int = 42,
) -> dict[str, Any]:
    """Create realistic input data with NaN patterns."""
    rng = np.random.default_rng(seed)

    generators = pd.Index(range(n_generators), name="generator")
    periods = pd.Index(range(n_periods), name="period")
    regions = pd.Index(range(n_regions), name="region")

    # Generator capacities - some generators unavailable in some periods (maintenance, etc.)
    gen_capacity = pd.DataFrame(
        rng.uniform(50, 500, size=(n_generators, n_periods)),
        index=generators,
        columns=periods,
    )
    # Set random entries to NaN (generator unavailable)
    nan_mask_gen = rng.random((n_generators, n_periods)) < nan_fraction_gen
    gen_capacity.values[nan_mask_gen] = np.nan

    # Generator costs
    gen_cost = pd.Series(rng.uniform(10, 100, n_generators), index=generators)

    # Generator to region mapping
    gen_region = pd.Series(rng.integers(0, n_regions, n_generators), index=generators)

    # Demand per region per period
    demand = pd.DataFrame(
        rng.uniform(100, 1000, size=(n_regions, n_periods)),
        index=regions,
        columns=periods,
    )

    # Transmission line capacities - sparse network (not all regions connected)
    # Use distinct dimension names to avoid xarray duplicate dimension issues
    regions_from = pd.Index(range(n_regions), name="region_from")
    regions_to = pd.Index(range(n_regions), name="region_to")

    line_capacity = pd.DataFrame(
        np.nan,
        index=regions_from,
        columns=regions_to,
        dtype=float,  # Start with all NaN
    )
    # Only some region pairs have lines
    for i in range(n_regions):
        for j in range(n_regions):
            if i != j and rng.random() > nan_fraction_lines:
                line_capacity.loc[i, j] = rng.uniform(100, 500)

    return {
        "generators": generators,
        "periods": periods,
        "regions": regions,
        "regions_from": regions_from,
        "regions_to": regions_to,
        "gen_capacity": gen_capacity,
        "gen_cost": gen_cost,
        "gen_region": gen_region,
        "demand": demand,
        "line_capacity": line_capacity,
    }


def build_model_manual_mask(data: dict[str, Any]) -> Model:
    """Build model using manual masking (traditional approach)."""
    m = Model()

    generators = data["generators"]
    periods = data["periods"]
    regions = data["regions"]
    regions_from = data["regions_from"]
    regions_to = data["regions_to"]
    gen_capacity = data["gen_capacity"]
    gen_cost = data["gen_cost"]
    gen_region = data["gen_region"]
    demand = data["demand"]
    line_capacity = data["line_capacity"]

    # Generator dispatch variables - manually mask where capacity is NaN
    gen_mask = gen_capacity.notnull()
    dispatch = m.add_variables(
        lower=0,
        upper=gen_capacity,
        coords=[generators, periods],
        name="dispatch",
        mask=gen_mask,
    )

    # Flow variables between regions - manually mask where no line exists
    flow_mask = line_capacity.notnull()
    flow = m.add_variables(
        lower=-line_capacity.abs(),
        upper=line_capacity.abs(),
        coords=[regions_from, regions_to],
        name="flow",
        mask=flow_mask,
    )

    # Energy balance constraint per region per period
    for r in regions:
        gens_in_region = generators[gen_region == r]
        gen_sum = dispatch.loc[gens_in_region, :].sum("generator")

        # Net flow into region
        flow_in = flow.loc[:, r].sum("region_from")
        flow_out = flow.loc[r, :].sum("region_to")

        m.add_constraints(
            gen_sum + flow_in - flow_out,
            GREATER_EQUAL,
            demand.loc[r],
            name=f"balance_r{r}",
        )

    # Objective: minimize generation cost
    obj = (dispatch * gen_cost).sum()
    m.add_objective(obj)

    return m


def build_model_auto_mask(data: dict[str, Any]) -> Model:
    """Build model using auto_mask=True (new approach)."""
    m = Model(auto_mask=True)

    generators = data["generators"]
    periods = data["periods"]
    regions = data["regions"]
    regions_from = data["regions_from"]
    regions_to = data["regions_to"]
    gen_capacity = data["gen_capacity"]
    gen_cost = data["gen_cost"]
    gen_region = data["gen_region"]
    demand = data["demand"]
    line_capacity = data["line_capacity"]

    # Generator dispatch variables - auto-masked where capacity is NaN
    dispatch = m.add_variables(
        lower=0,
        upper=gen_capacity,  # NaN values will be auto-masked
        coords=[generators, periods],
        name="dispatch",
    )

    # Flow variables between regions - auto-masked where no line exists
    flow = m.add_variables(
        lower=-line_capacity.abs(),
        upper=line_capacity.abs(),  # NaN values will be auto-masked
        coords=[regions_from, regions_to],
        name="flow",
    )

    # Energy balance constraint per region per period
    for r in regions:
        gens_in_region = generators[gen_region == r]
        gen_sum = dispatch.loc[gens_in_region, :].sum("generator")

        # Net flow into region
        flow_in = flow.loc[:, r].sum("region_from")
        flow_out = flow.loc[r, :].sum("region_to")

        m.add_constraints(
            gen_sum + flow_in - flow_out,
            GREATER_EQUAL,
            demand.loc[r],
            name=f"balance_r{r}",
        )

    # Objective: minimize generation cost
    obj = (dispatch * gen_cost).sum()
    m.add_objective(obj)

    return m


def build_model_no_mask(data: dict[str, Any]) -> Model:
    """Build model WITHOUT any masking (NaN values left in place)."""
    m = Model()

    generators = data["generators"]
    periods = data["periods"]
    regions = data["regions"]
    regions_from = data["regions_from"]
    regions_to = data["regions_to"]
    gen_capacity = data["gen_capacity"]
    gen_cost = data["gen_cost"]
    gen_region = data["gen_region"]
    demand = data["demand"]
    line_capacity = data["line_capacity"]

    # Generator dispatch variables - NO masking, NaN bounds left in place
    dispatch = m.add_variables(
        lower=0,
        upper=gen_capacity,  # Contains NaN values
        coords=[generators, periods],
        name="dispatch",
    )

    # Flow variables between regions - NO masking
    flow = m.add_variables(
        lower=-line_capacity.abs(),
        upper=line_capacity.abs(),  # Contains NaN values
        coords=[regions_from, regions_to],
        name="flow",
    )

    # Energy balance constraint per region per period
    for r in regions:
        gens_in_region = generators[gen_region == r]
        gen_sum = dispatch.loc[gens_in_region, :].sum("generator")

        # Net flow into region
        flow_in = flow.loc[:, r].sum("region_from")
        flow_out = flow.loc[r, :].sum("region_to")

        m.add_constraints(
            gen_sum + flow_in - flow_out,
            GREATER_EQUAL,
            demand.loc[r],
            name=f"balance_r{r}",
        )

    # Objective: minimize generation cost
    obj = (dispatch * gen_cost).sum()
    m.add_objective(obj)

    return m


def benchmark(
    n_generators: int = 500,
    n_periods: int = 100,
    n_regions: int = 20,
    n_runs: int = 3,
    solve: bool = True,
) -> dict[str, Any]:
    """Run benchmark comparing no masking, manual masking, and auto masking."""
    print("=" * 70)
    print("BENCHMARK: No Masking vs Manual Masking vs Auto-Masking")
    print("=" * 70)
    print("\nModel size:")
    print(f"  - Generators: {n_generators}")
    print(f"  - Time periods: {n_periods}")
    print(f"  - Regions: {n_regions}")
    print(f"  - Potential dispatch vars: {n_generators * n_periods:,}")
    print(f"  - Potential flow vars: {n_regions * n_regions:,}")
    print(f"\nRunning {n_runs} iterations each...\n")

    # Generate data once
    data = create_nan_data(
        n_generators=n_generators,
        n_periods=n_periods,
        n_regions=n_regions,
    )

    # Count NaN entries
    gen_nan_count = data["gen_capacity"].isna().sum().sum()
    gen_total = data["gen_capacity"].size
    line_nan_count = data["line_capacity"].isna().sum().sum()
    line_total = data["line_capacity"].size

    print("NaN statistics:")
    print(
        f"  - Generator capacity: {gen_nan_count:,}/{gen_total:,} "
        f"({100 * gen_nan_count / gen_total:.1f}% NaN)"
    )
    print(
        f"  - Line capacity: {line_nan_count:,}/{line_total:,} "
        f"({100 * line_nan_count / line_total:.1f}% NaN)"
    )
    print()

    # Benchmark NO masking (baseline)
    no_mask_times = []
    for i in range(n_runs):
        start = time.perf_counter()
        m_no_mask = build_model_no_mask(data)
        elapsed = time.perf_counter() - start
        no_mask_times.append(elapsed)
        if i == 0:
            # Can't use nvars directly as it will fail with NaN values
            # Instead count total variable labels (including those with NaN bounds)
            no_mask_nvars = sum(
                m_no_mask.variables[k].labels.size for k in m_no_mask.variables
            )
            no_mask_ncons = m_no_mask.ncons

    # Benchmark manual masking
    manual_times = []
    for i in range(n_runs):
        start = time.perf_counter()
        m_manual = build_model_manual_mask(data)
        elapsed = time.perf_counter() - start
        manual_times.append(elapsed)
        if i == 0:
            manual_nvars = m_manual.nvars
            manual_ncons = m_manual.ncons

    # Benchmark auto masking
    auto_times = []
    for i in range(n_runs):
        start = time.perf_counter()
        m_auto = build_model_auto_mask(data)
        elapsed = time.perf_counter() - start
        auto_times.append(elapsed)
        if i == 0:
            auto_nvars = m_auto.nvars
            auto_ncons = m_auto.ncons

    # Results
    print("-" * 70)
    print("RESULTS: Model Building Time")
    print("-" * 70)

    print("\nNo masking (baseline):")
    print(f"  - Mean time: {np.mean(no_mask_times):.3f}s")
    print(f"  - Variables: {no_mask_nvars:,} (includes NaN-bounded vars)")
    print(f"  - Constraints: {no_mask_ncons:,}")

    print("\nManual masking:")
    print(f"  - Mean time: {np.mean(manual_times):.3f}s")
    print(f"  - Variables: {manual_nvars:,}")
    print(f"  - Constraints: {manual_ncons:,}")
    manual_overhead = np.mean(manual_times) - np.mean(no_mask_times)
    print(f"  - Overhead vs no-mask: {manual_overhead * 1000:+.1f}ms")

    print("\nAuto masking:")
    print(f"  - Mean time: {np.mean(auto_times):.3f}s")
    print(f"  - Variables: {auto_nvars:,}")
    print(f"  - Constraints: {auto_ncons:,}")
    auto_overhead = np.mean(auto_times) - np.mean(no_mask_times)
    print(f"  - Overhead vs no-mask: {auto_overhead * 1000:+.1f}ms")

    # Comparison
    print("\nComparison (Auto vs Manual):")
    speedup = np.mean(manual_times) / np.mean(auto_times)
    diff = np.mean(auto_times) - np.mean(manual_times)
    if speedup > 1:
        print(f"  - Auto-mask is {speedup:.2f}x FASTER than manual")
    else:
        print(f"  - Auto-mask is {1 / speedup:.2f}x SLOWER than manual")
    print(f"  - Time difference: {diff * 1000:+.1f}ms")

    # Verify models are equivalent
    print("\nVerification:")
    print(f"  - Manual == Auto variables: {manual_nvars == auto_nvars}")
    print(f"  - Manual == Auto constraints: {manual_ncons == auto_ncons}")
    print(f"  - Variables masked out: {no_mask_nvars - manual_nvars:,}")

    results = {
        "n_generators": n_generators,
        "n_periods": n_periods,
        "potential_vars": n_generators * n_periods,
        "no_mask_time": np.mean(no_mask_times),
        "manual_time": np.mean(manual_times),
        "auto_time": np.mean(auto_times),
        "nvars": manual_nvars,
        "masked_out": no_mask_nvars - manual_nvars,
    }

    # LP file write benchmark
    print("\n" + "-" * 70)
    print("RESULTS: LP File Write Time & Size")
    print("-" * 70)

    import os
    import tempfile

    # Write LP file for manual masked model
    with tempfile.NamedTemporaryFile(suffix=".lp", delete=False) as f:
        manual_lp_path = f.name
    start = time.perf_counter()
    m_manual.to_file(manual_lp_path)
    manual_write_time = time.perf_counter() - start
    manual_lp_size = os.path.getsize(manual_lp_path) / (1024 * 1024)  # MB
    os.unlink(manual_lp_path)

    # Write LP file for auto masked model
    with tempfile.NamedTemporaryFile(suffix=".lp", delete=False) as f:
        auto_lp_path = f.name
    start = time.perf_counter()
    m_auto.to_file(auto_lp_path)
    auto_write_time = time.perf_counter() - start
    auto_lp_size = os.path.getsize(auto_lp_path) / (1024 * 1024)  # MB
    os.unlink(auto_lp_path)

    print("\nManual masking:")
    print(f"  - Write time: {manual_write_time:.3f}s")
    print(f"  - File size:  {manual_lp_size:.2f} MB")

    print("\nAuto masking:")
    print(f"  - Write time: {auto_write_time:.3f}s")
    print(f"  - File size:  {auto_lp_size:.2f} MB")

    print(f"\nFiles identical: {abs(manual_lp_size - auto_lp_size) < 0.01}")

    results["manual_write_time"] = manual_write_time
    results["auto_write_time"] = auto_write_time
    results["lp_size_mb"] = manual_lp_size

    # Quick solve comparison
    if solve:
        print("\n" + "-" * 70)
        print("RESULTS: Solve Time (single run)")
        print("-" * 70)

        start = time.perf_counter()
        m_manual.solve("highs", io_api="direct")
        manual_solve = time.perf_counter() - start

        start = time.perf_counter()
        m_auto.solve("highs", io_api="direct")
        auto_solve = time.perf_counter() - start

        print(f"\nManual masking solve: {manual_solve:.3f}s")
        print(f"Auto masking solve:   {auto_solve:.3f}s")

        if m_manual.objective.value is not None and m_auto.objective.value is not None:
            print(
                f"Objective values match: "
                f"{np.isclose(m_manual.objective.value, m_auto.objective.value)}"
            )
            print(f"  - Manual: {m_manual.objective.value:.2f}")
            print(f"  - Auto:   {m_auto.objective.value:.2f}")

    return results


def benchmark_code_simplicity() -> None:
    """Show the code simplicity benefit of auto_mask."""
    print("\n" + "=" * 70)
    print("CODE COMPARISON: Manual vs Auto-Mask")
    print("=" * 70)

    manual_code = """
# Manual masking - must create mask explicitly
gen_mask = gen_capacity.notnull()
dispatch = m.add_variables(
    lower=0,
    upper=gen_capacity,
    coords=[generators, periods],
    name="dispatch",
    mask=gen_mask,  # Extra step required
)
"""

    auto_code = """
# Auto masking - just pass the data with NaN
m = Model(auto_mask=True)
dispatch = m.add_variables(
    lower=0,
    upper=gen_capacity,  # NaN auto-masked
    coords=[generators, periods],
    name="dispatch",
)
"""

    print("\nManual masking approach:")
    print(manual_code)
    print("Auto-mask approach:")
    print(auto_code)
    print("Benefits of auto_mask:")
    print("  - Less boilerplate code")
    print("  - No need to manually track which arrays have NaN")
    print("  - Reduces risk of forgetting to mask")
    print("  - Cleaner, more declarative style")


def benchmark_constraint_masking(n_runs: int = 3) -> None:
    """Benchmark auto-masking of constraints with NaN in RHS."""
    print("\n" + "=" * 70)
    print("BENCHMARK: Constraint Auto-Masking (NaN in RHS)")
    print("=" * 70)

    n_vars = 1000
    n_constraints = 5000
    nan_fraction = 0.3

    rng = np.random.default_rng(42)
    idx = pd.Index(range(n_vars), name="i")
    con_idx = pd.Index(range(n_constraints), name="c")

    # Create RHS with NaN values
    rhs = pd.Series(rng.uniform(1, 100, n_constraints), index=con_idx)
    nan_mask = rng.random(n_constraints) < nan_fraction
    rhs.values[nan_mask] = np.nan

    print("\nModel size:")
    print(f"  - Variables: {n_vars}")
    print(f"  - Potential constraints: {n_constraints}")
    print(f"  - NaN in RHS: {nan_mask.sum()} ({100 * nan_fraction:.0f}%)")
    print(f"\nRunning {n_runs} iterations each...\n")

    # Manual masking
    manual_times = []
    for i in range(n_runs):
        start = time.perf_counter()
        m = Model()
        x = m.add_variables(lower=0, coords=[idx], name="x")
        coeffs = pd.DataFrame(
            rng.uniform(0.1, 1, (n_constraints, n_vars)), index=con_idx, columns=idx
        )
        con_mask = rhs.notnull()  # Manual mask creation
        m.add_constraints((coeffs * x).sum("i"), GREATER_EQUAL, rhs, mask=con_mask)
        m.add_objective(x.sum())
        elapsed = time.perf_counter() - start
        manual_times.append(elapsed)
        if i == 0:
            manual_ncons = m.ncons

    # Auto masking
    auto_times = []
    for i in range(n_runs):
        start = time.perf_counter()
        m = Model(auto_mask=True)
        x = m.add_variables(lower=0, coords=[idx], name="x")
        coeffs = pd.DataFrame(
            rng.uniform(0.1, 1, (n_constraints, n_vars)), index=con_idx, columns=idx
        )
        m.add_constraints((coeffs * x).sum("i"), GREATER_EQUAL, rhs)  # No mask needed
        m.add_objective(x.sum())
        elapsed = time.perf_counter() - start
        auto_times.append(elapsed)
        if i == 0:
            auto_ncons = m.ncons

    print("-" * 70)
    print("RESULTS: Constraint Building Time")
    print("-" * 70)
    print("\nManual masking:")
    print(f"  - Mean time: {np.mean(manual_times):.3f}s")
    print(f"  - Active constraints: {manual_ncons:,}")

    print("\nAuto masking:")
    print(f"  - Mean time: {np.mean(auto_times):.3f}s")
    print(f"  - Active constraints: {auto_ncons:,}")

    overhead = np.mean(auto_times) - np.mean(manual_times)
    print(f"\nOverhead: {overhead * 1000:.1f}ms")
    print(f"Same constraint count: {manual_ncons == auto_ncons}")


def print_summary_table(results: list[dict[str, Any]]) -> None:
    """Print a summary table of all benchmark results."""
    print("\n" + "=" * 110)
    print("SUMMARY TABLE: Model Building & LP Write Times")
    print("=" * 110)
    print(
        f"{'Model':<12} {'Pot.Vars':>10} {'Act.Vars':>10} {'Masked':>8} "
        f"{'No-Mask':>9} {'Manual':>9} {'Auto':>9} {'Diff':>8} "
        f"{'LP Write':>9} {'LP Size':>9}"
    )
    print("-" * 110)
    for r in results:
        name = f"{r['n_generators']}x{r['n_periods']}"
        lp_write = r.get("manual_write_time", 0) * 1000
        lp_size = r.get("lp_size_mb", 0)
        print(
            f"{name:<12} {r['potential_vars']:>10,} {r['nvars']:>10,} "
            f"{r['masked_out']:>8,} {r['no_mask_time'] * 1000:>8.0f}ms "
            f"{r['manual_time'] * 1000:>8.0f}ms {r['auto_time'] * 1000:>8.0f}ms "
            f"{(r['auto_time'] - r['manual_time']) * 1000:>+7.0f}ms "
            f"{lp_write:>8.0f}ms {lp_size:>8.1f}MB"
        )
    print("-" * 110)
    print("Pot.Vars = Potential variables, Act.Vars = Active (non-masked) variables")
    print("Masked = Variables masked out due to NaN bounds")
    print("Diff = Auto-mask time minus Manual mask time (negative = faster)")
    print("LP Write = Time to write LP file, LP Size = LP file size in MB")


if __name__ == "__main__":
    all_results = []

    # Run benchmarks with different sizes
    print("\n### SMALL MODEL ###")
    all_results.append(
        benchmark(n_generators=100, n_periods=50, n_regions=10, n_runs=5, solve=False)
    )

    print("\n\n### MEDIUM MODEL ###")
    all_results.append(
        benchmark(n_generators=500, n_periods=100, n_regions=20, n_runs=3, solve=False)
    )

    print("\n\n### LARGE MODEL ###")
    all_results.append(
        benchmark(n_generators=1000, n_periods=200, n_regions=30, n_runs=3, solve=False)
    )

    print("\n\n### VERY LARGE MODEL ###")
    all_results.append(
        benchmark(n_generators=2000, n_periods=500, n_regions=40, n_runs=3, solve=False)
    )

    print("\n\n### EXTRA LARGE MODEL ###")
    all_results.append(
        benchmark(n_generators=5000, n_periods=500, n_regions=50, n_runs=2, solve=False)
    )

    # Print summary table
    print_summary_table(all_results)

    # Run constraint benchmark
    benchmark_constraint_masking()

    # Show code comparison
    benchmark_code_simplicity()
