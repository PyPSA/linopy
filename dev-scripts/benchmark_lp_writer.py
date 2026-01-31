#!/usr/bin/env python3
"""
Benchmark script for LP file writing performance.

Benchmarks both synthetic models and a realistic PyPSA network model.

Usage:
    python dev-scripts/benchmark_lp_writer.py
"""

import tempfile
import time
from pathlib import Path

import numpy as np
from numpy.random import default_rng

from linopy import Model

rng = default_rng(125)


def basic_model(n: int) -> Model:
    """Create a basic model with 2*n^2 variables and 2*n^2 constraints."""
    m = Model()
    N = np.arange(n)
    x = m.add_variables(coords=[N, N], name="x")
    y = m.add_variables(coords=[N, N], name="y")
    m.add_constraints(x - y >= N, name="c1")
    m.add_constraints(x + y >= 0, name="c2")
    m.add_objective((2 * x).sum() + y.sum())
    return m


def knapsack_model(n: int) -> Model:
    """Create a knapsack model with n binary variables and 1 constraint."""
    m = Model()
    packages = m.add_variables(coords=[np.arange(n)], binary=True)
    weight = rng.integers(1, 100, size=n)
    value = rng.integers(1, 100, size=n)
    m.add_constraints((weight * packages).sum() <= 200)
    m.add_objective(-(value * packages).sum())
    return m


def pypsa_model() -> Model | None:
    """Create a model from the PyPSA SciGrid-DE example network."""
    try:
        import pypsa
    except ImportError:
        return None
    n = pypsa.examples.scigrid_de()
    n.optimize.create_model()
    return n.model


def benchmark_model(label: str, m: Model, iterations: int = 10) -> tuple[float, float]:
    """Benchmark LP file writing for a single model. Returns (mean, std)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Warmup
        m.to_file(Path(tmpdir) / "warmup.lp", progress=False)

        times = []
        for i in range(iterations):
            fn = Path(tmpdir) / f"bench_{i}.lp"
            start = time.perf_counter()
            m.to_file(fn, progress=False)
            times.append(time.perf_counter() - start)

    avg = np.mean(times)
    std = np.std(times)
    print(
        f"  {label:55s} ({m.nvars:>9,} vars, {m.ncons:>9,} cons): "
        f"{avg * 1000:7.1f}ms ± {std * 1000:5.1f}ms"
    )
    return avg, std


def main() -> None:
    iterations = 10
    print(f"LP file writing benchmark ({iterations} iterations each)")
    print("=" * 90)

    print("\nbasic_model (2 x N^2 vars, 2 x N^2 constraints):")
    for n in [50, 100, 200, 500, 1000]:
        benchmark_model(f"N={n}", basic_model(n), iterations)

    print("\nknapsack_model (N binary vars, 1 constraint with N terms):")
    for n in [100, 1000, 10000, 50000, 100000]:
        benchmark_model(f"N={n}", knapsack_model(n), iterations)

    print("\nPyPSA SciGrid-DE (realistic power system model):")
    m = pypsa_model()
    if m is not None:
        benchmark_model("scigrid-de (24 snapshots)", m, iterations)
    else:
        print("  (skipped, pypsa not installed)")


if __name__ == "__main__":
    main()
