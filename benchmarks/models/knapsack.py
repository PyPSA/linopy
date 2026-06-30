"""Knapsack benchmark model: N binary variables, 1 constraint (MILP, binary)."""

from __future__ import annotations

import numpy as np

import linopy
from benchmarks.registry import DEFAULT_PHASES, BenchSpec, register

SIZES = (100, 10_000)


def build_knapsack(n: int) -> linopy.Model:
    """Build a knapsack model with N items."""
    rng = np.random.default_rng(42)
    weights = rng.integers(1, 100, size=n)
    values = rng.integers(1, 100, size=n)
    capacity = int(weights.sum() * 0.5)

    m = linopy.Model()
    x = m.add_variables(coords=[range(n)], dims=["item"], binary=True, name="x")
    m.add_constraints((x * weights).sum() <= capacity, name="capacity")
    m.add_objective(-(x * values).sum())
    return m


SPEC = register(
    BenchSpec(
        name="knapsack",
        build=build_knapsack,
        sweep=SIZES,
        phases=DEFAULT_PHASES,  # HiGHS handles binary; matrices handles MILP
    )
)
