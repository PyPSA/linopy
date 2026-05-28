"""Knapsack benchmark model: N binary variables, 1 constraint (MILP, binary)."""

from __future__ import annotations

import numpy as np

import linopy
from benchmarks.registry import BINARY, DEFAULT_PHASES, ModelSpec, register

SIZES = (100, 1_000, 10_000, 100_000, 1_000_000)


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
    ModelSpec(
        name="knapsack",
        build=build_knapsack,
        sizes=SIZES,
        features=frozenset({BINARY}),
        phases=DEFAULT_PHASES,  # HiGHS handles binary; matrices handles MILP
        quick_threshold=100,
        long_threshold=10_000,
    )
)
