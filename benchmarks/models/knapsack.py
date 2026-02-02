"""Knapsack benchmark model: N binary variables."""

from __future__ import annotations

import numpy as np

import linopy

LABEL = "knapsack N={n}"
SIZES = [{"n": n} for n in [10, 50, 100, 500, 1000]]
QUICK_SIZES = [{"n": n} for n in [10, 50]]
DESCRIPTION = "N binary variables — integer programming stress test"


def build(n: int) -> linopy.Model:
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
