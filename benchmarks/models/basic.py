"""Basic benchmark model: 2*N^2 variables and constraints."""

from __future__ import annotations

import linopy

LABEL = "basic N={n}"
SIZES = [{"n": n} for n in [10, 50, 100, 250, 500, 1000, 1600]]
QUICK_SIZES = [{"n": n} for n in [10, 50, 100]]
DESCRIPTION = "2*N^2 vars/cons — simple dense model"


def build(n: int) -> linopy.Model:
    """Build a basic N×N model."""
    m = linopy.Model()
    x = m.add_variables(coords=[range(n), range(n)], dims=["i", "j"], name="x")
    y = m.add_variables(coords=[range(n), range(n)], dims=["i", "j"], name="y")
    m.add_constraints(x + y <= 10, name="upper")
    m.add_constraints(x - y >= -5, name="lower")
    m.add_objective(x.sum() + 2 * y.sum())
    return m
