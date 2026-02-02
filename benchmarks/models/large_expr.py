"""Large expression benchmark: many-term expression stress test."""

from __future__ import annotations

import linopy

LABEL = "large_expr N={n_constraints} K={terms_per_constraint}"
SIZES = [
    {"n_constraints": 100, "terms_per_constraint": 100},
    {"n_constraints": 500, "terms_per_constraint": 200},
    {"n_constraints": 1000, "terms_per_constraint": 500},
    {"n_constraints": 2000, "terms_per_constraint": 1000},
    {"n_constraints": 5000, "terms_per_constraint": 1000},
]
QUICK_SIZES = [
    {"n_constraints": 100, "terms_per_constraint": 100},
    {"n_constraints": 500, "terms_per_constraint": 200},
]
DESCRIPTION = "N constraints each summing K variables — expression building stress test"


def build(n_constraints: int, terms_per_constraint: int) -> linopy.Model:
    """Build a model with many-term expressions."""
    m = linopy.Model()

    # Create variables: one per (constraint, term)
    x = m.add_variables(
        lower=0,
        coords=[range(n_constraints), range(terms_per_constraint)],
        dims=["constraint", "term"],
        name="x",
    )

    # Each constraint sums all terms for that constraint index
    expr = x.sum("term")
    m.add_constraints(expr <= 1, name="sum_limit")

    # Objective: sum everything
    m.add_objective(x.sum())
    return m
