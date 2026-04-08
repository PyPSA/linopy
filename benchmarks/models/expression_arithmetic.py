"""Expression arithmetic benchmark: stress-tests +, *, sum, broadcasting."""

from __future__ import annotations

import numpy as np

import linopy

SIZES = [10, 50, 100, 250, 500, 1000]


def build_expression_arithmetic(n: int) -> linopy.Model:
    """Build a model that exercises expression arithmetic heavily."""
    m = linopy.Model()

    # Variables on different dimensions to trigger broadcasting
    x = m.add_variables(coords=[range(n), range(n)], dims=["i", "j"], name="x")
    y = m.add_variables(coords=[range(n)], dims=["i"], name="y")
    z = m.add_variables(coords=[range(n)], dims=["j"], name="z")

    # Expression arithmetic: broadcasting y (dim i) and z (dim j) against x (dim i,j)
    coeffs = np.linspace(-1, 1, n * n).reshape(n, n)
    expr1 = x * coeffs + y - z
    expr2 = 2 * x - 3 * y + z
    combined = expr1 + expr2

    m.add_constraints(combined <= 100, name="combined")
    m.add_constraints(expr1.sum("j") >= -10, name="row_sum")
    m.add_objective(combined.sum())
    return m
