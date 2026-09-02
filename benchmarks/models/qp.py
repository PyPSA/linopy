"""
QP benchmark: continuous quadratic objective on a portfolio-style model.

Decision variables:
    x_i  >= 0   (weight on asset i, continuous)

Constraints:
    sum_i x_i  == 1
    x_i        <= 0.3        (no asset > 30% of portfolio)

Objective:
    minimize  sum_i q_i * x_i^2  -  sum_i r_i * x_i

A pure diagonal quadratic — enough to exercise the QP build / write / matrix
paths without paying for cross-terms. Cross-term coupling needs single-term
factors on both sides (see ``LinearExpression._multiply_by_linear_expression``),
which is awkward to set up cleanly via the public API.
"""

from __future__ import annotations

import numpy as np

import linopy
from benchmarks.registry import (
    DEFAULT_PHASES,
    BenchSpec,
    register,
)

SIZES = (10, 1_000)


def build_qp(n_assets: int) -> linopy.Model:
    rng = np.random.default_rng(42)
    q = rng.uniform(0.5, 2.0, size=n_assets)
    r = rng.uniform(0.05, 0.15, size=n_assets)

    m = linopy.Model()
    x = m.add_variables(
        lower=0,
        upper=0.3,
        coords=[range(n_assets)],
        dims=["asset"],
        name="x",
    )

    m.add_constraints(x.sum() == 1, name="budget")

    m.add_objective((q * x**2).sum() - (r * x).sum())
    return m


SPEC = register(
    BenchSpec(
        name="qp",
        build=build_qp,
        sweep=SIZES,
        phases=DEFAULT_PHASES,
    )
)
