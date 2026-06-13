"""
Piecewise-linear benchmark: generation with piecewise fuel-cost curves.

Each generator has a piecewise fuel cost curve pinned via
``add_piecewise_formulation``. The default ``method="auto"`` picks an
SOS2 or incremental expansion, generating auxiliary variables and
constraints — that overhead is what we want to measure.

Decision variables:
    power[gen]  in [0, 100]      (continuous)
    fuel[gen]   in [0, inf)      (continuous, pinned to piecewise curve)

Constraints:
    sum_gen power[gen]  >=  demand
    piecewise:  fuel[gen] = f(power[gen])    for each gen

Objective:
    minimize  sum_gen fuel[gen]
"""

from __future__ import annotations

import warnings

import linopy
from benchmarks.registry import (
    DEFAULT_PHASES,
    BenchSpec,
    register,
)

SIZES = (10, 1_000)

_API_AVAILABLE = hasattr(linopy.Model, "add_piecewise_formulation") and hasattr(
    linopy, "EvolvingAPIWarning"
)


def build_piecewise(n_gens: int) -> linopy.Model:
    # Shared breakpoints, broadcast across generators.
    x_pts = [0.0, 30.0, 60.0, 100.0]
    y_pts = [0.0, 36.0, 84.0, 170.0]  # convex-ish fuel curve

    m = linopy.Model()
    power = m.add_variables(
        lower=0,
        upper=100,
        coords=[range(n_gens)],
        dims=["gen"],
        name="power",
    )
    fuel = m.add_variables(
        lower=0,
        coords=[range(n_gens)],
        dims=["gen"],
        name="fuel",
    )

    demand = 0.5 * n_gens * x_pts[-1]
    m.add_constraints(power.sum() >= demand, name="demand")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=linopy.EvolvingAPIWarning)
        m.add_piecewise_formulation(
            (power, x_pts),
            (fuel, y_pts),
        )

    m.add_objective(fuel.sum())
    return m


# ``add_piecewise_formulation`` is a recent (still-evolving) API. Skip
# registration silently on older linopy so the rest of the suite stays usable.
SPEC: BenchSpec | None
if _API_AVAILABLE:
    SPEC = register(
        BenchSpec(
            name="piecewise",
            build=build_piecewise,
            sweep=SIZES,
            # Monotonic breakpoints + ``method="auto"`` → incremental
            # reformulation (pure MILP with binaries), which every supported
            # solver handles.
            phases=DEFAULT_PHASES,
        )
    )
else:
    SPEC = None
