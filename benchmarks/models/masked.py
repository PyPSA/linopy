"""
Masked-variables benchmark: transportation with sparse allowed routes.

A standard transportation LP, but only a sparse subset of (origin, dest) pairs
are valid routes. The ``mask=`` keyword on ``add_variables`` skips the rest,
keeping the variable count sub-quadratic.

Decision variables:
    x[origin, dest] >= 0   continuous, only created for allowed routes

Constraints:
    sum_dest x[o, .]   <= supply[o]
    sum_orig x[., d]   == demand[d]

Objective:
    minimize  sum cost[o, d] * x[o, d]

The mask is dense at small sizes and sparser at large sizes, mimicking
real-world transport networks where each origin only serves a fixed
fan-out regardless of total node count.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

import linopy
from benchmarks.registry import (
    DEFAULT_PHASES,
    BenchSpec,
    register,
)

SIZES = (10, 100)


def build_masked(n: int) -> linopy.Model:
    rng = np.random.default_rng(42)
    origins = np.arange(n)
    dests = np.arange(n)

    # Each origin serves at most ~min(20, n) destinations.
    fan_out = min(20, n)
    mask_np = np.zeros((n, n), dtype=bool)
    for o in range(n):
        # Deterministic fan-out so size determines connectivity.
        targets = rng.choice(n, size=fan_out, replace=False)
        mask_np[o, targets] = True

    mask = xr.DataArray(mask_np, coords=[("origin", origins), ("dest", dests)])
    cost = xr.DataArray(
        rng.uniform(1, 10, size=(n, n)),
        coords=[("origin", origins), ("dest", dests)],
    )

    # Supply scaled so the problem stays feasible at any size:
    # each origin can ship up to ``demand_per_dest * fan_out`` units.
    demand_per_dest = 5.0
    supply_per_origin = demand_per_dest * n  # plenty of slack
    supply = xr.DataArray(np.full(n, supply_per_origin), coords=[("origin", origins)])
    demand = xr.DataArray(np.full(n, demand_per_dest), coords=[("dest", dests)])

    m = linopy.Model()
    x = m.add_variables(
        lower=0,
        coords=[("origin", origins), ("dest", dests)],
        mask=mask,
        name="x",
    )

    m.add_constraints(x.sum("dest") <= supply, name="supply", mask=mask.any("dest"))
    m.add_constraints(x.sum("origin") == demand, name="demand", mask=mask.any("origin"))

    m.add_objective((cost * x).sum())
    return m


SPEC = register(
    BenchSpec(
        name="masked",
        build=build_masked,
        sweep=SIZES,
        phases=DEFAULT_PHASES,
    )
)
