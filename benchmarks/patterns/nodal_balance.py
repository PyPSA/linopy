"""
Nodal-balance pattern — grouped-sum padding under bus-connectivity skew (#745).

The idiom: sum each bus's generators (``groupby(bus).sum()``) and balance the
result against demand. ``LinearExpression.groupby(...).sum()`` pads every group
to the largest group's term count, so as generators concentrate on one hub the
result's ``_term`` axis blows up — most of it fill. ``severity`` dials that
skew; the build's peak memory is expected to climb steeply with it on the
current (dense) kernel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

import linopy
from benchmarks.registry import SEVERITIES, BenchSpec, register_pattern

N_GEN = 2000
N_BUS = 50
N_TIME = 8  # broadcast/volume dim — the groupby pathology is on gen, not time


def _bus_of_gen(severity: int) -> np.ndarray:
    """
    Assign each generator to a bus, skewed toward one hub by ``severity``.

    - ``severity == 0``  → round-robin: every bus holds ~``N_GEN / N_BUS``.
    - ``severity == 100`` → bus 0 holds almost all generators.

    The first ``N_BUS`` generators anchor one bus each, so no bus is ever empty
    — the constraint *shape* (``N_BUS`` rows) is fixed across the sweep and only
    the per-group term count (the padding) varies.
    """
    rng = np.random.default_rng(0)
    bus = np.arange(N_GEN) % N_BUS  # uniform baseline
    anchor = np.zeros(N_GEN, dtype=bool)
    anchor[:N_BUS] = True  # pin one generator per bus
    move = (~anchor) & (rng.random(N_GEN) < severity / 100)
    bus[move] = 0  # reassign a severity-fraction of the rest onto the hub
    return bus


def build_nodal_balance(severity: int) -> linopy.Model:
    gens = pd.RangeIndex(N_GEN, name="gen")
    time = pd.RangeIndex(N_TIME, name="time")
    buses = pd.RangeIndex(N_BUS, name="bus")
    rng = np.random.default_rng(1)

    m = linopy.Model()
    gen = m.add_variables(lower=0, coords=[gens, time], name="gen")

    bus_of_gen = pd.Series(_bus_of_gen(severity), index=gens, name="bus")
    supply = (1 * gen).groupby(bus_of_gen).sum()
    demand = xr.DataArray(
        rng.uniform(10.0, 100.0, size=(N_BUS, N_TIME)), coords=[buses, time]
    )
    m.add_constraints(supply == demand, name="balance")
    m.add_objective(gen.sum())
    return m


SPEC = register_pattern(
    BenchSpec(
        name="nodal_balance",
        build=build_nodal_balance,
        sweep=SEVERITIES,
        axis="severity",
    )
)
