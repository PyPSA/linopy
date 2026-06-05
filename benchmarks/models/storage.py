"""
Storage state-of-charge model — intertemporal coupling via ``.shift()``.

A fleet of storage units, each with a bidiagonal SoC recursion
``soc[t] - decay*soc[t-1] - eff*charge[t] + discharge[t]/eff == 0`` built with
``soc.shift(time=1)`` (``t=0`` falls off as the boundary). This is the one op
family no other model exercises — the ``.shift()``/``.isel()`` intertemporal
coupling that PyPSA's SoC and flixopt's ``charge_state.isel`` recursion lean on.

It is a *model*, not a pattern: each balance row has a fixed ~4 terms regardless
of horizon or unit count, so it scales with ``size`` (units × timesteps) and has
no benign→worst data-shape dial. ``size`` is the number of storage units.
"""

from __future__ import annotations

import pandas as pd

import linopy
from benchmarks.registry import CONTINUOUS, ModelSpec, register

SIZES = (10, 50, 100, 250, 500, 1000)
N_TIME = 168
DECAY = 0.99
ETA = 0.95


def build_storage(n_storage: int) -> linopy.Model:
    storages = pd.RangeIndex(n_storage, name="storage")
    time = pd.RangeIndex(N_TIME, name="time")

    m = linopy.Model()
    soc = m.add_variables(lower=0, upper=100, coords=[storages, time], name="soc")
    charge = m.add_variables(lower=0, upper=50, coords=[storages, time], name="charge")
    discharge = m.add_variables(
        lower=0, upper=50, coords=[storages, time], name="discharge"
    )

    prev = soc.shift(time=1)  # soc[t-1]; t=0 shifted out (initial-SoC boundary)
    m.add_constraints(
        soc - DECAY * prev - ETA * charge + discharge / ETA == 0, name="soc_balance"
    )
    m.add_objective((charge + discharge).sum())
    return m


SPEC = register(
    ModelSpec(
        name="storage",
        build=build_storage,
        sizes=SIZES,
        features=frozenset({CONTINUOUS}),
        quick_threshold=10,
        long_threshold=500,
        description="storage SoC recursion via .shift() — bidiagonal intertemporal coupling",
    )
)
