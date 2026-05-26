"""Sparse network benchmark: variables on mismatched coordinate subsets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

import linopy

SIZES = [10, 50, 100, 250, 500, 1000]


def build_sparse_network(n_buses: int) -> linopy.Model:
    """Build a ring network model with mismatched bus/line coordinate subsets."""
    rng = np.random.default_rng(42)
    n_lines = n_buses  # ring topology
    n_time = min(n_buses, 24)

    buses = pd.RangeIndex(n_buses, name="bus")
    lines = pd.RangeIndex(n_lines, name="line")
    time = pd.RangeIndex(n_time, name="time")

    # Ring topology: line i connects bus i -> bus (i+1) % n
    bus_from = np.arange(n_lines)
    bus_to = (bus_from + 1) % n_buses

    m = linopy.Model()

    # Bus-level variables (bus × time)
    gen = m.add_variables(lower=0, coords=[buses, time], name="gen")

    # Line-level variables (line × time)
    flow = m.add_variables(lower=-100, upper=100, coords=[lines, time], name="flow")

    # Incidence matrix (bus × line): +1 for incoming, -1 for outgoing
    incidence = np.zeros((n_buses, n_lines))
    incidence[bus_to, np.arange(n_lines)] = 1  # incoming
    incidence[bus_from, np.arange(n_lines)] = -1  # outgoing
    incidence_da = xr.DataArray(incidence, coords=[buses, lines])

    # Vectorized flow balance: gen - demand + incidence @ flow == 0
    demand = xr.DataArray(
        rng.uniform(10, 100, size=(n_buses, n_time)), coords=[buses, time]
    )
    net_flow = (flow * incidence_da).sum("line")
    m.add_constraints(gen + net_flow == demand, name="balance")

    m.add_objective(gen.sum())
    return m
