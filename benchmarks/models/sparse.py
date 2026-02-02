"""Sparse topology benchmark: ring network with bus balance constraints."""

from __future__ import annotations

import numpy as np
import xarray as xr

import linopy

LABEL = "sparse N={n_buses} T={n_time}"
SIZES = [
    {"n_buses": 50, "n_time": 50},
    {"n_buses": 100, "n_time": 100},
    {"n_buses": 250, "n_time": 250},
    {"n_buses": 500, "n_time": 500},
    {"n_buses": 1000, "n_time": 1000},
    {"n_buses": 1600, "n_time": 1600},
]
QUICK_SIZES = [
    {"n_buses": 50, "n_time": 50},
    {"n_buses": 100, "n_time": 100},
]
DESCRIPTION = "Sparse ring network — exercises outer-join alignment"


def build(n_buses: int, n_time: int) -> linopy.Model:
    """
    Build a ring-topology network model.

    N buses connected in a ring, each with generation and demand.
    Flow variables on each line connect adjacent buses.
    """
    m = linopy.Model()

    buses = range(n_buses)
    time = range(n_time)
    # Ring topology: line i connects bus i to bus (i+1) % n_buses
    n_lines = n_buses
    lines = range(n_lines)

    gen = m.add_variables(
        lower=0, coords=[buses, time], dims=["bus", "time"], name="gen"
    )
    flow = m.add_variables(coords=[lines, time], dims=["line", "time"], name="flow")

    # Flow capacity
    m.add_constraints(flow <= 100, name="flow_upper")
    m.add_constraints(flow >= -100, name="flow_lower")

    # Bus balance: gen[b] + inflow[b] - outflow[b] = demand[b]
    # In a ring: line b-1 flows into bus b, line b flows out of bus b
    # Rename line→bus so dimensions align for vectorized constraint
    inflow = flow.roll(line=1).assign_coords(line=list(buses)).rename(line="bus")
    outflow = flow.assign_coords(line=list(buses)).rename(line="bus")
    balance = gen + inflow - outflow

    rng = np.random.default_rng(42)
    demand = xr.DataArray(
        rng.uniform(10, 50, size=(n_buses, n_time)),
        coords=[list(buses), list(time)],
        dims=["bus", "time"],
    )
    m.add_constraints(balance == demand, name="balance")

    # Generation cost
    m.add_objective(gen.sum("time"))
    return m
