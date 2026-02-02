"""Sparse topology benchmark: ring network with bus balance constraints."""

from __future__ import annotations

import numpy as np

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

    # Bus balance: gen[b] + inflow - outflow = demand[b]
    rng = np.random.default_rng(42)
    demand = rng.uniform(10, 50, size=(n_buses, n_time))

    for b in buses:
        # Lines into bus b: line (b-1) % n_buses flows into b
        # Lines out of bus b: line b flows out of b
        line_in = (b - 1) % n_buses
        line_out = b
        balance = gen.sel(bus=b) + flow.sel(line=line_in) - flow.sel(line=line_out)
        m.add_constraints(balance == demand[b], name=f"balance_{b}")

    # Generation cost (sum over time first, then weight by bus cost)
    m.add_objective(gen.sum("time"))
    return m
