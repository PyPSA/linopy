"""
Cumulative-sum fold — ``.cumsum(dim)`` stacks a growing window into ``_term``.

A running total over time — cumulative energy, a rolling budget:
``(1 * x).cumsum("time")``. linopy currently routes ``cumsum`` through
``rolling(window=full_dim)`` (``expressions.py``), so its ``_term`` grows
triangularly to the dim size. It is benchmarked as its own op — not folded into
``rolling`` — because it is a distinct public op and a natural de-densification
target (a prefix sum need not materialise the triangle), so this is the
instrument that would show such a kernel change land. ``severity`` dials the
size of the cumulated dimension.
"""

from __future__ import annotations

import pandas as pd

import linopy
from benchmarks.registry import SEVERITIES, BenchSpec, register_pattern

N_ROW = 64  # broadcast/volume dim — the triangular fold is on t, not row
DIM_MAX = 200


def build_cumsum(severity: int) -> linopy.Model:
    rows = pd.RangeIndex(N_ROW, name="row")
    n = max(2, round(severity / 100 * DIM_MAX))

    m = linopy.Model()
    x = m.add_variables(coords=[rows, pd.RangeIndex(n, name="t")], name="x")
    running = (1 * x).cumsum("t")  # (row, t); _term grows triangularly to n
    m.add_constraints(running == 0, name="cumulative")
    m.add_objective((1 * x).sum())
    return m


SPEC = register_pattern(
    BenchSpec(
        name="cumsum",
        build=build_cumsum,
        sweep=SEVERITIES,
        axis="severity",
    )
)
