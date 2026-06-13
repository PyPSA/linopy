"""
Rolling-window coupling — ``rolling(K).sum()`` stacks K terms into ``_term``.

The *windowed* form of intertemporal coupling (unlike the 1-step storage SoC,
this one has a real density dial): minimum up/down time and windowed energy /
ramp limits sum a variable over a sliding window of K timesteps
(PyPSA ``status.rolling(K).sum()`` for min-up-time, ``constraints.py:450``).
``rolling(K).sum()`` builds a result with **K terms per row** — so the window
width is a clean severity dial. ``severity`` dials K from a single step to the
full horizon.
"""

from __future__ import annotations

import pandas as pd

import linopy
from benchmarks.registry import SEVERITIES, BenchSpec, register_pattern

N_UNIT = 8  # broadcast dim — the window densification is on time, not unit
N_TIME = 1000
MIN_WINDOW = 1


def build_rolling(severity: int) -> linopy.Model:
    units = pd.RangeIndex(N_UNIT, name="unit")
    time = pd.RangeIndex(N_TIME, name="time")
    window = max(MIN_WINDOW, round(MIN_WINDOW + severity / 100 * (N_TIME - MIN_WINDOW)))

    m = linopy.Model()
    status = m.add_variables(lower=0, upper=1, coords=[units, time], name="status")
    # min-up-time style: every K-step window carries at most K active steps.
    windowed = status.rolling(time=window).sum()
    m.add_constraints(windowed <= window, name="window_limit")
    m.add_objective(status.sum())
    return m


SPEC = register_pattern(
    BenchSpec(
        name="rolling",
        build=build_rolling,
        sweep=SEVERITIES,
        axis="severity",
    )
)
