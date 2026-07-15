"""
KVL-cycles pattern — sparse ``@`` densifies the result to a full ``_term`` (#748).

The idiom: contract a per-branch flow against a (branch × cycle) cycle matrix —
Kirchhoff's voltage law, ``flow @ C``. ``__matmul__`` is ``(flow * C).sum(...)``,
which stacks *every* branch into ``_term`` regardless of whether ``C`` is zero
there. ``severity`` dials ``C``'s sparsity: at 0 it is dense (every branch in
every cycle — nothing to gain), at 100 only ~3 branches per cycle carry a
nonzero (the real grid shape), yet the current kernel still produces
``_term == n_branch``. So the *cost is flat* across severity on today's kernel
— the win from a sparse-aware ``@`` is what grows with it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

import linopy
from benchmarks.registry import SEVERITIES, BenchSpec, register_pattern

N_BRANCH = 300
N_CYCLE = 100
N_TIME = 168  # snapshot horizon — sets the always-paid flat level (the
# densification width is branch; severity dials C's sparsity, which today's
# kernel ignores, so memory stays flat across severity)
MIN_PER_CYCLE = 3


def _cycle_matrix(severity: int, branches: pd.Index, cycles: pd.Index) -> xr.DataArray:
    """
    Branch×cycle incidence whose density falls as ``severity`` rises.

    - ``severity == 0``  → dense: every branch participates in every cycle.
    - ``severity == 100`` → ~``MIN_PER_CYCLE`` branches per cycle (real KVL).

    Entries are ±1. The number of nonzeros per cycle interpolates linearly
    between ``N_BRANCH`` (dense) and ``MIN_PER_CYCLE`` (sparse).
    """
    rng = np.random.default_rng(0)
    n_branch = len(branches)
    per_cycle = round(n_branch - severity / 100 * (n_branch - MIN_PER_CYCLE))
    per_cycle = max(MIN_PER_CYCLE, per_cycle)
    c_mat = np.zeros((n_branch, len(cycles)))
    for col in range(len(cycles)):
        idx = rng.choice(n_branch, size=per_cycle, replace=False)
        c_mat[idx, col] = rng.choice([-1.0, 1.0], size=per_cycle)
    return xr.DataArray(c_mat, coords=[branches, cycles])


def build_kvl_cycles(severity: int) -> linopy.Model:
    branches = pd.RangeIndex(N_BRANCH, name="branch")
    cycles = pd.RangeIndex(N_CYCLE, name="cycle")
    time = pd.RangeIndex(N_TIME, name="time")

    m = linopy.Model()
    flow = m.add_variables(lower=-100, upper=100, coords=[time, branches], name="flow")
    cycle_matrix = _cycle_matrix(severity, branches, cycles)
    kvl = (flow * cycle_matrix).sum("branch")
    m.add_constraints(kvl == 0.0, name="kvl")
    m.add_objective(flow.sum())
    return m


SPEC = register_pattern(
    BenchSpec(
        name="kvl_cycles",
        build=build_kvl_cycles,
        sweep=SEVERITIES,
        axis="severity",
    )
)
