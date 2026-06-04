"""
Fold sum — ``.sum(dim)`` stacks the summed dim into ``_term`` (2→202 cliff).

A global aggregation sums an expression over a dimension, which folds that
dimension's whole size into the result's ``_term`` axis: PyPSA's CO2 /
operational limits ``(p * weightings).sum()`` (global_constraints.py), flixopt's
``flow_rate.sum(['time', 'cluster'])`` for total flow hours. ``severity`` dials
the size of the folded dimension.
"""

from __future__ import annotations

import pandas as pd

import linopy
from benchmarks.registry import PatternSpec, register_pattern

N_ROW = 200
FOLD_MAX = 200


def build_flow_sum(severity: int) -> linopy.Model:
    rows = pd.RangeIndex(N_ROW, name="row")
    fold = max(2, round(severity / 100 * FOLD_MAX))

    m = linopy.Model()
    x = m.add_variables(coords=[rows, pd.RangeIndex(fold, name="fold")], name="x")
    # (row, fold) with one term, folded over ``fold`` → (row,) with ``fold`` terms.
    agg = (1 * x).sum("fold")
    m.add_constraints(agg == 0, name="aggregate")
    m.add_objective((1 * x).sum())
    return m


SPEC = register_pattern(
    PatternSpec(
        name="flow_sum",
        build=build_flow_sum,
        description=(
            "folded-dim size — 0: sum a size-2 dim (nterm 2), "
            "100: sum a size-200 dim (nterm 200)"
        ),
    )
)
