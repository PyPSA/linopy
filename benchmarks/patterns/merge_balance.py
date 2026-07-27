"""
Ragged merge — concat of mixed-width blocks pads all to the global max (#749).

The documented build peak: a balance assembled by merging sub-expressions of
*different* ``_term`` widths along a shared dim. PyPSA's nodal balance does
``merge(gen + storage + lines + links, join="outer")`` (the single largest
allocation in a SciGRID build); flixopt's bus balance is the sibling
``sum([flow_rate for flow in flows])``. Merging along a non-``_term`` dim makes
linopy align the ``_term`` axes by padding every block to the widest one — so
one fat block leaves the narrow blocks mostly fill. ``severity`` dials the
widest block's term count.
"""

from __future__ import annotations

import pandas as pd

import linopy
from benchmarks.registry import SEVERITIES, BenchSpec, register_pattern

N_BLOCKS = 30
N_ROW = 128  # broadcast/volume dim — the ragged padding is on _term, not row
NARROW = 3
WIDE = 200


def _block(
    m: linopy.Model, rows: pd.Index, name: str, width: int
) -> linopy.LinearExpression:
    """A ``(row,)`` expression with ``width`` terms (a ``(row, k)`` var folded over ``k``)."""
    k = pd.RangeIndex(width, name=f"k_{name}")
    x = m.add_variables(coords=[rows, k], name=name)
    return (1 * x).sum(f"k_{name}")


def build_merge_balance(severity: int) -> linopy.Model:
    rows = pd.RangeIndex(N_ROW, name="row")
    widest = max(NARROW, round(NARROW + severity / 100 * (WIDE - NARROW)))

    m = linopy.Model()
    blocks = [_block(m, rows, f"narrow{i}", NARROW) for i in range(N_BLOCKS - 1)]
    blocks.append(_block(m, rows, "wide", widest))

    lhs = linopy.merge(blocks, dim="block", join="outer")
    m.add_constraints(lhs == 0, name="balance")
    m.add_objective(blocks[0])
    return m


SPEC = register_pattern(
    BenchSpec(
        name="merge_balance",
        build=build_merge_balance,
        sweep=SEVERITIES,
        axis="severity",
    )
)
