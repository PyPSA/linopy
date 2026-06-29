#!/usr/bin/env python3
"""
MultiIndex -> flat dim + aux coords: feasibility for v1 (#744).

Can v1 drop first-class ``pd.MultiIndex`` snapshots for a flat ``snapshot`` dim
with ``period``/``timestep`` as auxiliary level coords? Each test below is an
explicit equality check (MI form vs flat+aux form), run under both legacy and v1
semantics.

The matrix, findings, and pinned PyPSA call sites live in the discussion
artifact: ``arithmetics-design/multiindex-feasibility.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model, available_solvers
from linopy.testing import assert_linequal

PERIODS = [2020, 2030]
N = 6  # 2 periods x 3 timesteps
PERIOD_OF = np.repeat(PERIODS, 3)  # period per flat snapshot
STEP_OF = np.tile(["t1", "t2", "t3"], 2)
DEMAND = {2020: 5.0, 2030: 7.0}

needs_highs = pytest.mark.skipif(
    "highs" not in available_solvers, reason="highs solver not available"
)


def _mi() -> pd.MultiIndex:
    return pd.MultiIndex.from_product(
        [PERIODS, ["t1", "t2", "t3"]], names=["period", "timestep"]
    )


def _flat() -> dict:
    """Flat snapshot dim with period/timestep as aux coords."""
    s = pd.RangeIndex(N, name="snapshot")
    return {
        "snapshot": s,
        "period": xr.DataArray(PERIOD_OF, dims="snapshot", coords={"snapshot": s}),
        "timestep": xr.DataArray(STEP_OF, dims="snapshot", coords={"snapshot": s}),
    }


# --- data-model equivalence (xarray level) ---------------------------------- #
def test_entry_normalize() -> None:
    """reset_index *is* the conversion: levels -> aux coords, dim coordinate-less."""
    r = xr.DataArray(
        np.arange(N), coords={"snapshot": _mi()}, dims="snapshot"
    ).reset_index("snapshot")
    assert list(r.coords) == ["period", "timestep"]
    assert "snapshot" not in r.indexes  # coordinate-less; xarray virtualizes 0..N-1
    assert np.array_equal(r["snapshot"].values, np.arange(N))


def test_level_selection() -> None:
    """where(period == p) reproduces sel(snapshot=(p, slice))."""
    mi = xr.DataArray(np.arange(N), coords={"snapshot": _mi()}, dims="snapshot")
    flat = xr.DataArray(np.arange(N), coords=_flat(), dims="snapshot")
    for p in PERIODS:
        assert np.array_equal(
            mi.sel(snapshot=(p, slice(None))).values,
            flat.where(flat.period == p, drop=True).values,
        )


def test_per_period_roll() -> None:
    """PyPSA SOC pattern: groupby('period').roll == per-period sel-loop (needs #751)."""
    mi = xr.DataArray(np.arange(N), coords={"snapshot": _mi()}, dims="snapshot")
    flat = xr.DataArray(np.arange(N), coords=_flat(), dims="snapshot")
    mi_rolled = np.concatenate(
        [mi.sel(snapshot=(p, slice(None))).roll(snapshot=1).values for p in PERIODS]
    )
    flat_rolled = flat.groupby("period").map(lambda b: b.roll(snapshot=1)).values
    assert np.array_equal(mi_rolled, flat_rolled)


def test_groupby_level_name() -> None:
    """Group a LinearExpression by a level name == the slow fallback (#751)."""
    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(N, name="snapshot")], name="x")
    expr = (1.0 * x).assign_coords(period=xr.DataArray(PERIOD_OF, dims="snapshot"))
    assert_linequal(
        expr.groupby("period").sum(), expr.groupby("period").sum(use_fallback=True)
    )


def test_variable_mi_tuple_sel_not_forwarded() -> None:
    """Logged finding: Variable.sel can't MI-tuple-select -> PyPSA drops to .data (#752 §2)."""
    m = Model()
    x = m.add_variables(coords=[pd.Index(_mi(), name="snapshot")], name="x")
    with pytest.raises(pd.errors.InvalidIndexError):
        x.sel(snapshot=(2020, slice(None)))


# --- model equivalence (linopy level, solved) ------------------------------- #
def _solve(snapshot, add_demand) -> tuple[float, np.ndarray]:
    """min-cost x s.t. per-period demand; constraints built by ``add_demand``."""
    m = Model()
    x = m.add_variables(lower=0, upper=4.0, coords=[snapshot], name="x")
    add_demand(m, x)
    m.add_objective((np.arange(1.0, N + 1.0) * x).sum())
    m.solve(solver_name="highs", output_flag=False)
    return float(m.objective.value), np.sort(m.solution["x"].values)


@needs_highs
def test_per_period_lp_equivalent() -> None:
    """Same per-period-demand LP, MI vs flat+aux -> identical optimum."""

    def mi_demand(m, x):  # select by position -- MI tuple-sel is not forwarded
        for p, d in DEMAND.items():
            m.add_constraints(
                x.isel(snapshot=np.flatnonzero(PERIOD_OF == p)).sum() >= d
            )

    def flat_demand(m, x):  # the level rides as an aux coord -> groupby
        e = (1.0 * x).assign_coords(period=_flat()["period"])
        rhs = xr.DataArray(
            list(DEMAND.values()), dims="period", coords={"period": PERIODS}
        )
        m.add_constraints(e.groupby("period").sum() >= rhs)

    obj_mi, sol_mi = _solve(pd.Index(_mi(), name="snapshot"), mi_demand)
    obj_flat, sol_flat = _solve(_flat()["snapshot"], flat_demand)
    assert np.isclose(obj_mi, obj_flat)
    assert np.allclose(sol_mi, sol_flat)


@needs_highs
def test_output_restacks_to_mi() -> None:
    """
    output: a flat solution re-stacks to MI at the boundary, as PyPSA would.

    linopy returns a bare flat ``snapshot`` dim; the caller reconstructs the MI
    with the mapping it already owns (``n.snapshots``) -- one ``assign_coords``,
    values and order preserved. The inverse of the ``entry`` row.
    """
    m = Model()
    x = m.add_variables(lower=0, upper=4.0, coords=[_flat()["snapshot"]], name="x")
    e = (1.0 * x).assign_coords(period=_flat()["period"])
    rhs = xr.DataArray(list(DEMAND.values()), dims="period", coords={"period": PERIODS})
    m.add_constraints(e.groupby("period").sum() >= rhs)
    m.add_objective((np.arange(1.0, N + 1.0) * x).sum())
    m.solve(solver_name="highs", output_flag=False)

    sol = m.solution["x"]  # bare flat dim, no level coords carried through solve
    coords = xr.Coordinates.from_pandas_multiindex(
        _mi(), "snapshot"
    )  # explicit-index API
    restacked = sol.assign_coords(coords)  # caller's own snapshot mapping
    assert isinstance(restacked.indexes["snapshot"], pd.MultiIndex)
    assert np.array_equal(restacked.values, sol.values)
    assert list(restacked.to_dataframe().index.names) == ["period", "timestep"]
