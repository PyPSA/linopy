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


# --- period-start boundary: the per-period roll composed into a constraint ---- #
@pytest.mark.parametrize("boundary", ["cyclic", "non-cyclic", "ramp"])
def test_period_boundary_lp_identical(tmp_path, boundary) -> None:
    """
    Per-period roll, composed into a constraint: flat+aux builds the byte-identical
    LP to an explicit per-period roll, for every period boundary PyPSA spells.

    ``soc[t] = soc[t-1] + charge[t] - demand[t]`` with the previous value rolled per
    period; only the *period start* differs (constraints.py @ v1.2.4):

    * ``cyclic`` -- wrap to the period's last step (``cyclic_state_of_charge``).
    * ``non-cyclic`` -- drop the previous *term*, keep the row, initial SOC -> rhs:
      PyPSA's ``previous_soc...roll(1).where(include_previous_soc)`` (L1691). A
      ``where`` on the term, *not* a dropped row.
    * ``ramp`` -- drop the whole *row* (no previous to ramp from): the
      ``_period_start_mask`` used as a constraint ``mask`` (L838).

    ``flat`` builds the previous SOC with ``groupby("period").roll``; ``oracle`` with
    an explicit per-period positional roll. A byte-equal LP file is the whole proof.

    Teeth: each boundary differs from the ``cyclic`` baseline (mask/wrap/row-drop is
    not a no-op); and for ``cyclic`` a period-unaware global roll diverges at the
    period-start rows (the other two mask those rows, so it is no control there).
    """
    s = pd.RangeIndex(N, name="snapshot")
    period = xr.DataArray(PERIOD_OF, dims="snapshot", coords={"snapshot": s})
    demand = xr.DataArray(
        [1.0, 3.0, 2.0, 2.0, 1.0, 3.0], dims="snapshot", coords={"snapshot": s}
    )
    starts = [int(np.flatnonzero(PERIOD_OF == p)[0]) for p in PERIODS]
    is_start = xr.DataArray(
        np.isin(np.arange(N), starts), dims="snapshot", coords={"snapshot": s}
    )
    prev_pos = np.empty(N, int)  # per-period cyclic-previous position
    for p in PERIODS:
        pos = np.flatnonzero(PERIOD_OF == p)
        prev_pos[pos] = np.roll(pos, 1)

    def lp(kind: str, bnd: str) -> str:
        m = Model()
        soc = m.add_variables(lower=0, coords=[s], name="soc")
        charge = m.add_variables(lower=0, coords=[s], name="charge")
        if kind == "flat":  # per-period roll falls out of groupby
            prev = (1.0 * soc).assign_coords(period=period).groupby("period")
            prev = prev.roll(snapshot=1)
            if "period" in prev.coords:  # legacy keeps the aux coord, v1 consumes it
                prev = prev.drop_vars("period")
        elif kind == "oracle":  # explicit per-period positional roll
            prev = (1.0 * soc).isel(snapshot=prev_pos).assign_coords(snapshot=s)
        else:  # period-unaware global roll -- the wrong build
            prev = (1.0 * soc).roll(snapshot=1)
        if bnd == "non-cyclic":  # drop the previous term, keep the row (PyPSA .where)
            prev = prev.where(~is_start)
        con = 1.0 * soc - prev - 1.0 * charge == -demand
        if bnd == "ramp":  # drop the whole row at the period start (constraint mask)
            m.add_constraints(con, name="soc", mask=~is_start)
        else:
            m.add_constraints(con, name="soc")
        path = tmp_path / f"{kind}_{bnd}.lp"
        m.to_file(path, io_api="lp")
        return path.read_text()

    contrast = {"cyclic": "non-cyclic", "non-cyclic": "cyclic", "ramp": "cyclic"}
    assert lp("flat", boundary) == lp("oracle", boundary)  # flat+aux == explicit roll
    assert lp("flat", boundary) != lp("flat", contrast[boundary])  # boundary is real
    if boundary == "cyclic":  # period-unaware roll diverges (masked away otherwise)
        assert lp("global", "cyclic") != lp("oracle", "cyclic")


# --- snapshots param: the MI PyPSA parks inside a linopy object --------------- #
def test_snapshots_param_flat_rebuild() -> None:
    """
    Snapshots param: the snapshot index PyPSA stores on ``model.parameters`` need
    not be an MI -- a flat `snapshot` + `period`/`timestep` aux vars rebuild it.

    PyPSA parks `model.parameters.assign(snapshots=sns)` and reads it back with
    `parameters.snapshots.to_index()` (optimize.py L689 / L905). Today that lands a
    real `MultiIndex` *inside* the linopy object; flat+aux parks only flat
    `snapshot` + level vars and rebuilds the identical index on demand.
    """
    mi = _mi()

    # MI way (PyPSA today): the MI lives inside model.parameters, read back verbatim
    m = Model()
    m.parameters = m.parameters.assign(snapshots=mi)  # optimize.py L689
    assert isinstance(m.parameters.indexes["snapshots"], pd.MultiIndex)
    assert m.parameters.snapshots.to_index().equals(mi)  # optimize.py L905

    # flat+aux way: park flat snapshot + level vars; no MI inside the object
    m2 = Model()
    flat = _flat()
    m2.parameters = m2.parameters.assign(
        period=flat["period"], timestep=flat["timestep"]
    )
    assert isinstance(m2.parameters.indexes["snapshot"], pd.RangeIndex)
    assert "snapshots" not in m2.parameters  # no MI parked inside the linopy object
    rebuilt = pd.MultiIndex.from_arrays(
        [m2.parameters.period.values, m2.parameters.timestep.values],
        names=["period", "timestep"],
    )
    assert rebuilt.equals(mi)  # same index, rebuilt from the aux vars
