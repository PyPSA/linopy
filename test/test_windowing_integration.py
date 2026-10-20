#!/usr/bin/env python3
"""
Auto-executed windowing-equivalence integration tests.

Each case registers a pre-existing linopy model — from the test suite, the
examples module or the benchmark suite — via the :func:`windowed` decorator.
A single parametrized driver then runs every case under both pattern modes
(strict and union), with the suite's ``semantics`` fixture automatically
multiplying everything by legacy and v1 arithmetic, and assesses equivalence
against the monolithic solve:

* **decomposable** cases (no cross-tile terms): the chunked objectives sum to
  the monolithic objective, and primal and dual values match per position;
* **replay** cases (coupled models): the :meth:`Model.advance` chain must be
  indistinguishable from re-solving every window as a fresh
  :meth:`Model.restrict` with the same carried fix sequence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from linopy import Model, available_solvers
from linopy.examples import benchmark_model

needs_highs = pytest.mark.skipif(
    "highs" not in available_solvers, reason="highs solver not available"
)


@dataclass(frozen=True)
class WindowCase:
    """A model builder registered for windowing-equivalence testing."""

    name: str
    build: Callable[[], Model]
    dim: str
    tile: int
    coupling: Literal["decomposable", "replay"]
    fix_vars: tuple[str, ...] = ()


CASES: list[WindowCase] = []


def windowed(
    dim: str,
    tile: int,
    coupling: Literal["decomposable", "replay"],
    fix_vars: tuple[str, ...] = (),
) -> Callable[[Callable[[], Model]], Callable[[], Model]]:
    """Register a model builder as a windowing-equivalence case."""

    def deco(build: Callable[[], Model]) -> Callable[[], Model]:
        CASES.append(WindowCase(build.__name__, build, dim, tile, coupling, fix_vars))
        return build

    return deco


# ---------------------------------------------------------------------------
# cases: pre-existing models, sources cited
# ---------------------------------------------------------------------------


@windowed(
    dim="snapshot",
    tile=3,
    coupling="decomposable",
    fix_vars=("x",),
)
def mi_period_demand() -> Model:
    """
    From ``test/test_mi_feasibility.py``: per-period demand via groupby-sum.
    Windows align with the period boundaries; a period's demand row leaks
    into later windows only as a satisfied folded constant.
    """
    m = Model()
    x = m.add_variables(
        lower=0, upper=4.0, coords=[pd.RangeIndex(6, name="snapshot")], name="x"
    )
    period = xr.DataArray(
        np.repeat([2020, 2030], 3),
        dims="snapshot",
        coords={"snapshot": x.indexes["snapshot"]},
    )
    e = (1.0 * x).assign_coords(period=period)
    rhs = xr.DataArray([5.0, 7.0], dims="period", coords={"period": [2020, 2030]})
    m.add_constraints(e.groupby("period").sum() >= rhs)
    m.add_objective((np.arange(1.0, 7.0) * x).sum())
    return m


@windowed(dim="time", tile=1, coupling="decomposable")
def constraint_empty_rows() -> Model:
    """From ``test/test_constraint.py``: an empty constraint group solves."""
    m = Model()
    x = m.add_variables(
        lower=0.0, coords=[range(3), range(2)], dims=["time", "product"], name="x"
    )
    m.add_constraints(x.isel(time=range(1, 1)) == 0, name="empty")
    m.add_objective(x.sum())
    return m


@windowed(dim="dim_0", tile=5, coupling="decomposable")
def examples_benchmark() -> Model:
    """From ``linopy/examples.py``: the JOSS benchmark model."""
    return benchmark_model()


@windowed(
    dim="time",
    tile=24,
    coupling="replay",
    fix_vars=("soc",),
)
def benchmarks_storage() -> Model:
    """
    Replicated from ``benchmarks/models/storage.py`` (not importable from
    the test tree): the 168h shift-coupled SoC fleet — a real, non-test toy
    model with intertemporal coupling.
    """
    m = Model()
    storages = pd.RangeIndex(10, name="storage")
    time = pd.RangeIndex(168, name="time")
    soc = m.add_variables(lower=0, upper=100, coords=[storages, time], name="soc")
    charge = m.add_variables(lower=0, upper=50, coords=[storages, time], name="charge")
    discharge = m.add_variables(
        lower=0, upper=50, coords=[storages, time], name="discharge"
    )
    prev = soc.shift(time=1)
    m.add_constraints(
        soc - 0.99 * prev - 0.95 * charge + discharge / 0.95 == 0, name="soc_balance"
    )
    m.add_objective((charge + discharge).sum())
    return m


# ---------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------


def _dim_index(model: Model, case: WindowCase) -> pd.Index:
    """The case's windowing dimension index, from any carrier."""
    for _, var in model.variables.items():
        if case.dim in var.indexes:
            return var.indexes[case.dim]
    raise ValueError(f"No variable carries the dimension '{case.dim}'.")


def _initial_carried(model: Model, case: WindowCase) -> dict[str, xr.DataArray]:
    """
    Full-horizon carry arrays for the case's fix variables, initialized to a
    feasible guess (the variable's upper bound, 0 where unbounded): folded
    constant rows from not-yet-solved windows must come out satisfied.
    """
    carried = {}
    for name in case.fix_vars:
        var = model.variables[name]
        labels = var.labels
        upper = np.asarray(var.upper.values, dtype=float)
        fill = np.where(np.isfinite(upper), upper, 0.0)
        carried[name] = xr.DataArray(
            fill, coords=labels.coords, dims=labels.dims, name=name
        )
    return carried


def _solve_windows(
    model: Model,
    case: WindowCase,
    windows: list[pd.Index],
    carried: dict[str, xr.DataArray],
) -> list[dict[str, Any]]:
    """Solve the windows sequentially via restrict + advance."""
    fix = {name: carried[name] for name in case.fix_vars}
    w = model.restrict(**{case.dim: windows[0]}, fix=fix)
    results = []
    for i, window in enumerate(windows):
        if i > 0:
            w.advance(**{case.dim: window}, fix=fix)
        w.solve(solver_name="highs", output_flag=False)
        results.append(
            {
                "objective": w.objective.value,
                "solution": w.solution,
                "dual": w.dual,
            }
        )
        for name in case.fix_vars:
            carried[name].loc[{case.dim: window}] = w.solution[name].values
    return results


def _assert_chunked_equals_full(
    results: list[dict[str, Any]], full: Model, case: WindowCase
) -> None:
    """
    Objectives sum to the monolithic one; primal/dual values match per
    position for every array carrying the windowing dim.

    Arrays WITHOUT the dim (shared variables re-optimized per window,
    aggregate rows enforced per window) are per-window quantities: their
    equality with the monolithic solve is not implied by decomposability and
    is not asserted.
    """
    total = sum(r["objective"] for r in results)
    assert total == pytest.approx(full.objective.value)

    for name, da in full.solution.data_vars.items():
        if case.dim not in da.dims:
            continue
        parts = [r["solution"][name] for r in results]
        merged = xr.concat(parts, dim=da.indexes[case.dim])
        assert_allclose(merged.values, da.values, rtol=1e-7)
    for name, da in full.dual.data_vars.items():
        if case.dim not in da.dims:
            continue
        parts = [r["dual"][name] for r in results]
        merged = xr.concat(parts, dim=da.indexes[case.dim])
        assert_allclose(merged.values, da.values, rtol=1e-6)


def _assert_replay_equals_fresh(
    case: WindowCase,
    windows: list[pd.Index],
    carried_init: dict[str, xr.DataArray],
    results: list[dict[str, Any]],
) -> None:
    """
    Replay the chain's fix sequence with fresh restricts: every window must
    re-solve to the chain's own result.
    """
    carried = {name: da.copy() for name, da in carried_init.items()}
    for i, window in enumerate(windows):
        fix = {name: carried[name] for name in case.fix_vars}
        fresh = case.build().restrict(**{case.dim: window}, fix=fix)
        fresh.solve(solver_name="highs", output_flag=False)

        assert fresh.objective.value == pytest.approx(results[i]["objective"])
        for name, da in fresh.solution.data_vars.items():
            assert_allclose(da.values, results[i]["solution"][name].values, rtol=1e-7)
        for name, da in fresh.dual.data_vars.items():
            assert_allclose(da.values, results[i]["dual"][name].values, rtol=1e-6)
        for name in case.fix_vars:
            carried[name].loc[{case.dim: window}] = fresh.solution[name].values


@needs_highs
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_windowed_equivalence(case: WindowCase, semantics: str) -> None:
    """Every registered model is solved equivalently through the windows."""
    full = case.build()
    full.solve(solver_name="highs", output_flag=False)

    index = _dim_index(full, case)
    n_tiles = int(np.ceil(len(index) / case.tile))
    windows = [index[i * case.tile : (i + 1) * case.tile] for i in range(n_tiles)]

    model = case.build()
    carried = _initial_carried(model, case)
    results = _solve_windows(model, case, windows, carried)

    if case.coupling == "decomposable":
        _assert_chunked_equals_full(results, full, case)
    else:
        _assert_replay_equals_fresh(case, windows, carried, results)
