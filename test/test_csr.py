"""
Tests for sparse groupby-sum (linopy.csr): type stability, transparent
materialization, and direct CSR realization under freeze. v1-only feature.
"""

from __future__ import annotations

import re
import tracemalloc
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
import scipy.sparse
import xarray as xr
from xarray.core.types import JoinOptions

import linopy
from linopy import LinearExpression, Model, Variable
from linopy.constants import TERM_DIM
from linopy.constraints import Constraint, ConstraintBase, CSRConstraint
from linopy.csr import CSRLinearExpression, Grid
from linopy.semantics import is_v1
from linopy.testing import assert_conequal, assert_linequal, assert_quadequal


def require_v1() -> None:
    if not is_v1():
        pytest.skip("sparse groupby-sum is gated behind v1 semantics")


@dataclass
class Case:
    """Model with gen_p and flow on a ring; load ordered like the sorted groups (v1)."""

    m: Model
    gen_p: Variable
    flow: Variable
    flow_t: Variable
    eff: xr.DataArray
    gbus: pd.Series
    bus0: pd.Series
    bus1: pd.Series
    load: xr.DataArray

    def balance_lhs(self, sparse: bool | None) -> LinearExpression:
        return (
            (self.eff * self.gen_p).groupby(self.gbus).sum(sparse=sparse)
            + (1.0 * self.flow).groupby(self.bus0).sum(sparse=sparse)
            - (1.0 * self.flow).groupby(self.bus1).sum(sparse=sparse)
        )


def base_model(
    gens_per_bus: tuple[int, ...] = (7, 1, 3, 1, 2), n_snap: int = 3, seed: int = 0
) -> Case:
    rng = np.random.default_rng(seed)
    n_bus = len(gens_per_bus)
    buses = pd.Index([f"bus{i}" for i in range(n_bus)], name="bus")
    gen_bus = np.repeat(np.arange(n_bus), gens_per_bus)
    gens = pd.Index([f"gen{i}" for i in range(len(gen_bus))], name="gen")
    lines = pd.Index([f"line{i}" for i in range(n_bus)], name="line")
    snaps = pd.Index(range(n_snap), name="snapshot")

    m = linopy.Model()
    gen_p = m.add_variables(coords=[gens, snaps], name="gen_p")
    flow = m.add_variables(coords=[lines, snaps], name="flow")
    flow_t = m.add_variables(coords=[snaps, lines], name="flow_t")

    gbus = pd.Series(buses[gen_bus], index=gens, name="bus")
    bus0 = pd.Series(buses[np.arange(n_bus)], index=lines, name="bus")
    bus1 = pd.Series(buses[(np.arange(n_bus) + 1) % n_bus], index=lines, name="bus")
    load = xr.DataArray(
        rng.uniform(1, 10, (n_bus, n_snap)), coords=[buses, snaps], name="load"
    ).sortby("bus")
    eff = xr.DataArray(rng.uniform(0.5, 1.5, len(gens)), coords=[gens])
    return Case(m, gen_p, flow, flow_t, eff, gbus, bus0, bus1, load)


def canon(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["labels", "vars"])
        .agg(pl.col("coeffs").sum(), pl.col("sign").first(), pl.col("rhs").first())
        .sort(["labels", "vars"])
    )


def assert_frozen_equal(con1: Constraint, con2: CSRConstraint) -> None:
    d1, d2 = canon(con1.to_polars()), canon(con2.to_polars())
    assert d1["labels"].equals(d2["labels"])
    assert d1["vars"].equals(d2["vars"])
    assert np.allclose(d1["coeffs"], d2["coeffs"])
    assert (d1["sign"] == d2["sign"]).all()
    assert np.allclose(d1["rhs"], d2["rhs"])
    labels = con1.labels.values.ravel()
    assert np.array_equal(np.sort(labels[labels != -1]), np.sort(con2.active_labels()))


def reindexed_balance(c: Case, sparse: bool) -> LinearExpression:
    """Generation on all buses plus flow on two lines, both reindexed onto the load grid."""
    lines = ["line0", "line1"]
    gen = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=sparse)
    flow = (1.0 * c.flow.loc[lines]).groupby(c.bus0.loc[lines]).sum(sparse=sparse)
    parts = [gen.reindex(bus=c.load.bus), flow.reindex(bus=c.load.bus)]
    return linopy.merge(parts, join="outer", cls=LinearExpression)


def test_csr_requires_v1() -> None:
    c = base_model()
    if is_v1():
        res = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True)
        assert type(res) is LinearExpression
        return
    with pytest.raises(ValueError, match="requires v1 semantics"):
        (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True)
    linopy.options["sparse_groupby"] = True
    try:
        res = (c.eff * c.gen_p).groupby(c.gbus).sum()
    finally:
        linopy.options["sparse_groupby"] = False
    assert res._csr is None


def test_csr_is_plain_linear_expression_and_materializes_equivalently() -> None:
    require_v1()
    c = base_model()
    sparse = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True)
    eager = (c.eff * c.gen_p).groupby(c.gbus).sum()
    assert type(sparse) is LinearExpression
    assert_linequal(sparse, eager)


def test_csr_composition_materializes_equivalently() -> None:
    require_v1()
    c = base_model()
    sparse = c.balance_lhs(sparse=True)
    assert type(sparse) is LinearExpression
    assert_linequal(sparse, c.balance_lhs(sparse=False))


def test_scalar_ops_stay_csr() -> None:
    require_v1()
    c = base_model()
    sparse = -2.0 * (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True)
    assert sparse._csr is not None
    assert_linequal(sparse, -2.0 * (c.eff * c.gen_p).groupby(c.gbus).sum())


@pytest.mark.parametrize("sparse", [True, False], ids=["sparse", "dense"])
def test_zero_coefficient_rows_stay_active(sparse: bool) -> None:
    require_v1()
    c = base_model()
    lhs = (0.0 * c.gen_p).groupby(c.gbus).sum(sparse=sparse)
    lhs = lhs + (0.0 * c.flow).groupby(c.bus0).sum(sparse=sparse)
    con = c.m.add_constraints(lhs == c.load, name="bal", freeze=True)
    assert len(con.active_labels()) == c.load.size


def test_merge_keeps_absent_cell_absent() -> None:
    require_v1()
    c = base_model()
    dense = (c.eff * c.gen_p).groupby(c.gbus).sum()
    flow = (1.0 * c.flow).groupby(c.bus0).sum()
    mask = xr.DataArray(np.arange(len(c.load.bus)) % 2 == 0, coords=[c.load.bus])
    flow = flow.where(mask)

    sparse = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True)
    tot = linopy.merge([sparse, flow], join="outer", cls=LinearExpression)
    assert tot._csr is not None
    assert_linequal(
        tot, linopy.merge([dense, flow], join="outer", cls=LinearExpression)
    )

    con = c.m.add_constraints(tot >= c.load, name="bal", freeze=True)
    assert isinstance(con, CSRConstraint)
    assert con.ncons == int(mask.sum()) * c.load.sizes["snapshot"]


@pytest.mark.parametrize(
    "grouper, kwargs",
    [
        (["gen", "snapshot"], {}),
        ("gen", {"use_fallback": True}),
    ],
)
def test_explicit_sparse_raises_on_unsupported_grouper(
    grouper: str | list[str], kwargs: dict
) -> None:
    require_v1()
    c = base_model()
    with pytest.raises(ValueError, match="sparse=True supports only"):
        (1.0 * c.gen_p).groupby(grouper).sum(sparse=True, **kwargs)


def keyed_model(
    member_first: bool = True,
) -> tuple[Model, LinearExpression, xr.DataArray]:
    """
    ``(1 + x)`` over ``(s, snapshot)`` with ``period``/``season`` keys on ``s``;
    two of the six (period, season) combinations never occur.
    """
    n = 6
    s = pd.RangeIndex(n, name="s")
    snaps = pd.Index(range(2), name="snapshot")
    m = Model()
    coords = [s, snaps] if member_first else [snaps, s]
    x = m.add_variables(coords=coords, name="x")
    period = xr.DataArray(np.arange(n) // 2, dims="s", coords={"s": s})
    season = xr.DataArray(list("wswwww"), dims="s", coords={"s": s})
    expr = (1.0 * x + 1.0).assign_coords(period=period, season=season)
    keys = pd.MultiIndex.from_arrays([period.values, season.values])
    rhs = xr.DataArray(
        np.arange(1.0, 2 * len(keys.unique()) + 1).reshape(-1, 2),
        coords=[pd.RangeIndex(len(keys.unique()), name="group"), snaps],
    )
    return m, expr, rhs


@pytest.mark.parametrize("member_first", [True, False])
@pytest.mark.parametrize("observed", [False, True])
def test_namelist_sparse_matches_dense(observed: bool, member_first: bool) -> None:
    require_v1()
    _, expr, _ = keyed_model(member_first)
    keys = ["period", "season"]
    sparse = expr.groupby(keys).sum(sparse=True, observed=observed)
    dense = expr.groupby(keys).sum(observed=observed)
    csr = sparse._csr
    assert csr is not None
    if observed:
        assert set(csr.grid.aux) == {"period", "season"}
    else:
        assert np.isnan(csr.const).sum() == 2 * 2
    assert csr.grid.dims == dense.coord_dims
    assert_linequal(sparse, dense)


@pytest.mark.parametrize("as_namelist", [False, True])
def test_single_key_sparse_ignores_observed(as_namelist: bool) -> None:
    require_v1()
    c = base_model()
    expr = (c.eff * c.gen_p).assign_coords(bus=("gen", c.gbus.to_numpy()))
    grouper = ["bus"] if as_namelist else c.gbus
    sparse = expr.groupby(grouper).sum(sparse=True, observed=True)
    assert sparse._csr is not None
    assert_linequal(sparse, expr.groupby(grouper).sum())


def test_sparse_keeps_aux_coords_on_surviving_dims() -> None:
    require_v1()
    _, expr, _ = keyed_model()
    expr = expr.assign_coords(tag=("snapshot", list("ab")))
    for kwargs in ({}, {"observed": True}):
        sparse = expr.groupby(["period", "season"]).sum(sparse=True, **kwargs)
        dense = expr.groupby(["period", "season"]).sum(**kwargs)
        assert set(sparse.coords) == set(dense.coords)
        assert_linequal(sparse, dense)


def test_dataframe_grouper_sparse_stays_compact() -> None:
    require_v1()
    _, expr, _ = keyed_model()
    df = expr.data[["period", "season"]].to_dataframe()[["period", "season"]]
    sparse = expr.groupby(df).sum(sparse=True)
    assert sparse._csr is not None
    assert sparse._csr.shape == (4, 2)
    assert_linequal(sparse, expr.groupby(df).sum())


def test_namelist_sparse_observed_freezes_compact() -> None:
    require_v1()
    m1, e1, rhs = keyed_model()
    m2, e2, _ = keyed_model()
    dense = m1.add_constraints(
        e1.groupby(["period", "season"]).sum(observed=True) == rhs,
        name="c",
        freeze=True,
    )
    sparse = m2.add_constraints(
        e2.groupby(["period", "season"]).sum(sparse=True, observed=True) == rhs,
        name="c",
        freeze=True,
    )
    assert isinstance(sparse, CSRConstraint)
    assert sparse.ncons == rhs.size
    assert_conequal(dense, sparse, strict=False)


def test_namelist_sparse_grid_absent_cells_inactive() -> None:
    require_v1()
    m1, e1, _ = keyed_model()
    m2, e2, _ = keyed_model()
    keys = ["period", "season"]
    dense = m1.add_constraints(e1.groupby(keys).sum() == 0, name="c", freeze=True)
    sparse = m2.add_constraints(
        e2.groupby(keys).sum(sparse=True) == 0, name="c", freeze=True
    )
    assert isinstance(sparse, CSRConstraint)
    assert sparse.ncons == 4 * 2
    assert_conequal(dense, sparse, strict=False)
    assert np.array_equal(
        np.sort(dense.active_labels()), np.sort(sparse.active_labels())
    )


def test_namelist_sparse_observed_keeps_aux_coords_through_merge() -> None:
    require_v1()
    _, expr, _ = keyed_model()
    keys = ["period", "season"]
    sparse = expr.groupby(keys).sum(sparse=True, observed=True)
    dense = expr.groupby(keys).sum(observed=True)
    tot = sparse + dense
    assert tot._csr is not None
    assert set(tot._csr.grid.aux) == {"period", "season"}
    assert_linequal(tot, 2.0 * dense)

    other = dense.assign_coords(region=("group", list("abcd")))
    tot = sparse + other
    assert tot._csr is not None
    assert set(tot._csr.grid.aux) == {"period", "season", "region"}
    xr.testing.assert_equal(
        tot.data.coords.to_dataset(), (dense + other).data.coords.to_dataset()
    )

    conflicting = dense.assign_coords(season=("group", list("xyzw")))
    with pytest.raises(ValueError, match="conflicting values"):
        sparse + conflicting


def test_namelist_sparse_grid_warns_and_observed_silences() -> None:
    require_v1()
    n = 200
    s = pd.RangeIndex(n, name="s")
    m = Model()
    x = m.add_variables(coords=[s], name="x")
    expr = (1.0 * x).assign_coords(
        period=xr.DataArray(np.arange(n), dims="s", coords={"s": s}),
        season=xr.DataArray(np.arange(n), dims="s", coords={"s": s}),
    )
    with pytest.warns(UserWarning, match="dense .* grid"):
        expr.groupby(["period", "season"]).sum(sparse=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = expr.groupby(["period", "season"]).sum(sparse=True, observed=True)
    assert res._csr is not None
    assert res._csr.shape == (n,)


def test_nan_multikey_grouper_raises_eagerly() -> None:
    require_v1()
    _, expr, _ = keyed_model()
    period = expr.data["period"]
    expr = expr.assign_coords(period=period.where(period > 0))
    with pytest.raises(ValueError, match="NaN values"):
        expr.groupby(["period", "season"]).sum(sparse=True)


def test_freeze_realizes_csr_without_dense_rectangle() -> None:
    require_v1()
    c1, c2 = base_model(), base_model()
    con1 = c1.m.add_constraints(c1.balance_lhs(sparse=False) == c1.load, name="bal")
    con2 = c2.m.add_constraints(
        c2.balance_lhs(sparse=True) == c2.load, name="bal", freeze=True
    )

    assert isinstance(con2, CSRConstraint)
    assert_frozen_equal(con1, con2)


def test_freeze_false_falls_back_to_identical_dense_constraint() -> None:
    """The fallback is canonical-form, so compare mathematically (strict=False)."""
    require_v1()
    c1, c2 = base_model(), base_model()
    con1 = c1.m.add_constraints(c1.balance_lhs(sparse=False) == c1.load, name="bal")
    con2 = c2.m.add_constraints(c2.balance_lhs(sparse=True) == c2.load, name="bal")
    assert isinstance(con2, Constraint)
    assert_conequal(con1, con2, strict=False)
    assert np.array_equal(con1.labels.values, con2.labels.values)


def test_to_constraint_on_csr_lhs_is_unassigned_csr_constraint() -> None:
    require_v1()
    c1, c2 = base_model(), base_model()
    dense = c1.balance_lhs(sparse=False) == c1.load
    con = c2.balance_lhs(sparse=True) == c2.load
    assert isinstance(con, CSRConstraint)
    assert not con.is_assigned
    assert con.type == "Constraint (unassigned)"
    assert "None" not in repr(con)
    assert_conequal(dense, con, strict=False)
    with pytest.raises(ValueError, match="not been assigned"):
        con.active_labels()
    with pytest.raises(ValueError, match="not been assigned"):
        con.to_polars()
    with pytest.raises(ValueError, match="not been assigned"):
        c2.m.constraints.add(con)

    con1 = c1.m.add_constraints(dense, name="bal")
    con2 = c2.m.add_constraints(con, name="bal", freeze=True)
    assert isinstance(con2, CSRConstraint)
    assert con2.is_assigned
    assert np.array_equal(
        np.sort(con1.labels.values.ravel()), np.sort(con2.active_labels())
    )
    assert_conequal(con1, con2, strict=False)


@pytest.mark.parametrize("freeze", [False, True])
def test_group_without_terms_matches_dense_labels(freeze: bool) -> None:
    require_v1()

    def build(sparse: bool) -> ConstraintBase:
        c = base_model()
        gens = c.gen_p.indexes["gen"]
        gen_p = c.gen_p.where(xr.DataArray(gens != "gen7", coords=[gens]))
        lhs = (c.eff * gen_p).groupby(c.gbus).sum(sparse=sparse)
        return c.m.add_constraints(lhs == c.load, name="bal", freeze=freeze)

    dense, sparse = build(False), build(True)
    assert isinstance(sparse, CSRConstraint if freeze else Constraint)
    assert_conequal(dense, sparse, strict=False)
    np.testing.assert_array_equal(dense.labels.values, sparse.labels.values)


@pytest.mark.parametrize("sparse", [True, False], ids=["sparse", "dense"])
def test_frozen_invalid_infinite_rhs_raises(sparse: bool) -> None:
    require_v1()
    c = base_model()
    with pytest.raises(ValueError, match="incorrect infinite values"):
        c.m.add_constraints(c.balance_lhs(sparse) <= -np.inf, name="bal", freeze=True)


@pytest.mark.parametrize("sparse", [True, False], ids=["sparse", "dense"])
def test_frozen_constraint_applies_row_scaling(sparse: bool) -> None:
    require_v1()
    c = base_model()
    snaps = c.load.indexes["snapshot"]
    scaling = xr.DataArray(np.arange(1.0, len(snaps) + 1), coords=[snaps])
    con = c.m.add_constraints(
        c.balance_lhs(sparse) == c.load, name="bal", freeze=True, scaling=scaling
    )
    assert isinstance(con, CSRConstraint)
    expected = scaling.broadcast_like(con.scaling).transpose(*con.scaling.dims)
    xr.testing.assert_equal(con.scaling, expected)


def test_option_gates_csr_and_freeze_model_default() -> None:
    require_v1()
    c = base_model()
    c.m.freeze_constraints = True
    linopy.options["sparse_groupby"] = True
    try:
        con = c.m.add_constraints(c.balance_lhs(sparse=None) == c.load, name="bal")
    finally:
        linopy.options["sparse_groupby"] = False
    assert isinstance(con, CSRConstraint)


def test_materialized_csr_still_freezes_via_dense_path() -> None:
    require_v1()
    c = base_model()
    lhs = c.balance_lhs(sparse=True)
    _ = lhs.nterm
    con = c.m.add_constraints(lhs == c.load, name="bal", freeze=True)
    assert isinstance(con, CSRConstraint)


@pytest.mark.parametrize("sparse", [True, False], ids=["sparse", "dense"])
def test_nan_rhs_raises(sparse: bool) -> None:
    require_v1()
    c = base_model()
    load = c.load.copy()
    load[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        c.m.add_constraints(c.balance_lhs(sparse) == load, name="bal", freeze=True)


@pytest.mark.parametrize("sparse", [True, False], ids=["sparse", "dense"])
def test_reordered_rhs_raises(sparse: bool) -> None:
    require_v1()
    c = base_model()
    load = c.load.isel(bus=slice(None, None, -1))
    with pytest.raises(ValueError, match="[Cc]oordinate"):
        c.m.add_constraints(c.balance_lhs(sparse) == load, name="bal", freeze=True)


def test_nan_grouper_raises_eagerly() -> None:
    require_v1()
    c = base_model()
    gbus = c.gbus.copy()
    gbus.iloc[0] = np.nan
    with pytest.raises(ValueError, match="NaN values"):
        (1.0 * c.gen_p).groupby(gbus).sum(sparse=True)


def test_lp_files_identical(tmp_path: Path) -> None:
    require_v1()
    sizes = (7, 1, 3, 1, 2, 1, 1, 4, 1, 2, 1, 1)
    c1, c2 = base_model(gens_per_bus=sizes), base_model(gens_per_bus=sizes)
    c1.m.add_constraints(c1.balance_lhs(sparse=False) == c1.load, name="bal")
    c1.m.add_objective((1.0 * c1.gen_p).sum())
    c2.m.add_constraints(
        c2.balance_lhs(sparse=True) == c2.load, name="bal", freeze=True
    )
    c2.m.add_objective((1.0 * c2.gen_p).sum())

    term_line = re.compile(r"^[+-][0-9.e+-]+ x[0-9]+$")

    def canon_lp(text: str) -> list[str]:
        out: list[str] = []
        buf: list[str] = []
        for line in text.splitlines():
            if term_line.match(line):
                buf.append(line)
            else:
                out += sorted(buf) + [line]
                buf = []
        return out + sorted(buf)

    f1, f2 = tmp_path / "eager.lp", tmp_path / "sparse.lp"
    c1.m.to_file(f1)
    c2.m.to_file(f2)
    assert canon_lp(f1.read_text()) == canon_lp(f2.read_text())


@pytest.mark.parametrize(
    "indexers",
    [
        {"bus": ["bus3", "bus0", "bus1", "bus2", "bus4"]},
        {"bus": ["bus0", "bus1", "bus2", "bus3", "bus4", "bus9"]},
        {"bus": ["bus3", "bus0"]},
        {"bus": ["bus4", "bus0", "bus7"], "snapshot": [2, 0, 5]},
    ],
    ids=["reorder", "add", "drop", "multi_dim"],
)
def test_reindex_stays_csr_and_matches_dense(indexers: dict) -> None:
    require_v1()
    c = base_model()
    sparse = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True).reindex(indexers)
    assert sparse._csr is not None
    dense = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=False).reindex(indexers)
    assert_linequal(sparse, dense)


def test_reindex_falls_back_to_dense_for_unsupported_kwargs() -> None:
    require_v1()
    c = base_model()
    sparse = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True)
    res = sparse.reindex(bus=["bus3", "bus0"], copy=False)
    assert res._csr is None
    dense = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=False)
    assert_linequal(res, dense.reindex(bus=["bus3", "bus0"]))


def test_reindex_merge_chain_freezes_csr() -> None:
    require_v1()
    c1, c2 = base_model(), base_model()
    con1 = c1.m.add_constraints(reindexed_balance(c1, False) == c1.load, name="bal")
    tot = reindexed_balance(c2, True)
    assert tot._csr is not None
    con2 = c2.m.add_constraints(tot == c2.load, name="bal", freeze=True)
    assert isinstance(con2, CSRConstraint)
    assert_frozen_equal(con1, con2)


def test_reindex_merge_chain_peak_memory() -> None:
    require_v1()
    sizes = (200,) + (1,) * 299
    n_snap = 50
    dense_rectangle_bytes = len(sizes) * n_snap * max(sizes) * 16
    c = base_model(gens_per_bus=sizes, n_snap=n_snap)
    tracemalloc.start()
    try:
        c.m.add_constraints(
            reindexed_balance(c, True) == c.load, name="bal", freeze=True
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < dense_rectangle_bytes / 4


@pytest.mark.parametrize("value", [0.0, 3.5])
def test_fillna_stays_csr_and_matches_dense(value: float) -> None:
    require_v1()
    c = base_model()
    wide = {"bus": [f"bus{i}" for i in range(7)]}
    sparse = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True).reindex(wide)
    filled = sparse.fillna(value)
    assert filled._csr is not None
    dense = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=False).reindex(wide)
    assert_linequal(filled, dense.fillna(value))


def test_fillna_with_array_falls_back_to_dense() -> None:
    require_v1()
    c = base_model()
    fill = xr.zeros_like(c.load)
    sparse = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True)
    res = sparse.fillna(fill)
    assert res._csr is None
    dense = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=False)
    assert_linequal(res, dense.fillna(fill))


def test_rename_stays_csr_and_matches_dense() -> None:
    require_v1()
    c = base_model()
    sparse = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True).rename(bus="node")
    assert sparse._csr is not None
    dense = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=False).rename(bus="node")
    assert sparse.coord_dims == ("node", "snapshot")
    assert_linequal(sparse, dense)


@pytest.mark.parametrize(
    "op",
    [lambda e: e.reindex(group=[3, 0, 9]), lambda e: e.rename(group="g")],
    ids=["reindex", "rename"],
)
def test_namelist_sparse_observed_keeps_aux_coords_through_op(
    op: Callable[[LinearExpression], LinearExpression],
) -> None:
    require_v1()
    _, expr, _ = keyed_model()
    keys = ["period", "season"]
    sparse = op(expr.groupby(keys).sum(sparse=True, observed=True))
    dense = op(expr.groupby(keys).sum(sparse=False, observed=True))
    assert sparse._csr is not None
    for name in keys:
        xr.testing.assert_equal(sparse.coords[name], dense.coords[name])
    assert_linequal(sparse, dense)


def cross_grid_parts(
    c: Case, sparse: bool, lines: tuple[str, ...] = ("line1", "line2")
) -> list[LinearExpression]:
    """Generation on all buses, flow on a line subset only, snapshot-major on the flow side."""
    gen = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=sparse)
    flow_t = 1.0 * c.flow_t.loc[:, list(lines)]
    flow = flow_t.groupby(c.bus0.loc[list(lines)]).sum(sparse=sparse)
    return [gen, flow]


def assert_terms_equal(a: LinearExpression, b: LinearExpression) -> None:
    """``assert_linequal`` up to the width of the (non-contractual) term axis."""
    width = max(a.nterm, b.nterm)
    padded = []
    for e in (a, b):
        pad = {"_term": (0, width - e.nterm)}
        fill = {"vars": -1, "coeffs": np.nan}
        padded.append(LinearExpression(e.data.pad(pad, constant_values=fill), e.model))
    assert_linequal(*padded)


@pytest.mark.parametrize("join", ["outer", "inner", "left", "right"])
@pytest.mark.parametrize("order", ["gen-flow", "flow-gen"])
def test_cross_grid_merge_stays_csr_and_matches_dense(
    join: JoinOptions, order: str
) -> None:
    require_v1()
    c = base_model()
    sparse, dense = cross_grid_parts(c, True), cross_grid_parts(c, False)
    if order == "flow-gen":
        sparse, dense = sparse[::-1], dense[::-1]
    res = linopy.merge(sparse, join=join, cls=LinearExpression)
    assert res._csr is not None
    assert res.coord_dims == dense[0].coord_dims
    assert_terms_equal(res, linopy.merge(dense, join=join, cls=LinearExpression))


def test_transposed_grid_exact_merge_stays_csr_and_matches_dense() -> None:
    """Same labels in a transposed dim order stay sparse under the default join."""
    require_v1()
    c = base_model()
    a = (1.0 * c.flow).groupby(c.bus0).sum(sparse=True)
    b = (1.0 * c.flow_t).groupby(c.bus0).sum(sparse=True)
    assert a.coord_dims == b.coord_dims[::-1] != b.coord_dims
    res = linopy.merge([a, b], cls=LinearExpression)
    assert res._csr is not None
    dense = [
        (1.0 * c.flow).groupby(c.bus0).sum(),
        (1.0 * c.flow_t).groupby(c.bus0).sum(),
    ]
    assert_terms_equal(res, linopy.merge(dense, cls=LinearExpression))


@pytest.mark.parametrize("join", ["outer", "inner", "left", "right"])
def test_three_operand_cross_grid_merge_matches_dense(join: JoinOptions) -> None:
    require_v1()
    c = base_model()
    third_lines = ("line3", "line4")
    sparse = cross_grid_parts(c, True) + cross_grid_parts(c, True, third_lines)[1:]
    dense = cross_grid_parts(c, False) + cross_grid_parts(c, False, third_lines)[1:]
    res = linopy.merge(sparse, join=join, cls=LinearExpression)
    assert res._csr is not None
    assert_terms_equal(res, linopy.merge(dense, join=join, cls=LinearExpression))


def test_cross_grid_merge_absent_fill_matches_dense() -> None:
    require_v1()
    c = base_model()
    sparse, dense = cross_grid_parts(c, True), cross_grid_parts(c, False)
    res = linopy.merge(
        sparse, join="outer", fill_value=linopy.ABSENT, cls=LinearExpression
    )
    expected = linopy.merge(
        dense, join="outer", fill_value=linopy.ABSENT, cls=LinearExpression
    )
    assert res._csr is not None
    filled = res.fillna(0)
    assert filled._csr is not None
    assert_terms_equal(filled, expected.fillna(0))
    assert_terms_equal(res, expected)
    assert res.const.isnull().sum() == 3 * c.load.sizes["snapshot"]


def test_cross_grid_merge_keeps_absent_cell_absent() -> None:
    require_v1()
    c = base_model()
    sparse, dense = cross_grid_parts(c, True), cross_grid_parts(c, False)
    mask = xr.DataArray([True, False], coords=[dense[1].indexes["bus"]])
    dense[1] = dense[1].where(mask)
    res = linopy.merge([sparse[0], dense[1]], join="outer", cls=LinearExpression)
    assert res._csr is not None
    assert_terms_equal(res, linopy.merge(dense, join="outer", cls=LinearExpression))


def test_cross_grid_merge_mixed_dense_operand_stays_csr() -> None:
    require_v1()
    c = base_model()
    sparse, dense = cross_grid_parts(c, True), cross_grid_parts(c, False)
    res = sparse[0].add(dense[1], join="outer")
    assert isinstance(res, LinearExpression)
    assert res._csr is not None
    expected = dense[0].add(dense[1], join="outer")
    assert isinstance(expected, LinearExpression)
    assert_terms_equal(res, expected)


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({}, ValueError),
        ({"join": "exact"}, xr.AlignmentError),
        ({"join": "override"}, xr.AlignmentError),
    ],
    ids=["auto", "exact", "override"],
)
def test_cross_grid_merge_raises_like_dense(kwargs: dict, error: type) -> None:
    require_v1()
    c = base_model()
    with pytest.raises(error):
        linopy.merge(cross_grid_parts(c, False), **kwargs)
    with pytest.raises(error):
        linopy.merge(cross_grid_parts(c, True), **kwargs)


@pytest.mark.parametrize("join", ["outer", "inner", "left", "right"])
def test_cross_grid_merge_with_aux_coord_operand_stays_csr(join: JoinOptions) -> None:
    """The aux coord follows its rows onto the joined grid, like dense."""
    require_v1()
    c = base_model()
    sparse, dense = cross_grid_parts(c, True), cross_grid_parts(c, False)
    tag = xr.DataArray(["x", "y"], coords=[dense[1].indexes["bus"]])
    tagged = LinearExpression(dense[1].data.assign_coords(tag=tag), c.m)
    with no_densify():
        res = linopy.merge([sparse[0], tagged], join=join, cls=LinearExpression)
    assert res.is_sparse
    labels = res.indexes["bus"]
    want = tag.to_series().reindex(labels).to_numpy()
    assert pd.Series(res.coords["tag"].values).equals(pd.Series(want))
    want_expr = linopy.merge([dense[0], tagged], join=join, cls=LinearExpression)
    assert_sparse_matches(res, want_expr)


def test_cross_grid_merge_aux_coord_conflict_raises_like_dense() -> None:
    require_v1()
    sparse = SPARSE_BUILDS["aux"](base_model())
    assert sparse._csr is not None
    dense = sparse._csr.to_dense()
    parts = [sparse, sparse.isel(group=[2, 1])]
    with pytest.raises(ValueError) as want:
        linopy.merge([dense, dense.isel(group=[2, 1])], join="outer")
    with pytest.raises(ValueError) as got:
        linopy.merge(parts, join="outer")
    assert str(got.value) == str(want.value)


def test_cross_grid_merge_with_duplicate_labels_raises_like_dense() -> None:
    require_v1()
    c = base_model()
    dup = pd.Index(["bus1", "bus1", "bus2"], name="bus")
    shed = 1.0 * c.m.add_variables(
        coords=[dup, c.load.indexes["snapshot"]], name="shed"
    )
    gen = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True)
    dense = (c.eff * c.gen_p).groupby(c.gbus).sum()
    with pytest.raises(ValueError, match="cannot reindex or align"):
        linopy.merge([dense, shed], join="left")
    with pytest.raises(ValueError, match="cannot reindex or align"):
        linopy.merge([gen, shed], join="left")


def test_override_merge_same_shape_stays_csr() -> None:
    require_v1()
    c = base_model()
    gen = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True)
    flow = (1.0 * c.flow).groupby(c.bus1.str.upper()).sum(sparse=True)
    res = linopy.merge([gen, flow], join="override", cls=LinearExpression)
    assert res._csr is not None
    dense = [
        (c.eff * c.gen_p).groupby(c.gbus).sum(),
        (1.0 * c.flow).groupby(c.bus1.str.upper()).sum(),
    ]
    assert_terms_equal(res, linopy.merge(dense, join="override", cls=LinearExpression))


def test_cross_grid_balance_freezes_csr() -> None:
    require_v1()
    c1, c2 = base_model(), base_model()
    lhs1 = linopy.merge(cross_grid_parts(c1, False), join="outer")
    con1 = c1.m.add_constraints(lhs1 == c1.load, name="bal")
    lhs2 = linopy.merge(cross_grid_parts(c2, True), join="outer", cls=LinearExpression)
    assert lhs2._csr is not None
    con2 = c2.m.add_constraints(lhs2 == c2.load, name="bal", freeze=True)
    assert isinstance(con2, CSRConstraint)
    assert_frozen_equal(con1, con2)


def test_cross_grid_merge_peak_memory() -> None:
    require_v1()
    sizes = (200,) + (1,) * 299
    n_snap = 50
    dense_rectangle_bytes = len(sizes) * n_snap * max(sizes) * 16
    c = base_model(gens_per_bus=sizes, n_snap=n_snap)
    tracemalloc.start()
    try:
        lhs = linopy.merge(cross_grid_parts(c, True), join="outer")
        c.m.add_constraints(lhs == c.load, name="bal", freeze=True)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < dense_rectangle_bytes / 4


def cell_matrix(
    expr: LinearExpression, dims: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell coefficient row over raw variable labels, plus the flat constant."""
    ds = expr.data
    nterm = ds.sizes[TERM_DIM]
    vars_ = ds.vars.transpose(*dims, TERM_DIM).to_numpy().reshape(-1, nterm)
    coeffs = ds.coeffs.transpose(*dims, TERM_DIM).to_numpy().reshape(-1, nterm)
    active = (vars_ != -1) & ~np.isnan(coeffs)
    rows = np.broadcast_to(np.arange(len(vars_))[:, None], vars_.shape)
    matrix = np.zeros((len(vars_), expr.model._xCounter))
    np.add.at(matrix, (rows[active], vars_[active]), coeffs[active])
    const = np.nan_to_num(ds.const.transpose(*dims).to_numpy().reshape(-1))
    return matrix, const


def assert_cells_equal(
    res: LinearExpression, reference: LinearExpression, dims: tuple[str, ...]
) -> None:
    """Compare two expressions cell by cell, independently of the term layout."""
    got, got_const = cell_matrix(res, dims)
    want, want_const = cell_matrix(reference, dims)
    assert np.allclose(got, want)
    assert np.allclose(got_const, want_const)


def assert_contracted_equal(
    res: CSRLinearExpression, reference: LinearExpression, dims: tuple[str, ...]
) -> None:
    """Compare a contraction with a dense reference independently of term layout."""
    assert res.grid.dims == dims
    assert_cells_equal(res.to_dense(), reference, dims)


def flat_operand(
    da: xr.DataArray, contracted_dims: tuple[str, ...], new_dims: tuple[str, ...]
) -> scipy.sparse.csr_array:
    """The C-order flattening ``contracted`` expects, as a sparse matrix."""
    values = da.transpose(*contracted_dims, *new_dims).to_numpy()
    n_contracted = int(np.prod([da.sizes[d] for d in contracted_dims], dtype=int))
    return scipy.sparse.csr_array(values.reshape(n_contracted, -1))


def gen_csr(c: Case) -> CSRLinearExpression:
    return CSRLinearExpression.from_dense((c.eff * c.gen_p).data, c.m)


@pytest.mark.parametrize("n_snap", [3, 70], ids=["one-chunk", "chunk-boundary"])
@pytest.mark.parametrize("zeros", [False, True], ids=["dense-C", "sparse-C"])
def test_contracted_partial_matches_dense(n_snap: int, zeros: bool) -> None:
    require_v1()
    c = base_model(n_snap=n_snap)
    expr = c.eff * c.gen_p
    values = np.arange(1.0, expr.data.sizes["gen"] * 2 + 1).reshape(-1, 2)
    if zeros:
        values[::2] = 0.0
    operand = xr.DataArray(
        values, coords={"gen": expr.indexes["gen"], "loc": ["L1", "L2"]}
    )
    res = gen_csr(c).contracted(
        flat_operand(operand, ("gen",), ("loc",)),
        ["gen"],
        [pd.Index(["L1", "L2"], name="loc")],
    )
    assert_contracted_equal(res, (expr * operand).sum("gen"), ("snapshot", "loc"))


def test_contracted_full_yields_zero_dim_grid() -> None:
    require_v1()
    c = base_model()
    expr = c.eff * c.gen_p
    operand = xr.DataArray(
        np.arange(
            expr.data.sizes["gen"] * expr.data.sizes["snapshot"], dtype=float
        ).reshape(expr.data.sizes["gen"], -1),
        coords={"gen": expr.indexes["gen"], "snapshot": expr.indexes["snapshot"]},
    )
    res = gen_csr(c).contracted(
        flat_operand(operand, ("gen", "snapshot"), ()), ["gen", "snapshot"], []
    )
    assert res.grid.size == 1
    assert_contracted_equal(res, (expr * operand).sum(["gen", "snapshot"]), ())


def test_contracted_with_two_new_dims() -> None:
    require_v1()
    c = base_model()
    expr = c.eff * c.gen_p
    operand = xr.DataArray(
        np.arange(expr.data.sizes["gen"] * 6, dtype=float).reshape(-1, 3, 2),
        coords={
            "gen": expr.indexes["gen"],
            "loc": ["L1", "L2", "L3"],
            "scen": [0, 1],
        },
    )
    res = gen_csr(c).contracted(
        flat_operand(operand, ("gen",), ("loc", "scen")),
        ["gen"],
        [pd.Index(["L1", "L2", "L3"], name="loc"), pd.Index([0, 1], name="scen")],
    )
    assert_contracted_equal(
        res, (expr * operand).sum("gen"), ("snapshot", "loc", "scen")
    )


def test_contracted_zero_operand_leaves_empty_rows() -> None:
    require_v1()
    c = base_model()
    expr = c.eff * c.gen_p
    operand = xr.DataArray(
        np.zeros((expr.data.sizes["gen"], 2)),
        coords={"gen": expr.indexes["gen"], "loc": ["L1", "L2"]},
    )
    res = gen_csr(c).contracted(
        flat_operand(operand, ("gen",), ("loc",)),
        ["gen"],
        [pd.Index(["L1", "L2"], name="loc")],
    )
    assert res.csr.nnz == 0
    assert (res.const == 0).all()
    assert (res.to_dense().data.vars == -1).all()


def test_contracted_reorders_non_trailing_contracted_dim() -> None:
    """``flow_t`` is stored as (snapshot, line); contracting ``line`` needs a reorder."""
    require_v1()
    c = base_model()
    expr = 1.0 * c.flow_t
    csr = CSRLinearExpression.from_dense(expr.data, c.m)
    assert csr.grid.dims == ("snapshot", "line")
    operand = xr.DataArray(
        np.arange(expr.data.sizes["line"] * 2, dtype=float).reshape(-1, 2),
        coords={"line": expr.indexes["line"], "loc": ["L1", "L2"]},
    )
    res = csr.contracted(
        flat_operand(operand, ("line",), ("loc",)),
        ["line"],
        [pd.Index(["L1", "L2"], name="loc")],
    )
    assert_contracted_equal(res, (expr * operand).sum("line"), ("snapshot", "loc"))


LOC = pd.Index(["L1", "L2"], name="loc")


def loc_operand(index: pd.Index) -> xr.DataArray:
    return xr.DataArray(np.ones((len(index), len(LOC))), coords=[index, LOC])


def tagged_group(c: Case) -> LinearExpression:
    """Generation grouped into ``group`` with ``bus``/``tag`` as auxiliary coords."""
    grouper = pd.DataFrame({"bus": c.gbus, "tag": c.gbus})
    return (1.0 * c.gen_p).groupby(grouper).sum(sparse=True, observed=True)


def test_contracted_keeps_aux_coords_on_kept_dims_only() -> None:
    require_v1()
    c = base_model()
    csr = tagged_group(c)._csr
    assert csr is not None
    assert set(csr.grid.aux) == {"bus", "tag"}
    kept = csr.contracted(
        flat_operand(
            loc_operand(csr.grid.indexes["snapshot"]), ("snapshot",), ("loc",)
        ),
        ["snapshot"],
        [LOC],
    )
    assert set(kept.grid.aux) == {"bus", "tag"}
    dropped = csr.contracted(
        flat_operand(loc_operand(csr.grid.indexes["group"]), ("group",), ("loc",)),
        ["group"],
        [LOC],
    )
    assert dropped.grid.aux == {}


@pytest.mark.parametrize(
    ("op", "aux_dims"),
    [
        (lambda g: g.renamed({"group": "g"}), {"bus": "g", "tag": "g"}),
        (
            lambda g: g.reordered(["snapshot", "group"]),
            {"bus": "group", "tag": "group"},
        ),
        (lambda g: g.reordered(["snapshot"]), {}),
        (
            lambda g: g.with_indexes({"snapshot": [5, 6, 7]}),
            {"bus": "group", "tag": "group"},
        ),
        (lambda g: g.combined([g], "outer"), {}),
    ],
    ids=["renamed", "transposed", "dropped", "relabelled", "combined"],
)
def test_grid_ops_carry_aux_coords(
    op: Callable[[Grid], Grid], aux_dims: dict[str, str]
) -> None:
    require_v1()
    csr = tagged_group(base_model())._csr
    assert csr is not None
    assert {n: d for n, (d, _) in op(csr.grid).aux.items()} == aux_dims


def test_grid_conformed_reindexes_aux_coords_and_equality_sees_them() -> None:
    require_v1()
    csr = tagged_group(base_model())._csr
    assert csr is not None
    grid = csr.grid
    labels = grid.indexes["group"][::-1]
    conformed = grid.conformed(grid.with_indexes({"group": labels}))
    tags = pd.Series(grid.aux["tag"][1], index=grid.indexes["group"])
    assert np.array_equal(conformed.aux["tag"][1], tags.loc[labels].to_numpy())
    assert grid == replace(grid)
    assert grid != replace(grid, aux={})
    assert grid.same_layout(replace(grid, aux={}))


def test_matmul_absent_cells_have_const_zero_and_no_terms() -> None:
    require_v1()
    c = base_model()
    expr = c.eff * c.gen_p
    snaps = expr.indexes["snapshot"]
    expr = expr.where(xr.DataArray(snaps != 1, coords=[snaps]))
    operand = loc_operand(expr.indexes["gen"])

    res = expr @ operand

    absent = res.sel(snapshot=1)
    assert (absent.data.vars == -1).all()
    assert (absent.const == 0).all()
    assert_cells_equal(res, (expr * operand).sum("gen"), ("snapshot", "loc"))


def test_matmul_prunes_explicit_zero_coefficients() -> None:
    """Unlike ``added``, ``@`` drops zero terms; the cell stays active via ``const``."""
    require_v1()
    c = base_model()
    expr = 0.0 * c.gen_p + 1.0
    operand = loc_operand(expr.indexes["gen"])

    res = expr @ operand

    assert (res.data.vars == -1).all()
    assert (res.const == len(expr.indexes["gen"])).all()
    assert_cells_equal(res, (expr * operand).sum("gen"), ("snapshot", "loc"))


@pytest.mark.parametrize("kind", ["nan", "mismatch", "reorder", "aux"])
def test_matmul_operand_errors_match_multiplication(kind: str) -> None:
    """§5, §8 and §11 on the constant are raised as ``*`` raises them."""
    require_v1()
    c = base_model()
    expr = c.eff * c.gen_p
    gens = expr.indexes["gen"]
    values = np.ones((len(gens), len(LOC)))
    index = gens
    if kind == "nan":
        values[0, 0] = np.nan
    elif kind == "mismatch":
        index = pd.Index([f"g{i}" for i in range(len(gens))], name="gen")
    elif kind == "reorder":
        index = gens[::-1]
    operand = xr.DataArray(values, coords=[index, LOC])
    if kind == "aux":
        operand = operand.assign_coords(tag=("gen", ["X"] * len(gens)))
        expr = expr.assign_coords(tag=("gen", ["Y"] * len(gens)))

    with pytest.raises(ValueError) as dense:
        expr * operand
    with pytest.raises(ValueError) as sparse:
        expr._sparse_matmul(operand)
    assert str(sparse.value) == str(dense.value)


@pytest.mark.parametrize("kind", ["non-unique", "multiindex"])
def test_matmul_falls_back_to_dense_on_unsupported_labels(kind: str) -> None:
    require_v1()
    m = Model()
    labels = ["a", "a", "b"] if kind == "non-unique" else ["a", "b", "c"]
    x = m.add_variables(coords=[pd.Index(labels, name="d")], name="x")
    expr = 1.0 * x
    coords: Any = {"d": expr.indexes["d"]}
    if kind == "multiindex":
        keys = pd.MultiIndex.from_tuples(
            [(0, "a"), (0, "b"), (1, "a")], names=["p", "q"]
        )
        coords = xr.Coordinates.from_pandas_multiindex(keys, "d")
        expr = LinearExpression(expr.data.drop_vars("d").assign_coords(coords), m)
    operand = (
        xr.DataArray(np.ones((3, len(LOC))), dims=["d", "loc"])
        .assign_coords(coords)
        .assign_coords(loc=LOC)
    )

    res = expr @ operand

    assert res._csr is None
    assert_linequal(res, (expr * operand).sum("d"))


@pytest.mark.parametrize("empty", ["kept", "contracted"], ids=["kept", "contracted"])
def test_matmul_with_zero_length_dim_matches_dense(empty: str) -> None:
    """A zero-length grid dim contracts to the same empty result as the dense path."""
    require_v1()
    m = Model()
    gens = pd.Index(["g0", "g1"], name="gen")
    empties = pd.Index([], name="empty", dtype=object)
    x = m.add_variables(coords=[gens, empties], name="x")
    expr = 1.0 * x
    contracted = "gen" if empty == "kept" else "empty"
    operand = loc_operand(expr.indexes[contracted])

    res = expr @ operand

    reference = (expr * operand).sum(contracted)
    assert res.coord_dims == ("empty" if empty == "kept" else "gen", "loc")
    assert res.coord_sizes == reference.coord_sizes
    assert res.data.vars.size == reference.data.vars.size == 0
    assert np.allclose(res.const, reference.const)


def test_matmul_dimensionless_expression_matches_dense() -> None:
    """A grid-less expression has nothing to contract and stays on the dense path."""
    require_v1()
    c = base_model()
    expr = (1.0 * c.gen_p).sum()
    operand = xr.DataArray(np.arange(2.0), coords=[LOC])

    res = expr @ operand

    assert res._csr is None
    assert_linequal(res, (expr * operand).sum([]))


def test_matmul_keeps_aux_coords_on_kept_dims_only() -> None:
    require_v1()
    c = base_model()
    grouped = tagged_group(c)
    assert grouped._csr is not None
    indexes = grouped._csr.grid.indexes

    kept = grouped @ loc_operand(indexes["snapshot"])
    dropped = grouped @ loc_operand(indexes["group"])

    assert {"bus", "tag"} <= set(kept.coords)
    assert not {"bus", "tag"} & set(dropped.coords)


def test_matmul_output_dim_order_matches_dense() -> None:
    require_v1()
    c = base_model()
    expr = c.eff * c.gen_p
    operand = loc_operand(expr.indexes["gen"])

    res = expr @ operand

    assert res.data.vars.dims == ("snapshot", "loc", TERM_DIM)
    assert res.data.vars.dims == (expr * operand).sum("gen").data.vars.dims


def test_matmul_keeps_csr_backing_through_chain() -> None:
    require_v1()
    c = base_model()
    snaps = c.gen_p.indexes["snapshot"]
    operand = loc_operand(snaps)
    gen = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True)
    flow = (1.0 * c.flow).groupby(c.bus0).sum(sparse=True)

    res = gen @ operand
    assert res._csr is not None
    assert_linequal(res, gen.dot(operand))
    assert_cells_equal(
        res,
        (c.eff * c.gen_p).groupby(c.gbus).sum() @ operand,
        ("bus", "loc"),
    )

    total = res + (flow @ operand)
    assert total._csr is not None
    con = c.m.add_constraints(total <= 0, name="matmul", freeze=True)
    assert isinstance(con, CSRConstraint)


@pytest.mark.legacy
def test_matmul_legacy_stays_dense() -> None:
    c = base_model()
    expr = c.eff * c.gen_p
    operand = loc_operand(expr.indexes["gen"])

    res = expr @ operand

    assert res._csr is None
    assert_linequal(res, (expr * operand).sum("gen"))


def test_quadratic_matmul_stays_on_dense_path() -> None:
    require_v1()
    c = base_model()
    quad = (1.0 * c.gen_p) * (1.0 * c.gen_p)
    operand = loc_operand(quad.indexes["gen"])

    res = quad @ operand

    assert_quadequal(res, (quad * operand).sum("gen"))


def test_matmul_peak_memory() -> None:
    require_v1()
    n_branch, n_cycle, n_snap = 1000, 300, 100
    dense_rectangle_bytes = n_snap * n_branch * n_cycle * 16
    branches = pd.Index([f"br{i}" for i in range(n_branch)], name="branch")
    cycles = pd.Index(range(n_cycle), name="cycle")
    snaps = pd.Index(range(n_snap), name="snapshot")
    rng = np.random.default_rng(0)
    incidence = xr.DataArray(
        rng.uniform(size=(n_branch, n_cycle)) < 0.01, coords=[branches, cycles]
    ).astype(float)
    m = Model()
    expr = 1.0 * m.add_variables(coords=[branches, snaps], name="flow")

    tracemalloc.start()
    try:
        res = expr @ incidence
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert res.coord_dims == ("snapshot", "cycle")
    assert peak < dense_rectangle_bytes / 4


def full_contraction(c: Case) -> LinearExpression:
    grouped = (1.0 * c.gen_p).groupby(c.gbus).sum(sparse=True)
    ones = xr.DataArray(np.ones(grouped.shape[:2]), coords=grouped.coords)
    return grouped @ ones


SPARSE_BUILDS: dict[str, Callable[[Case], LinearExpression]] = {
    "grouped": lambda c: (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True),
    "aux": tagged_group,
    "absent": lambda c: keyed_model()[1].groupby(["period", "season"]).sum(sparse=True),
    "zero-dim": full_contraction,
}

METADATA: dict[str, Callable[[LinearExpression], Any]] = {
    "shape": lambda e: e.shape,
    "size": lambda e: e.size,
    "ndim": lambda e: e.ndim,
    "sizes": lambda e: dict(e.sizes),
    "dims": lambda e: e.dims,
    "coord_dims": lambda e: e.coord_dims,
    "coord_sizes": lambda e: e.coord_sizes,
    "coord_names": lambda e: e.coord_names,
    "coords": lambda e: xr.Dataset(coords=e.coords),
    "indexes": lambda e: {k: list(v) for k, v in e.indexes.items()},
    "isnull": lambda e: e.isnull(),
    "repr": lambda e: repr(e).splitlines()[2:],
}


@pytest.mark.parametrize("attr", list(METADATA))
@pytest.mark.parametrize("build", list(SPARSE_BUILDS))
def test_metadata_is_served_without_densifying(build: str, attr: str) -> None:
    require_v1()
    sparse = SPARSE_BUILDS[build](base_model())
    assert sparse._csr is not None
    dense = sparse._csr.to_dense()
    got, want = METADATA[attr](sparse), METADATA[attr](dense)
    assert sparse.is_sparse
    if isinstance(got, xr.Dataset | xr.DataArray):
        xr.testing.assert_identical(got, want)
    else:
        assert got == want


def test_is_sparse_tracks_backing_and_repr_marks_it() -> None:
    require_v1()
    c = base_model()
    sparse = SPARSE_BUILDS["grouped"](c)
    assert sparse.is_sparse
    assert repr(sparse).startswith("LinearExpression (sparse) [")
    sparse.data
    assert not sparse.is_sparse
    assert repr(sparse).startswith("LinearExpression [")
    assert not (1.0 * c.gen_p).is_sparse


DENSIFY_OPS: dict[str, tuple[Callable[[LinearExpression, Case], Any], str]] = {
    "data": (lambda e, c: e.data, "`.data` read"),
    "merge": (lambda e, c: e + 1.0 * c.flow, "over different dimensions"),
    "matmul": (lambda e, c: e @ loc_operand(c.gbus.index), "sharing no dimension"),
    "rhs": (lambda e, c: e <= 1.0 * c.gen_p.sum("gen"), "non-constant rhs"),
    "mutable": (lambda e, c: (e == c.load).mutable(), "`mutable\\(\\)`"),
    "new-dim": (lambda e, c: e * xr.DataArray([1.0, 2.0], coords=[LOC]), "new dim"),
    "sum-kwargs": (lambda e, c: e.sum(dims="bus"), "`.data` read"),
    "groupby-fallback": (
        lambda e, c: e.groupby(halves(e)).sum(use_fallback=True),
        "`.data` read",
    ),
    "isel-pointwise": (
        lambda e, c: e.isel(bus=xr.DataArray([0, 1], dims="pt")),
        "`isel` over MultiIndex labels, or introducing dimensions",
    ),
    "where-new-dim": (
        lambda e, c: e.where(xr.DataArray([True, False], coords=[LOC])),
        "`where` over MultiIndex labels, or introducing dimensions",
    ),
    "where-callable": (
        lambda e, c: e.where(lambda ds: ds.const > 0),
        "`where` with a condition that is no DataArray",
    ),
}


def halves(e: LinearExpression) -> pd.Series:
    """Alternating ``a``/``b`` grouper over the expression's first dimension."""
    index = e.indexes[e.coord_dims[0]]
    return pd.Series(np.arange(len(index)) % 2, index=index).map({0: "a", 1: "b"})


@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "default"])
@pytest.mark.parametrize("op", list(DENSIFY_OPS))
def test_warn_on_densify_names_the_reason(op: str, enabled: bool) -> None:
    require_v1()
    c = base_model()
    sparse = SPARSE_BUILDS["grouped"](c)
    func, reason = DENSIFY_OPS[op]
    with linopy.options as opts, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", linopy.PerformanceWarning)
        opts.set_value(warn_on_densify=enabled)
        func(sparse, c)
    notices = [w for w in caught if issubclass(w.category, linopy.PerformanceWarning)]
    if not enabled:
        assert notices == []
        return
    assert any(re.search(reason, str(w.message)) for w in notices)
    assert all(w.filename == __file__ for w in notices)


def grid_operand(e: LinearExpression) -> xr.DataArray:
    """Positive values over the expression's full grid."""
    values = np.random.default_rng(1).uniform(1, 2, e.shape[:-1])
    return xr.DataArray(values, coords=[e.indexes[d] for d in e.coord_dims])


def first_dim_operand(e: LinearExpression) -> xr.DataArray:
    return grid_operand(e).isel({e.coord_dims[-1]: 0}, drop=True)


OPERANDS: dict[str, Callable[[LinearExpression], Any]] = {
    "dataarray": grid_operand,
    "subset-dims": first_dim_operand,
    "series": lambda e: first_dim_operand(e).to_series(),
    "ndarray": lambda e: grid_operand(e).to_numpy(),
    "zeros": lambda e: xr.zeros_like(grid_operand(e)),
    "operand-aux": lambda e: first_dim_operand(e).assign_coords(
        extra=(e.coord_dims[0], np.arange(e.shape[0]))
    ),
}

ELEMENTWISE_OPS: dict[str, Callable[[LinearExpression, Any], LinearExpression]] = {
    "mul": lambda e, x: e * x,
    "rmul": lambda e, x: x * e,
    "truediv": lambda e, x: e / x,
    "div": lambda e, x: e.div(x, join="left"),
    "add": lambda e, x: e + x,
    "radd": lambda e, x: x + e,
    "sub": lambda e, x: e - x,
    "rsub": lambda e, x: x - e,
}


@pytest.mark.parametrize(
    ("build", "operand", "op"),
    [
        (b, o, op)
        for b in ["grouped", "aux", "absent"]
        for o in OPERANDS
        for op in ELEMENTWISE_OPS
        if (b, o) != ("absent", "ndarray") and not (o == "zeros" and "div" in op)
    ],
)
def test_elementwise_constant_ops_stay_csr_and_match_dense(
    build: str, operand: str, op: str
) -> None:
    require_v1()
    sparse = SPARSE_BUILDS[build](base_model())
    assert sparse._csr is not None
    dense = sparse._csr.to_dense()
    x = OPERANDS[operand](dense)
    func = ELEMENTWISE_OPS[op]
    res = func(sparse, x)
    assert res.is_sparse
    assert_linequal(res, func(dense, x))


@pytest.mark.parametrize("join", ["inner", "outer", "left", "right"])
@pytest.mark.parametrize("fill_value", [None, linopy.ABSENT], ids=["fill", "absent"])
@pytest.mark.parametrize("method", ["add", "sub", "mul", "div"])
def test_elementwise_join_on_mismatched_labels_stays_csr_and_matches_dense(
    method: str, fill_value: Any, join: JoinOptions
) -> None:
    require_v1()
    sparse = SPARSE_BUILDS["grouped"](base_model())
    assert sparse._csr is not None
    dense = sparse._csr.to_dense()
    x = first_dim_operand(dense).isel(bus=[1, 3]).reindex(bus=["bus3", "bus1", "x"])
    x = x.fillna(2.0)
    res = getattr(sparse, method)(x, join=join, fill_value=fill_value)
    want = getattr(dense, method)(x, join=join, fill_value=fill_value)
    assert res.is_sparse
    xr.testing.assert_identical(res.isnull(), want.isnull())
    assert_cells_equal(res, want, ("bus", "snapshot"))


@pytest.mark.parametrize("op", ["mul", "add"])
@pytest.mark.parametrize("kind", ["nan", "nan-scalar", "mismatch", "reorder", "aux"])
def test_elementwise_operand_errors_match_dense(kind: str, op: str) -> None:
    """§5, §8 and §11 on the constant are raised as on the dense path."""
    require_v1()
    sparse = SPARSE_BUILDS["aux"](base_model())
    assert sparse._csr is not None
    dense = sparse._csr.to_dense()
    x: Any = grid_operand(dense)
    if kind == "nan":
        x[0, 0] = np.nan
    elif kind == "nan-scalar":
        x = np.float32("nan")
    elif kind == "mismatch":
        x = x.isel(group=slice(1, None))
    elif kind == "reorder":
        x = x.isel(group=slice(None, None, -1))
    else:
        x = x.assign_coords(tag=("group", ["X"] * x.sizes["group"]))
    func = ELEMENTWISE_OPS[op]
    with pytest.raises(ValueError) as want:
        func(dense, x)
    with pytest.raises(ValueError) as got:
        func(sparse, x)
    assert str(got.value) == str(want.value)
    assert sparse.is_sparse


@pytest.mark.parametrize("op", ["mul", "add"])
def test_elementwise_operand_over_new_dim_falls_back_to_dense(op: str) -> None:
    require_v1()
    sparse = SPARSE_BUILDS["grouped"](base_model())
    assert sparse._csr is not None
    dense = sparse._csr.to_dense()
    x = xr.DataArray([1.0, 2.0], coords=[LOC])
    func = ELEMENTWISE_OPS[op]
    res = func(sparse, x)
    assert not res.is_sparse
    assert_linequal(res, func(dense, x))


def test_elementwise_ops_keep_absent_cells_termless() -> None:
    require_v1()
    sparse = SPARSE_BUILDS["absent"](base_model())
    csr = (sparse * grid_operand(sparse) + 1.0)._csr
    assert csr is not None
    absent = np.isnan(csr.const)
    assert absent.any()
    assert (np.diff(csr.csr.indptr)[absent] == 0).all()


def assert_sparse_matches(res: LinearExpression, want: LinearExpression) -> None:
    """CSR backing kept, and equal to the dense reference up to term layout."""
    assert res.is_sparse
    assert res.coord_dims == want.coord_dims
    xr.testing.assert_identical(
        xr.Dataset(coords=res.coords), xr.Dataset(coords=want.coords)
    )
    xr.testing.assert_identical(res.isnull(), want.isnull())
    assert_cells_equal(res, want, tuple(map(str, want.coord_dims)))


@contextmanager
def no_densify() -> Iterator[None]:
    """Fail on any densify notice inside the block."""
    with linopy.options as opts, warnings.catch_warnings():
        warnings.simplefilter("error", linopy.PerformanceWarning)
        opts.set_value(warn_on_densify=True)
        yield


SUM_DIMS: dict[str, Callable[[tuple[str, ...]], Any]] = {
    "first": lambda d: d[0],
    "last-list": lambda d: [d[-1]],
    "leading": lambda d: list(d[:-1]),
    "all": lambda d: None,
    "all-reversed": lambda d: list(d)[::-1],
    "ellipsis": lambda d: ...,
    "empty": lambda d: [],
    "term": lambda d: [TERM_DIM],
}


@pytest.mark.parametrize("dims", list(SUM_DIMS))
@pytest.mark.parametrize("build", ["grouped", "aux", "absent"])
def test_sum_stays_csr_and_matches_dense(build: str, dims: str) -> None:
    require_v1()
    sparse = SPARSE_BUILDS[build](base_model())
    assert sparse._csr is not None
    dense = sparse._csr.to_dense()
    dim = SUM_DIMS[dims](tuple(map(str, sparse.coord_dims)))
    with no_densify():
        res = sparse.sum(dim)
    assert_sparse_matches(res, dense.sum(dim))


@pytest.mark.parametrize("drop_zeros", [False, True])
def test_sum_keeps_explicit_zeros_unless_dropped(drop_zeros: bool) -> None:
    require_v1()
    c = base_model()
    sparse = (0.0 * c.gen_p).groupby(c.gbus).sum(sparse=True)
    assert sparse._csr is not None
    dense = sparse._csr.to_dense()
    res = sparse.sum("snapshot", drop_zeros=drop_zeros)
    assert res._csr is not None
    assert (res._csr.csr.nnz == 0) == drop_zeros
    assert_sparse_matches(res, dense.sum("snapshot", drop_zeros=drop_zeros))


def test_sum_unknown_dim_raises_like_dense() -> None:
    require_v1()
    sparse = SPARSE_BUILDS["grouped"](base_model())
    assert sparse._csr is not None
    dense = sparse._csr.to_dense()
    with pytest.raises(KeyError) as want:
        dense.sum("nodim")
    with pytest.raises(KeyError) as got:
        sparse.sum("nodim")
    assert str(got.value) == str(want.value)


CHAINS: dict[str, Callable[[LinearExpression], LinearExpression]] = {
    "group-group": lambda e: e.groupby(halves(e)).sum(),
    "group-sum": lambda e: e.groupby(halves(e)).sum().sum("snapshot"),
    "sum-group": lambda e: e.sum("snapshot").groupby(halves(e)).sum(),
}


@pytest.mark.parametrize("chain", list(CHAINS))
@pytest.mark.parametrize("build", ["grouped", "aux", "absent"])
def test_chained_groupby_stays_csr_and_matches_dense(build: str, chain: str) -> None:
    require_v1()
    sparse = SPARSE_BUILDS[build](base_model())
    assert sparse._csr is not None
    dense = sparse._csr.to_dense()
    func = CHAINS[chain]
    with no_densify():
        res = func(sparse)
    assert_sparse_matches(res, func(dense))


@pytest.mark.parametrize("observed", [False, True])
def test_chained_namelist_groupby_on_aux_coords_matches_dense(observed: bool) -> None:
    require_v1()
    sparse = SPARSE_BUILDS["aux"](base_model())
    assert sparse._csr is not None
    dense = sparse._csr.to_dense()
    res = sparse.groupby(["bus", "tag"]).sum(observed=observed)
    assert_sparse_matches(res, dense.groupby(["bus", "tag"]).sum(observed=observed))


def test_chained_groupby_sparse_false_densifies() -> None:
    require_v1()
    sparse = SPARSE_BUILDS["grouped"](base_model())
    assert sparse._csr is not None
    dense = sparse._csr.to_dense()
    res = sparse.groupby(halves(sparse)).sum(sparse=False)
    assert not res.is_sparse
    assert_linequal(res, dense.groupby(halves(dense)).sum())


def dim0(e: LinearExpression) -> str:
    return str(e.coord_dims[0])


def labels0(e: LinearExpression) -> pd.Index:
    return e.indexes[dim0(e)]


def alternating(e: LinearExpression) -> xr.DataArray:
    """True on every other label of the expression's first dimension."""
    labels = labels0(e)
    return xr.DataArray(np.arange(len(labels)) % 2 == 0, coords=[labels])


SELECTIONS: dict[str, Callable[[LinearExpression], LinearExpression]] = {
    "sel-scalar": lambda e: e.sel({dim0(e): labels0(e)[1]}),
    "sel-scalar-drop": lambda e: e.sel({dim0(e): labels0(e)[1]}, drop=True),
    "sel-list": lambda e: e.sel({dim0(e): list(labels0(e)[[2, 0]])}),
    "sel-slice": lambda e: e.sel({dim0(e): slice(labels0(e)[1], None)}),
    "isel-int": lambda e: e.isel({str(e.coord_dims[-1]): 0}),
    "isel-list": lambda e: e.isel({dim0(e): [2, 1, 1]}),
    "isel-slice": lambda e: e.isel({dim0(e): slice(None, None, -2)}),
    "isel-empty": lambda e: e.isel({dim0(e): []}),
    "loc-dict": lambda e: e.loc[{dim0(e): list(labels0(e)[:2])}],
    "loc-scalar": lambda e: e.loc[labels0(e)[1]],
    "getitem-int": lambda e: e[0],
    "getitem-tuple": lambda e: e[1:, [1, 0]],
    "where-dim": lambda e: e.where(alternating(e)),
    "where-grid": lambda e: e.where(grid_operand(e) > 1.5),
    "where-other": lambda e: e.where(alternating(e), 3.0),
    "where-drop": lambda e: e.where(alternating(e), drop=True),
    "where-drop-mismatch": lambda e: e.where(
        alternating(e).isel({dim0(e): slice(1, None)}), drop=True
    ),
    "sel-scalar-sum": lambda e: e.sel({dim0(e): labels0(e)[1]}).sum(),
}


@pytest.mark.parametrize("select", list(SELECTIONS))
@pytest.mark.parametrize("build", ["grouped", "aux", "absent"])
def test_selection_stays_csr_and_matches_dense(build: str, select: str) -> None:
    require_v1()
    sparse = SPARSE_BUILDS[build](base_model())
    assert sparse._csr is not None
    dense = sparse._csr.to_dense()
    func = SELECTIONS[select]
    with no_densify():
        res = func(sparse)
    assert_sparse_matches(res, func(dense))


def test_selection_keeps_absent_cells_termless() -> None:
    require_v1()
    sparse = SPARSE_BUILDS["absent"](base_model())
    csr = sparse.where(alternating(sparse), 3.0).isel(season=[1, 0, 1])._csr
    assert csr is not None
    absent = np.isnan(csr.const)
    assert absent.any()
    assert (np.diff(csr.csr.indptr)[absent] == 0).all()


def test_scalar_selection_coords_merge_like_dense() -> None:
    require_v1()
    c = base_model()
    sparse = SPARSE_BUILDS["grouped"](c)
    assert sparse._csr is not None
    dense = sparse._csr.to_dense()
    flow = (1.0 * c.flow).groupby(c.bus0).sum()
    with no_densify():
        res = sparse.sel(snapshot=1) + flow.sel(snapshot=1)
    assert_sparse_matches(res, dense.sel(snapshot=1) + flow.sel(snapshot=1))
    with pytest.raises(ValueError) as want:
        dense.sel(snapshot=1) + flow.sel(snapshot=2)
    with pytest.raises(ValueError) as got:
        sparse.sel(snapshot=1) + flow.sel(snapshot=2)
    assert str(got.value) == str(want.value)


def test_where_on_mismatched_labels_raises_like_dense() -> None:
    require_v1()
    sparse = SPARSE_BUILDS["grouped"](base_model())
    assert sparse._csr is not None
    dense = sparse._csr.to_dense()
    cond = alternating(dense).isel(bus=slice(1, None))
    with pytest.raises(ValueError) as want:
        dense.where(cond)
    with pytest.raises(ValueError) as got:
        sparse.where(cond)
    assert str(got.value) == str(want.value)
    assert sparse.is_sparse


SELECTION_FALLBACKS = ["isel-pointwise", "where-new-dim", "where-callable"]


@pytest.mark.parametrize("op", SELECTION_FALLBACKS)
def test_selection_fallbacks_match_dense(op: str) -> None:
    require_v1()
    c = base_model()
    sparse = SPARSE_BUILDS["grouped"](c)
    assert sparse._csr is not None
    dense = sparse._csr.to_dense()
    func = DENSIFY_OPS[op][0]
    res = func(sparse, c)
    assert not res.is_sparse
    assert_linequal(res, func(dense, c))


MASKS: dict[str, Callable[[LinearExpression], Any]] = {
    "dim": alternating,
    "grid": lambda e: grid_operand(e) > 1.5,
    "ndarray": lambda e: (grid_operand(e) > 1.5).to_numpy(),
}


@pytest.mark.parametrize("mask", list(MASKS))
def test_add_constraints_mask_freezes_sparse_and_matches_dense(mask: str) -> None:
    require_v1()
    c1, c2 = base_model(), base_model()
    lhs = c2.balance_lhs(sparse=True)
    m = MASKS[mask](lhs)
    con1 = c1.m.add_constraints(c1.balance_lhs(False), ">=", c1.load, "bal", mask=m)
    with no_densify():
        con2 = c2.m.add_constraints(lhs, ">=", c2.load, "bal", mask=m, freeze=True)
    assert isinstance(con2, CSRConstraint)
    assert_frozen_equal(con1, con2)
