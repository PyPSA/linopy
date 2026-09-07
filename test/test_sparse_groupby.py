"""
Tests for sparse groupby-sum (linopy.sparse_expression): type stability, transparent
materialization, and direct CSR realization under freeze. v1-only feature.
"""

from __future__ import annotations

import re
import tracemalloc
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
from xarray.core.types import JoinOptions

import linopy
from linopy import LinearExpression, Model, Variable
from linopy.constraints import Constraint, ConstraintBase, CSRConstraint
from linopy.semantics import is_v1
from linopy.testing import assert_conequal, assert_linequal


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
    tot = linopy.merge([sparse, flow], join="outer")
    assert tot._csr is not None
    assert_linequal(tot, linopy.merge([dense, flow], join="outer"))

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
        assert set(csr.coords) == {"period", "season"}
    else:
        assert np.isnan(csr.const).sum() == 2 * 2
    assert csr.grid_dims == dense.coord_dims
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
    assert set(tot._csr.coords) == {"period", "season"}
    assert_linequal(tot, 2.0 * dense)

    other = dense.assign_coords(region=("group", list("abcd")))
    tot = sparse + other
    assert tot._csr is not None
    assert set(tot._csr.coords) == {"period", "season", "region"}
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
    res = linopy.merge(sparse, join=join)
    assert res._csr is not None
    assert res.coord_dims == dense[0].coord_dims
    assert_terms_equal(res, linopy.merge(dense, join=join))


@pytest.mark.parametrize("join", ["outer", "inner", "left", "right"])
def test_three_operand_cross_grid_merge_matches_dense(join: JoinOptions) -> None:
    require_v1()
    c = base_model()
    third_lines = ("line3", "line4")
    sparse = cross_grid_parts(c, True) + cross_grid_parts(c, True, third_lines)[1:]
    dense = cross_grid_parts(c, False) + cross_grid_parts(c, False, third_lines)[1:]
    res = linopy.merge(sparse, join=join)
    assert res._csr is not None
    assert_terms_equal(res, linopy.merge(dense, join=join))


def test_cross_grid_merge_absent_fill_matches_dense() -> None:
    require_v1()
    c = base_model()
    sparse, dense = cross_grid_parts(c, True), cross_grid_parts(c, False)
    res = linopy.merge(sparse, join="outer", fill_value=linopy.ABSENT)
    expected = linopy.merge(dense, join="outer", fill_value=linopy.ABSENT)
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
    res = linopy.merge([sparse[0], dense[1]], join="outer")
    assert res._csr is not None
    assert_terms_equal(res, linopy.merge(dense, join="outer"))


def test_cross_grid_merge_mixed_dense_operand_stays_csr() -> None:
    require_v1()
    c = base_model()
    sparse, dense = cross_grid_parts(c, True), cross_grid_parts(c, False)
    res = sparse[0].add(dense[1], join="outer")
    assert res._csr is not None
    assert_terms_equal(res, dense[0].add(dense[1], join="outer"))


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


def test_merge_with_aux_coord_operand_raises_like_dense() -> None:
    require_v1()
    c = base_model()
    sparse, dense = cross_grid_parts(c, True), cross_grid_parts(c, False)
    tag = xr.DataArray(["x", "y"], coords=[dense[1].indexes["bus"]])
    tagged = LinearExpression(dense[1].data.assign_coords(tag=tag), c.m)
    with pytest.raises(xr.MergeError, match="conflicting values for variable 'tag'"):
        linopy.merge([dense[0], tagged], join="outer")
    with pytest.raises(xr.MergeError, match="conflicting values for variable 'tag'"):
        linopy.merge([sparse[0], tagged], join="outer")


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
    res = linopy.merge([gen, flow], join="override")
    assert res._csr is not None
    dense = [
        (c.eff * c.gen_p).groupby(c.gbus).sum(),
        (1.0 * c.flow).groupby(c.bus1.str.upper()).sum(),
    ]
    assert_terms_equal(res, linopy.merge(dense, join="override"))


def test_cross_grid_balance_freezes_csr() -> None:
    require_v1()
    c1, c2 = base_model(), base_model()
    lhs1 = linopy.merge(cross_grid_parts(c1, False), join="outer")
    con1 = c1.m.add_constraints(lhs1 == c1.load, name="bal")
    lhs2 = linopy.merge(cross_grid_parts(c2, True), join="outer")
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
