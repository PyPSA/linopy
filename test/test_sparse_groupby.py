"""
Tests for sparse groupby-sum (linopy.sparse_expression): type stability, transparent
materialization, and direct CSR realization under freeze. v1-only feature.
"""

from __future__ import annotations

import re
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

import linopy
from linopy import LinearExpression, Model, Variable
from linopy.constraints import Constraint, CSRConstraint
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

    gbus = pd.Series(buses[gen_bus], index=gens, name="bus")
    bus0 = pd.Series(buses[np.arange(n_bus)], index=lines, name="bus")
    bus1 = pd.Series(buses[(np.arange(n_bus) + 1) % n_bus], index=lines, name="bus")
    load = xr.DataArray(
        rng.uniform(1, 10, (n_bus, n_snap)), coords=[buses, snaps], name="load"
    ).sortby("bus")
    eff = xr.DataArray(rng.uniform(0.5, 1.5, len(gens)), coords=[gens])
    return Case(m, gen_p, flow, eff, gbus, bus0, bus1, load)


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
    assert res._payload is None


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
    assert sparse._payload is not None
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
    assert tot._payload is not None
    assert_linequal(tot, linopy.merge([dense, flow], join="outer"))

    con = c.m.add_constraints(tot >= c.load, name="bal", freeze=True)
    assert isinstance(con, CSRConstraint)
    assert con.ncons == int(mask.sum()) * c.load.sizes["snapshot"]


@pytest.mark.parametrize(
    "grouper, kwargs",
    [
        (["gen", "snapshot"], {}),
        ("gen", {"use_fallback": True}),
        (["gen"], {"observed": True}),
    ],
)
def test_explicit_sparse_raises_on_unsupported_grouper(
    grouper: str | list[str], kwargs: dict
) -> None:
    require_v1()
    c = base_model()
    with pytest.raises(ValueError, match="single-key grouper"):
        (1.0 * c.gen_p).groupby(grouper).sum(sparse=True, **kwargs)


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
    assert sparse._payload is not None
    dense = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=False).reindex(indexers)
    assert_linequal(sparse, dense)


def test_reindex_falls_back_to_dense_for_unsupported_kwargs() -> None:
    require_v1()
    c = base_model()
    sparse = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True)
    res = sparse.reindex(bus=["bus3", "bus0"], copy=False)
    assert res._payload is None
    dense = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=False)
    assert_linequal(res, dense.reindex(bus=["bus3", "bus0"]))


def test_reindex_merge_chain_freezes_csr() -> None:
    require_v1()
    c1, c2 = base_model(), base_model()
    con1 = c1.m.add_constraints(reindexed_balance(c1, False) == c1.load, name="bal")
    tot = reindexed_balance(c2, True)
    assert tot._payload is not None
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
    assert filled._payload is not None
    dense = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=False).reindex(wide)
    assert_linequal(filled, dense.fillna(value))


def test_fillna_with_array_falls_back_to_dense() -> None:
    require_v1()
    c = base_model()
    fill = xr.zeros_like(c.load)
    sparse = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True)
    res = sparse.fillna(fill)
    assert res._payload is None
    dense = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=False)
    assert_linequal(res, dense.fillna(fill))


def test_rename_stays_csr_and_matches_dense() -> None:
    require_v1()
    c = base_model()
    sparse = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True).rename(bus="node")
    assert sparse._payload is not None
    dense = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=False).rename(bus="node")
    assert sparse.coord_dims == ("node", "snapshot")
    assert_linequal(sparse, dense)
