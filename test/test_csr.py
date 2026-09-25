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
from linopy import LinearExpression, Model, QuadraticExpression, Variable
from linopy.constants import TERM_DIM
from linopy.constraints import Constraint, ConstraintBase, CSRConstraint
from linopy.csr import CSRLinearExpression, Grid, column_compacted_matmul
from linopy.semantics import is_v1
from linopy.testing import (
    assert_conequal,
    assert_linequal,
    assert_quadequal,
    assert_varequal,
)


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

    def balance_lhs(self) -> LinearExpression:
        return (
            self.gen_sum()
            + (1.0 * self.flow).groupby(self.bus0).sum()
            - (1.0 * self.flow).groupby(self.bus1).sum()
        )

    def gen_sum(self) -> LinearExpression:
        return (self.eff * self.gen_p).groupby(self.gbus).sum()


def base_model(
    gens_per_bus: tuple[int, ...] = (7, 1, 3, 1, 2),
    n_snap: int = 3,
    seed: int = 0,
    sparse: bool = False,
) -> Case:
    rng = np.random.default_rng(seed)
    n_bus = len(gens_per_bus)
    buses = pd.Index([f"bus{i}" for i in range(n_bus)], name="bus")
    gen_bus = np.repeat(np.arange(n_bus), gens_per_bus)
    gens = pd.Index([f"gen{i}" for i in range(len(gen_bus))], name="gen")
    lines = pd.Index([f"line{i}" for i in range(n_bus)], name="line")
    snaps = pd.Index(range(n_snap), name="snapshot")

    m = linopy.Model(sparse=sparse)
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


def twin_models(**kwargs: Any) -> tuple[Case, Case]:
    """A dense model and its sparse twin, built identically."""
    return base_model(**kwargs), base_model(sparse=True, **kwargs)


def canon(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["labels", "vars"])
        .agg(pl.col("coeffs").sum(), pl.col("sign").first(), pl.col("rhs").first())
        .sort(["labels", "vars"])
    )


def assert_frozen_equal(con1: ConstraintBase, con2: ConstraintBase) -> None:
    d1, d2 = canon(con1.to_polars()), canon(con2.to_polars())
    assert d1["labels"].equals(d2["labels"])
    assert d1["vars"].equals(d2["vars"])
    assert np.allclose(d1["coeffs"], d2["coeffs"])
    assert (d1["sign"] == d2["sign"]).all()
    assert np.allclose(d1["rhs"], d2["rhs"])
    labels = con1.labels.values.ravel()
    assert np.array_equal(np.sort(labels[labels != -1]), np.sort(con2.active_labels()))


def reindexed_balance(c: Case) -> LinearExpression:
    """Generation on all buses plus flow on two lines, both reindexed onto the load grid."""
    lines = ["line0", "line1"]
    gen = c.gen_sum()
    flow = (1.0 * c.flow.loc[lines]).groupby(c.bus0.loc[lines]).sum()
    parts = [gen.reindex(bus=c.load.bus), flow.reindex(bus=c.load.bus)]
    return linopy.merge(parts, join="outer", cls=LinearExpression)


def test_csr_requires_v1() -> None:
    c = base_model()
    deprecated = pytest.warns(FutureWarning, match="deprecated")
    if is_v1():
        with deprecated:
            res = (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True)
        assert type(res) is LinearExpression
        return
    with deprecated, pytest.raises(ValueError, match="requires v1 semantics"):
        (c.eff * c.gen_p).groupby(c.gbus).sum(sparse=True)
    with pytest.warns(FutureWarning, match="deprecated"):
        linopy.options["sparse_groupby"] = True
    try:
        res = c.gen_sum()
    finally:
        linopy.options["sparse_groupby"] = False
    assert res._csr is None


def test_csr_is_plain_linear_expression_and_materializes_equivalently() -> None:
    require_v1()
    c1, c2 = twin_models()
    sparse = c2.gen_sum()
    assert type(sparse) is LinearExpression
    assert sparse._csr is not None
    assert_linequal(sparse, c1.gen_sum())


def test_csr_composition_materializes_equivalently() -> None:
    require_v1()
    c1, c2 = twin_models()
    sparse = c2.balance_lhs()
    assert type(sparse) is LinearExpression
    assert sparse._csr is not None
    assert_linequal(sparse, c1.balance_lhs())


def test_scalar_ops_stay_csr() -> None:
    require_v1()
    c1, c2 = twin_models()
    sparse = -2.0 * c2.gen_sum()
    assert sparse._csr is not None
    assert_linequal(sparse, -2.0 * c1.gen_sum())


@pytest.mark.parametrize("sparse", [True, False], ids=["sparse", "dense"])
def test_zero_coefficient_rows_stay_active(sparse: bool) -> None:
    require_v1()
    c = base_model(sparse=sparse)
    lhs = (0.0 * c.gen_p).groupby(c.gbus).sum()
    lhs = lhs + (0.0 * c.flow).groupby(c.bus0).sum()
    nterm = lhs.nterm
    con = c.m.add_constraints(lhs == c.load, name="bal", freeze=True)
    assert len(con.active_labels()) == c.load.size
    assert lhs.nterm == nterm


def test_merge_keeps_absent_cell_absent() -> None:
    require_v1()
    c1, c2 = twin_models()
    mask = xr.DataArray(np.arange(len(c1.load.bus)) % 2 == 0, coords=[c1.load.bus])
    flow1, flow2 = (
        (1.0 * c.flow).groupby(c.bus0).sum(use_fallback=True).where(mask)
        for c in (c1, c2)
    )

    tot = linopy.merge([c2.gen_sum(), flow2], join="outer", cls=LinearExpression)
    assert tot._csr is not None and flow2._csr is None
    assert_linequal(
        tot, linopy.merge([c1.gen_sum(), flow1], join="outer", cls=LinearExpression)
    )

    con = c2.m.add_constraints(tot >= c2.load, name="bal")
    assert isinstance(con, CSRConstraint)
    assert con.ncons == int(mask.sum()) * c2.load.sizes["snapshot"]


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
    deprecated = pytest.warns(FutureWarning, match="deprecated")
    with deprecated, pytest.raises(ValueError, match="sparse=True supports only"):
        (1.0 * c.gen_p).groupby(grouper).sum(sparse=True, **kwargs)


def keyed_model(
    member_first: bool = True, sparse: bool = False
) -> tuple[Model, LinearExpression, xr.DataArray]:
    """
    ``(1 + x)`` over ``(s, snapshot)`` with ``period``/``season`` keys on ``s``;
    two of the six (period, season) combinations never occur.
    """
    n = 6
    s = pd.RangeIndex(n, name="s")
    snaps = pd.Index(range(2), name="snapshot")
    m = Model(sparse=sparse)
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
    _, dense_expr, _ = keyed_model(member_first)
    _, sparse_expr, _ = keyed_model(member_first, sparse=True)
    keys = ["period", "season"]
    sparse = sparse_expr.groupby(keys).sum(observed=observed)
    dense = dense_expr.groupby(keys).sum(observed=observed)
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
    c1, c2 = twin_models()
    dense, sparse = (
        (c.eff * c.gen_p).assign_coords(bus=("gen", c.gbus.to_numpy()))
        for c in (c1, c2)
    )
    grouper = ["bus"] if as_namelist else c1.gbus
    res = sparse.groupby(grouper).sum(observed=True)
    assert res._csr is not None
    assert_linequal(res, dense.groupby(grouper).sum())


def test_sparse_keeps_aux_coords_on_surviving_dims() -> None:
    require_v1()
    dense_expr, sparse_expr = (
        keyed_model(sparse=sparse_model)[1].assign_coords(tag=("snapshot", list("ab")))
        for sparse_model in (False, True)
    )
    for kwargs in ({}, {"observed": True}):
        sparse = sparse_expr.groupby(["period", "season"]).sum(**kwargs)
        dense = dense_expr.groupby(["period", "season"]).sum(**kwargs)
        assert set(sparse.coords) == set(dense.coords)
        assert_linequal(sparse, dense)


def test_dataframe_grouper_sparse_stays_compact() -> None:
    require_v1()
    _, dense, _ = keyed_model()
    _, expr, _ = keyed_model(sparse=True)
    df = dense.data[["period", "season"]].to_dataframe()[["period", "season"]]
    sparse = expr.groupby(df).sum()
    assert sparse._csr is not None
    assert sparse._csr.shape == (4, 2)
    assert_linequal(sparse, dense.groupby(df).sum())


def test_namelist_sparse_observed_freezes_compact() -> None:
    require_v1()
    m1, e1, rhs = keyed_model()
    m2, e2, _ = keyed_model(sparse=True)
    dense = m1.add_constraints(
        e1.groupby(["period", "season"]).sum(observed=True) == rhs,
        name="c",
        freeze=True,
    )
    sparse = m2.add_constraints(
        e2.groupby(["period", "season"]).sum(observed=True) == rhs, name="c"
    )
    assert isinstance(sparse, CSRConstraint)
    assert sparse.ncons == rhs.size
    assert_conequal(dense, sparse, strict=False)


def test_namelist_sparse_grid_absent_cells_inactive() -> None:
    require_v1()
    m1, e1, _ = keyed_model()
    m2, e2, _ = keyed_model(sparse=True)
    keys = ["period", "season"]
    dense = m1.add_constraints(e1.groupby(keys).sum() == 0, name="c", freeze=True)
    sparse = m2.add_constraints(e2.groupby(keys).sum() == 0, name="c")
    assert isinstance(sparse, CSRConstraint)
    assert sparse.ncons == 4 * 2
    assert_conequal(dense, sparse, strict=False)
    assert np.array_equal(
        np.sort(dense.active_labels()), np.sort(sparse.active_labels())
    )


def test_namelist_sparse_observed_keeps_aux_coords_through_merge() -> None:
    require_v1()
    _, dense_expr, _ = keyed_model()
    _, sparse_expr, _ = keyed_model(sparse=True)
    keys = ["period", "season"]
    sparse = sparse_expr.groupby(keys).sum(observed=True)
    dense = dense_expr.groupby(keys).sum(observed=True)
    assert sparse._csr is not None
    operand = sparse._csr.to_dense()
    tot = sparse + operand
    assert tot._csr is not None
    assert set(tot._csr.grid.aux) == {"period", "season"}
    assert_linequal(tot, 2.0 * dense)

    region = ("group", list("abcd"))
    tot = sparse + operand.assign_coords(region=region)
    assert tot._csr is not None
    assert set(tot._csr.grid.aux) == {"period", "season", "region"}
    xr.testing.assert_equal(
        tot.data.coords.to_dataset(),
        (dense + dense.assign_coords(region=region)).data.coords.to_dataset(),
    )

    conflicting = operand.assign_coords(season=("group", list("xyzw")))
    with pytest.raises(ValueError, match="conflicting values"):
        sparse + conflicting


def test_namelist_sparse_grid_warns_and_observed_silences() -> None:
    require_v1()
    n = 200
    s = pd.RangeIndex(n, name="s")
    m = Model(sparse=True)
    x = m.add_variables(coords=[s], name="x")
    expr = (1.0 * x).assign_coords(
        period=xr.DataArray(np.arange(n), dims="s", coords={"s": s}),
        season=xr.DataArray(np.arange(n), dims="s", coords={"s": s}),
    )
    with pytest.warns(UserWarning, match="dense .* grid"):
        expr.groupby(["period", "season"]).sum()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = expr.groupby(["period", "season"]).sum(observed=True)
    assert res._csr is not None
    assert res._csr.shape == (n,)


def test_nan_multikey_grouper_raises_eagerly() -> None:
    require_v1()
    _, expr, _ = keyed_model(sparse=True)
    period = expr.data["period"]
    expr = expr.assign_coords(period=period.where(period > 0))
    with pytest.raises(ValueError, match="NaN values"):
        expr.groupby(["period", "season"]).sum()


def test_freeze_realizes_csr_without_dense_rectangle() -> None:
    require_v1()
    c1, c2 = twin_models()
    con1 = c1.m.add_constraints(c1.balance_lhs() == c1.load, name="bal")
    con2 = c2.m.add_constraints(c2.balance_lhs() == c2.load, name="bal")

    assert isinstance(con2, CSRConstraint)
    assert_frozen_equal(con1, con2)


def test_freeze_false_falls_back_to_identical_dense_constraint() -> None:
    """The fallback is canonical-form, so compare mathematically (strict=False)."""
    require_v1()
    c1, c2 = twin_models()
    con1 = c1.m.add_constraints(c1.balance_lhs() == c1.load, name="bal")
    con2 = c2.m.add_constraints(c2.balance_lhs() == c2.load, name="bal", freeze=False)
    assert isinstance(con2, Constraint)
    assert_conequal(con1, con2, strict=False)
    assert np.array_equal(con1.labels.values, con2.labels.values)


def test_to_constraint_on_csr_lhs_is_unassigned_csr_constraint() -> None:
    require_v1()
    c1, c2 = twin_models()
    dense = c1.balance_lhs() == c1.load
    con = c2.balance_lhs() == c2.load
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
    con2 = c2.m.add_constraints(con, name="bal")
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
        c = base_model(sparse=sparse)
        gens = c.gen_p.indexes["gen"]
        gen_p = c.gen_p.where(xr.DataArray(gens != "gen7", coords=[gens]))
        lhs = (c.eff * gen_p).groupby(c.gbus).sum()
        return c.m.add_constraints(lhs == c.load, name="bal", freeze=freeze)

    dense, sparse = build(False), build(True)
    assert isinstance(sparse, CSRConstraint if freeze else Constraint)
    assert_conequal(dense, sparse, strict=False)
    np.testing.assert_array_equal(dense.labels.values, sparse.labels.values)


@pytest.mark.parametrize("sparse", [True, False], ids=["sparse", "dense"])
def test_frozen_invalid_infinite_rhs_raises(sparse: bool) -> None:
    require_v1()
    c = base_model(sparse=sparse)
    with pytest.raises(ValueError, match="incorrect infinite values"):
        c.m.add_constraints(c.balance_lhs() <= -np.inf, name="bal", freeze=True)


ROW_SCALINGS: dict[str, Callable[[xr.DataArray], Any]] = {
    "scalar": lambda load: 2.0,
    "snapshot": lambda load: xr.DataArray(
        np.arange(1.0, load.sizes["snapshot"] + 1), coords=[load.indexes["snapshot"]]
    ),
    "grid": lambda load: load,
}


@pytest.mark.parametrize("masked", [False, True], ids=["all", "masked"])
@pytest.mark.parametrize("scaling", list(ROW_SCALINGS))
@pytest.mark.parametrize("sparse", [True, False], ids=["sparse", "dense"])
def test_frozen_constraint_applies_row_scaling(
    sparse: bool, scaling: str, masked: bool
) -> None:
    require_v1()
    c = base_model(n_snap=6, sparse=sparse)
    row_scaling = ROW_SCALINGS[scaling](c.load)
    mask = c.load.bus == "bus1" if masked else None
    con = c.m.add_constraints(
        c.balance_lhs() == c.load,
        name="bal",
        freeze=True,
        scaling=row_scaling,
        mask=mask,
    )
    assert isinstance(con, CSRConstraint)
    active = con.labels != -1
    expected = xr.DataArray(row_scaling).broadcast_like(c.load).astype(float)
    xr.testing.assert_equal(
        con.scaling.where(active), expected.transpose(*con.scaling.dims).where(active)
    )


@pytest.mark.parametrize("grid", [False, True], ids=["scalar", "grid"])
@pytest.mark.parametrize("invalid", [-1.0, np.nan])
@pytest.mark.parametrize("sparse", [True, False], ids=["sparse", "dense"])
def test_frozen_constraint_rejects_invalid_row_scaling_on_masked_rows(
    sparse: bool, invalid: float, grid: bool
) -> None:
    require_v1()
    c = base_model(sparse=sparse)
    mask = c.load.bus == "bus1"
    scaling = c.load.where(mask, invalid) if grid else invalid
    with pytest.raises(ValueError, match="finite positive"):
        c.m.add_constraints(
            c.balance_lhs() == c.load, freeze=True, scaling=scaling, mask=mask
        )


@pytest.mark.parametrize("masked", [False, True], ids=["all", "masked"])
def test_freeze_does_not_copy_lhs_matrix(masked: bool) -> None:
    require_v1()
    c = base_model(gens_per_bus=(400,) * 50, n_snap=20, sparse=True)
    lhs = c.balance_lhs()
    assert lhs._csr is not None
    csr = lhs._csr.csr
    csr_bytes = csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes
    mask = c.load.bus == "bus1" if masked else None
    tracemalloc.start()
    try:
        c.m.add_constraints(lhs == c.load, name="bal", freeze=True, mask=mask)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < csr_bytes / 4


def sparse_model_results(c: Case) -> dict[str, Any]:
    weights = xr.DataArray(
        np.eye(len(c.gbus))[:, :2],
        coords=[c.gbus.index, pd.Index(["a", "b"], name="k")],
    )
    lhs = c.balance_lhs()
    return {
        "expression groupby": (1.0 * c.gen_p).groupby(c.gbus).sum(),
        "variable groupby": c.gen_p.groupby(c.gbus).sum(),
        "expression @": (1.0 * c.gen_p) @ weights,
        "variable @": c.gen_p @ weights,
        "constraint": c.m.add_constraints(lhs == c.load, name="bal"),
        "unfrozen": c.m.add_constraints(lhs >= 0, name="free", freeze=False),
        "copy": c.m.copy().constraints["bal"],
    }


@pytest.mark.parametrize("sparse", [False, True])
def test_model_sparse_switches_the_whole_build(sparse: bool) -> None:
    require_v1()
    res = sparse_model_results(base_model(sparse=sparse))
    unfrozen = res.pop("unfrozen")
    assert isinstance(unfrozen, Constraint)
    for key, obj in res.items():
        if isinstance(obj, LinearExpression):
            assert obj.is_sparse is sparse, key
        else:
            assert isinstance(obj, CSRConstraint) is sparse, key


def test_model_sparse_persists_through_netcdf(tmp_path: Path) -> None:
    require_v1()
    c = base_model(sparse=True)
    c.m.to_netcdf(tmp_path / "m.nc")
    read = linopy.read_netcdf(tmp_path / "m.nc")
    assert read.sparse and read.freeze_constraints
    assert read.copy().sparse
    linopy.options["semantics"] = "legacy"
    with pytest.raises(ValueError, match="requires v1 semantics"):
        linopy.read_netcdf(tmp_path / "m.nc")


def test_file_without_sparse_key_keeps_freeze_default(tmp_path: Path) -> None:
    with pytest.warns(FutureWarning, match="deprecated"):
        m = Model(freeze_constraints=True)
    m.to_netcdf(tmp_path / "m.nc")
    ds = xr.load_dataset(tmp_path / "m.nc")
    del ds.attrs["sparse"]
    ds.to_netcdf(tmp_path / "old.nc")
    read = linopy.read_netcdf(tmp_path / "old.nc")
    assert read.freeze_constraints and not read.sparse


@pytest.mark.parametrize(
    "op",
    [
        lambda c: (1.0 * c.gen_p).groupby(c.gbus).sum(),
        lambda c: (1.0 * c.gen_p) @ xr.DataArray(np.ones(len(c.gbus)), [c.gbus.index]),
    ],
    ids=["groupby", "matmul"],
)
def test_sparse_model_raises_under_legacy(op: Callable[[Case], Any]) -> None:
    require_v1()
    c = base_model(sparse=True)
    linopy.options["semantics"] = "legacy"
    with pytest.raises(ValueError, match="requires v1 semantics"):
        op(c)


@pytest.mark.parametrize(
    "build",
    [
        lambda: Model(sparse=True, chunk=10),
        lambda: Model(sparse=True, freeze_constraints=True),
        lambda: setattr(Model(sparse=True), "chunk", 10),
        lambda: setattr(Model(sparse=True), "freeze_constraints", False),
    ],
    ids=["chunk", "freeze_constraints", "set-chunk", "set-freeze_constraints"],
)
def test_sparse_model_rejects_conflicting_config(build: Callable[[], Any]) -> None:
    require_v1()
    with pytest.raises(ValueError, match="sparse"):
        build()


@pytest.mark.legacy
def test_model_sparse_requires_v1() -> None:
    with pytest.raises(ValueError, match="requires v1 semantics"):
        Model(sparse=True)


@pytest.mark.parametrize(
    "switch",
    [
        lambda c: Model(freeze_constraints=True),
        lambda c: setattr(c.m, "freeze_constraints", True),
        lambda c: (1.0 * c.gen_p).groupby(c.gbus).sum(sparse=False),
        lambda c: linopy.options.set_value(sparse_groupby=True),
    ],
    ids=[
        "Model(freeze_constraints)",
        "set-freeze_constraints",
        "sum(sparse)",
        "option",
    ],
)
def test_deprecated_sparse_switches_warn(switch: Callable[[Case], Any]) -> None:
    c = base_model()
    with linopy.options, pytest.warns(FutureWarning, match="deprecated"):
        switch(c)


def test_sparse_groupby_option_covers_groupby_only() -> None:
    require_v1()
    c = base_model()
    weights = xr.DataArray(np.ones(len(c.gbus)), [c.gbus.index])
    with linopy.options, pytest.warns(FutureWarning, match="deprecated"):
        linopy.options.set_value(sparse_groupby=True)
        assert (1.0 * c.gen_p).groupby(c.gbus).sum().is_sparse
        assert not ((1.0 * c.gen_p) @ weights).is_sparse


def test_sparse_model_notices_groupby_without_sparse_path() -> None:
    require_v1()
    c = base_model(sparse=True)
    with no_densify(), pytest.raises(linopy.PerformanceWarning, match="grouper"):
        (1.0 * c.gen_p).groupby(c.gbus).sum(use_fallback=True)


def test_materialized_csr_still_freezes_via_dense_path() -> None:
    require_v1()
    c = base_model(sparse=True)
    lhs = c.balance_lhs()
    _ = lhs.nterm
    con = c.m.add_constraints(lhs == c.load, name="bal")
    assert isinstance(con, CSRConstraint)


@pytest.mark.parametrize("sparse", [True, False], ids=["sparse", "dense"])
def test_nan_rhs_raises(sparse: bool) -> None:
    require_v1()
    c = base_model(sparse=sparse)
    load = c.load.copy()
    load[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        c.m.add_constraints(c.balance_lhs() == load, name="bal", freeze=True)


@pytest.mark.parametrize("sparse", [True, False], ids=["sparse", "dense"])
def test_reordered_rhs_raises(sparse: bool) -> None:
    require_v1()
    c = base_model(sparse=sparse)
    load = c.load.isel(bus=slice(None, None, -1))
    with pytest.raises(ValueError, match="[Cc]oordinate"):
        c.m.add_constraints(c.balance_lhs() == load, name="bal", freeze=True)


def test_nan_grouper_raises_eagerly() -> None:
    require_v1()
    c = base_model(sparse=True)
    gbus = c.gbus.copy()
    gbus.iloc[0] = np.nan
    with pytest.raises(ValueError, match="NaN values"):
        (1.0 * c.gen_p).groupby(gbus).sum()


TERM_LINE = re.compile(r"^[+-][0-9.e+-]+ x[0-9]+$")


def canon_lp(text: str) -> list[str]:
    """LP file lines with each block of term lines sorted."""
    out: list[str] = []
    buf: list[str] = []
    for line in text.splitlines():
        if TERM_LINE.match(line):
            buf.append(line)
        else:
            out += sorted(buf) + [line]
            buf = []
    return out + sorted(buf)


def test_lp_files_identical(tmp_path: Path) -> None:
    require_v1()
    sizes = (7, 1, 3, 1, 2, 1, 1, 4, 1, 2, 1, 1)
    c1, c2 = twin_models(gens_per_bus=sizes)
    c1.m.add_constraints(c1.balance_lhs() == c1.load, name="bal")
    c1.m.add_objective((1.0 * c1.gen_p).sum())
    c2.m.add_constraints(c2.balance_lhs() == c2.load, name="bal")
    c2.m.add_objective((1.0 * c2.gen_p).sum())

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
    c1, c2 = twin_models()
    sparse = c2.gen_sum().reindex(indexers)
    assert sparse._csr is not None
    dense = c1.gen_sum().reindex(indexers)
    assert_linequal(sparse, dense)


def test_reindex_falls_back_to_dense_for_unsupported_kwargs() -> None:
    require_v1()
    c1, c2 = twin_models()
    res = c2.gen_sum().reindex(bus=["bus3", "bus0"], copy=False)
    assert res._csr is None
    assert_linequal(res, c1.gen_sum().reindex(bus=["bus3", "bus0"]))


def test_reindex_merge_chain_freezes_csr() -> None:
    require_v1()
    c1, c2 = twin_models()
    con1 = c1.m.add_constraints(reindexed_balance(c1) == c1.load, name="bal")
    tot = reindexed_balance(c2)
    assert tot._csr is not None
    con2 = c2.m.add_constraints(tot == c2.load, name="bal")
    assert isinstance(con2, CSRConstraint)
    assert_frozen_equal(con1, con2)


def test_reindex_merge_chain_peak_memory() -> None:
    require_v1()
    sizes = (200,) + (1,) * 299
    n_snap = 50
    dense_rectangle_bytes = len(sizes) * n_snap * max(sizes) * 16
    c = base_model(gens_per_bus=sizes, n_snap=n_snap, sparse=True)
    tracemalloc.start()
    try:
        c.m.add_constraints(reindexed_balance(c) == c.load, name="bal")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < dense_rectangle_bytes / 4


@pytest.mark.parametrize("value", [0.0, 3.5])
def test_fillna_stays_csr_and_matches_dense(value: float) -> None:
    require_v1()
    c1, c2 = twin_models()
    wide = {"bus": [f"bus{i}" for i in range(7)]}
    filled = c2.gen_sum().reindex(wide).fillna(value)
    assert filled._csr is not None
    dense = c1.gen_sum().reindex(wide)
    assert_linequal(filled, dense.fillna(value))


def test_fillna_with_array_falls_back_to_dense() -> None:
    require_v1()
    c1, c2 = twin_models()
    fill = xr.zeros_like(c1.load)
    res = c2.gen_sum().fillna(fill)
    assert res._csr is None
    assert_linequal(res, c1.gen_sum().fillna(fill))


def test_rename_stays_csr_and_matches_dense() -> None:
    require_v1()
    c1, c2 = twin_models()
    sparse = c2.gen_sum().rename(bus="node")
    assert sparse._csr is not None
    dense = c1.gen_sum().rename(bus="node")
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
    _, dense_expr, _ = keyed_model()
    _, sparse_expr, _ = keyed_model(sparse=True)
    keys = ["period", "season"]
    sparse = op(sparse_expr.groupby(keys).sum(observed=True))
    dense = op(dense_expr.groupby(keys).sum(observed=True))
    assert sparse._csr is not None
    for name in keys:
        xr.testing.assert_equal(sparse.coords[name], dense.coords[name])
    assert_linequal(sparse, dense)


def cross_grid_parts(
    c: Case, lines: tuple[str, ...] = ("line1", "line2")
) -> list[LinearExpression]:
    """Generation on all buses, flow on a line subset only, snapshot-major on the flow side."""
    flow_t = 1.0 * c.flow_t.loc[:, list(lines)]
    return [c.gen_sum(), flow_t.groupby(c.bus0.loc[list(lines)]).sum()]


def twin_cross_grid_parts() -> tuple[list[LinearExpression], list[LinearExpression]]:
    """``cross_grid_parts`` of a sparse model and of its dense twin."""
    c1, c2 = twin_models()
    return cross_grid_parts(c2), cross_grid_parts(c1)


def densified(e: LinearExpression) -> LinearExpression:
    """A dense operand in the model of the CSR-backed ``e``."""
    assert e._csr is not None
    return e._csr.to_dense()


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
    sparse, dense = twin_cross_grid_parts()
    if order == "flow-gen":
        sparse, dense = sparse[::-1], dense[::-1]
    res = linopy.merge(sparse, join=join, cls=LinearExpression)
    assert res._csr is not None
    assert res.coord_dims == dense[0].coord_dims
    assert_terms_equal(res, linopy.merge(dense, join=join, cls=LinearExpression))


def test_transposed_grid_exact_merge_stays_csr_and_matches_dense() -> None:
    """Same labels in a transposed dim order stay sparse under the default join."""
    require_v1()
    c1, c2 = twin_models()
    a = (1.0 * c2.flow).groupby(c2.bus0).sum()
    b = (1.0 * c2.flow_t).groupby(c2.bus0).sum()
    assert a.coord_dims == b.coord_dims[::-1] != b.coord_dims
    res = linopy.merge([a, b], cls=LinearExpression)
    assert res._csr is not None
    dense = [
        (1.0 * c1.flow).groupby(c1.bus0).sum(),
        (1.0 * c1.flow_t).groupby(c1.bus0).sum(),
    ]
    assert_terms_equal(res, linopy.merge(dense, cls=LinearExpression))


@pytest.mark.parametrize("join", ["outer", "inner", "left", "right"])
def test_three_operand_cross_grid_merge_matches_dense(join: JoinOptions) -> None:
    require_v1()
    c1, c2 = twin_models()
    third_lines = ("line3", "line4")
    sparse = cross_grid_parts(c2) + cross_grid_parts(c2, third_lines)[1:]
    dense = cross_grid_parts(c1) + cross_grid_parts(c1, third_lines)[1:]
    res = linopy.merge(sparse, join=join, cls=LinearExpression)
    assert res._csr is not None
    assert_terms_equal(res, linopy.merge(dense, join=join, cls=LinearExpression))


def test_cross_grid_merge_absent_fill_matches_dense() -> None:
    require_v1()
    sparse, dense = twin_cross_grid_parts()
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
    assert res.const.isnull().sum() == 3 * res.sizes["snapshot"]


def test_cross_grid_merge_keeps_absent_cell_absent() -> None:
    require_v1()
    sparse, dense = twin_cross_grid_parts()
    mask = xr.DataArray([True, False], coords=[dense[1].indexes["bus"]])
    dense[1] = dense[1].where(mask)
    flow = densified(sparse[1]).where(mask)
    res = linopy.merge([sparse[0], flow], join="outer", cls=LinearExpression)
    assert res._csr is not None
    assert_terms_equal(res, linopy.merge(dense, join="outer", cls=LinearExpression))


def test_cross_grid_merge_mixed_dense_operand_stays_csr() -> None:
    require_v1()
    sparse, dense = twin_cross_grid_parts()
    res = sparse[0].add(densified(sparse[1]), join="outer")
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
    sparse, dense = twin_cross_grid_parts()
    with pytest.raises(error):
        linopy.merge(dense, **kwargs)
    with pytest.raises(error):
        linopy.merge(sparse, **kwargs)


@pytest.mark.parametrize("join", ["outer", "inner", "left", "right"])
def test_cross_grid_merge_with_aux_coord_operand_stays_csr(join: JoinOptions) -> None:
    """The aux coord follows its rows onto the joined grid, like dense."""
    require_v1()
    sparse, dense = twin_cross_grid_parts()
    tag = xr.DataArray(["x", "y"], coords=[dense[1].indexes["bus"]])
    tagged = densified(sparse[1]).assign_coords(tag=tag)
    with no_densify():
        res = linopy.merge([sparse[0], tagged], join=join, cls=LinearExpression)
    assert res.is_sparse
    labels = res.indexes["bus"]
    want = tag.to_series().reindex(labels).to_numpy()
    assert pd.Series(res.coords["tag"].values).equals(pd.Series(want))
    want_expr = linopy.merge(
        [dense[0], dense[1].assign_coords(tag=tag)], join=join, cls=LinearExpression
    )
    assert_sparse_matches(res, want_expr)


def test_cross_grid_merge_aux_coord_conflict_raises_like_dense() -> None:
    require_v1()
    sparse, dense = sparse_and_dense("aux")
    parts = [sparse, sparse.isel(group=[2, 1])]
    with pytest.raises(ValueError) as want:
        linopy.merge([dense, dense.isel(group=[2, 1])], join="outer")
    with pytest.raises(ValueError) as got:
        linopy.merge(parts, join="outer")
    assert str(got.value) == str(want.value)


def test_cross_grid_merge_with_duplicate_labels_raises_like_dense() -> None:
    require_v1()
    dup = pd.Index(["bus1", "bus1", "bus2"], name="bus")
    for c in twin_models():
        shed = 1.0 * c.m.add_variables(
            coords=[dup, c.load.indexes["snapshot"]], name="shed"
        )
        with pytest.raises(ValueError, match="cannot reindex or align"):
            linopy.merge([c.gen_sum(), shed], join="left")


def test_override_merge_same_shape_stays_csr() -> None:
    require_v1()
    dense, sparse = (
        [c.gen_sum(), (1.0 * c.flow).groupby(c.bus1.str.upper()).sum()]
        for c in twin_models()
    )
    res = linopy.merge(sparse, join="override", cls=LinearExpression)
    assert res._csr is not None
    assert_terms_equal(res, linopy.merge(dense, join="override", cls=LinearExpression))


def test_cross_grid_balance_freezes_csr() -> None:
    require_v1()
    c1, c2 = twin_models()
    lhs1 = linopy.merge(cross_grid_parts(c1), join="outer")
    con1 = c1.m.add_constraints(lhs1 == c1.load, name="bal")
    lhs2 = linopy.merge(cross_grid_parts(c2), join="outer", cls=LinearExpression)
    assert lhs2._csr is not None
    con2 = c2.m.add_constraints(lhs2 == c2.load, name="bal")
    assert isinstance(con2, CSRConstraint)
    assert_frozen_equal(con1, con2)


def test_cross_grid_merge_peak_memory() -> None:
    require_v1()
    sizes = (200,) + (1,) * 299
    n_snap = 50
    dense_rectangle_bytes = len(sizes) * n_snap * max(sizes) * 16
    c = base_model(gens_per_bus=sizes, n_snap=n_snap, sparse=True)
    tracemalloc.start()
    try:
        lhs = linopy.merge(cross_grid_parts(c), join="outer")
        c.m.add_constraints(lhs == c.load, name="bal")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < dense_rectangle_bytes / 4


def cell_matrix(
    expr: LinearExpression, dims: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell coefficient row over raw variable labels, plus the flat constant."""
    ds = expr.data
    shape = (int(np.prod([ds.sizes[d] for d in dims], dtype=int)), ds.sizes[TERM_DIM])
    vars_ = ds.vars.transpose(*dims, TERM_DIM).to_numpy().reshape(shape)
    coeffs = ds.coeffs.transpose(*dims, TERM_DIM).to_numpy().reshape(shape)
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


BIG = 2**31


@pytest.mark.parametrize(
    "n_col, right_rows, expected",
    [
        (
            1000,
            [{900: 1.0, 3: 2.0, 500: 3.0}, {3: 4.0, 999: 5.0}],
            [
                {3: 10.0, 500: 3.0, 900: 1.0, 999: 10.0},
                {3: 12.0, 999: 15.0},
                {3: 8.0, 500: 12.0, 900: 4.0},
            ],
        ),
        (1000, [{}, {}], [{}, {}, {}]),
        (
            BIG + 10,
            [{BIG + 5: 1.0, 7: 2.0}, {BIG: 3.0}],
            [{7: 2.0, BIG: 6.0, BIG + 5: 1.0}, {BIG: 9.0}, {7: 8.0, BIG + 5: 4.0}],
        ),
    ],
    ids=["wide", "empty", "beyond-int32"],
)
def test_column_compacted_matmul(
    n_col: int, right_rows: list[dict[int, float]], expected: list[dict[int, float]]
) -> None:
    right = scipy.sparse.csr_array(
        (
            [v for row in right_rows for v in row.values()],
            np.array([c for row in right_rows for c in row], dtype=np.int64),
            np.cumsum([0, *map(len, right_rows)]),
        ),
        shape=(2, n_col),
    )
    left = scipy.sparse.csr_array([[1.0, 2.0], [0.0, 3.0], [4.0, 0.0]])

    res = column_compacted_matmul(left, right)

    assert res.shape == (3, n_col)
    rows = [
        dict(zip(res.indices[a:b].tolist(), res.data[a:b].tolist()))
        for a, b in zip(res.indptr[:-1], res.indptr[1:])
    ]
    assert [list(r.items()) for r in rows] == [sorted(r.items()) for r in expected]


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
    return (1.0 * c.gen_p).groupby(grouper).sum(observed=True)


def test_contracted_keeps_aux_coords_on_kept_dims_only() -> None:
    require_v1()
    csr = tagged_group(base_model(sparse=True))._csr
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
    csr = tagged_group(base_model(sparse=True))._csr
    assert csr is not None
    assert {n: d for n, (d, _) in op(csr.grid).aux.items()} == aux_dims


def test_grid_conformed_reindexes_aux_coords_and_equality_sees_them() -> None:
    require_v1()
    csr = tagged_group(base_model(sparse=True))._csr
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
    grouped = tagged_group(base_model(sparse=True))
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
    c1, c2 = twin_models()
    operand = loc_operand(c2.gen_p.indexes["snapshot"])
    gen = c2.gen_sum()
    flow = (1.0 * c2.flow).groupby(c2.bus0).sum()

    res = gen @ operand
    assert res._csr is not None
    assert_linequal(res, gen.dot(operand))
    assert_cells_equal(res, c1.gen_sum() @ operand, ("bus", "loc"))

    total = res + (flow @ operand)
    assert total._csr is not None
    con = c2.m.add_constraints(total <= 0, name="matmul")
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
    grouped = (1.0 * c.gen_p).groupby(c.gbus).sum()
    ones = xr.DataArray(np.ones(grouped.shape[:2]), coords=grouped.coords)
    return grouped @ ones


SPARSE_BUILDS: dict[str, Callable[[Case], LinearExpression]] = {
    "grouped": Case.gen_sum,
    "aux": tagged_group,
    "absent": lambda c: (
        keyed_model(sparse=c.m.sparse)[1].groupby(["period", "season"]).sum()
    ),
    "zero-dim": full_contraction,
}


def sparse_and_dense(
    build: str, c: Case | None = None
) -> tuple[LinearExpression, LinearExpression]:
    """A sparse build and its dense conversion, the sparse one left sparse."""
    sparse = SPARSE_BUILDS[build](base_model(sparse=True) if c is None else c)
    csr = sparse._csr
    assert sparse.is_sparse and csr is not None
    return sparse, csr.to_dense()


def dense_build(build: str) -> LinearExpression:
    """The build in a dense model, on the dense path."""
    dense = SPARSE_BUILDS[build](base_model())
    assert not dense.is_sparse
    return dense


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
    sparse, dense = sparse_and_dense(build)
    got, want = METADATA[attr](sparse), METADATA[attr](dense)
    assert sparse.is_sparse
    if isinstance(got, xr.Dataset | xr.DataArray):
        xr.testing.assert_identical(got, want)
    else:
        assert got == want


def test_is_sparse_tracks_backing_and_repr_marks_it() -> None:
    require_v1()
    c = base_model(sparse=True)
    sparse = SPARSE_BUILDS["grouped"](c)
    assert sparse.is_sparse
    assert repr(sparse).startswith("LinearExpression (sparse) [")
    sparse.data
    assert not sparse.is_sparse
    assert repr(sparse).startswith("LinearExpression [")
    assert not (1.0 * c.gen_p).is_sparse


def add_chunked(e: LinearExpression, c: Case) -> Any:
    chunked = base_model()
    chunked.m.chunk = {"bus": 2}
    with pytest.warns(FutureWarning, match="deprecated"):
        lhs = (chunked.eff * chunked.gen_p).groupby(chunked.gbus).sum(sparse=True)
    return chunked.m.add_constraints(lhs >= 1, freeze=True)


DENSIFY_OPS: dict[str, tuple[Callable[[LinearExpression, Case], Any], str]] = {
    "data": (lambda e, c: e.data, "`.data` read"),
    "merge": (lambda e, c: e + 1.0 * c.flow, "over different dimensions"),
    "matmul": (lambda e, c: e @ loc_operand(c.gbus.index), "sharing no dimension"),
    "rhs": (
        lambda e, c: e <= xr.DataArray([1.0, 2.0], coords=[LOC]),
        "rhs over dimensions outside the grid",
    ),
    "rhs-expr": (lambda e, c: e <= 1.0 * c.gen_p.sum("gen"), "over different dim"),
    "mutable": (lambda e, c: (e == c.load).mutable(), "`mutable\\(\\)`"),
    "add-unfrozen": (
        lambda e, c: c.m.add_constraints(e >= 1, freeze=False),
        "constraint added unfrozen, `freeze=False`",
    ),
    "add-chunked": (add_chunked, "chunked model"),
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
    c = base_model(sparse=True)
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
    assert len(notices) == 1
    assert re.search(reason, str(notices[0].message))
    assert notices[0].filename == __file__


@pytest.mark.parametrize("scale", [1.0, 0.0], ids=["plain", "zeros"])
@pytest.mark.parametrize("build", list(SPARSE_BUILDS))
def test_flat_and_to_polars_served_without_densifying(build: str, scale: float) -> None:
    require_v1()
    sparse, _ = sparse_and_dense(build)
    sparse = scale * sparse
    assert sparse._csr is not None
    dense = sparse._csr.to_dense()
    with no_densify():
        got_pl, got_flat = sparse.to_polars(), sparse.flat
    assert sparse.is_sparse
    assert got_pl.sort("vars").equals(dense.to_polars().sort("vars"))
    pd.testing.assert_frame_equal(got_flat, dense.flat)


@pytest.mark.parametrize("build", list(SPARSE_BUILDS))
def test_sparse_store_indices_follow_model_label_dtype(build: str) -> None:
    require_v1()
    sparse, _ = sparse_and_dense(build)
    m = sparse.model
    con = m.add_constraints(sparse >= 1, name="con")
    A = m.matrices.A
    assert isinstance(con, CSRConstraint) and sparse._csr is not None
    assert A is not None
    dtypes = {sparse._csr.csr.indices.dtype, con._csr.indices.dtype}
    assert dtypes == {np.dtype(m.dtypes["labels"]), A.indices.dtype}


@pytest.mark.parametrize("label_dtype", [np.int32, np.int64])
def test_sparse_group_sum_indices_widen_with_model(label_dtype: type) -> None:
    require_v1()
    m = Model(dtypes={"labels": label_dtype}, sparse=True)
    x = m.add_variables(coords=[pd.RangeIndex(4, name="i")], name="x")
    group = xr.DataArray([0, 0, 1, 1], coords=[x.indexes["i"]], name="g")
    expr = x.groupby(group).sum()
    assert expr._csr is not None
    assert expr._csr.csr.indices.dtype == expr._csr.csr.indptr.dtype == label_dtype


def objective_twins(build: str, scale: float) -> tuple[Model, Model]:
    """Twin models, the first with the sparse build as objective, the second dense."""
    sparse, _ = sparse_and_dense(build)
    _, dense = sparse_and_dense(build)
    with no_densify():
        sparse.model.add_objective(scale * (sparse - sparse.const.fillna(0)))
    dense.model.add_objective(scale * (dense - dense.const.fillna(0)))
    return sparse.model, dense.model


def test_sparse_objective_rejects_constant_without_densifying() -> None:
    require_v1()
    sparse, _ = sparse_and_dense("absent")
    with no_densify(), pytest.raises(ValueError, match="Constant values"):
        sparse.model.add_objective(sparse)


@pytest.mark.parametrize("scale", [1.0, 0.0], ids=["plain", "zeros"])
@pytest.mark.parametrize("build", list(SPARSE_BUILDS))
def test_objective_stays_csr_and_exports_like_dense(
    build: str, scale: float, tmp_path: Path
) -> None:
    require_v1()
    ms, md = objective_twins(build, scale)
    with no_densify():
        c = ms.matrices.c
        terms = ms.objective.linear_terms()
        ms.to_file(tmp_path / "sparse.lp")
        ms.to_netcdf(tmp_path / "sparse.nc")
        copied = ms.copy()
        assert ms.objective.attrs == {"name": "objective"}
        repr(ms.objective)
    assert ms.objective.expression.is_sparse and copied.objective.expression.is_sparse
    md.to_file(tmp_path / "dense.lp")
    md.to_netcdf(tmp_path / "dense.nc")
    assert np.array_equal(c, md.matrices.c)
    dense_terms = md.objective.linear_terms()
    assert sorted(zip(*map(list, terms))) == sorted(zip(*map(list, dense_terms)))
    assert (dense_terms[1] != 0).all()
    lp_sparse, lp_dense = (tmp_path / f"{k}.lp" for k in ("sparse", "dense"))
    assert canon_lp(lp_sparse.read_text()) == canon_lp(lp_dense.read_text())
    rs, rd = (linopy.read_netcdf(tmp_path / f"{k}.nc") for k in ("sparse", "dense"))
    es, ed = rs.objective.expression, rd.objective.expression
    assert isinstance(es, LinearExpression) and isinstance(ed, LinearExpression)
    assert_cells_equal(es, ed, ())
    assert rs.objective.expression.attrs["name"] == "objective"


@pytest.mark.skipif("highs" not in linopy.available_solvers, reason="needs highs")
@pytest.mark.parametrize("io_api", ["lp", "direct"])
@pytest.mark.parametrize("build", list(SPARSE_BUILDS))
def test_sparse_objective_solves_like_dense(build: str, io_api: str) -> None:
    require_v1()
    values = []
    for m in objective_twins(build, 1.0):
        for var in m.variables.data.values():
            var.update(lower=0, upper=1)
        m.objective.sense = "max"
        with no_densify():
            m.solve("highs", io_api=io_api)
        values.append(m.objective.value)
    assert values[0] == pytest.approx(values[1])


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

ELEMENTWISE_OPS: dict[
    str, Callable[[LinearExpression, Any], LinearExpression | QuadraticExpression]
] = {
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
    sparse, dense = sparse_and_dense(build)
    x = OPERANDS[operand](dense)
    func = ELEMENTWISE_OPS[op]
    res = func(sparse, x)
    assert res.is_sparse
    assert_linequal(res, func(dense, x))


DENSE_ADDENDS: dict[str, Callable[[Case, LinearExpression], LinearExpression]] = {
    "same-grid": lambda c, dense: 2 * dense,
    "scalar-var": lambda c, dense: 2 * c.gen_p.isel(gen=0, snapshot=0),
}


@pytest.mark.parametrize(
    ("build", "addend"),
    [(b, "same-grid") for b in SPARSE_BUILDS] + [("zero-dim", "scalar-var")],
)
def test_merge_with_dense_expression_stays_csr_and_matches_dense(
    build: str, addend: str
) -> None:
    require_v1()
    c = base_model(sparse=True)
    sparse, dense = sparse_and_dense(build, c)
    other = DENSE_ADDENDS[addend](c, dense)
    with no_densify():
        res = sparse + other
    assert_sparse_matches(res, dense + other)


@pytest.mark.parametrize("join", ["inner", "outer", "left", "right"])
@pytest.mark.parametrize("fill_value", [None, linopy.ABSENT], ids=["fill", "absent"])
@pytest.mark.parametrize("method", ["add", "sub", "mul", "div"])
def test_elementwise_join_on_mismatched_labels_stays_csr_and_matches_dense(
    method: str, fill_value: Any, join: JoinOptions
) -> None:
    require_v1()
    sparse, dense = sparse_and_dense("grouped")
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
    sparse, dense = sparse_and_dense("aux")
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
    sparse, dense = sparse_and_dense("grouped")
    x = xr.DataArray([1.0, 2.0], coords=[LOC])
    func = ELEMENTWISE_OPS[op]
    res = func(sparse, x)
    assert not res.is_sparse
    assert_linequal(res, func(dense, x))


@pytest.mark.parametrize("join", ["outer", "inner", "left", "right"])
@pytest.mark.parametrize("op", ["add", "sub", "mul", "div"])
def test_join_with_constant_keeps_aux_coords_like_dense(
    op: str, join: JoinOptions
) -> None:
    require_v1()
    sparse, dense = sparse_and_dense("aux")
    n = sparse.sizes["group"]
    x = xr.DataArray(
        np.arange(1.0, n + 2), coords=[pd.RangeIndex(1, n + 2, name="group")]
    )
    with no_densify():
        res = getattr(sparse, op)(x, join=join)
    assert_sparse_matches(res, getattr(dense, op)(x, join=join))


@pytest.mark.parametrize("join", ["outer", "inner", "left", "right"])
def test_to_constraint_join_with_constant_matches_dense(join: JoinOptions) -> None:
    require_v1()
    cons = []
    for c in twin_models():
        lhs = c.gen_sum()
        with no_densify():
            con = lhs.to_constraint("<=", c.load.isel(bus=[0, 2]), join=join)
            cons.append(c.m.add_constraints(con, name="c"))
    dense, frozen = cons
    assert isinstance(frozen, CSRConstraint)
    xr.testing.assert_identical(
        xr.Dataset(coords=frozen.coords), xr.Dataset(coords=dense.coords)
    )
    assert_frozen_equal(dense, frozen)


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
    sparse, dense = sparse_and_dense(build)
    dim = SUM_DIMS[dims](tuple(map(str, sparse.coord_dims)))
    with no_densify():
        res = sparse.sum(dim)
    assert_sparse_matches(res, dense.sum(dim))


@pytest.mark.parametrize("drop_zeros", [False, True])
def test_sum_keeps_explicit_zeros_unless_dropped(drop_zeros: bool) -> None:
    require_v1()
    c = base_model(sparse=True)
    sparse = (0.0 * c.gen_p).groupby(c.gbus).sum()
    assert sparse._csr is not None
    dense = sparse._csr.to_dense()
    res = sparse.sum("snapshot", drop_zeros=drop_zeros)
    assert res._csr is not None
    assert (res._csr.csr.nnz == 0) == drop_zeros
    assert_sparse_matches(res, dense.sum("snapshot", drop_zeros=drop_zeros))


def test_sum_unknown_dim_raises_like_dense() -> None:
    require_v1()
    sparse, dense = sparse_and_dense("grouped")
    with pytest.raises(KeyError) as want:
        dense.sum("nodim")
    with pytest.raises(KeyError) as got:
        sparse.sum("nodim")
    assert str(got.value) == str(want.value)


CHAINS: dict[str, Callable[[LinearExpression], LinearExpression]] = {
    "group-group": lambda e: e.groupby(halves(e)).sum(),
    "group-sum": lambda e: e.groupby(halves(e)).sum().sum("snapshot"),
    "sum-group": lambda e: e.sum("snapshot").groupby(halves(e)).sum(),
    "empty-sum": lambda e: e.isel({dim0(e): []}).sum(dim0(e)),
    "empty-group": lambda e: (
        (empty := e.isel({dim0(e): []})).groupby(halves(empty)).sum()
    ),
}


@pytest.mark.parametrize("chain", list(CHAINS))
@pytest.mark.parametrize("build", ["grouped", "aux", "absent"])
def test_chained_groupby_stays_csr_and_matches_dense(build: str, chain: str) -> None:
    require_v1()
    sparse, _ = sparse_and_dense(build)
    func = CHAINS[chain]
    with no_densify():
        res = func(sparse)
    assert_sparse_matches(res, func(dense_build(build)))


@pytest.mark.parametrize("observed", [False, True])
def test_chained_namelist_groupby_on_aux_coords_matches_dense(observed: bool) -> None:
    require_v1()
    sparse, _ = sparse_and_dense("aux")
    dense = dense_build("aux")
    res = sparse.groupby(["bus", "tag"]).sum(observed=observed)
    assert_sparse_matches(res, dense.groupby(["bus", "tag"]).sum(observed=observed))


def test_chained_groupby_sparse_false_densifies() -> None:
    require_v1()
    sparse, _ = sparse_and_dense("grouped")
    dense = dense_build("grouped")
    with pytest.warns(FutureWarning, match="deprecated"):
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
    sparse, dense = sparse_and_dense(build)
    func = SELECTIONS[select]
    with no_densify():
        res = func(sparse)
    assert_sparse_matches(res, func(dense))


ABSENT_KEEPING_OPS: dict[str, Callable[[LinearExpression], LinearExpression]] = {
    "elementwise": lambda e: e * grid_operand(e) + 1.0,
    "selection": lambda e: e.where(alternating(e), 3.0).isel(season=[1, 0, 1]),
}


@pytest.mark.parametrize("op", list(ABSENT_KEEPING_OPS))
def test_absent_cells_stay_termless(op: str) -> None:
    require_v1()
    sparse = SPARSE_BUILDS["absent"](base_model(sparse=True))
    csr = ABSENT_KEEPING_OPS[op](sparse)._csr
    assert csr is not None
    absent = np.isnan(csr.const)
    assert absent.any()
    assert (np.diff(csr.csr.indptr)[absent] == 0).all()


def test_scalar_selection_coords_merge_like_dense() -> None:
    require_v1()
    c = base_model(sparse=True)
    sparse, dense = sparse_and_dense("grouped", c)
    flow = densified((1.0 * c.flow).groupby(c.bus0).sum())
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
    sparse, dense = sparse_and_dense("grouped")
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
    c = base_model(sparse=True)
    sparse, dense = sparse_and_dense("grouped", c)
    func = DENSIFY_OPS[op][0]
    res = func(sparse, c)
    assert not res.is_sparse
    assert_linequal(res, func(dense, c))


MASKS: dict[str, Callable[[LinearExpression], Any]] = {
    "dim": alternating,
    "grid": lambda e: grid_operand(e) > 1.5,
    "ndarray": lambda e: (grid_operand(e) > 1.5).to_numpy(),
}


@pytest.mark.parametrize("prebuilt", [False, True], ids=["lhs", "constraint"])
@pytest.mark.parametrize("mask", list(MASKS))
def test_add_constraints_mask_freezes_sparse_and_matches_dense(
    mask: str, prebuilt: bool
) -> None:
    require_v1()
    c1, c2 = twin_models()
    lhs = c2.balance_lhs()
    m = MASKS[mask](lhs)
    con1 = c1.m.add_constraints(c1.balance_lhs(), ">=", c1.load, "bal", mask=m)
    with no_densify():
        if prebuilt:
            con2 = c2.m.add_constraints(lhs >= c2.load, name="bal", mask=m)
        else:
            con2 = c2.m.add_constraints(lhs, ">=", c2.load, "bal", mask=m)
    assert isinstance(con2, CSRConstraint)
    assert_frozen_equal(con1, con2)


def observed_keys_group() -> LinearExpression:
    """The #941 reproducer: a multi-key observed grouping with aux coords on ``group``."""
    m = Model(sparse=True)
    x = m.add_variables(coords=[pd.RangeIndex(4, name="s")], name="x")
    expr = x.to_linexpr().assign_coords(
        period=("s", [1, 1, 2, 2]), region=("s", ["n", "s", "n", "s"])
    )
    return expr.groupby(["period", "region"]).sum(observed=True)


def dashed_group() -> LinearExpression:
    """An observed grouping whose dims and aux coords carry dashes, the netcdf name separator."""
    m = Model(sparse=True)
    s = pd.RangeIndex(4, name="my-s")
    x = m.add_variables(coords=[s], name="x")
    grouper = pd.DataFrame(
        {"my-bus": ["a", "a", "b", "b"], "tag": [1, 1, 2, 2]}, index=s
    )
    grouped = x.to_linexpr().groupby(grouper).sum(observed=True)
    return grouped.rename({"group": "my-group"})


AUX_BUILDS: dict[str, Callable[[], LinearExpression]] = {
    "observed-keys": observed_keys_group,
    "dashed": dashed_group,
    "aux": lambda: SPARSE_BUILDS["aux"](base_model(sparse=True)),
    "scalar": lambda: SPARSE_BUILDS["grouped"](base_model(sparse=True)).sel(snapshot=1),
}


@pytest.mark.parametrize("build", list(AUX_BUILDS))
def test_frozen_constraint_keeps_aux_coords_like_dense(
    build: str, tmp_path: Path
) -> None:
    require_v1()
    sparse = AUX_BUILDS[build]()
    m = sparse.model
    assert sparse._csr is not None
    ref = m.add_constraints(sparse._csr.to_dense() >= 1, name="dense", freeze=False)
    with no_densify():
        con = m.add_constraints(sparse >= 1, name="sparse")
    assert isinstance(con, CSRConstraint)
    want = xr.Dataset(coords=ref.coords)
    assert set(want.coords) > set(con.coord_names)
    for got in (con, con.mutable(), ref.freeze()):
        xr.testing.assert_identical(xr.Dataset(coords=got.coords), want)
    m.to_netcdf(tmp_path / "m.nc")
    read = linopy.read_netcdf(tmp_path / "m.nc").constraints["sparse"]
    assert isinstance(read, CSRConstraint)
    xr.testing.assert_identical(xr.Dataset(coords=read.coords), want)


def setter(attr: str) -> Callable[[CSRConstraint], None]:
    return lambda con: setattr(con, attr, 1.0)


FROZEN_MUTATIONS: dict[str, Callable[[CSRConstraint], Any]] = {
    "loc": lambda con: con.loc[{"bus": "bus0"}],
    "update": lambda con: con.update(rhs=2.0),
    **{
        attr: setter(attr)
        for attr in ["coeffs", "vars", "sign", "rhs", "lhs", "scaling"]
    },
}


@pytest.mark.parametrize("op", list(FROZEN_MUTATIONS))
def test_frozen_constraint_mutation_names_mutable(op: str) -> None:
    require_v1()
    c = base_model(sparse=True)
    con = c.m.add_constraints(c.balance_lhs() >= c.load)
    assert isinstance(con, CSRConstraint)
    with pytest.raises(AttributeError, match=rf"CSRConstraint\.{op} .*\.mutable\(\)"):
        FROZEN_MUTATIONS[op](con)


FROZEN_UNSUPPORTED: dict[str, tuple[Callable[[CSRConstraint], Any], str]] = {
    "from_rule": (
        lambda con: CSRConstraint.from_rule(con.model, lambda m, i: None, [[0]]),
        r"Constraint\.from_rule .*\.freeze\(\)",
    ),
}


@pytest.mark.parametrize("op", list(FROZEN_UNSUPPORTED))
def test_frozen_unsupported_names_the_working_route(op: str) -> None:
    require_v1()
    c = base_model(sparse=True)
    con = c.m.add_constraints(c.balance_lhs() >= c.load)
    assert isinstance(con, CSRConstraint)
    call, remedy = FROZEN_UNSUPPORTED[op]
    with pytest.raises(AttributeError, match=rf"CSRConstraint\.{op} .*{remedy}"):
        call(con)


EXPR_RHS: dict[str, Callable[[Case], LinearExpression]] = {
    "expr": lambda c: (1.0 * c.flow).groupby(c.bus1).sum(),
    "expr-const": lambda c: (1.0 * c.flow).groupby(c.bus1).sum() + c.load,
    "dense-expr-const": lambda c: (
        (1.0 * c.flow).groupby(c.bus1).sum(use_fallback=True) - 2.0
    ),
}


@pytest.mark.parametrize("form", ["operator", "add_constraints"])
@pytest.mark.parametrize("rhs", list(EXPR_RHS))
def test_expression_rhs_freezes_sparse_and_matches_dense(rhs: str, form: str) -> None:
    require_v1()
    c1, c2 = twin_models()
    con1 = c1.m.add_constraints(
        c1.gen_sum() <= EXPR_RHS[rhs](c1), name="c", freeze=True
    )
    lhs2 = c2.gen_sum()
    rhs2 = EXPR_RHS[rhs](c2)
    with no_densify():
        if form == "operator":
            con2 = c2.m.add_constraints(lhs2 <= rhs2, name="c")
        else:
            con2 = c2.m.add_constraints(lhs2, "<=", rhs2, name="c")
    assert isinstance(con2, CSRConstraint)
    assert_frozen_equal(con1, con2)


MASK_KINDS: dict[str, Callable[[Case], xr.DataArray | None]] = {
    "full": lambda c: None,
    "masked": lambda c: c.load > 4,
    "all-masked": lambda c: c.load > np.inf,
}


def softened(
    sign: str,
    mask_kind: str,
    route: str,
    max_violation: float | None,
    freeze: bool,
) -> tuple[Case, ConstraintBase]:
    c = base_model(sparse=freeze)
    c.m.add_objective(1.0 * c.gen_p.sum())
    mask = MASK_KINDS[mask_kind](c)
    args = (c.balance_lhs(), sign, c.load)
    kwargs: dict[str, Any] = dict(name="c", mask=mask)
    if route == "soften":
        con = c.m.add_constraints(*args, **kwargs)
        con.soften(penalty=2.0, max_violation=max_violation)
    else:
        kwargs["freeze"] = None if route == "penalty-default" else freeze
        con = c.m.add_constraints(*args, **kwargs, penalty=2.0)
    return c, con


SOFTEN_ROUTES = [
    ("soften", None),
    ("soften", 5.0),
    ("penalty", None),
    ("penalty-default", None),
]


@pytest.mark.parametrize(("route", "max_violation"), SOFTEN_ROUTES)
@pytest.mark.parametrize("mask_kind", list(MASK_KINDS))
@pytest.mark.parametrize("sign", ["<=", ">=", "=="])
def test_frozen_soften_matches_dense(
    sign: str, mask_kind: str, route: str, max_violation: float | None, tmp_path: Path
) -> None:
    require_v1()
    dense_case, dense = softened(sign, mask_kind, route, max_violation, freeze=False)
    with no_densify():
        case, con = softened(sign, mask_kind, route, max_violation, freeze=True)
        case.m.to_netcdf(tmp_path / "m.nc")
    assert isinstance(con, CSRConstraint)
    assert case.m.constraints["c"] is con
    assert_frozen_equal(dense, con)
    obj, want_obj = case.m.objective.expression, dense_case.m.objective.expression
    assert isinstance(obj, LinearExpression) and isinstance(want_obj, LinearExpression)
    assert_cells_equal(obj, want_obj, ())
    assert dense.slack is not None
    read = linopy.read_netcdf(tmp_path / "m.nc").constraints["c"]
    assert isinstance(read, CSRConstraint)
    assert_frozen_equal(dense, read)
    for frozen_con in (con, read, con.mutable(), dense.freeze()):
        slack = frozen_con.slack
        assert slack is not None
        assert_varequal(slack.positive, dense.slack.positive)
        assert (slack.negative is None) == (dense.slack.negative is None)
        if slack.negative is not None and dense.slack.negative is not None:
            assert_varequal(slack.negative, dense.slack.negative)


def test_frozen_soften_keeps_sparse_objective() -> None:
    require_v1()
    c = base_model(sparse=True)
    with no_densify():
        c.m.add_objective((1.0 * c.gen_p).groupby(c.gbus).sum().sum())
        c.m.add_constraints(c.balance_lhs() >= c.load, penalty=2.0)
    assert c.m.objective.expression.is_sparse


def test_frozen_soften_max_sense_with_array_penalty() -> None:
    require_v1()
    c = base_model(sparse=True)
    c.m.add_objective(1.0 * c.gen_p.sum(), sense="max")
    with no_densify():
        con = c.m.add_constraints(c.balance_lhs(), ">=", c.load, name="c")
    assert isinstance(con, CSRConstraint)
    penalty = xr.full_like(c.load, 2.0)
    slack = con.soften(penalty=penalty)
    expected_objective = 1.0 * c.gen_p.sum() - (penalty * slack.positive).sum()
    obj = c.m.objective.expression
    assert isinstance(obj, LinearExpression)
    assert_cells_equal(obj, expected_objective, ())


@pytest.mark.skipif("highs" not in linopy.available_solvers, reason="needs highs")
@pytest.mark.parametrize("sign", ["<=", ">=", "=="])
def test_frozen_soften_solves_like_dense(sign: str) -> None:
    require_v1()
    values = []
    for freeze in (False, True):
        c, _ = softened(sign, "masked", "soften", 20.0, freeze)
        for var in (c.gen_p, c.flow):
            var.update(lower=0, upper=1)
        c.m.solve("highs")
        values.append(c.m.objective.value)
    assert values[0] == pytest.approx(values[1])


def soften_twice(con: ConstraintBase) -> Any:
    con.soften(penalty=1.0)
    return con.soften(penalty=1.0)


SOFTEN_ERRORS: dict[str, tuple[Callable[[ConstraintBase], Any], type, str]] = {
    "twice": (soften_twice, ValueError, "already softened"),
    "zero-penalty": (lambda con: con.soften(penalty=0.0), ValueError, "not positive"),
    "no-objective": (
        lambda con: con.soften(penalty=1.0),
        ValueError,
        "Objective must be defined",
    ),
    "mixed-signs": (
        lambda con: con.soften(penalty=1.0),
        NotImplementedError,
        "mixed signs",
    ),
}

# "twice", "zero-penalty" and "no-objective" are already covered for dense
# constraints in test/test_constraint.py; only the frozen path needs them here.
FROZEN_ONLY_ERRORS = {"twice", "zero-penalty", "no-objective"}

SOFTEN_REJECT_CASES = [
    pytest.param(error, freeze, id=f"{error}-{'frozen' if freeze else 'dense'}")
    for error in SOFTEN_ERRORS
    for freeze in ((True,) if error in FROZEN_ONLY_ERRORS else (False, True))
]


@pytest.mark.parametrize(("error", "freeze"), SOFTEN_REJECT_CASES)
def test_soften_rejects(error: str, freeze: bool) -> None:
    require_v1()
    c = base_model()
    if error != "no-objective":
        c.m.add_objective(1.0 * c.gen_p.sum())
    sign: Any = ">="
    if error == "mixed-signs":
        sign = xr.DataArray(np.where(c.load > 4, ">=", "<="), coords=c.load.coords)
    con = c.m.add_constraints(c.balance_lhs(), sign, c.load, name="c", freeze=freeze)
    assert isinstance(con, CSRConstraint) == freeze
    call, exc, match = SOFTEN_ERRORS[error]
    with pytest.raises(exc, match=match):
        call(con)


def frozen_model(soften: bool) -> tuple[Model, CSRConstraint]:
    c = base_model(sparse=True)
    for var in (c.gen_p, c.flow):
        var.update(lower=0, upper=1)
    with no_densify():
        c.m.add_objective((1.0 * c.gen_p).groupby(c.gbus).sum().sum())
        con = c.m.add_constraints(c.balance_lhs(), ">=", c.load, name="c")
        if soften:
            con.soften(penalty=2.0, max_violation=20.0)
    assert isinstance(con, CSRConstraint)
    return c.m, con


@pytest.mark.skipif("highs" not in linopy.available_solvers, reason="needs highs")
@pytest.mark.parametrize("include_solution", [True, False])
@pytest.mark.parametrize("deep", [True, False])
def test_copy_keeps_frozen_constraints(deep: bool, include_solution: bool) -> None:
    require_v1()
    m, con = frozen_model(soften=True)
    m.solve("highs")
    with no_densify():
        c = m.copy(deep=deep, include_solution=include_solution)
    copied = c.constraints["c"]
    assert copied.model is c and copied.name == "c"
    assert c.objective.expression.is_sparse
    assert con.slack is not None and copied.slack is not None
    assert copied.slack.positive.labels.equals(con.slack.positive.labels)
    assert copied.slack.positive.model is c
    assert ("dual" in copied.mutable().data) == include_solution


@pytest.mark.parametrize("deep", [True, False])
def test_softening_frozen_copy_leaves_original(deep: bool) -> None:
    require_v1()
    m, con = frozen_model(soften=False)
    with no_densify():
        copied = m.copy(deep=deep)
        copied.constraints["c"].soften(penalty=2.0)
    assert con.slack is None and "c_slack_pos" not in m.variables
    assert m.objective.expression.nterm < copied.objective.expression.nterm
    assert con.nterm < copied.constraints["c"].nterm


@pytest.mark.parametrize("deep", [True, False])
def test_copy_frozen_matrices(deep: bool) -> None:
    require_v1()
    m = Model(sparse=True)
    i = pd.RangeIndex(4, name="i")
    x = m.add_variables(lower=0, coords=[i], name="x")
    b = m.add_variables(coords=[i], binary=True, name="b")
    lhs = (2 * x).assign_coords(aux=("i", [1.0, 2, 3, 4]))
    mask = xr.DataArray([True, False, True, True], coords=[i])
    scaling = xr.DataArray([1.0, 2, 3, 4], coords=[i])
    m.add_constraints(lhs >= 1, name="c", mask=mask, scaling=scaling)
    m.add_indicator_constraints(b, 1, x <= 5, name="ind")
    c = m.copy(deep=deep)
    for name, con in m.constraints.items():
        copied = c.constraints[name]
        assert isinstance(con, CSRConstraint) and isinstance(copied, CSRConstraint)
        assert np.shares_memory(con._csr.data, copied._csr.data) != deep
    got, want = c.matrices, m.matrices
    assert got.A is not None and want.A is not None
    assert got.indicator_A is not None and want.indicator_A is not None
    pairs = [
        (got.A.toarray(), want.A.toarray()),
        (got.b, want.b),
        (got.sense, want.sense),
        (got.clabels, want.clabels),
        (got.indicator_A.toarray(), want.indicator_A.toarray()),
        (got.indicator_b, want.indicator_b),
        (got.indicator_binvar, want.indicator_binvar),
    ]
    for g, w in pairs:
        assert np.array_equal(g, w)
    with pytest.raises(ValueError, match="read-only"):
        c.constraints["c"].coords["aux"].values[0] = -1
    assert c.constraints["c"].coords["aux"].equals(m.constraints["c"].coords["aux"])
