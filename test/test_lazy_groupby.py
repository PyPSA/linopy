"""
Tests for lazy groupby-sum (linopy.lazy): type stability, transparent
materialization, and direct CSR realization under freeze. v1-only feature.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

import linopy
from linopy import LinearExpression
from linopy.constraints import Constraint, CSRConstraint
from linopy.semantics import is_v1
from linopy.testing import assert_conequal, assert_linequal


def require_v1() -> None:
    if not is_v1():
        pytest.skip("lazy groupby-sum is gated behind v1 semantics")


def base_model(gens_per_bus=(7, 1, 3, 1, 2), n_snap=3, seed=0):
    """Model with gen_p (gen, snapshot) and flow (line, snapshot) on a ring."""
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
    # v1 requires explicit alignment: order the load like the (sorted) groups
    load = xr.DataArray(
        rng.uniform(1, 10, (n_bus, n_snap)), coords=[buses, snaps], name="load"
    ).sortby("bus")
    eff = xr.DataArray(rng.uniform(0.5, 1.5, len(gens)), coords=[gens])
    return m, gen_p, flow, eff, gbus, bus0, bus1, load, buses


def balance_lhs(gen_p, flow, eff, gbus, bus0, bus1, lazy):
    return (
        (eff * gen_p).groupby(gbus).sum(lazy=lazy)
        + (1.0 * flow).groupby(bus0).sum(lazy=lazy)
        - (1.0 * flow).groupby(bus1).sum(lazy=lazy)
    )


def canon(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["labels", "vars"])
        .agg(pl.col("coeffs").sum(), pl.col("sign").first(), pl.col("rhs").first())
        .sort(["labels", "vars"])
    )


def test_lazy_requires_v1():
    m, gen_p, flow, eff, gbus, *_ = base_model()
    if is_v1():
        res = (eff * gen_p).groupby(gbus).sum(lazy=True)
        assert type(res) is LinearExpression
        return
    with pytest.raises(ValueError, match="requires v1 semantics"):
        (eff * gen_p).groupby(gbus).sum(lazy=True)
    # the option is not honored under legacy: result is eager and dense
    linopy.options["lazy_groupby"] = True
    try:
        res = (eff * gen_p).groupby(gbus).sum()
    finally:
        linopy.options["lazy_groupby"] = False
    assert res._lazy is None


def test_lazy_is_plain_linear_expression_and_materializes_identically():
    require_v1()
    m, gen_p, flow, eff, gbus, *_ = base_model()
    lazy = (eff * gen_p).groupby(gbus).sum(lazy=True)
    eager = (eff * gen_p).groupby(gbus).sum()
    assert type(lazy) is LinearExpression
    # comparing data materializes; the result must be exactly today's
    assert_linequal(lazy, eager)


def test_lazy_composition_materializes_identically():
    require_v1()
    m, gen_p, flow, eff, gbus, bus0, bus1, *_ = base_model()
    lazy = balance_lhs(gen_p, flow, eff, gbus, bus0, bus1, lazy=True)
    eager = balance_lhs(gen_p, flow, eff, gbus, bus0, bus1, lazy=False)
    assert type(lazy) is LinearExpression
    assert_linequal(lazy, eager)


def test_scalar_ops_stay_lazy():
    require_v1()
    m, gen_p, flow, eff, gbus, *_ = base_model()
    lazy = -2.0 * (eff * gen_p).groupby(gbus).sum(lazy=True)
    assert lazy._lazy is not None  # still unmaterialized after neg/scale
    eager = -2.0 * (eff * gen_p).groupby(gbus).sum()
    # scaling after grouping normalizes padded-slot fills while scaling
    # before grouping keeps fresh pads; masked-slot fill is not contractual,
    # so compare with masked slots blanked on both sides
    from linopy.testing import _sort_by_vars_along_term

    a, b = _sort_by_vars_along_term(lazy), _sort_by_vars_along_term(eager)
    assert (a.vars == b.vars).all()
    xr.testing.assert_allclose(
        a.coeffs.where(a.vars != -1), b.coeffs.where(b.vars != -1)
    )
    xr.testing.assert_equal(a.const, b.const)


def test_freeze_realizes_csr_without_dense_rectangle():
    require_v1()
    m1, *rest = base_model()
    lhs1 = balance_lhs(*rest[:6], lazy=False)
    c1 = m1.add_constraints(lhs1 == rest[6], name="balance")

    m2, *rest2 = base_model()
    lhs2 = balance_lhs(*rest2[:6], lazy=True)
    c2 = m2.add_constraints(lhs2 == rest2[6], name="balance", freeze=True)

    assert isinstance(c2, CSRConstraint)
    d1, d2 = canon(c1.to_polars()), canon(c2.to_polars())
    assert d1["labels"].equals(d2["labels"])
    assert d1["vars"].equals(d2["vars"])
    assert np.allclose(d1["coeffs"], d2["coeffs"])
    assert (d1["sign"] == d2["sign"]).all()
    assert np.allclose(d1["rhs"], d2["rhs"])
    assert np.array_equal(
        np.sort(c1.labels.values.ravel()), np.sort(c2.active_labels())
    )


def test_freeze_false_falls_back_to_identical_dense_constraint():
    require_v1()
    m1, *rest = base_model()
    c1 = m1.add_constraints(
        balance_lhs(*rest[:6], lazy=False) == rest[6], name="balance"
    )

    m2, *rest2 = base_model()
    c2 = m2.add_constraints(
        balance_lhs(*rest2[:6], lazy=True) == rest2[6], name="balance"
    )
    assert isinstance(c2, Constraint)
    assert_conequal(c1, c2)


def test_option_gates_lazy_and_freeze_model_default():
    require_v1()
    m, *rest = base_model()
    m.freeze_constraints = True
    linopy.options["lazy_groupby"] = True
    try:
        lhs = balance_lhs(*rest[:6], lazy=None)
        con = m.add_constraints(lhs == rest[6], name="balance")
    finally:
        linopy.options["lazy_groupby"] = False
    assert isinstance(con, CSRConstraint)


def test_materialized_lazy_still_freezes_via_dense_path():
    require_v1()
    m, *rest = base_model()
    lhs = balance_lhs(*rest[:6], lazy=True)
    _ = lhs.nterm  # force materialization before constraint creation
    con = m.add_constraints(lhs == rest[6], name="balance", freeze=True)
    assert isinstance(con, CSRConstraint)


def test_nan_rhs_raises_on_both_paths():
    require_v1()
    m1, *rest = base_model()
    load = rest[6].copy()
    load[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        m1.add_constraints(
            balance_lhs(*rest[:6], lazy=False) == load, name="bal", freeze=True
        )

    m2, *rest2 = base_model()
    load2 = rest2[6].copy()
    load2[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        m2.add_constraints(
            balance_lhs(*rest2[:6], lazy=True) == load2, name="bal", freeze=True
        )


def test_reordered_rhs_raises_on_both_paths():
    require_v1()
    m1, *rest = base_model()
    load = rest[6].isel(bus=slice(None, None, -1))  # same labels, reversed
    with pytest.raises(ValueError, match="[Cc]oordinate"):
        m1.add_constraints(
            balance_lhs(*rest[:6], lazy=False) == load, name="bal", freeze=True
        )

    m2, *rest2 = base_model()
    load2 = rest2[6].isel(bus=slice(None, None, -1))
    with pytest.raises(ValueError, match="[Cc]oordinate"):
        m2.add_constraints(
            balance_lhs(*rest2[:6], lazy=True) == load2, name="bal", freeze=True
        )


def test_nan_grouper_raises_eagerly():
    require_v1()
    m, gen_p, flow, eff, gbus, *_ = base_model()
    gbus = gbus.copy()
    gbus.iloc[0] = np.nan
    with pytest.raises(ValueError, match="NaN values"):
        (1.0 * gen_p).groupby(gbus).sum(lazy=True)


def test_lp_files_identical(tmp_path):
    require_v1()
    sizes = (7, 1, 3, 1, 2, 1, 1, 4, 1, 2, 1, 1)

    m1, *rest = base_model(gens_per_bus=sizes)
    m1.add_constraints(balance_lhs(*rest[:6], lazy=False) == rest[6], name="balance")
    m1.add_objective((1.0 * rest[0]).sum())

    m2, *rest2 = base_model(gens_per_bus=sizes)
    m2.add_constraints(
        balance_lhs(*rest2[:6], lazy=True) == rest2[6], name="balance", freeze=True
    )
    m2.add_objective((1.0 * rest2[0]).sum())

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

    f1, f2 = tmp_path / "eager.lp", tmp_path / "lazy.lp"
    m1.to_file(f1)
    m2.to_file(f2)
    assert canon_lp(f1.read_text()) == canon_lp(f2.read_text())
