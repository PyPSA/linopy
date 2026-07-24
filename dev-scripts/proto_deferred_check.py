"""
Equivalence check: deferred groupby-sum constraint == dense groupby path.

Builds the same nodal-balance constraint twice on identical models:
  dense    : m.add_constraints(gen.groupby(bus).sum() + flow.groupby(bus0).sum()
                               - flow.groupby(bus1).sum() == load)
  deferred : add_deferred_constraints(...) via CSRConstraint, no padded rectangle

and compares labels, terms (coeffs per (label, var)), rhs and sign.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from proto_deferred_groupby import DeferredGroupbySum, add_deferred_constraints

import linopy


def build_base_model(n_bus: int, gens_per_bus: list[int], n_snap: int, seed: int = 0):
    """Model with gen_p (gen, snapshot) and flow (line, snapshot) on a ring."""
    rng = np.random.default_rng(seed)
    buses = pd.Index([f"bus{i}" for i in range(n_bus)], name="bus")
    gen_bus = np.repeat(np.arange(n_bus), gens_per_bus)
    gens = pd.Index([f"gen{i}" for i in range(len(gen_bus))], name="gen")
    lines = pd.Index([f"line{i}" for i in range(n_bus)], name="line")  # ring
    snaps = pd.Index(range(n_snap), name="snapshot")

    m = linopy.Model()
    gen_p = m.add_variables(coords=[gens, snaps], name="gen_p")
    flow = m.add_variables(coords=[lines, snaps], name="flow")

    gbus = pd.Series(buses[gen_bus], index=gens, name="bus")
    bus0 = pd.Series(buses[np.arange(n_bus)], index=lines, name="bus")
    bus1 = pd.Series(buses[(np.arange(n_bus) + 1) % n_bus], index=lines, name="bus")
    load = xr.DataArray(
        rng.uniform(1, 10, (n_bus, n_snap)), coords=[buses, snaps], name="load"
    )
    # per-gen efficiency coefficient so coefficients are not all 1
    eff = xr.DataArray(rng.uniform(0.5, 1.5, len(gens)), coords=[gens])
    return m, gen_p, flow, eff, gbus, bus0, bus1, load, buses


def canon(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by(["labels", "vars"])
        .agg(pl.col("coeffs").sum(), pl.col("sign").first(), pl.col("rhs").first())
        .sort(["labels", "vars"])
    )


def check(n_bus: int, gens_per_bus: list[int], n_snap: int) -> None:
    args = dict(n_bus=n_bus, gens_per_bus=gens_per_bus, n_snap=n_snap)

    # dense path
    m1, gen_p, flow, eff, gbus, bus0, bus1, load, buses = build_base_model(**args)
    lhs = (
        (eff * gen_p).groupby(gbus).sum()
        + (1.0 * flow).groupby(bus0).sum()
        - (1.0 * flow).groupby(bus1).sum()
    )
    c1 = m1.add_constraints(lhs == load, name="balance")
    m1.add_objective((1.0 * gen_p).sum())
    print("dense constraint dims:", dict(c1.data.sizes))

    # deferred path; the dense groupby sorts group labels, so use a sorted
    # group index to get an identical constraint grid
    m2, gen_p, flow, eff, gbus, bus0, bus1, load, buses = build_base_model(**args)
    parts = [
        DeferredGroupbySum(eff * gen_p, gbus),
        DeferredGroupbySum(1.0 * flow, bus0),
        DeferredGroupbySum(-1.0 * flow, bus1),
    ]
    c2 = add_deferred_constraints(m2, parts, "=", load, "balance", buses.sort_values())
    m2.add_objective((1.0 * gen_p).sum())
    print("deferred nterm (max row nnz):", c2.nterm)

    d1, d2 = canon(c1.to_polars()), canon(c2.to_polars())
    assert d1.shape == d2.shape, (d1.shape, d2.shape)
    assert (d1["labels"] == d2["labels"]).all()
    assert (d1["vars"] == d2["vars"]).all()
    assert np.allclose(d1["coeffs"], d2["coeffs"])
    assert (d1["sign"] == d2["sign"]).all()
    assert np.allclose(d1["rhs"], d2["rhs"])

    # label layout: same grid, same order
    assert c1.labels.dims == tuple(c2.coord_names), (c1.labels.dims, c2.coord_names)
    assert np.array_equal(
        np.sort(c1.labels.values.ravel()), np.sort(c2.active_labels())
    )

    # end to end: identical LP files (term order within a row is not
    # semantic -- CSR emits column-sorted, dense emits build-order -- so
    # sort term lines within each block before comparing)
    import re
    import tempfile
    from pathlib import Path

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

    with tempfile.TemporaryDirectory() as tmp:
        f1, f2 = Path(tmp) / "dense.lp", Path(tmp) / "deferred.lp"
        m1.to_file(f1)
        m2.to_file(f2)
        assert canon_lp(f1.read_text()) == canon_lp(f2.read_text()), "LP files differ"

    print(f"OK: {d1.height} term rows and LP files identical (dense vs deferred)")


def main() -> None:
    # small; bus names sort like creation order
    check(n_bus=5, gens_per_bus=[7, 1, 3, 1, 2], n_snap=3)
    # 12 buses: lexicographic group order ('bus10' < 'bus2') differs from
    # creation order, exercising label-based rhs alignment
    check(n_bus=12, gens_per_bus=[7, 1, 3, 1, 2, 1, 1, 4, 1, 2, 1, 1], n_snap=3)


if __name__ == "__main__":
    main()
