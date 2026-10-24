"""Long-form inputs: ``linopy.Table`` constants and relation group-by."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
from test_store_conformance import RELATION, TABLE_A, TABLE_AB, A, B, assert_same

from linopy import LinearExpression, Model, QuadraticExpression, Table, Variable
from linopy.testing import assert_equivalent

TABLES: dict[str, tuple[Table, xr.DataArray]] = {
    "all-keys-default": (
        TABLE_AB,
        xr.DataArray([[1.0, 4], [5, 3], [2, 5]], coords=[A, B]),
    ),
    "missing-key-column": (TABLE_A, xr.DataArray([2.0, 3, 1], coords=[A])),
    "polars": (
        Table(pl.DataFrame({"b": ["q", "p"], "a": [1, 1], "v": [3.0, 4]}), "v", 1),
        xr.DataArray([[1.0, 4, 1], [1, 3, 1]], coords=[B, A]),
    ),
}


def x(m: Model) -> Variable:
    return m.add_variables(coords=[A, B], name="x")


POSITIONS: dict[str, Callable[[Model, Any], Any]] = {
    "lower": lambda m, c: m.add_variables(lower=c, coords=[A, B], name="v").lower,
    "upper": lambda m, c: m.add_variables(upper=c, coords=[A, B], name="v").upper,
    "coefficient": lambda m, c: c * x(m),
    "to_linexpr": lambda m, c: x(m).to_linexpr(c),
    "add-constant": lambda m, c: 2 * x(m) + c,
    "sub-constant": lambda m, c: 2 * x(m) - c,
    "rsub-constant": lambda m, c: c - 2 * x(m),
    "rhs-add_constraints": lambda m, c: m.add_constraints(2 * x(m), "<=", c),
    "rhs-le": lambda m, c: 2 * x(m) <= c,
    "div": lambda m, c: x(m) / c,
    "quad-mul": lambda m, c: c * x(m) ** 2,
}


@pytest.mark.parametrize("table", list(TABLES))
@pytest.mark.parametrize("position", list(POSITIONS))
def test_table_matches_dataarray(backend: Model, position: str, table: str) -> None:
    tab, da = TABLES[table]
    want = POSITIONS[position](Model(sparse=backend.sparse), da)
    assert_same(POSITIONS[position](backend, tab), want)


INVALID: dict[str, tuple[Table, str]] = {
    "duplicate-keys": (
        Table(pd.DataFrame({"a": [0, 0], "v": [1.0, 2]}), "v"),
        "duplicate keys",
    ),
    "unknown-key-column": (
        Table(pd.DataFrame({"a": [0], "z": [0], "v": [1.0]}), "v"),
        "unmatched \\['z'\\]",
    ),
    "key-outside-coords": (
        Table(pd.DataFrame({"a": [7], "v": [1.0]}), "v"),
        "outside the target coordinates",
    ),
    "no-value-column": (Table(pd.DataFrame({"a": [0]}), "v"), "no value column"),
    "nan-default": (Table(pd.DataFrame({"a": [0], "v": [1.0]}), "v"), "NaN"),
}


@pytest.mark.parametrize("position", ["coefficient", "add-constant", "rhs-le"])
@pytest.mark.parametrize("case", list(INVALID))
def test_invalid_table_raises(backend: Model, case: str, position: str) -> None:
    tab, match = INVALID[case]
    with pytest.raises(ValueError, match=match):
        POSITIONS[position](backend, tab)


def test_table_reflected_comparison(backend: Model) -> None:
    assert_same(
        TABLE_AB >= 2 * x(backend), 2 * x(Model(sparse=backend.sparse)) <= TABLE_AB
    )


def test_table_penalty(backend: Model) -> None:
    def softened(m: Model, penalty: Any) -> LinearExpression | QuadraticExpression:
        v = x(m)
        m.add_objective(v.sum())
        m.add_constraints(v >= 0, penalty=penalty)
        return m.objective.expression

    want = softened(Model(sparse=backend.sparse), TABLES["all-keys-default"][1])
    assert_same(softened(backend, TABLE_AB), want)


def test_table_without_target_coords_raises() -> None:
    with pytest.raises(ValueError, match="needs target coordinates"):
        Model().add_variables(lower=TABLE_A)


def incidence(relation: pd.DataFrame, weights: Any = 1.0) -> xr.DataArray:
    member, group = relation.columns
    weighted = relation.assign(w=weights)
    matrix = weighted.pivot_table("w", member, group, aggfunc="sum", fill_value=0.0)
    return xr.DataArray(matrix)


@pytest.mark.parametrize(
    "relation", [RELATION, RELATION[RELATION.a != 1]], ids=["all", "unlisted"]
)
@pytest.mark.parametrize(
    "to_frame", [pd.DataFrame, lambda r: pl.from_dict(r.to_dict("list"))]
)
def test_relation_groupby_matches_incidence_matmul(
    backend: Model, to_frame: Callable[[pd.DataFrame], Any], relation: pd.DataFrame
) -> None:
    expr = 2 * backend.add_variables(coords=[B, A], name="x") + 1
    res = expr.groupby(to_frame(relation)).sum()
    assert res.is_sparse == backend.sparse
    assert_equivalent(res, expr @ incidence(relation).reindex(a=A, fill_value=0.0))


def test_relation_groupby_quadratic_raises(backend: Model) -> None:
    with pytest.raises(NotImplementedError, match="quadratic"):
        (x(backend) ** 2).groupby(RELATION).sum()


@pytest.mark.parametrize(
    ("relation", "kwargs", "match"),
    [
        (pd.DataFrame({"a": [0, 9], "g": ["u", "v"]}), {}, r"relation members \[9\]"),
        (pd.DataFrame({"a": [0, 1], "b": ["p", "q"]}), {}, "must not name"),
        (RELATION, {"use_fallback": True}, "not supported"),
        (RELATION, {"observed": True}, "not supported"),
    ],
)
def test_invalid_relation_groupby_raises(
    backend: Model, relation: pd.DataFrame, kwargs: dict[str, bool], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        (1 * x(backend)).groupby(relation).sum(**kwargs)


LINKS = pd.Index(["l0", "l1", "l2", "l3"], name="link")
SNAPSHOTS = pd.RangeIndex(2, name="snapshot")
PORTS = pd.DataFrame(
    {
        "link": list(LINKS) * 2,
        "port": [0] * 4 + [1] * 4,
        "bus": ["b0", "b0", "b1", "b2", "b1", "b2", "b2", "b0"],
    }
)
EFFICIENCY = pd.DataFrame({"link": LINKS, "eff": [0.9, 0.8, 0.95, 0.7]})


def port(i: int) -> pd.DataFrame:
    return PORTS.loc[PORTS.port == i, ["link", "bus"]]


def test_link_ports_to_bus(backend: Model) -> None:
    p = backend.add_variables(coords=[SNAPSHOTS, LINKS], name="p")
    eff = Table(EFFICIENCY, "eff")
    touching = p.groupby(PORTS[["link", "bus"]]).sum()
    balance = (-1 * p).groupby(port(0)).sum() + (eff * p).groupby(port(1)).sum()

    weights = np.where(PORTS.port == 0, -1.0, np.tile(EFFICIENCY.eff, 2))
    for res, matrix in [
        (touching, incidence(PORTS[["link", "bus"]])),
        (balance, incidence(PORTS[["link", "bus"]], weights)),
    ]:
        assert isinstance(res, LinearExpression)
        assert res.is_sparse == backend.sparse
        assert_equivalent(res, 1 * p @ matrix)
