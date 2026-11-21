"""The term tables of :mod:`linopy.spec.terms`: a sparse build gives the problem the dense build gives."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("mathspec")

from linopy import Model  # noqa: E402
from linopy.spec import terms  # noqa: E402

T = pd.Index([0, 1, 2, 3], name="t")
E = pd.Index(["e1", "e2", "e3"], name="e")
G = pd.Index(["north", "south"], name="g")
Q = xr.DataArray(
    [[1.0, 0.0, 2.0, 3.0], [0.0, 0.0, 4.0, 5.0], [6.0, 7.0, np.nan, 8.0]],
    coords=[E, T],
)
DATA: dict[str, Any] = {
    "t": T,
    "e": E,
    "g": G,
    "group_of": pd.DataFrame({"e": E, "g": ["north", "north", "south"]}),
    "p": xr.DataArray([2.0, 4.0, 8.0], coords=[E]),
    "q": Q,
    "r": Q.where(Q != 0, np.inf).fillna(1.0),
    "lag": xr.DataArray([0, 1, 2], coords=[E]),
    "w": xr.DataArray([[1.0, 2.0, np.nan, 4.0], [5.0, 6.0, 7.0, 8.0]], coords=[G, T]),
}
ROWS: dict[str, tuple[list[str], str]] = {
    "a-number-times-a-term": (["e"], "3 * x - x <= 1"),
    "a-term-over-data": (["e", "t"], "y / p + x / 4 <= 1"),
    "a-sum-of-a-term-without-the-dimension": (["e"], "sum(x + y, over=t) <= 1"),
    "a-factor-over-a-dimension-the-term-lacks": (["e", "t"], "q * x + y <= 1"),
    "a-factor-with-a-hole-summed": (["e"], "sum(q * y + 1, over=t) <= 1"),
    "a-sum-over-everything": ([], "sum(p * y) <= 1"),
    "a-grouped-sum": (["g", "t"], "sum(p * y, over=e, by=group_of[g]) + z <= 1"),
    "a-grouped-sum-of-a-broadcast-term": (
        ["g", "t"],
        "sum(x + q, over=e, by=group_of[g]) + z <= 1",
    ),
    "a-lookup-of-a-term": (["e", "t"], "y <= at(2 * z, by=group_of[g])"),
    "a-lookup-of-data-beside-a-term": (
        ["e"],
        "sum(u * at(w, by=group_of[g]), over=t) <= 1",
    ),
    "a-lookup-of-data-beside-a-term-and-a-constant": (
        ["e"],
        "sum((y + 1) * at(w, by=group_of[g]), over=t) <= 1",
    ),
    "a-lookup-of-a-term-with-a-constant": (
        ["e", "t"],
        "y <= at(2 * z + 1, by=group_of[g])",
    ),
    "a-lookup-of-data-summed-over-the-key": (
        ["t"],
        "sum(u * at(w, by=group_of[g]), over=e) + sum(z, over=g) <= 1",
    ),
    "a-shift": (["e", "t"], "y - shift(y, along=t, offset=1) <= 1"),
    "a-shift-with-an-edge-value": (
        ["e", "t"],
        "y - shift(y + 1, along=t, offset=lag, edge=0) <= 1",
    ),
    "a-wrapped-shift-of-a-term-without-the-axis": (
        ["e", "t"],
        "y <= shift(p * x + y, along=t, offset=1, edge='wrap')",
    ),
    "a-lookup-of-a-term-over-the-key": (["e", "t"], "at(z + x, by=group_of[g]) <= 1"),
}


def spec(dims: list[str], expression: str) -> dict[str, Any]:
    return {
        "dimensions": {
            "t": {"dtype": "int", "ordered": True},
            "e": {"dtype": "str"},
            "g": {"dtype": "str"},
        },
        "relations": {"group_of": {"key": "e", "values": "g"}},
        "parameters": {
            "p": {"dims": ["e"]},
            "q": {"dims": ["e", "t"], "missing": "absent"},
            "w": {"dims": ["g", "t"], "missing": "absent"},
            "r": {"dims": ["e", "t"]},
            "lag": {"dims": ["e"], "dtype": "int"},
        },
        "variables": {
            "x": {"dims": ["e"], "bounds": {"lower": 0}},
            "y": {"dims": ["e", "t"], "where": "q > 0", "bounds": {"lower": 0}},
            "z": {"dims": ["g", "t"], "bounds": {"lower": 0}},
            "u": {"dims": ["e", "t"], "bounds": {"lower": 0}},
        },
        "constraints": {"row": {"dims": dims, "expression": expression}},
        "objective": {"sense": "minimize", "expression": "sum(p * x) + sum(q * y)"},
    }


def problem(m: Model) -> tuple[np.ndarray, ...]:
    matrices = m.matrices
    assert matrices.A is not None
    return matrices.A.toarray(), matrices.b, matrices.sense, matrices.c


@pytest.fixture
def tables(monkeypatch: pytest.MonkeyPatch) -> list[terms.Table]:
    """The tables the build turns into expressions, on the term path."""
    monkeypatch.setenv("LINOPY_SPEC_TERMS", "1")
    monkeypatch.setenv("LINOPY_SPEC_TERMS_SCOPE", "*")
    made: list[terms.Table] = []
    materialised = terms.materialised

    def recorded(table: terms.Table, *args: Any) -> Any:
        made.append(table)
        return materialised(table, *args)

    monkeypatch.setattr(terms, "materialised", recorded)
    return made


@pytest.mark.parametrize("dense_keys", [terms.DENSE_KEYS, 0], ids=["dense", "sorted"])
@pytest.mark.parametrize(("dims", "expression"), ROWS.values(), ids=list(ROWS))
def test_a_row_from_term_tables_is_the_row_the_dense_build_gives(
    tables: list[terms.Table],
    monkeypatch: pytest.MonkeyPatch,
    dense_keys: int,
    dims: list[str],
    expression: str,
) -> None:
    monkeypatch.setattr(terms, "DENSE_KEYS", dense_keys)
    sparse = Model.from_spec(spec(dims, expression), DATA, sparse=True)
    assert len(tables) == 2
    dense = Model.from_spec(spec(dims, expression), DATA)
    for built, expected in zip(problem(sparse), problem(dense)):
        np.testing.assert_array_equal(built, expected)


@pytest.mark.parametrize(
    "expression",
    [
        pytest.param("sum_back(y, along=t, window=2) + x <= 1", id="a-window-sum"),
        pytest.param("y + r * x <= 1", id="infinite-data-beside-a-term"),
    ],
)
def test_a_row_the_tables_do_not_hold_is_built_by_the_evaluator(
    tables: list[terms.Table], expression: str
) -> None:
    sparse = Model.from_spec(spec(["e", "t"], expression), DATA, sparse=True)
    assert len(tables) == 1
    dense = Model.from_spec(spec(["e", "t"], expression), DATA)
    for built, expected in zip(problem(sparse), problem(dense)):
        np.testing.assert_array_equal(built, expected)
