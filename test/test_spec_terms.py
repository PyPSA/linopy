"""
The term path of a sparse build, :mod:`linopy.spec.terms`: it builds the
model the eager evaluator builds, which takes over wherever the term path does
not cover a declaration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("mathspec")

from spec_helpers import (  # noqa: E402
    CURVE_DATA,
    CURVE_SPEC,
    DISPATCH_DATA,
    FULL_X,
    FULL_Y,
    MASKED_CURVE_SPEC,
    RAGGED_X,
    RAGGED_Y,
    TT,
    needs_examples,
    with_,
    yaml_dict,
)
from test_spec_builder import (  # noqa: E402
    ABSENT_DATA,
    ABSENT_SPEC,
    BESIDE_A_VACATED_SHIFT,
    EXAMPLES,
    FULL_C,
    FULL_W,
    NAMED_WX,
    SPARSE_SPEC,
    T,
    example,
)
from test_spec_operators import (  # noqa: E402
    OPERATOR_DATA,
    RELATION_DATA,
    RELATION_SUMS,
    operator_spec,
    relations_spec,
)

from linopy import LinearExpression, Model, QuadraticExpression  # noqa: E402
from linopy.spec import terms  # noqa: E402
from linopy.testing import (  # noqa: E402
    assert_conequal,
    assert_equivalent,
    assert_quadequal,
    assert_varequal,
    synthetic_sources,
)


def assert_alike(
    a: LinearExpression | QuadraticExpression,
    b: LinearExpression | QuadraticExpression,
) -> None:
    if isinstance(a, LinearExpression) and isinstance(b, LinearExpression):
        assert_equivalent(a, b)
    else:
        assert_quadequal(a, b)


def assert_built_alike(spec: Any, data: dict[str, Any]) -> None:
    """
    *spec* built on the term path and on the eager path, equal as the solver
    reads it: the term path spreads a term onto the nonzero entries of a factor
    only, so it holds no term of coefficient zero where the eager path does.
    """
    with terms.eager_only():
        eager = Model.from_spec(spec, data, sparse=True)
    built = Model.from_spec(spec, data, sparse=True)
    assert list(built.variables) == list(eager.variables)
    for name in eager.variables:
        assert_varequal(built.variables[name], eager.variables[name])
    assert list(built.constraints) == list(eager.constraints)
    for name in eager.constraints:
        con = eager.constraints[name]
        assert_conequal(built.constraints[name], con, strict=False)
        xr.testing.assert_equal(built.constraints[name].labels, con.labels)
    assert list(built.expressions) == list(eager.expressions)
    for name, expression in eager.expressions.items():
        assert_alike(built.expressions[name], expression)
    assert_alike(built.objective.expression, eager.objective.expression)
    assert built.objective.sense == eager.objective.sense


@pytest.fixture
def tables(monkeypatch: pytest.MonkeyPatch) -> list[terms.Table]:
    """The tables the term path turns into expressions."""
    made: list[terms.Table] = []
    materialised = terms.materialised

    def recorded(table: terms.Table, *args: Any) -> Any:
        made.append(table)
        return materialised(table, *args)

    monkeypatch.setattr(terms, "materialised", recorded)
    return made


SPARSE_DATA = {"t": T, "w": FULL_W, "c": FULL_C}
PROGRAMS: dict[str, tuple[Any, dict[str, Any]]] = {
    "dispatch": (yaml_dict(), DISPATCH_DATA),
    "curve": (CURVE_SPEC, {**CURVE_DATA, "bp_x": FULL_X, "bp_y": FULL_Y}),
    "ragged-curve": (
        MASKED_CURVE_SPEC,
        {**CURVE_DATA, "bp_x": RAGGED_X, "bp_y": RAGGED_Y},
    ),
    "operators": (operator_spec(), OPERATOR_DATA),
    "relations": (relations_spec(), RELATION_DATA),
    **{f"relation-sum-{k}": v for k, v in RELATION_SUMS.items()},
    "sparse": (SPARSE_SPEC, SPARSE_DATA),
    "named-expression": (NAMED_WX, SPARSE_DATA),
    "beside-a-vacated-shift": (BESIDE_A_VACATED_SHIFT, SPARSE_DATA),
    "absent-rows": (
        with_(
            ABSENT_SPEC,
            constraints={"cap": {"dims": ["t"], "expression": "a * x <= 5"}},
        ),
        ABSENT_DATA,
    ),
}


@pytest.mark.parametrize(("spec", "data"), PROGRAMS.values(), ids=list(PROGRAMS))
def test_the_term_path_builds_the_model_the_eager_path_builds(
    spec: Any, data: dict[str, Any]
) -> None:
    assert_built_alike(spec, data)


@needs_examples
@pytest.mark.parametrize("path", [example(p) for p in EXAMPLES])
def test_every_mathspec_example_builds_alike_on_both_paths(path: Path) -> None:
    assert_built_alike(path, synthetic_sources(path))


E = pd.Index(["e1", "e2", "e3"], name="e")
G = pd.Index(["north", "south"], name="g")
Q = xr.DataArray(
    [[1.0, 0.0, 2.0, 3.0], [0.0, 0.0, 4.0, 5.0], [6.0, 7.0, np.nan, 8.0]],
    coords=[E, TT],
)
DATA: dict[str, Any] = {
    "t": TT,
    "e": E,
    "g": G,
    "group_of": pd.DataFrame({"e": E, "g": ["north", "north", "south"]}),
    "p": xr.DataArray([2.0, 4.0, 8.0], coords=[E]),
    "q": Q,
    "r": Q.where(Q != 0, np.inf).fillna(1.0),
    "lag": xr.DataArray([0, 1, 2], coords=[E]),
    "w": xr.DataArray([[1.0, 2.0, np.nan, 4.0], [5.0, 6.0, 7.0, 8.0]], coords=[G, TT]),
}


def row(dims: list[str], expression: str) -> dict[str, Any]:
    """A program of one constraint ``row`` over *dims*, beside a linear objective."""
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


@pytest.mark.parametrize("dense_keys", [terms.DENSE_KEYS, 0], ids=["dense", "sorted"])
@pytest.mark.parametrize(("dims", "expression"), ROWS.values(), ids=list(ROWS))
def test_a_row_from_term_tables_is_the_row_the_eager_path_builds(
    tables: list[terms.Table],
    monkeypatch: pytest.MonkeyPatch,
    dense_keys: int,
    dims: list[str],
    expression: str,
) -> None:
    monkeypatch.setattr(terms, "DENSE_KEYS", dense_keys)
    assert_built_alike(row(dims, expression), DATA)
    assert len(tables) == 2


EAGER: dict[str, dict[str, Any]] = {
    "a-window-sum": row(["e", "t"], "sum_back(y, along=t, window=2) + x <= 1"),
    "infinite-data-beside-a-term": row(["e", "t"], "y + r * x <= 1"),
    "a-product-of-two-terms": with_(
        row(["e"], "x <= 1"),
        objective={"sense": "minimize", "expression": "sum(x * x)"},
    ),
    "a-right-side-over-a-dimension-the-left-lacks": row(["e", "t"], "x <= q * y"),
    "a-row-of-zero-weights": row(["e", "t"], "q * x <= 1"),
}


@pytest.mark.parametrize("spec", EAGER.values(), ids=list(EAGER))
def test_a_row_the_term_path_does_not_cover_builds_on_the_eager_path(
    spec: dict[str, Any],
) -> None:
    assert_built_alike(spec, DATA)
