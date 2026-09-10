"""
Operators built as a constraint and folded as a named expression: sum,
grouped sum, ``at``, shift/translate, sum_back windows, and the where/mask
predicates that gate which rows exist.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

math_spec = pytest.importorskip("math_spec")
yaml = pytest.importorskip("yaml")

import linopy  # noqa: E402
from conftest import TT, WHERE_DATA, WHERE_SPEC, solved, with_  # noqa: E402
from linopy import LinearExpression, Model  # noqa: E402
from linopy.spec import SpecDataError  # noqa: E402

pytestmark = [
    pytest.mark.v1,
    pytest.mark.skipif("highs" not in linopy.available_solvers, reason="needs highs"),
]

# ---------------------------------------------------------------------------
# operators, built as a constraint and folded as a named expression
# ---------------------------------------------------------------------------

S = pd.Index(["a", "b"], name="s")
V = np.array([1.0, 2.0, 4.0, 8.0])
OPERATORS: dict[str, tuple[str, list[str], list[float]]] = {
    "shift-edge-0": ("shift(x, over=t, offset=1, edge=0)", ["t"], [0, 1, 2, 4]),
    "shift-ahead-edge-0": ("shift(x, over=t, offset=-1, edge=0)", ["t"], [2, 4, 8, 0]),
    "shift-wrap": ("shift(x, over=t, offset=1, edge='wrap')", ["t"], [8, 1, 2, 4]),
    "shift-wrap-in-groups": (
        "shift(x, over=t, offset=1, edge='wrap', by=season_of)",
        ["t"],
        [2, 1, 8, 4],
    ),
    "shift-by-group-offset": (
        "shift(x, over=t, offset=lag, edge=0, by=season_of)",
        ["t"],
        [0, 1, 0, 0],
    ),
    "sum-back": ("sum_back(x, over=t, within=2)", ["t"], [1, 3, 6, 12]),
    "sum-back-wrap": (
        "sum_back(x, over=t, within=2, edge='wrap')",
        ["t"],
        [9, 3, 6, 12],
    ),
    "sum-back-in-groups": (
        "sum_back(x, over=t, within=2, by=season_of)",
        ["t"],
        [1, 3, 4, 12],
    ),
    "sum-back-group-width": (
        "sum_back(x, over=t, within=width, by=season_of)",
        ["t"],
        [1, 2, 4, 12],
    ),
    "sum-by": ("sum(x, by=season_of)", ["s"], [3, 12]),
    "at": ("x * at(z, by=season_of)", ["t"], [10, 20, 80, 160]),
    "cases": ("x_state", ["t"], [100, 1, 2, 4]),
}


def operator_spec() -> dict[str, Any]:
    spec: dict[str, Any] = {
        "dimensions": {"t": {"dtype": "int"}, "s": {"dtype": "str"}},
        "lookups": {"season_of": {"over": "t", "into": "s"}},
        "parameters": {
            "v": {"dims": ["t"]},
            "z": {"dims": ["s"]},
            "lag": {"dims": ["s"], "dtype": "int"},
            "width": {"dims": ["s"], "dtype": "int"},
        },
        "variables": {"x": {"foreach": ["t"], "bounds": {"lower": 0, "upper": 100}}},
        "constraints": {"fix": {"foreach": ["t"], "expression": "x == v"}},
        "expressions": {
            "x_state": {
                "foreach": ["t"],
                "cases": {"first": {"when": "position(t) == 0", "expression": 100}},
                "otherwise": "shift(x, over=t, offset=1)",
            }
        },
        "objective": {"sense": "minimize", "expression": "sum(x)"},
    }
    for key, (expression, dims, _) in OPERATORS.items():
        name = key.replace("-", "_")
        spec["variables"][f"y_{name}"] = {
            "foreach": dims,
            "bounds": {"lower": -1000, "upper": 1000},
        }
        spec["constraints"][f"link_{name}"] = {
            "foreach": dims,
            "expression": f"y_{name} == {expression}",
        }
        spec["expressions"][f"probe_{name}"] = expression
    return spec


OPERATOR_DATA: dict[str, Any] = {
    "t": TT,
    "s": S,
    "season_of": pd.Series(["a", "a", "b", "b"], index=TT),
    "v": pd.Series(V, index=TT),
    "z": pd.Series([10.0, 20.0], index=S),
    "lag": pd.Series([1, 2], index=S),
    "width": pd.Series([1, 2], index=S),
}


@pytest.fixture(scope="module")
def operators_model() -> Model:
    with linopy.options as options:
        options["semantics"] = "v1"
        return solved(operator_spec(), OPERATOR_DATA, retain="all")


@pytest.mark.parametrize("key", OPERATORS)
def test_an_operator_builds_and_folds_alike(operators_model: Model, key: str) -> None:
    _, dims, expected = OPERATORS[key]
    name = key.replace("-", "_")
    want = xr.DataArray(expected, coords={dims[0]: OPERATOR_DATA[dims[0]]}, dims=dims)
    built = operators_model.solution[f"y_{name}"]
    folded = operators_model.spec.expressions[f"probe_{name}"].solution
    xr.testing.assert_allclose(built, want.rename(f"y_{name}"))
    xr.testing.assert_allclose(folded, want.rename(f"probe_{name}"))


AMOUNT_SPEC: dict[str, Any] = {
    "dimensions": {"t": {"dtype": "int"}, "g": {"dtype": "int"}},
    "lookups": {"grp": {"over": "t", "into": "g"}},
    "parameters": {"v": {"dims": ["t"]}, "lag": {"dims": ["g"], "dtype": "int"}},
    "variables": {
        "x": {"foreach": ["t"], "bounds": {"lower": 0, "upper": 100}},
        "y": {"foreach": ["t"], "bounds": {"lower": -100, "upper": 100}},
    },
    "constraints": {
        "fix": {"foreach": ["t"], "expression": "x == v"},
        "link": {
            "foreach": ["t"],
            "expression": "y == shift(x, over=t, offset=lag, edge=0, by=grp)",
        },
    },
    "objective": {"sense": "minimize", "expression": "sum(x)"},
}


def test_a_missing_shift_amount_is_refused() -> None:
    t = pd.Index([0, 1, 2], name="t")
    g = pd.Index([0, 1], name="g")
    data = {
        "t": t,
        "g": g,
        "grp": pd.Series([0, 0, 1], index=t),
        "v": pd.Series([0.0, 4.0, 5.0], index=t),
        "lag": pd.Series([1], index=g[:1]),
    }
    with pytest.raises(SpecDataError, match="parameter 'lag' is used as a coefficient"):
        Model.from_spec(AMOUNT_SPEC, data)


# ---------------------------------------------------------------------------
# grouped sum
# ---------------------------------------------------------------------------

GROUPED_SPEC: dict[str, Any] = {
    "dimensions": {"generator": {}, "bus": {"dtype": "str"}},
    "lookups": {"gen_bus": {"over": "generator", "into": "bus"}},
    "parameters": {"capacity": {"dims": ["generator"]}},
    "variables": {
        "imports": {"foreach": ["bus"], "bounds": {"lower": 0, "upper": 100}}
    },
    "constraints": {
        "import_limit": {
            "foreach": ["bus"],
            "expression": "imports <= sum(capacity, by=gen_bus)",
        }
    },
    "objective": {"sense": "maximize", "expression": "sum(imports, over=bus)"},
}
GENS = pd.Index(["g1", "g2"], name="generator")


def grouped_sources(capacity: pd.Series) -> dict[str, Any]:
    return {
        "bus": ["north", "south"],
        "generator": GENS,
        "gen_bus": pd.Series(["north", "north"], index=GENS),
        "capacity": capacity,
    }


def test_an_empty_group_on_the_constant_side_is_a_zero_and_not_a_gap() -> None:
    m = solved(GROUPED_SPEC, grouped_sources(pd.Series([3.0, 4.0], index=GENS)))
    assert m.objective.value == pytest.approx(7.0)
    assert float(m.solution["imports"].sel(bus="south")) == pytest.approx(0.0)


def test_a_lookup_that_maps_nothing_leaves_every_group_at_the_empty_sum() -> None:
    """Filtering to the mapped members leaves nothing, and nothing is what xarray will not group."""
    spec = with_(
        GROUPED_SPEC,
        variables={
            "out": {
                "foreach": ["generator"],
                "bounds": {"lower": 0, "upper": "capacity"},
            }
        },
        expressions={"per_bus": "sum(out, by=gen_bus)"},
    )
    sources = grouped_sources(pd.Series([3.0, 4.0], index=GENS))
    sources["gen_bus"] = pd.Series([], dtype=object)
    m = solved(spec, sources)

    assert m.objective.value == pytest.approx(0.0)
    per_bus = m.spec.expressions["per_bus"]
    assert isinstance(per_bus.expression, LinearExpression)
    assert per_bus.expression.nterm == 0
    assert per_bus.solution.indexes["bus"].tolist() == ["north", "south"]
    np.testing.assert_allclose(per_bus.solution.values, [0.0, 0.0])


def test_a_member_with_no_value_is_still_refused_through_a_group() -> None:
    with pytest.raises(SpecDataError, match="parameter 'capacity' covers 1 fewer"):
        Model.from_spec(GROUPED_SPEC, grouped_sources(pd.Series([3.0], index=GENS[:1])))


# ---------------------------------------------------------------------------
# where predicates
# ---------------------------------------------------------------------------

WHERE_CASES: dict[str, tuple[str, str, list[Any]]] = {
    "dimension-comparison": ("x", "t > 1", [2, 3]),
    "lookup-comparison": ("x", "season_of == 'a'", [0, 1]),
    "lookup-not-equal-skips-unmapped": ("x", "season_of != 'a'", [2]),
    "lookup-pair": ("x", "season_of != other_of", [1]),
    "lookup-defined": ("x", "season_of", [0, 1, 2]),
    "label-space-lookup": ("x", "tag == 'q'", [1]),
    "not": ("x", "NOT (t > 1)", [0, 1]),
    "and": ("x", "t > 0 AND t < 3", [1, 2]),
    "or": ("x", "t == 0 OR t == 3", [0, 3]),
    "position": ("x", "position(t) == -1", [3]),
    "position-in-groups": ("x", "position(t, by=season_of) == 0", [0, 2]),
    "bool-parameter": ("x", "flag", [0]),
    "float-parameter-must-be-finite": ("x", "cost", [0, 2]),
    "str-parameter": ("x", "label", [1, 2]),
    "parameter-comparison": ("x", "cost > 2", [2, 1]),
    "datetime-axis": ("y", "d >= '2030-01-03'", list(WHERE_DATA["d"][2:])),
}


@pytest.mark.parametrize("case", WHERE_CASES)
def test_a_where_picks_the_rows_it_names(case: str) -> None:
    variable, predicate, labels = WHERE_CASES[case]
    spec = with_(
        WHERE_SPEC,
        variables={variable: {**WHERE_SPEC["variables"][variable], "where": predicate}},
    )
    built = Model.from_spec(spec, WHERE_DATA).variables[variable]
    dim = built.dims[0]
    present = built.labels[dim][(built.labels != -1).to_numpy()]
    assert sorted(present.to_numpy().tolist()) == sorted(labels)


@pytest.mark.parametrize(
    ("predicate", "match"),
    [
        ("position(t) == 7", "names position 7 of 't', which has 4"),
        ("position(t, by=season_of) == 1", "shorter than that: \\['b'\\]"),
    ],
)
def test_a_position_no_coordinate_holds_is_refused(predicate: str, match: str) -> None:
    spec = with_(
        WHERE_SPEC,
        variables={"x": {**WHERE_SPEC["variables"]["x"], "where": predicate}},
    )
    with pytest.raises(SpecDataError, match=match):
        Model.from_spec(spec, WHERE_DATA)


# ---------------------------------------------------------------------------
# edges: partial lookups and an empty dimension beside a sum
# ---------------------------------------------------------------------------

PARTIAL_CASES: dict[str, list[float]] = {
    "sum-by": [3.0, 4.0],
    "at": [10.0, 20.0, 80.0, np.nan],
    "shift-wrap-in-groups": [2.0, 1.0, 4.0, np.nan],
    "sum-back-in-groups": [1.0, 3.0, 4.0, np.nan],
}


@pytest.mark.parametrize("key", PARTIAL_CASES)
def test_a_member_a_lookup_sends_nowhere_reaches_nothing(key: str) -> None:
    data = {**OPERATOR_DATA, "season_of": pd.Series(["a", "a", "b"], index=TT[:3])}
    m = solved(operator_spec(), data, retain="all")
    _, dims, _ = OPERATORS[key]
    name = key.replace("-", "_")
    folded = m.spec.expressions[f"probe_{name}"].solution
    want = xr.DataArray(
        PARTIAL_CASES[key], coords={dims[0]: OPERATOR_DATA[dims[0]]}, dims=dims
    )
    xr.testing.assert_allclose(folded, want.rename(folded.name))
    if dims == ["t"]:
        assert int(m.constraints[f"link_{name}"].labels.sel(t=3)) == -1


def test_a_sum_beside_an_empty_dimension_is_the_empty_sum() -> None:
    spec: dict[str, Any] = {
        "dimensions": {"t": {"dtype": "int"}, "s": {"dtype": "str"}},
        "variables": {"x": {"foreach": ["t", "s"], "bounds": {"lower": 0, "upper": 1}}},
        "constraints": {"cap": {"foreach": ["s"], "expression": "sum(x, over=t) <= 1"}},
        "objective": {"sense": "maximize", "expression": "sum(x)"},
    }
    m = Model.from_spec(spec, {"t": [0, 1], "s": pd.Index([], name="s", dtype=object)})
    assert "cap" not in m.constraints


def test_a_window_width_no_member_carries_is_a_window_of_nothing() -> None:
    data = {
        **OPERATOR_DATA,
        "season_of": pd.Series(
            [], index=pd.Index([], name="t", dtype=int), dtype=object
        ),
    }
    m = solved(operator_spec(), data, retain="all")
    folded = m.spec.expressions["probe_sum_back_group_width"].solution
    assert bool(folded.isnull().all())
