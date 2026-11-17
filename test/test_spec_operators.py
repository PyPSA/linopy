"""
Operators built as a constraint and folded as a named expression: sum,
grouped sum, ``at``, shift/translate, sum_back windows, and the where/mask
predicates that gate which rows exist.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("mathspec")
yaml = pytest.importorskip("yaml")
pytest.importorskip("highspy")

from spec_helpers import (  # noqa: E402
    TT,
    WHERE_DATA,
    WHERE_SPEC,
    S,
    solved,
    table,
    with_,
)

from linopy import (  # noqa: E402
    LinearExpression,
    Model,
    PerformanceWarning,
    QuadraticExpression,
)
from linopy.spec import SpecDataError, evaluate  # noqa: E402
from linopy.spec import evaluator as spec_evaluator  # noqa: E402
from linopy.spec import operators as spec_operators  # noqa: E402
from linopy.testing import (  # noqa: E402
    assert_conequal,
    assert_equivalent,
    assert_linequal,
    assert_quadequal,
)

# ---------------------------------------------------------------------------
# operators, built as a constraint and folded as a named expression
# ---------------------------------------------------------------------------

V = np.array([1.0, 2.0, 4.0, 8.0])
OPERATORS: dict[str, tuple[str, list[str], list[float]]] = {
    "shift-edge-0": ("shift(x, along=t, offset=1, edge=0)", ["t"], [0, 1, 2, 4]),
    "shift-ahead-edge-0": ("shift(x, along=t, offset=-1, edge=0)", ["t"], [2, 4, 8, 0]),
    "shift-wrap": ("shift(x, along=t, offset=1, edge='wrap')", ["t"], [8, 1, 2, 4]),
    "shift-wrap-in-groups": (
        "shift(x, along=t, offset=1, edge='wrap', within=season_of[s])",
        ["t"],
        [2, 1, 8, 4],
    ),
    "shift-by-group-offset": (
        "shift(x, along=t, offset=lag, edge=0, within=season_of[s])",
        ["t"],
        [0, 1, 0, 0],
    ),
    "sum-back": ("sum_back(x, along=t, window=2)", ["t"], [1, 3, 6, 12]),
    "sum-back-wrap": (
        "sum_back(x, along=t, window=2, edge='wrap')",
        ["t"],
        [9, 3, 6, 12],
    ),
    "sum-back-in-groups": (
        "sum_back(x, along=t, window=2, within=season_of[s])",
        ["t"],
        [1, 3, 4, 12],
    ),
    "sum-back-group-width": (
        "sum_back(x, along=t, window=width, within=season_of[s])",
        ["t"],
        [1, 2, 4, 12],
    ),
    "sum-by": ("sum(x, over=t, by=season_of[s])", ["s"], [3, 12]),
    "at": ("x * at(z, by=season_of[s])", ["t"], [10, 20, 80, 160]),
    "cases": ("x_state", ["t"], [100, 1, 2, 4]),
}


def operator_spec() -> dict[str, Any]:
    spec: dict[str, Any] = {
        "dimensions": {"t": {"dtype": "int", "ordered": True}, "s": {"dtype": "str"}},
        "relations": {"season_of": {"key": "t", "values": "s", "missing": "absent"}},
        "parameters": {
            "v": {"dims": ["t"]},
            "z": {"dims": ["s"]},
            "lag": {"dims": ["s"], "dtype": "int"},
            "width": {"dims": ["s"], "dtype": "int"},
        },
        "variables": {"x": {"dims": ["t"], "bounds": {"lower": 0, "upper": 100}}},
        "constraints": {"fix": {"dims": ["t"], "expression": "x == v"}},
        "expressions": {
            "x_state": {
                "dims": ["t"],
                "cases": {"first": {"when": "position(t) == 0", "expression": 100}},
                "otherwise": "shift(x, along=t, offset=1)",
            }
        },
        "objective": {"sense": "minimize", "expression": "sum(x)"},
    }
    for key, (expression, dims, _) in OPERATORS.items():
        name = key.replace("-", "_")
        spec["variables"][f"y_{name}"] = {
            "dims": dims,
            "bounds": {"lower": -1000, "upper": 1000},
        }
        spec["constraints"][f"link_{name}"] = {
            "dims": dims,
            "expression": f"y_{name} == {expression}",
        }
        spec["expressions"][f"probe_{name}"] = expression
    return spec


OPERATOR_DATA: dict[str, Any] = {
    "t": TT,
    "s": S,
    "season_of": pd.DataFrame({"t": TT, "s": ["a", "a", "b", "b"]}),
    "v": pd.DataFrame({"t": TT, "v": V}),
    "z": pd.DataFrame({"s": S, "z": [10.0, 20.0]}),
    "lag": pd.DataFrame({"s": S, "lag": [1, 2]}),
    "width": pd.DataFrame({"s": S, "width": [1, 2]}),
}


@pytest.fixture(scope="module")
def operators_model() -> Model:
    return solved(operator_spec(), OPERATOR_DATA)


@pytest.mark.parametrize("key", OPERATORS)
def test_an_operator_builds_and_folds_alike(operators_model: Model, key: str) -> None:
    _, dims, expected = OPERATORS[key]
    name = key.replace("-", "_")
    want = xr.DataArray(expected, coords={dims[0]: OPERATOR_DATA[dims[0]]}, dims=dims)
    built = operators_model.solution[f"y_{name}"]
    folded = evaluate(operator_spec(), operators_model, f"probe_{name}")[
        f"probe_{name}"
    ]
    xr.testing.assert_allclose(built, want.rename(f"y_{name}"))
    xr.testing.assert_allclose(folded, want.rename(f"probe_{name}"))


HOURS = pd.date_range("2030-01-01", periods=12, freq="h", name="t")
SEASON = ["a", "b", "a", "a", "c", "b", "a", "a", "b", "a", "a", "a"]
HOURLY = pd.DataFrame({"s": SEASON, "v": np.arange(1.0, 13.0)}, index=HOURS)
EDGES = {"none": "", "fill": ", edge=0", "wrap": ", edge='wrap'"}


def hourly_shift(offset: int, edge: str, within: str) -> dict[str, Any]:
    shifted = f"shift(x, along=t, offset={offset}{EDGES[edge]}{within})"
    return {
        "dimensions": {
            "t": {"dtype": "datetime", "ordered": True},
            "s": {"dtype": "str"},
        },
        "relations": {"season_of": {"key": "t", "values": "s"}},
        "parameters": {"v": {"dims": ["t"]}},
        "variables": {
            "x": {"dims": ["t"], "bounds": {"lower": 0, "upper": 100}},
            "y": {"dims": ["t"], "bounds": {"lower": -100, "upper": 100}},
        },
        "constraints": {
            "fix": {"dims": ["t"], "expression": "x == v"},
            "link": {"dims": ["t"], "expression": f"y == {shifted}"},
        },
        "objective": {"sense": "minimize", "expression": "sum(x)"},
    }


def pandas_shift(
    values: pd.api.typing.SeriesGroupBy, offset: int, edge: str
) -> pd.Series:
    if edge == "wrap":
        return values.transform(lambda group: np.roll(group.to_numpy(), offset))
    return values.shift(offset, fill_value=0.0 if edge == "fill" else None)


@pytest.mark.parametrize("grouped", [True, False], ids=["in-groups", "whole-axis"])
@pytest.mark.parametrize("edge", EDGES)
@pytest.mark.parametrize("offset", [-7, -1, 0, 1, 7])
def test_a_shift_on_a_datetime_axis_matches_a_pandas_shift(
    offset: int, edge: str, grouped: bool
) -> None:
    within = ", within=season_of[s]" if grouped else ""
    sources = {
        "t": HOURS,
        "s": ["a", "b", "c"],
        "season_of": HOURLY.s.rename_axis("t").reset_index(),
        "v": HOURLY.v.rename_axis("t").reset_index(),
    }
    m = solved(hourly_shift(offset, edge, within), sources)
    groups = HOURLY.s if grouped else np.zeros(len(HOURLY))
    want = pandas_shift(HOURLY.groupby(groups).v, offset, edge)
    rows = m.constraints["link"].labels != -1
    np.testing.assert_allclose(m.solution["y"].where(rows), want, equal_nan=True)


AMOUNT_SPEC: dict[str, Any] = {
    "dimensions": {"t": {"dtype": "int", "ordered": True}, "g": {"dtype": "int"}},
    "relations": {"grp": {"key": "t", "values": "g", "missing": "absent"}},
    "parameters": {"v": {"dims": ["t"]}, "lag": {"dims": ["g"], "dtype": "int"}},
    "variables": {
        "x": {"dims": ["t"], "bounds": {"lower": 0, "upper": 100}},
        "y": {"dims": ["t"], "bounds": {"lower": -100, "upper": 100}},
    },
    "constraints": {
        "fix": {"dims": ["t"], "expression": "x == v"},
        "link": {
            "dims": ["t"],
            "expression": "y == shift(x, along=t, offset=lag, edge=0, within=grp[g])",
        },
    },
    "objective": {"sense": "minimize", "expression": "sum(x)"},
}


@pytest.mark.parametrize(
    ("where", "refused"),
    [({}, True), ({"where": "t < 2"}, False)],
    ids=["read", "masked-out"],
)
def test_a_missing_shift_amount_is_refused_where_a_live_row_reads_it(
    where: dict[str, str], refused: bool
) -> None:
    t = pd.Index([0, 1, 2], name="t")
    g = pd.Index([0, 1], name="g")
    data = {
        "t": t,
        "g": g,
        "grp": pd.DataFrame({"t": t, "g": [0, 0, 1]}),
        "v": pd.DataFrame({"t": t, "v": [0.0, 4.0, 5.0]}),
        "lag": pd.DataFrame({"g": g[:1], "lag": [1]}),
    }
    link = {**AMOUNT_SPEC["constraints"]["link"], **where}
    spec = with_(AMOUNT_SPEC, constraints={"link": link})
    if refused:
        with pytest.raises(
            SpecDataError, match="constraint 'link' reads parameter 'lag' at g=1, where"
        ):
            Model.from_spec(spec, data)
    else:
        m = Model.from_spec(spec, data)
        assert (m.constraints["link"].labels != -1).values.tolist() == [
            True,
            True,
            False,
        ]


WINDOW_SPEC: dict[str, Any] = {
    "dimensions": {"t": {"dtype": "int", "ordered": True}},
    "parameters": {"v": {"dims": ["t"]}},
    "variables": {"x": {"dims": ["t"], "bounds": {"lower": 0, "upper": 100}}},
    "constraints": {"fix": {"dims": ["t"], "expression": "x == v"}},
    "objective": {"sense": "minimize", "expression": "sum(x)"},
}
WINDOW_DATA: dict[str, Any] = {"t": TT, "v": pd.DataFrame({"t": TT, "v": V})}


def window(expression: str) -> dict[str, Any]:
    return with_(WINDOW_SPEC, expressions={"recent": expression})


def test_a_trailing_sum_wider_than_its_axis_is_refused() -> None:
    with pytest.raises(
        SpecDataError, match=r"asks for a window of 5 position\(s\) where 't' holds 4"
    ):
        Model.from_spec(window("sum_back(x, along=t, window=5)"), WINDOW_DATA)


def test_a_trailing_sum_wider_than_its_axis_wraps_onto_the_whole_axis() -> None:
    spec = window("sum_back(x, along=t, window=5, edge='wrap')")
    want = xr.DataArray([V.sum()] * len(TT), coords={"t": TT}, name="recent")
    xr.testing.assert_allclose(
        evaluate(spec, solved(spec, WINDOW_DATA), "recent")["recent"], want
    )


# ---------------------------------------------------------------------------
# grouped sum
# ---------------------------------------------------------------------------

GROUPED_SPEC: dict[str, Any] = {
    "dimensions": {"generator": {}, "bus": {"dtype": "str"}},
    "relations": {
        "gen_bus": {"key": "generator", "values": "bus", "missing": "absent"}
    },
    "parameters": {"capacity": {"dims": ["generator"]}},
    "variables": {"imports": {"dims": ["bus"], "bounds": {"lower": 0, "upper": 100}}},
    "constraints": {
        "import_limit": {
            "dims": ["bus"],
            "expression": "imports <= sum(capacity, over=generator, by=gen_bus[bus])",
        }
    },
    "objective": {"sense": "maximize", "expression": "sum(imports, over=bus)"},
}
GENS = pd.Index(["g1", "g2"], name="generator")


def grouped_sources(capacity: pd.Series) -> dict[str, Any]:
    return {
        "bus": ["north", "south"],
        "generator": GENS,
        "gen_bus": pd.DataFrame({"generator": GENS, "bus": ["north", "north"]}),
        "capacity": table(capacity, "capacity"),
    }


def test_an_empty_group_on_the_constant_side_is_a_zero_and_not_a_gap() -> None:
    m = solved(GROUPED_SPEC, grouped_sources(pd.Series([3.0, 4.0], index=GENS)))
    assert m.objective.value == pytest.approx(7.0)
    assert float(m.solution["imports"].sel(bus="south")) == pytest.approx(0.0)


def test_a_relation_that_maps_nothing_leaves_every_group_at_the_empty_sum() -> None:
    """Filtering to the mapped members leaves nothing, and nothing is what xarray will not group."""
    spec = with_(
        GROUPED_SPEC,
        variables={
            "out": {
                "dims": ["generator"],
                "bounds": {"lower": 0, "upper": "capacity"},
            }
        },
        expressions={"per_bus": "sum(out, over=generator, by=gen_bus[bus])"},
    )
    sources = grouped_sources(pd.Series([3.0, 4.0], index=GENS))
    sources["gen_bus"] = pd.DataFrame({"generator": [], "bus": []}, dtype=object)
    m = solved(spec, sources)

    assert m.objective.value == pytest.approx(0.0)
    assert isinstance(m.expressions["per_bus"], LinearExpression)
    assert m.expressions["per_bus"].nterm == 0
    per_bus = evaluate(spec, m, "per_bus")["per_bus"]
    assert per_bus.indexes["bus"].tolist() == ["north", "south"]
    np.testing.assert_allclose(per_bus.values, [0.0, 0.0])


def test_a_quadratic_term_sums_by_group_and_stays_quadratic() -> None:
    spec = {
        "dimensions": GROUPED_SPEC["dimensions"],
        "relations": GROUPED_SPEC["relations"],
        "variables": {"p": {"dims": ["generator"], "bounds": {"lower": 1, "upper": 2}}},
        "objective": {
            "sense": "minimize",
            "expression": "sum(sum(p * p, over=generator, by=gen_bus[bus]), over=bus)",
        },
        "expressions": {"per_bus": "sum(p * p, over=generator, by=gen_bus[bus])"},
    }
    sources = grouped_sources(pd.Series(dtype=float))
    del sources["capacity"]
    m = solved(spec, sources)

    assert m.is_quadratic
    assert m.objective.value == pytest.approx(2.0)
    assert isinstance(m.expressions["per_bus"], QuadraticExpression)
    per_bus = evaluate(spec, m, "per_bus")["per_bus"]
    np.testing.assert_allclose(per_bus.values, [2.0, 0.0])


@pytest.mark.parametrize(
    ("where", "refused"),
    [({}, True), ({"where": "bus == 'south'"}, False)],
    ids=["read", "group-masked-out"],
)
def test_a_member_with_no_value_is_refused_where_its_group_is_read(
    where: dict[str, str], refused: bool
) -> None:
    limit = {**GROUPED_SPEC["constraints"]["import_limit"], **where}
    spec = with_(GROUPED_SPEC, constraints={"import_limit": limit})
    sources = grouped_sources(pd.Series([3.0], index=GENS[:1]))
    if refused:
        with pytest.raises(
            SpecDataError, match="parameter 'capacity' at generator='g2'"
        ):
            Model.from_spec(spec, sources)
    else:
        m = Model.from_spec(spec, sources)
        assert (m.constraints["import_limit"].labels != -1).values.tolist() == [
            False,
            True,
        ]


def test_a_sum_over_several_axes_collapses_each() -> None:
    spec = {
        "dimensions": {"t": {"dtype": "int"}, "s": {"dtype": "str"}},
        "variables": {"x": {"dims": ["t", "s"], "bounds": {"lower": 0, "upper": 1}}},
        "constraints": {"cap": {"dims": [], "expression": "sum(x, over=[t, s]) <= 3"}},
        "objective": {"sense": "maximize", "expression": "sum(x, over=[s, t])"},
        "expressions": {"total": "sum(x, over=[t, s])"},
    }
    m = solved(spec, {"t": TT, "s": S})
    assert m.constraints["cap"].lhs.nterm == len(TT) * len(S)
    assert float(evaluate(spec, m, "total")["total"]) == pytest.approx(3.0)


SELF_MAP_SPEC: dict[str, Any] = {
    "dimensions": {"t": {"dtype": "int", "ordered": True}},
    "relations": {"next_of": {"key": "t", "values": {"next": "t"}}},
    "parameters": {"v": {"dims": ["t"]}},
    "variables": {
        "x": {"dims": ["t"], "bounds": {"lower": 0, "upper": 10}},
        "y": {"dims": ["t"], "bounds": {"lower": 0, "upper": 100}},
        "a": {"dims": ["t"], "bounds": {"lower": 0, "upper": 100}},
    },
    "constraints": {
        "fix": {"dims": ["t"], "expression": "x == v"},
        "link": {"dims": ["t"], "expression": "y == sum(x, over=t, by=next_of[next])"},
        "ahead": {"dims": ["t"], "expression": "a == at(x, by=next_of[next])"},
    },
    "objective": {"sense": "minimize", "expression": "sum(y)"},
    "expressions": {
        "terms": "sum(x, over=t, by=next_of[next])",
        "data": "sum(v, over=t, by=next_of[next])",
        "ahead_terms": "at(x, by=next_of[next])",
        "ahead_data": "at(v, by=next_of[next])",
    },
}


@pytest.mark.parametrize(
    ("built", "folds", "want"),
    [
        ("y", ["terms", "data"], [0.0, 1.0, 6.0]),
        ("a", ["ahead_terms", "ahead_data"], [2.0, 4.0, 4.0]),
    ],
)
def test_a_relation_into_its_own_dimension_joins_on_it(
    backend: Model, built: str, folds: list[str], want: list[float]
) -> None:
    t = TT[:3]
    data = {
        "t": t,
        "v": pd.DataFrame({"t": t, "v": V[:3]}),
        "next_of": pd.DataFrame({"t": t, "next": [1, 2, 2]}),
    }
    m = Model.from_spec(SELF_MAP_SPEC, data, sparse=backend.sparse)
    m.solve(solver_name="highs", output_flag=False)
    assert m.solution[built].values.tolist() == pytest.approx(want)
    for folded in evaluate(SELF_MAP_SPEC, m, *folds, sources=data).values():
        assert folded.values.tolist() == pytest.approx(want)


E = pd.Index(["e1", "e2", "e3"], name="e")
RELATION_SUMS: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "into-another-dim": (
        with_(
            GROUPED_SPEC,
            variables={"out": {"dims": ["generator"], "bounds": {"lower": 0}}},
            expressions={"summed": "sum(out, over=generator, by=gen_bus[bus])"},
        ),
        {
            **grouped_sources(pd.Series([3.0, 4.0], index=GENS)),
            "gen_bus": pd.DataFrame({"generator": GENS, "bus": ["south", "south"]}),
        },
    ),
    "into-its-own-dim": (
        {
            **SELF_MAP_SPEC,
            "constraints": {},
            "expressions": {"summed": "sum(x, over=t, by=next_of[next])"},
        },
        {
            "t": TT,
            "v": pd.DataFrame({"t": TT, "v": V}),
            "next_of": pd.DataFrame({"t": TT, "next": [1, 2, 2, 0]}),
        },
    ),
    "beside-a-relation-that-maps-nothing": (
        with_(
            GROUPED_SPEC,
            variables={"out": {"dims": ["generator"], "bounds": {"lower": 0}}},
            expressions={
                "summed": "imports + sum(out, over=generator, by=gen_bus[bus])"
            },
        ),
        {
            **grouped_sources(pd.Series([3.0, 4.0], index=GENS)),
            "gen_bus": pd.DataFrame({"generator": [], "bus": []}, dtype=object),
        },
    ),
    **{
        name: (
            {
                "dimensions": {
                    "t": {"dtype": "int", "ordered": True},
                    "s": {"dtype": "str"},
                    "e": {"dtype": "str"},
                },
                "relations": {
                    "season_of": {"key": "t", "values": "s", "missing": "absent"}
                },
                "parameters": {"lag": {"dims": ["e"], "dtype": "int"}},
                "variables": {"x": {"dims": ["t", "e"]}},
                "expressions": {
                    "summed": f"x + shift(2 * x, along=t, offset=lag, {edge})"
                },
            },
            {
                "t": TT,
                "s": S,
                "e": E,
                "season_of": pd.DataFrame({"t": TT, "s": ["a", "a", "b", "b"]}),
                "lag": pd.DataFrame({"e": E, "lag": [0, 1, 1]}),
            },
        )
        for name, edge in {
            "shift-by-an-entity-offset": "edge=0",
            "shift-in-groups-by-an-entity-offset": "edge='wrap', within=season_of[s]",
        }.items()
    },
    "at-of-a-term": (
        with_(
            GROUPED_SPEC,
            variables={"out": {"dims": ["generator"], "bounds": {"lower": 0}}},
            expressions={"summed": "out + at(2 * imports, by=gen_bus[bus])"},
        ),
        grouped_sources(pd.Series([3.0, 4.0], index=GENS)),
    ),
    "at-along-its-own-key": (
        {
            "dimensions": {
                "t": {"dtype": "int", "ordered": True},
                "e": {"dtype": "str"},
            },
            "relations": {
                "source_of": {
                    "key": ["e", "t"],
                    "values": {"source": "t"},
                    "missing": "absent",
                }
            },
            "variables": {"x": {"dims": ["e", "t"]}},
            "expressions": {
                "summed": "at(shift(2 * x, along=t, offset=1), by=source_of[source])"
            },
        },
        {
            "t": TT,
            "e": E,
            "source_of": pd.DataFrame(
                {"e": ["e1", "e1", "e3"], "t": [1, 3, 0], "source": [0, 2, 3]}
            ),
        },
    ),
    "cases-beside-data-over-more-dimensions": (
        {
            "dimensions": {
                "t": {"dtype": "int", "ordered": True},
                "e": {"dtype": "str"},
            },
            "parameters": {
                "lag": {"dims": ["e"], "dtype": "int"},
                "v": {"dims": ["t"]},
            },
            "variables": {"x": {"dims": ["e"]}},
            "expressions": {
                "summed": {
                    "dims": ["e", "t"],
                    "cases": {"lagged": {"when": "lag > 0", "expression": "x"}},
                    "otherwise": "v",
                }
            },
        },
        {
            "t": TT,
            "e": E,
            "lag": pd.DataFrame({"e": E, "lag": [0, 1, 1]}),
            "v": pd.DataFrame({"t": TT, "v": V}),
        },
    ),
}


@pytest.mark.parametrize(
    ("spec", "data"), RELATION_SUMS.values(), ids=list(RELATION_SUMS)
)
def test_a_relation_sum_keeps_the_backing_and_matches_dense(
    backend: Model, spec: dict[str, Any], data: dict[str, Any]
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        m = Model.from_spec(spec, data, sparse=backend.sparse)
    summed = m.expressions["summed"]
    assert summed.is_sparse == backend.sparse
    assert_linequal(summed, Model.from_spec(spec, data).expressions["summed"])


def test_a_rhs_over_a_dimension_the_term_lacks_keeps_the_backing(
    backend: Model,
) -> None:
    """A sparse term met a rhs over another dimension by densifying."""
    spec = {
        "dimensions": {"t": {"dtype": "int", "ordered": True}, "e": {"dtype": "str"}},
        "parameters": {"cap": {"dims": ["t", "e"]}},
        "variables": {"x": {"dims": ["t"]}},
        "constraints": {"limit": {"dims": ["t", "e"], "expression": "2 * x <= cap"}},
    }
    cap = xr.DataArray(np.arange(12.0).reshape(4, 3), coords=[TT, E], name="cap")
    data = {"t": TT, "e": E, "cap": cap}
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        m = Model.from_spec(spec, data, sparse=backend.sparse)
    limit = m.constraints["limit"]
    assert limit.lhs.is_sparse == backend.sparse
    assert_conequal(limit, Model.from_spec(spec, data).constraints["limit"])


# ---------------------------------------------------------------------------
# a chain of sums, merged in one step
# ---------------------------------------------------------------------------

CHAIN = "(x + a) * 1 + (y - b) * 1 - a - (z + c) * 1 + x - y"
STEPS = {
    "s1": "(x + a) * 1 + (y - b) * 1",
    "s2": "s1 - a",
    "s3": "s2 - (z + c) * 1",
    "s4": "s3 + x",
}
CHAIN_SPEC: dict[str, Any] = {
    "dimensions": {"t": {"dtype": "int", "ordered": True}, "s": {"dtype": "str"}},
    "parameters": {
        "a": {"dims": ["t"]},
        "b": {"dims": ["s", "t"]},
        "c": {"dims": ["s"]},
    },
    "variables": {
        "x": {"dims": ["t"]},
        "y": {"dims": ["s", "t"], "where": "b < 0"},
        "z": {"dims": ["s"]},
    },
    "expressions": {
        "chained": {"dims": ["s", "t"], "expression": CHAIN},
        **{name: {"dims": ["s", "t"], "expression": e} for name, e in STEPS.items()},
        "stepwise": {"dims": ["s", "t"], "expression": "s4 - y"},
    },
}
CHAIN_DATA = {
    "t": TT,
    "s": S,
    "a": xr.DataArray([0.1, 1e16, 0.7, -3.3], coords=[TT]),
    "b": xr.DataArray([[-0.2, 1.0, -0.3, -1.1], [-0.6, -1e-3, 2.0, -0.3]], [S, TT]),
    "c": xr.DataArray([-0.3, 5.1], coords=[S]),
}


def spied(monkeypatch: pytest.MonkeyPatch, owner: Any, name: str) -> list[Any]:
    """What each call of *owner*'s function *name* returns, recorded as it is called."""
    made: list[Any] = []
    function = getattr(owner, name)

    def recorded(*args: Any, **kwargs: Any) -> Any:
        made.append(function(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(owner, name, recorded)
    return made


def test_a_sum_chain_merges_its_terms_in_one_step(
    backend: Model, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each of ``s1`` to ``s4`` is a reference, so ``stepwise`` adds one operand at a time."""
    chains = spied(monkeypatch, spec_evaluator, "_added")
    m = Model.from_spec(CHAIN_SPEC, CHAIN_DATA, sparse=backend.sparse)
    chained, stepwise = m.expressions["chained"], m.expressions["stepwise"]
    assert chains
    assert chained.is_sparse == backend.sparse
    assert_linequal(chained, stepwise)
    np.testing.assert_array_equal(chained.const.values, stepwise.const.values)


# ---------------------------------------------------------------------------
# a sum of a term scaled by sparse data, built from its live cells
# ---------------------------------------------------------------------------

SITES = pd.Index(["a", "b", "c"], name="s")
BUSES = pd.Index(["u", "v"], name="b")
SPARSE_SPEC: dict[str, Any] = {
    "dimensions": {
        "t": {"dtype": "int", "ordered": True},
        "s": {"dtype": "str"},
        "b": {"dtype": "str"},
    },
    "relations": {"bus_of": {"key": "s", "values": "b"}},
    "parameters": {
        "p": {"dims": ["s", "t"], "missing": "absent"},
        "q": {"dims": ["t"]},
        "c": {"dims": ["s"]},
    },
    "variables": {
        "x": {"dims": ["s", "t"], "where": "c > -1", "bounds": {"lower": 0}},
        "y": {"dims": ["t"], "bounds": {"lower": 0, "upper": 1}},
        "z": {"dims": ["s"], "bounds": {"lower": 0, "upper": 1}},
    },
}
SPARSE_DATA = {
    "t": TT,
    "s": SITES,
    "b": BUSES,
    "bus_of": pd.DataFrame({"s": SITES, "b": ["u", "v", "u"]}),
    "p": xr.DataArray(
        [[0, 1.5, 0, 0], [np.nan, 0, -0.3, 0], [0, 0, 0.7, np.nan]], [SITES, TT]
    ),
    "q": xr.DataArray([1.0, 3.0, 0.5, 7.0], coords=[TT]),
    "c": xr.DataArray([0.1, -3.0, 0.2], coords=[SITES]),
}
SCALED_SUMS: dict[str, tuple[str, list[str], str]] = {
    "over-one-dim": ("x * p", ["s"], "sum({}, over=t)"),
    "over-every-dim": ("x * p", [], "sum({}, over=[s, t])"),
    "data-first": ("p * x", ["t"], "sum({}, over=s)"),
    "scaled-twice": ("q * (x * p) / 2", ["s"], "sum({}, over=t)"),
    "divided": ("x * p / q", ["s"], "sum({}, over=t)"),
    "broadcast-term": ("y * p", ["s"], "sum({}, over=t)"),
    "term-with-constant": ("(x - c + 0.1) * p", ["t"], "sum({}, over=s)"),
    "through-a-relation": ("x * p", ["b", "t"], "sum({}, over=s, by=bus_of[b])"),
    "constant-through-a-relation": (
        "(x + c) * p",
        ["b", "t"],
        "sum({}, over=s, by=bus_of[b])",
    ),
    "broadcast-through-a-relation": (
        "z * p",
        ["b"],
        "sum({}, over=[s, t], by=bus_of[b])",
    ),
}


def scaled_sum_spec(scaled: str, dims: list[str], summed: str) -> dict[str, Any]:
    """``fused`` sums *scaled* in place; ``general`` sums it through the reference ``scaled``, operator by operator."""
    return with_(
        SPARSE_SPEC,
        expressions={
            "scaled": {"dims": ["s", "t"], "expression": scaled},
            "fused": {"dims": dims, "expression": summed.format(scaled)},
            "general": {"dims": dims, "expression": summed.format("scaled")},
        },
    )


@pytest.mark.parametrize("case", SCALED_SUMS)
def test_a_sum_of_a_term_scaled_by_sparse_data_is_built_from_its_live_cells(
    backend: Model, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """The zero coefficients the operators keep are left out; the rest is the same, the constant to the last bit."""
    fused_sums = spied(monkeypatch, spec_operators, "weighted_sum")
    spec = scaled_sum_spec(*SCALED_SUMS[case])
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        m = Model.from_spec(spec, SPARSE_DATA, sparse=backend.sparse)
    fused, general = m.expressions["fused"], m.expressions["general"]
    assert isinstance(fused, LinearExpression)
    assert isinstance(general, LinearExpression)
    assert any(made is not None for made in fused_sums)
    assert fused.is_sparse == backend.sparse
    assert_equivalent(fused, general)
    np.testing.assert_array_equal(fused.const.values, general.const.values)


def minimized(expression: str) -> dict[str, Any]:
    return {"objective": {"sense": "minimize", "expression": expression}}


ZEROED_ENTRIES: dict[str, Callable[[str], dict[str, Any]]] = {
    "emptied-row": lambda scaled: {
        "constraints": {
            "row": {"dims": ["t"], "expression": f"sum({scaled}, over=s) <= 5"}
        }
    },
    "objective": lambda scaled: minimized(f"sum({scaled})"),
    "factor": lambda scaled: minimized(f"sum(y * sum({scaled}, over=s))"),
}


@pytest.mark.parametrize("entry", ZEROED_ENTRIES)
def test_a_sum_whose_live_cells_leave_out_every_term_is_built_operator_by_operator(
    backend: Model, entry: str
) -> None:
    """A term with a zero coefficient still counts as a term of its row, its objective and its factor."""
    zeroed = {**SPARSE_DATA, "p": xr.zeros_like(SPARSE_DATA["p"])}

    def built(scaled: str) -> Model:
        spec = with_(
            SPARSE_SPEC,
            expressions={"scaled": {"dims": ["s", "t"], "expression": "x * p"}},
            **ZEROED_ENTRIES[entry](scaled),
        )
        return Model.from_spec(spec, zeroed, sparse=backend.sparse)

    fused, general = built("x * p"), built("scaled")
    for name in general.constraints:
        assert_conequal(fused.constraints[name], general.constraints[name])
    assert_linequal(fused.objective.expression, general.objective.expression)


READERS: dict[str, dict[str, Any]] = {
    "two-constraints": {
        "constraints": {
            "r1": {"dims": ["t"], "expression": "e <= 5"},
            "r2": {"dims": ["t"], "expression": "e <= 6"},
        }
    },
    "a-live-reader-first": {
        "constraints": {
            "r1": {"dims": ["t"], "expression": "e + y <= 5"},
            "r2": {"dims": ["t"], "expression": "e <= 6"},
        }
    },
    "a-constraint-and-the-objective": {
        "constraints": {"r1": {"dims": ["t"], "expression": "e + y <= 5"}},
        **minimized("sum(e)"),
    },
}


@pytest.mark.parametrize("readers", READERS)
def test_every_reader_of_a_named_sum_the_data_empties_sees_its_terms(
    backend: Model, readers: str
) -> None:
    """The first reader to find an emptied row builds the named expression again, operator by operator, for every reader."""
    zeroed = {**SPARSE_DATA, "p": xr.zeros_like(SPARSE_DATA["p"])}

    def built(scaled: str) -> Model:
        spec = with_(
            SPARSE_SPEC,
            expressions={
                "e": {"dims": ["t"], "expression": f"sum({scaled}, over=s)"},
                "scaled": {"dims": ["s", "t"], "expression": "x * p"},
            },
            **READERS[readers],
        )
        return Model.from_spec(spec, zeroed, sparse=backend.sparse)

    fused, general = built("x * p"), built("scaled")
    for name in general.constraints:
        got, want = fused.constraints[name], general.constraints[name]
        xr.testing.assert_equal(got.labels, want.labels)
        xr.testing.assert_equal(got.rhs, want.rhs)
        assert_equivalent(got.lhs, want.lhs)
    if "objective" in READERS[readers]:
        got_objective = fused.objective.expression
        want_objective = general.objective.expression
        assert isinstance(got_objective, LinearExpression)
        assert isinstance(want_objective, LinearExpression)
        assert_equivalent(got_objective, want_objective)


@pytest.mark.parametrize(
    "product",
    [
        pytest.param("y * {}", id="by-a-variable"),
        pytest.param("{} * {}", id="of-two-sums"),
        pytest.param("{} ** 2", id="squared"),
    ],
)
def test_a_product_of_a_fast_sum_is_what_the_operators_make(product: str) -> None:
    """A product of terms is built on the slots they hold, which a fast sum leaves out."""

    def made(scaled: str) -> Any:
        summed = f"sum({scaled}, over=s)"
        spec = with_(
            SPARSE_SPEC,
            expressions={
                "e": {"dims": ["t"], "expression": product.format(summed, summed)},
                "scaled": {"dims": ["s", "t"], "expression": "x * p"},
            },
        )
        try:
            return Model.from_spec(spec, SPARSE_DATA).expressions["e"]
        except TypeError as error:
            return str(error)

    fused, general = made("x * p"), made("scaled")
    if isinstance(general, str):
        assert fused == general
    else:
        assert_quadequal(fused, general)


AGGREGATES: dict[str, tuple[str, list[str]]] = {
    "variable": ("sum(x, over=s, by=bus_of[b])", ["b", "t"]),
    "term-with-constant": ("sum(x + c, over=s, by=bus_of[b])", ["b", "t"]),
    "beside-a-plain-sum": ("sum(x - c, over=[s, t], by=bus_of[b])", ["b"]),
    "joined-behind-a-dim": ("sum(x + c, over=t, by=half_of[h])", ["h", "s"]),
}


@pytest.mark.parametrize("case", AGGREGATES)
def test_a_sparse_term_summed_through_a_relation_is_one_aggregate(
    monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Rows of the relation land in their groups out of order, beside masked cells; the dense sum gathers them."""
    sums = spied(monkeypatch, spec_operators, "grouped_sum")
    aggregated = spied(monkeypatch, spec_operators, "_aggregated")
    summed, dims = AGGREGATES[case]
    spec = with_(
        SPARSE_SPEC,
        dimensions={"h": {"dtype": "str"}},
        relations={"half_of": {"key": "t", "values": "h"}},
        expressions={"summed": {"dims": dims, "expression": summed}},
    )
    halves = pd.DataFrame({"t": TT, "h": ["late", "early", "late", "early"]})
    data = {**SPARSE_DATA, "h": ["early", "late"], "half_of": halves}
    with warnings.catch_warnings():
        warnings.simplefilter("error", PerformanceWarning)
        Model.from_spec(spec, data, sparse=True)
    assert aggregated
    sparse = sums.pop()
    Model.from_spec(spec, data)
    dense = sums.pop()
    assert sparse.coord_dims == dense.coord_dims
    assert_linequal(sparse, dense)
    np.testing.assert_array_equal(sparse.const.values, dense.const.values)


# ---------------------------------------------------------------------------
# relations of several columns: bare, multi-key and multi-value
# ---------------------------------------------------------------------------

K = pd.Index(["p", "q"], name="k")
RELATIONS: dict[str, dict[str, Any]] = {
    "cover": {"key": {"start": "t", "covered": "t"}, "missing": "absent"},
    "ends": {"key": "t", "values": {"from": "s", "to": "s"}},
    "slot": {"key": "t", "values": ["s", "k"]},
    "zone_of": {"key": ["t", "k"], "values": "s"},
}
RELATION_DATA: dict[str, Any] = {
    "t": TT,
    "s": S,
    "k": K,
    "cover": pd.DataFrame({"start": [0, 0, 1, 1, 2, 3], "covered": [0, 1, 1, 2, 2, 3]}),
    "ends": pd.DataFrame(
        {"t": TT, "from": ["a", "b", "a", "b"], "to": ["b", "a", "a", "b"]}
    ),
    "slot": pd.DataFrame(
        {"t": TT, "s": ["a", "a", "b", "a"], "k": ["p", "q", "p", "p"]}
    ),
    "zone_of": table(
        pd.Series(
            list("aabbbbaa"), index=pd.MultiIndex.from_product([K, TT]).swaplevel()
        ),
        "s",
    ),
    "v": pd.DataFrame({"t": TT, "v": V}),
    "z": pd.DataFrame({"s": S, "z": [10.0, 20.0]}),
    "w": pd.DataFrame({"k": K, "w": [1.0, 2.0]}),
    "zk": xr.DataArray([[1.0, 2.0], [3.0, 4.0]], coords=[S, K]),
}
JOINS: dict[str, tuple[str, list[str], list[Any]]] = {
    "sum-through-a-bare-relation": (
        "sum(x, over=start, by=cover[covered])",
        ["t"],
        [1, 3, 6, 8],
    ),
    "at-one-of-two-columns": ("x * at(z, by=ends[to])", ["t"], [20, 20, 40, 160]),
    "sum-by-one-of-two-columns": ("sum(x, over=t, by=ends[to])", ["s"], [6, 9]),
    "at-a-variable-from-both-ends": (
        "at(u, by=ends[from]) - at(u, by=ends[to])",
        ["t"],
        [-10, 10, 0, 0],
    ),
    "sum-by-a-column-list": (
        "sum(x, over=t, by=slot[s, k])",
        ["s", "k"],
        [[9, 2], [4, 0]],
    ),
    "sum-keeping-a-key-column": (
        "sum(x * w, over=t, by=zone_of[s])",
        ["s", "k"],
        [[3, 24], [12, 6]],
    ),
    "at-adding-a-key-column": (
        "x * at(z, by=zone_of[s])",
        ["t", "k"],
        [[10, 20], [20, 40], [80, 40], [160, 80]],
    ),
    "at-joining-on-a-key-column": (
        "x * at(zk, by=zone_of[s])",
        ["t", "k"],
        [[1, 4], [2, 8], [12, 8], [24, 16]],
    ),
}


def relations_spec() -> dict[str, Any]:
    spec: dict[str, Any] = {
        "dimensions": {
            "t": {"dtype": "int", "ordered": True},
            "s": {"dtype": "str"},
            "k": {"dtype": "str"},
        },
        "relations": RELATIONS,
        "parameters": {
            "v": {"dims": ["t"]},
            "z": {"dims": ["s"]},
            "w": {"dims": ["k"]},
            "zk": {"dims": ["s", "k"]},
        },
        "variables": {
            "x": {"dims": ["t"], "bounds": {"lower": 0, "upper": 100}},
            "u": {"dims": ["s"], "bounds": {"lower": 0, "upper": 100}},
        },
        "constraints": {
            "fix": {"dims": ["t"], "expression": "x == v"},
            "fix_u": {"dims": ["s"], "expression": "u == z"},
        },
        "expressions": {},
        "objective": {"sense": "minimize", "expression": "sum(x)"},
    }
    for key, (expression, dims, _) in JOINS.items():
        name = key.replace("-", "_")
        spec["variables"][f"y_{name}"] = {
            "dims": dims,
            "bounds": {"lower": -1000, "upper": 1000},
        }
        spec["constraints"][f"link_{name}"] = {
            "dims": dims,
            "expression": f"y_{name} == {expression}",
        }
        spec["expressions"][f"probe_{name}"] = {"dims": dims, "expression": expression}
    return spec


@pytest.fixture(scope="module")
def relations_model() -> Model:
    return solved(relations_spec(), RELATION_DATA, store=False)


@pytest.mark.parametrize("key", JOINS)
def test_a_join_through_several_columns_builds_and_folds_alike(
    relations_model: Model, key: str
) -> None:
    _, dims, expected = JOINS[key]
    name = key.replace("-", "_")
    coords = {d: RELATION_DATA[d] for d in dims}
    want = xr.DataArray(np.asarray(expected, dtype=float), coords=coords, dims=dims)
    want = want.transpose(*sorted(dims))
    built = relations_model.solution[f"y_{name}"]
    probe = f"probe_{name}"
    folded = evaluate(relations_spec(), relations_model, probe, sources=RELATION_DATA)
    xr.testing.assert_allclose(built, want.rename(f"y_{name}"))
    xr.testing.assert_allclose(folded[probe], want.rename(probe))


@pytest.mark.parametrize(
    ("predicate", "labels"),
    [
        ("ends[to] == 'a'", [1, 2]),
        ("ends[from] != ends[to]", [0, 1]),
    ],
)
def test_a_where_reads_one_column_of_several(predicate: str, labels: list[int]) -> None:
    spec = with_(
        relations_spec(),
        variables={"x": {"dims": ["t"], "where": predicate, "bounds": {"lower": 0}}},
    )
    x = Model.from_spec(spec, RELATION_DATA).variables["x"]
    assert x.labels.t[(x.labels != -1).to_numpy()].values.tolist() == labels


@pytest.mark.parametrize(
    ("relation", "rows", "match"),
    [
        pytest.param(
            "zone_of",
            RELATION_DATA["zone_of"].iloc[:-1],
            r"no row for 1 \('k', 't'\) label\(s\), the first at k='q', t=3",
            id="multi-key-missing-a-tuple",
        ),
        pytest.param(
            "cover",
            pd.DataFrame({"start": [0, 0], "covered": [1, 1]}),
            r"maps 1 \('covered', 'start'\) label\(s\) more than once: \(1, 0\)",
            id="bare-row-twice",
        ),
        pytest.param(
            "cover",
            pd.DataFrame({"start": [0], "covered": [9]}),
            r"maps 9, which are not labels of 't'",
            id="bare-stray-label",
        ),
        pytest.param(
            "ends",
            pd.DataFrame({"t": TT, "from": list("abab")}),
            r"arrived as a DataFrame with the columns \['t', 'from'\]",
            id="multi-value-frame-lacking-a-column",
        ),
        pytest.param(
            "ends",
            pd.Series(list("abab"), index=TT),
            r"cannot read rows out of Series",
            id="multi-value-as-a-series",
        ),
    ],
)
def test_a_relation_of_several_columns_is_checked_row_by_row(
    relation: str, rows: Any, match: str
) -> None:
    with pytest.raises(SpecDataError, match=match):
        Model.from_spec(relations_spec(), {**RELATION_DATA, relation: rows})


@pytest.mark.parametrize("relation", ["cover", "ends", "slot", "zone_of"])
def test_store_refuses_a_relation_no_single_array_holds(relation: str) -> None:
    spec = {
        "dimensions": relations_spec()["dimensions"],
        "relations": {relation: RELATIONS[relation]},
        "variables": {"x": {"dims": ["t"], "bounds": {"lower": 0}}},
        "objective": {"sense": "minimize", "expression": "sum(x)"},
    }
    data = {d: RELATION_DATA[d] for d in ("t", "s", "k", relation)}
    m = Model.from_spec(spec, data)
    assert m.variables["x"].shape == (4,)
    with pytest.raises(KeyError, match=f"no '{relation}', and store=True stores no"):
        evaluate(spec, m)
    with pytest.raises(
        SpecDataError,
        match=f"relation '{relation}' has the key column.* model.parameters",
    ):
        Model.from_spec(spec, data, store=True)


def test_store_keeps_a_single_valued_relation_read_from_a_frame() -> None:
    frame = pd.DataFrame({"t": TT, "s": ["a", "a", "b", "b"]})
    m = Model.from_spec(WHERE_SPEC, {**WHERE_DATA, "season_of": frame}, store=True)
    assert m.parameters["season_of"].values.tolist() == ["a", "a", "b", "b"]


# ---------------------------------------------------------------------------
# where predicates
# ---------------------------------------------------------------------------

WHERE_CASES: dict[str, tuple[str, str, list[Any]]] = {
    "dimension-comparison": ("x", "t > 1", [2, 3]),
    "relation-comparison": ("x", "season_of == 'a'", [0, 1]),
    "relation-not-equal-skips-unmapped": ("x", "season_of != 'a'", [2]),
    "relation-pair": ("x", "season_of != other_of", [1]),
    "relation-defined": ("x", "season_of", [0, 1, 2]),
    "pulled-back-skips-unmapped": (
        "x",
        "at(s != 'a', by=season_of[s])",
        [2],
    ),
    "not": ("x", "NOT (t > 1)", [0, 1]),
    "and": ("x", "t > 0 AND t < 3", [1, 2]),
    "or": ("x", "t == 0 OR t == 3", [0, 3]),
    "position": ("x", "position(t) == -1", [3]),
    "position-in-groups": ("x", "position(t, within=season_of[s]) == 0", [0, 2]),
    "bool-parameter": ("x", "flag", [0]),
    "float-parameter-present-and-finite": ("x", "cost", [0, 2]),
    "str-parameter": ("x", "label", [1, 2]),
    "parameter-comparison": ("x", "cost > 2", [2, 1]),
    "condition": ("x", "late", [2, 3]),
    "condition-in-a-conjunction": ("x", "late AND t < 3", [2]),
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
    ("predicate", "rows"),
    [
        pytest.param("false AND t > 1", None, id="false-flag-first"),
        pytest.param("t > 1 AND false", None, id="false-flag-last"),
        pytest.param("t < 2 AND t > 1", None, id="disjoint"),
        pytest.param("flag AND late", None, id="disjoint-bool-parameter"),
        pytest.param("t > 0 AND t < 3", [1, 2], id="overlapping"),
        pytest.param("late AND cost > 2 AND t < 3", [2], id="overlapping-three"),
    ],
)
def test_a_where_joined_by_and_builds_only_the_rows_all_its_conditions_hold(
    predicate: str, rows: list[int] | None
) -> None:
    row = {"dims": ["t"], "expression": "x >= 0", "where": predicate}
    m = Model.from_spec(with_(WHERE_SPEC, constraints={"k": row}), WHERE_DATA)
    if rows is None:
        assert "k" not in m.constraints
        return
    labels = m.constraints["k"].labels
    assert labels["t"][(labels != -1).to_numpy()].to_numpy().tolist() == rows


@pytest.mark.parametrize(
    ("predicate", "match"),
    [
        ("position(t) == 7", "names position 7 of 't', which has 4"),
        ("position(t, within=season_of[s]) == 1", "shorter than that: \\['b'\\]"),
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
# edges: partial relations and an empty dimension beside a sum
# ---------------------------------------------------------------------------

PARTIAL_CASES: dict[str, list[float]] = {
    "sum-by": [3.0, 4.0],
    "at": [10.0, 20.0, 80.0, np.nan],
    "shift-wrap-in-groups": [2.0, 1.0, 4.0, np.nan],
    "sum-back-in-groups": [1.0, 3.0, 4.0, np.nan],
}


@pytest.mark.parametrize("key", PARTIAL_CASES)
def test_a_member_a_relation_sends_nowhere_reaches_nothing(key: str) -> None:
    data = {**OPERATOR_DATA, "season_of": pd.DataFrame({"t": TT[:3], "s": list("aab")})}
    m = solved(operator_spec(), data)
    _, dims, _ = OPERATORS[key]
    name = key.replace("-", "_")
    folded = evaluate(operator_spec(), m, f"probe_{name}")[f"probe_{name}"]
    want = xr.DataArray(
        PARTIAL_CASES[key], coords={dims[0]: OPERATOR_DATA[dims[0]]}, dims=dims
    )
    xr.testing.assert_allclose(folded, want.rename(folded.name))
    if dims == ["t"]:
        assert int(m.constraints[f"link_{name}"].labels.sel(t=3)) == -1


def test_a_sum_beside_an_empty_dimension_is_the_empty_sum() -> None:
    spec: dict[str, Any] = {
        "dimensions": {"t": {"dtype": "int", "ordered": True}, "s": {"dtype": "str"}},
        "variables": {"x": {"dims": ["t", "s"], "bounds": {"lower": 0, "upper": 1}}},
        "constraints": {"cap": {"dims": ["s"], "expression": "sum(x, over=t) <= 1"}},
        "objective": {"sense": "maximize", "expression": "sum(x)"},
    }
    m = Model.from_spec(spec, {"t": [0, 1], "s": pd.Index([], name="s", dtype=object)})
    assert "cap" not in m.constraints


def test_a_window_width_no_member_carries_is_a_window_of_nothing() -> None:
    data = {
        **OPERATOR_DATA,
        "season_of": pd.DataFrame(
            {"t": pd.Series([], dtype=int), "s": pd.Series([], dtype=object)}
        ),
    }
    m = solved(operator_spec(), data)
    folded = evaluate(operator_spec(), m, "probe_sum_back_group_width")[
        "probe_sum_back_group_width"
    ]
    assert bool(folded.isnull().all())


def test_a_lookup_into_an_empty_dimension_reads_nothing() -> None:
    """`at` gathered by position, and an empty target has no position 0 to stand in for a hole."""
    spec = {
        "dimensions": {"outage": {}, "line": {}, "transformer": {}},
        "relations": {
            "Outage_line": {"key": "outage", "values": "line", "missing": "absent"},
            "Outage_transformer": {
                "key": "outage",
                "values": "transformer",
                "missing": "absent",
            },
        },
        "variables": {"s": {"dims": ["line"]}, "t": {"dims": ["transformer"]}},
        "expressions": {
            "flow": {
                "dims": ["outage"],
                "cases": {
                    "line": {
                        "when": "Outage_line",
                        "expression": "at(s, by=Outage_line[line])",
                    }
                },
                "otherwise": "at(t, by=Outage_transformer[transformer])",
            }
        },
        "constraints": {"cap": {"dims": ["outage"], "expression": "flow <= 1"}},
    }
    sources = {
        "outage": ["a"],
        "line": ["a"],
        "transformer": pd.Index([], dtype=object),
        "Outage_line": pd.DataFrame({"outage": ["a"], "line": ["a"]}),
        "Outage_transformer": pd.DataFrame(
            {"outage": [], "transformer": []}, dtype=object
        ),
    }
    m = Model.from_spec(spec, sources)
    assert int((m.constraints["cap"].labels != -1).sum()) == 1
