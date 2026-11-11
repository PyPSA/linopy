"""
A multi-dimension: its table attaches as tuple labels, and a model over it
builds the same problem as the flat twin over its members confined to the
listed combinations.
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Any

import pandas as pd
import polars as pl
import pytest
import xarray as xr

pytest.importorskip("mathspec")
pytest.importorskip("highspy")

from spec_helpers import solved  # noqa: E402

from linopy import Model  # noqa: E402
from linopy.spec import SpecDataError, evaluate  # noqa: E402
from linopy.testing import synthetic_sources  # noqa: E402

DIMENSIONS: dict[str, Any] = {
    "unit": {},
    "bus": {},
    "snapshot": {"dtype": "int", "ordered": True},
    "zone": {},
}
RELATIONS: dict[str, Any] = {"zone_of": {"key": "bus", "values": "zone"}}
PARAMETERS: dict[str, Any] = {
    "capacity": {"dims": ["unit"]},
    "load": {"dims": ["bus", "snapshot"]},
    "cost": {"dims": ["unit"]},
    "zone_cap": {"dims": ["zone"]},
}
EXPRESSIONS: dict[str, str] = {
    "balance": "sum(efficiency * flow, over=unit) == load",
    "cap": "flow <= capacity",
    "total": "sum(flow, over=bus) <= capacity",
    "ramp": "flow - shift(flow, along=snapshot, offset=1) <= 0.6 * capacity",
    "zonal": "sum(flow, over=[unit, bus], by=zone_of[zone]) <= zone_cap",
}
GUARD = "count(connection, over=unit) >= 1"
MULTI: dict[str, Any] = {
    "dimensions": {**DIMENSIONS, "connection": {"over": ["unit", "bus"]}},
    "relations": RELATIONS,
    "parameters": {
        **PARAMETERS,
        "efficiency": {"dims": ["connection", "snapshot"], "missing": 1},
        "output": {"dims": ["connection"], "dtype": "bool", "missing": False},
    },
    "variables": {
        "flow": {"dims": ["connection", "snapshot"], "bounds": {"lower": 0}},
        "spill": {
            "dims": ["bus", "snapshot", "unit"],
            "where": "connection AND NOT output",
            "bounds": {"lower": 0, "upper": 1},
        },
    },
    "constraints": {
        "balance": {
            "dims": ["bus", "snapshot"],
            "expression": EXPRESSIONS["balance"],
            "where": GUARD,
        },
        "cap": {"dims": ["connection", "snapshot"], "expression": EXPRESSIONS["cap"]},
        "total": {"dims": ["snapshot", "unit"], "expression": EXPRESSIONS["total"]},
        "ramp": {"dims": ["connection", "snapshot"], "expression": EXPRESSIONS["ramp"]},
        "spilled": {"dims": ["bus", "snapshot", "unit"], "expression": "spill <= flow"},
        "zonal": {"dims": ["snapshot", "zone"], "expression": EXPRESSIONS["zonal"]},
        "spill_cap": {
            "dims": ["snapshot"],
            "expression": "sum(spill, over=connection) <= 1",
        },
    },
    "expressions": {
        "share": {"dims": ["connection", "snapshot"], "expression": "flow / capacity"}
    },
    "objective": {
        "expression": "sum(cost * flow, over=[connection, snapshot]) - sum(spill)"
    },
}
FLAT: dict[str, Any] = {
    "dimensions": DIMENSIONS,
    "relations": {
        **RELATIONS,
        "connection": {"key": ["unit", "bus"], "missing": "absent"},
    },
    "parameters": {
        **PARAMETERS,
        "efficiency": {
            "dims": ["bus", "snapshot", "unit"],
            "missing": 1,
            "where": "connection",
        },
        "output": {"dims": ["bus", "unit"], "dtype": "bool", "missing": False},
    },
    "variables": {
        "flow": {
            "dims": ["bus", "snapshot", "unit"],
            "where": "connection",
            "bounds": {"lower": 0},
        },
        "spill": MULTI["variables"]["spill"],
    },
    "constraints": {
        "balance": MULTI["constraints"]["balance"],
        "cap": {
            "dims": ["bus", "snapshot", "unit"],
            "expression": EXPRESSIONS["cap"],
            "where": "connection",
        },
        "total": MULTI["constraints"]["total"],
        "ramp": {
            "dims": ["bus", "snapshot", "unit"],
            "expression": EXPRESSIONS["ramp"],
            "where": "connection",
        },
        "spilled": MULTI["constraints"]["spilled"],
        "zonal": MULTI["constraints"]["zonal"],
        "spill_cap": {
            "dims": ["snapshot"],
            "expression": "sum(spill, over=[bus, unit]) <= 1",
        },
    },
    "objective": {
        "expression": "sum(cost * flow, over=[bus, snapshot, unit]) - sum(spill)"
    },
}
UNIT = pd.Index(["u1", "u2"], name="unit")
BUS = pd.Index(["b1", "b2", "b3"], name="bus")
CONNECTION = pd.DataFrame({"unit": ["u1", "u1", "u2"], "bus": ["b1", "b2", "b2"]})
LISTED = pd.Index(
    [("b1", "u1"), ("b2", "u1"), ("b2", "u2")], dtype=object, tupleize_cols=False
)
SOURCES: dict[str, Any] = {
    "unit": UNIT,
    "bus": BUS,
    "snapshot": [0, 1, 2],
    "connection": CONNECTION,
    "zone": ["z1", "z2"],
    "zone_of": pd.DataFrame({"bus": BUS, "zone": ["z1", "z2", "z2"]}),
    "zone_cap": pd.DataFrame({"zone": ["z1", "z2"], "zone_cap": [20.0, 9.0]}),
    "capacity": pd.DataFrame({"unit": UNIT, "capacity": [10.0, 5.0]}),
    "cost": pd.DataFrame({"unit": UNIT, "cost": [1.0, 2.0]}),
    "load": pd.DataFrame(
        {
            "bus": [*BUS.repeat(3)],
            "snapshot": [0, 1, 2] * 3,
            "load": [1.0, 4, 2, 3, 6, 4, 0, 0, 0],
        }
    ),
    "efficiency": pd.DataFrame(
        {"unit": ["u1"], "bus": ["b2"], "snapshot": [1], "efficiency": [0.8]}
    ),
    "output": pd.DataFrame({"unit": ["u2"], "bus": ["b2"], "output": [True]}),
}


def _plain(
    name: str, coords: dict[str, Any]
) -> tuple[str, tuple[tuple[str, Any], ...]]:
    """*name* at *coords*, a ``connection`` label read as its members, as the flat twin labels it."""
    flat = {str(d): v for d, v in coords.items() if d != "connection"}
    if "connection" in coords:
        flat |= dict(zip(["bus", "unit"], coords["connection"]))
    return name, tuple(
        sorted((d, v.item() if hasattr(v, "item") else v) for d, v in flat.items())
    )


def _problem(m: Model) -> Counter[Any]:
    """Each constraint row of *m* as its name, coordinate, terms, sign and right-hand side, read over plain dimensions."""
    rows: dict[int, Any] = {}
    for row in m.constraints.flat.itertuples():
        at = _plain(*m.constraints.get_label_position(int(row.labels)))
        term = (_plain(*m.variables.get_label_position(int(row.vars))), row.coeffs)
        rows.setdefault(row.labels, (at, row.sign, row.rhs, []))[3].append(term)
    return Counter(
        (at, sign, rhs, frozenset(terms)) for at, sign, rhs, terms in rows.values()
    )


def test_a_multi_dimension_builds_the_problem_of_its_flat_twin() -> None:
    multi = solved(MULTI, SOURCES, store=False)
    flat = solved(FLAT, SOURCES, store=False)
    assert _problem(multi) == _problem(flat)
    assert multi.objective.value == pytest.approx(flat.objective.value)


def test_a_variable_over_a_multi_dimension_has_a_row_per_listed_combination() -> None:
    m = Model.from_spec(MULTI, SOURCES)
    assert m.variables["flow"].indexes["connection"].equals(LISTED)
    spill = m.variables["spill"].mask.sel(snapshot=0).to_pandas()
    assert spill.stack()[lambda s: s].index.tolist() == [("b1", "u1"), ("b2", "u1")], (
        "NOT output holds off connection"
    )


def test_a_named_expression_over_a_multi_dimension_folds_from_the_stored_data() -> None:
    m = solved(MULTI, SOURCES)
    share = evaluate(MULTI, m, "share")["share"]
    assert share.indexes["connection"].equals(LISTED)
    xr.testing.assert_allclose(share, m.expressions["share"].solution)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(CONNECTION, id="pandas"),
        pytest.param(pl.from_pandas(CONNECTION), id="polars"),
        pytest.param(pd.MultiIndex.from_frame(CONNECTION), id="multiindex"),
        pytest.param(CONNECTION[["bus", "unit"]], id="columns-in-any-order"),
    ],
)
def test_a_multi_dimension_attaches_from_its_table(source: Any) -> None:
    m = Model.from_spec(MULTI, {**SOURCES, "connection": source})
    assert m.variables["flow"].indexes["connection"].equals(LISTED)


def test_a_grid_over_a_multi_dimension_may_be_stacked() -> None:
    grid = xr.DataArray(
        [[0.5, 0.5, 1.0], [1.0, 1.0, 1.0]], coords={"unit": UNIT, "bus": BUS}
    )
    stacked = grid.stack(connection=["unit", "bus"]).sel(
        connection=list(CONNECTION.itertuples(index=False))
    )
    m = Model.from_spec(MULTI, {**SOURCES, "output": stacked > 0.9}, store=True)
    assert m.parameters["output"].to_series().tolist() == [False, False, True]


@pytest.mark.parametrize(
    ("patch", "fragment"),
    [
        pytest.param(
            {"connection": list(LISTED)},
            "Pass a pandas or polars DataFrame",
            id="a-list-of-tuples",
        ),
        pytest.param(
            {"connection": CONNECTION[["unit"]]},
            "exactly the columns ['bus', 'unit']",
            id="a-member-missing",
        ),
        pytest.param(
            {"connection": pd.DataFrame({"unit": ["u1"], "bus": ["b9"]})},
            "not coordinates of it: 'b9'",
            id="a-label-its-member-lacks",
        ),
        pytest.param(
            {"connection": pd.concat([CONNECTION, CONNECTION.head(1)])},
            "lists ('b1', 'u1') more than once",
            id="a-combination-twice",
        ),
        pytest.param(
            {"connection": pd.DataFrame({"unit": ["u1", None], "bus": ["b1", "b2"]})},
            "with no label",
            id="a-hole",
        ),
        pytest.param(
            {"output": pd.DataFrame({"unit": ["u2"], "bus": ["b1"], "output": [True]})},
            "that 'connection' does not list: bus='b1', unit='u2'",
            id="a-parameter-row-off-the-listed-combinations",
        ),
        pytest.param(
            {
                "output": pd.DataFrame(
                    {"connection": list(LISTED), "output": [True] * 3}
                )
            },
            "exactly the columns ['bus', 'unit', 'output']",
            id="a-parameter-table-without-member-columns",
        ),
    ],
)
def test_data_a_multi_dimension_cannot_take_is_refused(
    patch: dict[str, Any], fragment: str
) -> None:
    with pytest.raises(SpecDataError) as caught:
        Model.from_spec(MULTI, {**SOURCES, **patch})
    assert fragment in str(caught.value)


def test_synthetic_sources_list_a_strict_subset_of_the_combinations() -> None:
    sources = synthetic_sources(MULTI, n=3)
    assert len(sources["connection"]) == 5, "every other of the 3 x 3 combinations"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Model.from_spec({**MULTI, "constraints": {}}, sources)
    assert m.variables["flow"].sizes["connection"] == 5
