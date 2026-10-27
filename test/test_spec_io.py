"""
Round trips of a spec-built model through netcdf and through ``copy``.

A spec-built model built with ``store=True`` keeps its raw parameters and
relations in ``model.parameters``, so the netcdf file holds the data a model
is rebuilt from with the spec. Labels are the delicate part, a partial relation holds
NaN in an array of strings, so every relation shape is checked value by value,
on both netcdf engines ``test_io`` uses. No test here asserts a model
container's dtype after a round trip: the scipy engine narrows int64.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

mathspec = pytest.importorskip("mathspec")
pytest.importorskip("highspy")

from spec_helpers import (  # noqa: E402
    DISPATCH_DATA,
    EXAMPLE_DISPATCH,
    EXAMPLES_DIR,
    WHERE_DATA,
    WHERE_SPEC,
    solved,
    with_,
)

from linopy import Model, read_netcdf  # noqa: E402
from linopy.spec import evaluate  # noqa: E402
from linopy.testing import (  # noqa: E402
    assert_linequal,
    assert_model_equal,
    synthetic_sources,
)

HAS_NETCDF4 = importlib.util.find_spec("netCDF4") is not None
ENGINES = [
    pytest.param(
        "netcdf4",
        marks=pytest.mark.skipif(not HAS_NETCDF4, reason="netCDF4 not installed"),
    ),
    "scipy",
]

S1 = pd.Index(["a", "b", "c"], name="s1")
S2 = pd.Index(["p", "q"], name="s2")
I1 = pd.Index([10, 20, 30], name="i1")
I2 = pd.Index([1, 2], name="i2")

RELATION_SPEC: dict[str, Any] = {
    "dimensions": {
        "s1": {"dtype": "str"},
        "s2": {"dtype": "str"},
        "i1": {"dtype": "int"},
        "i2": {"dtype": "int"},
    },
    "relations": {
        "str_to_str": {"key": "s1", "values": "s2", "missing": "absent"},
        "str_to_int": {"key": "s1", "values": "i2", "missing": "absent"},
        "int_to_str": {"key": "i1", "values": "s2", "missing": "absent"},
        "int_to_int": {"key": "i1", "values": "i2", "missing": "absent"},
    },
    "parameters": {"cost": {"dims": ["s1"]}},
    "variables": {"x": {"dims": ["s1"], "bounds": {"lower": 0, "upper": 1}}},
    "objective": {"sense": "minimize", "expression": "sum(x * cost)"},
}
RELATION_OVER = {"str_to_str": S1, "str_to_int": S1, "int_to_str": I1, "int_to_int": I1}
RELATION_INTO = {"str_to_str": S2, "str_to_int": I2, "int_to_str": S2, "int_to_int": I2}

DTYPE_SPEC: dict[str, Any] = {
    "dimensions": {"s1": {"dtype": "str"}},
    "parameters": {
        "count": {"dims": ["s1"], "dtype": "int"},
        "flag": {"dims": ["s1"], "dtype": "bool"},
        "cost": {"dims": ["s1"]},
        "tag": {"dims": ["s1"], "dtype": "str"},
        "sparse_count": {"dims": ["s1"], "dtype": "int", "missing": "neutral"},
        "sparse_flag": {"dims": ["s1"], "dtype": "bool", "missing": "absent"},
    },
    "variables": {"x": {"dims": ["s1"], "bounds": {"lower": 0, "upper": 1}}},
    "objective": {"sense": "minimize", "expression": "sum(x * cost)"},
    "expressions": {
        "report": {
            "dims": ["s1"],
            "cases": {
                "on": {"when": "flag AND tag == 'u'", "expression": "cost * count"}
            },
            "otherwise": 0,
        }
    },
}
TAGGED_SPEC = with_(
    WHERE_SPEC,
    expressions={
        "tagged": {
            "dims": ["t"],
            "cases": {"u": {"when": "label == 'u'", "expression": "1"}},
            "otherwise": 0,
        }
    },
)
DTYPE_DATA: dict[str, Any] = {
    "s1": S1,
    "count": pd.Series([1, 2, 3], index=S1),
    "flag": pd.Series([True, False, True], index=S1),
    "cost": pd.Series([1.0, 2.0, 3.0], index=S1),
    "tag": pd.Series(["u", "v", "w"], index=S1),
    "sparse_count": pd.Series([1, 2], index=S1[:2]),
    "sparse_flag": pd.Series([True, False], index=S1[:2]),
}


def relation_sources(mapped: int) -> dict[str, Any]:
    """Data for ``RELATION_SPEC``, each relation mapping only its first *mapped* labels."""
    sources: dict[str, Any] = {
        "s1": S1,
        "s2": S2,
        "i1": I1,
        "i2": I2,
        "cost": pd.Series([1.0, 2.0, 3.0], index=S1),
    }
    for name, over in RELATION_OVER.items():
        into = RELATION_INTO[name]
        sources[name] = pd.Series(
            [into[i % len(into)] for i in range(mapped)], index=over[:mapped]
        )
    return sources


def roundtrip(m: Model, tmp_path: Path, engine: str) -> Model:
    path = tmp_path / f"model-{engine}.nc"
    m.to_netcdf(path, engine=engine)
    return read_netcdf(path)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_spec_built_model_round_trips(tmp_path: Path, engine: str) -> None:
    m = solved(EXAMPLE_DISPATCH, DISPATCH_DATA)
    assert_model_equal(m, roundtrip(m, tmp_path, engine))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("mapped", [3, 2, 0], ids=["full", "partial", "empty"])
@pytest.mark.parametrize("name", RELATION_OVER)
def test_a_relation_round_trips_exactly(
    tmp_path: Path, engine: str, mapped: int, name: str
) -> None:
    m = Model.from_spec(RELATION_SPEC, relation_sources(mapped), store=True)
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    xr.testing.assert_identical(p.parameters[name], m.parameters[name])


AT_SPEC = with_(
    RELATION_SPEC,
    variables={"y": {"dims": ["s2"], "bounds": {"lower": 0, "upper": 1}}},
    expressions={"pulled": "x + at(y, by=str_to_str[s2])"},
)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_named_expression_read_through_a_lookup_round_trips(
    tmp_path: Path, engine: str
) -> None:
    m = Model.from_spec(AT_SPEC, relation_sources(2))
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    assert_linequal(p.expressions["pulled"], m.expressions["pulled"])


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "labels", [[10, 11, 12], [0, 1]], ids=["relabelled", "shorter"]
)
def test_a_hand_added_variable_keeps_its_own_labels(
    tmp_path: Path, engine: str, labels: list[int]
) -> None:
    """A container sharing a master dimension's name but not its labels is left alone."""
    own = pd.Index(labels, name="snapshot")
    m = Model.from_spec(EXAMPLE_DISPATCH, DISPATCH_DATA, store=True)
    m.add_variables(coords=[own], name="side")
    p = roundtrip(m, tmp_path, engine)

    master = m.parameters.indexes["snapshot"]
    assert p.variables["side"].indexes["snapshot"].equals(own)
    assert p.parameters.indexes["snapshot"].equals(master)
    assert p.variables["p"].indexes["snapshot"].equals(master)


@pytest.mark.parametrize("engine", ENGINES)
def test_every_container_shares_the_master_coordinates(
    tmp_path: Path, engine: str, backend: Model
) -> None:
    """The master coordinates are canonical: no container may disagree with them."""
    if backend.sparse and engine == "scipy":
        pytest.skip(
            "netCDF3 holds no unicode-array attr, and a sparse constraint writes one"
        )
    m = Model.from_spec(
        EXAMPLE_DISPATCH, DISPATCH_DATA, store=True, sparse=backend.sparse
    )
    m.solve(solver_name="highs", output_flag=False)
    p = roundtrip(m, tmp_path, engine)

    master = p.parameters.indexes
    holders = [
        *p.variables.data.values(),
        *p.constraints.data.values(),
        p.objective.expression,
    ]
    for holder in holders:
        for dim, index in holder.indexes.items():
            assert index.equals(master[dim]), f"{dim} differs on {holder}"


@pytest.mark.parametrize("engine", ENGINES)
def test_labelled_parameters_and_unreached_coordinates_round_trip(
    tmp_path: Path, engine: str
) -> None:
    """A str parameter with holes, and a dimension only a relation reaches."""
    m = Model.from_spec(TAGGED_SPEC, WHERE_DATA, store=True)
    assert "label" in m.parameters
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    assert set(p.parameters.indexes) == set(m.parameters.indexes)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    ("position", "value"), [(0, np.nan), (3, 0.0)], ids=["a-new-hole", "a-filled-hole"]
)
def test_a_hole_round_trips_and_is_compared(
    tmp_path: Path, engine: str, position: int, value: float
) -> None:
    m = Model.from_spec(
        with_(WHERE_SPEC, expressions={"twice": "cost * 2"}), WHERE_DATA, store=True
    )
    p = roundtrip(m, tmp_path, engine)
    assert_model_equal(m, p)
    assert p.parameters["cost"].isnull().values.tolist() == [False, False, False, True]

    p.parameters["cost"].values[position] = value
    with pytest.raises(AssertionError):
        assert_model_equal(m, p)


REBUILT: dict[str, tuple[Any, Any]] = {
    "dispatch": (EXAMPLE_DISPATCH, DISPATCH_DATA),
    "dtypes": (DTYPE_SPEC, DTYPE_DATA),
    "relations": (RELATION_SPEC, relation_sources(2)),
    "tagged": (TAGGED_SPEC, WHERE_DATA),
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(("spec", "sources"), REBUILT.values(), ids=REBUILT.keys())
def test_a_model_rebuilds_from_its_netcdf_parameters(
    tmp_path: Path, engine: str, spec: Any, sources: Any
) -> None:
    m = Model.from_spec(spec, sources, store=True)
    rebuilt = Model.from_spec(
        spec, roundtrip(m, tmp_path, engine).parameters, store=True
    )
    assert_model_equal(rebuilt, m)


def copied(deep: bool) -> Callable[[Model, Path, str], Model]:
    return lambda m, tmp_path, engine: m.copy(include_solution=True, deep=deep)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "transform",
    [roundtrip, copied(deep=True), copied(deep=False)],
    ids=["netcdf", "deep-copy", "shallow-copy"],
)
def test_evaluate_after_a_round_trip_and_a_copy(
    tmp_path: Path, engine: str, transform: Callable[[Model, Path, str], Model]
) -> None:
    m = solved(EXAMPLE_DISPATCH, DISPATCH_DATA)
    expected = evaluate(EXAMPLE_DISPATCH, m)
    got = evaluate(EXAMPLE_DISPATCH, transform(m, tmp_path, engine))

    assert list(got) == list(expected)
    for name, value in expected.items():
        xr.testing.assert_allclose(got[name], value)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "transform", [roundtrip, copied(deep=True)], ids=["netcdf", "deep-copy"]
)
def test_an_edit_to_stored_parameters_takes_effect_on_rebuild(
    tmp_path: Path, engine: str, transform: Callable[[Model, Path, str], Model]
) -> None:
    m = solved(EXAMPLE_DISPATCH, DISPATCH_DATA)
    p = transform(m, tmp_path, engine)
    p.parameters["cost"].loc["wind"] = 10.0
    rebuilt = solved(EXAMPLE_DISPATCH, p.parameters)

    assert m.objective.value == pytest.approx(2500.0)
    assert rebuilt.objective.value == pytest.approx(4800.0)
    xr.testing.assert_identical(rebuilt.parameters, p.parameters)


@pytest.mark.skipif(
    EXAMPLES_DIR is None, reason="set MATHSPEC_EXAMPLES to a mathspec examples dir"
)
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("example", ["commitment.yaml", "operators/sum_by.yaml"])
def test_a_mathspec_example_round_trips(
    tmp_path: Path, engine: str, example: str
) -> None:
    path = Path(EXAMPLES_DIR or "", example)
    m = Model.from_spec(path, synthetic_sources(path, 3), store=True)
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    assert set(p.parameters.indexes) == set(m.parameters.indexes)
