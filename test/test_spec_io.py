"""
Round trips of a spec-built model through netcdf and through ``copy``.

The spec itself is persisted as its YAML text and lowered again on read, so
what has to survive besides the model is data: the master coordinates, the
lookups and the retained parameters. Labels are the delicate part — a partial
lookup holds NaN in an array of strings — so every lookup shape is checked
value by value and dtype by dtype, on both netcdf engines ``test_io`` uses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import xarray as xr

math_spec = pytest.importorskip("math_spec")

from test_spec_builder import (  # noqa: E402
    DISPATCH_DATA,
    EXAMPLE_DISPATCH,
    EXAMPLES_DIR,
    WHERE_DATA,
    WHERE_SPEC,
    solved,
    synthetic_sources,
)

import linopy  # noqa: E402
from linopy import Model, read_netcdf  # noqa: E402
from linopy.io import SPEC_ATTR  # noqa: E402
from linopy.testing import assert_model_equal  # noqa: E402

pytestmark = [
    pytest.mark.v1,
    pytest.mark.skipif("highs" not in linopy.available_solvers, reason="needs highs"),
]

ENGINES = ["netcdf4", "scipy"]

S1 = pd.Index(["a", "b", "c"], name="s1")
S2 = pd.Index(["p", "q"], name="s2")
I1 = pd.Index([10, 20, 30], name="i1")
I2 = pd.Index([1, 2], name="i2")

LOOKUP_SPEC: dict[str, Any] = {
    "dimensions": {
        "s1": {"dtype": "str"},
        "s2": {"dtype": "str"},
        "i1": {"dtype": "int"},
        "i2": {"dtype": "int"},
    },
    "lookups": {
        "str_to_str": {"over": "s1", "into": "s2"},
        "str_to_int": {"over": "s1", "into": "i2"},
        "int_to_str": {"over": "i1", "into": "s2"},
        "int_to_int": {"over": "i1", "into": "i2"},
    },
    "parameters": {"cost": {"dims": ["s1"]}},
    "variables": {"x": {"foreach": ["s1"], "bounds": {"lower": 0, "upper": 1}}},
    "objective": {"sense": "minimize", "expression": "sum(x * cost)"},
}
LOOKUP_OVER = {"str_to_str": S1, "str_to_int": S1, "int_to_str": I1, "int_to_int": I1}
LOOKUP_INTO = {"str_to_str": S2, "str_to_int": I2, "int_to_str": S2, "int_to_int": I2}


def lookup_sources(mapped: int) -> dict[str, Any]:
    """Data for ``LOOKUP_SPEC``, each lookup mapping only its first *mapped* labels."""
    sources: dict[str, Any] = {
        "s1": S1,
        "s2": S2,
        "i1": I1,
        "i2": I2,
        "cost": pd.Series([1.0, 2.0, 3.0], index=S1),
    }
    for name, over in LOOKUP_OVER.items():
        into = LOOKUP_INTO[name]
        sources[name] = pd.Series(
            [into[i % len(into)] for i in range(mapped)], index=over[:mapped]
        )
    return sources


def roundtrip(m: Model, tmp_path: Path, engine: str) -> Model:
    path = tmp_path / f"model-{engine}.nc"
    m.to_netcdf(path, engine=engine)
    return read_netcdf(path)


def assert_arrayequal(a: xr.DataArray, b: xr.DataArray) -> None:
    """Assert equal values and dtype — the dtype is what a netcdf type drops."""
    assert a.dtype == b.dtype, f"dtypes differ: {a.dtype} != {b.dtype}"
    xr.testing.assert_equal(a, b)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("retain", ["report", "all"])
def test_a_spec_built_model_round_trips(
    tmp_path: Path, engine: str, retain: str
) -> None:
    m = solved(EXAMPLE_DISPATCH, DISPATCH_DATA, retain=retain)
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    assert p.spec.text == m.spec.text
    assert p.spec.program.constraints == m.spec.program.constraints
    assert set(p.spec.expressions) == set(m.spec.expressions)
    for name in m.spec.expressions:
        assert_arrayequal(m.spec.expressions[name], p.spec.expressions[name])
    for dim, index in m.spec.coords.items():
        pd.testing.assert_index_equal(index, p.spec.coords[dim])


@pytest.mark.parametrize("engine", ENGINES)
def test_a_retain_none_model_evaluates_after_a_round_trip(
    tmp_path: Path, engine: str
) -> None:
    m = solved(EXAMPLE_DISPATCH, DISPATCH_DATA, retain="none")
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    assert not p.spec.parameters.data_vars
    assert_arrayequal(
        m.spec.evaluate("spend", DISPATCH_DATA), p.spec.evaluate("spend", DISPATCH_DATA)
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("mapped", [3, 2, 0], ids=["full", "partial", "empty"])
@pytest.mark.parametrize("name", LOOKUP_OVER)
def test_a_lookup_round_trips_exactly(
    tmp_path: Path, engine: str, mapped: int, name: str
) -> None:
    m = Model.from_spec(LOOKUP_SPEC, lookup_sources(mapped), retain="all")
    over = str(LOOKUP_OVER[name].name)
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    assert_arrayequal(m.spec.lookups[over][name], p.spec.lookups[over][name])


@pytest.mark.parametrize("engine", ENGINES)
def test_labelled_parameters_and_unreached_coordinates_round_trip(
    tmp_path: Path, engine: str
) -> None:
    m = Model.from_spec(WHERE_SPEC, WHERE_DATA, retain="all")
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    assert_arrayequal(m.spec.parameters["label"], p.spec.parameters["label"])
    for dim, index in m.spec.coords.items():
        pd.testing.assert_index_equal(index, p.spec.coords[dim])


@pytest.mark.parametrize("engine", ENGINES)
def test_a_solved_model_reports_the_same_expression(
    tmp_path: Path, engine: str
) -> None:
    m = solved(EXAMPLE_DISPATCH, DISPATCH_DATA)
    p = roundtrip(m, tmp_path, engine)

    assert p.objective.value == m.objective.value
    assert_arrayequal(m.spec.expressions["spend"], p.spec.expressions["spend"])
    assert float(p.spec.expressions["spend"].sum()) == pytest.approx(2500.0)


@pytest.mark.parametrize("deep", [True, False])
def test_a_copy_carries_the_spec(deep: bool) -> None:
    m = Model.from_spec(WHERE_SPEC, WHERE_DATA, retain="all")
    p = m.copy(deep=deep)
    m.parameters = m.parameters.drop_vars("cost")

    assert p.spec.text == m.spec.text
    assert "cost" in p.spec.parameters, "the copy's spec reads the copy, not the source"
    for over, by_name in m.spec.lookups.items():
        for name, lookup in by_name.items():
            assert_arrayequal(lookup, p.spec.lookups[over][name])


def test_a_model_without_a_spec_carries_none(tmp_path: Path) -> None:
    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(3, name="i")], name="x")
    m.add_objective(x.sum())
    path = tmp_path / "plain.nc"
    m.to_netcdf(path)

    assert SPEC_ATTR not in xr.load_dataset(path).attrs
    assert read_netcdf(path)._spec is None
    assert m.copy()._spec is None


@pytest.mark.skipif(
    EXAMPLES_DIR is None, reason="set MATH_SPEC_EXAMPLES to a math-spec examples dir"
)
@pytest.mark.parametrize("engine", ENGINES)
def test_the_pypsa_example_round_trips(tmp_path: Path, engine: str) -> None:
    """Nine lookups into one dimension, a datetime axis, bool and str parameters."""
    path = Path(EXAMPLES_DIR or "", "pypsa.yaml")
    program = math_spec.to_program(str(path))
    m = Model.from_spec(path, synthetic_sources(program, 3), retain="all")
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    for dim, index in m.spec.coords.items():
        pd.testing.assert_index_equal(index, p.spec.coords[dim])
    for over, by_name in m.spec.lookups.items():
        for name, lookup in by_name.items():
            assert_arrayequal(lookup, p.spec.lookups[over][name])
