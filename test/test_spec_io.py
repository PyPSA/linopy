"""
Round trips of a spec-built model through netcdf and through ``copy``.

The spec itself is persisted as its YAML text and lowered again on read, so
what has to survive besides the model is data: the master coordinates, the
relations and the kept parameters. Labels are the delicate part — a partial
relation holds NaN in an array of strings — so every relation shape is checked
value by value, on both netcdf engines ``test_io`` uses. No test here asserts a
model container's dtype after a round trip: the scipy engine narrows int64.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

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
    yaml_dict,
)

import linopy  # noqa: E402
from linopy import Model, read_netcdf  # noqa: E402
from linopy.spec import lower  # noqa: E402
from linopy.spec.testing import synthetic_sources  # noqa: E402
from linopy.testing import assert_linequal, assert_model_equal  # noqa: E402

ENGINES = ["netcdf4", "scipy"]

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
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    assert p.spec.text == m.spec.text
    assert p.spec.program.constraints == m.spec.program.constraints
    assert set(p.spec.expressions) == set(m.spec.expressions)
    for name in m.spec.expressions:
        xr.testing.assert_equal(
            m.spec.expressions[name].solution, p.spec.expressions[name].solution
        )


@pytest.mark.parametrize("engine", ENGINES)
def test_the_caller_parameters_and_the_spec_ones_stay_apart(
    tmp_path: Path, engine: str
) -> None:
    """A model parameter of the caller's is written beside the spec's, not into them."""
    m = solved(with_(yaml_dict(), expressions={"twice": "cost * 2"}), DISPATCH_DATA)
    m.parameters["cost"] = xr.DataArray([1.0, 2.0, 3.0], dims=["own"])
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    xr.testing.assert_equal(p.parameters["cost"], m.parameters["cost"])
    xr.testing.assert_equal(p.spec.parameters["cost"], m.spec.parameters["cost"])


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("mapped", [3, 2, 0], ids=["full", "partial", "empty"])
@pytest.mark.parametrize("name", RELATION_OVER)
def test_a_relation_round_trips_exactly(
    tmp_path: Path, engine: str, mapped: int, name: str
) -> None:
    m = Model.from_spec(RELATION_SPEC, relation_sources(mapped))
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    assert name in p.spec.relations


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
@pytest.mark.parametrize("name", ["count", "flag", "cost", "tag"])
def test_a_parameter_keeps_its_dtype(tmp_path: Path, engine: str, name: str) -> None:
    m = Model.from_spec(DTYPE_SPEC, DTYPE_DATA)
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    assert p.spec.parameters[name].dtype == m.spec.parameters[name].dtype


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "labels", [[10, 11, 12], [0, 1]], ids=["relabelled", "shorter"]
)
def test_a_hand_added_variable_keeps_its_own_labels(
    tmp_path: Path, engine: str, labels: list[int]
) -> None:
    """A container sharing a master dimension's name but not its labels is left alone."""
    own = pd.Index(labels, name="snapshot")
    m = Model.from_spec(EXAMPLE_DISPATCH, DISPATCH_DATA)
    m.add_variables(coords=[own], name="side")
    p = roundtrip(m, tmp_path, engine)

    assert p.variables["side"].indexes["snapshot"].equals(own)
    assert p.spec.coords["snapshot"].equals(m.spec.coords["snapshot"])
    assert p.variables["p"].indexes["snapshot"].equals(m.spec.coords["snapshot"])


@pytest.mark.parametrize("engine", ENGINES)
def test_every_container_shares_the_master_coordinates(
    tmp_path: Path, engine: str, backend: Model
) -> None:
    """The master coordinates are canonical: no container may disagree with them."""
    if backend.sparse and engine == "scipy":
        pytest.skip(
            "netCDF3 holds no unicode-array attr, and a sparse constraint writes one"
        )
    m = backend.add_spec(EXAMPLE_DISPATCH, DISPATCH_DATA)
    m.solve(solver_name="highs", output_flag=False)
    p = roundtrip(m, tmp_path, engine)

    master = p.spec.coords
    holders = [
        *p.variables.data.values(),
        *p.constraints.data.values(),
        p.objective.expression,
    ]
    for holder in holders:
        for dim, index in holder.indexes.items():
            assert index.equals(master[str(dim)]), f"{dim} differs on {holder}"


def test_a_model_whose_expressions_all_carry_variables_writes_no_parameters(
    tmp_path: Path,
) -> None:
    path = tmp_path / "model.nc"
    Model.from_spec(EXAMPLE_DISPATCH, DISPATCH_DATA).to_netcdf(path)
    written = {str(n) for n in xr.load_dataset(path).data_vars}
    assert {n for n in written if n.startswith("spec-")} == {
        "spec-coords__snapshot",
        "spec-coords__generator",
    }


@pytest.mark.parametrize("engine", ENGINES)
def test_labelled_parameters_and_unreached_coordinates_round_trip(
    tmp_path: Path, engine: str
) -> None:
    """A str parameter with holes, and a dimension only a relation reaches."""
    m = Model.from_spec(TAGGED_SPEC, WHERE_DATA)
    assert "label" in m.spec.parameters
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    assert set(p.spec.coords) == set(m.spec.coords)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_presence_mask_round_trips_and_is_compared(
    tmp_path: Path, engine: str
) -> None:
    m = Model.from_spec(
        with_(WHERE_SPEC, expressions={"twice": "cost * 2"}), WHERE_DATA
    )
    p = roundtrip(m, tmp_path, engine)
    assert_model_equal(m, p)

    p.spec.presence["cost"].values[:] = True
    with pytest.raises(AssertionError):
        assert_model_equal(m, p)


@pytest.mark.parametrize("deep", [True, False])
def test_a_copy_carries_the_spec(deep: bool) -> None:
    """The copy's spec reads the copy, and only a deep copy owns its buffers."""
    m = Model.from_spec(TAGGED_SPEC, WHERE_DATA)
    p = m.copy(deep=deep)
    p.spec.parameters["label"].values[1] = "changed"

    assert p.spec.text == m.spec.text
    assert p.spec.parameters["label"].values[1] == "changed"
    assert m.spec.parameters["label"].values[1] == ("u" if deep else "changed")


def test_a_model_without_a_spec_carries_none(tmp_path: Path) -> None:
    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(3, name="i")], name="x")
    m.add_objective(x.sum())
    path = tmp_path / "plain.nc"
    m.to_netcdf(path)

    assert "spec-text" not in xr.load_dataset(path).attrs
    assert read_netcdf(path)._spec is None
    assert m.copy()._spec is None


def rewritten(tmp_path: Path, m: Model, **attrs: Any) -> Path:
    """*m* written out, its file attributes updated with *attrs*, as another writer would have left them."""
    path = tmp_path / "current.nc"
    m.to_netcdf(path)
    ds = xr.load_dataset(path).assign_attrs(attrs)
    other = tmp_path / "other.nc"
    ds.to_netcdf(other)
    return other


def test_the_file_names_the_mathspec_version(tmp_path: Path) -> None:
    path = tmp_path / "model.nc"
    Model.from_spec(EXAMPLE_DISPATCH, DISPATCH_DATA).to_netcdf(path)
    assert xr.load_dataset(path).attrs["spec-mathspec"] == mathspec.__version__
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        read_netcdf(path)


def test_a_file_from_another_mathspec_version_warns_and_still_reads(
    tmp_path: Path,
) -> None:
    m = Model.from_spec(EXAMPLE_DISPATCH, DISPATCH_DATA)
    path = rewritten(tmp_path, m, **{"spec-mathspec": "0.0.0"})
    with pytest.warns(UserWarning, match="lowered by mathspec 0.0.0"):
        p = read_netcdf(path)
    assert_model_equal(m, p)


def test_a_spec_file_reads_as_a_plain_model_without_mathspec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m = Model.from_spec(EXAMPLE_DISPATCH, DISPATCH_DATA)
    path = tmp_path / "model.nc"
    m.to_netcdf(path)
    monkeypatch.setattr(linopy.io, "find_spec", lambda name: None)
    with pytest.warns(UserWarning, match="mathspec is not installed"):
        p = read_netcdf(path)

    assert p._spec is None
    assert list(p.variables) == list(m.variables)
    assert list(p.constraints) == list(m.constraints)
    assert_linequal(p.objective.expression, m.objective.expression)
    p.remove_constraints("power_balance")
    again = tmp_path / "plain.nc"
    p.to_netcdf(again)
    assert "spec-text" not in xr.load_dataset(again).attrs
    monkeypatch.undo()
    assert read_netcdf(again)._spec is None


@pytest.mark.skipif(
    EXAMPLES_DIR is None, reason="set MATHSPEC_EXAMPLES to a mathspec examples dir"
)
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("example", ["commitment.yaml", "operators/sum_by.yaml"])
def test_a_mathspec_example_round_trips(
    tmp_path: Path, engine: str, example: str
) -> None:
    path = Path(EXAMPLES_DIR or "", example)
    m = Model.from_spec(path, synthetic_sources(lower(str(path)), 3))
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    assert set(p.spec.coords) == set(m.spec.coords)


TRANSFORMS: dict[str, Callable[[Model, Path], Model]] = {
    "built": lambda m, tmp_path: m,
    "copy": lambda m, tmp_path: m.copy(),
    "netcdf": lambda m, tmp_path: roundtrip(m, tmp_path, "netcdf4"),
}


@pytest.mark.parametrize("transform", TRANSFORMS.values(), ids=TRANSFORMS.keys())
def test_the_stamp_names_the_spec_file_through_a_copy_and_a_round_trip(
    tmp_path: Path, backend: Model, transform: Callable[[Model, Path], Model]
) -> None:
    path = tmp_path / "dispatch.yaml"
    path.write_text(EXAMPLE_DISPATCH)
    p = transform(backend.add_spec(path, DISPATCH_DATA), tmp_path)
    assert p.spec.name == "dispatch"
    assert p.variables["p"].spec == "dispatch"
    assert p.constraints["power_balance"].spec == "dispatch"
    assert p.expressions["spend"].spec == "dispatch"
    assert p.objective.expression.spec == "dispatch"
    assert p.spec.expressions["spend"].expression is p.expressions["spend"]
