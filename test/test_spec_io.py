"""
Round trips of a spec-built model through netcdf and through ``copy``.

The spec itself is persisted as its YAML text and lowered again on read, so
what has to survive besides the model is data: the master coordinates, the
lookups and the retained parameters. Labels are the delicate part — a partial
lookup holds NaN in an array of strings — so every lookup shape is checked
value by value and dtype by dtype, on both netcdf engines ``test_io`` uses.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import xarray as xr

math_spec = pytest.importorskip("math_spec")

from test_spec_builder import (  # noqa: E402
    BASE_MODEL,
    DISPATCH_DATA,
    EXAMPLE_DISPATCH,
    EXAMPLES_DIR,
    GENERATOR,
    SECOND_DATA,
    SECOND_SPEC,
    SOS_SPEC,
    THREE,
    WHERE_DATA,
    WHERE_SPEC,
    extended,
    solved,
    subset_bound,
)

import linopy  # noqa: E402
from linopy import Model, read_netcdf  # noqa: E402
from linopy.io import (  # noqa: E402
    LAYER_BOUND_ATTR,
    LAYER_TEXT_ATTR,
    SPEC_ATTR,
    SPEC_LAYERS_ATTR,
    SPEC_OBJECTIVE_ATTR,
    SPEC_WHOLE_ATTR,
)
from linopy.spec import SpecDataError  # noqa: E402
from linopy.spec.netcdf import LEGACY_OBJECTIVE_ATTR  # noqa: E402
from linopy.spec.testing import synthetic_sources  # noqa: E402
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

DTYPE_SPEC: dict[str, Any] = {
    "dimensions": {"s1": {"dtype": "str"}},
    "parameters": {
        "count": {"dims": ["s1"], "dtype": "int"},
        "flag": {"dims": ["s1"], "dtype": "bool"},
        "cost": {"dims": ["s1"]},
        "tag": {"dims": ["s1"], "dtype": "str"},
    },
    "variables": {"x": {"foreach": ["s1"], "bounds": {"lower": 0, "upper": 1}}},
    "objective": {"sense": "minimize", "expression": "sum(x * cost)"},
}
DTYPE_DATA: dict[str, Any] = {
    "s1": S1,
    "count": pd.Series([1, 2, 3], index=S1),
    "flag": pd.Series([True, False, True], index=S1),
    "cost": pd.Series([1.0, 2.0, 3.0], index=S1),
    "tag": pd.Series(["u", "v", "w"], index=S1),
}


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


def whole() -> Model:
    """The dispatch example built from its spec, one layer describing the whole model."""
    return Model.from_spec(EXAMPLE_DISPATCH, DISPATCH_DATA, retain="all")


def layered(layers: int, whole_: bool) -> Model:
    """A model under *layers* spec layers, the first the whole model or extending a hand-built one."""
    if layers == 0:
        return BASE_MODEL()
    m = whole() if whole_ else extended()
    if layers == 2:
        data = {**SECOND_DATA, "p": m.variables["p"]}
        m.add_spec(SECOND_SPEC, data, retain="all", name="second")
    return m


def netcdf_path(tmp_path: Path, engine: str) -> Path:
    return tmp_path / f"model-{engine}.nc"


def roundtrip(m: Model, tmp_path: Path, engine: str) -> Model:
    path = netcdf_path(tmp_path, engine)
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
        assert_arrayequal(
            m.spec.expressions[name].solution, p.spec.expressions[name].solution
        )


@pytest.mark.parametrize("engine", ENGINES)
def test_a_retain_none_model_evaluates_after_a_round_trip(
    tmp_path: Path, engine: str
) -> None:
    """A file is where retain bites: the sources the built model still read are gone."""
    m = solved(EXAMPLE_DISPATCH, DISPATCH_DATA, retain="none")
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    assert not p.spec.parameters.data_vars
    with pytest.raises(SpecDataError, match="no longer holds the sources"):
        p.spec.expressions["spend"].solution
    assert_arrayequal(
        m.spec.expressions["spend"].solution,
        p.spec.evaluate("spend", DISPATCH_DATA).solution,
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_the_caller_parameters_and_the_spec_ones_stay_apart(
    tmp_path: Path, engine: str
) -> None:
    """A model parameter of the caller's is written beside the spec's, not into them."""
    m = solved(EXAMPLE_DISPATCH, DISPATCH_DATA, retain="all")
    m.parameters["cost"] = xr.DataArray([1, 2, 3], dims=["own"])
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    assert_arrayequal(p.parameters["cost"], m.parameters["cost"])
    assert_arrayequal(p.spec.parameters["cost"], m.spec.parameters["cost"])


@pytest.mark.parametrize("engine", ENGINES)
def test_a_replaced_objective_is_still_known_after_a_round_trip(
    tmp_path: Path, engine: str
) -> None:
    """Nothing else in the file would say the typeset objective is not the model's."""
    m = Model.from_spec(EXAMPLE_DISPATCH, DISPATCH_DATA, retain="all")
    m.add_objective(m.variables["p"].sum() * 2.0, overwrite=True)
    p = roundtrip(m, tmp_path, engine)

    assert m.spec.unspecified.objective
    assert p.spec.unspecified.objective
    assert p.spec.objective_owner is None
    assert_model_equal(m, p)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    ("layers", "whole_"),
    [(0, False), (1, True), (1, False), (2, True), (2, False)],
    ids=["none", "1-whole", "1-extended", "2-whole", "2-extended"],
)
def test_layers_round_trip(
    tmp_path: Path, engine: str, layers: int, whole_: bool
) -> None:
    """Every layer comes back in order with its text, its data and its bindings; a model without any writes none."""
    m = layered(layers, whole_)
    p = roundtrip(m, tmp_path, engine)
    attrs = xr.load_dataset(netcdf_path(tmp_path, engine)).attrs

    assert_model_equal(m, p)
    if layers == 0:
        assert p._spec is None
        assert not [k for k in attrs if k.startswith(SPEC_ATTR)]
        return
    assert list(p.spec.layers) == list(m.spec.layers)
    for name, layer in m.spec.layers.items():
        other = p.spec[name]
        assert other.text == layer.text
        assert attrs[LAYER_TEXT_ATTR.format(name)] == layer.text
        assert LAYER_BOUND_ATTR.format(name) in attrs
        assert dict(other.names) == dict(layer.names)
        assert set(other.coords) == set(layer.coords)
        for dim, index in layer.coords.items():
            assert other.coords[dim].equals(index)
            assert other.coords[dim].dtype == index.dtype
        for pname, arr in layer.parameters.items():
            assert_arrayequal(other.parameters[pname], arr)
    assert p.spec.whole is whole_
    assert p.spec.objective_owner == ("spec" if whole_ else None)


@pytest.mark.parametrize("replaced", [False, True], ids=["own", "replaced"])
def test_a_legacy_file_reads_as_one_layer(tmp_path: Path, replaced: bool) -> None:
    """A file written before layers existed reads as the single layer ``spec`` describing the whole model."""
    m = whole()
    if replaced:
        m.add_objective(m.variables["p"].sum() * 2.0, overwrite=True)
    path = tmp_path / "current.nc"
    m.to_netcdf(path)
    ds = xr.load_dataset(path)
    old = "spec-spec-"
    ds = ds.rename(
        {
            k: "spec-" + str(k)[len(old) :]
            for k in [*ds.data_vars, *ds.dims]
            if str(k).startswith(old)
        }
    )
    for attr in (
        SPEC_LAYERS_ATTR,
        SPEC_WHOLE_ATTR,
        SPEC_OBJECTIVE_ATTR,
        LAYER_TEXT_ATTR.format("spec"),
        LAYER_BOUND_ATTR.format("spec"),
    ):
        del ds.attrs[attr]
    ds.attrs[SPEC_ATTR] = m.spec.text
    ds.attrs[LEGACY_OBJECTIVE_ATTR] = int(replaced)
    legacy = tmp_path / "legacy.nc"
    ds.to_netcdf(legacy)
    p = read_netcdf(legacy)

    assert list(p.spec.layers) == ["spec"]
    assert p.spec.whole is True
    assert p.spec.objective_owner == (None if replaced else "spec")
    assert_model_equal(m, p)


@pytest.mark.parametrize("engine", ENGINES)
def test_a_bound_sos_variable_keeps_its_attrs_after_a_round_trip(
    tmp_path: Path, engine: str
) -> None:
    p = roundtrip(extended(SOS_SPEC), tmp_path, engine)

    assert p.variables["p"].attrs["sos_type"] == 1
    assert p.variables["p"].attrs["sos_dim"] == "generator"
    assert p.spec.unspecified.sos == ()


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
    assert name in p.spec.lookups[over]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", ["count", "flag", "cost", "tag"])
def test_a_parameter_keeps_its_dtype(tmp_path: Path, engine: str, name: str) -> None:
    m = Model.from_spec(DTYPE_SPEC, DTYPE_DATA, retain="all")
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    assert p.spec.parameters[name].dtype == m.spec.parameters[name].dtype


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("build", [whole, extended], ids=["whole", "extended"])
@pytest.mark.parametrize(
    "labels", [[10, 11, 12], [0, 1]], ids=["relabelled", "shorter"]
)
def test_a_hand_added_variable_keeps_its_own_labels(
    tmp_path: Path, engine: str, build: Callable[[], Model], labels: list[int]
) -> None:
    """A container sharing a master dimension's name but not its labels is left alone."""
    own = pd.Index(labels, name="snapshot")
    m = build()
    m.add_variables(coords=[own], name="side")
    p = roundtrip(m, tmp_path, engine)

    assert p.variables["side"].indexes["snapshot"].equals(own)
    assert p.spec.coords["snapshot"].equals(m.spec.coords["snapshot"])
    assert p.variables["p"].indexes["snapshot"].equals(m.spec.coords["snapshot"])


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "build, csr",
    [
        (
            functools.partial(solved, EXAMPLE_DISPATCH, DISPATCH_DATA, retain="all"),
            False,
        ),
        (
            functools.partial(
                solved,
                EXAMPLE_DISPATCH,
                DISPATCH_DATA,
                retain="all",
                freeze_constraints=True,
            ),
            True,
        ),
        (extended, False),
    ],
    ids=["dataset", "csr", "extended"],
)
def test_every_container_shares_the_master_coordinate_dtypes(
    tmp_path: Path, engine: str, build: Callable[[], Model], csr: bool
) -> None:
    """The master coordinates are canonical: no container may disagree with them, a bound one included."""
    if csr and engine == "scipy":
        pytest.skip(
            "netCDF3 holds no unicode-array attr, and a CSR constraint writes one"
        )
    m = build()
    p = roundtrip(m, tmp_path, engine)

    master = {dim: index.dtype for dim, index in p.spec.coords.items()}
    holders = [
        *(v.data for _, v in p.variables.items()),
        *(c.data for _, c in p.constraints.items()),
        p.objective.expression.data,
    ]
    assert master == {dim: index.dtype for dim, index in m.spec.coords.items()}
    for data in holders:
        for dim, index in data.indexes.items():
            if str(dim) in master:
                assert index.dtype == master[str(dim)], f"{dim} differs on {data}"


@pytest.mark.parametrize("engine", ENGINES)
def test_a_subset_bound_variable_keeps_its_own_index_and_still_folds(
    tmp_path: Path, engine: str
) -> None:
    """A bound variable spanning less than the master keeps its labels at the master's dtype, and is read onto the master again after the read."""
    m = subset_bound()
    m.solve(solver_name="highs", output_flag=False)
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    assert p.spec.coords["generator"].equals(THREE)
    own = p.variables["p"].indexes["generator"]
    assert (
        own.equals(GENERATOR)
        and own.dtype == m.variables["p"].indexes["generator"].dtype
    )
    assert_arrayequal(
        p.spec.expressions["total"].solution, m.spec.expressions["total"].solution
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_labelled_parameters_and_unreached_coordinates_round_trip(
    tmp_path: Path, engine: str
) -> None:
    """A str parameter with holes, and a dimension only a lookup reaches."""
    m = Model.from_spec(WHERE_SPEC, WHERE_DATA, retain="all")
    p = roundtrip(m, tmp_path, engine)

    assert_model_equal(m, p)
    assert set(p.spec.coords) == set(m.spec.coords)


@pytest.mark.parametrize("deep", [True, False])
def test_a_copy_carries_the_spec(deep: bool) -> None:
    """The copy's spec reads the copy, and only a deep copy owns its buffers."""
    m = Model.from_spec(WHERE_SPEC, WHERE_DATA, retain="all")
    p = m.copy(deep=deep)
    p.spec.parameters["label"].values[1] = "changed"

    assert p.spec.text == m.spec.text
    assert p.spec.parameters["label"].values[1] == "changed"
    assert m.spec.parameters["label"].values[1] == ("u" if deep else "changed")


@pytest.mark.parametrize("deep", [True, False])
def test_a_copy_carries_every_layer(deep: bool) -> None:
    """Both layers, their order, their bindings and the model-level state survive a copy."""
    m = layered(2, whole_=False)
    p = m.copy(deep=deep)

    assert_model_equal(m, p)
    assert list(p.spec.layers) == ["extra", "second"]
    assert dict(p.spec["second"].names) == {"p": "p"}
    assert p.spec["second"].model is p
    assert p.spec.whole is False
    assert p.spec.objective_owner is None


def test_a_copy_can_still_read_what_retain_dropped() -> None:
    """A copy keeps the sources, so it folds an unretained parameter like its original."""
    m = solved(EXAMPLE_DISPATCH, DISPATCH_DATA, retain="none")

    assert_arrayequal(
        m.copy(include_solution=True).spec.expressions["spend"].solution,
        m.spec.expressions["spend"].solution,
    )


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
    assert set(p.spec.coords) == set(m.spec.coords)
