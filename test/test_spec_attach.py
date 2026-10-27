"""Binding user data to a mathspec program."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("mathspec")

from spec_helpers import with_  # noqa: E402

from linopy.spec import SpecDataError  # noqa: E402
from linopy.spec.api import lower  # noqa: E402
from linopy.spec.attach import Attached, attach, fill  # noqa: E402

SPEC: dict[str, Any] = {
    "dimensions": {
        "f": {"dtype": "str"},
        "t": {"dtype": "int", "ordered": True},
        "g": {"dtype": "str"},
    },
    "relations": {"grp": {"key": "f", "values": "g", "missing": "absent"}},
    "parameters": {
        "cost": {"dims": ["f"], "missing": "absent"},
        "cap": {"dims": ["f", "t"], "missing": "absent"},
        "flag": {"dims": ["f"], "dtype": "bool", "missing": "absent"},
        "rate": {"dims": []},
        "lead": {"dims": ["f"], "dtype": "int", "missing": "absent"},
    },
    "variables": {
        "x": {
            "dims": ["f", "t"],
            "where": "flag",
            "bounds": {"lower": 0, "upper": "cap"},
        }
    },
    "constraints": {
        "k": {"dims": ["g", "t"], "expression": "sum(x, over=f, by=grp[g]) <= 10"},
        "s": {
            "dims": ["f", "t"],
            "expression": "shift(x, along=t, offset=lead, edge=0) >= 0",
        },
    },
    "objective": {"sense": "maximize", "expression": "sum(x * cost)"},
    "expressions": {
        "spend": "sum(x * cost, over=t)",
        "total": "sum(spend, over=f) * rate",
    },
}

F = pd.Index(["b", "a", "c"], name="f")
T = pd.Index([0, 1, 2], name="t")
CAP = xr.DataArray(np.arange(9.0).reshape(3, 3), coords={"f": F, "t": T}, name="cap")
COST = pd.Series([1.0, 2.0, 3.0], index=F)

DUP_ROWS = pd.Series([1.0, 9.0, 2.0], index=pd.Index(["a", "a", "b"], name="f"))
NULL_ROW = pd.Series({"a": 1.0, "b": None})
NAN_ROW = pd.Series({"a": 1.0, "b": np.nan})
NULL_FRAME = pd.DataFrame({"f": ["a", "b"], "value": [1.0, None]})
STRAY_ROW = pd.Series({"a": 1.0, "zz": 2.0})
DEEP_INDEX = pd.MultiIndex.from_tuples([("a", 0), ("b", 0)], names=["f", "k"])
DEEP_ROWS = pd.Series([5.0, 5.0], index=DEEP_INDEX)


def sources_from(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """*base* with *override* applied; a ``None`` value drops the key instead."""
    merged = {**base, **override}
    for key, value in override.items():
        if value is None:
            merged.pop(key)
    return merged


@pytest.fixture(scope="module")
def program() -> Any:
    return lower(SPEC)


@pytest.fixture
def good() -> dict[str, Any]:
    return {
        "f": list(F),
        "t": list(T),
        "g": ["n", "e"],
        "cost": COST,
        "cap": CAP,
        "flag": pd.Series([True, False, True], index=F),
        "rate": 0.5,
        "lead": pd.Series([1, 0, 1], index=F),
        "grp": pd.Series(["n", "e", "n"], index=F),
    }


def filled(attached: Attached, name: str) -> xr.DataArray:
    return fill(attached.program, name, attached.read(name))


def read_all(
    program: Any, sources: Mapping[str, Any] | xr.Dataset
) -> list[xr.DataArray]:
    attached = attach(program, sources)
    return [filled(attached, name) for name in program.parameters]


def as_dataset(sources: Mapping[str, Any]) -> xr.Dataset:
    dims = {d: sources[d] for d in SPEC["dimensions"]}
    values = {k: xr.DataArray(v) for k, v in sources.items() if k not in dims}
    return xr.Dataset(values, coords=dims)


CAP_SHAPES = {
    "dataarray": CAP,
    "dataarray-transposed": CAP.transpose("t", "f"),
    "series": CAP.to_series(),
    "series-transposed": CAP.transpose("t", "f").to_series(),
    "series-unnamed-levels": CAP.to_series().rename_axis([None, None]),
    "tidy-frame": CAP.to_series().reset_index(name="value"),
    "tidy-frame-extra-column": CAP.to_series()
    .reset_index(name="value")
    .assign(note="x"),
    "tidy-frame-named-column": CAP.to_series().reset_index(name="cap"),
    "tidy-frame-named-column-beside-value": CAP.to_series()
    .reset_index(name="cap")
    .assign(value=-1.0),
    "wide-frame": CAP.to_pandas(),
    "wide-frame-transposed": CAP.to_pandas().T,
    "dict": CAP.to_series().to_dict(),
}


@pytest.mark.parametrize("cap", CAP_SHAPES.values(), ids=CAP_SHAPES.keys())
def test_rank_two_shapes_attach_alike(
    program: Any, good: dict[str, Any], cap: Any
) -> None:
    got = filled(attach(program, {**good, "cap": cap}), "cap")
    xr.testing.assert_equal(got, CAP)
    assert got.dims == ("f", "t")


OBLONG_SPEC: dict[str, Any] = {
    "dimensions": {"f": {"dtype": "str"}, "t": {"dtype": "int", "ordered": True}},
    "parameters": {"cap": {"dims": ["f", "t"]}},
    "variables": {"x": {"dims": ["f", "t"]}},
    "objective": {"sense": "maximize", "expression": "sum(x * cap)"},
}


def test_an_unnamed_wide_frame_of_unequal_widths_is_read_by_position() -> None:
    f, t = pd.Index(["a", "b"], name="f"), pd.Index([0, 1, 2], name="t")
    values = np.arange(6.0).reshape(2, 3)
    frame = pd.DataFrame(values, index=list(f), columns=list(t))
    sources = {"f": list(f), "t": list(t), "cap": frame}
    got = filled(attach(lower(OBLONG_SPEC), sources), "cap")
    xr.testing.assert_equal(
        got, xr.DataArray(values, coords={"f": f, "t": t}, name="cap")
    )


COST_SHAPES = {
    "series": COST,
    "series-unnamed": COST.rename_axis(None),
    "dataarray": xr.DataArray(COST),
    "dict": COST.to_dict(),
    "whole-numbers": COST.astype(int),
    "tidy-frame": COST.reset_index(name="value"),
    "indexed-frame": COST.to_frame("value"),
    "indexed-frame-named-column": COST.to_frame("cost"),
}


@pytest.mark.parametrize("cost", COST_SHAPES.values(), ids=COST_SHAPES.keys())
def test_rank_one_shapes_attach_alike(
    program: Any, good: dict[str, Any], cost: Any
) -> None:
    got = filled(attach(program, {**good, "cost": cost}), "cost")
    xr.testing.assert_equal(got, xr.DataArray(COST, name="cost"))


def test_a_square_long_frame_is_read_long(program: Any, good: dict[str, Any]) -> None:
    diagonal = pd.DataFrame({"f": list(F), "t": list(T), "cap": [1.0, 2.0, 3.0]})
    values, present = attach(program, {**good, "cap": diagonal}).read("cap")
    np.testing.assert_array_equal(present.values, np.eye(3, dtype=bool))
    np.testing.assert_array_equal(np.diag(values.values), [1.0, 2.0, 3.0])


def test_a_hole_in_a_wide_frame_is_a_missing_row(
    program: Any, good: dict[str, Any]
) -> None:
    wide = CAP.to_pandas()
    wide.loc["a", 1] = np.nan
    values, present = attach(program, {**good, "cap": wide}).read("cap")
    xr.testing.assert_equal(present, CAP != 4.0)
    xr.testing.assert_equal(values, CAP.where(CAP != 4.0))


DIMENSION_SHAPES = {
    "index": F,
    "list": list(F),
    "tuple": tuple(F),
    "ndarray": F.to_numpy(),
    "series": pd.Series(F),
    "dataarray": xr.DataArray(list(F), dims=["f"]),
}


@pytest.mark.parametrize("f", DIMENSION_SHAPES.values(), ids=DIMENSION_SHAPES.keys())
def test_dimension_shapes_keep_source_order(
    program: Any, good: dict[str, Any], f: Any
) -> None:
    coords = attach(program, {**good, "f": f}).coords
    assert coords["f"].tolist() == ["b", "a", "c"]
    assert coords["f"].name == "f"
    assert list(coords) == ["f", "t", "g"]


def arrow_strings() -> Any:
    pytest.importorskip("pyarrow")
    return pd.array(list(F), dtype=pd.StringDtype("pyarrow"))


DIMENSION_SOURCES: dict[str, tuple[str, Callable[[], Any], list[Any]]] = {
    "set": ("f", lambda: set(F), ["a", "b", "c"]),
    "frozenset": ("f", lambda: frozenset(F), ["a", "b", "c"]),
    "string-python": ("f", lambda: pd.array(list(F), dtype="string"), list(F)),
    "string-pyarrow": ("f", arrow_strings, list(F)),
    "categorical": ("f", lambda: pd.Categorical(list(F)), list(F)),
    "masked-int": ("t", lambda: pd.array(list(T), dtype="Int64"), list(T)),
}


@pytest.mark.parametrize("case", DIMENSION_SOURCES)
def test_dimension_sources_beyond_sequences(
    program: Any, good: dict[str, Any], case: str
) -> None:
    dim, make, expected = DIMENSION_SOURCES[case]
    attached = attach(program, {**good, dim: make()})
    assert attached.coords[dim].tolist() == expected
    xr.testing.assert_equal(filled(attached, "cap"), CAP.sel({dim: expected}))


def test_relation_is_padded_onto_the_dimension(
    program: Any, good: dict[str, Any]
) -> None:
    attached = attach(program, {**good, "grp": {"a": "n"}})
    grp = attached.relations["grp"]
    assert grp.dims == ("f",)
    assert grp.sel(f="a").item() == "n"
    assert pd.isna(grp.sel(f=["b", "c"])).all()


RELATION_SHAPES = {
    "series": pd.Series(["n", "e", "n"], index=F),
    "series-unnamed": pd.Series(["n", "e", "n"], index=F.rename(None)),
    "dict": {"b": "n", "a": "e", "c": "n"},
    "dataarray": xr.DataArray(["n", "e", "n"], coords={"f": F}),
}


@pytest.mark.parametrize("grp", RELATION_SHAPES.values(), ids=RELATION_SHAPES.keys())
def test_relation_shapes_attach_alike(
    program: Any, good: dict[str, Any], grp: Any
) -> None:
    got = attach(program, {**good, "grp": grp}).relations["grp"]
    assert got.values.tolist() == ["n", "e", "n"]


@pytest.mark.parametrize("storage", ["python", "pyarrow"])
@pytest.mark.parametrize("shape", ["series", "dataarray"])
def test_extension_strings_attach_as_numpy_objects(
    program: Any, good: dict[str, Any], storage: str, shape: str
) -> None:
    if storage == "pyarrow":
        pytest.importorskip("pyarrow")
    series = pd.Series(["n", "e"], index=F[:2], dtype=pd.StringDtype(storage))
    grp = xr.DataArray(series) if shape == "dataarray" else series
    got = attach(program, {**good, "grp": grp}).relations["grp"]
    assert got.dtype == np.dtype(object)
    assert got.values[:2].tolist() == ["n", "e"]
    assert pd.isna(got.values[2])
    assert got.sel(f=["b", "a"]).values.tolist() == ["n", "e"]


@pytest.mark.parametrize("storage", ["python", "pyarrow"])
@pytest.mark.parametrize("shape", ["series", "tidy-frame", "dict", "dataarray"])
def test_string_parameter_rows_attach_as_numpy_objects(
    good: dict[str, Any], storage: str, shape: str
) -> None:
    if storage == "pyarrow":
        pytest.importorskip("pyarrow")
    tagged = with_(
        SPEC, parameters={"tag": {"dims": ["f"], "dtype": "str", "missing": "absent"}}
    )
    series = pd.Series(["n", "e"], index=F[:2], dtype=pd.StringDtype(storage))
    shapes = {
        "series": series,
        "tidy-frame": series.reset_index(name="value"),
        "dict": series.to_dict(),
        "dataarray": xr.DataArray(series),
    }
    values, present = attach(lower(tagged), {**good, "tag": shapes[shape]}).read("tag")
    assert values.dtype == np.dtype(object)
    assert values.values[:2].tolist() == ["n", "e"]
    assert pd.isna(values.values[2])
    assert present.values.tolist() == [True, True, False]


def test_missing_rows_become_nan_and_false(program: Any, good: dict[str, Any]) -> None:
    sparse = {
        **good,
        "lead": pd.Series({"a": 1}),
        "flag": pd.Series({"a": True}),
        "cap": CAP.sel(t=[0, 1]),
    }
    attached = attach(program, sparse)
    lead = filled(attached, "lead")
    assert lead.dtype == np.float64
    assert lead.sel(f="a").item() == 1.0
    assert lead.sel(f=["b", "c"]).isnull().all()
    unfilled = attached.read("flag").values
    assert unfilled.dtype == np.float64
    np.testing.assert_array_equal(unfilled.values, [np.nan, 1.0, np.nan])
    flag = filled(attached, "flag")
    assert flag.dtype == bool
    assert flag.values.tolist() == [False, True, False]
    cap = filled(attached, "cap")
    assert cap.dims == ("f", "t")
    assert cap.sel(t=2).isnull().all()


@pytest.mark.parametrize(
    ("name", "value", "expected_dtype"),
    [
        ("cost", 2, float),
        ("cap", 1.5, float),
        ("flag", True, bool),
        ("lead", 3, np.int64),
    ],
)
def test_scalar_is_broadcast_over_declared_dims(
    program: Any, good: dict[str, Any], name: str, value: Any, expected_dtype: Any
) -> None:
    got = filled(attach(program, {**good, name: value}), name)
    assert got.dims == tuple(SPEC["parameters"][name]["dims"])
    assert got.dtype == expected_dtype
    assert (got == value).all()


def test_scalar_parameter_stays_scalar(program: Any, good: dict[str, Any]) -> None:
    got = filled(attach(program, good), "rate")
    assert got.dims == ()
    assert got.item() == 0.5


EMPTY_SOURCES = {
    "dict": {},
    "object-series": pd.Series(dtype=object),
    "float-series": pd.Series(dtype=float),
}


MODES_DATA: dict[str, Any] = {"f": ["a", "b"], "cost": {"a": 1.0}}


def with_missing(**missing: Any) -> Any:
    return lower(
        {
            "dimensions": {"f": {"dtype": "str"}},
            "parameters": {"cost": {"dims": ["f"], **missing}},
            "variables": {"x": {"dims": ["f"], "bounds": {"lower": 0, "upper": 1}}},
            "objective": {"sense": "maximize", "expression": "sum(x * cost)"},
        }
    )


@pytest.mark.parametrize(
    ("missing", "expected"),
    [
        ("refused", [1.0, np.nan]),
        ("absent", [1.0, np.nan]),
        ("neutral", [1.0, 0.0]),
        (7.5, [1.0, 7.5]),
    ],
)
def test_a_missing_row_reads_the_way_missing_says(
    missing: Any, expected: list[float]
) -> None:
    attached = attach(with_missing(missing=missing), MODES_DATA)
    np.testing.assert_array_equal(filled(attached, "cost").values, expected)
    assert attached.read("cost").present.values.tolist() == [True, False]


STR_HOLE = np.array(["x", np.nan, "z"], dtype=object)
HOLES = [
    pytest.param("float", "absent", [1.0, np.nan, 3.0], id="float"),
    pytest.param("float", "neutral", [1.0, 0.0, 3.0], id="float-0"),
    pytest.param("int", "absent", [1.0, np.nan, 3.0], id="int"),
    pytest.param("int", "neutral", [1, 0, 3], id="int-0"),
    pytest.param("int", 7, [1, 7, 3], id="int-7"),
    pytest.param("bool", "absent", [True, False, False], id="bool"),
    pytest.param("str", "absent", STR_HOLE, id="str"),
]
HELD = {
    "float": [1.0, np.nan, 3.0],
    "int": [1.0, np.nan, 3.0],
    "bool": [1.0, np.nan, 0.0],
}


@pytest.mark.parametrize("source", ["dataarray", "dataset"])
@pytest.mark.parametrize(("dtype", "missing", "expected"), HOLES)
def test_a_hole_in_a_labelled_source_is_a_missing_row(
    good: dict[str, Any], dtype: str, missing: Any, expected: Any, source: str
) -> None:
    spec = with_(
        SPEC, parameters={"p": {"dims": ["f"], "dtype": dtype, "missing": missing}}
    )
    sources = {**good, "p": xr.DataArray(HELD.get(dtype, STR_HOLE), coords={"f": F})}
    attached = attach(
        lower(spec), as_dataset(sources) if source == "dataset" else sources
    )
    assert attached.read("p").present.values.tolist() == [True, False, True]
    got = filled(attached, "p")
    assert got.dtype == np.asarray(expected).dtype
    xr.testing.assert_equal(got, xr.DataArray(expected, coords={"f": F}))


UNSUPPORTED: dict[str, tuple[dict[str, Any], str]] = {
    "given": ({"given": {"expressions": {"e": {"dims": ["f"]}}}}, "given:"),
    "multi-key": (
        {"relations": {"grp": {"key": ["f", "t"], "values": "g"}}},
        "one key column mapping into one value column",
    ),
    "bare": (
        {"relations": {"grp": {"key": ["f", "g"]}}},
        "one key column mapping into one value column",
    ),
}


@pytest.mark.parametrize("case", UNSUPPORTED)
def test_a_program_linopy_cannot_attach_is_refused(
    good: dict[str, Any], case: str
) -> None:
    override, match = UNSUPPORTED[case]
    spec = {**SPEC, "constraints": {}, **override}
    with pytest.raises(SpecDataError, match=match):
        attach(lower(spec), good)


@pytest.mark.parametrize("cost", EMPTY_SOURCES.values(), ids=EMPTY_SOURCES.keys())
def test_empty_source_attaches_as_all_nan(
    program: Any, good: dict[str, Any], cost: Any
) -> None:
    got = filled(attach(program, {**good, "cost": cost}), "cost")
    assert got.dtype == np.float64
    assert got.isnull().all()
    assert got.indexes["f"].equals(F)


def test_missing_parameter_is_refused_when_read(
    program: Any, good: dict[str, Any]
) -> None:
    good.pop("cost")
    attached = attach(program, good)
    with pytest.raises(SpecDataError, match="no data provided for parameter 'cost'"):
        attached.read("cost")


def test_undeclared_parameter_is_refused_with_a_hint(
    program: Any, good: dict[str, Any]
) -> None:
    with pytest.raises(KeyError, match="unknown parameter 'csot'.*'cost'"):
        attach(program, good).read("csot")


REFUSALS = [
    pytest.param(
        {"f": ["a", "a", "b"]},
        r"dimension 'f' lists 'a' more than once",
        id="duplicate-member",
    ),
    pytest.param(
        {"cost": STRAY_ROW}, r"parameter 'cost'.*'f'.*'zz'", id="unknown-label"
    ),
    pytest.param(
        {"cap": CAP.assign_coords(t=[0, 1, 9])},
        r"parameter 'cap'.*'t'.*\b9\b",
        id="unknown-label-dense",
    ),
    pytest.param(
        {"cap": CAP.assign_coords(t=[9, 0, 7])},
        r"not coordinates of it: 9, 7\.",
        id="unknown-labels-dense-in-source-order",
    ),
    pytest.param(
        {"cap": CAP.to_series().reset_index(name="value").assign(t=[9, 0, 7] * 3)},
        r"not coordinates of it: 9, 7\.",
        id="unknown-labels-rows-in-source-order",
    ),
    pytest.param(
        {"cap": CAP.assign_coords(t=[0.0, 1.0, 5.0])},
        r"not coordinates of it: 5\.0\.",
        id="unknown-float-label-among-matching-ones",
    ),
    pytest.param(
        {"cost": DUP_ROWS},
        r"parameter 'cost' has more than one row for a coordinate: f='a' \(2 rows\)",
        id="duplicated-coordinate-row",
    ),
    pytest.param(
        {"cap": xr.DataArray([1.0, 2.0], coords={"f": ["a", "a"]})},
        r"parameter 'cap' arrived as a DataArray over \['f'\]",
        id="dense-wrong-dims",
    ),
    pytest.param(
        {"cap": xr.DataArray(np.ones((3, 3)), coords={"f": list(F), "t": [0, 0, 1]})},
        r"parameter 'cap' has more than one row for a coordinate: t=0 \(2 rows\)",
        id="dense-duplicate-coordinate",
    ),
    pytest.param(
        {"cap": COST},
        r"parameter 'cap'.*1 level\(s\) where 'cap' is over \['f', 't'\]",
        id="wrong-rank",
    ),
    pytest.param(
        {"cap": DEEP_ROWS.rename_axis(["f", "q"])},
        r"parameter 'cap' is indexed by \['f', 'q'\]",
        id="wrong-level-names",
    ),
    pytest.param(
        {"rate": COST},
        r"parameter 'rate' is declared with no dims.*3 rows",
        id="rows-for-scalar",
    ),
    pytest.param(
        {"cost": {"a", "b"}},
        r"parameter 'cost': cannot adapt set",
        id="unsupported-shape",
    ),
    pytest.param(
        {"cost": pd.DataFrame({"f": ["a"], "amount": [1.0]})},
        r"parameter 'cost' arrived as a DataFrame with columns \['f', 'amount'\]",
        id="frame-without-value-column",
    ),
    pytest.param(
        {"cap": CAP.to_pandas().rename_axis(index="f", columns="q")},
        r"parameter 'cap' arrived as a wide DataFrame with index 'f' and columns 'q'",
        id="wide-frame-wrong-axis-names",
    ),
    pytest.param(
        {"cap": CAP.to_pandas().rename_axis(index=None, columns=None)},
        r"parameter 'cap' arrived as a 3x3 wide DataFrame with neither axis named.*square "
        r"frame does not say which axis is which",
        id="wide-frame-unnamed-square",
    ),
    pytest.param(
        {"lead": xr.DataArray([1.0, 0.0, 1.0], coords={"f": F})},
        r"'lead' is declared 'int'.*'float'",
        id="float-grid-without-hole-for-int",
    ),
    pytest.param(
        {"lead": xr.DataArray([1.5, np.nan, 1.0], coords={"f": F})},
        r"'lead' is declared 'int'.*1\.5 are not whole numbers",
        id="fraction-in-a-grid-with-a-hole-for-int",
    ),
    pytest.param(
        {"flag": xr.DataArray([2.0, np.nan, 1.0], coords={"f": F})},
        r"'flag' is declared 'bool'.*2\.0 are not 0 or 1",
        id="two-in-a-grid-with-a-hole-for-bool",
    ),
    pytest.param(
        {"cap": CAP.assign_coords(t=["0", "1", "2"])},
        r"parameter 'cap' has 'str' label\(s\) in dimension 't'.*differ in type, not spelling",
        id="str-labels-on-an-int-dimension",
    ),
    pytest.param(
        {"cap": xr.DataArray(np.ones((3, 3)), dims=["f", "t"])},
        r"parameter 'cap' has no coordinate labels along 'f'",
        id="dense-without-labels",
    ),
    pytest.param(
        {"cost": NULL_ROW}, r"parameter 'cost' carries 1 row.*f='b'", id="null-row"
    ),
    pytest.param({"cost": NAN_ROW}, r"parameter 'cost' carries 1 row", id="nan-row"),
    pytest.param(
        {"cost": {"a": 1.0, "b": None}},
        r"parameter 'cost' carries 1 row",
        id="null-in-a-dict",
    ),
    pytest.param(
        {"cost": NULL_FRAME},
        r"parameter 'cost' carries 1 row",
        id="null-in-a-tidy-frame",
    ),
    pytest.param(
        {"cost": [1.0, np.nan, 2.0]},
        r"parameter 'cost': cannot adapt list",
        id="sequence",
    ),
    pytest.param(
        {"cost": DEEP_ROWS},
        r"parameter 'cost'.*2 level\(s\) where 'cost' is over \['f'\]",
        id="series-too-deep",
    ),
    pytest.param(
        {"rate": float("nan")},
        r"parameter 'rate' is one value and that value is a hole",
        id="nan-scalar",
    ),
    pytest.param(
        {"rate": pd.DataFrame({"value": [None]})},
        r"parameter 'rate' is one value and that value is a hole",
        id="nan-scalar-frame",
    ),
    pytest.param(
        {"lead": pd.Series([1.5, 0.0, 1.0], index=F)},
        r"'lead' is declared 'int'.*'float'",
        id="float-for-int",
    ),
    pytest.param(
        {"flag": pd.Series([1, 0, 1], index=F)},
        r"'flag' is declared 'bool'.*'int'",
        id="int-for-bool",
    ),
    pytest.param(
        {"flag": 1.0}, r"'flag' is declared 'bool'.*'float'", id="float-scalar-for-bool"
    ),
    pytest.param(
        {"rate": "1.5"}, r"'rate' is declared 'float'.*'str'", id="numeric-str-scalar"
    ),
    pytest.param(
        {"rate": True},
        r"'rate' is declared 'float'.*'bool'",
        id="bool-scalar-for-float",
    ),
    pytest.param(
        {"rate": "abc"}, r"'rate' is declared 'float'.*'str'", id="str-scalar-for-float"
    ),
    pytest.param(
        {"cost": pd.Series(["x", "y", "z"], index=F)},
        r"'cost' is declared 'float'.*'str'",
        id="str-for-float",
    ),
    pytest.param({"f": None}, r"dimension 'f' has no index", id="missing-dimension"),
    pytest.param(
        {"f": {"a", 1}},
        r"index for dimension 'f' arrived as a set.*do not sort",
        id="dimension-set-that-does-not-sort",
    ),
    pytest.param(
        {"f": {"a": 1}},
        r"index for dimension 'f': cannot read labels out of dict",
        id="dimension-shape",
    ),
    pytest.param(
        {"f": np.ones((2, 2))},
        r"index for dimension 'f' is 2-dimensional",
        id="dimension-rank",
    ),
    pytest.param(
        {"f": DEEP_INDEX},
        r"index for dimension 'f' is a MultiIndex; a spec dimension is one flat axis",
        id="dimension-multiindex",
    ),
    pytest.param(
        {"f": [("a", 0), ("b", 1)]},
        r"index for dimension 'f' is a MultiIndex",
        id="dimension-tuples",
    ),
    pytest.param(
        {"f": [1, 2, 3]},
        r"dimension 'f' is declared 'str'.*'int'",
        id="dimension-int-for-str",
    ),
    pytest.param(
        {"t": ["a", "b", "c"]},
        r"dimension 't' is declared 'int'.*'str'",
        id="dimension-str-for-int",
    ),
    pytest.param(
        {"grp": xr.DataArray(["n"], coords={"t": [0]})},
        r"relation 'grp' arrived as a DataArray over \['t'\]",
        id="relation-wrong-dataarray-dim",
    ),
    pytest.param(
        {"grp": None}, r"no data provided for relation 'grp'", id="missing-relation"
    ),
    pytest.param(
        {"grp": {"zz": "n"}},
        r"relation 'grp' maps 'zz', which are not labels of 'f'",
        id="relation-stray-key",
    ),
    pytest.param(
        {"grp": {1: "n"}},
        r"relation 'grp' has 'int' label\(s\) in dimension 'f'.*differ in type",
        id="relation-int-key-on-a-str-dimension",
    ),
    pytest.param(
        {"grp": {"a": "zz"}},
        r"relation 'grp' has value\(s\) that are not 'g' labels: 'zz'",
        id="relation-stray-value",
    ),
    pytest.param(
        {"grp": pd.Series(["n", "e"], index=pd.Index(["a", "a"], name="f"))},
        r"relation 'grp' maps 1 'f' label\(s\) more than once: 'a'",
        id="relation-two-values",
    ),
    pytest.param(
        {"grp": {"a": None, "b": "n"}},
        r"relation 'grp' carries 1 row\(s\) with a null in 'g': f='a'",
        id="relation-null",
    ),
    pytest.param(
        {"grp": pd.DataFrame({"f": ["a"], "g": ["n"]})},
        r"relation 'grp': cannot adapt DataFrame",
        id="relation-shape",
    ),
    pytest.param(
        {"grp": pd.Series(["n"], index=pd.Index(["a"], name="t"))},
        r"relation 'grp' is a Series indexed by 't'",
        id="relation-wrong-index",
    ),
]


@pytest.mark.parametrize(("override", "match"), REFUSALS)
def test_malformed_data_is_refused_naming_the_symbol(
    program: Any, good: dict[str, Any], override: dict[str, Any], match: str
) -> None:
    with pytest.raises(SpecDataError, match=match):
        read_all(program, sources_from(good, override))


def test_int_labels_are_shown_as_written(program: Any, good: dict[str, Any]) -> None:
    with pytest.raises(SpecDataError, match=r"\b99\b") as error:
        read_all(program, {**good, "cap": CAP.assign_coords(t=[0, 1, 99])})
    assert "int64" not in str(error.value)


def test_dataset_is_a_source(program: Any, good: dict[str, Any]) -> None:
    from_dataset = attach(program, as_dataset(good))
    from_mapping = attach(program, good)
    assert from_dataset.coords["f"].equals(from_mapping.coords["f"])
    for name in program.parameters:
        xr.testing.assert_equal(filled(from_dataset, name), filled(from_mapping, name))
    xr.testing.assert_equal(
        from_dataset.relations["grp"], from_mapping.relations["grp"]
    )


class Unlisted(Mapping[str, Any]):
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.reads: list[str] = []

    def __getitem__(self, key: str) -> Any:
        self.reads.append(key)
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("sources must not be iterated")

    def __len__(self) -> int:
        return len(self.data)

    def keys(self) -> Any:
        return self.data.keys()


def test_sources_are_read_by_key_once_never_iterated(
    program: Any, good: dict[str, Any]
) -> None:
    source = Unlisted(good)
    read_all(program, source)
    assert Counter(source.reads) == dict.fromkeys(good, 1)


def test_a_source_key_close_to_a_declared_name_warns(
    program: Any, good: dict[str, Any]
) -> None:
    sources = sources_from(good, {"csot": COST, "cost": None})
    with pytest.warns(UserWarning, match=r"csot -> cost"):
        attached = attach(program, sources)
    with pytest.raises(SpecDataError, match="no data provided for parameter 'cost'"):
        attached.read("cost")


def test_a_source_key_like_no_declared_name_is_silent(
    program: Any, good: dict[str, Any], recwarn: Any
) -> None:
    attach(program, {**good, "notes": COST})
    assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []


def test_unreached_dimension_needs_no_source(good: dict[str, Any]) -> None:
    dimensions = {**SPEC["dimensions"], "z": {"dtype": "int"}}
    program = lower({**SPEC, "dimensions": dimensions})
    assert list(attach(program, good).coords) == ["f", "t", "g"]


def test_master_coordinate_dtype_wins(program: Any, good: dict[str, Any]) -> None:
    source = CAP.assign_coords(t=T.astype("int32"))
    got = filled(attach(program, {**good, "cap": source}), "cap")
    assert got.indexes["t"].dtype == np.int64


ACCEPTED = "accepted"
FLAG_SPEC = {
    "dimensions": {"g": {"dtype": "str"}},
    "parameters": {"active": {"dims": ["g"], "dtype": "bool"}},
    "variables": {
        "x": {"dims": ["g"], "where": "active", "bounds": {"lower": 0, "upper": 1}}
    },
    "objective": {"sense": "maximize", "expression": "sum(x)"},
}


@pytest.mark.parametrize(
    ("column", "verdict"),
    [
        pytest.param(pd.Series({"a": True, "b": False}), ACCEPTED, id="a-bool-column"),
        pytest.param(pd.Series({"a": 1, "b": 0}), SpecDataError, id="a-1-0-int-column"),
        pytest.param(
            pd.Series({"a": 1.0, "b": 0.0}), SpecDataError, id="a-1-0-float-column"
        ),
    ],
)
def test_a_flag_attaches_by_its_declaration(column: pd.Series, verdict: Any) -> None:
    program = lower(FLAG_SPEC)
    sources = {"g": ["a", "b"], "active": column}
    if verdict is ACCEPTED:
        assert filled(attach(program, sources), "active").dtype == bool
        return
    with pytest.raises(SpecDataError, match="declared 'bool'"):
        read_all(program, sources)


RELATION_SPEC = {
    "dimensions": {"g": {}, "b": {"dtype": "str"}},
    "relations": {"gen_bus": {"key": "g", "values": "b", "missing": "absent"}},
    "parameters": {"p_max": {"dims": ["g"]}},
    "variables": {"x": {"dims": ["g"], "bounds": {"lower": 0, "upper": "p_max"}}},
    "constraints": {
        "k": {"dims": ["b"], "expression": "sum(x, over=g, by=gen_bus[b]) <= 10"}
    },
    "objective": {"sense": "maximize", "expression": "sum(x)"},
}
G_TWICE = pd.Index(["w", "w", "s"], name="g")
RELATION_GOOD = {
    "p_max": pd.Series({"w": 5.0, "s": 5.0}),
    "g": ["w", "s"],
    "b": ["n", "e"],
    "gen_bus": pd.Series({"w": "n", "s": "e"}),
}


@pytest.mark.parametrize(
    ("override", "match"),
    [
        pytest.param(
            {"g": None, "b": None}, "dimension 'g' has no index", id="a-map-no-labels"
        ),
        pytest.param(
            {"gen_bus": None}, "no data provided for relation", id="an-index-no-map"
        ),
        pytest.param(
            {"gen_bus": pd.Series({"w": "n", "s": "zz"})},
            "not 'b' labels",
            id="a-stray-value",
        ),
        pytest.param(
            {"gen_bus": pd.Series(["n", "e", "e"], index=G_TWICE)},
            "more than once",
            id="two-values-for-one-label",
        ),
        pytest.param(
            {"gen_bus": pd.Series({"w": None, "s": "e"})},
            "null in 'b'",
            id="mapping-a-label-to-nothing",
        ),
        pytest.param(
            {"gen_bus": pd.Series([None, "n", "n"], index=G_TWICE)},
            "null in 'b'",
            id="a-label-held-twice-with-a-null",
        ),
    ],
)
def test_a_relation_defect_is_refused(override: dict[str, Any], match: str) -> None:
    program = lower(RELATION_SPEC)
    with pytest.raises(SpecDataError, match=match):
        read_all(program, sources_from(RELATION_GOOD, override))


def test_a_relation_with_nan_in_a_dataarray_is_partial(good: dict[str, Any]) -> None:
    grp = xr.DataArray(np.array(["n", np.nan, "n"], dtype=object), coords={"f": F})
    got = attach(lower(SPEC), {**good, "grp": grp}).relations["grp"]
    expected = attach(lower(SPEC), {**good, "grp": {"b": "n", "c": "n"}}).relations[
        "grp"
    ]
    xr.testing.assert_identical(got, expected)
    refused = with_(SPEC, relations={"grp": {"key": "f", "values": "g"}})
    with pytest.raises(
        SpecDataError, match=r"relation 'grp' has no row for 1 'f' label"
    ):
        attach(lower(refused), {**good, "grp": grp})


def test_datetime_labels_given_as_text_are_a_type_mismatch() -> None:
    snapshot = pd.date_range("2025-01-01", periods=3, freq="h", name="snapshot")
    spec = {
        "dimensions": {"snapshot": {"dtype": "datetime"}},
        "parameters": {"load": {"dims": ["snapshot"]}},
        "variables": {
            "p": {"dims": ["snapshot"], "bounds": {"lower": 0, "upper": "load"}}
        },
        "objective": {"sense": "maximize", "expression": "sum(p)"},
    }
    from_sql = pd.DataFrame(
        {"snapshot": snapshot.astype(str), "value": [1.0, 2.0, 3.0]}
    )
    attached = attach(lower(spec), {"snapshot": snapshot, "load": from_sql})
    with pytest.raises(
        SpecDataError, match=r"differ in type, not spelling.*pd\.to_datetime"
    ):
        attached.read("load")


def test_a_stray_relation_value_over_an_int_target_is_shown_as_written() -> None:
    program = lower({**RELATION_SPEC, "dimensions": {"g": {}, "b": {"dtype": "int"}}})
    numbered = {"b": [1, 2], "gen_bus": pd.Series({"w": 1, "s": 99})}
    sources = sources_from(RELATION_GOOD, numbered)
    with pytest.raises(SpecDataError, match=r"not 'b' labels: 99\b") as error:
        read_all(program, sources)
    assert "int64" not in str(error.value)
