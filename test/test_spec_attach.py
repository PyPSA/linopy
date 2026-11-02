"""Binding user data to a mathspec program."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

pytest.importorskip("mathspec")

from spec_helpers import table, with_  # noqa: E402

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
CAP_TABLE = table(CAP.to_series(), "cap")
COST_TABLE = table(COST, "cost")
GRP_TABLE = pd.DataFrame({"f": F, "g": ["n", "e", "n"]})

DUP_ROWS = pd.DataFrame({"f": ["a", "a", "b"], "cost": [1.0, 9.0, 2.0]})
DEEP_INDEX = pd.MultiIndex.from_tuples([("a", 0), ("b", 0)], names=["f", "k"])


def polars(frame: pd.DataFrame) -> pl.DataFrame:
    """*frame* as a polars DataFrame."""
    return pl.DataFrame({str(c): frame[c].tolist() for c in frame.columns})


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
        "cost": COST_TABLE,
        "cap": CAP,
        "flag": pd.DataFrame({"f": F, "flag": [True, False, True]}),
        "rate": 0.5,
        "lead": pd.DataFrame({"f": F, "lead": [1, 0, 1]}),
        "grp": GRP_TABLE,
    }


def filled(attached: Attached, name: str) -> xr.DataArray:
    return fill(attached.program, name, attached.read(name))


def read_all(
    program: Any, sources: Mapping[str, Any] | xr.Dataset
) -> list[xr.DataArray]:
    attached = attach(program, sources)
    return [filled(attached, name) for name in program.parameters]


CAP_SHAPES = {
    "dataarray": CAP,
    "dataarray-transposed": CAP.transpose("t", "f"),
    "table": CAP_TABLE,
    "table-columns-reordered": CAP_TABLE[["cap", "t", "f"]],
    "table-dims-reordered": table(CAP.transpose("t", "f").to_series(), "cap"),
    "table-unnamed-index": CAP_TABLE.set_axis(range(10, 19)),
    "polars-table": polars(CAP_TABLE),
}


@pytest.mark.parametrize("cap", CAP_SHAPES.values(), ids=CAP_SHAPES.keys())
def test_rank_two_shapes_attach_alike(
    program: Any, good: dict[str, Any], cap: Any
) -> None:
    got = filled(attach(program, {**good, "cap": cap}), "cap")
    xr.testing.assert_equal(got, CAP)
    assert got.dims == ("f", "t")


COST_SHAPES = {
    "table": COST_TABLE,
    "dataarray": xr.DataArray(COST),
    "whole-numbers": table(COST.astype(int), "cost"),
    "polars-table": polars(COST_TABLE),
    "polars-whole-numbers": polars(table(COST.astype(int), "cost")),
}


@pytest.mark.parametrize("cost", COST_SHAPES.values(), ids=COST_SHAPES.keys())
def test_rank_one_shapes_attach_alike(
    program: Any, good: dict[str, Any], cost: Any
) -> None:
    got = filled(attach(program, {**good, "cost": cost}), "cost")
    xr.testing.assert_equal(got, xr.DataArray(COST, name="cost"))


def test_a_table_leaving_a_coordinate_out_has_a_missing_row(
    program: Any, good: dict[str, Any]
) -> None:
    diagonal = pd.DataFrame({"f": list(F), "t": list(T), "cap": [1.0, 2.0, 3.0]})
    values, present = attach(program, {**good, "cap": diagonal}).read("cap")
    np.testing.assert_array_equal(present.values, np.eye(3, dtype=bool))
    np.testing.assert_array_equal(np.diag(values.values), [1.0, 2.0, 3.0])


DIMENSION_SHAPES = {
    "index": F,
    "list": list(F),
    "tuple": tuple(F),
    "ndarray": F.to_numpy(),
    "series": pd.Series(F),
    "polars-series": pl.Series("f", list(F)),
}


@pytest.mark.parametrize("f", DIMENSION_SHAPES.values(), ids=DIMENSION_SHAPES.keys())
def test_dimension_shapes_keep_source_order(
    program: Any, good: dict[str, Any], f: Any
) -> None:
    coords = attach(program, {**good, "f": f}).coords
    assert coords["f"].tolist() == ["b", "a", "c"]
    assert coords["f"].name == "f"
    assert list(coords) == ["f", "t", "g"]


def test_a_table_column_gives_a_dimension_through_unique(
    program: Any, good: dict[str, Any]
) -> None:
    column = pd.Series(["b", "a", "b", "c", "a"])
    for labels in (
        column.unique(),
        pl.Series(column.tolist()).unique(maintain_order=True),
    ):
        assert attach(program, {**good, "f": labels}).coords["f"].tolist() == list(F)


def arrow_strings() -> Any:
    return pd.array(list(F), dtype=pd.StringDtype("pyarrow"))


DIMENSION_SOURCES: dict[str, tuple[str, Callable[[], Any], list[Any]]] = {
    "string-python": ("f", lambda: pd.array(list(F), dtype="string"), list(F)),
    "string-pyarrow": ("f", arrow_strings, list(F)),
    "categorical": ("f", lambda: pd.Categorical(list(F)), list(F)),
    "masked-int": ("t", lambda: pd.array(list(T), dtype="Int64"), list(T)),
    "polars-int": ("t", lambda: pl.Series("t", list(T)), list(T)),
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
    grp = pd.DataFrame({"f": ["a"], "g": ["n"]})
    column = attach(program, {**good, "grp": grp}).relations["grp"].column("g")
    assert column.dims == ("f",)
    assert column.sel(f="a").item() == "n"
    assert pd.isna(column.sel(f=["b", "c"])).all()


RELATION_SHAPES = {
    "table": GRP_TABLE,
    "table-columns-reordered": pd.DataFrame(
        {"g": ["e", "n", "n"], "f": ["a", "b", "c"]}
    ),
    "table-unnamed-index": GRP_TABLE.set_axis([7, 3, 5]),
    "polars-table": polars(GRP_TABLE),
}


@pytest.mark.parametrize("grp", RELATION_SHAPES.values(), ids=RELATION_SHAPES.keys())
def test_relation_shapes_attach_alike(
    program: Any, good: dict[str, Any], grp: Any
) -> None:
    got = attach(program, {**good, "grp": grp}).relations["grp"].column("g")
    assert got.values.tolist() == ["n", "e", "n"]


@pytest.mark.parametrize("storage", ["python", "pyarrow"])
def test_extension_strings_attach_as_numpy_objects(
    program: Any, good: dict[str, Any], storage: str
) -> None:
    values = pd.array(["n", "e"], dtype=pd.StringDtype(storage))
    grp = pd.DataFrame({"f": F[:2], "g": values})
    got = attach(program, {**good, "grp": grp}).relations["grp"].column("g")
    assert got.dtype == np.dtype(object)
    assert got.values[:2].tolist() == ["n", "e"]
    assert pd.isna(got.values[2])
    assert got.sel(f=["b", "a"]).values.tolist() == ["n", "e"]


@pytest.mark.parametrize("storage", ["python", "pyarrow"])
@pytest.mark.parametrize("shape", ["table", "dataarray", "polars-table"])
def test_string_parameter_rows_attach_as_numpy_objects(
    good: dict[str, Any], storage: str, shape: str
) -> None:
    tagged = with_(
        SPEC, parameters={"tag": {"dims": ["f"], "dtype": "str", "missing": "absent"}}
    )
    series = pd.Series(["n", "e"], index=F[:2], dtype=pd.StringDtype(storage))
    shapes = {
        "table": lambda: table(series, "tag"),
        "dataarray": lambda: xr.DataArray(series),
        "polars-table": lambda: polars(table(series, "tag")),
    }
    values, present = attach(lower(tagged), {**good, "tag": shapes[shape]()}).read(
        "tag"
    )
    assert values.dtype == np.dtype(object)
    assert values.values[:2].tolist() == ["n", "e"]
    assert pd.isna(values.values[2])
    assert present.values.tolist() == [True, True, False]


def test_missing_rows_become_nan_and_false(program: Any, good: dict[str, Any]) -> None:
    sparse = {
        **good,
        "lead": pd.DataFrame({"f": ["a"], "lead": [1]}),
        "flag": pd.DataFrame({"f": ["a"], "flag": [True]}),
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
    "rate", [0.5, np.float64(0.5), xr.DataArray(0.5)], ids=["float", "numpy", "0-d"]
)
def test_scalar_parameter_stays_scalar(
    program: Any, good: dict[str, Any], rate: Any
) -> None:
    got = filled(attach(program, {**good, "rate": rate}), "rate")
    assert got.dims == ()
    assert got.item() == 0.5


EMPTY_SOURCES = {
    "float-table": pd.DataFrame(
        {"f": pd.Series(dtype=object), "cost": pd.Series(dtype=float)}
    ),
    "object-table": pd.DataFrame(
        {"f": pd.Series(dtype=object), "cost": pd.Series(dtype=object)}
    ),
    "polars-table": pl.DataFrame(
        {"f": pl.Series([], dtype=pl.Utf8), "cost": pl.Series([], dtype=pl.Float64)}
    ),
}


MODES_DATA: dict[str, Any] = {
    "f": ["a", "b"],
    "cost": pd.DataFrame({"f": ["a"], "cost": [1.0]}),
}


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


def grid_spec(**parameters: Any) -> Any:
    """A spec over 'f' alone with *parameters*, read from a grid or a Dataset."""
    return lower(
        {
            "dimensions": {"f": {"dtype": "str"}},
            "parameters": parameters,
            "variables": {"x": {"dims": ["f"], "bounds": {"lower": 0, "upper": 1}}},
            "objective": {"sense": "maximize", "expression": "sum(x)"},
        }
    )


ROW_DTYPES = {"float": float, "int": int, "bool": bool, "str": object}


@pytest.mark.parametrize("source", ["dataarray", "dataset", "table"])
@pytest.mark.parametrize(("dtype", "missing", "expected"), HOLES)
def test_a_hole_in_a_labelled_source_is_a_missing_row(
    dtype: str, missing: Any, expected: Any, source: str
) -> None:
    program = grid_spec(p={"dims": ["f"], "dtype": dtype, "missing": missing})
    p = xr.DataArray(HELD.get(dtype, STR_HOLE), coords={"f": F})
    rows = p.to_series().dropna().astype(ROW_DTYPES[dtype])
    sources: Any = {
        "dataarray": {"f": F, "p": p},
        "dataset": xr.Dataset({"p": p}, coords={"f": F}),
        "table": {"f": F, "p": rows.rename("p").reset_index()},
    }[source]
    attached = attach(program, sources)
    assert attached.read("p").present.values.tolist() == [True, False, True]
    got = filled(attached, "p")
    assert got.dtype == np.asarray(expected).dtype
    xr.testing.assert_equal(got, xr.DataArray(expected, coords={"f": F}))


UNSUPPORTED: dict[str, tuple[dict[str, Any], str]] = {
    "given": ({"given": {"expressions": {"e": {"dims": ["f"]}}}}, "given:"),
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


EXACT_COLUMNS = r"it is read from exactly the columns"
IN_INDEX = r"indexed by \['f'\].*never in its index.*reset_index\(\)"
UNIQUE = r"\.unique\(\)"
REFUSALS = [
    pytest.param(
        {"f": ["a", "a", "b"]},
        rf"dimension 'f' lists 'a' more than once.*{UNIQUE}",
        id="duplicate-member",
    ),
    pytest.param(
        {"f": pl.Series("f", ["b", "a", "b"])},
        r"dimension 'f' lists 'b' more than once.*maintain_order=True",
        id="duplicate-member-in-a-polars-column",
    ),
    pytest.param(
        {"cost": pd.DataFrame({"f": ["a", "zz"], "cost": [1.0, 2.0]})},
        r"parameter 'cost'.*'f'.*'zz'",
        id="unknown-label",
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
        {"cap": CAP_TABLE.assign(t=[9, 0, 7] * 3)},
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
        {"cap": table(COST, "cap")},
        rf"parameter 'cap' arrived as a DataFrame with the columns \['f', 'cap'\], and "
        rf"{EXACT_COLUMNS} \['f', 't', 'cap'\]",
        id="table-missing-a-dim",
    ),
    pytest.param(
        {"cap": CAP_TABLE.rename(columns={"t": "q"})},
        rf"parameter 'cap'.*{EXACT_COLUMNS} \['f', 't', 'cap'\]",
        id="table-wrong-dim-column",
    ),
    pytest.param(
        {"cost": COST_TABLE.assign(note="x")},
        rf"parameter 'cost' arrived as a DataFrame with the columns "
        rf"\['f', 'cost', 'note'\], and {EXACT_COLUMNS} \['f', 'cost'\]",
        id="table-extra-column",
    ),
    pytest.param(
        {"cost": polars(COST_TABLE.assign(note="x"))},
        rf"parameter 'cost'.*{EXACT_COLUMNS} \['f', 'cost'\]",
        id="polars-table-extra-column",
    ),
    pytest.param(
        {"cost": COST_TABLE.rename(columns={"cost": "value"})},
        rf"parameter 'cost'.*\['f', 'value'\].*{EXACT_COLUMNS} \['f', 'cost'\]",
        id="value-column",
    ),
    pytest.param(
        {"cost": COST.to_frame("cost")},
        rf"parameter 'cost' arrived as a DataFrame {IN_INDEX}",
        id="dims-in-the-index",
    ),
    pytest.param(
        {"cap": CAP.to_pandas()},
        rf"parameter 'cap' arrived as a DataFrame {IN_INDEX}",
        id="wide-table",
    ),
    pytest.param(
        {"cost": COST},
        r"parameter 'cost': cannot read Series as values over \['f'\].*DataFrame "
        r"with exactly the columns \['f', 'cost'\]",
        id="series",
    ),
    pytest.param(
        {"cost": COST.to_dict()},
        r"parameter 'cost': cannot read dict",
        id="dict",
    ),
    pytest.param(
        {"cost": 2.0},
        r"parameter 'cost': cannot read float as values over \['f'\]",
        id="scalar-for-dims",
    ),
    pytest.param(
        {"rate": table(COST, "rate")},
        r"parameter 'rate' is declared with no dims.*arrived as DataFrame",
        id="table-for-scalar",
    ),
    pytest.param(
        {"rate": [0.5]},
        r"parameter 'rate' is declared with no dims.*arrived as list",
        id="list-for-scalar",
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
        {"cost": pd.DataFrame({"f": ["a", "b"], "cost": [1.0, None]})},
        r"parameter 'cost' carries 1 row.*f='b'",
        id="null-row",
    ),
    pytest.param(
        {"cost": pd.DataFrame({"f": ["a", "b"], "cost": [1.0, np.nan]})},
        r"parameter 'cost' carries 1 row",
        id="nan-row",
    ),
    pytest.param(
        {"cost": pl.DataFrame({"f": ["a", "b"], "cost": [1.0, None]})},
        r"parameter 'cost' carries 1 row.*f='b'",
        id="null-in-a-polars-table",
    ),
    pytest.param(
        {"cost": [1.0, np.nan, 2.0]},
        r"parameter 'cost': cannot read list",
        id="sequence",
    ),
    pytest.param(
        {"rate": float("nan")},
        r"parameter 'rate' is one value and that value is a hole",
        id="nan-scalar",
    ),
    pytest.param(
        {"lead": pd.DataFrame({"f": F, "lead": [1.5, 0.0, 1.0]})},
        r"'lead' is declared 'int'.*'float'",
        id="float-for-int",
    ),
    pytest.param(
        {"flag": pd.DataFrame({"f": F, "flag": [1, 0, 1]})},
        r"'flag' is declared 'bool'.*'int'",
        id="int-for-bool",
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
        {"cost": pd.DataFrame({"f": F, "cost": ["x", "y", "z"]})},
        r"'cost' is declared 'float'.*'str'",
        id="str-for-float",
    ),
    pytest.param(
        {"f": None},
        rf"dimension 'f' has no index.*{UNIQUE}",
        id="missing-dimension",
    ),
    pytest.param(
        {"f": set(F)},
        r"index for dimension 'f' arrived as a set, which has no order.*sorted",
        id="dimension-set",
    ),
    pytest.param(
        {"f": frozenset(F)},
        r"index for dimension 'f' arrived as a frozenset, which has no order",
        id="dimension-frozenset",
    ),
    pytest.param(
        {"f": xr.DataArray(list(F), dims=["f"])},
        r"index for dimension 'f': cannot read labels out of DataArray",
        id="dimension-dataarray",
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
        {"grp": None}, r"no data provided for relation 'grp'", id="missing-relation"
    ),
    pytest.param(
        {"grp": pd.Series(["n", "e", "n"], index=F)},
        r"relation 'grp': cannot read rows out of Series; pass a pandas or polars "
        r"DataFrame with exactly the columns \['f', 'g'\]",
        id="relation-series",
    ),
    pytest.param(
        {"grp": {"b": "n", "a": "e", "c": "n"}},
        r"relation 'grp': cannot read rows out of dict",
        id="relation-dict",
    ),
    pytest.param(
        {"grp": xr.DataArray(["n", "e", "n"], coords={"f": F})},
        r"relation 'grp': cannot read rows out of DataArray",
        id="relation-dataarray",
    ),
    pytest.param(
        {"grp": ["n"]},
        r"relation 'grp': cannot read rows out of list",
        id="relation-shape",
    ),
    pytest.param(
        {"grp": pd.DataFrame({"f": ["zz"], "g": ["n"]})},
        r"relation 'grp' maps 'zz', which are not labels of 'f'",
        id="relation-stray-key",
    ),
    pytest.param(
        {"grp": pd.DataFrame({"f": [1], "g": ["n"]})},
        r"relation 'grp' has 'int' label\(s\) in dimension 'f'.*differ in type",
        id="relation-int-key-on-a-str-dimension",
    ),
    pytest.param(
        {"grp": pd.DataFrame({"f": ["a"], "g": ["zz"]})},
        r"relation 'grp' has value\(s\) that are not 'g' labels: 'zz'",
        id="relation-stray-value",
    ),
    pytest.param(
        {"grp": pd.DataFrame({"f": ["a", "a"], "g": ["n", "e"]})},
        r"relation 'grp' maps 1 'f' label\(s\) more than once: 'a'",
        id="relation-two-values",
    ),
    pytest.param(
        {"grp": pd.DataFrame({"f": ["a", "b"], "g": [None, "n"]})},
        r"relation 'grp' carries 1 row\(s\) with a null in 'g': f='a'",
        id="relation-null",
    ),
    pytest.param(
        {"grp": pd.DataFrame({"f": ["a"], "bus": ["n"]})},
        rf"relation 'grp' arrived as a DataFrame with the columns \['f', 'bus'\], and "
        rf"{EXACT_COLUMNS} \['f', 'g'\]",
        id="relation-table-without-a-column",
    ),
    pytest.param(
        {"grp": GRP_TABLE.assign(note="x")},
        rf"relation 'grp'.*{EXACT_COLUMNS} \['f', 'g'\]",
        id="relation-table-extra-column",
    ),
    pytest.param(
        {"grp": polars(GRP_TABLE.assign(note="x"))},
        rf"relation 'grp'.*{EXACT_COLUMNS} \['f', 'g'\]",
        id="relation-polars-table-extra-column",
    ),
    pytest.param(
        {"grp": pd.DataFrame({"g": ["n", "e", "n"]}, index=F)},
        rf"relation 'grp' arrived as a DataFrame {IN_INDEX}",
        id="relation-key-in-the-index",
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


def test_dataset_is_a_source() -> None:
    program = grid_spec(cost={"dims": ["f"], "missing": "absent"}, rate={"dims": []})
    cost = xr.DataArray([1.0, np.nan, 3.0], coords={"f": F})
    from_dataset = attach(program, xr.Dataset({"cost": cost, "rate": 0.5}))
    from_mapping = attach(program, {"f": F, "cost": cost, "rate": 0.5})
    assert from_dataset.coords["f"].equals(from_mapping.coords["f"])
    for name in program.parameters:
        xr.testing.assert_equal(filled(from_dataset, name), filled(from_mapping, name))


def test_a_relation_in_a_dataset_reads_as_its_value_column_over_the_key(
    good: dict[str, Any],
) -> None:
    grp = xr.DataArray(np.array(["n", np.nan, "n"], dtype=object), coords={"f": F})
    dataset = xr.Dataset({"grp": grp}, coords={"t": T, "g": ["n", "e"]})
    from_dataset = attach(lower(SPEC), dataset).relations["grp"]
    rows = pd.DataFrame({"f": ["b", "c"], "g": ["n", "n"]})
    from_table = attach(lower(SPEC), {**good, "grp": rows}).relations["grp"]
    xr.testing.assert_identical(from_dataset.column("g"), from_table.column("g"))


def test_a_relation_over_another_dimension_in_a_dataset_is_refused() -> None:
    grp = xr.DataArray(["n"], coords={"t": [0]})
    dataset = xr.Dataset({"grp": grp}, coords={"f": F, "g": ["n", "e"]})
    with pytest.raises(SpecDataError, match=r"relation 'grp': cannot read rows"):
        attach(lower(SPEC), dataset)


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
    sources = sources_from(good, {"csot": COST_TABLE, "cost": None})
    with pytest.warns(UserWarning, match=r"csot -> cost"):
        attached = attach(program, sources)
    with pytest.raises(SpecDataError, match="no data provided for parameter 'cost'"):
        attached.read("cost")


@pytest.mark.parametrize("key", ["notes", "csot"], ids=["unlike", "beside-its-name"])
def test_a_source_key_that_leaves_no_name_without_data_is_silent(
    program: Any, good: dict[str, Any], recwarn: Any, key: str
) -> None:
    attach(program, {**good, key: COST_TABLE})
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
    ("values", "verdict"),
    [
        pytest.param([True, False], ACCEPTED, id="a-bool-column"),
        pytest.param([1, 0], SpecDataError, id="a-1-0-int-column"),
        pytest.param([1.0, 0.0], SpecDataError, id="a-1-0-float-column"),
    ],
)
def test_a_flag_attaches_by_its_declaration(values: list[Any], verdict: Any) -> None:
    program = lower(FLAG_SPEC)
    sources = {
        "g": ["a", "b"],
        "active": pd.DataFrame({"g": ["a", "b"], "active": values}),
    }
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
G_TWICE = ["w", "w", "s"]
RELATION_GOOD = {
    "p_max": pd.DataFrame({"g": ["w", "s"], "p_max": [5.0, 5.0]}),
    "g": ["w", "s"],
    "b": ["n", "e"],
    "gen_bus": pd.DataFrame({"g": ["w", "s"], "b": ["n", "e"]}),
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
            {"gen_bus": pd.DataFrame({"g": ["w", "s"], "b": ["n", "zz"]})},
            "not 'b' labels",
            id="a-stray-value",
        ),
        pytest.param(
            {"gen_bus": pd.DataFrame({"g": G_TWICE, "b": ["n", "e", "e"]})},
            "more than once",
            id="two-values-for-one-label",
        ),
        pytest.param(
            {"gen_bus": pd.DataFrame({"g": ["w", "s"], "b": [None, "e"]})},
            "null in 'b'",
            id="mapping-a-label-to-nothing",
        ),
        pytest.param(
            {"gen_bus": pd.DataFrame({"g": G_TWICE, "b": [None, "n", "n"]})},
            "null in 'b'",
            id="a-label-held-twice-with-a-null",
        ),
    ],
)
def test_a_relation_defect_is_refused(override: dict[str, Any], match: str) -> None:
    program = lower(RELATION_SPEC)
    with pytest.raises(SpecDataError, match=match):
        read_all(program, sources_from(RELATION_GOOD, override))


def test_a_relation_leaving_a_key_out_is_partial(good: dict[str, Any]) -> None:
    grp = pd.DataFrame({"f": ["b", "c"], "g": ["n", "n"]})
    column = attach(lower(SPEC), {**good, "grp": grp}).relations["grp"].column("g")
    assert column.sel(f=["b", "c"]).values.tolist() == ["n", "n"]
    assert pd.isna(column.sel(f="a").item())
    refused = with_(SPEC, relations={"grp": {"key": "f", "values": "g"}})
    with pytest.raises(
        SpecDataError, match=r"relation 'grp' has no row for 1 'f' label"
    ):
        attach(lower(refused), {**good, "grp": grp})


SNAPSHOTS = pd.date_range("2025-01-01", periods=3, freq="h", name="snapshot")
LOAD_SPEC: dict[str, Any] = {
    "dimensions": {"snapshot": {"dtype": "datetime"}},
    "parameters": {"load": {"dims": ["snapshot"]}},
    "variables": {"p": {"dims": ["snapshot"], "bounds": {"lower": 0, "upper": "load"}}},
    "objective": {"sense": "maximize", "expression": "sum(p)"},
}


def test_datetime_labels_given_as_text_are_a_type_mismatch() -> None:
    from_sql = pd.DataFrame(
        {"snapshot": SNAPSHOTS.astype(str), "load": [1.0, 2.0, 3.0]}
    )
    attached = attach(lower(LOAD_SPEC), {"snapshot": SNAPSHOTS, "load": from_sql})
    with pytest.raises(
        SpecDataError, match=r"differ in type, not spelling.*pd\.to_datetime"
    ):
        attached.read("load")


def test_polars_datetimes_attach_like_pandas_ones() -> None:
    load = pd.DataFrame({"snapshot": SNAPSHOTS, "load": [1.0, 2.0, 3.0]})
    program = lower(LOAD_SPEC)
    expected = filled(attach(program, {"snapshot": SNAPSHOTS, "load": load}), "load")
    from_polars = {
        "snapshot": pl.Series("snapshot", SNAPSHOTS.to_list()),
        "load": polars(load),
    }
    xr.testing.assert_equal(filled(attach(program, from_polars), "load"), expected)


def test_a_stray_relation_value_over_an_int_target_is_shown_as_written() -> None:
    program = lower({**RELATION_SPEC, "dimensions": {"g": {}, "b": {"dtype": "int"}}})
    numbered = {"b": [1, 2], "gen_bus": pd.DataFrame({"g": ["w", "s"], "b": [1, 99]})}
    sources = sources_from(RELATION_GOOD, numbered)
    with pytest.raises(SpecDataError, match=r"not 'b' labels: 99\b") as error:
        read_all(program, sources)
    assert "int64" not in str(error.value)


def test_a_table_label_equal_to_a_coordinate_in_value_only_is_refused() -> None:
    program = lower(
        {
            "dimensions": {"y": {"dtype": "int"}},
            "parameters": {"p": {"dims": ["y"]}},
            "variables": {"x": {"dims": ["y"], "bounds": {"lower": 0, "upper": 1}}},
            "objective": {"sense": "maximize", "expression": "sum(x)"},
        }
    )
    rows = pd.DataFrame({"y": pd.Series([True], dtype=object), "p": [1.0]})
    attached = attach(program, {"y": pd.Index([1, 2], name="y"), "p": rows})
    with pytest.raises(SpecDataError, match="in value but not in type: True"):
        attached.check("p")
