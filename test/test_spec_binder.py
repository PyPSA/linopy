"""Binding user data to a math-spec program."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

math_spec = pytest.importorskip("math_spec")

from linopy.spec import SpecDataError, bind  # noqa: E402

SPEC: dict[str, Any] = {
    "dimensions": {"f": {"dtype": "str"}, "t": {"dtype": "int"}, "g": {"dtype": "str"}},
    "lookups": {"grp": {"over": "f", "into": "g"}},
    "parameters": {
        "cost": {"dims": ["f"]},
        "cap": {"dims": ["f", "t"]},
        "flag": {"dims": ["f"], "dtype": "bool"},
        "rate": {"dims": []},
        "lead": {"dims": ["f"], "dtype": "int"},
    },
    "variables": {
        "x": {
            "foreach": ["f", "t"],
            "where": "flag",
            "bounds": {"lower": 0, "upper": "cap"},
        }
    },
    "constraints": {
        "k": {"foreach": ["g", "t"], "expression": "sum(x, by=grp) <= 10"},
        "s": {
            "foreach": ["f", "t"],
            "expression": "shift(x, over=t, offset=lead, edge=0) >= 0",
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
    return math_spec.to_program(SPEC)


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


def read_all(program: Any, sources: Mapping[str, Any]) -> list[xr.DataArray]:
    bound = bind(program, sources)
    return [bound.parameter(name) for name in program.parameters]


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
    "wide-frame": CAP.to_pandas(),
    "wide-frame-transposed": CAP.to_pandas().T,
    "wide-frame-unnamed": CAP.to_pandas().rename_axis(index=None, columns=None),
    "dict": CAP.to_series().to_dict(),
}


@pytest.mark.parametrize("cap", CAP_SHAPES.values(), ids=CAP_SHAPES.keys())
def test_rank_two_shapes_bind_alike(
    program: Any, good: dict[str, Any], cap: Any
) -> None:
    got = bind(program, {**good, "cap": cap}).parameter("cap")
    xr.testing.assert_equal(got, CAP)
    assert got.dims == ("f", "t")


COST_SHAPES = {
    "series": COST,
    "series-unnamed": COST.rename_axis(None),
    "dataarray": xr.DataArray(COST),
    "dict": COST.to_dict(),
    "tidy-frame": COST.reset_index(name="value"),
    "indexed-frame": COST.to_frame("value"),
}


@pytest.mark.parametrize("cost", COST_SHAPES.values(), ids=COST_SHAPES.keys())
def test_rank_one_shapes_bind_alike(
    program: Any, good: dict[str, Any], cost: Any
) -> None:
    got = bind(program, {**good, "cost": cost}).parameter("cost")
    xr.testing.assert_equal(got, xr.DataArray(COST, name="cost"))


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
    coords = bind(program, {**good, "f": f}).coords
    assert coords["f"].tolist() == ["b", "a", "c"]
    assert coords["f"].name == "f"
    assert list(coords) == ["f", "t", "g"]


def test_lookup_is_padded_onto_the_dimension(
    program: Any, good: dict[str, Any]
) -> None:
    bound = bind(program, {**good, "grp": {"a": "n"}})
    grp = bound.lookups["f"]["grp"]
    assert grp.dims == ("f",)
    assert grp.sel(f="a").item() == "n"
    assert pd.isna(grp.sel(f=["b", "c"])).all()


LOOKUP_SHAPES = {
    "series": pd.Series(["n", "e", "n"], index=F),
    "series-unnamed": pd.Series(["n", "e", "n"], index=F.rename(None)),
    "dict": {"b": "n", "a": "e", "c": "n"},
    "dataarray": xr.DataArray(["n", "e", "n"], coords={"f": F}),
}


@pytest.mark.parametrize("grp", LOOKUP_SHAPES.values(), ids=LOOKUP_SHAPES.keys())
def test_lookup_shapes_bind_alike(program: Any, good: dict[str, Any], grp: Any) -> None:
    got = bind(program, {**good, "grp": grp}).lookups["f"]["grp"]
    assert got.values.tolist() == ["n", "e", "n"]


def test_missing_rows_become_nan_and_false(program: Any, good: dict[str, Any]) -> None:
    sparse = {
        **good,
        "cost": pd.Series({"a": 1.0}),
        "lead": pd.Series({"a": 1}),
        "flag": pd.Series({"a": True}),
        "cap": CAP.sel(t=[0, 1]),
    }
    bound = bind(program, sparse)
    cost = bound.parameter("cost")
    assert cost.sel(f="a").item() == 1.0
    assert cost.sel(f=["b", "c"]).isnull().all()
    lead = bound.parameter("lead")
    assert lead.dtype == np.float64
    assert lead.sel(f="a").item() == 1.0
    assert lead.sel(f=["b", "c"]).isnull().all()
    flag = bound.parameter("flag")
    assert flag.dtype == bool
    assert flag.values.tolist() == [False, True, False]
    cap = bound.parameter("cap")
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
    got = bind(program, {**good, name: value}).parameter(name)
    assert got.dims == tuple(SPEC["parameters"][name]["dims"])
    assert got.dtype == expected_dtype
    assert (got == value).all()


def test_scalar_parameter_stays_scalar(program: Any, good: dict[str, Any]) -> None:
    got = bind(program, good).parameter("rate")
    assert got.dims == ()
    assert got.item() == 0.5


EMPTY_SOURCES = {
    "dict": {},
    "object-series": pd.Series(dtype=object),
    "float-series": pd.Series(dtype=float),
}


@pytest.mark.parametrize("cost", EMPTY_SOURCES.values(), ids=EMPTY_SOURCES.keys())
def test_empty_source_binds_as_all_nan(
    program: Any, good: dict[str, Any], cost: Any
) -> None:
    got = bind(program, {**good, "cost": cost}).parameter("cost")
    assert got.dtype == np.float64
    assert got.isnull().all()
    assert got.indexes["f"].equals(F)


def test_missing_parameter_is_refused_when_read(
    program: Any, good: dict[str, Any]
) -> None:
    good.pop("cost")
    bound = bind(program, good)
    with pytest.raises(SpecDataError, match="no data provided for parameter 'cost'"):
        bound.parameter("cost")


def test_undeclared_parameter_is_refused_with_a_hint(
    program: Any, good: dict[str, Any]
) -> None:
    with pytest.raises(SpecDataError, match="unknown parameter 'csot'.*'cost'"):
        bind(program, good).parameter("csot")


def test_retain_is_validated_before_binding(program: Any, good: dict[str, Any]) -> None:
    with pytest.raises(SpecDataError, match=r"'report', 'all', 'none'") as error:
        bind(program, good, retain="reports")  # type: ignore[arg-type]
    assert "Did you mean 'report'?" in str(error.value)


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
        {"cap": xr.DataArray(np.ones((3, 3)), dims=["f", "t"])},
        r"parameter 'cap' has no coordinate labels along 'f'",
        id="dense-without-labels",
    ),
    pytest.param(
        {"cost": NULL_ROW}, r"parameter 'cost' carries 1 row.*f='b'", id="null-row"
    ),
    pytest.param({"cost": NAN_ROW}, r"parameter 'cost' carries 1 row", id="nan-row"),
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
    pytest.param(
        {"csot": COST}, r"source key 'csot'.*Did you mean 'cost'", id="unknown-key"
    ),
    pytest.param({"f": None}, r"dimension 'f' has no index", id="missing-dimension"),
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
        {"grp": xr.DataArray(["n"], coords={"t": [0]})},
        r"lookup 'grp' arrived as a DataArray over \['t'\]",
        id="lookup-wrong-dataarray-dim",
    ),
    pytest.param(
        {"grp": None}, r"no data provided for lookup 'grp'", id="missing-lookup"
    ),
    pytest.param(
        {"grp": {"zz": "n"}},
        r"lookup 'grp' maps 'zz', which are not labels of 'f'",
        id="lookup-stray-key",
    ),
    pytest.param(
        {"grp": {"a": "zz"}},
        r"lookup 'grp' has value\(s\) that are not 'g' labels: 'zz'",
        id="lookup-stray-value",
    ),
    pytest.param(
        {"grp": pd.Series(["n", "e"], index=pd.Index(["a", "a"], name="f"))},
        r"lookup 'grp' maps 1 'f' label\(s\) more than once: 'a'",
        id="lookup-two-values",
    ),
    pytest.param(
        {"grp": {"a": None, "b": "n"}},
        r"lookup 'grp' carries 1 row\(s\) with a null in 'g': f='a'",
        id="lookup-null",
    ),
    pytest.param(
        {"grp": pd.DataFrame({"f": ["a"], "g": ["n"]})},
        r"lookup 'grp': cannot adapt DataFrame",
        id="lookup-shape",
    ),
    pytest.param(
        {"grp": pd.Series(["n"], index=pd.Index(["a"], name="t"))},
        r"lookup 'grp' is a Series indexed by 't'",
        id="lookup-wrong-index",
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
    dims = {"f": F, "t": T, "g": ["n", "e"]}
    values = {k: xr.DataArray(v) for k, v in good.items() if k not in dims}
    from_dataset = bind(program, xr.Dataset(values, coords=dims))
    from_mapping = bind(program, good)
    assert from_dataset.coords["f"].equals(from_mapping.coords["f"])
    for name in program.parameters:
        xr.testing.assert_equal(
            from_dataset.parameter(name), from_mapping.parameter(name)
        )
    xr.testing.assert_equal(
        from_dataset.lookups["f"]["grp"], from_mapping.lookups["f"]["grp"]
    )


class Counting(Mapping[str, Any]):
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.pulled: list[str] = []

    def __getitem__(self, key: str) -> Any:
        self.pulled.append(key)
        return self.data[key]

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("sources must not be iterated")

    def __len__(self) -> int:
        return len(self.data)

    def keys(self) -> Any:
        return self.data.keys()


def test_sources_are_pulled_by_key_on_demand(
    program: Any, good: dict[str, Any]
) -> None:
    sources = Counting(good)
    bound = bind(program, sources)
    assert set(sources.pulled) == {"f", "t", "g", "grp"}
    bound.parameter("cap")
    bound.parameter("cap")
    assert sources.pulled.count("cap") == 2


@pytest.mark.parametrize(
    ("retain", "expected"),
    [
        ("report", {"cost", "rate", "grp"}),
        ("all", {"cost", "cap", "flag", "rate", "lead", "grp"}),
        ("none", {"grp"}),
    ],
)
def test_retained_follows_the_named_expressions(
    program: Any, good: dict[str, Any], retain: Any, expected: set[str]
) -> None:
    retained = bind(program, good, retain=retain).retained()
    assert set(retained.data_vars) == expected
    assert retained.coords["f"].values.tolist() == ["b", "a", "c"]


def test_report_closure_reads_names_and_masks() -> None:
    spec = {
        "dimensions": {"f": {"dtype": "str"}, "t": {"dtype": "int"}},
        "parameters": {
            "cost": {"dims": ["f"]},
            "lag": {"dims": ["f"], "dtype": "int"},
            "span": {"dims": ["f"], "dtype": "int"},
            "on": {"dims": ["f"], "dtype": "bool"},
            "other": {"dims": ["f"]},
        },
        "variables": {"x": {"foreach": ["f", "t"], "bounds": {"lower": 0, "upper": 1}}},
        "objective": {"sense": "maximize", "expression": "sum(x * other)"},
        "expressions": {
            "recent": "sum_back(x, over=t, within=span)",
            "late": {
                "foreach": ["f", "t"],
                "cases": {
                    "active": {
                        "when": "on",
                        "expression": "shift(x, over=t, offset=lag, edge=0)",
                    }
                },
                "otherwise": "x * cost",
            },
        },
    }
    program = math_spec.to_program(spec)
    f = pd.Index(["a"], name="f")
    sources = {
        "f": f,
        "t": [0, 1],
        "cost": pd.Series([1.0], index=f),
        "lag": pd.Series([1], index=f),
        "span": pd.Series([2], index=f),
        "on": pd.Series([True], index=f),
        "other": pd.Series([2.0], index=f),
    }
    retained = bind(program, sources).retained()
    assert set(retained.data_vars) == {"cost", "lag", "span", "on"}


def test_unreached_dimension_needs_no_source() -> None:
    dimensions = {**PARITY_SPEC["dimensions"], "z": {"dtype": "int"}}
    program = math_spec.to_program({**PARITY_SPEC, "dimensions": dimensions})
    assert list(bind(program, GOOD).coords) == ["f"]


@pytest.mark.parametrize("shape", ["dataarray", "dataarray-transposed", "wide-frame"])
def test_aligned_array_is_not_copied(
    program: Any, good: dict[str, Any], shape: str
) -> None:
    source = CAP_SHAPES[shape]
    bound = bind(program, {**good, "cap": source})
    assert np.shares_memory(np.asarray(source), bound.parameter("cap").values)
    assert np.shares_memory(np.asarray(source), bound.parameter("cap").values)


def test_master_coordinate_dtype_wins_without_a_copy(
    program: Any, good: dict[str, Any]
) -> None:
    source = CAP.assign_coords(t=T.astype("int32"))
    got = bind(program, {**good, "cap": source}).parameter("cap")
    assert got.indexes["t"].dtype == np.int64
    assert np.shares_memory(np.asarray(source), got.values)


def test_derived_parameter_is_not_bound_from_sources() -> None:
    spec = {
        "dimensions": {"bp": {"dtype": "int"}},
        "parameters": {"bp_x": {"dims": ["bp"]}, "bp_y": {"dims": ["bp"]}},
        "variables": {
            "x": {"foreach": [], "bounds": {"lower": 0, "upper": 10}},
            "y": {"foreach": []},
        },
        "piecewise": {
            "curve": {
                "over": "bp",
                "method": "lp",
                "points": "bp_x",
                "links": [["x", "bp_x"], ["y", "bp_y", ">="]],
            }
        },
        "objective": {"sense": "minimize", "expression": "y"},
    }
    program = math_spec.to_program(spec)
    derived = [n for n, p in program.parameters.items() if p.derivation is not None]
    assert derived
    bp = pd.Index([0, 1, 2], name="bp")
    sources = {
        "bp": bp,
        "bp_x": pd.Series([0.0, 5.0, 10.0], index=bp),
        "bp_y": pd.Series([0.0, 2.0, 8.0], index=bp),
    }
    bound = bind(program, sources, retain="all")
    assert set(bound.retained().data_vars) == {"bp_x", "bp_y"}
    with pytest.raises(SpecDataError, match="emitted by piecewise block 'curve'"):
        bound.parameter(derived[0])
    with pytest.raises(SpecDataError, match=derived[0]):
        bind(program, {**sources, derived[0]: 1.0})


# ---------------------------------------------------------------------------
# lpspec data-parity cases, eager representation
# ---------------------------------------------------------------------------

PARITY_SPEC: dict[str, Any] = {
    "dimensions": {"f": {"dtype": "str"}},
    "parameters": {"cost": {"dims": ["f"]}, "cap": {"dims": ["f"]}},
    "variables": {"x": {"foreach": ["f"], "bounds": {"lower": 0, "upper": "cap"}}},
    "constraints": {"k": {"foreach": ["f"], "expression": "x <= cap"}},
    "objective": {"sense": "maximize", "expression": "sum(x * cost)"},
}

GOOD = {
    "f": ["a", "b"],
    "cost": pd.Series({"a": 1.0, "b": 2.0}),
    "cap": pd.Series({"a": 5.0, "b": 5.0}),
}
ACCEPTED = "accepted"

PARITY_CASES = [
    pytest.param({}, ACCEPTED, id="valid"),
    pytest.param({"cap": None}, SpecDataError, id="parameter-missing-entirely"),
    pytest.param({"cost": pd.Series({"a": 1.0})}, ACCEPTED, id="coefficient-sparse"),
    pytest.param({"cost": DUP_ROWS}, SpecDataError, id="duplicated-coordinate-row"),
    pytest.param({"cost": STRAY_ROW}, SpecDataError, id="label-not-in-the-dimension"),
    pytest.param({"cost": NULL_ROW}, SpecDataError, id="a-null-value"),
    pytest.param({"cost": NAN_ROW}, SpecDataError, id="a-nan-value"),
    pytest.param({"cap": NULL_ROW}, SpecDataError, id="a-hole-in-a-bound"),
    pytest.param({"cost": float("nan")}, SpecDataError, id="a-hole-as-a-scalar"),
    pytest.param({"cost": [1.0, np.nan]}, SpecDataError, id="a-hole-in-a-sequence"),
    pytest.param({"cost": {"a": 1.0, "b": None}}, SpecDataError, id="a-hole-in-a-dict"),
    pytest.param({"cost": NULL_FRAME}, SpecDataError, id="a-hole-in-a-tidy-frame"),
    pytest.param({"cost": pd.Series({"a": 1, "b": 2})}, ACCEPTED, id="whole-numbers"),
    pytest.param({"csot": COST}, SpecDataError, id="an-undeclared-source-key"),
    pytest.param({"cost": DEEP_ROWS}, SpecDataError, id="a-series-too-deep"),
]


@pytest.mark.parametrize(("override", "verdict"), PARITY_CASES)
def test_parity_with_lpspec_data_verdicts(
    override: dict[str, Any], verdict: Any
) -> None:
    program = math_spec.to_program(PARITY_SPEC)
    sources = sources_from(GOOD, override)
    if verdict is ACCEPTED:
        read_all(program, sources)
        return
    with pytest.raises(verdict):
        read_all(program, sources)


def test_a_hole_is_named_where_it_sits() -> None:
    program = math_spec.to_program(PARITY_SPEC)
    with pytest.raises(SpecDataError, match="parameter 'cost'") as error:
        read_all(program, {**GOOD, "cost": NULL_ROW})
    assert "divisor" not in str(error.value)
    assert "f='b'" in str(error.value)


FLAG_SPEC = {
    "dimensions": {"g": {"dtype": "str"}},
    "parameters": {"active": {"dims": ["g"], "dtype": "bool"}},
    "variables": {
        "x": {"foreach": ["g"], "where": "active", "bounds": {"lower": 0, "upper": 1}}
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
def test_a_flag_binds_by_its_declaration(column: pd.Series, verdict: Any) -> None:
    program = math_spec.to_program(FLAG_SPEC)
    sources = {"g": ["a", "b"], "active": column}
    if verdict is ACCEPTED:
        assert bind(program, sources).parameter("active").dtype == bool
        return
    with pytest.raises(SpecDataError, match="declared 'bool'"):
        read_all(program, sources)


LOOKUP_SPEC = {
    "dimensions": {"g": {}, "b": {"dtype": "str"}},
    "lookups": {"gen_bus": {"over": "g", "into": "b"}},
    "parameters": {"p_max": {"dims": ["g"]}},
    "variables": {"x": {"foreach": ["g"], "bounds": {"lower": 0, "upper": "p_max"}}},
    "constraints": {"k": {"foreach": ["b"], "expression": "sum(x, by=gen_bus) <= 10"}},
    "objective": {"sense": "maximize", "expression": "sum(x)"},
}
G_TWICE = pd.Index(["w", "w", "s"], name="g")
LOOKUP_GOOD = {
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
            {"gen_bus": None}, "no data provided for lookup", id="an-index-no-map"
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
def test_a_lookup_defect_is_refused(override: dict[str, Any], match: str) -> None:
    program = math_spec.to_program(LOOKUP_SPEC)
    with pytest.raises(SpecDataError, match=match):
        read_all(program, sources_from(LOOKUP_GOOD, override))


TAG_SPEC = {
    **LOOKUP_SPEC,
    "lookups": {"tag": {"over": "g", "dtype": "int"}},
    "constraints": {"k": {"foreach": ["g"], "expression": "x <= 10"}},
}
TAG_GOOD = sources_from(LOOKUP_GOOD, {"gen_bus": None, "b": None})


def test_a_label_space_lookup_is_padded_onto_the_dimension() -> None:
    program = math_spec.to_program(TAG_SPEC)
    bound = bind(program, {**TAG_GOOD, "tag": {"s": 7}})
    tag = bound.lookups["g"]["tag"]
    assert tag.dims == ("g",)
    assert tag.indexes["g"].tolist() == ["w", "s"]
    assert np.isnan(tag.sel(g="w").item())
    assert tag.sel(g="s").item() == 7


@pytest.mark.parametrize(
    ("tag", "match"),
    [
        pytest.param({"w": None, "s": 7}, "null in 'tag': g='w'", id="a-null"),
        pytest.param(
            {"w": "x", "s": "y"}, "lookup 'tag' is declared 'int'.*'str'", id="a-str"
        ),
    ],
)
def test_a_label_space_lookup_defect_is_refused(
    tag: dict[str, Any], match: str
) -> None:
    program = math_spec.to_program(TAG_SPEC)
    with pytest.raises(SpecDataError, match=match):
        bind(program, {**TAG_GOOD, "tag": tag})


def test_a_stray_lookup_value_over_an_int_target_is_shown_as_written() -> None:
    program = math_spec.to_program(
        {**LOOKUP_SPEC, "dimensions": {"g": {}, "b": {"dtype": "int"}}}
    )
    numbered = {"b": [1, 2], "gen_bus": pd.Series({"w": 1, "s": 99})}
    sources = sources_from(LOOKUP_GOOD, numbered)
    with pytest.raises(SpecDataError, match=r"not 'b' labels: 99\b") as error:
        read_all(program, sources)
    assert "int64" not in str(error.value)
