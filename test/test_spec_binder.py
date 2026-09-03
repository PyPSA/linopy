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

SPEC = {
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
        "flag": pd.Series({"a": True}),
        "cap": CAP.sel(t=[0, 1]),
    }
    bound = bind(program, sparse)
    cost = bound.parameter("cost")
    assert cost.sel(f="a").item() == 1.0
    assert cost.sel(f=["b", "c"]).isnull().all()
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


def test_missing_parameter_is_refused_when_read(
    program: Any, good: dict[str, Any]
) -> None:
    good.pop("cost")
    bound = bind(program, good)
    with pytest.raises(SpecDataError, match="no data provided for parameter 'cost'"):
        bound.parameter("cost")


REFUSALS = [
    pytest.param(
        {"f": ["a", "a", "b"]},
        r"dimension 'f' lists 'a' more than once",
        id="duplicate-member",
    ),
    pytest.param(
        {"cost": pd.Series({"a": 1.0, "zz": 2.0})},
        r"parameter 'cost'.*'f'.*'zz'",
        id="unknown-label",
    ),
    pytest.param(
        {"cap": CAP.assign_coords(t=[0, 1, 9])},
        r"parameter 'cap'.*'t'.*\b9\b",
        id="unknown-label-dense",
    ),
    pytest.param(
        {"cost": pd.Series([1.0, 9.0, 2.0], index=pd.Index(["a", "a", "b"], name="f"))},
        r"parameter 'cost' has more than one row for a coordinate: f='a' \(2 rows\)",
        id="duplicated-coordinate-row",
    ),
    pytest.param(
        {"cap": xr.DataArray([1.0, 2.0], coords={"f": ["a", "a"]})},
        r"parameter 'cap' arrived as a DataArray over \['f'\]",
        id="dense-wrong-dims",
    ),
    pytest.param(
        {
            "cap": xr.DataArray(
                np.ones((2, 3)), coords={"f": ["a", "a"], "t": [0, 1, 2]}
            )
        },
        r"parameter 'cap' has more than one row",
        id="dense-duplicate-coordinate",
    ),
    pytest.param(
        {"cap": COST},
        r"parameter 'cap'.*1 level\(s\) where 'cap' is over \['f', 't'\]",
        id="wrong-rank",
    ),
    pytest.param(
        {
            "cap": pd.Series(
                [1.0], index=pd.MultiIndex.from_tuples([("a", 0)], names=["f", "q"])
            )
        },
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
        {"cap": xr.DataArray(np.ones((3, 3)), dims=["f", "t"])},
        r"parameter 'cap' has no coordinate labels along 'f'",
        id="dense-without-labels",
    ),
    pytest.param(
        {"cost": pd.Series({"a": 1.0, "b": None})},
        r"parameter 'cost' carries 1 row.*f='b'",
        id="null-row",
    ),
    pytest.param(
        {"cost": pd.Series({"a": 1.0, "b": np.nan})},
        r"parameter 'cost' carries 1 row",
        id="nan-row",
    ),
    pytest.param(
        {"rate": float("nan")},
        r"parameter 'rate' is one value and that value is a hole",
        id="nan-scalar",
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
    sources = {**good, **override}
    for key, value in override.items():
        if value is None:
            sources.pop(key)
    with pytest.raises(SpecDataError, match=match):
        read_all(program, sources)


def test_int_labels_are_shown_as_written(program: Any, good: dict[str, Any]) -> None:
    with pytest.raises(SpecDataError, match=r"\b99\b") as error:
        read_all(program, {**good, "cap": CAP.assign_coords(t=[0, 1, 99])})
    assert "int64" not in str(error.value)


def test_dataset_is_a_source(program: Any, good: dict[str, Any]) -> None:
    ds = xr.Dataset(
        {
            "cost": xr.DataArray(COST),
            "cap": CAP,
            "flag": xr.DataArray(good["flag"]),
            "rate": 0.5,
            "lead": xr.DataArray(good["lead"]),
            "grp": xr.DataArray(good["grp"]),
        },
        coords={"f": F, "t": T, "g": ["n", "e"]},
    )
    from_dataset = bind(program, ds)
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
            "on": {"dims": ["f"], "dtype": "bool"},
            "other": {"dims": ["f"]},
        },
        "variables": {"x": {"foreach": ["f", "t"], "bounds": {"lower": 0, "upper": 1}}},
        "objective": {"sense": "maximize", "expression": "sum(x * other)"},
        "expressions": {
            "late": {
                "foreach": ["f", "t"],
                "cases": {
                    "active": {
                        "when": "on",
                        "expression": "shift(x, over=t, offset=lag, edge=0)",
                    }
                },
                "otherwise": "x * cost",
            }
        },
    }
    program = math_spec.to_program(spec)
    f = pd.Index(["a"], name="f")
    sources = {
        "f": f,
        "t": [0, 1],
        "cost": pd.Series([1.0], index=f),
        "lag": pd.Series([1], index=f),
        "on": pd.Series([True], index=f),
        "other": pd.Series([2.0], index=f),
    }
    assert set(bind(program, sources).retained().data_vars) == {"cost", "lag", "on"}


def test_aligned_array_is_not_copied(program: Any, good: dict[str, Any]) -> None:
    bound = bind(program, good)
    assert np.shares_memory(CAP.values, bound.parameter("cap").values)
    assert np.shares_memory(CAP.values, bound.parameter("cap").values)
    permuted = CAP.transpose("t", "f")
    assert np.shares_memory(
        permuted.values,
        bind(program, {**good, "cap": permuted}).parameter("cap").values,
    )
    wide = CAP.to_pandas()
    assert np.shares_memory(
        wide.values, bind(program, {**good, "cap": wide}).parameter("cap").values
    )


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

PARITY_SPEC = {
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
    pytest.param(GOOD, ACCEPTED, id="valid"),
    pytest.param(
        {"f": ["a", "b"], "cost": GOOD["cost"]},
        SpecDataError,
        id="parameter-missing-entirely",
    ),
    pytest.param(
        {**GOOD, "cost": pd.Series({"a": 1.0})}, ACCEPTED, id="coefficient-sparse"
    ),
    pytest.param(
        {
            **GOOD,
            "cost": pd.Series(
                [1.0, 9.0, 2.0], index=pd.Index(["a", "a", "b"], name="f")
            ),
        },
        SpecDataError,
        id="duplicated-coordinate-row",
    ),
    pytest.param(
        {**GOOD, "cost": pd.Series({"a": 1.0, "zz": 2.0})},
        SpecDataError,
        id="label-the-dimension-does-not-have",
    ),
    pytest.param(
        {**GOOD, "cost": pd.Series({"a": 1.0, "b": None})},
        SpecDataError,
        id="a-null-value",
    ),
    pytest.param(
        {**GOOD, "cost": pd.Series({"a": 1.0, "b": float("nan")})},
        SpecDataError,
        id="a-nan-value",
    ),
    pytest.param(
        {**GOOD, "cap": pd.Series({"a": 5.0, "b": None})},
        SpecDataError,
        id="a-hole-in-a-bound",
    ),
    pytest.param(
        {**GOOD, "cost": float("nan")}, SpecDataError, id="a-hole-as-a-scalar"
    ),
    pytest.param(
        {**GOOD, "cost": [1.0, float("nan")]}, SpecDataError, id="a-hole-in-a-sequence"
    ),
    pytest.param(
        {**GOOD, "cost": {"a": 1.0, "b": None}}, SpecDataError, id="a-hole-in-a-dict"
    ),
    pytest.param(
        {**GOOD, "cost": pd.DataFrame({"f": ["a", "b"], "value": [1.0, None]})},
        SpecDataError,
        id="a-hole-in-a-tidy-frame",
    ),
    pytest.param(
        {**GOOD, "cost": pd.Series({"a": 1, "b": 2})},
        ACCEPTED,
        id="whole-numbers-serve-a-float-declaration",
    ),
    pytest.param(
        {**GOOD, "csot": GOOD["cost"]},
        SpecDataError,
        id="a-source-key-the-model-does-not-declare",
    ),
    pytest.param(
        {
            **GOOD,
            "cost": pd.Series(
                [5.0, 5.0],
                index=pd.MultiIndex.from_tuples([("a", 0), ("b", 0)], names=["f", "k"]),
            ),
        },
        SpecDataError,
        id="a-series-deeper-than-the-declared-dims",
    ),
]


@pytest.mark.parametrize(("sources", "verdict"), PARITY_CASES)
def test_parity_with_lpspec_data_verdicts(
    sources: dict[str, Any], verdict: Any
) -> None:
    program = math_spec.to_program(PARITY_SPEC)
    if verdict is ACCEPTED:
        read_all(program, sources)
        return
    with pytest.raises(verdict):
        read_all(program, sources)


def test_a_hole_is_named_where_it_sits() -> None:
    program = math_spec.to_program(PARITY_SPEC)
    with pytest.raises(SpecDataError, match="parameter 'cost'") as error:
        read_all(program, {**GOOD, "cost": pd.Series({"a": 1.0, "b": None})})
    assert "divisor" not in str(error.value)
    assert "f='b'" in str(error.value)


def test_a_hole_in_a_scalar_parameter_is_refused() -> None:
    spec = {
        "dimensions": {"f": {"dtype": "str"}},
        "parameters": {"rate": {"dims": []}},
        "variables": {"x": {"foreach": ["f"], "bounds": {"lower": 0, "upper": 1}}},
        "objective": {"sense": "maximize", "expression": "sum(x * rate)"},
    }
    program = math_spec.to_program(spec)
    with pytest.raises(SpecDataError, match="hole"):
        read_all(program, {"f": ["a", "b"], "rate": pd.DataFrame({"value": [None]})})


@pytest.mark.parametrize(
    ("column", "verdict"),
    [
        pytest.param(
            pd.Series({"a": True, "b": False}), ACCEPTED, id="a-boolean-column"
        ),
        pytest.param(pd.Series({"a": 1, "b": 0}), SpecDataError, id="a-1-0-int-column"),
        pytest.param(
            pd.Series({"a": 1.0, "b": 0.0}), SpecDataError, id="a-1-0-float-column"
        ),
    ],
)
def test_a_flag_binds_by_its_declaration(column: pd.Series, verdict: Any) -> None:
    spec = {
        "dimensions": {"g": {"dtype": "str"}},
        "parameters": {"active": {"dims": ["g"], "dtype": "bool"}},
        "variables": {
            "x": {
                "foreach": ["g"],
                "where": "active",
                "bounds": {"lower": 0, "upper": 1},
            }
        },
        "objective": {"sense": "maximize", "expression": "sum(x)"},
    }
    program = math_spec.to_program(spec)
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
P_MAX = {"p_max": pd.Series({"w": 5.0, "s": 5.0})}
INDEX = {"g": ["w", "s"], "b": ["n", "e"]}
MAP = {"gen_bus": pd.Series({"w": "n", "s": "e"})}


@pytest.mark.parametrize(
    ("sources", "match"),
    [
        pytest.param(
            {**P_MAX, **MAP}, "dimension 'g' has no index", id="a-map-and-no-labels"
        ),
        pytest.param(
            {**P_MAX, **INDEX}, "no data provided for lookup", id="an-index-and-no-map"
        ),
        pytest.param(
            {**P_MAX, **INDEX, "gen_bus": pd.Series({"w": "n", "s": "zz"})},
            "not 'b' labels",
            id="a-stray-value",
        ),
        pytest.param(
            {
                **P_MAX,
                **INDEX,
                "gen_bus": pd.Series(
                    ["n", "e", "e"], index=pd.Index(["w", "w", "s"], name="g")
                ),
            },
            "more than once",
            id="two-values-for-one-label",
        ),
        pytest.param(
            {**P_MAX, **INDEX, "gen_bus": pd.Series({"w": None, "s": "e"})},
            "null in 'b'",
            id="mapping-a-label-to-nothing",
        ),
        pytest.param(
            {
                **P_MAX,
                **INDEX,
                "gen_bus": pd.Series(
                    [None, "n", "n"], index=pd.Index(["w", "w", "s"], name="g")
                ),
            },
            "null in 'b'",
            id="a-label-held-twice-with-a-null",
        ),
    ],
)
def test_a_lookup_defect_is_refused(sources: dict[str, Any], match: str) -> None:
    program = math_spec.to_program(LOOKUP_SPEC)
    with pytest.raises(SpecDataError, match=match):
        read_all(program, sources)


def test_a_stray_lookup_value_over_an_int_target_is_shown_as_written() -> None:
    program = math_spec.to_program(
        {**LOOKUP_SPEC, "dimensions": {"g": {}, "b": {"dtype": "int"}}}
    )
    sources = {
        **P_MAX,
        "g": ["w", "s"],
        "b": [1, 2],
        "gen_bus": pd.Series({"w": 1, "s": 99}),
    }
    with pytest.raises(SpecDataError, match=r"not 'b' labels: 99\b") as error:
        read_all(program, sources)
    assert "int64" not in str(error.value)
