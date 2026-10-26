"""Long-form inputs: ``linopy.from_long`` constants and ``linopy.Relation`` group-by."""

from __future__ import annotations

import datetime as dt
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
from test_store_conformance import RELATION, RELATION_FRAME, A, B, assert_same

from linopy import (
    EvolvingAPIWarning,
    LinearExpression,
    Model,
    QuadraticExpression,
    Relation,
    Variable,
    from_long,
)
from linopy.constants import _emitted_evolving_warnings
from linopy.long import Column
from linopy.testing import assert_equivalent


def to_polars(frame: pd.DataFrame) -> pl.DataFrame:
    return pl.from_dict(frame.astype(object).where(frame.notna(), None).to_dict("list"))


FRAME_LIBS: dict[str, Callable[[pd.DataFrame], Any]] = {
    "pandas": lambda f: f,
    "polars": to_polars,
}
FRAME_AB = pd.DataFrame(
    {"a": [0, 2, 1, 0], "b": ["p", "p", "q", "q"], "v": [1.0, 2, 3, 4]}
)
FRAME_A = pd.DataFrame({"a": [2, 0, 1], "v": [1.0, 2, 3]})

CONSTANTS: dict[str, tuple[pd.DataFrame, float, xr.DataArray]] = {
    "all-keys-default": (
        FRAME_AB,
        5.0,
        xr.DataArray([[1.0, 4], [5, 3], [2, 5]], coords=[A, B]),
    ),
    "missing-key-column": (FRAME_A, np.nan, xr.DataArray([2.0, 3, 1], coords=[A])),
    "column-order": (
        pd.DataFrame({"b": ["q", "p"], "a": [1, 1], "v": [3.0, 4]}),
        1.0,
        xr.DataArray([[1.0, 4, 1], [1, 3, 1]], coords=[B, A]),
    ),
}


@pytest.mark.parametrize("lib", list(FRAME_LIBS))
@pytest.mark.parametrize("case", list(CONSTANTS))
def test_from_long(case: str, lib: str) -> None:
    frame, default, want = CONSTANTS[case]
    res = from_long(FRAME_LIBS[lib](frame), coords=[A, B], default=default)
    xr.testing.assert_identical(res, want.rename("v"))


@pytest.mark.parametrize("lib", list(FRAME_LIBS))
def test_from_long_variable_coords(lib: str) -> None:
    frame = FRAME_AB.rename(columns={"v": "cost"})
    var = x(Model())
    res = from_long(FRAME_LIBS[lib](frame), coords=var.coords)
    want = from_long(frame, coords=[A, B])
    assert res.name == "cost"
    xr.testing.assert_identical(res, want)


SNAPSHOTS = pd.date_range("2030-01-01", periods=3, freq="h", name="k")
KEY_COORDS: dict[str, pd.Index] = {
    "int": pd.Index([10, 20, 30], name="k"),
    "range": pd.RangeIndex(5, 8, name="k"),
    "str": pd.Index(["x", "y", "z"], name="k"),
    "datetime": SNAPSHOTS,
}


@pytest.mark.parametrize("lib", list(FRAME_LIBS))
@pytest.mark.parametrize("dtype", list(KEY_COORDS))
def test_from_long_key_dtypes(dtype: str, lib: str) -> None:
    coord = KEY_COORDS[dtype]
    frame = pd.DataFrame({"k": coord[[2, 0]], "v": [1.0, 2.0]})
    res = from_long(FRAME_LIBS[lib](frame), coords=[coord], default=0)
    want = xr.DataArray([2.0, 0, 1], coords=[coord], name="v")
    xr.testing.assert_identical(res, want)


@pytest.mark.parametrize("fill", [1.0, 0.5], ids=["full", "partial"])
def test_from_long_sorted_full_cube_matches_scatter(fill: float) -> None:
    coords = [pd.RangeIndex(40, name="g"), pd.date_range("2030", periods=30, name="t")]
    cube = pd.MultiIndex.from_product(coords).to_frame(index=False)
    cube["v"] = np.arange(len(cube), dtype=float)
    rows = cube.sample(frac=fill, random_state=0).sort_index()
    frame = to_polars(rows)
    shuffled = frame.sample(fraction=1.0, shuffle=True, seed=0)
    res = from_long(frame, coords=coords, default=-1)
    xr.testing.assert_identical(res, from_long(shuffled, coords=coords, default=-1))
    want = cube.v.where(cube.index.isin(rows.index), -1.0).to_numpy()
    np.testing.assert_array_equal(res.values, want.reshape(40, 30))


def x(m: Model) -> Variable:
    return m.add_variables(coords=[A, B], name="x")


def softened(m: Model, penalty: Any) -> LinearExpression | QuadraticExpression:
    v = x(m)
    m.add_objective(v.sum())
    m.add_constraints(v >= 0, penalty=penalty)
    return m.objective.expression


POSITIONS: dict[str, Callable[[Model, Any], Any]] = {
    "lower": lambda m, c: m.add_variables(lower=c, coords=[A, B], name="v").lower,
    "upper": lambda m, c: m.add_variables(upper=c, coords=[A, B], name="v").upper,
    "coefficient": lambda m, c: c * x(m),
    "to_linexpr": lambda m, c: x(m).to_linexpr(c),
    "add-constant": lambda m, c: 2 * x(m) + c,
    "sub-constant": lambda m, c: 2 * x(m) - c,
    "rsub-constant": lambda m, c: c - 2 * x(m),
    "rhs-add_constraints": lambda m, c: m.add_constraints(2 * x(m), "<=", c),
    "rhs-le": lambda m, c: 2 * x(m) <= c,
    "reflected-ge": lambda m, c: c >= 2 * x(m),
    "div": lambda m, c: x(m) / c,
    "quad-mul": lambda m, c: c * x(m) ** 2,
    "penalty": softened,
}


@pytest.mark.parametrize("case", ["all-keys-default", "missing-key-column"])
@pytest.mark.parametrize("position", list(POSITIONS))
def test_from_long_positions(backend: Model, position: str, case: str) -> None:
    frame, default, want = CONSTANTS[case]
    const = from_long(frame, coords=[A, B], default=default)
    if position == "penalty":
        const = const.fillna(1.0)
        want = want.fillna(1.0)
    expected = POSITIONS[position](Model(sparse=backend.sparse), want)
    assert_same(POSITIONS[position](backend, const), expected)


INVALID: dict[str, tuple[pd.DataFrame, type[Exception], str]] = {
    "no-value-column": (
        pd.DataFrame({"a": [0], "b": ["p"]}),
        ValueError,
        r"exactly one value column besides the key columns \['a', 'b'\] .*got \[\]",
    ),
    "no-key-column": (pd.DataFrame({"v": [1.0]}), ValueError, "no key column"),
    "two-value-columns": (
        pd.DataFrame({"a": [0], "z": [0], "v": [1.0]}),
        ValueError,
        r"exactly one value column .*got \['z', 'v'\]",
    ),
    "null-key": (
        pd.DataFrame({"b": ["p", None], "v": [1.0, 2]}),
        ValueError,
        "'b' has null values",
    ),
    "non-numeric-value": (
        pd.DataFrame({"a": [0], "v": ["1"]}),
        TypeError,
        "non-numeric",
    ),
    "duplicate-keys": (
        pd.DataFrame({"a": [0, 0], "v": [1.0, 2]}),
        ValueError,
        "duplicate keys",
    ),
    "key-outside-coords": (
        pd.DataFrame({"a": [7, 1, 9], "v": [1.0, 2, 3]}),
        ValueError,
        r"Values \[7, 9\] of column 'a' are not in",
    ),
    "null-value": (
        pd.DataFrame({"a": [0, 1], "v": pd.array([1, None], dtype="Int64")}),
        ValueError,
        "'v' has null values",
    ),
    "key-dtype-family": (
        pd.DataFrame({"a": ["0"], "v": [1.0]}),
        TypeError,
        "does not match",
    ),
    "complex-value": (
        pd.DataFrame({"a": [0], "v": [1 + 1j]}),
        TypeError,
        "'v' has unsupported dtype",
    ),
}


@pytest.mark.parametrize("lib", list(FRAME_LIBS))
@pytest.mark.parametrize("case", list(INVALID))
def test_from_long_invalid_raises(case: str, lib: str) -> None:
    frame, error, match = INVALID[case]
    with pytest.raises(error, match=match):
        from_long(FRAME_LIBS[lib](frame), coords=[A, B])


MIDNIGHT = pd.date_range("2030-01-01", periods=2, name="k")
SUB_MS = pd.DatetimeIndex(["2030-01-01 00:00:00.0004", "2030-01-01"], name="k")
BERLIN = pd.date_range("2030-01-01", periods=2, freq="h", name="k", tz="Europe/Berlin")
DAY = pl.Series([dt.date(2030, 1, 1)])
DAYS = pd.Index([dt.date(2030, 1, 1), dt.date(2030, 1, 2)], name="k", dtype=object)
DATE_STRINGS = pd.Index(["2030-01-01", "2030-01-02"], name="k")
MS = pl.Series([dt.datetime(2030, 1, 1)], dtype=pl.Datetime("ms"))
NOT_IN = (ValueError, "not in the coordinates")
NO_MATCH = (TypeError, "does not match")

KEY_DTYPES: dict[str, tuple[Column, pd.Index, list[float] | tuple[type, str]]] = {
    "int32": (pl.Series([300], dtype=pl.Int32), pd.Index([10, 300]), [np.nan, 1]),
    "uint8-range": (
        pl.Series([2], dtype=pl.UInt8),
        pd.RangeIndex(3),
        [np.nan, np.nan, 1],
    ),
    "uint64-range-overflow": (
        pl.Series([2**63 + 5], dtype=pl.UInt64),
        pd.RangeIndex(3),
        NOT_IN,
    ),
    "float-range": (pl.Series([2.5]), pd.RangeIndex(3), NOT_IN),
    "int-float-lossy": (pl.Series([10]), pd.Index([10.7, 20.0]), NOT_IN),
    "int-float-lossy-pandas": (pd.Series([10]), pd.Index([10.7, 20.0]), NOT_IN),
    "int-float": (pl.Series([1]), pd.Index([1.5, 1.0]), [np.nan, 1]),
    "date": (DAY, MIDNIGHT, NO_MATCH),
    "date-pandas": (pd.Series(DAY.to_list()), MIDNIGHT, NO_MATCH),
    "date-object": (DAY, DAYS, [1, np.nan]),
    "date-object-pandas": (pd.Series(DAY.to_list()), DAYS, [1, np.nan]),
    "date-string": (DAY, DATE_STRINGS, NO_MATCH),
    "date-string-pandas": (pd.Series(DAY.to_list()), DATE_STRINGS, NO_MATCH),
    "ms-sub-ms": (MS, SUB_MS, [np.nan, 1]),
    "ms-us": (MS, MIDNIGHT, [1, np.nan]),
    "tz": (MS.dt.replace_time_zone("Europe/Berlin"), BERLIN, [1, np.nan]),
    "tz-pandas": (pd.Series(BERLIN[[1]]), BERLIN, [np.nan, 1]),
    "tz-naive": (MS.dt.replace_time_zone("UTC"), MIDNIGHT, NO_MATCH),
    "naive-tz-pandas": (pd.Series(MIDNIGHT[[0]]), BERLIN, NO_MATCH),
    "duration": (pl.Series([1], dtype=pl.Duration("ns")), MIDNIGHT, NO_MATCH),
    "categorical": (pl.Series(["q"], dtype=pl.Categorical), B, [np.nan, 1]),
    "categorical-pandas": (pd.Series(pd.Categorical(["q"])), B, [np.nan, 1]),
    "categorical-int-pandas": (
        pd.Series(pd.Categorical([20])),
        pd.Index([10, 20]),
        [np.nan, 1],
    ),
    "categorical-int-str-pandas": (pd.Series(pd.Categorical([1])), B, NO_MATCH),
}


@pytest.mark.parametrize("case", list(KEY_DTYPES))
def test_from_long_column_dtypes(case: str) -> None:
    column, index, want = KEY_DTYPES[case]
    index = index.rename("k")
    data = {"k": column.rename("k"), "v": [1.0] * len(column)}
    frame = pl.DataFrame(data) if isinstance(column, pl.Series) else pd.DataFrame(data)
    if isinstance(want, tuple):
        with pytest.raises(want[0], match=want[1]):
            from_long(frame, coords=[index])
    else:
        res = from_long(frame, coords=[index])
        xr.testing.assert_identical(res, xr.DataArray(want, coords=[index], name="v"))


@pytest.mark.parametrize(
    "frame",
    [
        to_polars(FRAME_A).lazy(),
        FRAME_A.set_index("a").v,
        FRAME_A.set_index("a"),
        FRAME_AB.set_index(["a", "b"]),
    ],
    ids=["lazyframe", "series", "named-index", "multiindex"],
)
def test_from_long_rejects_frame_forms(frame: Any) -> None:
    with pytest.raises(TypeError):
        from_long(frame, coords=[A, B])


def incidence(relation: pd.DataFrame, weights: Any = 1.0) -> xr.DataArray:
    member, group = relation.columns
    weighted = relation.assign(w=weights)
    matrix = weighted.pivot_table("w", member, group, aggfunc="sum", fill_value=0.0)
    return xr.DataArray(matrix)


@pytest.mark.parametrize("lib", list(FRAME_LIBS))
@pytest.mark.parametrize("unlisted", [False, True], ids=["all", "unlisted"])
@pytest.mark.parametrize(
    "coords", [[B, A], [A, B]], ids=["member-last", "member-first"]
)
def test_relation_groupby_matches_incidence_matmul(
    backend: Model, coords: list[pd.Index], unlisted: bool, lib: str
) -> None:
    frame = RELATION_FRAME[RELATION_FRAME.a != 1] if unlisted else RELATION_FRAME
    expr = 2 * backend.add_variables(coords=coords, name="x") + 1
    res = expr.groupby(Relation(FRAME_LIBS[lib](frame), "a", "g")).sum()
    assert res.is_sparse == backend.sparse
    assert res.coord_dims == tuple("g" if c.name == "a" else c.name for c in coords)
    want = expr @ incidence(frame).reindex(a=A, fill_value=0.0)
    transposed = want._store.transpose([str(d) for d in res.coord_dims])
    assert_equivalent(res, LinearExpression._from_store(transposed, backend))


@pytest.mark.parametrize("lib", list(FRAME_LIBS))
def test_groupby_long_frame_raises(backend: Model, lib: str) -> None:
    with pytest.raises(TypeError, match="linopy.Relation"):
        (1 * x(backend)).groupby(FRAME_LIBS[lib](RELATION_FRAME)).sum()


def test_relation_groupby_quadratic_raises(backend: Model) -> None:
    with pytest.raises(NotImplementedError, match="quadratic"):
        (x(backend) ** 2).groupby(RELATION).sum()


@pytest.mark.parametrize(
    ("relation", "kwargs", "error", "match"),
    [
        (
            Relation(pd.DataFrame({"a": [0, 9], "g": ["u", "v"]}), "a", "g"),
            {},
            ValueError,
            r"Values \[9\] of column 'a' are not in",
        ),
        (
            Relation(pd.DataFrame({"a": [0, None], "g": ["u", "v"]}), "a", "g"),
            {},
            ValueError,
            "'a' has null values",
        ),
        (
            Relation(pd.DataFrame({"a": ["0"], "g": ["u"]}), "a", "g"),
            {},
            TypeError,
            "does not match",
        ),
        (
            Relation(pd.DataFrame({"a": [0, 1], "b": ["p", "q"]}), "a", "b"),
            {},
            ValueError,
            "must not name",
        ),
        (
            Relation(pd.DataFrame({"c": [0], "g": ["u"]}), "c", "g"),
            {},
            ValueError,
            "must name a dimension",
        ),
        (RELATION, {"use_fallback": True}, ValueError, "not supported"),
        (RELATION, {"observed": True}, ValueError, "not supported"),
    ],
    ids=[
        "outside",
        "null-member",
        "member-dtype",
        "group-is-dim",
        "member-not-dim",
        "fallback",
        "observed",
    ],
)
def test_invalid_relation_groupby_raises(
    backend: Model,
    relation: Relation,
    kwargs: dict[str, bool],
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        (1 * x(backend)).groupby(relation).sum(**kwargs)


@pytest.mark.parametrize("lib", list(FRAME_LIBS))
@pytest.mark.parametrize(
    ("frame", "group", "match"),
    [
        (pd.DataFrame({"a": [0]}), "g", r"no columns \['g'\]"),
        (pd.DataFrame({"a": [0], "g": [None]}), "g", "'g' has null values"),
        (pd.DataFrame({"a": [0]}), "a", "must differ, both are 'a'"),
        (
            pd.DataFrame({"a": [0], "g": ["u"], "w": [1.0]}),
            "g",
            r"exactly the columns 'a' and 'g', got \['a', 'g', 'w'\]",
        ),
    ],
    ids=["missing-column", "null-group", "same-column", "extra-column"],
)
def test_invalid_relation_raises(
    frame: pd.DataFrame, group: str, match: str, lib: str
) -> None:
    with pytest.raises(ValueError, match=match):
        Relation(FRAME_LIBS[lib](frame), "a", group)


DUPLICATE_A = pd.Index([0, 0, 1], name="a")
DUPLICATE_LABELS: dict[str, Callable[[Model, Callable[[pd.DataFrame], Any]], Any]] = {
    "from_long": lambda m, lib: from_long(lib(FRAME_A), coords=[DUPLICATE_A]),
    "relation": lambda m, lib: (
        m.add_variables(coords=[DUPLICATE_A], name="x")
        .groupby(Relation(lib(RELATION_FRAME), "a", "g"))
        .sum()
    ),
}


@pytest.mark.parametrize("lib", list(FRAME_LIBS))
@pytest.mark.parametrize("case", list(DUPLICATE_LABELS))
def test_duplicate_coordinate_labels_raise(backend: Model, case: str, lib: str) -> None:
    with pytest.raises(ValueError, match="dimension 'a' have duplicate labels"):
        DUPLICATE_LABELS[case](backend, FRAME_LIBS[lib])


LINKS = pd.Index(["l0", "l1", "l2", "l3"], name="link")
TIMES = pd.RangeIndex(2, name="snapshot")
PORTS = pd.DataFrame(
    {
        "link": list(LINKS) * 2,
        "port": [0] * 4 + [1] * 4,
        "bus": ["b0", "b0", "b1", "b2", "b1", "b2", "b2", "b0"],
    }
)
EFFICIENCY = pd.DataFrame({"link": LINKS, "eff": [0.9, 0.8, 0.95, 0.7]})


LINK_BUS = PORTS[["link", "bus"]]


def port(i: int) -> Relation:
    return Relation(LINK_BUS[PORTS.port == i], "link", "bus")


def test_link_ports_to_bus(backend: Model) -> None:
    p = backend.add_variables(coords=[TIMES, LINKS], name="p")
    eff = from_long(EFFICIENCY, coords=[TIMES, LINKS])
    touching = p.groupby(Relation(LINK_BUS, "link", "bus")).sum()
    balance = (-1 * p).groupby(port(0)).sum() + (eff * p).groupby(port(1)).sum()

    weights = np.where(PORTS.port == 0, -1.0, np.tile(EFFICIENCY.eff, 2))
    for res, matrix in [
        (touching, incidence(LINK_BUS)),
        (balance, incidence(LINK_BUS, weights)),
    ]:
        assert isinstance(res, LinearExpression)
        assert res.is_sparse == backend.sparse
        assert_equivalent(res, 1 * p @ matrix)


EVOLVING: dict[str, Callable[[], object]] = {
    "from_long": lambda: from_long(FRAME_A, coords=[pd.RangeIndex(3, name="a")]),
    "Relation": lambda: Relation(RELATION_FRAME, "a", "g"),
}


@pytest.mark.parametrize("api", list(EVOLVING))
def test_long_form_warns_evolving_api_once(api: str) -> None:
    _emitted_evolving_warnings.discard(api)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", EvolvingAPIWarning)
        EVOLVING[api]()
        EVOLVING[api]()
    messages = [str(w.message) for w in caught if w.category is EvolvingAPIWarning]
    assert len(messages) == 1
    assert messages[0].startswith(f"long-form: {api} ")
    assert caught[0].filename == __file__
