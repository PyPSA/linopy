"""
``sel`` leaves out indexers that select every label of their dimension in
order; if all are such, it returns a shallow copy without indexing.

A CSR-backed expression gathering every row in place keeps its store.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import linopy.common
from linopy import Model, Variable
from linopy.common import is_full_indexer
from linopy.constraints import Constraint
from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.testing import (
    assert_conequal,
    assert_linequal,
    assert_quadequal,
    assert_varequal,
)

A = pd.Index([10, 20, 30], name="a")
B = pd.Index(["x", "y"], name="b")
KINDS = (Variable, LinearExpression, QuadraticExpression, Constraint)


@pytest.fixture(params=[False, True], ids=["dense", "sparse"])
def m(request: pytest.FixtureRequest) -> Model:
    return Model(sparse=True) if request.param else Model()


@pytest.fixture
def x(m: Model) -> Variable:
    return m.add_variables(lower=0, upper=5, coords=[A, B], name="x")


@pytest.fixture
def objects(m: Model, x: Variable) -> dict[str, Any]:
    return {
        "variable": x,
        "linear": 2 * x + 1,
        "quadratic": x * x,
        "constraint": m.add_constraints(2 * x >= 1, name="c"),
    }


ASSERT_EQUAL: dict[str, Callable[[Any, Any], None]] = {
    "variable": assert_varequal,
    "linear": assert_linequal,
    "quadratic": assert_quadequal,
    "constraint": assert_conequal,
}


def _plain_sel(
    obj: Any, args: tuple, kwargs: dict, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """``obj.sel(*args, **kwargs)`` with no indexer taken as full."""
    with monkeypatch.context() as patch:
        patch.setattr(linopy.common, "is_full_indexer", lambda *_: False)
        return obj.sel(*args, **kwargs)


def _no_shortcut(self: Any) -> Any:
    raise AssertionError("sel took the full-selection shortcut")


FULL_SELECTIONS = {
    "list": ((), {"a": [10, 20, 30]}),
    "array": ((), {"a": np.array([10, 20, 30])}),
    "index": ((), {"a": pd.Index([10, 20, 30])}),
    "same-named index": ((), {"a": pd.Index([10, 20, 30], name="a")}),
    "own index": ((), {"a": A}),
    "full slice": ((), {"a": slice(None)}),
    "two dims": ((), {"a": [10, 20, 30], "b": ["x", "y"]}),
    "dict": (({"b": ["x", "y"]},), {}),
    "indexers keyword": ((), {"indexers": {"b": ["x", "y"]}}),
    "drop": ((), {"a": [10, 20, 30], "drop": True}),
}


@pytest.mark.parametrize("kind", list(ASSERT_EQUAL))
@pytest.mark.parametrize(
    "selection", FULL_SELECTIONS.values(), ids=list(FULL_SELECTIONS)
)
def test_full_sel_is_shallow_copy(
    objects: dict[str, Any],
    kind: str,
    selection: tuple[tuple, dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obj = objects[kind]
    args, kwargs = selection
    result = obj.sel(*args, **kwargs)
    assert type(result) is type(obj)
    assert result is not obj
    ASSERT_EQUAL[kind](result, obj)
    ASSERT_EQUAL[kind](result, _plain_sel(obj, args, kwargs, monkeypatch))


PARTIAL_SELECTIONS = {
    "full and subset": ({"a": [10, 20, 30], "b": ["x"]}, {"b": ["x"]}),
    "full and scalar": ({"a": [10, 20, 30], "b": "y"}, {"b": "y"}),
    "full and slice": ({"a": A, "b": slice("y", None)}, {"b": slice("y", None)}),
}


@pytest.mark.parametrize("kind", list(ASSERT_EQUAL))
@pytest.mark.parametrize(
    "selection", PARTIAL_SELECTIONS.values(), ids=list(PARTIAL_SELECTIONS)
)
def test_full_indexers_are_left_out(
    objects: dict[str, Any],
    kind: str,
    selection: tuple[dict, dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obj = objects[kind]
    indexers, rest = selection
    expected = _plain_sel(obj, (), indexers, monkeypatch)
    for cls in KINDS:
        monkeypatch.setattr(cls, "_shallow_copy", _no_shortcut)
    ASSERT_EQUAL[kind](obj.sel(indexers), expected)
    ASSERT_EQUAL[kind](obj.sel(rest), expected)


OTHER_SELECTIONS = {
    "subset": ((), {"a": [10, 20]}),
    "reordered": ((), {"a": [30, 20, 10]}),
    "duplicates": ((), {"a": [10, 10, 20]}),
    "scalar": ((), {"a": 10}),
    "label slice": ((), {"a": slice(10, 30)}),
    "tuple": ((), {"a": (10, 20, 30)}),
    "other-named index": ((), {"a": pd.Index([10, 20, 30], name="z")}),
    "dataarray": ((), {"a": xr.DataArray([10, 20, 30], dims="a")}),
    "renaming dataarray": ((), {"a": xr.DataArray([10, 20, 30], dims="z")}),
    "nearest": ((), {"a": [10, 20, 30], "method": "nearest"}),
    "tolerance": ((), {"a": [10, 20, 30], "method": "nearest", "tolerance": 1}),
    "positional method": (({"a": [10, 20, 30]}, "nearest"), {}),
    "dict and keywords": (({"a": [10, 20, 30]},), {"b": ["x", "y"]}),
    "2d array": ((), {"a": np.array([[10, 20, 30]])}),
}


@pytest.mark.parametrize("kind", list(ASSERT_EQUAL))
@pytest.mark.parametrize(
    "selection", OTHER_SELECTIONS.values(), ids=list(OTHER_SELECTIONS)
)
def test_other_sel_unchanged(
    objects: dict[str, Any],
    kind: str,
    selection: tuple[tuple, dict],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obj = objects[kind]
    args, kwargs = selection
    try:
        expected = _plain_sel(obj, args, kwargs, monkeypatch)
    except Exception as err:
        expected = err
    for cls in KINDS:
        monkeypatch.setattr(cls, "_shallow_copy", _no_shortcut)
    if isinstance(expected, Exception):
        with pytest.raises(type(expected)):
            obj.sel(*args, **kwargs)
        return
    ASSERT_EQUAL[kind](obj.sel(*args, **kwargs), expected)


def test_full_sel_shares_data(x: Variable) -> None:
    result = x.sel(a=[10, 20, 30])
    assert np.shares_memory(result.labels.values, x.labels.values)
    assert result.name == x.name

    expr = 2 * x
    selected = expr.sel(a=[10, 20, 30])
    assert selected.is_sparse == expr.is_sparse
    if expr.is_sparse:
        assert selected._store is expr._store
    else:
        assert np.shares_memory(selected.coeffs.values, expr.coeffs.values)


def test_full_sel_is_independent(m: Model, x: Variable) -> None:
    result = x.sel(a=[10, 20, 30])
    result.update(lower=3)
    result.attrs["note"] = "copy"
    assert (x.lower == 0).all()
    assert (m.variables["x"].lower == 0).all()
    assert "note" not in x.attrs

    expr = 2 * x
    selected = expr.sel(a=[10, 20, 30])
    selected.const = selected.const + 1
    assert (expr.const == 0).all()

    con = m.add_constraints(x >= 1, name="c")
    copy = con.sel(a=[10, 20, 30])
    copy.update(rhs=2)
    copy.attrs["note"] = "copy"
    assert (con.rhs == 1).all()
    assert (m.constraints["c"].rhs == 1).all()
    assert "note" not in con.attrs


def test_is_full_indexer() -> None:
    labels = [10, 20, 30]
    assert is_full_indexer(A, labels)
    assert is_full_indexer(A, slice(None))
    assert not is_full_indexer(None, labels)
    assert not is_full_indexer(A, slice(10, None))
    assert not is_full_indexer(A, (10, 20, 30))
    assert not is_full_indexer(A, np.array([labels]))
    multi = pd.MultiIndex.from_product([[1], [2, 3]], names=["m0", "m1"])
    assert not is_full_indexer(multi, [(1, 2), (1, 3)])
    assert not is_full_indexer(multi, multi)
    duplicated = pd.Index([10, 10], name="a")
    assert not is_full_indexer(duplicated, duplicated)
    assert not is_full_indexer(duplicated, [10, 10])
    ints01 = pd.Index([0, 1], name="a")
    assert not is_full_indexer(ints01, [False, True])
    assert not is_full_indexer(ints01, np.array([False, True]))
    bools = pd.Index([False, True], name="a")
    assert not is_full_indexer(bools, bools)
    assert not is_full_indexer(bools, [False, True])


def test_sel_of_boolean_mask_unchanged(m: Model) -> None:
    v = m.add_variables(coords=[pd.Index([0, 1], name="a")], name="v")
    assert v.sel(a=[False, True]).sizes["a"] == 1


def test_sparse_gather_in_place_keeps_store() -> None:
    m = Model(sparse=True)
    x = m.add_variables(coords=[A, B], name="x")
    expr = 2 * x + 1
    assert expr.is_sparse
    for kept in (
        expr.isel(a=slice(None)),
        expr.isel(a=slice(0, 3)),
        expr.sel(a=slice(10, 30)),
        expr.roll(a=0),
    ):
        assert kept._store is expr._store
    moved = expr.isel(a=slice(1, None))
    assert moved._store is not expr._store
    dense = LinearExpression(expr._to_dataset().isel(a=slice(1, None)), m)
    assert_linequal(moved, dense)
