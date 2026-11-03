"""
``sel`` of every label, in order, returns a shallow copy without indexing.

Any other selection keeps going through xarray's ``sel``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model, Variable
from linopy.common import selects_everything
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


@pytest.fixture(params=[False, True], ids=["dense", "sparse"])
def m(request: pytest.FixtureRequest) -> Model:
    return Model(sparse=True) if request.param else Model()


@pytest.fixture
def x(m: Model) -> Variable:
    return m.add_variables(lower=0, upper=5, coords=[A, B], name="x")


@pytest.fixture
def objects(m: Model, x: Variable) -> dict[str, Any]:
    expr = 2 * x + 1
    return {
        "variable": x,
        "linear": expr,
        "quadratic": x * x,
        "constraint": m.add_constraints(2 * x >= 1, name="c"),
    }


ASSERT_EQUAL = {
    "variable": assert_varequal,
    "linear": assert_linequal,
    "quadratic": assert_quadequal,
    "constraint": assert_conequal,
}


IDENTITY_SELECTIONS = {
    "list": ((), {"a": [10, 20, 30]}),
    "array": ((), {"a": np.array([10, 20, 30])}),
    "index": ((), {"a": pd.Index([10, 20, 30])}),
    "same-named index": ((), {"a": pd.Index([10, 20, 30], name="a")}),
    "own index": ((), {"a": A}),
    "two dims": ((), {"a": [10, 20, 30], "b": ["x", "y"]}),
    "dict": (({"b": ["x", "y"]},), {}),
    "indexers keyword": ((), {"indexers": {"b": ["x", "y"]}}),
    "default options": ((), {"a": [10, 20, 30], "method": None, "drop": False}),
}


@pytest.mark.parametrize("kind", list(ASSERT_EQUAL))
@pytest.mark.parametrize(
    "selection", IDENTITY_SELECTIONS.values(), ids=list(IDENTITY_SELECTIONS)
)
def test_identity_sel_is_shortcut(
    objects: dict[str, Any], kind: str, selection: tuple[tuple, dict]
) -> None:
    obj = objects[kind]
    args, kwargs = selection
    result = obj.sel(*args, **kwargs)
    assert type(result) is type(obj)
    assert result is not obj
    ASSERT_EQUAL[kind](result, obj)
    ASSERT_EQUAL[kind](result, obj._sel(*args, **kwargs))


def test_identity_sel_shares_data(m: Model, x: Variable) -> None:
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


def test_identity_sel_is_independent(m: Model, x: Variable) -> None:
    result = x.sel(a=[10, 20, 30])
    result.update(lower=3)
    assert (x.lower == 0).all()
    assert (m.variables["x"].lower == 0).all()

    expr = 2 * x
    selected = expr.sel(a=[10, 20, 30])
    selected.const = selected.const + 1
    assert (expr.const == 0).all()

    con = m.add_constraints(x >= 1, name="c")
    copy = con.sel(a=[10, 20, 30])
    copy.update(rhs=2)
    assert (con.rhs == 1).all()
    assert (m.constraints["c"].rhs == 1).all()


NON_IDENTITY_SELECTIONS = {
    "subset": ((), {"a": [10, 20]}),
    "reordered": ((), {"a": [30, 20, 10]}),
    "duplicates": ((), {"a": [10, 10, 20]}),
    "scalar": ((), {"a": 10}),
    "slice": ((), {"a": slice(None)}),
    "full slice": ((), {"a": slice(10, 30)}),
    "tuple": ((), {"a": (10, 20, 30)}),
    "other-named index": ((), {"a": pd.Index([10, 20, 30], name="z")}),
    "dataarray": ((), {"a": xr.DataArray([10, 20, 30], dims="a")}),
    "renaming dataarray": ((), {"a": xr.DataArray([10, 20, 30], dims="z")}),
    "one identity, one subset": ((), {"a": [10, 20, 30], "b": ["x"]}),
    "nearest": ((), {"a": [10, 20, 30], "method": "nearest"}),
    "tolerance": ((), {"a": [10, 20, 30], "method": "nearest", "tolerance": 1}),
    "drop": ((), {"a": [10, 20, 30], "drop": True}),
    "2d array": ((), {"a": np.array([[10, 20, 30]])}),
}


@pytest.mark.parametrize("kind", list(ASSERT_EQUAL))
@pytest.mark.parametrize(
    "selection", NON_IDENTITY_SELECTIONS.values(), ids=list(NON_IDENTITY_SELECTIONS)
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
        expected = obj._sel(*args, **kwargs)
    except Exception as err:
        expected = err
    for cls in (Variable, LinearExpression, QuadraticExpression, Constraint):
        monkeypatch.setattr(cls, "_shallow_copy", _no_shortcut)
    if isinstance(expected, Exception):
        with pytest.raises(type(expected)):
            obj.sel(*args, **kwargs)
        return
    ASSERT_EQUAL[kind](obj.sel(*args, **kwargs), expected)


def _no_shortcut(self: Any) -> Any:
    raise AssertionError("sel took the identity shortcut")


def test_selects_everything_rejects() -> None:
    indexes = {"a": A, "m": pd.MultiIndex.from_product([[1], [2, 3]])}
    labels = [10, 20, 30]
    assert selects_everything(indexes, (), {"a": labels})
    assert not selects_everything(indexes, (), {})
    assert not selects_everything(indexes, (), {"c": labels})
    assert not selects_everything(indexes, ({"a": labels}, "nearest"), {})
    assert not selects_everything(indexes, ({"a": labels},), {"a": labels})
    assert not selects_everything(indexes, (), {"m": [(1, 2), (1, 3)]})
    assert not selects_everything(indexes, (), {"m": indexes["m"]})
    duplicated = {"a": pd.Index([10, 10], name="a")}
    assert not selects_everything(duplicated, (), {"a": duplicated["a"]})
    assert not selects_everything(duplicated, (), {"a": [10, 10]})
    ints01 = {"a": pd.Index([0, 1], name="a")}
    assert not selects_everything(ints01, (), {"a": [False, True]})
    assert not selects_everything(ints01, (), {"a": np.array([False, True])})
    bools = {"a": pd.Index([False, True], name="a")}
    assert not selects_everything(bools, (), {"a": bools["a"]})
    assert not selects_everything(bools, (), {"a": [False, True]})


def test_sel_of_boolean_mask_unchanged(m: Model) -> None:
    v = m.add_variables(coords=[pd.Index([0, 1], name="a")], name="v")
    assert_varequal(v.sel(a=[False, True]), v._sel(a=[False, True]))
    assert v.sel(a=[False, True]).sizes["a"] == 1
