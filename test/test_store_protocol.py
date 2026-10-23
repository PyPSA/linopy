"""
The store protocol: :class:`~linopy.store.DenseStore` and
:class:`~linopy.csr.CSRStore` give equivalent results for every store method.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import xarray as xr
from test_store_conformance import COND, OVER_AB, SERIES, B, C, grid, no_densify

from linopy import LinearExpression, Model, PerformanceWarning
from linopy.csr import CSRStore, Grid
from linopy.store import DenseStore, Store
from linopy.testing import assert_equivalent

BUILDS: dict[str, Callable[[Model], LinearExpression]] = {
    "grouped": lambda m: grid(m).groupby(SERIES).sum(),
    "aux": grid,
    "absent": lambda m: grid(m).where(COND),
}

STORE_CLASSES: list[type[Store]] = [DenseStore, CSRStore]


def build_store(build: str, cls: type[Store]) -> Store:
    expr = BUILDS[build](Model())
    store = cls.from_dense(expr.data, expr.model)
    assert isinstance(store, cls)
    return store


def as_expr(store: Store) -> LinearExpression:
    return LinearExpression._from_store(store, store.model)


def collapsed(s: Store, dim: str) -> tuple[Grid, np.ndarray]:
    target = s.grid.select(d for d in s.grid.dims if d != dim)
    return target, target.indexer(s.grid)[0]


def concatenated(s: Store) -> Store:
    other = s.with_grid(s.grid.with_indexes({"b": ["r", "s"]}))
    target = s.grid.with_indexes({"b": ["p", "q", "r", "s"]})
    res = s.concat([other], "b", target)
    assert not isinstance(res, str)
    return res


def solution(s: Store) -> np.ndarray:
    values = np.append(np.arange(s.model._xCounter, dtype=float), np.nan)
    return s.solution(lambda labels: values)


def taken(s: Store) -> Store:
    rows = np.arange(s.n_cells)[::-1].copy()
    rows[0] = -1
    return s.take(rows, s.grid)


SAME_GRID: dict[str, Callable[[Store], Store]] = {
    "scale": lambda s: s.scale(2.0),
    "scale-cells": lambda s: s.scale(np.arange(s.n_cells, dtype=float) % 3),
    "scale-div": lambda s: s.scale(
        np.arange(s.n_cells, dtype=float) % 3, operator.truediv
    ),
    "assign-const": lambda s: s.assign_const(
        np.where(np.arange(s.n_cells) % 2, np.nan, 1.0)
    ),
    "assign-const-fillna": lambda s: s.assign_const(
        np.where(np.arange(s.n_cells) % 2, np.nan, 1.0)
    ).fillna(0.0),
    "fillna": lambda s: s.fillna(3.0),
    "prune": lambda s: s.scale(np.arange(s.n_cells, dtype=float) % 2).prune(),
    "canonical": lambda s: s.add(s).canonical(),
    "add": lambda s: s.add(s.scale(-2.0), s),
    "take": taken,
}

OPS: dict[str, Callable[[Store], Store]] = SAME_GRID | {
    "with-grid": lambda s: s.with_grid(replace(s.grid, aux={})),
    "broadcast": lambda s: s.broadcast(Grid({**s.grid.indexes, "c": C})),
    "reindex": lambda s: s.reindex(s.grid.with_indexes({"b": ["q", "r"]})),
    "reindex-fill": lambda s: s.reindex(s.grid.with_indexes({"b": ["q", "r"]}), 0.0),
    "rename": lambda s: s.rename({"b": "bb"}),
    "sum": lambda s: s.sum(["b"]),
    "sum-all": lambda s: s.sum(s.grid.dims),
    "aggregate": lambda s: s.aggregate(*collapsed(s, "b")),
    "concat": concatenated,
    "dot": lambda s: s.dot(
        scipy.sparse.csr_array(np.ones((len(B), 1))),
        ["b"],
        [pd.Index(["z"], name="z")],
    ),
    "from-grouper": lambda s: type(s).from_grouper(
        s, s.model, pd.Series(["x", "x"], index=B), "g2", True, s.grid.dims
    ),
}


@pytest.mark.parametrize("cls", STORE_CLASSES)
@pytest.mark.parametrize("build", list(BUILDS))
def test_round_trip_through_dataset(build: str, cls: type[Store]) -> None:
    store = build_store(build, cls)
    back = cls.from_dense(store.to_dataset(), store.model)
    assert isinstance(back, cls)
    assert back.grid == store.grid
    assert_equivalent(as_expr(back), as_expr(store))


@pytest.mark.parametrize("build", list(BUILDS))
def test_conversions_agree(build: str) -> None:
    dense, sparse = (build_store(build, cls) for cls in STORE_CLASSES)
    assert isinstance(sparse.to_dense(), DenseStore)
    assert isinstance(dense.to_sparse(), CSRStore)
    assert dense.to_dense() is dense and sparse.to_sparse() is sparse
    assert_equivalent(as_expr(sparse.to_dense()), as_expr(dense))
    assert_equivalent(as_expr(dense.to_sparse()), as_expr(sparse))


@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize("build", list(BUILDS))
def test_store_methods_agree(build: str, op: str) -> None:
    dense, sparse = (OPS[op](build_store(build, cls)) for cls in STORE_CLASSES)
    assert isinstance(dense, DenseStore) and isinstance(sparse, CSRStore)
    assert dense.grid == sparse.grid
    assert_equivalent(as_expr(dense), as_expr(sparse))


@pytest.mark.parametrize("cls", STORE_CLASSES)
@pytest.mark.parametrize("op", list(SAME_GRID))
@pytest.mark.parametrize("build", list(BUILDS))
def test_grid_preserved(build: str, op: str, cls: type[Store]) -> None:
    store = build_store(build, cls)
    assert SAME_GRID[op](store).grid == store.grid


QUERIES: dict[str, Callable[[Store], np.ndarray]] = {
    "const": lambda s: s.const,
    "has-terms": lambda s: s.has_terms(),
    "solution": solution,
    "n-cells": lambda s: np.array(s.n_cells),
}


@pytest.mark.parametrize("query", list(QUERIES))
@pytest.mark.parametrize("build", list(BUILDS))
def test_queries_agree(build: str, query: str) -> None:
    dense, sparse = (QUERIES[query](build_store(build, cls)) for cls in STORE_CLASSES)
    np.testing.assert_array_equal(dense, sparse)


@pytest.mark.parametrize("build", list(BUILDS))
def test_cells_agree(build: str) -> None:
    dense, sparse = (build_store(build, cls) for cls in STORE_CLASSES)
    for indices in np.ndindex(*dense.grid.shape):
        (dc, dv, dk), (sc, sv, sk) = dense.cell(indices), sparse.cell(indices)
        live = dv != -1
        np.testing.assert_array_equal(np.sort(dv[live]), sv)
        np.testing.assert_array_equal(dk, sk)
        assert sorted(dc[live]) == sorted(sc)


@pytest.mark.parametrize("cls", STORE_CLASSES)
def test_from_labels(cls: type[Store]) -> None:
    m = Model()
    x = m.add_variables(coords=[COND.indexes["a"], B], name="x")
    labels = x.labels.where(COND, -1)
    ds = xr.Dataset({"coeffs": xr.full_like(labels, 2.0, dtype=float), "vars": labels})
    want = (2 * x).where(COND)
    assert_equivalent(as_expr(cls.from_labels(ds, m)), want)


CONSTS: dict[str, xr.DataArray] = {
    "plain": OVER_AB,
    "new-aux": OVER_AB.assign_coords(extra=("b", ["u", "v"])),
    "same-aux": OVER_AB.assign_coords(tag=("a", ["u", "v", "u"])),
}


@pytest.mark.parametrize("const", list(CONSTS))
def test_const_setter_swaps_the_store(backend: Model, const: str) -> None:
    expr, want = grid(backend), grid(Model())
    with no_densify():
        expr.const = CONSTS[const]
    want.const = CONSTS[const]
    assert expr.is_sparse == backend.sparse
    assert_equivalent(expr, want)
    assert set(expr.coords) == set(want.coords)


def test_const_setter_densifies_2d_coords() -> None:
    const = OVER_AB.assign_coords(two=(("a", "b"), np.arange(6).reshape(3, 2)))
    expr, want = grid(Model(sparse=True)), grid(Model())
    with pytest.warns(PerformanceWarning, match="assignment of `const`"):
        expr.const = const
    want.const = const
    assert not expr.is_sparse
    assert_equivalent(expr, want)
    assert set(expr.coords) == set(want.coords)


def test_const_setter_rejects_clashing_aux(backend: Model) -> None:
    expr = grid(backend)
    with pytest.raises(xr.MergeError, match="tag"):
        expr.const = OVER_AB.assign_coords(tag=("a", ["w", "w", "w"]))


def test_dense_store_grid_rejects_2d_coords() -> None:
    expr = grid(Model()).assign_coords(two=(("a", "b"), np.arange(6).reshape(3, 2)))
    with pytest.raises(ValueError, match="cannot hold the coordinates"):
        DenseStore(expr.data, expr.model).grid


def test_vars_setter_densifies_with_reason() -> None:
    expr, dense = grid(Model(sparse=True)), grid(Model())
    with pytest.warns(PerformanceWarning, match="assignment of `vars`"):
        expr.vars = dense.vars
    assert not expr.is_sparse
    assert_equivalent(expr, dense)


@pytest.mark.parametrize("cls", STORE_CLASSES)
def test_absent_cell_drops_its_terms(cls: type[Store]) -> None:
    store = build_store("aux", cls)
    cells, labels, _ = store.terms()
    absent = np.arange(store.n_cells) % 2 == 1
    expr = as_expr(store.assign_const(np.where(absent, np.nan, 1.0)))
    live = np.sort(labels[~absent[cells]])
    np.testing.assert_array_equal(expr.has_terms.values.ravel(), ~absent)
    np.testing.assert_array_equal(np.sort(expr.flat["vars"]), live)
    summed = expr.sum().vars.values
    np.testing.assert_array_equal(np.sort(summed[summed != -1]), live)
