"""
The store protocol: :class:`~linopy.store.DenseStore` and
:class:`~linopy.csr.CSRStore` give equivalent results for every store method.
"""

from __future__ import annotations

import operator
import warnings
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
import xarray as xr
from test_store_conformance import (
    COND,
    OVER_AB,
    SERIES,
    B,
    C,
    grid,
    no_densify,
    with_2d_coord,
)

import linopy
from linopy import LinearExpression, Model, PerformanceWarning
from linopy.csr import CSRStore, Grid
from linopy.store import DenseStore, Store, as_sparse
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


def cell_values(s: Store, values: np.ndarray) -> xr.DataArray:
    return s.dataarray(values, "values")


def combined(s: Store, reindex: bool) -> Store:
    target = s.reindex(s.grid.with_indexes({"b": ["q", "r"]})) if reindex else s
    const = cell_values(target, np.arange(target.n_cells, dtype=float))
    factor = cell_values(target, np.arange(target.n_cells, dtype=float) % 3)
    res = s.combine(const, operator.mul, factor, reindex)
    assert not isinstance(res, str)
    return res


def taken(s: Store) -> Store:
    rows = np.arange(s.n_cells)[::-1].copy()
    rows[0] = -1
    return s.take(rows, s.grid)


SAME_GRID: dict[str, Callable[[Store], Store]] = {
    "scale": lambda s: s.scale(2.0),
    "scale-cells": lambda s: s.scale(np.arange(s.n_cells, dtype=float) % 3),
    "scale-div": lambda s: s.scale(
        np.arange(s.n_cells, dtype=float) % 3 + 1, operator.truediv
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
    "absorbed": lambda s: s.assign_const(
        np.where(np.arange(s.n_cells) % 2, np.nan, 1.0)
    ).absorbed(),
    "combine": lambda s: combined(s, False),
    "take": taken,
    "sum-no-dim": lambda s: s.sum([]),
}

OPS: dict[str, Callable[[Store], Store]] = SAME_GRID | {
    "with-grid": lambda s: s.with_grid(replace(s.grid, aux={})),
    "broadcast": lambda s: s.broadcast(Grid({**s.grid.indexes, "c": C})),
    "reindex": lambda s: s.reindex(s.grid.with_indexes({"b": ["q", "r"]})),
    "reindex-fill": lambda s: s.reindex(s.grid.with_indexes({"b": ["q", "r"]}), 0.0),
    "rename": lambda s: s.rename({"b": "bb"}),
    "sum": lambda s: s.sum(["b"]),
    "sum-all": lambda s: s.sum(s.grid.dims),
    "sum-none": lambda s: s.sum(),
    "combine-reindex": lambda s: combined(s, True),
    "aggregate": lambda s: s.aggregate(*collapsed(s, "b")),
    "concat": concatenated,
    "from-matmul": lambda s: type(s).from_matmul(
        as_sparse(s),
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
    converted = dense.to_sparse()
    assert isinstance(converted, CSRStore)
    assert dense.to_dense() is dense and sparse.to_sparse() is sparse
    assert_equivalent(as_expr(sparse.to_dense()), as_expr(dense))
    assert_equivalent(as_expr(converted), as_expr(sparse))


def test_model_store_cls(backend: Model) -> None:
    assert backend.store_cls is (CSRStore if backend.sparse else DenseStore)


def test_dense_to_sparse_names_off_grid_coords() -> None:
    expr = with_2d_coord(as_expr(build_store("aux", DenseStore)))
    with no_densify():
        assert expr._store.to_sparse("why") == "coordinates the Grid cannot hold: two"


@pytest.mark.parametrize("reason", [None, "why"])
def test_dense_to_sparse_notice(reason: str | None) -> None:
    dense = build_store("aux", DenseStore)
    with linopy.options as opts, warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        opts.set_value(warn_on_densify=True)
        assert isinstance(dense.to_sparse(reason), CSRStore)
    notices = [str(w.message) for w in rec]
    assert notices == (
        [] if reason is None else [f"Dense backing sparsified: {reason}."]
    )


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


def buffers(s: Store) -> list[np.ndarray]:
    if isinstance(s, CSRStore):
        return [s.csr.data, s.const]
    assert isinstance(s, DenseStore)
    return [s.ds.coeffs.values, s.ds.const.values]


@pytest.mark.parametrize("deep", [True, False])
@pytest.mark.parametrize("cls", STORE_CLASSES)
@pytest.mark.parametrize("build", list(BUILDS))
def test_copy_moves_to_model(build: str, cls: type[Store], deep: bool) -> None:
    store, model = build_store(build, cls), Model()
    copied = store.copy(model, deep)
    assert type(copied) is cls and copied.model is model
    assert_equivalent(as_expr(copied), as_expr(store))
    shared = [np.shares_memory(a, b) for a, b in zip(buffers(copied), buffers(store))]
    assert shared == [not deep] * len(shared)


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
    with pytest.warns(PerformanceWarning, match=r"assignment\(const\)"):
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
    with pytest.warns(PerformanceWarning, match=r"assignment\(vars\)"):
        expr.vars = dense.vars
    assert not expr.is_sparse
    assert_equivalent(expr, dense)


@pytest.mark.parametrize("cls", STORE_CLASSES)
def test_absent_cell_drops_its_terms(cls: type[Store]) -> None:
    store = build_store("aux", cls)
    counts, labels, _ = store.terms()
    cells = np.repeat(np.arange(store.n_cells), counts)
    absent = np.arange(store.n_cells) % 2 == 1
    expr = as_expr(store.assign_const(np.where(absent, np.nan, 1.0)))
    live = np.sort(labels[~absent[cells]])
    np.testing.assert_array_equal(expr.has_terms.values.ravel(), ~absent)
    np.testing.assert_array_equal(np.sort(expr.flat["vars"]), live)
    summed = expr.sum().vars.values
    np.testing.assert_array_equal(np.sort(summed[summed != -1]), live)


def zero_coeff(m: Model) -> LinearExpression:
    """A grid with a zero coefficient in some cells, kept in the store."""
    expr = grid(m)
    return expr.assign(coeffs=expr.coeffs.where(expr.coeffs.a != 1, 0.0))


TERM_BUILDS = {**BUILDS, "zero-coeff": zero_coeff}


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("build", list(TERM_BUILDS))
def test_terms_agree(build: str, active: bool) -> None:
    expr = TERM_BUILDS[build](Model())
    stores: list[Store] = []
    for cls in STORE_CLASSES:
        store = cls.from_dense(expr.data, expr.model)
        assert isinstance(store, cls)
        stores.append(store)
    n_cells = stores[0].n_cells
    mask = np.arange(n_cells) % 3 != 0 if active else None
    dense, sparse = (s.terms(mask) for s in stores)
    for got, want in zip(sparse, dense):
        np.testing.assert_array_equal(got, want)
    counts, labels, coeffs = dense
    assert counts.shape == (n_cells,)
    assert counts.sum() == len(labels) == len(coeffs)
    assert (labels != -1).all() and (coeffs != 0).all()
    absent = np.isnan(stores[0].const)
    if mask is not None:
        absent |= ~mask
    assert (counts[absent] == 0).all()


@pytest.mark.parametrize("cls", STORE_CLASSES)
def test_terms_counts_match_stored_cells(cls: type[Store]) -> None:
    store = build_store("absent", cls)
    counts, labels, coeffs = store.terms()
    cells = np.repeat(np.arange(store.n_cells), counts)
    for cell, label, coeff in zip(cells, labels, coeffs):
        cell_coeffs, cell_labels, _ = store.cell(
            np.unravel_index(cell, store.grid.shape)
        )
        assert coeff in cell_coeffs[cell_labels == label]


def masked_model(sparse: bool) -> Model:
    """
    Constraints with a masked row, a row dropped for an infinite rhs while
    its lhs keeps live terms, a stored zero coefficient and duplicate terms.
    """
    m = Model(sparse=sparse)
    lhs = zero_coeff(m)
    x, y = m.variables["x"], m.variables["y"]
    lhs = lhs + x
    rhs = OVER_AB.where(OVER_AB != 3, np.inf)
    m.add_constraints(lhs <= rhs, name="c", mask=OVER_AB != 2)
    m.constraints["c"].sanitize_infinities()
    m.add_constraints(x.sum("b") + x.sum("b") + y <= 4, name="dup")
    m.add_objective(x.sum())
    return m


def test_constraint_to_polars_backends_agree() -> None:
    dense, sparse = masked_model(False), masked_model(True)
    for name in dense.constraints:
        got = sparse.constraints[name].to_polars()
        want = dense.constraints[name].to_polars()
        key = ["labels", "vars"]
        assert got.sort(key).equals(want.sort(key))
        assert (want["labels"] != -1).all()
        assert (want["coeffs"] != 0).all()
    con = dense.constraints["c"]
    counts = con.lhs._store.terms()[0]
    masked = con.labels.values.ravel() == -1
    assert counts[masked].sum() > 0, "masked rows keep live lhs terms"
    assert con.to_polars()["labels"].n_unique() == (~masked).sum()


def test_lp_file_backends_agree(tmp_path: Path) -> None:
    paths = [tmp_path / f"{kind}.lp" for kind in ("dense", "sparse")]
    for path, sparse in zip(paths, (False, True)):
        masked_model(sparse).to_file(path, io_api="lp")
    assert paths[0].read_text() == paths[1].read_text()
