"""
``@`` / ``dot`` against a constant backed by a pydata/sparse array.

A labelled :class:`xarray.DataArray` whose data is a ``sparse.COO`` takes the
same sparse contraction as a dense operand without being densified, and gives
the same result.
"""

from __future__ import annotations

import time
import tracemalloc
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import linopy
from linopy import LinearExpression, Model, Variable
from linopy.alignment import is_sparse_backed
from linopy.testing import assert_equivalent, assert_quadequal

sparse = pytest.importorskip("sparse")

A = pd.Index(range(4), name="a")
B = pd.Index(["b0", "b1", "b2"], name="b")
C = pd.Index(range(5), name="c")
E = pd.Index(["e0", "e1"], name="e")


def to_sparse(da: xr.DataArray, fill_value: float = 0.0) -> xr.DataArray:
    return da.copy(data=sparse.COO.from_numpy(da.values, fill_value=fill_value))


def mostly_zero(*indexes: pd.Index, seed: int = 0) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    shape = tuple(len(i) for i in indexes)
    values = rng.integers(-3, 4, size=shape).astype(float)
    values[rng.uniform(size=shape) < 0.6] = 0.0
    return xr.DataArray(values, coords=list(indexes))


@pytest.fixture(params=[False, True], ids=["dense-model", "sparse-model"])
def m(request: pytest.FixtureRequest) -> Model:
    return Model(sparse=request.param)


@pytest.fixture
def x(m: Model) -> Variable:
    return m.add_variables(coords=[A, B], name="x")


def receiver(x: Variable, kind: str) -> Variable | LinearExpression:
    if kind == "variable":
        return x
    return 2.0 * x + 1.0


OPERANDS = {
    "one-dim": lambda: mostly_zero(A, C),
    "two-dims": lambda: mostly_zero(B, A),
    "two-dims-new": lambda: mostly_zero(A, C, B),
    "one-dim-two-new": lambda: mostly_zero(C, A, E),
}


@pytest.mark.parametrize("kind", ["variable", "expression"])
@pytest.mark.parametrize("operand", list(OPERANDS))
def test_sparse_operand_matches_dense(x: Variable, kind: str, operand: str) -> None:
    lhs = receiver(x, kind)
    dense = OPERANDS[operand]()
    with warnings.catch_warnings():
        warnings.simplefilter("error", linopy.PerformanceWarning)
        res = lhs @ to_sparse(dense)
    assert_equivalent(res, lhs @ dense)
    assert_equivalent(lhs.dot(to_sparse(dense)), lhs @ dense)
    assert res.is_sparse == x.model.sparse


def test_sparse_operand_keeps_kept_dim_aux_coords(x: Variable) -> None:
    expr = (1.0 * x).assign_coords(tag=("b", ["p", "q", "r"]))
    dense = mostly_zero(A, C)
    assert_equivalent(expr @ to_sparse(dense), expr @ dense)


def test_sparse_operand_accepts_other_sparse_formats(x: Variable) -> None:
    dense = mostly_zero(A, C)
    gcxs = dense.copy(data=sparse.GCXS.from_numpy(dense.values))
    assert is_sparse_backed(gcxs)
    assert_equivalent(x @ gcxs, x @ dense)


@pytest.mark.parametrize("kind", ["mismatch", "reorder", "aux"])
def test_sparse_operand_label_errors_match_dense(x: Variable, kind: str) -> None:
    expr = 1.0 * x
    index = A
    if kind == "mismatch":
        index = pd.Index([10, 11, 12, 13], name="a")
    elif kind == "reorder":
        index = A[::-1]
    dense = mostly_zero(index, C)
    if kind == "aux":
        dense = dense.assign_coords(tag=("a", ["X"] * len(A)))
        expr = expr.assign_coords(tag=("a", ["Y"] * len(A)))

    with pytest.raises(ValueError) as want:
        expr @ dense
    with pytest.raises(ValueError) as got:
        expr @ to_sparse(dense)
    assert str(got.value) == str(want.value)


@pytest.mark.parametrize("where", ["stored", "fill"])
def test_sparse_operand_nan_matches_dense(x: Variable, where: str) -> None:
    dense = mostly_zero(A, C)
    if where == "stored":
        dense[0, 0] = np.nan
        operand = to_sparse(dense)
    else:
        dense = dense.where(dense != 0)
        operand = to_sparse(dense, fill_value=np.nan)
        assert np.isnan(operand.data.fill_value)

    with pytest.raises(ValueError) as want:
        x @ dense
    with pytest.raises(ValueError) as got:
        x @ operand
    assert str(got.value) == str(want.value)


def test_sparse_operand_nonzero_fill_value_raises(x: Variable) -> None:
    operand = to_sparse(mostly_zero(A, C) + 1.0, fill_value=1.0)
    with pytest.raises(ValueError, match="fill value 0"):
        x @ operand


FALLBACKS: dict[str, Any] = {
    # (receiver from x, operand) pairs the sparse product refuses
    "no-shared-dim": (lambda x: 1.0 * x, lambda: mostly_zero(C, E)),
    "zero-dim": (lambda x: (1.0 * x).sum(), lambda: mostly_zero(C)),
}


@pytest.mark.filterwarnings("ignore:Sparse \\(CSR\\) backing densified")
@pytest.mark.parametrize("case", list(FALLBACKS))
def test_dense_fallback_densifies_operand_with_warning(x: Variable, case: str) -> None:
    make_expr, make_operand = FALLBACKS[case]
    expr, dense = make_expr(x), make_operand()

    with pytest.warns(linopy.PerformanceWarning, match="pydata/sparse"):
        res = expr @ to_sparse(dense)
    assert_equivalent(res, expr @ dense)


@pytest.mark.filterwarnings("ignore:Sparse \\(CSR\\) backing densified")
def test_dense_fallback_warning_can_be_silenced(x: Variable) -> None:
    expr, dense = 1.0 * x, mostly_zero(C, E)
    with linopy.options as opts, warnings.catch_warnings():
        opts.set_value(warn_on_densify=False)
        warnings.simplefilter("error", linopy.PerformanceWarning)
        res = expr @ to_sparse(dense)
    assert_equivalent(res, expr @ dense)


def test_dense_fallback_on_non_unique_labels() -> None:
    m = Model()
    x = m.add_variables(coords=[pd.Index(["p", "p", "q"], name="d")], name="x")
    dense = xr.DataArray(
        np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]]),
        dims=["d", "c"],
        coords={"d": x.indexes["d"], "c": [0, 1]},
    )
    with pytest.warns(linopy.PerformanceWarning, match="non-unique labels"):
        res = x @ to_sparse(dense)
    assert_equivalent(res, x @ dense)


def test_quadratic_matmul_densifies_operand_with_warning(x: Variable) -> None:
    quad = x * x
    dense = mostly_zero(A, C)
    with pytest.warns(linopy.PerformanceWarning, match="quadratic expression"):
        res = quad @ to_sparse(dense)
    assert_quadequal(res, quad @ dense)


def test_is_sparse_backed() -> None:
    dense = mostly_zero(A, C)
    assert not is_sparse_backed(dense)
    assert not is_sparse_backed(dense.values)
    assert is_sparse_backed(to_sparse(dense))


def test_large_mostly_zero_operand_is_never_densified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 5000 x 3000 incidence with 5000 non-zeros contracts in its non-zeros."""
    n_link, n_bus, nnz = 5000, 3000, 5000
    links = pd.Index(range(n_link), name="link")
    buses = pd.Index(range(n_bus), name="bus")
    rng = np.random.default_rng(0)
    coo = sparse.COO(
        np.stack([np.arange(nnz), rng.integers(0, n_bus, nnz)]),
        rng.choice([-1.0, 1.0], nnz),
        shape=(n_link, n_bus),
    )
    incidence = xr.DataArray(coo, coords=[links, buses])
    dense_bytes = n_link * n_bus * 8

    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the sparse operand was densified")

    monkeypatch.setattr(sparse.COO, "todense", refuse)
    m = Model()
    p = m.add_variables(coords=[links], name="p")

    tracemalloc.start()
    start = time.perf_counter()
    try:
        res = p @ incidence
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert res.coord_dims == ("bus",)
    assert peak < dense_bytes / 10
    assert elapsed < 10
    coeffs = res.coeffs.values
    assert np.nansum(np.abs(coeffs)) == nnz
