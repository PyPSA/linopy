"""
``@`` / ``dot`` against a constant backed by a pydata/sparse array.

A labelled :class:`xarray.DataArray` whose data is a ``sparse.COO`` takes the
same sparse contraction as a dense operand without being densified, and gives
the same result.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import linopy
from linopy import LinearExpression, Model, QuadraticExpression, Variable
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
    "one-dim": lambda: to_sparse(mostly_zero(A, C)),
    "two-dims": lambda: to_sparse(mostly_zero(B, A)),
    "two-dims-new": lambda: to_sparse(mostly_zero(A, C, B)),
    "one-dim-two-new": lambda: to_sparse(mostly_zero(C, A, E)),
    "empty": lambda: xr.DataArray(sparse.zeros((len(A), len(C))), coords=[A, C]),
    "explicit-zeros": lambda: xr.DataArray(
        sparse.COO([[0, 2], [1, 3]], [0.0, 2.0], shape=(len(A), len(C))),
        coords=[A, C],
    ),
}


def refuse(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("the sparse operand was densified")


@pytest.mark.parametrize("kind", ["variable", "expression"])
@pytest.mark.parametrize("operand", list(OPERANDS))
def test_sparse_operand_matches_dense(
    x: Variable, kind: str, operand: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    lhs = receiver(x, kind)
    op = OPERANDS[operand]()
    want = lhs @ op.copy(data=op.data.todense())
    monkeypatch.setattr(sparse.COO, "todense", refuse)
    with warnings.catch_warnings():
        warnings.simplefilter("error", linopy.PerformanceWarning)
        res = lhs @ op
    dotted = lhs.dot(op)
    assert isinstance(res, LinearExpression)
    assert isinstance(dotted, LinearExpression)
    assert_equivalent(res, want)
    assert_equivalent(dotted, want)
    assert res.is_sparse == x.model.sparse


def test_sparse_operand_keeps_kept_dim_aux_coords(x: Variable) -> None:
    expr = (1.0 * x).assign_coords(tag=("b", ["p", "q", "r"]))
    dense = mostly_zero(A, C)
    assert_equivalent(expr @ to_sparse(dense), expr @ dense)


def test_sparse_operand_accepts_other_sparse_formats(x: Variable) -> None:
    dense = mostly_zero(A, C)
    gcxs = dense.copy(data=sparse.GCXS.from_numpy(dense.values))
    assert is_sparse_backed(gcxs)
    assert not is_sparse_backed(dense) and not is_sparse_backed(dense.values)
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


def non_unique(x: Variable) -> tuple[Any, xr.DataArray, str]:
    y = x.model.add_variables(coords=[pd.Index(["p", "p", "q"], name="d")])
    dense = xr.DataArray(
        np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]]),
        dims=["d", "c"],
        coords={"d": y.indexes["d"], "c": [0, 1]},
    )
    return y, dense, "non-unique labels"


FALLBACKS: dict[str, Any] = {
    # x -> (receiver, dense operand, warning match) the sparse product refuses
    "no-shared-dim": lambda x: (1.0 * x, mostly_zero(C, E), "pydata/sparse"),
    "zero-dim": lambda x: ((1.0 * x).sum(), mostly_zero(C), "pydata/sparse"),
    "non-unique": non_unique,
    "quadratic": lambda x: (x * x, mostly_zero(A, C), "quadratic expression"),
}


@pytest.mark.filterwarnings("ignore:Sparse \\(CSR\\) backing densified")
@pytest.mark.parametrize("case", list(FALLBACKS))
def test_dense_fallback_densifies_operand_with_notice(x: Variable, case: str) -> None:
    expr, dense, match = FALLBACKS[case](x)
    with linopy.options as opts, pytest.warns(linopy.PerformanceWarning, match=match):
        opts.set_value(warn_on_densify=True)
        res = expr @ to_sparse(dense)
    quad = isinstance(res, QuadraticExpression)
    (assert_quadequal if quad else assert_equivalent)(res, expr @ dense)
