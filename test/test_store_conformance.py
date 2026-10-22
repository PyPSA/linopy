"""
Conformance of the dense and the sparse (CSR) backend: every operation either
keeps the sparse backing or names why it falls back to dense, and both
backends agree on the result.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import linopy
from linopy import LinearExpression, Model, QuadraticExpression, Variable
from linopy.constants import TERM_DIM
from linopy.constraints import ConstraintBase, CSRConstraint
from linopy.testing import assert_equivalent

KEPT = "kept"
ELLIPSIS: Any = ...
DATA_READ = "`.data` read, e.g. by an operation without a sparse path"
WHERE = (
    "`where` with a condition that is no DataArray, an `other` that is no "
    "scalar, or extra arguments"
)
NEW_DIMS = "elementwise operation with a constant over new dimensions"

A = pd.Index([0, 1, 2], name="a")
B = pd.Index(["p", "q"], name="b")
C = pd.Index([10, 20], name="c")
TAG = ("a", ["u", "v", "u"])
COND = xr.DataArray([True, False, True], coords=[A])
OVER_A = xr.DataArray([1.0, 2.0, 3.0], coords=[A])
OVER_AB = xr.DataArray(np.arange(6.0).reshape(3, 2), coords=[A, B])
OVER_B = xr.DataArray([1.0, 2.0], coords=[B])
OVER_BC = xr.DataArray(np.arange(4.0).reshape(2, 2), coords=[B, C])
OVER_C = xr.DataArray([1.0, 2.0], coords=[C])
SHIFTED_A = xr.DataArray([1.0, 2.0], coords=[pd.Index([1, 5], name="a")])
SERIES = pd.Series([0, 1, 0], index=A, name="g")
FRAME = pd.DataFrame({"k1": [0, 1, 0], "k2": [1, 1, 2]}, index=A)


def var(e: LinearExpression, name: str) -> Variable:
    return e.model.variables[name]


def grid(m: Model) -> LinearExpression:
    x = m.add_variables(coords=[A, B], name="x").assign_coords(tag=TAG)
    y = m.add_variables(coords=[A], name="y").assign_coords(tag=TAG)
    return 2 * x + y


def single(m: Model) -> LinearExpression:
    return 1 * m.add_variables(coords=[A], name="y")


def non_unique(m: Model) -> LinearExpression:
    a = pd.Index([0, 0, 1], name="a")
    m.add_variables(coords=[a, B], name="w")
    return 1 * m.add_variables(coords=[a], name="n")


def zero_size(m: Model) -> LinearExpression:
    return 1 * m.add_variables(coords=[pd.Index([], dtype=int, name="a")], name="n")


def frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(list(df.columns), ignore_index=True)


def const_assigned(
    value: xr.DataArray,
    then: Callable[[LinearExpression], LinearExpression] = lambda e: e,
) -> Callable[[LinearExpression], Any]:
    def op(e: LinearExpression) -> LinearExpression:
        e.const = value
        return then(e).fillna(0.0)

    return op


class Case(NamedTuple):
    build: Callable[[Model], LinearExpression]
    op: Callable[[LinearExpression], Any]
    expected: str
    sparse_xfail: str | None = None


NAN_A = OVER_A.where(OVER_A != 2.0)
AFTER_NAN_CONST: dict[
    str, tuple[Callable[[LinearExpression], LinearExpression], str]
] = {
    "isel": (lambda e: e.isel(a=[0, 1, 2]), KEPT),
    "roll": (lambda e: e.roll(a=1), KEPT),
    "mul": (lambda e: e * 2, KEPT),
    "reindex": (lambda e: e.reindex(b=["q", "p"]), KEPT),
    "add-absent": (
        lambda e: e.add(SHIFTED_A, join="outer", fill_value=linopy.ABSENT),
        KEPT,
    ),
    "data": (lambda e: LinearExpression(e.data, e.model), DATA_READ),
}

SIGN_EQ = "CSRConstraint normalizes the sign '==' to '=' while Constraint keeps it"

CASES: dict[str, Case] = {
    "sel": Case(grid, lambda e: e.sel(a=[0, 2]), KEPT),
    "isel": Case(grid, lambda e: e.isel(a=slice(0, 2)), KEPT),
    "isel-scalar": Case(grid, lambda e: e.isel(b=0), KEPT),
    "getitem": Case(grid, lambda e: e[0:2], KEPT),
    "roll": Case(grid, lambda e: e.roll(a=1), KEPT),
    "sel-pointwise": Case(
        grid,
        lambda e: e.sel(a=xr.DataArray([0, 1], dims="c")),
        "`sel` introducing dimensions or coordinates",
    ),
    "isel-pointwise": Case(
        grid,
        lambda e: e.isel(a=xr.DataArray([0, 1], dims="c")),
        "`isel` introducing dimensions or coordinates",
    ),
    "shift": Case(grid, lambda e: e.shift(a=1), KEPT),
    "shift-fill-value": Case(
        grid, lambda e: e.shift(a=1, fill_value=0), "`shift` with a fill_value"
    ),
    "where": Case(grid, lambda e: e.where(COND), KEPT),
    "where-scalar-other": Case(grid, lambda e: e.where(COND, 1.0), KEPT),
    "where-drop": Case(grid, lambda e: e.where(COND, drop=True), KEPT),
    "where-callable": Case(grid, lambda e: e.where(lambda _: COND), WHERE),
    "where-array-other": Case(grid, lambda e: e.where(COND, OVER_AB), WHERE),
    "where-drop-scalar-other": Case(
        grid, lambda e: e.where(COND, 1.0, drop=True), WHERE
    ),
    "sum-dim": Case(grid, lambda e: e.sum("a"), KEPT),
    "sum-dims": Case(grid, lambda e: e.sum(["b", "a"]), KEPT),
    "sum-all": Case(grid, lambda e: e.sum(), KEPT),
    "sum-ellipsis": Case(grid, lambda e: e.sum(ELLIPSIS), KEPT),
    "sum-term": Case(grid, lambda e: e.sum(TERM_DIM), KEPT),
    "sum-drop-zeros": Case(grid, lambda e: (0 * e).sum("b", drop_zeros=True), KEPT),
    "fillna-scalar": Case(grid, lambda e: e.where(COND).fillna(1.0), KEPT),
    "fillna-array": Case(
        grid, lambda e: e.where(COND).fillna(OVER_AB), "fillna with a non-scalar value"
    ),
    "fillna-expression": Case(
        grid, lambda e: e.where(COND).fillna(3 * e), "fillna with a non-scalar value"
    ),
    "reindex": Case(grid, lambda e: e.reindex(a=[0, 1, 5]), KEPT),
    "reindex-method": Case(
        grid, lambda e: e.reindex(a=[0, 1, 3], method="nearest"), DATA_READ
    ),
    "reindex-tolerance": Case(
        grid, lambda e: e.reindex(a=[0, 1, 3], method="nearest", tolerance=1), DATA_READ
    ),
    "reindex-fill-value": Case(
        grid,
        lambda e: e.reindex(a=[0, 5], fill_value={"vars": -1, "coeffs": 0, "const": 0}),
        DATA_READ,
    ),
    "reindex-copy": Case(grid, lambda e: e.reindex(a=[0, 1, 5], copy=False), DATA_READ),
    "rename-dim": Case(grid, lambda e: e.rename(a="d"), KEPT),
    "rename-aux": Case(grid, lambda e: e.rename(tag="label"), DATA_READ),
    "rename-const": Case(grid, lambda e: e.rename(const="k"), DATA_READ),
    "drop-vars-aux": Case(grid, lambda e: e.drop_vars("tag"), KEPT),
    "drop-vars-dim": Case(grid, lambda e: e.drop_vars("b"), DATA_READ),
    "drop-vars-const": Case(grid, lambda e: e.drop_vars("const"), DATA_READ),
    "add-scalar": Case(grid, lambda e: e + 1.5, KEPT),
    "rsub-scalar": Case(grid, lambda e: 3 - e, KEPT),
    "div-scalar": Case(grid, lambda e: e / 2, KEPT),
    "add-array": Case(grid, lambda e: e + OVER_A, KEPT),
    "mul-array": Case(grid, lambda e: e * OVER_AB, KEPT),
    "div-array": Case(grid, lambda e: e / OVER_B, KEPT),
    "sub-join-outer": Case(grid, lambda e: e.sub(SHIFTED_A, join="outer"), KEPT),
    "mul-join-inner": Case(grid, lambda e: e.mul(SHIFTED_A, join="inner"), KEPT),
    "add-join-left": Case(grid, lambda e: e.add(SHIFTED_A, join="left"), KEPT),
    "add-array-new-dim": Case(grid, lambda e: e + OVER_C, NEW_DIMS),
    "mul-array-new-dim": Case(grid, lambda e: e * OVER_C, NEW_DIMS),
    "neg": Case(grid, lambda e: -e, KEPT),
    "mul-scalar": Case(grid, lambda e: e * 3, KEPT),
    "matmul-shared-dim": Case(grid, lambda e: e @ OVER_B, KEPT),
    "matmul-new-dim": Case(grid, lambda e: e @ OVER_BC, KEPT),
    "matmul-zero-dim": Case(
        grid, lambda e: e.sum() @ 2, "`@` on a zero-dimensional expression"
    ),
    "matmul-non-unique": Case(
        non_unique,
        lambda e: e @ xr.DataArray([1.0, 2.0, 3.0], coords=[e.indexes["a"]]),
        "`@` over non-unique labels",
    ),
    "matmul-zero-size": Case(
        zero_size,
        lambda e: e @ xr.DataArray([], coords=[e.indexes["a"]]),
        "`@` over a zero-size grid",
    ),
    "matmul-no-shared-dim": Case(
        grid,
        lambda e: e @ OVER_C,
        "`@` with an operand sharing no dimension or with an unlabelled one",
    ),
    "add-expr-same-grid": Case(grid, lambda e: e + 2 * e, KEPT),
    "add-expr-cross-grid": Case(grid, lambda e: e + 1 * var(e, "y"), KEPT),
    "add-expr-join-outer": Case(
        grid,
        lambda e: e.add(1 * var(e, "y").sel(a=[1, 2]), join="outer"),
        KEPT,
    ),
    "add-dense-operand": Case(
        grid, lambda e: e + var(e, "y").to_linexpr(_dense=True), KEPT
    ),
    "merge-along-grid-dim": Case(
        grid, lambda e: linopy.merge([e.sel(a=[0]), e.sel(a=[1, 2])], dim="a"), KEPT
    ),
    "merge-kwargs": Case(
        grid,
        lambda e: linopy.merge([e, e], combine_attrs="drop"),
        "merge with extra arguments",
    ),
    "merge-dataset-operand": Case(
        grid,
        lambda e: linopy.merge([e, var(e, "y").to_linexpr(_dense=True).data]),
        "merge with an operand that is not a LinearExpression",
    ),
    "merge-spread-along-dim": Case(
        grid,
        lambda e: linopy.merge([e.drop_vars("tag"), 1 * var(e, "y")], dim="a"),
        "merge of operands over different dimensions",
    ),
    "merge-new-dim": Case(
        grid, lambda e: linopy.merge([e, e], dim="c"), "merge along a new dimension"
    ),
    "merge-non-unique": Case(
        non_unique, lambda e: e + 1 * var(e, "w"), "merge over non-unique labels"
    ),
    "to-constraint-scalar": Case(grid, lambda e: e.to_constraint("<=", 1), KEPT),
    "to-constraint-array": Case(grid, lambda e: e.to_constraint(">=", OVER_A), KEPT),
    "to-constraint-expression": Case(
        grid, lambda e: e.to_constraint("<=", 1 * var(e, "y")), KEPT
    ),
    "to-constraint-expression-eq": Case(
        grid, lambda e: e.to_constraint("==", 1 * var(e, "y")), KEPT, SIGN_EQ
    ),
    "to-constraint-rhs-off-grid": Case(
        grid,
        lambda e: e.to_constraint("<=", OVER_C),
        "constraint with an rhs over dimensions outside the grid",
    ),
    "groupby-series": Case(grid, lambda e: e.groupby(SERIES).sum(), KEPT),
    "groupby-dataframe": Case(grid, lambda e: e.groupby(FRAME).sum(), KEPT),
    "groupby-dataframe-observed": Case(
        grid, lambda e: e.groupby(FRAME).sum(observed=True), KEPT
    ),
    "groupby-dataarray": Case(
        grid, lambda e: e.groupby(xr.DataArray(SERIES)).sum(), KEPT
    ),
    "groupby-use-fallback": Case(
        grid,
        lambda e: e.groupby(SERIES).sum(use_fallback=True),
        "groupby-sum with a grouper without a sparse path",
    ),
    "groupby-2d-grouper": Case(
        grid,
        lambda e: e.groupby(xr.DataArray(OVER_AB % 2, name="g")).sum(),
        "groupby-sum with a grouper without a sparse path",
    ),
    "coeffs": Case(grid, lambda e: e.coeffs.sum(TERM_DIM), DATA_READ),
    "vars": Case(grid, lambda e: e.vars.max(TERM_DIM), DATA_READ),
    "to-quadexpr": Case(grid, lambda e: e.to_quadexpr(), DATA_READ),
    "simplify": Case(grid, lambda e: e.simplify(), DATA_READ),
    "densify-terms": Case(grid, lambda e: e.densify_terms(), DATA_READ),
    "loc": Case(grid, lambda e: e.loc[[0, 1]], KEPT),
    "cumsum": Case(grid, lambda e: e.cumsum("a"), DATA_READ),
    "rolling": Case(grid, lambda e: e.rolling(a=2).sum(), DATA_READ),
    "diff": Case(grid, lambda e: e.diff("a"), KEPT),
    "mul-variable": Case(grid, lambda e: e * var(e, "y"), DATA_READ),
    "pow": Case(single, lambda e: e**2, DATA_READ),
    "to-polars": Case(
        grid, lambda e: frame(pd.DataFrame(e.to_polars().to_dict())), KEPT
    ),
    "flat": Case(grid, lambda e: frame(e.flat), KEPT),
    "linear-terms": Case(
        grid, lambda e: frame(pd.DataFrame(dict(zip("vc", e.linear_terms())))), KEPT
    ),
    "has-terms": Case(grid, lambda e: e.where(COND).has_terms, KEPT),
    "isnull": Case(grid, lambda e: e.where(COND).isnull(), KEPT),
    "const": Case(grid, lambda e: (e + OVER_A).const, KEPT),
    "const-assign-nan": Case(grid, const_assigned(NAN_A), KEPT),
    "const-assign-partial": Case(grid, const_assigned(OVER_A.sel(a=[0, 1])), KEPT),
    "const-assign-new-dim": Case(grid, const_assigned(OVER_C), "assignment of `const`"),
    **{
        f"const-assign-nan-{name}": Case(grid, const_assigned(NAN_A, then), expected)
        for name, (then, expected) in AFTER_NAN_CONST.items()
    },
}


@contextmanager
def no_densify() -> Iterator[None]:
    """Fail on any densify notice inside the block."""
    with linopy.options as opts, warnings.catch_warnings():
        warnings.simplefilter("error", linopy.PerformanceWarning)
        opts.set_value(warn_on_densify=True)
        yield


def assert_same(res: Any, want: Any) -> None:
    if isinstance(res, QuadraticExpression):
        pd.testing.assert_frame_equal(res.flat, want.flat)
    elif isinstance(res, LinearExpression):
        assert_equivalent(res, want)
    elif isinstance(res, ConstraintBase):
        assert_equivalent(res.lhs, want.lhs)
        xr.testing.assert_equal(res.sign, want.sign)
        xr.testing.assert_allclose(res.rhs, want.rhs)
    elif isinstance(res, pd.DataFrame):
        pd.testing.assert_frame_equal(res, want)
    else:
        xr.testing.assert_allclose(res, want)


@pytest.mark.parametrize("case", list(CASES))
def test_operation(backend: Model, case: str, request: pytest.FixtureRequest) -> None:
    build, op, expected, sparse_xfail = CASES[case]
    if backend.sparse and sparse_xfail is not None:
        request.applymarker(
            pytest.mark.xfail(strict=True, raises=AssertionError, reason=sparse_xfail)
        )
    want = op(build(Model()))
    expr = build(backend)
    assert expr.is_sparse == backend.sparse
    if not backend.sparse or expected == KEPT:
        with no_densify():
            res = op(expr)
    else:
        with pytest.warns(linopy.PerformanceWarning, match=re.escape(expected)) as rec:
            res = op(expr)
        notices = [w for w in rec if issubclass(w.category, linopy.PerformanceWarning)]
        assert len(notices) == 1
    assert_same(res, want)
    if backend.sparse and expected == KEPT:
        assert expr.is_sparse
        if isinstance(res, LinearExpression):
            assert res.is_sparse
        elif isinstance(res, ConstraintBase):
            assert isinstance(res, CSRConstraint)


DIFFERENCES: dict[str, Callable[[LinearExpression], LinearExpression]] = {
    "coeffs": lambda e: e * 2,
    "const": lambda e: e + 1,
    "null-mask": lambda e: e.where(COND),
    "aux-coord": lambda e: e.drop_vars("tag"),
}


@pytest.mark.parametrize("diff", list(DIFFERENCES))
def test_assert_equivalent_detects_difference(backend: Model, diff: str) -> None:
    expr = grid(backend)
    with no_densify():
        other = DIFFERENCES[diff](expr)
        assert expr.is_sparse == other.is_sparse == backend.sparse
        with pytest.raises(AssertionError):
            assert_equivalent(expr, other)
