"""
Conformance of the dense and the sparse (CSR) backend: every operation either
keeps the sparse backing or names why it falls back to dense, and both
backends agree on the result.
"""

from __future__ import annotations

import inspect
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
from linopy.constraints import Constraint
from linopy.csr import CSRStore
from linopy.store import DenseStore
from linopy.testing import assert_equivalent, assert_quadequal

KEPT = "kept"
ELLIPSIS: Any = ...
DATA_READ = "data(read)"
QUADRATIC = "mul(quadratic product)"
RENAME = "rename(non-dimension names)"
DROP_VARS = "drop_vars(dimension or data variable)"
ASSIGNMENT = "assignment(const)"
ASSIGN_COEFFS = "assignment(coeffs)"
SPARSIFIED = "sparsified: merge with a sparse operand"

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
COEFFS = OVER_AB.expand_dims({TERM_DIM: 2}, axis=-1)
SERIES = pd.Series([0, 1, 0], index=A, name="g")
FRAME = pd.DataFrame({"k1": [0, 1, 0], "k2": [1, 1, 2]}, index=A)


def var(e: LinearExpression, name: str) -> Variable:
    return e.model.variables[name]


def dense_linexpr(v: Variable) -> LinearExpression:
    return v._linexpr(DenseStore)


def dense_merged(e: LinearExpression) -> pd.DataFrame:
    res = dense_linexpr(var(e, "x")) + dense_linexpr(var(e, "y"))
    assert not res.is_sparse
    return frame(res.flat)


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


def without_tag(e: LinearExpression) -> LinearExpression:
    return e.drop_vars("tag")


def with_2d_coord(e: LinearExpression) -> LinearExpression:
    return e.assign_coords(two=(("a", "b"), np.arange(6).reshape(3, 2)))


def coords_reassigned(e: LinearExpression) -> LinearExpression:
    named = e.model.add_expressions(e, name="named")
    e.model.assign_coords(a=[7, 8, 9])
    assert isinstance(named, LinearExpression)
    return named


def frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(list(df.columns), ignore_index=True)


def const_assigned(value: xr.DataArray) -> Callable[[LinearExpression], Any]:
    def op(e: LinearExpression) -> LinearExpression:
        e.const = value
        return e.fillna(0.0)

    return op


def coeffs_assigned(e: LinearExpression) -> LinearExpression:
    e.coeffs = COEFFS
    return e


def merged_along_a(e: LinearExpression) -> LinearExpression:
    other = 1 * var(e, "y")
    res = linopy.merge([e.drop_vars("tag"), other], dim="a")
    assert other.is_sparse == e.is_sparse
    assert isinstance(res, LinearExpression)
    return res


def override_merge_raises(e: LinearExpression) -> LinearExpression:
    with pytest.raises(xr.AlignmentError):
        linopy.merge([e, e.sel(a=[0, 1])], join="override")
    return e


class Case(NamedTuple):
    build: Callable[[Model], LinearExpression]
    op: Callable[[LinearExpression], Any]
    expected: str


CASES: dict[str, Case] = {
    "sel": Case(grid, lambda e: e.sel(a=[0, 2]), KEPT),
    "isel": Case(grid, lambda e: e.isel(a=slice(0, 2)), KEPT),
    "isel-scalar": Case(grid, lambda e: e.isel(b=0), KEPT),
    "getitem": Case(grid, lambda e: e[0:2], KEPT),
    "roll": Case(grid, lambda e: e.roll(a=1), KEPT),
    "roll-coords": Case(grid, lambda e: e.roll(a=1, roll_coords=True), KEPT),
    "sel-scalar": Case(grid, lambda e: e.sel(a=1), KEPT),
    "sel-2d-aux-densifies": Case(
        grid,
        lambda e: with_2d_coord(e).sel(a=[0, 2]),
        DATA_READ,
    ),
    "sel-pointwise": Case(
        grid,
        lambda e: e.sel(a=xr.DataArray([0, 1], dims="c")),
        "sel(new dimensions or coordinates)",
    ),
    "isel-pointwise": Case(
        grid,
        lambda e: e.isel(a=xr.DataArray([0, 1], dims="c")),
        "isel(new dimensions or coordinates)",
    ),
    "getitem-new-dim": Case(
        grid,
        lambda e: e[xr.DataArray([0, 1], coords=[C])],
        "getitem(new dimensions or coordinates)",
    ),
    "shift": Case(grid, lambda e: e.shift(a=1), KEPT),
    "shift-negative": Case(grid, lambda e: e.shift(a=-1), KEPT),
    "shift-fill-value": Case(
        grid, lambda e: e.shift(a=1, fill_value=0), "shift(fill_value)"
    ),
    "where": Case(grid, lambda e: e.where(COND), KEPT),
    "where-scalar-other": Case(grid, lambda e: e.where(COND, 1.0), KEPT),
    "where-drop": Case(grid, lambda e: e.where(COND, drop=True), KEPT),
    "where-callable": Case(
        grid, lambda e: e.where(lambda _: COND), "where(condition no DataArray)"
    ),
    "where-array-other": Case(
        grid, lambda e: e.where(COND, OVER_AB), "where(non-scalar other)"
    ),
    "where-expression-other": Case(
        grid, lambda e: e.where(COND, 3 * e), "where(non-scalar other)"
    ),
    "where-new-dim": Case(
        grid,
        lambda e: e.where(xr.DataArray([True, False], coords=[C])),
        "where(new dimensions or coordinates)",
    ),
    "where-drop-scalar-other": Case(
        grid, lambda e: e.where(COND, 1.0, drop=True), "where(other with drop)"
    ),
    "sum-dim": Case(grid, lambda e: e.sum("a"), KEPT),
    "sum-dims": Case(grid, lambda e: e.sum(["b", "a"]), KEPT),
    "sum-all": Case(grid, lambda e: e.sum(), KEPT),
    "sum-ellipsis": Case(grid, lambda e: e.sum(ELLIPSIS), KEPT),
    "sum-term": Case(grid, lambda e: e.sum(TERM_DIM), KEPT),
    "sum-drop-zeros": Case(grid, lambda e: (0 * e).sum("b", drop_zeros=True), KEPT),
    "fillna-scalar": Case(grid, lambda e: e.where(COND).fillna(1.0), KEPT),
    "fillna-array": Case(
        grid, lambda e: e.where(COND).fillna(OVER_AB), "fillna(non-scalar value)"
    ),
    "fillna-expression": Case(
        grid, lambda e: e.where(COND).fillna(3 * e), "fillna(non-scalar value)"
    ),
    "reindex": Case(grid, lambda e: e.reindex(a=[0, 1, 5]), KEPT),
    "reindex-method": Case(
        grid, lambda e: e.reindex(a=[0, 1, 3], method="nearest"), "reindex(method=...)"
    ),
    "reindex-tolerance": Case(
        grid,
        lambda e: e.reindex(a=[0, 1, 3], method="nearest", tolerance=1),
        "reindex(method=...)",
    ),
    "reindex-fill-value": Case(
        grid,
        lambda e: e.reindex(a=[0, 5], fill_value={"vars": -1, "coeffs": 0, "const": 0}),
        "reindex(fill_value=...)",
    ),
    "reindex-copy": Case(
        grid, lambda e: e.reindex(a=[0, 1, 5], copy=False), "reindex(copy=False)"
    ),
    "reindex-off-grid": Case(
        grid, lambda e: e.reindex(c=C), "reindex(indexers off the grid)"
    ),
    "rename-dim": Case(grid, lambda e: e.rename(a="d"), KEPT),
    "rename-aux": Case(grid, lambda e: e.rename(tag="label"), RENAME),
    "rename-const": Case(grid, lambda e: e.rename(const="k"), RENAME),
    "drop-vars-aux": Case(grid, lambda e: e.drop_vars("tag"), KEPT),
    "drop-vars-dim": Case(grid, lambda e: e.drop_vars("b"), DROP_VARS),
    "drop-vars-callable": Case(
        grid, lambda e: e.drop_vars(lambda ds: "tag"), "drop_vars(callable)"
    ),
    "drop-vars-const": Case(grid, lambda e: e.drop_vars("const"), DROP_VARS),
    "add-scalar": Case(grid, lambda e: e + 1.5, KEPT),
    "rsub-scalar": Case(grid, lambda e: 3 - e, KEPT),
    "div-scalar": Case(grid, lambda e: e / 2, KEPT),
    "add-array": Case(grid, lambda e: e + OVER_A, KEPT),
    "mul-array": Case(grid, lambda e: e * OVER_AB, KEPT),
    "div-array": Case(grid, lambda e: e / OVER_B, KEPT),
    "sub-join-outer": Case(grid, lambda e: e.sub(SHIFTED_A, join="outer"), KEPT),
    "mul-join-inner": Case(grid, lambda e: e.mul(SHIFTED_A, join="inner"), KEPT),
    "add-join-left": Case(grid, lambda e: e.add(SHIFTED_A, join="left"), KEPT),
    "add-array-new-dim": Case(
        grid, lambda e: e + OVER_C, "add(constant over new dimensions)"
    ),
    "mul-array-new-dim": Case(
        grid, lambda e: e * OVER_C, "mul(constant over new dimensions)"
    ),
    "div-array-new-dim": Case(
        grid, lambda e: e / OVER_C, "truediv(constant over new dimensions)"
    ),
    "neg": Case(grid, lambda e: -e, KEPT),
    "mul-scalar": Case(grid, lambda e: e * 3, KEPT),
    "mul-constant-expression": Case(
        grid, lambda e: e * LinearExpression(OVER_A, e.model), KEPT
    ),
    "rmul-constant-expression": Case(
        single, lambda e: LinearExpression(OVER_A, e.model) * e, KEPT
    ),
    "matmul-shared-dim": Case(grid, lambda e: e @ OVER_B, KEPT),
    "matmul-new-dim": Case(grid, lambda e: e @ OVER_BC, KEPT),
    "matmul-zero-dim": Case(
        grid, lambda e: e.sum() @ 2, "matmul(zero-dimensional expression)"
    ),
    "matmul-non-unique": Case(
        non_unique,
        lambda e: e @ xr.DataArray([1.0, 2.0, 3.0], coords=[e.indexes["a"]]),
        "matmul(non-unique labels)",
    ),
    "matmul-zero-size": Case(
        zero_size,
        lambda e: e @ xr.DataArray([], coords=[e.indexes["a"]]),
        "matmul(zero-size grid)",
    ),
    "matmul-no-shared-dim": Case(
        grid,
        lambda e: e @ OVER_C,
        "matmul(operand sharing no dimension or with an unlabelled one)",
    ),
    "add-expr-same-grid": Case(grid, lambda e: e + 2 * e, KEPT),
    "add-expr-cross-grid": Case(grid, lambda e: e + 1 * var(e, "y"), KEPT),
    "add-expr-join-outer": Case(
        grid,
        lambda e: e.add(1 * var(e, "y").sel(a=[1, 2]), join="outer"),
        KEPT,
    ),
    "add-dense-operand": Case(
        grid, lambda e: e + dense_linexpr(var(e, "y")), SPARSIFIED
    ),
    "add-dense-operands": Case(grid, dense_merged, KEPT),
    "merge-along-grid-dim": Case(
        grid, lambda e: linopy.merge([e.sel(a=[0]), e.sel(a=[1, 2])], dim="a"), KEPT
    ),
    "merge-kwargs": Case(
        grid,
        lambda e: linopy.merge([e, e], combine_attrs="drop"),
        "merge(extra arguments)",
    ),
    "merge-dataset-operand": Case(
        grid,
        lambda e: linopy.merge([e, dense_linexpr(var(e, "y")).data]),
        "merge(operand not a LinearExpression)",
    ),
    "merge-spread-along-dim": Case(
        grid,
        merged_along_a,
        "merge(operands over different dimensions)",
    ),
    "merge-new-dim": Case(
        grid, lambda e: linopy.merge([e, e], dim="c"), "merge(new dimension)"
    ),
    "merge-non-unique": Case(
        non_unique, lambda e: e + 1 * var(e, "w"), "merge(non-unique labels)"
    ),
    "merge-non-unique-dense-operand": Case(
        non_unique,
        lambda e: e + dense_linexpr(var(e, "w")),
        "merge(non-unique labels)",
    ),
    "merge-dense-operands-around-sparse": Case(
        grid,
        lambda e: linopy.merge(
            [dense_linexpr(var(e, "y")), e, dense_linexpr(var(e, "x"))]
        ),
        SPARSIFIED,
    ),
    "merge-differing-grids": Case(
        lambda m: without_tag(grid(m)),
        lambda e: linopy.merge([e.sum("a"), e.sel(a=[0, 1]), e.sel(a=[1, 2])]),
        "merge(differing grids with join=None)",
    ),
    "merge-override-differing-shapes": Case(
        lambda m: without_tag(grid(m)),
        override_merge_raises,
        'merge(join="override" with differently shaped grids)',
    ),
    "merge-dense-2d-coord": Case(
        grid,
        lambda e: e + with_2d_coord(dense_linexpr(var(e, "x"))),
        "merge(coordinates the Grid cannot hold: two)",
    ),
    "merge-factor-dim": Case(
        grid,
        lambda e: linopy.merge([e, e], dim="_factor", cls=QuadraticExpression),
        "merge(into a QuadraticExpression)",
    ),
    "to-constraint-scalar": Case(grid, lambda e: e.to_constraint("<=", 1), KEPT),
    "to-constraint-array": Case(grid, lambda e: e.to_constraint(">=", OVER_A), KEPT),
    "to-constraint-expression": Case(
        grid, lambda e: e.to_constraint("<=", 1 * var(e, "y")), KEPT
    ),
    "to-constraint-expression-eq": Case(
        grid, lambda e: e.to_constraint("==", 1 * var(e, "y")), KEPT
    ),
    "to-constraint-update-coeffs": Case(
        grid,
        lambda e: e.to_constraint("<=", 1).update(coeffs=2.0),
        "update(coeffs=, variables=)",
    ),
    "to-constraint-update-lhs-off-grid": Case(
        grid,
        lambda e: e.to_constraint("<=", 1).update(
            lhs=with_2d_coord(dense_linexpr(var(e, "x")))
        ),
        "update(lhs=)",
    ),
    "constraint-data": Case(grid, lambda e: e.to_constraint("<=", 1).data, DATA_READ),
    "constraint-sel": Case(
        grid, lambda e: e.to_constraint("<=", 1).sel(a=[0, 1]), DATA_READ
    ),
    "constraint-getitem": Case(
        grid, lambda e: e.to_constraint("<=", 1)[{"a": [0, 1]}], "__getitem__"
    ),
    "to-constraint-array-sign": Case(
        grid,
        lambda e: e.to_constraint(xr.DataArray(["<=", ">=", "<="], coords=[A]), 1),
        "to_constraint(non-scalar sign)",
    ),
    "to-constraint-rhs-off-grid": Case(
        grid,
        lambda e: e.to_constraint("<=", OVER_C),
        "to_constraint(rhs over dimensions outside the grid)",
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
        "groupby.sum(grouper without a sparse path)",
    ),
    "groupby-2d-grouper": Case(
        grid,
        lambda e: e.groupby(xr.DataArray(OVER_AB % 2, name="g")).sum(),
        "groupby.sum(grouper without a sparse path)",
    ),
    "coeffs": Case(grid, lambda e: e.coeffs.sum(TERM_DIM), DATA_READ),
    "vars": Case(grid, lambda e: e.vars.max(TERM_DIM), DATA_READ),
    "to-quadexpr": Case(grid, lambda e: e.to_quadexpr(), DATA_READ),
    "simplify": Case(grid, lambda e: e.simplify(), DATA_READ),
    "densify-terms": Case(grid, lambda e: (e + 0 * e).densify_terms(), KEPT),
    "reset-const": Case(grid, lambda e: (e + 1).where(COND).reset_const(), KEPT),
    "equals": Case(grid, lambda e: e.equals(e), "equals(dense comparison)"),
    "iter": Case(grid, lambda e: list(e), KEPT),
    "assign-coords": Case(grid, coords_reassigned, KEPT),
    "loc": Case(grid, lambda e: e.loc[[0, 1]], KEPT),
    "cumsum": Case(grid, lambda e: e.cumsum("a"), DATA_READ),
    "rolling": Case(grid, lambda e: e.rolling(a=2).sum(), DATA_READ),
    "diff": Case(grid, lambda e: e.diff("a"), KEPT),
    "mul-variable": Case(grid, lambda e: e * var(e, "y"), QUADRATIC),
    "pow": Case(single, lambda e: e**2, QUADRATIC),
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
    "const-assign-new-dim": Case(grid, const_assigned(OVER_C), ASSIGNMENT),
    "coeffs-assign": Case(grid, coeffs_assigned, ASSIGN_COEFFS),
}

CONST_REJECTED: dict[str, tuple[Any, str]] = {
    "nan": (OVER_A.where(OVER_A != 2.0), "NaN found"),
    "nan-scalar": (np.nan, "NaN found"),
    "partial": (OVER_A.sel(a=[0, 1]), "does not cover"),
    "partial-off-grid": (SHIFTED_A, "does not cover"),
}


@contextmanager
def no_densify() -> Iterator[None]:
    """Fail on any conversion notice, densify or sparsify, inside the block."""
    with linopy.options as opts, warnings.catch_warnings():
        warnings.simplefilter("error", linopy.PerformanceWarning)
        opts.set_value(warn_on_densify=True)
        yield


@contextmanager
def sparsified() -> Iterator[None]:
    """Expect exactly one conversion notice inside the block, the sparsify one."""
    with linopy.options as opts, warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        opts.set_value(warn_on_densify=True)
        yield
    notices = [str(w.message) for w in rec if w.category is linopy.PerformanceWarning]
    assert notices == [f"Dense backing {SPARSIFIED}."]


def assert_same(res: Any, want: Any) -> None:
    if isinstance(res, QuadraticExpression):
        pd.testing.assert_frame_equal(res.flat, want.flat)
    elif isinstance(res, LinearExpression):
        assert_equivalent(res, want)
    elif isinstance(res, Constraint):
        assert_equivalent(res.lhs, want.lhs)
        xr.testing.assert_equal(res.sign, want.sign)
        xr.testing.assert_allclose(res.rhs, want.rhs)
    elif isinstance(res, pd.DataFrame):
        pd.testing.assert_frame_equal(res, want)
    elif isinstance(res, bool | list):
        assert res == want
    else:
        xr.testing.assert_allclose(res, want)


@pytest.mark.parametrize("case", list(CASES))
def test_operation(backend: Model, case: str) -> None:
    build, op, expected = CASES[case]
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
    if backend.sparse and expected not in (ASSIGNMENT, ASSIGN_COEFFS):
        assert expr.is_sparse
    if backend.sparse and expected in (KEPT, SPARSIFIED):
        if isinstance(res, LinearExpression):
            assert res.is_sparse
        elif isinstance(res, Constraint):
            assert res.lhs.is_sparse


CONST_ASSIGNERS: dict[str, Callable[[LinearExpression, Any], Any]] = {
    "setter": lambda e, c: setattr(e, "const", c),
    "assign": lambda e, c: e.assign(const=c),
}


@pytest.mark.parametrize("assigner", list(CONST_ASSIGNERS))
@pytest.mark.parametrize("value", list(CONST_REJECTED))
def test_const_assign_rejects(backend: Model, value: str, assigner: str) -> None:
    const, match = CONST_REJECTED[value]
    expr = grid(backend)
    with no_densify(), pytest.raises(ValueError, match=match):
        CONST_ASSIGNERS[assigner](expr, const)
    assert expr.is_sparse == backend.sparse


def test_nan_constant_operand_rejects(backend: Model) -> None:
    expr = grid(backend)
    with no_densify(), pytest.raises(ValueError, match="NaN found"):
        expr + CONST_REJECTED["nan"][0]
    assert expr.is_sparse == backend.sparse


NONFINITE_OPS: dict[str, Callable[[LinearExpression], LinearExpression]] = {
    "mul-inf": lambda e: e * OVER_A.where(OVER_A != 2.0, np.inf),
    "div-zero": lambda e: e / OVER_A.where(OVER_A != 2.0, 0.0),
    "add-inf-then-sub-inf": lambda e: (e + np.inf) - np.inf,
    "mul-inf-zero-coeff": lambda e: (0 * e + 1) * np.inf,
    "mul-array-inf-zero-coeff": lambda e: (
        (0 * e + 1) * OVER_A.where(OVER_A != 2.0, np.inf)
    ),
}


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("op", list(NONFINITE_OPS))
def test_nonfinite_constant_op_raises(backend: Model, op: str) -> None:
    expr = grid(backend)
    with no_densify(), pytest.raises(ValueError, match="produced NaN"):
        NONFINITE_OPS[op](expr)
    assert expr.is_sparse == backend.sparse


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_nonfinite_constant_op_keeps_absent_cell(backend: Model) -> None:
    expr = (grid(backend) + 1).where(COND)
    with no_densify():
        res = expr * np.inf
    assert res.is_sparse == backend.sparse
    absent = ~COND.broadcast_like(res.const)
    np.testing.assert_array_equal(res.const.isnull(), absent)
    np.testing.assert_array_equal(res.has_terms, ~absent)


REASON_SOURCES = [linopy.expressions, linopy.store, linopy.csr, linopy.constraints]
REASON = re.compile(
    r"""(?<!\w)f?(["'])((?:[a-z_.]+|\{[^{}]+\})\((?:[^"'{}\n]|\{[^{}]+\}|(?!\1)["'])+\))\1"""
)


def as_pattern(reason: str) -> re.Pattern[str]:
    return re.compile(".+".join(map(re.escape, re.split(r"\{[^{}]+\}", reason))))


def test_every_fallback_case_is_listed() -> None:
    sources = [inspect.getsource(module) for module in REASON_SOURCES]
    reasons = {m[2] for source in sources for m in REASON.finditer(source)}
    expected = {c.expected for c in CASES.values()}
    missing = [
        r for r in reasons if not any(as_pattern(r).fullmatch(e) for e in expected)
    ]
    assert len(reasons) > 30
    assert missing == []


@pytest.mark.parametrize("dim", ["zzz", ["a", "zzz"]])
def test_sum_over_missing_dim_raises_without_densifying(
    backend: Model, dim: str | list[str]
) -> None:
    expr = grid(backend)
    with no_densify(), pytest.raises(KeyError, match="zzz"):
        expr.sum(dim)
    assert expr.is_sparse == backend.sparse


def _squared(sparse: bool) -> LinearExpression | QuadraticExpression:
    m = Model(sparse=sparse)
    x = m.add_variables(coords=[A], name="x")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", linopy.PerformanceWarning)
        return x * x


def test_quadratic_groupby_sum_densifies() -> None:
    group = pd.Series(SERIES)
    with no_densify():
        dense = _squared(False).groupby(group).sum()
    with pytest.warns(
        linopy.PerformanceWarning,
        match=re.escape("groupby.sum(grouper without a sparse path)"),
    ):
        sparse = _squared(True).groupby(group).sum()

    assert isinstance(dense, QuadraticExpression)
    assert isinstance(sparse, QuadraticExpression)
    assert not sparse.is_sparse
    assert_quadequal(sparse, dense)

    labels = dense.data.vars.sel(_factor=0).values
    assert set(labels[0][labels[0] >= 0].tolist()) == {0, 2}
    assert set(labels[1][labels[1] >= 0].tolist()) == {1}


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


def assign_coeffs(e: LinearExpression) -> None:
    e.coeffs = 2 * e.coeffs


def assign_vars(e: LinearExpression) -> None:
    e.vars = e.vars


def assign_const_off_grid(e: LinearExpression) -> None:
    e.const = OVER_C


@pytest.mark.parametrize("assign", [assign_coeffs, assign_vars, assign_const_off_grid])
def test_named_expression_keeps_name_through_assignment(
    backend: Model, assign: Callable[[LinearExpression], None]
) -> None:
    named = backend.add_expressions(grid(backend), name="named")
    assert isinstance(named, LinearExpression)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", linopy.PerformanceWarning)
        assert named.data.attrs["name"] == "named"
        assign(named)
    assert named.name == backend.expressions["named"].name == "named"


def positional_model() -> tuple[Model, LinearExpression]:
    m = Model()
    x = m.add_variables(coords=[A], name="x")
    y = m.add_variables(coords=[A], name="y")
    return m, 2 * dense_linexpr(x) + dense_linexpr(y) - dense_linexpr(x)


ROWS = {"all": [True, True, True], "masked": [True, False, True]}


@pytest.mark.parametrize("rows", list(ROWS))
def test_positional_csr_agrees(rows: str) -> None:
    m, expr = positional_model()
    active = np.array(ROWS[rows])
    label_index = m.variables.label_index
    dense = expr._store.positional_csr(label_index, active, "c")
    sparse_store = expr._store.to_sparse()
    assert isinstance(sparse_store, CSRStore)
    canonical = sparse_store.canonical()
    sparse = canonical.positional_csr(label_index, active, "c")
    assert dense.shape == sparse.shape == (active.sum(), 6)
    assert (dense != sparse).nnz == 0
    assert expr._store.nnz(active) == 3 * active.sum()
    assert canonical.nnz(active) >= sparse.nnz


@pytest.mark.parametrize("rows", list(ROWS))
def test_sparse_positional_csr_cached_while_held(rows: str) -> None:
    m, expr = positional_model()
    sparse_store = expr._store.to_sparse()
    assert isinstance(sparse_store, CSRStore)
    store = sparse_store.canonical()
    active = np.array(ROWS[rows])
    first = store.positional_csr(m.variables.label_index, active, "c")
    again = store.positional_csr(m.variables.label_index, active, "c")
    assert all(
        a is b
        for a, b in zip(
            (first.data, first.indices, first.indptr),
            (again.data, again.indices, again.indptr),
        )
    )
    other_rows = store.positional_csr(m.variables.label_index, ~active, "c")
    assert other_rows.indices is not first.indices
    m.add_variables(name="z")
    rebuilt = store.positional_csr(m.variables.label_index, active, "c")
    assert rebuilt.indices is not first.indices
    np.testing.assert_array_equal(rebuilt.indices, first.indices)
