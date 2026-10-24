from __future__ import annotations

import numpy as np
import scipy.sparse
import xarray as xr
from xarray.testing import assert_equal

from linopy.constants import TERM_DIM
from linopy.constraints import Constraint, _con_unwrap
from linopy.expressions import LinearExpression, QuadraticExpression, _expr_unwrap
from linopy.model import Model
from linopy.variables import Variable, _var_unwrap


def _align_dim_order(target: xr.Dataset, other: xr.Dataset) -> xr.Dataset:
    """
    Reorder ``other``'s dimensions to match ``target`` when they share a dim set.

    linopy expressions inherit xarray's broadcasting, whose dimension order
    follows operand order (``x + y`` yields ``(x_dim, y_dim)`` while ``y + x``
    yields ``(y_dim, x_dim)``). That difference is not semantically meaningful,
    so it should not make two expressions compare unequal. If the dimension
    sets differ the inputs are genuinely unequal; leave ``other`` untouched so
    :func:`assert_equal` reports the real difference.
    """
    if set(target.dims) == set(other.dims):
        return other.transpose(*target.dims)
    return other


def _sort_by_vars_along_term(expr: LinearExpression) -> LinearExpression:
    """Sort a linear expression's terms by variable labels along _term."""
    ds = expr.data
    if TERM_DIM not in ds.dims:
        return expr
    order = np.argsort(ds["vars"].values, axis=-1, kind="stable")
    sorted_vars = np.take_along_axis(ds["vars"].values, order, axis=-1)
    sorted_coeffs = np.take_along_axis(ds["coeffs"].values, order, axis=-1)
    new_ds = ds.copy()
    new_ds["vars"] = (ds["vars"].dims, sorted_vars)
    new_ds["coeffs"] = (ds["coeffs"].dims, sorted_coeffs)
    return LinearExpression(new_ds, expr.model)


def assert_varequal(a: Variable, b: Variable) -> None:
    """Assert that two variables are equal."""
    return assert_equal(_var_unwrap(a), _var_unwrap(b))


def assert_linequal(
    a: LinearExpression | QuadraticExpression, b: LinearExpression | QuadraticExpression
) -> None:
    """
    Assert that two linear expressions are semantically equal.

    Terms are sorted by variable labels along _term and dimension order is
    aligned before comparing, so expressions with different term orderings or
    dimension orderings but identical mathematical meaning are considered equal.
    """
    assert isinstance(a, LinearExpression)
    assert isinstance(b, LinearExpression)
    a_ds = _expr_unwrap(_sort_by_vars_along_term(a))
    b_ds = _expr_unwrap(_sort_by_vars_along_term(b))
    return assert_equal(a_ds, _align_dim_order(a_ds, b_ds))


def _cell_matrix(expr: LinearExpression, n_vars: int) -> scipy.sparse.csr_array:
    """Per-cell term matrix with duplicate labels summed and zeros pruned."""
    cells, vars_, coeffs = expr._store.terms()
    shape = (int(expr.const.size), n_vars)
    matrix = scipy.sparse.coo_array((coeffs, (cells, vars_)), shape=shape).tocsr()
    matrix.eliminate_zeros()
    return matrix


def assert_equivalent(a: LinearExpression, b: LinearExpression) -> None:
    """
    Assert that two linear expressions are equivalent independently of their
    backing (dense or sparse) and term layout.

    Compares the coordinates including auxiliary ones, the dimension order,
    the absence mask, the constants and the per-cell term matrix after
    summing duplicate variables and dropping zero coefficients. ``nterm``
    and the term order are ignored. Never densifies a sparse operand.
    """
    assert isinstance(a, LinearExpression)
    assert isinstance(b, LinearExpression)
    xr.testing.assert_identical(
        xr.Dataset(coords=a.coords), xr.Dataset(coords=b.coords)
    )
    assert a.coord_dims == b.coord_dims
    xr.testing.assert_identical(a.isnull(), b.isnull())
    xr.testing.assert_allclose(a.const, b.const)
    n_vars = max(a.model._xCounter, b.model._xCounter)
    got, want = _cell_matrix(a, n_vars), _cell_matrix(b, n_vars)
    np.testing.assert_array_equal(got.indptr, want.indptr)
    np.testing.assert_array_equal(got.indices, want.indices)
    np.testing.assert_allclose(got.data, want.data)


def assert_quadequal(
    a: LinearExpression | QuadraticExpression, b: LinearExpression | QuadraticExpression
) -> None:
    """Assert that two quadratic or linear expressions are equal."""
    return assert_equal(_expr_unwrap(a), _expr_unwrap(b))


def assert_exprequal(
    a: LinearExpression | QuadraticExpression,
    b: LinearExpression | QuadraticExpression,
    check_name: bool = True,
) -> None:
    """
    Assert that two expressions are equal, dispatching on linear vs quadratic.

    xarray's assert_equal ignores attrs, so the stored name (which lives in
    ``attrs["name"]``) is compared explicitly unless ``check_name=False``.
    """
    assert type(a) is type(b), f"expression types differ: {type(a)} != {type(b)}"
    if check_name:
        assert a.name == b.name, f"expression names differ: {a.name!r} != {b.name!r}"
    if isinstance(a, QuadraticExpression):
        assert_quadequal(a, b)
    else:
        assert_linequal(a, b)


def assert_conequal(a: Constraint, b: Constraint, strict: bool = True) -> None:
    """
    Assert that two constraints are equal.

    Parameters
    ----------
        a: Constraint
            The first constraint.
        b: Constraint
            The second constraint.
        strict: bool
            Whether to compare the constraints strictly. If not, only compare
            mathematically relevant parts: the lhs by :func:`assert_equivalent`,
            independently of its store, and the sign and rhs.
    """
    if strict:
        assert_equal(_con_unwrap(a), _con_unwrap(b))
    else:
        assert_equivalent(a.lhs, b.lhs)
        assert_equal(a.sign, b.sign)
        assert_equal(a.rhs, b.rhs)


def assert_model_equal(a: Model, b: Model) -> None:
    """Assert that two models are equal."""
    for k in a.dataset_attrs:
        assert_equal(getattr(a, k), getattr(b, k))

    assert list(a.variables) == list(b.variables)
    assert list(a.constraints) == list(b.constraints)

    for v in a.variables:
        assert_varequal(a.variables[v], b.variables[v])

    for c in a.constraints:
        assert_conequal(a.constraints[c], b.constraints[c])

    assert list(a.expressions) == list(b.expressions)

    for e in a.expressions:
        assert_exprequal(a.expressions[e], b.expressions[e])

    assert_exprequal(a.objective.expression, b.objective.expression, check_name=False)
    assert a.objective.sense == b.objective.sense
    assert a.objective.value == b.objective.value

    assert a.status == b.status
    assert a.termination_condition == b.termination_condition

    assert a.type == b.type
