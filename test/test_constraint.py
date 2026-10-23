#!/usr/bin/env python3
"""
Created on Tue Nov  2 22:38:48 2021.

@author: fabian
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
from xarray.testing import assert_equal

import linopy
from linopy import EQUAL, GREATER_EQUAL, LESS_EQUAL, LinearExpression, Model
from linopy.constants import (
    HELPER_DIMS,
    TERM_DIM,
    long_EQUAL,
    short_GREATER_EQUAL,
    short_LESS_EQUAL,
    sign_replace_dict,
)
from linopy.constraints import (
    AnonymousScalarConstraint,
    Constraint,
    Constraints,
)
from linopy.csr import canonical
from linopy.testing import assert_linequal, assert_varequal


@pytest.fixture
def m() -> Model:
    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(10, name="first")], name="x")
    m.add_variables(coords=[pd.Index([1, 2, 3], name="second")], name="y")
    m.add_variables(0, 10, name="z")
    m.add_constraints(x >= 0, name="c")
    return m


@pytest.fixture
def x(m: Model) -> linopy.Variable:
    return m.variables["x"]


@pytest.fixture
def y(m: Model) -> linopy.Variable:
    return m.variables["y"]


@pytest.fixture
def c(m: Model) -> linopy.constraints.Constraint:
    return m.constraints["c"]


@pytest.fixture
def mc(m: Model) -> linopy.constraints.Constraint:
    return m.constraints["c"]


def test_constraint_repr(c: linopy.constraints.Constraint) -> None:
    c.__repr__()


def test_constraint_repr_equivalent_across_stores(
    c: linopy.constraints.Constraint,
) -> None:
    """The repr is the same for a dense and a sparse lhs."""
    dense = repr(c)
    assert repr(c.to_sparse()) == dense


def test_constraints_repr(m: Model) -> None:
    m.constraints.__repr__()


def test_add_constraints_holds_lhs_on_model_store(backend: Model) -> None:
    x = backend.add_variables(coords=[pd.RangeIndex(10, name="first")], name="x")
    c = backend.add_constraints(1 * x >= 1, name="c")
    assert type(c) is Constraint
    assert c.lhs.is_sparse == backend.sparse
    assert backend.constraints["c"] is c
    assert c.ncons == 10


def test_add_constraints_penalty_softens_constraint(
    m: Model, y: linopy.Variable
) -> None:
    """`penalty=` on add_constraints is a shortcut for calling `.soften()`."""
    m.add_objective(y.sum(), sense="min")
    penalty_coeff = 10
    constraint = m.add_constraints(y >= 10, name="constraint", penalty=penalty_coeff)

    assert isinstance(constraint, linopy.constraints.Constraint)
    # The slack term was added to the lhs, same effect as calling .soften() directly:
    assert constraint.lhs.nterm == 2
    expected_objective = (
        y.sum() + penalty_coeff * m.variables["constraint_slack_pos"].sum()
    )
    assert_linequal(m.objective.expression, expected_objective)


def test_constraint_name(c: linopy.constraints.Constraint) -> None:
    assert c.name == "c"


def test_empty_constraints_repr() -> None:
    # test empty contraints
    Model().constraints.__repr__()


def test_constraint_handles_empty_rows(backend: Model) -> None:
    """An empty constraint group must be accepted and solve cleanly."""

    m = backend
    x = m.add_variables(
        lower=0.0,
        coords=[range(3), range(2)],
        dims=["time", "product"],
        name="x",
    )
    empty = x.isel(time=range(1, 1))
    c = m.add_constraints(empty == 0, name="empty")
    assert isinstance(c, linopy.constraints.Constraint)
    assert c.size == 0
    # Solving a model with only an empty constraint group is also fine.
    m.add_objective(x.sum())
    m.solve("highs", io_api="direct", output_flag=False)
    assert m.status == "ok"


def test_cannot_create_constraint_without_variable() -> None:
    model = linopy.Model()
    with pytest.raises(ValueError):
        _ = linopy.LinearExpression(12, model) == linopy.LinearExpression(13, model)


def test_constraints_getter(m: Model, c: linopy.constraints.Constraint) -> None:
    assert c.shape == (10,)
    assert isinstance(m.constraints[["c"]], Constraints)


def test_anonymous_constraint_from_linear_expression_le(
    x: linopy.Variable, y: linopy.Variable
) -> None:
    expr = 10 * x + y
    con = expr <= 10
    assert isinstance(con.lhs, LinearExpression)
    assert (con.sign == LESS_EQUAL).all()
    assert (con.rhs == 10).all()


def test_anonymous_constraint_from_linear_expression_ge(
    x: linopy.Variable, y: linopy.Variable
) -> None:
    expr = 10 * x + y
    con = expr >= 10
    assert isinstance(con.lhs, LinearExpression)
    assert (con.sign == GREATER_EQUAL).all()
    assert (con.rhs == 10).all()


def test_anonymous_constraint_from_linear_expression_eq(
    x: linopy.Variable, y: linopy.Variable
) -> None:
    expr = 10 * x + y
    con = expr == 10
    assert isinstance(con.lhs, LinearExpression)
    assert (con.sign == EQUAL).all()
    assert (con.rhs == 10).all()


def test_anonymous_constraint_from_variable_le(x: linopy.Variable) -> None:
    con = x <= 10
    assert isinstance(con.lhs, LinearExpression)
    assert (con.sign == LESS_EQUAL).all()
    assert (con.rhs == 10).all()


def test_anonymous_constraint_from_variable_ge(x: linopy.Variable) -> None:
    con = x >= 10
    assert isinstance(con.lhs, LinearExpression)
    assert (con.sign == GREATER_EQUAL).all()
    assert (con.rhs == 10).all()


def test_anonymous_constraint_from_variable_eq(x: linopy.Variable) -> None:
    con = x == 10
    assert isinstance(con.lhs, LinearExpression)
    assert (con.sign == EQUAL).all()
    assert (con.rhs == 10).all()


def test_anonymous_constraint_with_variable_on_rhs(
    x: linopy.Variable, y: linopy.Variable
) -> None:
    expr = 10 * x + y
    con = expr == x
    assert isinstance(con.lhs, LinearExpression)
    assert (con.sign == EQUAL).all()
    assert (con.rhs == 0).all()


def test_anonymous_constraint_with_constant_on_lhs(
    x: linopy.Variable, y: linopy.Variable
) -> None:
    expr = 10 * x + y + 10
    con = expr == 0
    assert isinstance(con.lhs, LinearExpression)
    assert (con.lhs.const == 0.0).all()
    assert (con.sign == EQUAL).all()
    assert (con.rhs == -10).all()


def test_anonymous_constraint_with_constant_on_rhs(
    x: linopy.Variable, y: linopy.Variable
) -> None:
    expr = 10 * x + y
    con = expr == 10
    assert isinstance(con.lhs, LinearExpression)
    assert (con.sign == EQUAL).all()
    assert (con.rhs == 10).all()


def test_anonymous_constraint_with_expression_on_both_sides(
    x: linopy.Variable, y: linopy.Variable
) -> None:
    expr = 10 * x + y + 10
    con = expr == expr
    assert isinstance(con.lhs, LinearExpression)
    assert con.lhs.nterm == 4  # are stacked on top of each other
    assert (con.coeffs.sum(con.term_dim) == 0).all()
    assert (con.sign == EQUAL).all()
    assert (con.rhs == 0).all()


def test_anonymous_scalar_constraint_with_scalar_variable_on_rhs(
    x: linopy.Variable, y: linopy.Variable
) -> None:
    expr = 10 * x.at[0] + y.at[1]
    with pytest.raises(TypeError):
        expr == x.at[0]  # type: ignore
        # assert isinstance(con.lhs, LinearExpression)
        # assert (con.sign == EQUAL).all()
        # assert (con.rhs == 0).all()


def test_constraint_inherited_properties(
    x: linopy.Variable, y: linopy.Variable
) -> None:
    con = 10 * x + y <= 10
    assert isinstance(con.attrs, dict)
    assert isinstance(con.coords, xr.Coordinates)
    assert isinstance(con.indexes, xr.core.indexes.Indexes)
    assert isinstance(con.sizes, xr.core.utils.Frozen)
    assert isinstance(con.ndim, int)
    assert isinstance(con.nterm, int)
    assert isinstance(con.shape, tuple)
    assert isinstance(con.size, int)
    assert isinstance(con.dims, xr.core.utils.Frozen)


def test_constraint_assign_replaces_term_fields(
    x: linopy.Variable, y: linopy.Variable
) -> None:
    con = 10 * x + y <= 10
    assert isinstance(con, Constraint)
    res = con.assign(
        vars=con.vars.isel({TERM_DIM: [0]}), coeffs=con.coeffs.isel({TERM_DIM: [0]})
    )
    assert isinstance(res, Constraint)
    assert res.sizes[TERM_DIM] == 1
    assert_equal(res.vars, con.vars.isel({TERM_DIM: [0]}))


@pytest.mark.parametrize(
    "build",
    [lambda x: x, lambda x: 1 * x, lambda x: x >= 0],
    ids=["var", "expr", "con"],
)
def test_set_index_swaps_dim_coord(x: linopy.Variable, build: Callable) -> None:
    obj = build(x)
    alt = np.arange(10, 20)
    res = obj.assign_coords(alt=("first", alt)).set_index(first="alt")
    assert type(res) is type(obj)
    assert res.indexes["first"].tolist() == alt.tolist()


def test_constraint_wrapped_methods(x: linopy.Variable, y: linopy.Variable) -> None:
    con = 10 * x + y <= 10
    assert isinstance(con, Constraint)

    # Test wrapped methods
    con.assign(rhs=xr.zeros_like(con.rhs))
    con.assign_attrs({"new_attr": "value"})
    con.assign_coords(
        {"new_coord": xr.DataArray(np.zeros((2, 2)), coords=[range(2), range(2)])}
    )
    # con.bfill(dim="first")
    con.broadcast_like(con.data)
    con.chunk()
    con.drop_sel({"first": 0})
    con.drop_isel({"first": 0})
    con.expand_dims("new_dim")
    # con.ffill(dim="first")
    con.shift({"first": 1})
    con.reindex({"first": [0, 1]})
    con.reindex_like(con.data)
    con.rename({"first": "new_labels"})
    con.rename_dims({"first": "new_labels"})
    con.roll({"first": 1})
    con.stack(new_dim=("first", "second")).unstack("new_dim")


def test_anonymous_constraint_sel(x: linopy.Variable, y: linopy.Variable) -> None:
    expr = 10 * x + y
    con = expr <= 10
    assert isinstance(con, Constraint)
    assert isinstance(con.sel(first=[1, 2]), Constraint)


def test_anonymous_constraint_swap_dims(x: linopy.Variable, y: linopy.Variable) -> None:
    expr = 10 * x + y
    con = expr <= 10
    assert isinstance(con, Constraint)
    con = con.assign_coords({"third": ("second", con.indexes["second"] + 100)})
    con = con.swap_dims({"second": "third"})
    assert isinstance(con, Constraint)
    assert con.coord_dims == ("first", "third")


def test_anonymous_constraint_loc(x: linopy.Variable, y: linopy.Variable) -> None:
    expr = 10 * x + y
    con = expr <= 10
    assert isinstance(con, Constraint)
    assert isinstance(con.loc[[1, 2]], Constraint)


def test_anonymous_constraint_getitem(x: linopy.Variable, y: linopy.Variable) -> None:
    expr = 10 * x + y
    con = expr <= 10
    assert isinstance(con[1], Constraint)


def test_constraint_from_rule(m: Model, x: linopy.Variable, y: linopy.Variable) -> None:
    def bound(m: Model, i: int, j: int) -> AnonymousScalarConstraint:
        return (i - 1) * x.at[i - 1] + y.at[j] >= 0 if i % 2 else i * x.at[i] >= 0

    coords = [x.coords["first"], y.coords["second"]]
    con = Constraint.from_rule(m, bound, coords)
    assert isinstance(con, Constraint)
    assert con.lhs.nterm == 2
    repr(con)  # test repr


def test_constraint_from_rule_with_none_return(
    m: Model, x: linopy.Variable, y: linopy.Variable
) -> None:
    def bound(m: Model, i: int, j: int) -> AnonymousScalarConstraint | None:
        if i % 2:
            return i * x.at[i] + y.at[j] >= 0
        return None

    coords = [x.coords["first"], y.coords["second"]]
    con = Constraint.from_rule(m, bound, coords)
    assert isinstance(con, Constraint)
    assert isinstance(con.lhs.vars, xr.DataArray)
    assert con.lhs.nterm == 2
    assert (con.lhs.vars.loc[0, :] == -1).all()
    assert (con.lhs.vars.loc[1, :] != -1).all()
    repr(con)  # test repr


def test_constraint_vars_getter(
    mc: linopy.constraints.Constraint, x: linopy.Variable
) -> None:
    assert_equal(mc.vars.squeeze(), x.labels)


def test_constraint_coeffs_getter(mc: linopy.constraints.Constraint) -> None:
    assert (mc.coeffs == 1).all()


def test_constraint_sign_getter(c: linopy.constraints.Constraint) -> None:
    assert (c.sign == GREATER_EQUAL).all()


def test_constraint_rhs_getter(c: linopy.constraints.Constraint) -> None:
    assert (c.rhs == 0).all()


def test_constraint_update_variables(
    mc: linopy.constraints.Constraint, x: linopy.Variable
) -> None:
    mc.update(variables=x)
    assert_equal(mc.vars.isel({TERM_DIM: 0}), x.labels)


@pytest.mark.parametrize("as_type", [lambda v: v, pd.DataFrame])
def test_constraint_update_variables_invalid(
    mc: linopy.constraints.Constraint,
    x: linopy.Variable,
    as_type: Callable[[xr.DataArray], Any],
) -> None:
    with pytest.raises(TypeError, match="expects a Variable"):
        mc.update(variables=as_type(x.labels))


@pytest.mark.parametrize(
    "attr", ["coeffs", "vars", "sign", "rhs", "lhs"], ids=lambda a: f"{a}-setter"
)
def test_constraint_field_setters_removed(
    mc: linopy.constraints.Constraint, attr: str
) -> None:
    with pytest.raises(AttributeError):
        setattr(mc, attr, getattr(mc, attr))


def test_constraint_update_coeffs(mc: linopy.constraints.Constraint) -> None:
    mc.update(coeffs=3)
    assert (mc.coeffs == 3).all()


def test_constraint_update_lhs(
    mc: linopy.constraints.Constraint, x: linopy.Variable, y: linopy.Variable
) -> None:
    mc.update(lhs=x + y)
    assert mc.lhs.nterm == 2
    assert mc.vars.notnull().all().item()
    assert mc.coeffs.notnull().all().item()


def test_constraint_update_lhs_with_variable(
    mc: linopy.constraints.Constraint, x: linopy.Variable
) -> None:
    mc.update(lhs=x)
    assert mc.lhs.nterm == 1


def test_constraint_update_lhs_with_constant(
    mc: linopy.constraints.Constraint,
) -> None:
    sizes = mc.sizes
    mc.update(lhs=10)
    assert (mc.rhs == -10).all()
    assert mc.lhs.nterm == 0
    assert mc.sizes["first"] == sizes["first"]


@pytest.mark.parametrize("sign", [EQUAL, long_EQUAL])
def test_constraint_update_sign(mc: linopy.constraints.Constraint, sign: str) -> None:
    mc.update(sign=sign)
    assert (mc.sign == EQUAL).all()


def test_constraint_update_sign_invalid(
    mc: linopy.constraints.Constraint,
) -> None:
    with pytest.raises(ValueError):
        mc.update(sign="asd")


def test_constraint_update_rhs(mc: linopy.constraints.Constraint) -> None:
    sizes = mc.sizes
    mc.update(rhs=2)
    assert (mc.rhs == 2).all()
    assert mc.sizes == sizes


def test_constraint_update_rhs_and_sign(mc: linopy.constraints.Constraint) -> None:
    mc.update(rhs=5, sign=EQUAL)
    assert (mc.rhs == 5).all()
    assert (mc.sign == EQUAL).all()


def test_constraint_update_no_kwargs_is_noop(
    mc: linopy.constraints.Constraint,
) -> None:
    old_rhs = mc.rhs.copy()
    old_sign = mc.sign.copy()
    mc.update()
    assert (mc.rhs == old_rhs).all()
    assert (mc.sign == old_sign).all()


def test_constraint_update_rearranges_variable_rhs(
    mc: linopy.constraints.Constraint, x: linopy.Variable
) -> None:
    """
    Variable / Expression rhs is moved onto lhs; only the constant
    part lands on rhs (mirrors add_constraints and the .rhs setter).
    """
    mc.update(rhs=x + 3)
    assert (mc.rhs == 3).all()
    assert mc.lhs.nterm == 2  # original term + the rearranged -x


def test_constraint_update_returns_self(
    mc: linopy.constraints.Constraint,
) -> None:
    out = mc.update(rhs=7)
    assert out is mc


def test_constraint_update_positional_constraint_expression(
    mc: linopy.constraints.Constraint, x: linopy.Variable, y: linopy.Variable
) -> None:
    """``c.update(x + 5 <= 3)`` replaces lhs / sign / rhs in one call."""
    mc.update(x + y <= 7)
    assert (mc.rhs == 7).all()
    assert (mc.sign == LESS_EQUAL).all()
    assert mc.lhs.nterm == 2


def test_constraint_update_positional_rejects_mixing_kwargs(
    mc: linopy.constraints.Constraint, x: linopy.Variable
) -> None:
    """Positional constraint can't be combined with keyword updates."""
    with pytest.raises(TypeError, match="cannot be combined with keyword"):
        mc.update(x <= 3, sign=EQUAL)


def test_constraint_update_positional_rejects_non_constraint(
    mc: linopy.constraints.Constraint,
) -> None:
    """Random objects are rejected with a clear error."""
    with pytest.raises(TypeError, match="must be a ConstraintLike"):
        mc.update("not a constraint")  # type: ignore


def test_constraint_update_lhs_only(
    mc: linopy.constraints.Constraint, x: linopy.Variable, y: linopy.Variable
) -> None:
    """lhs= alone replaces the expression; rhs and sign untouched."""
    old_rhs = mc.rhs.copy()
    old_sign = mc.sign.copy()
    mc.update(lhs=5 * x + 7 * y)
    assert (mc.rhs == old_rhs).all()
    assert (mc.sign == old_sign).all()
    assert mc.lhs.nterm == 2


def test_constraint_update_coeffs_only_keeps_values(
    mc: linopy.constraints.Constraint,
) -> None:
    """coeffs= alone replaces the coef array element-wise; vars untouched."""
    old_vars = mc.vars.copy()
    mc.update(coeffs=mc.coeffs * 10)
    assert (mc.vars == old_vars).all()
    # original was mc.lhs with leading coeff; *10 → all coeffs *10
    assert mc.coeffs.max() >= 10


def test_constraint_update_lhs_and_sign_together(
    mc: linopy.constraints.Constraint, x: linopy.Variable
) -> None:
    """Compound updates compose: lhs replacement + sign flip in one call."""
    mc.update(lhs=2 * x, sign=EQUAL)
    assert (mc.sign == EQUAL).all()
    assert mc.lhs.nterm == 1


def test_constraint_update_lhs_and_coeffs_rejected(
    mc: linopy.constraints.Constraint, x: linopy.Variable
) -> None:
    """lhs= (full replacement) and coeffs= (partial) are mutually exclusive."""
    with pytest.raises(TypeError, match="lhs.*coeffs.*variables"):
        mc.update(lhs=2 * x, coeffs=mc.coeffs * 2)


def test_constraint_update_lhs_and_variables_rejected(
    mc: linopy.constraints.Constraint, x: linopy.Variable
) -> None:
    """lhs= (full replacement) and variables= (partial) are mutually exclusive."""
    with pytest.raises(TypeError, match="lhs.*coeffs.*variables"):
        mc.update(lhs=2 * x, variables=mc.vars)  # type: ignore[arg-type]


def test_constraint_update_rhs_with_variable(
    mc: linopy.constraints.Constraint, x: linopy.Variable
) -> None:
    mc.update(rhs=x)
    assert (mc.rhs == 0).all()
    assert (mc.coeffs.isel({mc.term_dim: -1}) == -1).all()
    assert mc.lhs.nterm == 2


def test_constraint_update_rhs_with_expression(
    mc: linopy.constraints.Constraint, x: linopy.Variable, y: linopy.Variable
) -> None:
    mc.update(rhs=x + y)
    assert (mc.rhs == 0).all()
    assert (mc.coeffs.isel({mc.term_dim: -1}) == -1).all()
    assert mc.lhs.nterm == 3


def test_constraint_update_rhs_with_expression_and_constant(
    mc: linopy.constraints.Constraint, x: linopy.Variable
) -> None:
    mc.update(rhs=x + 1)
    assert (mc.rhs == 1).all()
    assert (mc.coeffs.sum(mc.term_dim) == 0).all()
    assert mc.lhs.nterm == 2


def test_constraint_update_rhs_broadcasts_missing_dim() -> None:
    """Rhs update broadcasts against the constraint coords: missing dims expand."""
    m = Model()
    x = m.add_variables(
        coords=[pd.RangeIndex(2, name="i"), pd.RangeIndex(3, name="j")], name="x"
    )
    con = m.add_constraints(1 * x >= 0, name="con")

    con.update(rhs=xr.DataArray([1.0, 2.0], dims=["i"], coords={"i": [0, 1]}))

    assert dict(con.rhs.sizes) == {"i": 2, "j": 3}
    assert (con.rhs.sel(i=1) == 2.0).all()


def test_constraint_labels_setter_invalid(c: linopy.constraints.Constraint) -> None:
    with pytest.raises(AttributeError):
        c.labels = c.labels  # type: ignore


def test_constraint_sel(c: linopy.constraints.Constraint) -> None:
    assert isinstance(c.sel(first=[1, 2]), Constraint)
    assert isinstance(c.isel(first=[1, 2]), Constraint)


def test_constraint_flat_removed(c: linopy.constraints.Constraint) -> None:
    with pytest.raises(AttributeError):
        c.flat  # type: ignore[attr-defined]


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_iterate_slices(mc: linopy.constraints.Constraint, sparse: bool) -> None:
    con = mc.to_sparse() if sparse else mc
    slices = list(con.iterate_slices(slice_size=2))
    assert len(slices) > 1
    for i in slices:
        assert isinstance(i, Constraint)
        assert con.coord_dims == i.coord_dims
        assert i.lhs.is_sparse == sparse
    sliced = pl.concat([i.to_polars() for i in slices])
    assert sliced.equals(con.to_polars())


def test_constraint_to_polars(c: linopy.constraints.Constraint) -> None:
    assert isinstance(c.to_polars(), pl.DataFrame)


def test_constraint_to_polars_mixed_signs(m: Model, x: linopy.Variable) -> None:
    """Test to_polars when a constraint has mixed sign values across dims."""
    # Use Constraint so sign data can be patched
    con = m.add_constraints(x >= 0, name="mixed")
    # Replace sign data with mixed signs across the first dimension
    n = con.sizes["first"]
    signs = np.array(["<=" if i % 2 == 0 else ">=" for i in range(n)])
    con.update(sign=xr.DataArray(signs, dims=con.sign.dims))
    df = con.to_polars()
    assert isinstance(df, pl.DataFrame)
    assert set(df["sign"].to_list()) == {"<=", ">="}


def test_constraint_assignment_with_anonymous_constraints(
    m: Model, x: linopy.Variable, y: linopy.Variable
) -> None:
    m.add_constraints(x + y == 0, name="c2")
    assert m.constraints["c2"].vars.notnull().all()
    assert m.constraints["c2"].coeffs.notnull().all()


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_constraint_assignment_sanitize_zeros(
    m: Model, x: linopy.Variable, y: linopy.Variable, sparse: bool
) -> None:
    con = m.add_constraints(0 * x + y == 0, name="c2")
    if sparse:
        con.to_sparse()
    m.constraints.sanitize_zeros()
    c2 = m.constraints["c2"]
    assert c2.nterm == 1
    assert c2.has_variable(y)
    assert not c2.has_variable(x)
    csr, _ = c2.to_matrix(m.variables.label_index)
    assert (csr.data == 1).all()


def test_constraint_assignment_with_args(
    m: Model, x: linopy.Variable, y: linopy.Variable
) -> None:
    lhs = x + y
    m.add_constraints(lhs, EQUAL, 0, name="c2")
    assert m.constraints["c2"].vars.notnull().all()
    assert m.constraints["c2"].coeffs.notnull().all()
    assert (m.constraints["c2"].sign == EQUAL).all()
    assert (m.constraints["c2"].rhs == 0).all()


def test_constraint_assignment_with_args_and_constant(
    m: Model, x: linopy.Variable, y: linopy.Variable
) -> None:
    lhs = x + y + 10
    m.add_constraints(lhs, EQUAL, 0, name="c2")
    assert m.constraints["c2"].vars.notnull().all()
    assert m.constraints["c2"].coeffs.notnull().all()
    assert (m.constraints["c2"].sign == EQUAL).all()
    assert (m.constraints["c2"].rhs == -10).all()


def test_constraint_assignment_with_args_valid_sign(
    m: Model, x: linopy.Variable, y: linopy.Variable
) -> None:
    lhs = x + y
    for i, sign in enumerate([EQUAL, GREATER_EQUAL, LESS_EQUAL]):
        m.add_constraints(lhs, sign, 0, name=f"c{i}")
        assert m.constraints[f"c{i}"].vars.notnull().all()
        assert m.constraints[f"c{i}"].coeffs.notnull().all()
        assert (m.constraints[f"c{i}"].sign == sign).all()
        assert (m.constraints[f"c{i}"].rhs == 0).all()


def test_constraint_assignment_with_args_alternative_sign(
    m: Model, x: linopy.Variable, y: linopy.Variable
) -> None:
    lhs = x + y

    for i, sign in enumerate([long_EQUAL, short_GREATER_EQUAL, short_LESS_EQUAL]):
        m.add_constraints(lhs, sign, 0, name=f"c{i}")
        assert m.constraints[f"c{i}"].vars.notnull().all()
        assert m.constraints[f"c{i}"].coeffs.notnull().all()
        assert (m.constraints[f"c{i}"].sign == sign_replace_dict[sign]).all()
        assert (m.constraints[f"c{i}"].rhs == 0).all()


def test_constraint_assignment_assert_sign_rhs_not_none(
    m: Model, x: linopy.Variable, y: linopy.Variable
) -> None:
    lhs = x + y
    with pytest.raises(ValueError):
        m.add_constraints(lhs, EQUAL, None)


def test_constraint_assignment_callable_assert_sign_rhs_not_none(
    m: Model, x: linopy.Variable, y: linopy.Variable
) -> None:
    def lhs(x: linopy.Variable) -> None:
        return None

    coords = [x.coords["first"], y.coords["second"]]
    with pytest.raises(ValueError):
        m.add_constraints(lhs, EQUAL, None, coords=coords)


def test_constraint_assignment_tuple_assert_sign_rhs_not_none(
    m: Model, x: linopy.Variable, y: linopy.Variable
) -> None:
    lhs = [(1, x), (2, y)]
    with pytest.raises(ValueError):
        m.add_constraints(lhs, EQUAL, None)


def test_constraint_assignment_assert_sign_rhs_none(
    m: Model, x: linopy.Variable, y: linopy.Variable
) -> None:
    con = x + y >= 0
    with pytest.raises(ValueError):
        m.add_constraints(con, EQUAL, None)

    with pytest.raises(ValueError):
        m.add_constraints(con, None, 0)


def test_constraint_assignment_scalar_constraints_assert_sign_rhs_none(
    m: Model, x: linopy.Variable, y: linopy.Variable
) -> None:
    con = x.at[0] + y.at[1] >= 0
    with pytest.raises(ValueError):
        m.add_constraints(con, EQUAL, None)

    with pytest.raises(ValueError):
        m.add_constraints(con, None, 0)


def test_constraint_assignment_with_args_invalid_sign(
    m: Model, x: linopy.Variable, y: linopy.Variable
) -> None:
    lhs = x + y
    with pytest.raises(ValueError):
        m.add_constraints(lhs, ",", 0)


def test_constraint_with_helper_dims_as_coords(m: Model) -> None:
    coords = [pd.Index([0], name="a"), pd.Index([1, 2], name=TERM_DIM)]
    coeffs = xr.DataArray(np.array([[1, 2]]), coords=coords)
    vars = xr.DataArray(np.array([[1, 2]]), coords=coords)
    sign = xr.DataArray("==", coords=[coords[0]])
    rhs = xr.DataArray(np.array([0]), coords=[coords[0]])

    data = xr.Dataset({"coeffs": coeffs, "vars": vars, "sign": sign, "rhs": rhs})
    assert set(HELPER_DIMS).intersection(set(data.coords))
    con = Constraint.from_dataset(data, m, "c")

    expr = m.add_constraints(con)
    assert not set(HELPER_DIMS).intersection(set(expr.coords))


def test_constraint_matrix(m: Model) -> None:
    # Returns (csr_array, con_labels) — dense: active rows and active-var columns
    A, con_labels = m.constraints.to_matrix()
    n_active_vars = len(m.variables.label_index.vlabels)
    assert A.shape == (10, n_active_vars)
    assert len(con_labels) == 10


def test_constraint_matrix_masked_variables() -> None:
    """
    Test constraint matrix with missing variables.

    In this case the variables that are used in the constraints are
    missing. The matrix shoud not be built for constraints which have
    variables which are missing.
    """
    m = Model()
    mask = pd.Series([False] * 5 + [True] * 5)
    x = m.add_variables(coords=[range(10)], mask=mask)
    m.add_variables()
    m.add_constraints(x, EQUAL, 0)
    # Returns dense matrix: active rows only, all active-var columns
    A, con_labels = m.constraints.to_matrix()
    n_active_vars = len(m.variables.label_index.vlabels)
    assert A.shape == (m.ncons, n_active_vars)
    assert len(con_labels) == m.ncons


def test_constraint_matrix_masked_constraints() -> None:
    """
    Test constraint matrix with missing constraints.
    """
    m = Model()
    mask = pd.Series([False] * 5 + [True] * 5)
    x = m.add_variables(coords=[range(10)])
    m.add_variables()
    m.add_constraints(x, EQUAL, 0, mask=mask)
    # active cons are indices 5-9, which reference vars 5-9 only (all active)
    A, con_labels = m.constraints.to_matrix()
    n_active_vars = len(m.variables.label_index.vlabels)
    assert A.shape == (m.ncons, n_active_vars)
    assert len(con_labels) == m.ncons


def test_constraint_matrix_masked_constraints_and_variables() -> None:
    """
    Test constraint matrix with missing constraints and variables.
    """
    m = Model()
    mask = pd.Series([False] * 5 + [True] * 5)
    x = m.add_variables(coords=[range(10)], mask=mask)
    m.add_variables()
    m.add_constraints(x, EQUAL, 0, mask=mask)
    # both masks align: 5 active cons x all active vars (5 x + 1 scalar)
    A, con_labels = m.constraints.to_matrix()
    n_active_vars = len(m.variables.label_index.vlabels)
    assert A.shape == (m.ncons, n_active_vars)
    assert len(con_labels) == m.ncons


def test_get_name_by_label() -> None:
    m = Model()
    x = m.add_variables(coords=[range(10)])
    y = m.add_variables(coords=[range(10)])

    m.add_constraints(x + y <= 10, name="first")
    m.add_constraints(x - y >= 5, name="second")

    assert m.constraints.get_name_by_label(4) == "first"
    assert m.constraints.get_name_by_label(14) == "second"

    with pytest.raises(ValueError):
        m.constraints.get_name_by_label(30)

    with pytest.raises(ValueError):
        m.constraints.get_name_by_label("first")  # type: ignore


def test_constraints_inequalities(m: Model) -> None:
    assert isinstance(m.constraints.inequalities, Constraints)


def test_constraints_equalities(m: Model) -> None:
    assert isinstance(m.constraints.equalities, Constraints)


@pytest.mark.parametrize("masked", [False, True], ids=["all", "masked"])
def test_to_sparse_to_dense_roundtrip(masked: bool) -> None:
    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(5, name="i")], name="x")
    mask = xr.DataArray([True, False, True, False, True], dims=["i"])
    con = m.add_constraints((x.where(mask) if masked else 1 * x) >= 0, name="c")
    before = con.data
    assert con.to_sparse() is con
    assert con.lhs.is_sparse
    assert con.ncons == (3 if masked else 5)
    assert_equal(con.labels, before.labels)
    assert_equal(con.rhs, before.rhs)
    assert_equal(con.sign, before.sign)
    assert not con.to_dense().lhs.is_sparse
    assert_equal(con.data, before)


def test_coeff_dtype_preserved() -> None:
    m = Model()
    i = pd.RangeIndex(4, name="i")
    x = m.add_variables(coords=[i], name="x")
    coeff = xr.DataArray(np.arange(1, 5, dtype=np.float32), coords=[i])
    con = m.add_constraints(coeff * x >= 1, name="c")
    assert con.coeffs.dtype == np.float32


def test_sparse_to_matrix_has_active_variable_columns(
    m: Model, c: linopy.constraints.Constraint
) -> None:
    csr, _ = c.to_sparse().to_matrix(m.variables.label_index)
    assert csr.shape[1] == m.variables.label_index.n_active_vars


def _model_with_mutation(sparse: bool, mutation: str) -> Model:
    m = Model(sparse=sparse)
    i = pd.RangeIndex(3, name="i")
    m.add_variables(coords=[i], name="a")
    b = m.add_variables(coords=[i], name="b")
    m.add_constraints(2 * b >= 1, name="c1")
    d = m.add_variables(coords=[pd.RangeIndex(2, name="i")], name="d")
    m.add_constraints(d <= 5, name="c2")
    if mutation == "remove":
        m.remove_variables("a")
    return m


@pytest.mark.parametrize("mutation", ["add", "remove"])
def test_sparse_matrices_after_variable_mutation(mutation: str) -> None:
    sparse = _model_with_mutation(True, mutation)
    dense = _model_with_mutation(False, mutation)
    a_sparse, a_dense = sparse.matrices.A, dense.matrices.A
    assert a_sparse is not None and a_dense is not None
    np.testing.assert_array_equal(a_sparse.toarray(), a_dense.toarray())
    np.testing.assert_array_equal(sparse.matrices.b, dense.matrices.b)
    np.testing.assert_array_equal(sparse.matrices.clabels, dense.matrices.clabels)
    np.testing.assert_array_equal(sparse.matrices.vlabels, dense.matrices.vlabels)


@pytest.mark.parametrize("coeff", [1.0, 0.0], ids=["term", "zero-term"])
def test_constraint_removed_with_referenced_variable(
    backend: Model, coeff: float
) -> None:
    m = backend
    i = pd.RangeIndex(3, name="i")
    a = m.add_variables(coords=[i], name="a")
    b = m.add_variables(coords=[i], name="b")
    m.add_constraints(coeff * a + b >= 1, name="c")
    m.add_constraints(1 * b >= 0, name="d")
    with pytest.warns(UserWarning, match="also removes constraints"):
        m.remove_variables("a")
    assert "c" not in m.constraints
    assert m.matrices.A is not None


def test_to_sparse_keeps_mixed_signs() -> None:
    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(3, name="i")], name="x")
    con = m.add_constraints(x >= 0, name="mixed")
    con.update(sign=xr.DataArray(["<=", ">=", "<="], dims=["i"]))
    sign = con.sign
    con.to_sparse()
    assert_equal(con.sign, sign)
    _, _, _, sense = con.to_matrix_with_rhs(m.variables.label_index)
    assert list(sense) == ["<", ">", "<"]


def test_variable_label_index(m: Model) -> None:
    li = m.variables.label_index
    assert li.n_active_vars > 0
    assert len(li.vlabels) == li.n_active_vars
    assert li.label_to_pos.shape[0] == m._xCounter
    for lbl in li.vlabels:
        assert li.label_to_pos[lbl] >= 0
    assert (li.label_to_pos[li.vlabels] == np.arange(li.n_active_vars)).all()


def test_variable_label_index_invalidation(m: Model) -> None:
    li = m.variables.label_index
    old_vlabels = li.vlabels.copy()
    m.add_variables(name="w")
    li.invalidate()
    assert len(li.vlabels) > len(old_vlabels)


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_to_matrix_with_rhs(m: Model, sparse: bool) -> None:
    c = m.constraints["c"]
    if sparse:
        c.to_sparse()
    li = m.variables.label_index
    csr, con_labels, b, sense = c.to_matrix_with_rhs(li)
    assert csr.shape[0] == len(con_labels)
    assert csr.shape[0] == len(b)
    assert csr.shape[0] == len(sense)
    assert all(s in ("<", ">", "=") for s in sense)
    np.testing.assert_array_equal(b, c.rhs.values.ravel()[c.active_row_mask()])


def test_constraint_repr_shows_variable_names(m: Model) -> None:
    c = m.constraints["c"]
    r = repr(c)
    assert "x" in r


def test_sparse_mixed_signs_from_rule() -> None:
    m = Model(sparse=True)
    x = m.add_variables(coords=[pd.RangeIndex(4, name="i")], name="x")
    coords = [pd.RangeIndex(4, name="i")]

    def bound(m: Model, i: int) -> AnonymousScalarConstraint:
        if i % 2:
            return x.at[i] >= i
        return x.at[i] == 0.0

    con = m.add_constraints(bound, coords=coords, name="mixed_rule")
    assert con.lhs.is_sparse
    assert con.ncons == 4
    np.testing.assert_array_equal(con.sign.values, ["=", ">=", "=", ">="])


def test_mixed_sign_to_matrix_with_rhs() -> None:
    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(4, name="i")], name="x")
    coords = [pd.RangeIndex(4, name="i")]

    def bound(m: Model, i: int) -> AnonymousScalarConstraint:
        if i % 2:
            return x.at[i] >= i
        return x.at[i] == 0.0

    con = m.add_constraints(bound, coords=coords, name="c")
    li = m.variables.label_index
    csr, con_labels, b, sense = con.to_matrix_with_rhs(li)
    assert len(sense) == 4
    assert list(sense) == ["=", ">", "=", ">"]


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_mixed_sign_sanitize_infinities(sparse: bool) -> None:
    m = Model(sparse=sparse)
    x = m.add_variables(coords=[pd.RangeIndex(4, name="i")], name="x")
    mc = m.add_constraints(x >= 0, name="c")
    mc.update(
        sign=xr.DataArray(["<=", ">=", "<=", ">="], dims=["i"]),
        rhs=xr.DataArray([np.inf, -np.inf, 1.0, 2.0], dims=["i"]),
    )
    mc.sanitize_infinities()
    assert mc.lhs.is_sparse == sparse
    assert mc.ncons == 2
    np.testing.assert_array_equal(mc.rhs.values[mc.active_row_mask()], [1.0, 2.0])


def test_mixed_sign_repr() -> None:
    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(4, name="i")], name="x")
    coords = [pd.RangeIndex(4, name="i")]

    def bound(m: Model, i: int) -> AnonymousScalarConstraint:
        if i % 2:
            return x.at[i] >= i
        return x.at[i] == 0.0

    con = m.add_constraints(bound, coords=coords, name="c")
    r = repr(con)
    assert "≥" in r
    assert "=" in r


# Constraint.soften method's tests


def test_constraint_soften_returns_slack_for_le_and_ge(
    m: Model, y: linopy.Variable
) -> None:
    """
    Checks that a constraint of type 'less or equal' or 'great and equal' returns
    one slack each.
    """
    z = m.variables["z"]
    m.add_objective((z + y).sum(), sense="min")

    # create new constraint, and add slack for 'greater or equal'
    z_constraint = m.add_constraints(z >= -10, name="constraint_over_z")
    z_slack = z_constraint.soften(penalty=10)
    assert isinstance(z_slack.positive, linopy.Variable)
    assert z_slack.negative is None

    # create new constraint, and add slack for 'less or equal':
    y_constraint = m.add_constraints(y <= 0, name="constraint_over_y")
    y_slack = y_constraint.soften(penalty=10)
    assert isinstance(y_slack.positive, linopy.Variable)
    assert y_slack.negative is None


def test_constraint_soften_returns_slack_for_eq(m: Model, y: linopy.Variable) -> None:
    """Checks that a constraint of type 'equal'' returns two slack variables"""
    m.add_objective(y.sum(), sense="min")
    y_contraint = m.add_constraints(y == 10, name="equality_contraint")
    y_slack = y_contraint.soften(penalty=10)

    assert isinstance(y_slack.positive, linopy.Variable)
    assert isinstance(y_slack.negative, linopy.Variable)


def test_constraint_soften_raises_on_detached_constraint(
    m: Model, x: linopy.Variable, c: linopy.Constraint
) -> None:
    """
    A selection returns a detached copy that is not registered in
    `model.constraints`; softening it would silently fail to affect the
    actual model, so `soften` must raise instead.
    """
    m.add_objective(x.sum(), sense="min")

    with pytest.raises(ValueError, match="not the constraint registered"):
        c.isel(first=slice(None)).soften(penalty=10)


def test_constraint_soften_twice_raises(m: Model, y: linopy.Variable) -> None:
    """
    Softening an already-softened constraint must raise instead of silently
    stacking a second, redundant slack term onto the same lhs.
    """
    m.add_objective(y.sum(), sense="min")
    constraint = m.add_constraints(y >= 10, name="constraint")
    constraint.soften(penalty=10, name="first_slack")

    with pytest.raises(ValueError, match="already softened"):
        constraint.soften(penalty=5, name="second_slack")


def test_constraint_soften_twice_raises_via_add_constraints_penalty(
    m: Model, y: linopy.Variable
) -> None:
    """The same guard applies when the first soften came from `add_constraints(penalty=...)`."""
    m.add_objective(y.sum(), sense="min")
    constraint = m.add_constraints(y >= 10, name="constraint", penalty=10)

    with pytest.raises(ValueError, match="already softened"):
        constraint.soften(penalty=5, name="second")


def test_constraint_soften_updates_lhs(m: Model, y: linopy.Variable) -> None:
    """
    Tests that the left hand side gains the slack term(s) with the expected sign
    per constraint direction.
    """
    m.add_objective(y.sum(), sense="min")

    # '<=' : lhs -> lhs - positive_slack
    le_constraint = m.add_constraints(y <= 10, name="le_constraint")
    le_slack = le_constraint.soften(penalty=10)

    # Assert that the new lhs has 1 extra term:
    assert le_constraint.lhs.nterm == 2

    # Assert tht the label for the constraint's new variable is the same as the slack:
    assert_equal(
        le_constraint.vars.isel({le_constraint.term_dim: -1}),
        le_slack.positive.labels,
    )

    # Since the it's less or equal, we expect the coeff of the variable to be -1
    assert (le_constraint.coeffs.isel({le_constraint.term_dim: -1}) == -1).all()

    # '>=' : lhs -> lhs + positive_slack
    ge_constraint = m.add_constraints(y >= -10, name="ge_constraint")
    ge_slack = ge_constraint.soften(penalty=10)

    assert ge_constraint.lhs.nterm == 2

    assert_equal(
        ge_constraint.vars.isel({ge_constraint.term_dim: -1}),
        ge_slack.positive.labels,
    )
    assert (ge_constraint.coeffs.isel({ge_constraint.term_dim: -1}) == 1).all()

    # '==' : lhs -> lhs - positive_slack + negative_slack
    eq_constraint = m.add_constraints(y == 0, name="eq_constraint")
    eq_slack = eq_constraint.soften(penalty=10)
    assert eq_slack.negative is not None

    # Assert there's two extra terms:
    assert eq_constraint.lhs.nterm == 3

    # The positive slack is added first to the lhs, therefore, it should be on position
    # -2 of the lhs:
    assert_equal(
        eq_constraint.vars.isel({eq_constraint.term_dim: -2}),
        eq_slack.positive.labels,
    )

    # The negative slack is added next to the lhs, therefore, it should be on position
    # -1 of the lhs:
    assert_equal(
        eq_constraint.vars.isel({eq_constraint.term_dim: -1}),
        eq_slack.negative.labels,
    )

    assert (eq_constraint.coeffs.isel({eq_constraint.term_dim: -2}) == -1).all()
    assert (eq_constraint.coeffs.isel({eq_constraint.term_dim: -1}) == 1).all()


def test_constraint_soften_updates_objective_min_sense(
    m: Model, y: linopy.Variable
) -> None:
    penalty_coeff = 10
    original_objective = y.sum()
    m.add_objective(original_objective, sense="min")
    constraint = m.add_constraints(y >= 10, name="constraint")
    slack = constraint.soften(penalty=penalty_coeff)

    # For sense='min', the penalty must added with positive sign:
    expected_objective = original_objective + penalty_coeff * slack.positive.sum()
    assert_linequal(m.objective.expression, expected_objective)


def test_constraint_soften_updates_objective_max_sense(
    m: Model, y: linopy.Variable
) -> None:
    penalty_coeff = 10
    original_objective = y.sum()
    m.add_objective(original_objective, sense="max")
    constraint = m.add_constraints(y <= 10, name="constraint")
    slack = constraint.soften(penalty=penalty_coeff)

    # For sense='max', the penalty must added with negative sign:
    expected_objective = original_objective - penalty_coeff * slack.positive.sum()
    assert_linequal(m.objective.expression, expected_objective)


def test_constraint_soften_max_violation_bounds_slack(
    m: Model, x: linopy.Variable
) -> None:
    """max_violation sets the slack variable's upper bound; default is unbounded (inf)."""
    m.add_objective(x.sum(), sense="min")

    bounded_constraint = m.add_constraints(x >= 0, name="bounded_constraint")
    bounded_slack = bounded_constraint.soften(penalty=10, max_violation=5)
    assert (bounded_slack.positive.upper == 5).all()

    unbounded_constraint = m.add_constraints(x >= 0, name="unbounded_constraint")
    unbounded_slack = unbounded_constraint.soften(penalty=10)
    assert np.isinf(unbounded_slack.positive.upper).all()


def test_constraint_soften_negative_penalty_raises(
    m: Model, y: linopy.Variable
) -> None:
    """Setting penalty < 0 raises ValueError."""
    m.add_objective(y.sum(), sense="min")
    constraint = m.add_constraints(y >= 10, name="constraint")
    with pytest.raises(ValueError, match="Penalty is not positive"):
        constraint.soften(penalty=-10)


def test_constraint_soften_without_objective_raises(
    m: Model, y: linopy.Variable
) -> None:
    """Calling soften before model.add_objective raises ValueError."""
    constraint = m.add_constraints(y <= 10, name="constraint")
    with pytest.raises(ValueError, match="Objective must be defined"):
        constraint.soften(penalty=10)


def test_constraint_soften_respects_mask(m: Model, x: linopy.Variable) -> None:
    """Masked constraint entries produce masked slack variables (mask propagated)."""
    m.add_objective(x.sum(), sense="min")
    mask = pd.Series([False] * 5 + [True] * 5)
    constraint = m.add_constraints(x >= 0, name="masked_constraint", mask=mask)
    slack = constraint.soften(penalty=10)

    assert_equal(slack.positive.mask, constraint.mask)
    assert constraint.mask is not None
    assert (constraint.mask.values == mask.values).all()


def test_constraint_slack_none_before_soften(m: Model, x: linopy.Variable) -> None:
    """A constraint that was never softened exposes no slack variable."""
    constraint = m.add_constraints(x >= 0, name="constraint")
    assert constraint.slack is None


def test_constraint_slack_matches_returned_slack_for_le_and_ge(
    m: Model, x: linopy.Variable, y: linopy.Variable
) -> None:
    """
    `.slack` resolves to the same variables `soften` returned, for both
    inequality directions (no negative slack).
    """
    m.add_objective((x + y).sum(), sense="min")

    le_constraint = m.add_constraints(y <= 0, name="le_constraint")
    le_slack = le_constraint.soften(penalty=10)
    assert le_constraint.slack is not None
    assert_varequal(le_constraint.slack.positive, le_slack.positive)
    assert le_constraint.slack.negative is None

    ge_constraint = m.add_constraints(x >= -10, name="ge_constraint")
    ge_slack = ge_constraint.soften(penalty=10)
    assert ge_constraint.slack is not None
    assert_varequal(ge_constraint.slack.positive, ge_slack.positive)
    assert ge_constraint.slack.negative is None


def test_constraint_slack_matches_returned_slack_for_eq(
    m: Model, y: linopy.Variable
) -> None:
    """
    For an equality constraint, `.slack` carries both the positive and the
    negative slack variable.
    """
    m.add_objective(y.sum(), sense="min")
    constraint = m.add_constraints(y == 10, name="eq_constraint")
    slack = constraint.soften(penalty=10)

    resolved = constraint.slack
    assert resolved is not None
    assert_varequal(resolved.positive, slack.positive)
    assert slack.negative is not None
    assert resolved.negative is not None
    assert_varequal(resolved.negative, slack.negative)


def test_constraint_lhs_is_new_wrapper_on_held_store(
    m: Model, x: linopy.Variable
) -> None:
    con = m.add_constraints(x >= 0, name="d")
    assert con.lhs is not con.lhs
    assert con.lhs._store is con.lhs._store


def test_constraint_canonicalizes_sparse_lhs() -> None:
    m = Model(sparse=True)
    x = m.add_variables(coords=[pd.RangeIndex(3, name="i")], name="x")
    lhs = x + 2 * x
    assert lhs.is_sparse and lhs.nterm == 2
    con = Constraint(lhs, "<=", 1, m)
    store = con.lhs._store
    assert canonical(store.csr) is store.csr
    assert store.nterm == 1
    con.update(lhs=x - 3 * x)
    updated = con.lhs._store
    assert updated.nterm == 1
    assert (updated.csr.data == -2).all()


def test_constraint_data_roundtrips_through_from_dataset(
    m: Model, x: linopy.Variable
) -> None:
    con = m.add_constraints(2 * x >= 1, name="d")
    con.dual = xr.zeros_like(con.rhs)
    restored = Constraint.from_dataset(con.data, m, con.name)
    xr.testing.assert_identical(restored.data, con.data)
    assert restored.attrs == con.attrs


def test_from_dataset_rejects_unknown_field(m: Model, x: linopy.Variable) -> None:
    con = m.add_constraints(x >= 1, name="d")
    with pytest.raises(ValueError, match="unknown constraint fields"):
        Constraint.from_dataset(con.data.assign(foo=con.rhs), m)


def test_constraint_rejects_lhs_with_constant(m: Model, x: linopy.Variable) -> None:
    with pytest.raises(ValueError, match="zero constant"):
        Constraint(x + 1, "<=", 0, m)


def test_set_blocks_writes_blocks_field() -> None:
    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(3, name="i")], name="x")
    y = m.add_variables(coords=[pd.RangeIndex(3, name="i")], name="y")
    m.add_constraints(x >= 0, name="cx")
    m.add_constraints(x + y >= 0, name="cxy")
    m.constraints.set_blocks(np.array([1, 1, 1, 2, 2, 2]))
    assert m.constraints["cx"].data.blocks.values.tolist() == [1, 1, 1]
    assert m.constraints["cxy"].data.blocks.values.tolist() == [3, 3, 3]


@pytest.mark.parametrize(
    "rhs",
    [
        xr.DataArray(np.ones((10, 2)), dims=["first", "extra"]),
        xr.DataArray([5.0, 6.0], coords={"first": [0, 1]}),
        xr.DataArray(np.ones(10), coords={"first": np.arange(1, 11)}),
    ],
    ids=["extra-dim", "subset-coords", "shifted-coords"],
)
def test_constraint_rejects_field_off_lhs_grid(
    m: Model, x: linopy.Variable, rhs: xr.DataArray
) -> None:
    with pytest.raises(ValueError, match="Constraint rhs"):
        Constraint(1 * x, "<=", rhs, m)


def test_to_matrix_names_constraint_with_removed_variable(backend: Model) -> None:
    m = backend
    i = pd.RangeIndex(3, name="i")
    a = m.add_variables(coords=[i], name="a")
    b = m.add_variables(coords=[i], name="b")
    con = m.add_constraints(a + b >= 1, name="c")
    m.variables.remove("a")
    with pytest.raises(ValueError, match="Constraint 'c' references variable"):
        con.to_matrix(m.variables.label_index)
