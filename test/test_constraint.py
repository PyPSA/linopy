#!/usr/bin/env python3
"""
Created on Tue Nov  2 22:38:48 2021.

@author: fabian
"""

from __future__ import annotations

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
    ConstraintBase,
    Constraints,
)


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
def c(m: Model) -> linopy.constraints.ConstraintBase:
    return m.constraints["c"]


@pytest.fixture
def mc(m: Model) -> linopy.constraints.MutableConstraint:
    return m.constraints["c"].mutable()


def test_constraint_repr(c: linopy.constraints.CSRConstraint) -> None:
    c.__repr__()


def test_constraint_repr_equivalent_to_mutable(
    c: linopy.constraints.CSRConstraint,
) -> None:
    """Constraint (CSR-backed) and MutableConstraint repr must be identical."""
    frozen = c.freeze()
    assert repr(frozen) == repr(c)


def test_constraints_repr(m: Model) -> None:
    m.constraints.__repr__()


def test_add_constraints_freeze(m: Model, x: linopy.Variable) -> None:
    c = m.add_constraints(x >= 1, name="frozen_c", freeze=True)
    assert isinstance(c, linopy.constraints.CSRConstraint)
    assert isinstance(m.constraints["frozen_c"], linopy.constraints.CSRConstraint)
    assert c.ncons == 10


def test_constraint_name(c: linopy.constraints.CSRConstraint) -> None:
    assert c.name == "c"


def test_empty_constraints_repr() -> None:
    # test empty contraints
    Model().constraints.__repr__()


def test_cannot_create_constraint_without_variable() -> None:
    model = linopy.Model()
    with pytest.raises(ValueError):
        _ = linopy.LinearExpression(12, model) == linopy.LinearExpression(13, model)


def test_constraints_getter(m: Model, c: linopy.constraints.CSRConstraint) -> None:
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


def test_constraint_wrapped_methods(x: linopy.Variable, y: linopy.Variable) -> None:
    con: Constraint = 10 * x + y <= 10

    # Test wrapped methods
    con.assign({"new_var": xr.DataArray(np.zeros((2, 2)), coords=[range(2), range(2)])})
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
    assert isinstance(con.sel(first=[1, 2]), ConstraintBase)


def test_anonymous_constraint_swap_dims(x: linopy.Variable, y: linopy.Variable) -> None:
    expr = 10 * x + y
    con = expr <= 10
    con = con.assign_coords({"third": ("second", con.indexes["second"] + 100)})
    con = con.swap_dims({"second": "third"})
    assert isinstance(con, ConstraintBase)
    assert con.coord_dims == ("first", "third")


def test_anonymous_constraint_set_index(x: linopy.Variable, y: linopy.Variable) -> None:
    expr = 10 * x + y
    con = expr <= 10
    con = con.assign_coords({"third": ("second", con.indexes["second"] + 100)})
    con = con.set_index({"multi": ["second", "third"]})
    assert isinstance(con, ConstraintBase)
    assert con.coord_dims == (
        "first",
        "multi",
    )
    assert isinstance(con.indexes["multi"], pd.MultiIndex)


def test_anonymous_constraint_loc(x: linopy.Variable, y: linopy.Variable) -> None:
    expr = 10 * x + y
    con = expr <= 10
    assert isinstance(con.loc[[1, 2]], ConstraintBase)


def test_anonymous_constraint_getitem(x: linopy.Variable, y: linopy.Variable) -> None:
    expr = 10 * x + y
    con = expr <= 10
    assert isinstance(con[1], ConstraintBase)


def test_constraint_from_rule(m: Model, x: linopy.Variable, y: linopy.Variable) -> None:
    def bound(m: Model, i: int, j: int) -> AnonymousScalarConstraint:
        return (i - 1) * x.at[i - 1] + y.at[j] >= 0 if i % 2 else i * x.at[i] >= 0

    coords = [x.coords["first"], y.coords["second"]]
    con = Constraint.from_rule(m, bound, coords)
    assert isinstance(con, ConstraintBase)
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
    assert isinstance(con, ConstraintBase)
    assert isinstance(con.lhs.vars, xr.DataArray)
    assert con.lhs.nterm == 2
    assert (con.lhs.vars.loc[0, :] == -1).all()
    assert (con.lhs.vars.loc[1, :] != -1).all()
    repr(con)  # test repr


def test_constraint_vars_getter(
    mc: linopy.constraints.MutableConstraint, x: linopy.Variable
) -> None:
    assert_equal(mc.vars.squeeze(), x.labels)


def test_constraint_coeffs_getter(mc: linopy.constraints.MutableConstraint) -> None:
    assert (mc.coeffs == 1).all()


def test_constraint_sign_getter(c: linopy.constraints.CSRConstraint) -> None:
    assert (c.sign == GREATER_EQUAL).all()


def test_constraint_rhs_getter(c: linopy.constraints.CSRConstraint) -> None:
    assert (c.rhs == 0).all()


def test_constraint_vars_setter(
    mc: linopy.constraints.MutableConstraint, x: linopy.Variable
) -> None:
    mc.vars = x
    assert_equal(mc.vars, x.labels)


def test_constraint_vars_setter_with_array(
    mc: linopy.constraints.MutableConstraint, x: linopy.Variable
) -> None:
    mc.vars = x.labels
    assert_equal(mc.vars, x.labels)


def test_constraint_vars_setter_invalid(
    mc: linopy.constraints.MutableConstraint, x: linopy.Variable
) -> None:
    with pytest.raises(TypeError):
        mc.vars = pd.DataFrame(x.labels)


def test_constraint_coeffs_setter(mc: linopy.constraints.MutableConstraint) -> None:
    mc.coeffs = 3
    assert (mc.coeffs == 3).all()


def test_constraint_lhs_setter(
    mc: linopy.constraints.MutableConstraint, x: linopy.Variable, y: linopy.Variable
) -> None:
    mc.lhs = x + y
    assert mc.lhs.nterm == 2
    assert mc.vars.notnull().all().item()
    assert mc.coeffs.notnull().all().item()


def test_constraint_lhs_setter_with_variable(
    mc: linopy.constraints.MutableConstraint, x: linopy.Variable
) -> None:
    mc.lhs = x
    assert mc.lhs.nterm == 1


def test_constraint_lhs_setter_with_constant(
    mc: linopy.constraints.MutableConstraint,
) -> None:
    sizes = mc.sizes
    mc.lhs = 10
    assert (mc.rhs == -10).all()
    assert mc.lhs.nterm == 0
    assert mc.sizes["first"] == sizes["first"]


def test_constraint_sign_setter(mc: linopy.constraints.MutableConstraint) -> None:
    mc.sign = EQUAL
    assert (mc.sign == EQUAL).all()


def test_constraint_sign_setter_alternative(
    mc: linopy.constraints.MutableConstraint,
) -> None:
    mc.sign = long_EQUAL
    assert (mc.sign == EQUAL).all()


def test_constraint_sign_setter_invalid(
    mc: linopy.constraints.MutableConstraint,
) -> None:
    # Test that assigning lhs with other type that LinearExpression raises TypeError
    with pytest.raises(ValueError):
        mc.sign = "asd"


def test_constraint_rhs_setter(mc: linopy.constraints.MutableConstraint) -> None:
    sizes = mc.sizes
    mc.rhs = 2  # type: ignore
    assert (mc.rhs == 2).all()
    assert mc.sizes == sizes


def test_constraint_rhs_setter_with_variable(
    mc: linopy.constraints.MutableConstraint, x: linopy.Variable
) -> None:
    mc.rhs = x  # type: ignore
    assert (mc.rhs == 0).all()
    assert (mc.coeffs.isel({mc.term_dim: -1}) == -1).all()
    assert mc.lhs.nterm == 2


def test_constraint_rhs_setter_with_expression(
    mc: linopy.constraints.MutableConstraint, x: linopy.Variable, y: linopy.Variable
) -> None:
    mc.rhs = x + y
    assert (mc.rhs == 0).all()
    assert (mc.coeffs.isel({mc.term_dim: -1}) == -1).all()
    assert mc.lhs.nterm == 3


def test_constraint_rhs_setter_with_expression_and_constant(
    mc: linopy.constraints.MutableConstraint, x: linopy.Variable
) -> None:
    mc.rhs = x + 1
    assert (mc.rhs == 1).all()
    assert (mc.coeffs.sum(mc.term_dim) == 0).all()
    assert mc.lhs.nterm == 2


def test_constraint_labels_setter_invalid(c: linopy.constraints.CSRConstraint) -> None:
    # Test that assigning labels raises AttributeError (Constraint is frozen)
    with pytest.raises(AttributeError):
        c.labels = c.labels  # type: ignore


def test_constraint_sel(c: linopy.constraints.CSRConstraint) -> None:
    assert isinstance(c.mutable().sel(first=[1, 2]), ConstraintBase)
    assert isinstance(c.mutable().isel(first=[1, 2]), ConstraintBase)


def test_constraint_flat(c: linopy.constraints.CSRConstraint) -> None:
    assert isinstance(c.flat, pd.DataFrame)


def test_iterate_slices(mc: linopy.constraints.MutableConstraint) -> None:
    for i in mc.iterate_slices(slice_size=2):
        assert isinstance(i, ConstraintBase)
        assert mc.coord_dims == i.coord_dims


def test_constraint_to_polars(c: linopy.constraints.CSRConstraint) -> None:
    assert isinstance(c.to_polars(), pl.DataFrame)


def test_constraint_to_polars_mixed_signs(m: Model, x: linopy.Variable) -> None:
    """Test to_polars when a constraint has mixed sign values across dims."""
    # Use MutableConstraint so sign data can be patched
    con = m.add_constraints(x >= 0, name="mixed", freeze=False)
    # Replace sign data with mixed signs across the first dimension
    n = con.data.sizes["first"]
    signs = np.array(["<=" if i % 2 == 0 else ">=" for i in range(n)])
    con.data["sign"] = xr.DataArray(signs, dims=con.data["sign"].dims)
    df = con.to_polars()
    assert isinstance(df, pl.DataFrame)
    assert set(df["sign"].to_list()) == {"<=", ">="}


def test_constraint_assignment_with_anonymous_constraints(
    m: Model, x: linopy.Variable, y: linopy.Variable
) -> None:
    m.add_constraints(x + y == 0, name="c2")
    assert m.constraints["c2"].vars.notnull().all()
    assert m.constraints["c2"].coeffs.notnull().all()


def test_constraint_assignment_sanitize_zeros(
    m: Model, x: linopy.Variable, y: linopy.Variable
) -> None:
    m.add_constraints(0 * x + y == 0, name="c2")
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
    con = Constraint(data, m, "c")

    expr = m.add_constraints(con)
    assert not set(HELPER_DIMS).intersection(set(expr.data.coords))


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


def test_freeze_mutable_roundtrip(m: Model) -> None:
    frozen = m.constraints["c"]
    assert isinstance(frozen, linopy.constraints.CSRConstraint)
    mc = frozen.mutable()
    assert isinstance(mc, Constraint)
    refrozen = linopy.constraints.CSRConstraint.from_mutable(mc, frozen._cindex)
    assert_equal(frozen.labels, refrozen.labels)
    assert_equal(frozen.rhs, refrozen.rhs)
    assert_equal(frozen.sign, refrozen.sign)
    np.testing.assert_array_equal(frozen._csr.toarray(), refrozen._csr.toarray())
    np.testing.assert_array_equal(frozen._con_labels, refrozen._con_labels)


def test_freeze_mutable_roundtrip_with_masking() -> None:
    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(5, name="i")], name="x")
    mask = xr.DataArray([True, False, True, False, True], dims=["i"])
    m.add_constraints(x.where(mask) >= 0, name="c")
    frozen = m.constraints["c"]
    mc = frozen.mutable()
    refrozen = linopy.constraints.CSRConstraint.from_mutable(mc, frozen._cindex)
    assert_equal(frozen.labels, refrozen.labels)
    assert_equal(frozen.rhs, refrozen.rhs)
    assert frozen.ncons == refrozen.ncons == 3


def test_from_mutable_mixed_signs() -> None:
    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(3, name="i")], name="x")
    m.add_constraints(x >= 0, name="mixed", freeze=False)
    mc = m.constraints["mixed"]
    assert isinstance(mc, Constraint)
    mc._data["sign"] = xr.DataArray(["<=", ">=", "<="], dims=["i"])
    frozen = linopy.constraints.CSRConstraint.from_mutable(mc)
    assert isinstance(frozen._sign, np.ndarray)
    assert list(frozen._sign) == ["<=", ">=", "<="]
    assert_equal(frozen.sign, mc.sign)


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


def test_to_matrix_with_rhs(m: Model) -> None:
    c = m.constraints["c"]
    li = m.variables.label_index
    csr, con_labels, b, sense = c.to_matrix_with_rhs(li)
    assert csr.shape[0] == len(con_labels)
    assert csr.shape[0] == len(b)
    assert csr.shape[0] == len(sense)
    assert all(s in ("<", ">", "=") for s in sense)
    np.testing.assert_array_equal(b, c._rhs)


def test_to_matrix_with_rhs_mutable(m: Model) -> None:
    mc = m.constraints["c"].mutable()
    li = m.variables.label_index
    csr, con_labels, b, sense = mc.to_matrix_with_rhs(li)
    assert csr.shape[0] == len(con_labels)
    assert csr.shape[0] == len(b)
    assert csr.shape[0] == len(sense)


def test_constraint_repr_shows_variable_names(m: Model) -> None:
    c = m.constraints["c"]
    r = repr(c)
    assert "x" in r


def test_freeze_mixed_signs_from_rule() -> None:
    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(4, name="i")], name="x")
    coords = [pd.RangeIndex(4, name="i")]

    def bound(m, i):
        if i % 2:
            return x.at[i] >= i
        return x.at[i] == 0.0

    con = m.add_constraints(bound, coords=coords, name="mixed_rule")
    assert isinstance(con, linopy.constraints.CSRConstraint)
    assert isinstance(con._sign, np.ndarray)
    assert con.ncons == 4
    expected_signs = ["=", ">=", "=", ">="]
    assert list(con._sign) == expected_signs
    np.testing.assert_array_equal(con.sign.values, expected_signs)


def test_frozen_rhs_setter() -> None:
    m = Model()
    time = pd.RangeIndex(5, name="t")
    x = m.add_variables(lower=0, coords=[time], name="x")
    con = m.add_constraints(x >= 1, name="c")
    assert isinstance(con, linopy.constraints.CSRConstraint)
    con.rhs = 10
    np.testing.assert_array_equal(con._rhs, np.full(5, 10.0))
    factor = pd.Series(range(5), index=time)
    con.rhs = 2 * factor
    np.testing.assert_array_equal(con._rhs, 2 * np.arange(5, dtype=float))


def test_frozen_lhs_setter() -> None:
    m = Model()
    time = pd.RangeIndex(5, name="t")
    x = m.add_variables(lower=0, coords=[time], name="x")
    y = m.add_variables(lower=0, coords=[time], name="y")
    con = m.add_constraints(x >= 0, name="c")
    assert isinstance(con, linopy.constraints.CSRConstraint)
    con.lhs = 3 * x + 2 * y
    lhs = con.mutable().lhs
    assert lhs.nterm == 2


def test_frozen_setter_invalidates_dual() -> None:
    m = Model()
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(3, name="i")], name="x")
    con = m.add_constraints(x >= 0, name="c")
    con._dual = np.array([1.0, 2.0, 3.0])
    con.rhs = 10
    assert con._dual is None


def test_mixed_sign_to_matrix_with_rhs() -> None:
    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(4, name="i")], name="x")
    coords = [pd.RangeIndex(4, name="i")]

    def bound(m, i):
        if i % 2:
            return x.at[i] >= i
        return x.at[i] == 0.0

    con = m.add_constraints(bound, coords=coords, name="c")
    li = m.variables.label_index
    csr, con_labels, b, sense = con.to_matrix_with_rhs(li)
    assert len(sense) == 4
    assert list(sense) == ["=", ">", "=", ">"]


def test_mixed_sign_sanitize_infinities() -> None:
    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(4, name="i")], name="x")
    m.add_constraints(x >= 0, name="c", freeze=False)
    mc = m.constraints["c"]
    mc._data["sign"] = xr.DataArray(["<=", ">=", "<=", ">="], dims=["i"])
    mc._data["rhs"] = xr.DataArray([np.inf, -np.inf, 1.0, 2.0], dims=["i"])
    frozen = mc.freeze()
    frozen.sanitize_infinities()
    assert frozen.ncons == 2
    np.testing.assert_array_equal(frozen._rhs, [1.0, 2.0])


def test_mixed_sign_repr() -> None:
    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(4, name="i")], name="x")
    coords = [pd.RangeIndex(4, name="i")]

    def bound(m, i):
        if i % 2:
            return x.at[i] >= i
        return x.at[i] == 0.0

    con = m.add_constraints(bound, coords=coords, name="c")
    r = repr(con)
    assert "≥" in r
    assert "=" in r
