"""
Algebraic properties of linopy arithmetic.

All standard algebraic laws should hold for linopy expressions,
including in the presence of absent slots (NaN from shift/where/reindex).

This file serves as both specification and test suite.

Notation:
    x[A], y[A], z[A]  — linopy variables with dimension A
    g[A,B]             — linopy variable with dimensions A and B
    c[B]               — constant (DataArray) with dimension B
    s                  — scalar (int/float)
    xs                 — x.shift(time=1), variable with absent slot

SPECIFICATION
=============

1. Commutativity
   a + b == b + a                     for any linopy operands a, b
   a * c == c * a                     for variable/expression a, constant c

2. Associativity
   (a + b) + c == a + (b + c)         for any linopy operands a, b, c
   Including with absent slots: (xs + s) + y == xs + (s + y)

3. Distributivity
   c * (a + b) == c*a + c*b           for constant c, linopy operands a, b
   Including with absent slots: s * (xs + c) == s*xs + s*c

4. Identity
   a + 0 == a                         additive identity
   a * 1 == a                         multiplicative identity

5. Negation
   a - b == a + (-b)                  subtraction is addition of negation
   -(-a) == a                         double negation

6. Zero
   a * 0 == 0                         multiplication by zero

7. NaN / absent slot behavior
   Addition uses additive identity (0) to fill NaN const:
     xs + s  revives absent slot with const=s
     xs - s  revives absent slot with const=-s
   Multiplication propagates NaN:
     xs * s  keeps absent slot absent
     xs / s  keeps absent slot absent
   Merge (expression + expression):
     xs + y       — absent x term doesn't poison valid y term
     xs + ys      — fully absent when ALL terms absent
   Variable and expression paths are consistent.

8. fillna
   Variable.fillna(numeric) returns LinearExpression
   Expression.fillna(value) fills const at absent slots

9. Named methods with fill_value
   .add(v, fill_value=f)  fills const before adding
   .mul(v, fill_value=f)  fills const before multiplying
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model
from linopy.expressions import LinearExpression
from linopy.variables import Variable

pytestmark = pytest.mark.v1_only


@pytest.fixture
def m() -> Model:
    return Model()


@pytest.fixture
def time() -> pd.RangeIndex:
    return pd.RangeIndex(3, name="time")


@pytest.fixture
def tech() -> pd.Index:
    return pd.Index(["solar", "wind"], name="tech")


@pytest.fixture
def x(m: Model, time: pd.RangeIndex) -> Variable:
    """Variable with dims [time]."""
    return m.add_variables(lower=0, coords=[time], name="x")


@pytest.fixture
def y(m: Model, time: pd.RangeIndex) -> Variable:
    """Variable with dims [time]."""
    return m.add_variables(lower=0, coords=[time], name="y")


@pytest.fixture
def z(m: Model, time: pd.RangeIndex) -> Variable:
    """Variable with dims [time]."""
    return m.add_variables(lower=0, coords=[time], name="z")


@pytest.fixture
def g(m: Model, time: pd.RangeIndex, tech: pd.Index) -> Variable:
    """Variable with dims [time, tech]."""
    return m.add_variables(lower=0, coords=[time, tech], name="g")


@pytest.fixture
def c(tech: pd.Index) -> xr.DataArray:
    """Constant (DataArray) with dims [tech]."""
    return xr.DataArray([2.0, 3.0], dims=["tech"], coords={"tech": tech})


def assert_linequal(a: LinearExpression, b: LinearExpression) -> None:
    """
    Assert two linear expressions are algebraically equivalent.

    Checks dimensions, coordinates, coefficients, variable references, and constants.
    """
    assert set(a.dims) == set(b.dims), f"dims differ: {a.dims} vs {b.dims}"
    for dim in a.dims:
        if isinstance(dim, str) and dim.startswith("_"):
            continue
        np.testing.assert_array_equal(
            sorted(a.coords[dim].values), sorted(b.coords[dim].values)
        )
    # Simplify both to canonical form for coefficient/variable comparison
    a_s = a.simplify()
    b_s = b.simplify()
    assert a_s.nterm == b_s.nterm, f"nterm differs: {a_s.nterm} vs {b_s.nterm}"
    np.testing.assert_array_almost_equal(
        np.sort(a_s.coeffs.values, axis=None),
        np.sort(b_s.coeffs.values, axis=None),
        err_msg="coefficients differ",
    )
    np.testing.assert_array_equal(
        np.sort(a_s.vars.values, axis=None),
        np.sort(b_s.vars.values, axis=None),
    )
    np.testing.assert_array_almost_equal(
        a.const.values, b.const.values, err_msg="constants differ"
    )


# ============================================================
# 1. Commutativity
# ============================================================


class TestCommutativity:
    def test_add_var_var(self, x: Variable, y: Variable) -> None:
        """X + y == y + x"""
        assert_linequal(x + y, y + x)

    def test_mul_var_constant(self, g: Variable, c: xr.DataArray) -> None:
        """G * c == c * g"""
        assert_linequal(g * c, c * g)

    def test_add_var_constant(self, g: Variable, c: xr.DataArray) -> None:
        """G + c == c + g"""
        assert_linequal(g + c, c + g)

    def test_add_var_scalar(self, x: Variable) -> None:
        """X + 5 == 5 + x"""
        assert_linequal(x + 5, 5 + x)

    def test_mul_var_scalar(self, x: Variable) -> None:
        """X * 3 == 3 * x"""
        assert_linequal(x * 3, 3 * x)


# ============================================================
# 2. Associativity
# ============================================================


class TestAssociativity:
    def test_add_same_dims(self, x: Variable, y: Variable, z: Variable) -> None:
        """(x + y) + z == x + (y + z)"""
        assert_linequal((x + y) + z, x + (y + z))

    def test_add_with_constant(self, x: Variable, g: Variable, c: xr.DataArray) -> None:
        """(x[A] + c[B]) + g[A,B] == x[A] + (c[B] + g[A,B])"""
        assert_linequal((x + c) + g, x + (c + g))

    def test_add_shifted_scalar_var(self, x: Variable, y: Variable) -> None:
        """(x.shift(1) + 5) + y == x.shift(1) + (5 + y)"""
        lhs = (x.shift(time=1) + 5) + y
        rhs = x.shift(time=1) + (5 + y)
        assert_linequal(lhs, rhs)

    def test_add_shifted_scalar_var_reordered(self, x: Variable, y: Variable) -> None:
        """(x.shift(1) + y) + 5 == x.shift(1) + (y + 5)"""
        lhs = (x.shift(time=1) + y) + 5
        rhs = x.shift(time=1) + (y + 5)
        assert_linequal(lhs, rhs)

    def test_add_three_scalars_shifted(self, x: Variable) -> None:
        """(x.shift(1) + 3) + 7 == x.shift(1) + 10"""
        lhs = (x.shift(time=1) + 3) + 7
        rhs = x.shift(time=1) + 10
        assert_linequal(lhs, rhs)

    def test_sub_shifted_scalar_var(self, x: Variable, y: Variable) -> None:
        """(x.shift(1) - 5) + y == x.shift(1) + (y - 5)"""
        lhs = (x.shift(time=1) - 5) + y
        rhs = x.shift(time=1) + (y - 5)
        assert_linequal(lhs, rhs)


# ============================================================
# 3. Distributivity
# ============================================================


class TestDistributivity:
    def test_scalar(self, x: Variable, y: Variable) -> None:
        """S * (x + y) == s*x + s*y"""
        assert_linequal(3 * (x + y), 3 * x + 3 * y)

    def test_constant_subset_dims(self, g: Variable, c: xr.DataArray) -> None:
        """c[B] * (g[A,B] + g[A,B]) == c*g + c*g"""
        assert_linequal(c * (g + g), c * g + c * g)

    def test_constant_mixed_dims(
        self, x: Variable, g: Variable, c: xr.DataArray
    ) -> None:
        """c[B] * (x[A] + g[A,B]) == c*x + c*g"""
        assert_linequal(c * (x + g), c * x + c * g)

    def test_scalar_shifted_add_constant(self, x: Variable) -> None:
        """3 * (x.shift(1) + 5) == 3*x.shift(1) + 15"""
        lhs = 3 * (x.shift(time=1) + 5)
        rhs = 3 * x.shift(time=1) + 15
        assert_linequal(lhs, rhs)

    def test_scalar_shifted_add_var(self, x: Variable, y: Variable) -> None:
        """3 * (x.shift(1) + y) == 3*x.shift(1) + 3*y"""
        lhs = 3 * (x.shift(time=1) + y)
        rhs = 3 * x.shift(time=1) + 3 * y
        assert_linequal(lhs, rhs)


# ============================================================
# 4. Identity
# ============================================================


class TestIdentity:
    def test_additive(self, x: Variable) -> None:
        """X + 0 == x"""
        result = x + 0
        assert isinstance(result, LinearExpression)
        assert (result.const == 0).all()
        np.testing.assert_array_equal(result.coeffs.squeeze().values, [1, 1, 1])

    def test_multiplicative(self, x: Variable) -> None:
        """X * 1 == x"""
        result = x * 1
        assert isinstance(result, LinearExpression)
        np.testing.assert_array_equal(result.coeffs.squeeze().values, [1, 1, 1])

    def test_additive_shifted(self, x: Variable) -> None:
        """x.shift(1) + 0 revives absent slot as zero expression."""
        result = x.shift(time=1) + 0
        assert not result.isnull().values[0]
        assert result.const.values[0] == 0


# ============================================================
# 5. Negation
# ============================================================


class TestNegation:
    def test_subtraction_is_add_negation(self, x: Variable, y: Variable) -> None:
        """X - y == x + (-y)"""
        assert_linequal(x - y, x + (-y))

    def test_subtraction_definition(self, x: Variable, y: Variable) -> None:
        """X - y == x + (-1) * y"""
        assert_linequal(x - y, x + (-1) * y)

    def test_double_negation(self, x: Variable) -> None:
        """-(-x) has same coefficients as x"""
        result = -(-x)
        np.testing.assert_array_equal(
            result.coeffs.squeeze().values,
            (1 * x).coeffs.squeeze().values,
        )


# ============================================================
# 6. Zero
# ============================================================


class TestZero:
    def test_multiplication_by_zero(self, x: Variable) -> None:
        """X * 0 has zero coefficients"""
        result = x * 0
        assert (result.coeffs == 0).all()


# ============================================================
# 7. NaN / absent slot behavior
# ============================================================


class TestAbsentSlotAddition:
    """Addition fills const with 0 (additive identity) → revives absent slots."""

    def test_add_scalar_revives(self, x: Variable) -> None:
        result = x.shift(time=1) + 5
        assert not result.isnull().values[0]
        assert result.const.values[0] == 5

    def test_add_array_revives(self, x: Variable) -> None:
        arr = xr.DataArray([10.0, 20.0, 30.0], dims=["time"])
        result = (1 * x).shift(time=1) + arr
        assert not result.isnull().values[0]
        assert result.const.values[0] == 10.0

    def test_sub_scalar_revives(self, x: Variable) -> None:
        result = x.shift(time=1) - 5
        assert not result.isnull().values[0]
        assert result.const.values[0] == -5

    def test_add_zero_revives(self, x: Variable) -> None:
        """+ 0 revives to a zero expression (not absent)."""
        result = x.shift(time=1) + 0
        assert not result.isnull().values[0]
        assert result.const.values[0] == 0

    def test_variable_and_expression_paths_consistent_add(self, x: Variable) -> None:
        var_result = x.shift(time=1) + 5
        expr_result = (1 * x).shift(time=1) + 5
        np.testing.assert_array_equal(
            var_result.isnull().values, expr_result.isnull().values
        )
        np.testing.assert_array_equal(var_result.const.values, expr_result.const.values)


class TestAbsentSlotMultiplication:
    """Multiplication propagates NaN → absent stays absent."""

    def test_mul_scalar_propagates(self, x: Variable) -> None:
        result = x.shift(time=1) * 3
        assert result.isnull().values[0]
        assert not result.isnull().values[1]

    def test_mul_array_propagates(self, x: Variable) -> None:
        arr = xr.DataArray([2.0, 2.0, 2.0], dims=["time"])
        result = (1 * x).shift(time=1) * arr
        assert result.isnull().values[0]

    def test_div_scalar_propagates(self, x: Variable) -> None:
        result = (1 * x).shift(time=1) / 2
        assert result.isnull().values[0]

    def test_variable_and_expression_paths_consistent_mul(self, x: Variable) -> None:
        var_result = x.shift(time=1) * 3
        expr_result = (1 * x).shift(time=1) * 3
        np.testing.assert_array_equal(
            var_result.isnull().values, expr_result.isnull().values
        )


class TestAbsentSlotMerge:
    """Merging expressions: absent terms don't poison valid terms."""

    def test_partial_absent(self, x: Variable, y: Variable) -> None:
        """X + y.shift(1): x is valid everywhere → no absent slots."""
        result = x + (1 * y).shift(time=1)
        assert not result.isnull().any()

    def test_all_absent(self, x: Variable, y: Variable) -> None:
        """x.shift(1) + y.shift(1): all terms absent at time=0 → absent."""
        result = (1 * x).shift(time=1) + (1 * y).shift(time=1)
        assert result.isnull().values[0]
        assert not result.isnull().values[1]

    def test_shifted_const_lost(self, x: Variable, y: Variable) -> None:
        """X + (y+5).shift(1): shifted constant is lost at the gap."""
        result = x + (1 * y + 5).shift(time=1)
        # time=0: only x's const (0), shifted 5 is lost
        assert result.const.values[0] == 0
        # time=1: both consts survive (0 + 5 = 5)
        assert result.const.values[1] == 5


class TestAbsentSlotMixed:
    """Combined add/mul with absent slots."""

    def test_add_then_mul(self, x: Variable) -> None:
        """(x.shift(1) + 5) * 3 → +15 at absent slot."""
        result = (x.shift(time=1) + 5) * 3
        assert not result.isnull().values[0]
        assert result.const.values[0] == 15

    def test_mul_then_add(self, x: Variable) -> None:
        """x.shift(1) * 3 + 5 → +5 at absent slot."""
        result = x.shift(time=1) * 3 + 5
        assert not result.isnull().values[0]
        assert result.const.values[0] == 5

    def test_where_add_revives(self, x: Variable) -> None:
        mask = xr.DataArray([True, False, True], dims=["time"])
        result = (1 * x).where(mask) + 10
        assert not result.isnull().any()
        assert result.const.values[1] == 10

    def test_where_mul_propagates(self, x: Variable) -> None:
        mask = xr.DataArray([True, False, True], dims=["time"])
        result = (1 * x).where(mask) * 3
        assert not result.isnull().values[0]
        assert result.isnull().values[1]
        assert not result.isnull().values[2]


# ============================================================
# 8. fillna
# ============================================================


class TestFillNA:
    """fillna revives absent slots with explicit values."""

    def test_variable_fillna_numeric_returns_expression(self, x: Variable) -> None:
        result = x.shift(time=1).fillna(0)
        assert isinstance(result, LinearExpression)

    def test_variable_fillna_revives(self, x: Variable) -> None:
        result = x.shift(time=1).fillna(0)
        assert not result.isnull().any()
        assert result.const.values[0] == 0

    def test_variable_fillna_custom_value(self, x: Variable) -> None:
        result = x.shift(time=1).fillna(42)
        assert result.const.values[0] == 42
        assert result.const.values[1] == 0  # valid slots unaffected

    def test_expression_fillna_revives(self, x: Variable) -> None:
        result = (1 * x).shift(time=1).fillna(0) + 5
        assert not result.isnull().any()
        assert result.const.values[0] == 5

    def test_variable_fillna_variable_returns_variable(
        self, x: Variable, y: Variable
    ) -> None:
        result = x.shift(time=1).fillna(y)
        assert isinstance(result, Variable)

    def test_fillna_then_add_equals_fillna_sum(self, x: Variable) -> None:
        """fillna(0) + 5 == fillna(5) at absent slots."""
        a = (1 * x).shift(time=1).fillna(0) + 5
        b = (1 * x).shift(time=1).fillna(5)
        assert a.const.values[0] == 5
        assert b.const.values[0] == 5


# ============================================================
# 9. Named methods with fill_value
# ============================================================


class TestFillValueParam:
    """Named methods (.add, .sub, .mul, .div) accept fill_value."""

    def test_add_fill_value(self, x: Variable) -> None:
        expr = (1 * x).shift(time=1)
        result = expr.add(5, fill_value=0)
        assert not result.isnull().any()
        assert result.const.values[0] == 5

    def test_sub_fill_value(self, x: Variable) -> None:
        expr = (1 * x).shift(time=1)
        result = expr.sub(5, fill_value=0)
        assert not result.isnull().any()
        assert result.const.values[0] == -5

    def test_mul_fill_value(self, x: Variable) -> None:
        expr = (1 * x).shift(time=1)
        result = expr.mul(3, fill_value=0)
        assert not result.isnull().any()
        assert result.const.values[0] == 0

    def test_div_fill_value(self, x: Variable) -> None:
        expr = (1 * x).shift(time=1)
        result = expr.div(2, fill_value=0)
        assert not result.isnull().any()
        assert result.const.values[0] == 0

    def test_add_without_fill_value_still_revives(self, x: Variable) -> None:
        """add() always fills const with 0 (additive identity)."""
        expr = (1 * x).shift(time=1)
        result = expr.add(5)
        assert not result.isnull().values[0]
        assert result.const.values[0] == 5

    def test_fill_value_only_affects_absent(self, x: Variable) -> None:
        expr = (1 * x).shift(time=1)
        result = expr.add(5, fill_value=0)
        assert result.const.values[1] == 5  # valid slot: 0 + 5
        assert result.coeffs.values[1, 0] == 1  # coeff unchanged


# ============================================================
# 10. Division with absent slots
# ============================================================


class TestDivisionAbsentSlots:
    """Division propagates NaN same as multiplication."""

    def test_div_scalar_propagates(self, x: Variable) -> None:
        result = (1 * x).shift(time=1) / 2
        assert result.isnull().values[0]
        assert not result.isnull().values[1]

    def test_div_array_propagates(self, x: Variable) -> None:
        arr = xr.DataArray([2.0, 2.0, 2.0], dims=["time"])
        result = (1 * x).shift(time=1) / arr
        assert result.isnull().values[0]

    def test_div_consistent_paths(self, x: Variable) -> None:
        """Variable and expression paths give same result for division."""
        var_result = x.shift(time=1) / 2
        expr_result = (1 * x).shift(time=1) / 2
        assert_linequal(var_result, expr_result)

    def test_div_equals_mul_reciprocal(self, x: Variable) -> None:
        """Shifted / 2 == shifted * 0.5"""
        shifted = (1 * x).shift(time=1)
        assert_linequal(shifted / 2, shifted * 0.5)


# ============================================================
# 11. Subtraction with absent slots
# ============================================================


class TestSubtractionAbsentSlots:
    """Subtraction with shifted coords preserves associativity."""

    def test_sub_scalar_revives(self, x: Variable) -> None:
        result = x.shift(time=1) - 5
        assert not result.isnull().values[0]
        assert result.const.values[0] == -5

    def test_sub_associativity_shifted(self, x: Variable, y: Variable) -> None:
        """(x.shift(1) - 5) + y == x.shift(1) + (y - 5)"""
        xs = x.shift(time=1)
        assert_linequal((xs - 5) + y, xs + (y - 5))

    def test_sub_equals_add_neg_shifted(self, x: Variable) -> None:
        """x.shift(1) - 5 == x.shift(1) + (-5)"""
        xs = x.shift(time=1)
        assert_linequal(xs - 5, xs + (-5))


# ============================================================
# 12. Multi-dimensional absent slots
# ============================================================


@pytest.fixture
def g2(m: Model, time: pd.RangeIndex, tech: pd.Index) -> Variable:
    """Second variable with dims [time, tech]."""
    return m.add_variables(lower=0, coords=[time, tech], name="g2")


class TestMultiDimensionalAbsentSlots:
    """2D variables with shift: add revives, mul propagates."""

    def test_2d_add_revives(self, g: Variable) -> None:
        shifted = (1 * g).shift(time=1)
        result = shifted + 5
        assert not result.isnull().isel(time=0).any()
        assert (result.const.isel(time=0) == 5).all()

    def test_2d_mul_propagates(self, g: Variable) -> None:
        shifted = (1 * g).shift(time=1)
        result = shifted * 3
        assert result.isnull().isel(time=0).all()

    def test_2d_associativity(self, g: Variable, g2: Variable) -> None:
        """(g.shift(1) + g2) + 5 == g.shift(1) + (g2 + 5) in 2D."""
        gs = g.shift(time=1)
        assert_linequal((gs + g2) + 5, gs + (g2 + 5))

    def test_2d_distributivity(self, g: Variable, g2: Variable) -> None:
        """2 * (g.shift(1) + g2) == 2*g.shift(1) + 2*g2 in 2D."""
        gs = g.shift(time=1)
        assert_linequal(2 * (gs + g2), 2 * gs + 2 * g2)
