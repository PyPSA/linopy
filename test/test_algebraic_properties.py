"""
Algebraic properties of linopy arithmetic.

All standard algebraic laws should hold for linopy expressions.
This file serves as both specification and test suite.

Notation:
    x[A], y[A], z[A]  — linopy variables with dimension A
    g[A,B]             — linopy variable with dimensions A and B
    c[B]               — constant (DataArray) with dimension B
    s                  — scalar (int/float)

SPECIFICATION
=============

1. Commutativity
   a + b == b + a                     for any linopy operands a, b
   a * c == c * a                     for variable/expression a, constant c

2. Associativity
   (a + b) + c == a + (b + c)         for any linopy operands a, b, c
   Including mixed: (x[A] + c[B]) + g[A,B] == x[A] + (c[B] + g[A,B])

3. Distributivity
   c * (a + b) == c*a + c*b           for constant c, linopy operands a, b
   s * (a + b) == s*a + s*b           for scalar s

4. Identity
   a + 0 == a                         additive identity
   a * 1 == a                         multiplicative identity

5. Negation
   a - b == a + (-b)                  subtraction is addition of negation
   -(-a) == a                         double negation

6. Zero
   a * 0 == 0                         multiplication by zero
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
    def test_add_expr_expr(self, x: Variable, y: Variable) -> None:
        """X + y == y + x"""
        assert_linequal(x + y, y + x)

    def test_mul_expr_constant(self, g: Variable, c: xr.DataArray) -> None:
        """G * c == c * g"""
        assert_linequal(g * c, c * g)

    def test_add_expr_constant(self, g: Variable, c: xr.DataArray) -> None:
        """G + c == c + g"""
        assert_linequal(g + c, c + g)


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
# 7. NaN propagation
# ============================================================


class TestNaNPropagation:
    """Absent slots (from shift/where/reindex) propagate through bare operators."""

    def test_variable_add_scalar_propagates(self, x: Variable) -> None:
        """x.shift(1) + 5 keeps absent slot absent."""
        result = x.shift(time=1) + 5
        assert result.isnull().values[0]
        assert not result.isnull().values[1]

    def test_expression_add_scalar_propagates(self, x: Variable) -> None:
        """(1*x).shift(1) + 5 keeps absent slot absent."""
        result = (1 * x).shift(time=1) + 5
        assert result.isnull().values[0]
        assert not result.isnull().values[1]

    def test_variable_mul_scalar_propagates(self, x: Variable) -> None:
        """x.shift(1) * 3 keeps absent slot absent."""
        result = x.shift(time=1) * 3
        assert result.isnull().values[0]
        assert not result.isnull().values[1]

    def test_expression_mul_scalar_propagates(self, x: Variable) -> None:
        """(1*x).shift(1) * 3 keeps absent slot absent."""
        result = (1 * x).shift(time=1) * 3
        assert result.isnull().values[0]
        assert not result.isnull().values[1]

    def test_variable_and_expression_paths_consistent(self, x: Variable) -> None:
        """Variable and expression paths produce the same result."""
        var_result = x.shift(time=1) + 5
        expr_result = (1 * x).shift(time=1) + 5
        np.testing.assert_array_equal(
            var_result.isnull().values, expr_result.isnull().values
        )
        np.testing.assert_array_equal(var_result.const.values, expr_result.const.values)

    def test_add_zero_propagates(self, x: Variable) -> None:
        """x.shift(1) + 0 keeps absent slot absent (no implicit revival)."""
        result = x.shift(time=1) + 0
        assert result.isnull().values[0]

    def test_merge_all_absent_stays_absent(self, x: Variable, y: Variable) -> None:
        """x.shift(1) + y.shift(1) is absent where all terms are absent."""
        result = (1 * x).shift(time=1) + (1 * y).shift(time=1)
        assert result.isnull().values[0]
        assert not result.isnull().values[1]

    def test_merge_partial_absent_not_absent(self, x: Variable, y: Variable) -> None:
        """X + y.shift(1): valid term from x prevents coordinate from being absent."""
        result = x + (1 * y).shift(time=1)
        assert not result.isnull().any()

    def test_where_propagates(self, x: Variable) -> None:
        """Masked slots stay absent through arithmetic."""
        mask = xr.DataArray([True, False, True], dims=["time"])
        result = (1 * x).where(mask) + 10
        assert not result.isnull().values[0]
        assert result.isnull().values[1]
        assert not result.isnull().values[2]


# ============================================================
# 8. fillna
# ============================================================


class TestFillNA:
    """fillna revives absent slots with explicit values."""

    def test_variable_fillna_numeric_returns_expression(self, x: Variable) -> None:
        """Variable.fillna(numeric) returns a LinearExpression."""
        result = x.shift(time=1).fillna(0)
        assert isinstance(result, LinearExpression)

    def test_variable_fillna_revives_with_constant(self, x: Variable) -> None:
        """Variable.fillna(0) turns absent slot into a zero constant."""
        result = x.shift(time=1).fillna(0)
        assert not result.isnull().any()
        assert result.const.values[0] == 0

    def test_variable_fillna_custom_value(self, x: Variable) -> None:
        """Variable.fillna(42) fills absent slot with 42."""
        result = x.shift(time=1).fillna(42)
        assert result.const.values[0] == 42
        # Valid slots are unaffected
        assert result.const.values[1] == 0

    def test_expression_fillna_revives(self, x: Variable) -> None:
        """Expression.fillna(0) + 5 gives +5 at formerly absent slot."""
        result = (1 * x).shift(time=1).fillna(0) + 5
        assert not result.isnull().any()
        assert result.const.values[0] == 5

    def test_variable_fillna_variable_returns_variable(
        self, x: Variable, y: Variable
    ) -> None:
        """Variable.fillna(Variable) still returns a Variable."""
        result = x.shift(time=1).fillna(y)
        assert isinstance(result, Variable)

    def test_fillna_then_arithmetic(self, x: Variable) -> None:
        """fillna(0) + 5 and fillna(5) produce the same result."""
        a = (1 * x).shift(time=1).fillna(0) + 5
        b = (1 * x).shift(time=1).fillna(5)
        # At absent slot: both should give const=5
        assert a.const.values[0] == 5
        assert b.const.values[0] == 5


# ============================================================
# 9. Named methods with fill_value
# ============================================================


class TestFillValueParam:
    """Named methods (.add, .sub, .mul, .div) accept fill_value."""

    def test_add_fill_value(self, x: Variable) -> None:
        """expr.add(5, fill_value=0) revives absent slot."""
        expr = (1 * x).shift(time=1)
        result = expr.add(5, fill_value=0)
        assert not result.isnull().any()
        assert result.const.values[0] == 5

    def test_sub_fill_value(self, x: Variable) -> None:
        """expr.sub(5, fill_value=0) revives absent slot."""
        expr = (1 * x).shift(time=1)
        result = expr.sub(5, fill_value=0)
        assert not result.isnull().any()
        assert result.const.values[0] == -5

    def test_mul_fill_value(self, x: Variable) -> None:
        """expr.mul(3, fill_value=0) revives absent slot with 0."""
        expr = (1 * x).shift(time=1)
        result = expr.mul(3, fill_value=0)
        assert not result.isnull().any()
        assert result.const.values[0] == 0

    def test_div_fill_value(self, x: Variable) -> None:
        """expr.div(2, fill_value=0) revives absent slot with 0."""
        expr = (1 * x).shift(time=1)
        result = expr.div(2, fill_value=0)
        assert not result.isnull().any()
        assert result.const.values[0] == 0

    def test_add_without_fill_value_propagates(self, x: Variable) -> None:
        """expr.add(5) without fill_value still propagates NaN."""
        expr = (1 * x).shift(time=1)
        result = expr.add(5)
        assert result.isnull().values[0]

    def test_fill_value_only_affects_absent(self, x: Variable) -> None:
        """fill_value does not change valid slots."""
        expr = (1 * x).shift(time=1)
        result = expr.add(5, fill_value=0)
        # Valid slot: const should be 0 + 5 = 5
        assert result.const.values[1] == 5
        # Coefficients at valid slots unchanged
        assert result.coeffs.values[1, 0] == 1
