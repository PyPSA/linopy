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

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import linopy
from linopy import Model
from linopy.expressions import LinearExpression


@pytest.fixture(autouse=True)
def _use_exact_join():
    """Use exact arithmetic join for all tests in this module."""
    linopy.options["arithmetic_convention"] = "v1"
    yield
    linopy.options["arithmetic_convention"] = "legacy"


@pytest.fixture
def m():
    return Model()


@pytest.fixture
def time():
    return pd.RangeIndex(3, name="time")


@pytest.fixture
def tech():
    return pd.Index(["solar", "wind"], name="tech")


@pytest.fixture
def x(m, time):
    """Variable with dims [time]."""
    return m.add_variables(lower=0, coords=[time], name="x")


@pytest.fixture
def y(m, time):
    """Variable with dims [time]."""
    return m.add_variables(lower=0, coords=[time], name="y")


@pytest.fixture
def z(m, time):
    """Variable with dims [time]."""
    return m.add_variables(lower=0, coords=[time], name="z")


@pytest.fixture
def g(m, time, tech):
    """Variable with dims [time, tech]."""
    return m.add_variables(lower=0, coords=[time, tech], name="g")


@pytest.fixture
def c(tech):
    """Constant (DataArray) with dims [tech]."""
    return xr.DataArray([2.0, 3.0], dims=["tech"], coords={"tech": tech})


def assert_linequal(a: LinearExpression, b: LinearExpression) -> None:
    """Assert two linear expressions are algebraically equivalent."""
    assert set(a.dims) == set(b.dims), f"dims differ: {a.dims} vs {b.dims}"
    for dim in a.dims:
        if dim.startswith("_"):
            continue
        np.testing.assert_array_equal(
            sorted(a.coords[dim].values), sorted(b.coords[dim].values)
        )
    assert a.const.sum().item() == pytest.approx(b.const.sum().item())


# ============================================================
# 1. Commutativity
# ============================================================


class TestCommutativity:
    def test_add_expr_expr(self, x, y):
        """X + y == y + x"""
        assert_linequal(x + y, y + x)

    def test_mul_expr_constant(self, g, c):
        """G * c == c * g"""
        assert_linequal(g * c, c * g)

    def test_add_expr_constant(self, g, c):
        """G + c == c + g"""
        assert_linequal(g + c, c + g)


# ============================================================
# 2. Associativity
# ============================================================


class TestAssociativity:
    def test_add_same_dims(self, x, y, z):
        """(x + y) + z == x + (y + z)"""
        assert_linequal((x + y) + z, x + (y + z))

    def test_add_with_constant(self, x, g, c):
        """(x[A] + c[B]) + g[A,B] == x[A] + (c[B] + g[A,B])"""
        assert_linequal((x + c) + g, x + (c + g))


# ============================================================
# 3. Distributivity
# ============================================================


class TestDistributivity:
    def test_scalar(self, x, y):
        """S * (x + y) == s*x + s*y"""
        assert_linequal(3 * (x + y), 3 * x + 3 * y)

    def test_constant_subset_dims(self, g, c):
        """c[B] * (g[A,B] + g[A,B]) == c*g + c*g"""
        assert_linequal(c * (g + g), c * g + c * g)

    def test_constant_mixed_dims(self, x, g, c):
        """c[B] * (x[A] + g[A,B]) == c*x + c*g"""
        assert_linequal(c * (x + g), c * x + c * g)


# ============================================================
# 4. Identity
# ============================================================


class TestIdentity:
    def test_additive(self, x):
        """X + 0 == x"""
        result = x + 0
        assert isinstance(result, LinearExpression)
        assert (result.const == 0).all()
        np.testing.assert_array_equal(result.coeffs.squeeze().values, [1, 1, 1])

    def test_multiplicative(self, x):
        """X * 1 == x"""
        result = x * 1
        assert isinstance(result, LinearExpression)
        np.testing.assert_array_equal(result.coeffs.squeeze().values, [1, 1, 1])


# ============================================================
# 5. Negation
# ============================================================


class TestNegation:
    def test_subtraction_is_add_negation(self, x, y):
        """X - y == x + (-y)"""
        assert_linequal(x - y, x + (-y))

    def test_subtraction_definition(self, x, y):
        """X - y == x + (-1) * y"""
        assert_linequal(x - y, x + (-1) * y)

    def test_double_negation(self, x):
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
    def test_multiplication_by_zero(self, x):
        """X * 0 has zero coefficients"""
        result = x * 0
        assert (result.coeffs == 0).all()
