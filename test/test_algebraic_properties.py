"""
Tests for algebraic properties of the arithmetic convention.

Properties that hold are tested normally.
Properties that break (by design) are marked with xfail to document
the known limitation and detect if a future change fixes them.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model
from linopy.expressions import LinearExpression


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
    return m.add_variables(lower=0, coords=[time], name="x")


@pytest.fixture
def y(m, time):
    return m.add_variables(lower=0, coords=[time], name="y")


@pytest.fixture
def z(m, time):
    return m.add_variables(lower=0, coords=[time], name="z")


@pytest.fixture
def g(m, time, tech):
    return m.add_variables(lower=0, coords=[time, tech], name="g")


@pytest.fixture
def c(tech):
    """Constant DataArray with dims not in x but in g."""
    return xr.DataArray([2.0, 3.0], dims=["tech"], coords={"tech": tech})


def assert_linequal(a: LinearExpression, b: LinearExpression) -> None:
    """Assert two linear expressions are equivalent (same terms, same const)."""
    assert set(a.dims) == set(b.dims)
    for dim in a.dims:
        if dim.startswith("_"):
            continue
        np.testing.assert_array_equal(
            sorted(a.coords[dim].values), sorted(b.coords[dim].values)
        )
    assert a.const.sum().item() == pytest.approx(b.const.sum().item())


# ============================================================
# Properties that hold
# ============================================================


class TestPropertiesThatHold:
    def test_commutativity_addition(self, x, y):
        """X + y == y + x"""
        assert_linequal(x + y, y + x)

    def test_commutativity_multiplication(self, g, c):
        """G * c == c * g"""
        assert_linequal(g * c, c * g)

    def test_associativity_addition_same_dims(self, x, y, z):
        """(x + y) + z == x + (y + z)"""
        assert_linequal((x + y) + z, x + (y + z))

    def test_additive_identity(self, x):
        """X + 0 == x"""
        result = x + 0
        assert isinstance(result, LinearExpression)
        assert (result.const == 0).all()
        np.testing.assert_array_equal(result.coeffs.squeeze().values, [1, 1, 1])

    def test_multiplicative_identity(self, x):
        """X * 1 == x"""
        result = x * 1
        assert isinstance(result, LinearExpression)
        np.testing.assert_array_equal(result.coeffs.squeeze().values, [1, 1, 1])

    def test_negation(self, x, y):
        """X - y == x + (-y)"""
        assert_linequal(x - y, x + (-y))

    def test_scalar_distributivity(self, x, y):
        """S * (x + y) == s*x + s*y"""
        assert_linequal(3 * (x + y), 3 * x + 3 * y)

    def test_constant_distributivity_subset_dims(self, g, c):
        """c[B] * (g + g) == c*g + c*g  (c has subset dims of g)"""
        assert_linequal(c * (g + g), c * g + c * g)

    def test_subtraction_definition(self, x, y):
        """X - y == x + (-1 * y)"""
        assert_linequal(x - y, x + (-1) * y)

    def test_multiplication_by_zero(self, x):
        """X * 0 has zero coefficients"""
        result = x * 0
        assert (result.coeffs == 0).all()

    def test_double_negation(self, x):
        """-(-x) has same coefficients as x"""
        result = -(-x)
        np.testing.assert_array_equal(
            result.coeffs.squeeze().values,
            (1 * x).coeffs.squeeze().values,
        )


# ============================================================
# Properties that break (by design)
# ============================================================


class TestPropertiesThatBreak:
    @pytest.mark.xfail(
        reason="Rule 2: (x[A] + c[B]) raises because c introduces dim B into x",
        strict=True,
    )
    def test_associativity_with_constant(self, x, g, c):
        """
        (x[A] + c[B]) + g[A,B] should equal x[A] + (c[B] + g[A,B])

        Currently: left grouping raises, right grouping works.
        """
        lhs = (x + c) + g
        rhs = x + (c + g)
        assert_linequal(lhs, rhs)

    @pytest.mark.xfail(
        reason="Rule 2: c[B]*x[A] raises because c introduces dim B into x",
        strict=True,
    )
    def test_distributivity_with_constant(self, x, g, c):
        """
        c[B] * (x[A] + g[A,B]) should equal c[B]*x[A] + c[B]*g[A,B]

        Currently: undistributed form works, distributed form raises.
        """
        lhs = c * (x + g)
        rhs = c * x + c * g
        assert_linequal(lhs, rhs)

    def test_associativity_right_grouping_works(self, x, g, c):
        """x[A] + (c[B] + g[A,B]) works — the valid grouping."""
        result = x + (c + g)
        assert isinstance(result, LinearExpression)
        assert "time" in result.dims
        assert "tech" in result.dims

    def test_distributivity_undistributed_works(self, x, g, c):
        """c[B] * (x[A] + g[A,B]) works — apply constant to combined expr."""
        result = c * (x + g)
        assert isinstance(result, LinearExpression)
        assert "time" in result.dims
        assert "tech" in result.dims
