import pandas as pd
import pytest

from linopy import Model
from linopy.testing import assert_exprequal, assert_linequal, assert_model_equal


@pytest.fixture
def model() -> Model:
    return Model()


def test_assert_linequal_ignores_dimension_order(model: Model) -> None:
    """
    Commutative arithmetic yields different dimension orders (``x + y`` gives
    ``(i, j)`` while ``y + x`` gives ``(j, i)``), inherited from xarray
    broadcasting. That is not semantically meaningful, so ``assert_linequal``
    must treat the two as equal.
    """
    a = model.add_variables(coords=[pd.Index([0, 1], name="i")], name="a")
    b = model.add_variables(coords=[pd.Index([0, 1, 2], name="j")], name="b")

    assert (a + b).data.coeffs.dims != (b + a).data.coeffs.dims
    assert_linequal(a + b, b + a)
    assert_linequal(2 * a + 3 * b, 3 * b + 2 * a)


def test_assert_linequal_still_detects_real_differences(model: Model) -> None:
    """Aligning dimension order must not mask genuinely unequal expressions."""
    a = model.add_variables(coords=[pd.Index([0, 1], name="i")], name="a")
    c = model.add_variables(coords=[pd.Index([0, 1], name="k")], name="c")

    with pytest.raises(AssertionError):
        assert_linequal(1 * a, 1 * c)  # different dimension sets
    with pytest.raises(AssertionError):
        assert_linequal(1 * a, 2 * a)  # different coefficients


def test_assert_exprequal_detects_type_mismatch(model: Model) -> None:
    """A linear and a quadratic expression must never compare equal."""
    a = model.add_variables(coords=[pd.Index([0, 1], name="i")], name="a")
    b = model.add_variables(coords=[pd.Index([0, 1], name="i")], name="b")

    with pytest.raises(AssertionError, match="expression types differ"):
        assert_exprequal(a + 1, a * b)


def test_assert_exprequal_detects_name_mismatch(model: Model) -> None:
    """Expressions with identical values but different stored names differ."""
    a = model.add_variables(coords=[pd.Index([0, 1], name="i")], name="a")

    lhs = model.add_expressions(a + 1, name="first")
    rhs = model.add_expressions(a + 1, name="second")

    with pytest.raises(AssertionError, match="expression names differ"):
        assert_exprequal(lhs, rhs)

    # names deliberately ignored
    assert_exprequal(lhs, rhs, check_name=False)


def test_assert_model_equal_detects_expression_difference() -> None:
    """assert_model_equal must fail when expressions differ between models."""
    m1 = Model()
    a1 = m1.add_variables(coords=[pd.Index([0, 1], name="i")], name="a")
    m1.add_expressions(a1 + 1, name="expr")
    m1.add_objective(a1.sum())

    m2 = Model()
    a2 = m2.add_variables(coords=[pd.Index([0, 1], name="i")], name="a")
    m2.add_expressions(a2 + 2, name="expr")  # different coefficients
    m2.add_objective(a2.sum())

    with pytest.raises(AssertionError):
        assert_model_equal(m1, m2)

    m3 = Model()
    a3 = m3.add_variables(coords=[pd.Index([0, 1], name="i")], name="a")
    m3.add_objective(a3.sum())  # no "expr" at all

    with pytest.raises(AssertionError):
        assert_model_equal(m1, m3)
