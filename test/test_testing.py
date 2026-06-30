import pandas as pd
import pytest

from linopy import Model
from linopy.testing import assert_linequal


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
