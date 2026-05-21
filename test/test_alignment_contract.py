#!/usr/bin/env python3
"""
Alignment-contract tests.

Pin the *intended* contract for the coordinate-alignment bug cluster reported in
PyPSA/linopy issues #586, #706, #707, #708, and #709.

The contract this file locks in:

1. Binary operators (``+``, ``-``, ``*``, ``/``) between a ``Variable``/expression
   and a labelled coefficient (``pd.Series`` / ``xr.DataArray``) align operands
   **by label**, never by position. Algebraically identical expressions
   (``x - a <= 0`` vs ``x <= a``; ``x / a`` vs ``x * (1 / a)``) must therefore
   build identical constraints / expressions.

2. ``add_variables`` builds a variable whose dimension order matches ``coords``,
   regardless of the *type* of the ``lower`` / ``upper`` bounds and regardless
   of whether the bounds are missing some of the ``coords`` dimensions.
   pandas bounds with missing dimensions must be broadcast to the full
   ``coords`` shape, not silently dropped.

Each parameter case is marked ``xfail(strict=True)`` so it fires as long as the
bug reproduces; when the upstream fix lands the case starts passing, the strict
xfail turns into ``XPASS`` and CI fails, forcing the maintainer to remove the
xfail mark in the same commit that closes the issue. That is the right state
machine for a contract test.

The tests are deliberately parametrized over *labelled* coefficient / bound
containers (``pd.Series``, ``pd.DataFrame``, ``xr.DataArray``) and over a couple
of dtypes (``float64``, ``int64``) because each container reaches the alignment
code via a different conversion path in linopy. Unlabelled inputs (``np.ndarray``,
plain ``list``, scalar) are intentionally *not* parametrized: they have no
labels for the contract to align against, so positional behaviour is correct
for them and they would silently turn into XPASS noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import linopy
from linopy import Model

# ---------------------------------------------------------------------------
# Coefficient parametrisation (operator-alignment tests, #707, #708)
# ---------------------------------------------------------------------------
#
# Every case represents the *same* mapping ``{"A100": 100, "A1": 1, "A5": 5,
# "A11": 11}`` in a different container and dtype. The variable below is built
# in lexical order ``["A1", "A5", "A11", "A100"]``; the order mismatch is what
# distinguishes label-alignment from positional-alignment. ``coef["A1"] == 1``
# is the invariant the tests assert against — a positional-alignment bug pairs
# ``x[A1]`` with the container's first positional entry (``A100 → 100``).

_COEF_LABELS = ["A100", "A1", "A5", "A11"]
_COEF_VALUES = [100, 1, 5, 11]
_VAR_LABELS = ["A1", "A5", "A11", "A100"]


def _series(dtype: str) -> pd.Series:
    return pd.Series(np.asarray(_COEF_VALUES, dtype=dtype), index=_COEF_LABELS)


def _dataarray(dtype: str) -> xr.DataArray:
    return xr.DataArray(
        np.asarray(_COEF_VALUES, dtype=dtype),
        coords=[("dim_0", _COEF_LABELS)],
    )


COEF_CASES = [
    pytest.param(_series("float64"), id="series-float64"),
    pytest.param(_series("int64"), id="series-int64"),
    pytest.param(_dataarray("float64"), id="dataarray-float64"),
    pytest.param(_dataarray("int64"), id="dataarray-int64"),
]


@pytest.fixture
def x_unsorted() -> tuple[Model, linopy.Variable]:
    """Variable in lexical order — mismatched against ``_COEF_LABELS`` order."""
    m = Model()
    x = m.add_variables(coords=[_VAR_LABELS], name="x")
    return m, x


# ---------------------------------------------------------------------------
# Operator alignment (#707, #708)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="see PyPSA/linopy#707: `-` aligns positionally, `<=` aligns by label",
    strict=True,
)
@pytest.mark.parametrize("coef", COEF_CASES)
def test_subtract_then_le_matches_direct_le(
    x_unsorted: tuple[Model, linopy.Variable],
    coef: pd.Series | xr.DataArray,
) -> None:
    """Regression test for #707: ``x - a <= 0`` must equal ``x <= a``."""
    _, x = x_unsorted
    c1 = x - coef <= 0
    c2 = x <= coef
    # Same constraint either way: rhs at A1 must equal coef@A1 == 1.
    rhs1 = float(c1.rhs.sel(dim_0="A1").item())
    rhs2 = float(c2.rhs.sel(dim_0="A1").item())
    assert rhs1 == pytest.approx(1.0)
    assert rhs2 == pytest.approx(1.0)
    assert rhs1 == pytest.approx(rhs2)


@pytest.mark.xfail(
    reason="see PyPSA/linopy#708: `+` aligns positionally instead of by label",
    strict=True,
)
@pytest.mark.parametrize("coef", COEF_CASES)
def test_add_aligns_by_label(
    x_unsorted: tuple[Model, linopy.Variable],
    coef: pd.Series | xr.DataArray,
) -> None:
    """Regression test for #708: ``x + a`` must align by label."""
    _, x = x_unsorted
    const_at_A1 = float((x + coef).const.sel(dim_0="A1").item())
    # coef@A1 == 1, so x[A1] + coef must contribute +1 to the const term.
    assert const_at_A1 == pytest.approx(1.0)


@pytest.mark.xfail(
    reason="see PyPSA/linopy#708: `-` aligns positionally instead of by label",
    strict=True,
)
@pytest.mark.parametrize("coef", COEF_CASES)
def test_subtract_aligns_by_label(
    x_unsorted: tuple[Model, linopy.Variable],
    coef: pd.Series | xr.DataArray,
) -> None:
    """Regression test for #708: ``x - a`` must align by label."""
    _, x = x_unsorted
    const_at_A1 = float((x - coef).const.sel(dim_0="A1").item())
    assert const_at_A1 == pytest.approx(-1.0)


@pytest.mark.xfail(
    reason="see PyPSA/linopy#708: `/` aligns positionally instead of by label",
    strict=True,
)
@pytest.mark.parametrize("coef", COEF_CASES)
def test_divide_aligns_by_label(
    x_unsorted: tuple[Model, linopy.Variable],
    coef: pd.Series | xr.DataArray,
) -> None:
    """
    Regression test for #708: ``x / a`` must align by label.

    Also pins the algebraic identity ``x / a == x * (1 / a)`` — currently
    broken because ``*`` aligns by label and ``/`` aligns by position.
    The ``1 / coef`` term needs floating-point inversion, so int-dtype cases
    use the same algebra after the implicit cast.
    """
    _, x = x_unsorted
    inv = 1.0 / coef
    coeff_div = float((x / coef).coeffs.sel(dim_0="A1").values.ravel()[0])
    coeff_mul = float((x * inv).coeffs.sel(dim_0="A1").values.ravel()[0])
    # 1 / coef@A1 == 1, so coefficient on x[A1] must be 1.
    assert coeff_div == pytest.approx(1.0)
    assert coeff_div == pytest.approx(coeff_mul)


# ---------------------------------------------------------------------------
# add_variables dimension order / shape (#706, #709)
# ---------------------------------------------------------------------------
#
# Parametrise over the labelled bound containers that reach distinct code paths
# in ``add_variables`` — see ``_iter_inputs`` / ``as_dataarray`` in
# ``linopy.variables``. The contract: dim order follows ``coords`` and missing
# dims get broadcast, regardless of bound type or dtype.

_X = pd.Index(["a", "b", "c"], name="x")
_Y = pd.Index(["X", "Y"], name="y")


def _series_missing_y(dtype: str) -> pd.Series:
    return pd.Series(np.asarray([1, 2, 3], dtype=dtype), index=_X)


def _dataframe_missing_dim_in_index(dtype: str) -> pd.DataFrame:
    # DataFrame whose index covers "x" but columns are *not* "y" labels:
    # bound is effectively 1-D over "x", broadcast over "y" expected.
    return pd.DataFrame(
        np.asarray([[1], [2], [3]], dtype=dtype), index=_X, columns=["v"]
    )


def _dataarray_missing_y(dtype: str) -> xr.DataArray:
    return xr.DataArray(np.asarray([1, 2, 3], dtype=dtype), coords=[("x", list(_X))])


PANDAS_BOUND_CASES = [
    pytest.param(_series_missing_y("float64"), id="series-float64"),
    pytest.param(_series_missing_y("int64"), id="series-int64"),
    pytest.param(_dataframe_missing_dim_in_index("float64"), id="dataframe-float64"),
]

DATAARRAY_BOUND_CASES = [
    pytest.param(_dataarray_missing_y("float64"), id="dataarray-float64"),
    pytest.param(_dataarray_missing_y("int64"), id="dataarray-int64"),
]


@pytest.mark.xfail(
    reason="see PyPSA/linopy#706: DataArray bounds with missing dim get the dim prepended",
    strict=True,
)
@pytest.mark.parametrize("bound", DATAARRAY_BOUND_CASES)
def test_add_variables_dataarray_bounds_preserve_coords_order(
    bound: xr.DataArray,
) -> None:
    """Regression test for #706: dim order must follow ``coords``."""
    m = Model()
    v = m.add_variables(lower=-bound, upper=bound, coords=[_X, _Y], name="v")
    assert v.dims == ("x", "y")
    assert v.sizes == {"x": 3, "y": 2}


@pytest.mark.xfail(
    reason="see PyPSA/linopy#709: pandas bounds with missing dim are silently dropped",
    strict=True,
)
@pytest.mark.parametrize("bound", PANDAS_BOUND_CASES)
def test_add_variables_pandas_bounds_broadcast_missing_dim(
    bound: pd.Series | pd.DataFrame,
) -> None:
    """Regression test for #709: pandas bounds must broadcast against ``coords``."""
    m = Model()
    v = m.add_variables(lower=-bound, upper=bound, coords=[_X, _Y], name="v")
    assert v.dims == ("x", "y")
    assert v.sizes == {"x": 3, "y": 2}
