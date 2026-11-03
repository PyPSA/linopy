"""Tests for :meth:`linopy.LinearExpression.from_coo`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import LinearExpression, Model
from linopy.testing import assert_equivalent

TIME = pd.RangeIndex(3, name="time")
BUS = pd.Index(["a", "b"], name="bus")
GEN = pd.Index(["g1", "g2", "g3"], name="gen")
GEN_BUS = np.array([0, 1, 1])  # bus position of each generator
LINE = pd.Index(["l1"], name="line")  # from bus a to bus b


@pytest.fixture(params=[False, True], ids=["dense", "sparse"])
def m(request: pytest.FixtureRequest) -> Model:
    m = Model(sparse=request.param)
    m.add_variables(0, 10, coords=[TIME, GEN], name="p")
    m.add_variables(-5, 5, coords=[TIME, LINE], name="f")
    return m


def balance_triplets(m: Model) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nodal balance ``sum(p at bus) - f out + f in`` as COO triplets."""
    nb = len(BUS)
    t = np.arange(len(TIME))[:, None]
    p = m.variables["p"].labels.transpose("time", "gen").values
    f = m.variables["f"].labels.transpose("time", "line").values
    rows = [(t * nb + GEN_BUS).ravel(), (t * nb + 0).ravel(), (t * nb + 1).ravel()]
    vars_ = [p.ravel(), f.ravel(), f.ravel()]
    coeffs = [np.ones(p.size), -np.ones(f.size), np.ones(f.size)]
    return tuple(map(np.concatenate, (rows, vars_, coeffs)))  # type: ignore[return-value]


def balance_arithmetic(m: Model) -> LinearExpression:
    p, f = m.variables["p"], m.variables["f"]
    gen = p.groupby(xr.DataArray(BUS[GEN_BUS], coords=[GEN], name="bus")).sum()
    out = (-f).groupby(xr.DataArray(["a"], coords=[LINE], name="bus")).sum()
    inn = f.groupby(xr.DataArray(["b"], coords=[LINE], name="bus")).sum()
    return gen.add(out, join="outer").add(inn, join="outer").reindex(bus=BUS)


def test_matches_arithmetic(m: Model) -> None:
    rows, vars_, coeffs = balance_triplets(m)
    expr = LinearExpression.from_coo(m, [TIME, BUS], rows, vars_, coeffs)
    assert expr.is_sparse is m.sparse
    assert isinstance(expr._store, m.store_cls)
    assert_equivalent(expr, balance_arithmetic(m))


def test_unsorted_rows(m: Model) -> None:
    rows, vars_, coeffs = balance_triplets(m)
    order = np.random.default_rng(0).permutation(rows.size)
    expr = LinearExpression.from_coo(
        m, [TIME, BUS], rows[order], vars_[order], coeffs[order]
    )
    assert_equivalent(expr, balance_arithmetic(m))


def test_mapping_coords(m: Model) -> None:
    rows, vars_, coeffs = balance_triplets(m)
    expr = LinearExpression.from_coo(
        m, {"time": TIME, "bus": BUS.values}, rows, vars_, coeffs
    )
    assert_equivalent(expr, balance_arithmetic(m))


def test_duplicates_sum(m: Model) -> None:
    p = m.variables["p"]
    label = int(p.labels.values[0, 0])
    expr = LinearExpression.from_coo(
        m, [TIME], [0, 0, 0], [label, label, label], [1.0, 2.0, 0.5]
    )
    want = (3.5 * p.isel(gen=0, drop=True)).where(xr.DataArray([1, 0, 0], [TIME]) > 0)
    assert_equivalent(expr, want)


def test_masked_labels_and_empty_cells(m: Model) -> None:
    p = m.variables["p"]
    labels = p.labels.isel(gen=0, drop=True).values
    expr = LinearExpression.from_coo(
        m, [TIME], [0, 99, 2], [labels[0], -1, labels[2]], [1.0, np.nan, 1.0]
    )
    mask = xr.DataArray([True, False, True], [TIME])
    assert_equivalent(expr, p.isel(gen=0, drop=True).to_linexpr().where(mask))
    assert expr.const.isnull().values.tolist() == [False, True, False]


def test_no_terms(m: Model) -> None:
    expr = LinearExpression.from_coo(m, [TIME, BUS], [], [], [])
    assert expr.const.shape == (3, 2)
    assert expr.const.isnull().values.all()


def test_scalar_grid(m: Model) -> None:
    p = m.variables["p"]
    labels = p.labels.values[0, :2]
    expr = LinearExpression.from_coo(m, [], [0, 0], labels, [1.0, 2.0])
    assert expr.const.dims == ()
    assert_equivalent(
        expr, p.isel(time=0, gen=0, drop=True) + 2 * p.isel(time=0, gen=1, drop=True)
    )


def test_does_not_share_input_memory(m: Model) -> None:
    rows, vars_, coeffs = balance_triplets(m)
    expr = LinearExpression.from_coo(m, [TIME, BUS], rows, vars_, coeffs)
    coeffs[:] = 99.0
    assert_equivalent(expr, balance_arithmetic(m))


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"rows": [0, 1]}, ValueError, "same length"),
        ({"rows": [0, 6, 1]}, IndexError, r"\[0, 6\)"),
        ({"rows": [0, -1, 1]}, IndexError, r"\[0, 6\)"),
        ({"rows": [0.0, 1.0, 2.0]}, TypeError, "`rows` must hold integers"),
        ({"vars": [0, 999, 1]}, ValueError, "variable labels"),
        ({"vars": [0, -2, 1]}, ValueError, "variable labels"),
        ({"coeffs": [1.0, np.nan, 1.0]}, ValueError, "`coeffs` must be finite"),
        ({"coeffs": [1.0, np.inf, 1.0]}, ValueError, "`coeffs` must be finite"),
        ({"coeffs": ["a", "b", "c"]}, ValueError, "convert"),
        ({"coords": [TIME, TIME]}, ValueError, "repeat"),
        ({"coords": [[0, 1, 2], BUS]}, ValueError, "name every dimension"),
        (
            {"coords": {"uv": pd.MultiIndex.from_product([[0], [1, 2]])}},
            ValueError,
            "MultiIndex",
        ),
    ],
)
def test_errors(
    m: Model, kwargs: dict[str, object], error: type[Exception], match: str
) -> None:
    args: dict[str, object] = {
        "coords": [TIME, BUS],
        "rows": [0, 1, 2],
        "vars": [0, 1, 2],
        "coeffs": [1.0, 1.0, 1.0],
    }
    with pytest.raises(error, match=match):
        LinearExpression.from_coo(m, **(args | kwargs))  # type: ignore[arg-type]


def test_constraint(m: Model) -> None:
    rows, vars_, coeffs = balance_triplets(m)
    lhs = LinearExpression.from_coo(m, [TIME, BUS], rows, vars_, coeffs)
    via_coo = m.add_constraints(lhs == 3, name="coo")
    via_ops = m.add_constraints(balance_arithmetic(m) == 3, name="ops")
    assert_equivalent(via_coo.lhs, via_ops.lhs)
    xr.testing.assert_equal(via_coo.rhs, via_ops.rhs)
