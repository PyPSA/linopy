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


def test_mapping_coords(m: Model) -> None:
    rows, vars_, coeffs = balance_triplets(m)
    expr = LinearExpression.from_coo(
        m, {"time": TIME, "bus": BUS.values}, rows, vars_, coeffs
    )
    assert_equivalent(expr, balance_arithmetic(m))


CELL = pd.RangeIndex(6, name="cell")


@pytest.mark.parametrize(
    ("rows", "pos", "coeffs"),
    [
        ([0, 1, 4], [0, 1, 2], [1.0, 2.0, 3.0]),
        ([0, 1], [0, -1], [1.0, 1.0]),  # cell 1 addressed only by a masked term
        ([0, 1], [0, -1], [1.0, np.nan]),
        ([0, 1], [0, 1], [1.0, 0.0]),
        ([2, 2, 2], [1, 1, 1], [1.0, 2.0, 0.5]),
        ([5, 0, 3, 0], [3, 0, 2, 1], [1.0, 1.0, -1.0, 2.0]),
        ([0, 0], [1, 2], [np.inf, -np.inf]),
    ],
    ids=["plain", "masked-only", "nan-on-masked", "zero", "dupes", "unsorted", "inf"],
)
def test_conforms_to_groupby_sum(
    m: Model, rows: list[int], pos: list[int], coeffs: list[float]
) -> None:
    labels = np.where(
        np.array(pos) >= 0, m.variables["p"].labels.values.ravel()[pos], -1
    )
    got = LinearExpression.from_coo(m, [CELL], rows, labels, coeffs)
    k = pd.RangeIndex(len(rows), name="k")
    one = xr.Dataset(
        {
            "vars": (("k", "_term"), labels[:, None]),
            "coeffs": (("k", "_term"), np.array(coeffs)[:, None]),
        },
        coords={"k": k},
    )
    want = (
        LinearExpression(one, m)
        .groupby(xr.DataArray(rows, coords=[k], name="cell"))
        .sum()
        .reindex(cell=CELL)
    )
    assert got.isnull().values.tolist() == want.isnull().values.tolist()
    assert_equivalent(got, want)
    x = m.add_variables(coords=[CELL], name="x")
    via_coo = m.add_constraints(got - x == 0, name="coo")
    via_ops = m.add_constraints(want - x == 0, name="ops")
    assert (via_coo.labels.values >= 0).tolist() == (
        via_ops.labels.values >= 0
    ).tolist()


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
        ({"rows": [0, 99, 1], "vars": [0, -1, 1]}, IndexError, r"\[0, 6\)"),
        ({"coeffs": [1.0, np.nan, 1.0]}, ValueError, "NaN found"),
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
