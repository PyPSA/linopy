#!/usr/bin/env python3
"""
Created on Wed Mar 10 11:23:13 2021.

@author: fabulous
"""

from typing import Any

import dask
import dask.array.core
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.core.types import JoinOptions

from linopy import EQUAL, GREATER_EQUAL, LESS_EQUAL, Model, available_solvers
from linopy.testing import assert_conequal

T = pd.Index([0, 1, 2], name="t")

# Test model functions


def test_constraint_assignment() -> None:
    m: Model = Model()

    lower: xr.DataArray = xr.DataArray(
        np.zeros((10, 10)), coords=[range(10), range(10)]
    )
    upper: xr.DataArray = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(name="y")

    con0 = m.add_constraints(1 * x + 10 * y, EQUAL, 0)

    for attr in m.constraints.dataset_attrs:
        assert "con0" in getattr(m.constraints, attr)

    assert m.constraints.labels.con0.shape == (10, 10)
    assert np.issubdtype(m.constraints.labels.con0.dtype, np.integer)
    assert np.issubdtype(m.constraints.coeffs.con0.dtype, np.number)
    assert np.issubdtype(m.constraints.vars.con0.dtype, np.number)
    assert np.issubdtype(m.constraints.rhs.con0.dtype, np.number)

    assert_conequal(m.constraints.con0, con0)


def test_constraint_equality() -> None:
    m: Model = Model()

    lower: xr.DataArray = xr.DataArray(
        np.zeros((10, 10)), coords=[range(10), range(10)]
    )
    upper: xr.DataArray = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(name="y")

    con0 = m.add_constraints(1 * x + 10 * y, EQUAL, 0)

    assert_conequal(con0, 1 * x + 10 * y == 0, strict=False)
    assert_conequal(1 * x + 10 * y == 0, 1 * x + 10 * y == 0, strict=False)

    with pytest.raises(AssertionError):
        assert_conequal(con0, 1 * x + 10 * y <= 0, strict=False)

    with pytest.raises(AssertionError):
        assert_conequal(con0, 1 * x + 10 * y >= 0, strict=False)

    with pytest.raises(AssertionError):
        assert_conequal(10 * y + 2 * x == 0, 1 * x + 10 * y == 0, strict=False)


def test_constraints_getattr_formatted() -> None:
    m: Model = Model()
    x = m.add_variables(0, 10, name="x")
    m.add_constraints(1 * x == 0, name="con-0")
    assert_conequal(m.constraints.con_0, m.constraints["con-0"])


def test_anonymous_constraint_assignment() -> None:
    m: Model = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(name="y")
    con = 1 * x + 10 * y == 0
    m.add_constraints(con)

    for attr in m.constraints.dataset_attrs:
        assert "con0" in getattr(m.constraints, attr)

    assert m.constraints.labels.con0.shape == (10, 10)
    assert np.issubdtype(m.constraints.labels.con0.dtype, np.integer)
    assert np.issubdtype(m.constraints.coeffs.con0.dtype, np.number)
    assert np.issubdtype(m.constraints.vars.con0.dtype, np.number)
    assert np.issubdtype(m.constraints.rhs.con0.dtype, np.number)


def test_constraint_assignment_with_tuples() -> None:
    m: Model = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper)
    y = m.add_variables()

    m.add_constraints([(1, x), (10, y)], EQUAL, 0, name="c")
    for attr in m.constraints.dataset_attrs:
        assert "c" in getattr(m.constraints, attr)
    assert m.constraints.labels.c.shape == (10, 10)


def test_constraint_assignment_chunked() -> None:
    # setting bounds with one pd.DataFrame and one pd.Series
    m: Model = Model(chunk=5)
    lower = pd.DataFrame(np.zeros((10, 10)))
    upper = pd.Series(np.ones(10))
    x = m.add_variables(lower, upper)
    m.add_constraints(x, GREATER_EQUAL, 0, name="c")
    assert m.constraints.coeffs.c.data.shape == (
        10,
        10,
        1,
    )
    assert isinstance(m.constraints.coeffs.c.data, dask.array.core.Array)


def test_constraint_assignment_with_reindex() -> None:
    m: Model = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(name="y")

    m.add_constraints(1 * x + 10 * y, EQUAL, 0)

    shuffled_coords = [2, 1, 3, 4, 6, 5, 7, 9, 8, 0]

    con = x.loc[shuffled_coords] + y >= 10
    assert (con.coords["dim_0"].values == shuffled_coords).all()


@pytest.mark.parametrize(
    "rhs_factory",
    [
        pytest.param(lambda m, v: xr.DataArray(v, dims=["dim_0"]), id="dataarray"),
        pytest.param(lambda m, v: pd.Series(v, index=v), id="series"),
        pytest.param(
            lambda m, v: m.add_variables(coords=[v]),
            id="variable",
        ),
        pytest.param(
            lambda m, v: 2 * m.add_variables(coords=[v]) + 1,
            id="linexpr",
        ),
    ],
)
def test_constraint_rhs_lower_dim(rhs_factory: Any) -> None:
    m = Model()
    naxis = np.arange(10, dtype=float)
    maxis = np.arange(10).astype(str)
    x = m.add_variables(coords=[naxis, maxis])
    y = m.add_variables(coords=[naxis, maxis])

    c = m.add_constraints(x - y >= rhs_factory(m, naxis))
    assert c.shape == (10, 10)


def test_constraint_rhs_unlabeled_lower_dim_ambiguous_raises(backend: Model) -> None:
    """Both dims are size 10, so a length-10 rhs cannot be paired by size."""
    m = backend
    naxis = np.arange(10, dtype=float)
    maxis = np.arange(10).astype(str)
    x = m.add_variables(coords=[naxis, maxis])
    y = m.add_variables(coords=[naxis, maxis])

    with pytest.raises(ValueError, match=r"sizes alone cannot decide"):
        m.add_constraints(x - y >= naxis)


@pytest.mark.parametrize("join", [None, "outer"])
@pytest.mark.parametrize(
    "rhs",
    [
        pytest.param(0.0, id="scalar"),
        pytest.param(xr.DataArray([0.0, 0.0, 0.0], coords=[T]), id="dataarray"),
        pytest.param(pd.Series([0.0, 0.0, 0.0], index=T), id="series"),
    ],
)
def test_constraint_zero_rhs_is_unsigned(
    backend: Model, rhs: Any, join: JoinOptions | None
) -> None:
    x = backend.add_variables(coords=[T])
    c = (2 * x).to_constraint(LESS_EQUAL, rhs, join=join)
    assert not np.signbit(c.rhs.values).any()


@pytest.mark.parametrize(
    "rhs_factory",
    [
        pytest.param(lambda m: pd.DataFrame(np.ones((5, 3))), id="dataframe"),
    ],
)
def test_constraint_rhs_higher_dim_constant_warns(
    rhs_factory: Any, caplog: Any
) -> None:
    m = Model()
    x = m.add_variables(coords=[range(5)], name="x")

    with caplog.at_level("WARNING", logger="linopy.expressions"):
        m.add_constraints(x >= rhs_factory(m))
    assert "dimensions" in caplog.text


def test_constraint_rhs_unlabeled_higher_dim_raises() -> None:
    """The (5, 3) rhs has an axis of length 3 matching no dim of x."""
    m = Model()
    x = m.add_variables(coords=[range(5)], name="x")

    with pytest.raises(ValueError, match=r"no unambiguous dimension match"):
        m.add_constraints(x >= np.ones((5, 3)))


def test_constraint_rhs_higher_dim_dataarray_reindexes() -> None:
    """DataArray RHS with extra dims reindexes to expression coords (no raise)."""
    m = Model()
    x = m.add_variables(coords=[range(5)], name="x")
    rhs = xr.DataArray(np.ones((5, 3)), dims=["dim_0", "extra"])

    c = m.add_constraints(x >= rhs)
    assert c.shape == (5, 3)


@pytest.mark.parametrize(
    "rhs_factory",
    [
        pytest.param(
            lambda m: m.add_variables(coords=[range(5), range(3)]),
            id="variable",
        ),
        pytest.param(
            lambda m: 2 * m.add_variables(coords=[range(5), range(3)]) + 1,
            id="linexpr",
        ),
    ],
)
def test_constraint_rhs_higher_dim_expression(rhs_factory: Any) -> None:
    m = Model()
    x = m.add_variables(coords=[range(5)], name="x")

    c = m.add_constraints(x >= rhs_factory(m))
    assert c.shape == (5, 3)


def test_wrong_constraint_assignment_repeated() -> None:
    # repeated variable assignment is forbidden
    m: Model = Model()
    x = m.add_variables()
    m.add_constraints(x, LESS_EQUAL, 0, name="con")
    with pytest.raises(ValueError):
        m.add_constraints(x, LESS_EQUAL, 0, name="con")


def test_masked_constraints() -> None:
    m: Model = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper)
    y = m.add_variables()

    mask = pd.Series([True] * 5 + [False] * 5)
    m.add_constraints(1 * x + 10 * y, EQUAL, 0, mask=mask)
    assert (m.constraints.labels.con0[0:5, :] != -1).all()
    assert (m.constraints.labels.con0[5:10, :] == -1).all()


def test_masked_constraints_broadcast() -> None:
    m: Model = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper)
    y = m.add_variables()

    mask = pd.Series([True] * 5 + [False] * 5)
    m.add_constraints(1 * x + 10 * y, EQUAL, 0, name="bc1", mask=mask)
    assert (m.constraints.labels.bc1[0:5, :] != -1).all()
    assert (m.constraints.labels.bc1[5:10, :] == -1).all()

    mask2 = xr.DataArray([True] * 5 + [False] * 5, dims=["dim_1"])
    m.add_constraints(1 * x + 10 * y, EQUAL, 0, name="bc2", mask=mask2)
    assert (m.constraints.labels.bc2[:, 0:5] != -1).all()
    assert (m.constraints.labels.bc2[:, 5:10] == -1).all()

    # Pandas Series with named index missing a dim is broadcast to data.coords.
    mask_pd = pd.Series(
        [True, False, True] + [False] * 7, index=pd.RangeIndex(10, name="dim_0")
    )
    m.add_constraints(1 * x + 10 * y, EQUAL, 0, name="bc_pd", mask=mask_pd)
    assert (m.constraints.labels.bc_pd[[0, 2], :] != -1).all()
    assert (m.constraints.labels.bc_pd[[1, 3, 4, 5, 6, 7, 8, 9], :] == -1).all()

    # Mask with sparse coords (subset of data's coords) now raises instead of
    # emitting a FutureWarning — the rule from the bounds path applies here too.
    mask3 = xr.DataArray(
        [True, True, False, False, False],
        dims=["dim_0"],
        coords={"dim_0": range(5)},
    )
    with pytest.raises(
        ValueError, match=r"mask: coordinate values for dimension 'dim_0'"
    ):
        m.add_constraints(1 * x + 10 * y, EQUAL, 0, name="bc3", mask=mask3)

    # Mask with extra dimension not in data should raise
    mask4 = xr.DataArray([True, False], dims=["extra_dim"])
    with pytest.raises(ValueError, match=r"mask has dimension\(s\) \['extra_dim'\]"):
        m.add_constraints(1 * x + 10 * y, EQUAL, 0, name="bc4", mask=mask4)


def test_non_aligned_constraints() -> None:
    m: Model = Model()

    lower = xr.DataArray(np.zeros(10), coords=[range(10)])
    x = m.add_variables(lower, name="x")

    lower = xr.DataArray(np.zeros(8), coords=[range(8)])
    y = m.add_variables(lower, name="y")

    m.add_constraints(x == 0.0)
    m.add_constraints(y == 0.0)

    with pytest.warns(UserWarning):
        m.constraints.labels

        for dtype in m.constraints.labels.dtypes.values():
            assert np.issubdtype(dtype, np.integer)

        for dtype in m.constraints.coeffs.dtypes.values():
            assert np.issubdtype(dtype, np.floating)

        for dtype in m.constraints.vars.dtypes.values():
            assert np.issubdtype(dtype, np.integer)

        for dtype in m.constraints.rhs.dtypes.values():
            assert np.issubdtype(dtype, np.floating)


def test_constraints_flat() -> None:
    m: Model = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper)
    y = m.add_variables()

    assert isinstance(m.constraints.flat, pd.DataFrame)
    assert m.constraints.flat.empty
    with pytest.raises(ValueError):
        m.constraints.to_matrix()

    m.add_constraints(1 * x + 10 * y, EQUAL, 0)
    m.add_constraints(1 * x + 10 * y, LESS_EQUAL, 0)
    m.add_constraints(1 * x + 10 * y, GREATER_EQUAL, 0)

    assert isinstance(m.constraints.flat, pd.DataFrame)
    assert not m.constraints.flat.empty


def test_sanitize_infinities() -> None:
    m: Model = Model()

    lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
    upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(name="y")

    # Test correct infinities
    m.add_constraints(x <= np.inf, name="con_inf")
    m.add_constraints(y >= -np.inf, name="con_neg_inf")
    m.constraints.sanitize_infinities()
    assert (m.constraints["con_inf"].labels == -1).all()
    assert (m.constraints["con_neg_inf"].labels == -1).all()

    # Test incorrect infinities
    with pytest.raises(ValueError):
        m.add_constraints(x >= np.inf, name="con_wrong_inf")
    with pytest.raises(ValueError):
        m.add_constraints(y <= -np.inf, name="con_wrong_neg_inf")


def test_sanitize_infinities_refreshes_label_index() -> None:
    m = Model()
    x = m.add_variables(lower=0, coords=[range(3)], name="x")
    m.add_constraints(x <= np.array([np.inf, 5.0, np.inf]), name="up")
    m.add_constraints(x >= 1, name="lo")

    assert m.constraints.label_index.n_active_cons == 6
    m.constraints.sanitize_infinities()
    np.testing.assert_array_equal(m.constraints.label_index.clabels, [1, 3, 4, 5])
    np.testing.assert_array_equal(
        m.constraints.label_index.label_to_pos, [-1, 0, -1, 1, 2, 3]
    )


def test_sanitize_missings_refreshes_label_index() -> None:
    m = Model()
    x = m.add_variables(lower=0, coords=[range(3)], name="x")
    con = m.add_constraints(x <= 5, name="up")
    con.update(lhs=x.where(xr.DataArray([True, False, True], coords=x.coords)))

    assert m.constraints.label_index.n_active_cons == 3
    m.constraints.sanitize_missings()
    np.testing.assert_array_equal(m.constraints.label_index.clabels, [0, 2])


@pytest.mark.skipif("highs" not in available_solvers, reason="HiGHS is not installed")
@pytest.mark.parametrize("io_api", ["direct", "lp"])
def test_matrices_before_solve_with_infinite_rhs(io_api: str) -> None:
    # https://github.com/PyPSA/linopy/issues/1178
    m = Model()
    x = m.add_variables(lower=0, coords=[range(3)], name="x")
    m.add_constraints(x <= np.array([np.inf, 5.0, np.inf]), name="up")
    m.add_constraints(x >= 1, name="lo")
    m.add_objective(x.sum())

    assert m.matrices.A is not None
    m.solve("highs", io_api=io_api)

    np.testing.assert_allclose(m.constraints["lo"].dual, [1, 1, 1])
    np.testing.assert_allclose(m.constraints["up"].dual.values[1], 0)


def test_infinities_masked_restores_labels() -> None:
    m = Model()
    x = m.add_variables(lower=0, coords=[range(3)], name="x")
    up = m.add_constraints(x <= np.array([np.inf, 5.0, np.inf]), name="up")
    m.add_constraints(x >= 1, name="lo")

    assert m.constraints.label_index.n_active_cons == 6
    with pytest.raises(RuntimeError, match="boom"):
        with m.constraints.infinities_masked():
            np.testing.assert_array_equal(up.labels, [-1, 1, -1])
            assert m.constraints.label_index.n_active_cons == 4
            raise RuntimeError("boom")

    np.testing.assert_array_equal(up.labels, [0, 1, 2])
    np.testing.assert_array_equal(m.constraints.label_index.clabels, [0, 1, 2, 3, 4, 5])


def _model_with_infinite_row() -> tuple[Model, Any]:
    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[range(3)], name="x")
    m.add_constraints(x >= np.array([1.0, -np.inf, 3.0]), name="c")
    m.add_objective(x.sum())
    return m, x


@pytest.mark.skipif("highs" not in available_solvers, reason="HiGHS is not installed")
@pytest.mark.parametrize("io_api", ["direct", "lp"])
def test_solve_keeps_infinite_row_for_next_solve(io_api: str) -> None:
    m, x = _model_with_infinite_row()
    c = m.constraints["c"]

    m.solve("highs", io_api=io_api)
    np.testing.assert_array_equal(c.labels, [0, 1, 2])
    np.testing.assert_allclose(x.solution, [1, 0, 3])
    assert np.isnan(c.dual.values[1])

    c.update(rhs=np.array([1.0, 2.0, 3.0]))
    m.solve("highs", io_api=io_api)
    np.testing.assert_allclose(x.solution, [1, 2, 3])


@pytest.mark.skipif("highs" not in available_solvers, reason="HiGHS is not installed")
def test_persistent_update_after_model_solve_with_infinite_row() -> None:
    from linopy.solvers import Solver

    m, x = _model_with_infinite_row()
    c = m.constraints["c"]
    solver = Solver.from_name("highs", model=m, io_api="direct", track_updates=True)

    m.solve("highs", io_api="direct")
    c.update(rhs=np.array([1.0, 2.0, 3.0]))
    solver.update(m)
    assert solver._rebuilds == 0
    solver.solve(assign=True)
    np.testing.assert_allclose(x.solution, [1, 2, 3])


def test_masked_incorrect_infinities() -> None:
    # https://github.com/PyPSA/linopy/issues/1116
    m: Model = Model()
    x = m.add_variables(name="x", coords=[[0, 1]])
    rhs = xr.DataArray([-np.inf, -1000], coords=x.coords)

    con = m.add_constraints(x <= rhs, name="masked", mask=x.coords["dim_0"] == 1)
    assert (con.labels.values == [-1, con.labels.values[1]]).all()
    assert con.labels.values[1] != -1

    with pytest.raises(ValueError, match="incorrect infinite values"):
        m.add_constraints(x <= rhs, name="unmasked")
    with pytest.raises(ValueError, match="incorrect infinite values"):
        m.add_constraints(x <= rhs, name="wrong_mask", mask=x.coords["dim_0"] == 0)
