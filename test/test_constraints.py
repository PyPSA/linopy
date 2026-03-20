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

from linopy import EQUAL, GREATER_EQUAL, LESS_EQUAL, Model, Variable, available_solvers
from linopy.testing import assert_conequal

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
    assert m.constraints.labels.con0.dtype == int
    assert m.constraints.coeffs.con0.dtype in (int, float)
    assert m.constraints.vars.con0.dtype in (int, float)
    assert m.constraints.rhs.con0.dtype in (int, float)

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
    assert m.constraints.labels.con0.dtype == int
    assert m.constraints.coeffs.con0.dtype in (int, float)
    assert m.constraints.vars.con0.dtype in (int, float)
    assert m.constraints.rhs.con0.dtype in (int, float)


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
        pytest.param(lambda m, v: v, id="numpy"),
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


@pytest.mark.parametrize(
    "rhs_factory",
    [
        pytest.param(lambda m: np.ones((5, 3)), id="numpy"),
        pytest.param(lambda m: pd.DataFrame(np.ones((5, 3))), id="dataframe"),
    ],
)
@pytest.mark.legacy_only
def test_constraint_rhs_higher_dim_constant_warns(
    rhs_factory: Any, caplog: Any
) -> None:
    """Legacy: higher-dim constant RHS warns about dimensions."""
    m = Model()
    x = m.add_variables(coords=[range(5)], name="x")

    with caplog.at_level("WARNING", logger="linopy.expressions"):
        m.add_constraints(x >= rhs_factory(m))
    assert "dimensions" in caplog.text


@pytest.mark.v1_only
def test_constraint_rhs_higher_dim_constant_broadcasts_v1() -> None:
    """V1: higher-dim constant RHS broadcasts (creates redundant constraints)."""
    m = Model()
    x = m.add_variables(coords=[range(5)], name="x")
    rhs = xr.DataArray(np.ones((5, 3)), dims=["dim_0", "extra"])
    c = m.add_constraints(x >= rhs, name="broadcast_con")
    assert "extra" in c.dims


@pytest.mark.legacy_only
def test_constraint_rhs_higher_dim_dataarray_reindexes() -> None:
    """Legacy: DataArray RHS with extra dims reindexes to expression coords."""
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

    mask3 = xr.DataArray(
        [True, True, False, False, False],
        dims=["dim_0"],
        coords={"dim_0": range(5)},
    )
    with pytest.warns(FutureWarning, match="Missing values will be filled"):
        m.add_constraints(1 * x + 10 * y, EQUAL, 0, name="bc3", mask=mask3)
    assert (m.constraints.labels.bc3[0:2, :] != -1).all()
    assert (m.constraints.labels.bc3[2:5, :] == -1).all()
    assert (m.constraints.labels.bc3[5:10, :] == -1).all()

    # Mask with extra dimension not in data should raise
    mask4 = xr.DataArray([True, False], dims=["extra_dim"])
    with pytest.raises(AssertionError, match="not a subset"):
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


class TestConstraintCoordinateAlignment:
    """Tests for constraint behavior when variable and RHS coordinates differ."""

    @pytest.fixture(params=["xarray", "pandas_series"], ids=["da", "series"])
    def subset(self, request: Any) -> xr.DataArray | pd.Series:
        if request.param == "xarray":
            return xr.DataArray([10.0, 30.0], dims=["dim_2"], coords={"dim_2": [1, 3]})
        return pd.Series([10.0, 30.0], index=pd.Index([1, 3], name="dim_2"))

    @pytest.fixture(params=["xarray", "pandas_series"], ids=["da", "series"])
    def superset(self, request: Any) -> xr.DataArray | pd.Series:
        if request.param == "xarray":
            return xr.DataArray(
                np.arange(25, dtype=float),
                dims=["dim_2"],
                coords={"dim_2": range(25)},
            )
        return pd.Series(
            np.arange(25, dtype=float), index=pd.Index(range(25), name="dim_2")
        )

    # -- var <= subset --

    @pytest.mark.legacy_only
    def test_var_le_subset_fills_nan(self, v: Variable, subset: xr.DataArray) -> None:
        con = v <= subset
        assert con.sizes["dim_2"] == v.sizes["dim_2"]
        assert con.rhs.sel(dim_2=1).item() == 10.0
        assert con.rhs.sel(dim_2=3).item() == 30.0
        assert np.isnan(con.rhs.sel(dim_2=0).item())

    @pytest.mark.v1_only
    def test_var_le_subset_raises(self, v: Variable) -> None:
        subset = xr.DataArray([10.0, 30.0], dims=["dim_2"], coords={"dim_2": [1, 3]})
        with pytest.raises(ValueError, match="exact"):
            v <= subset

    @pytest.mark.v1_only
    def test_var_le_subset_join_left(self, v: Variable) -> None:
        subset = xr.DataArray([10.0, 30.0], dims=["dim_2"], coords={"dim_2": [1, 3]})
        con = v.to_linexpr().le(subset, join="left")
        assert con.sizes["dim_2"] == v.sizes["dim_2"]
        assert con.rhs.sel(dim_2=1).item() == 10.0
        assert con.rhs.sel(dim_2=3).item() == 30.0
        assert np.isnan(con.rhs.sel(dim_2=0).item())

    # -- var comparison (all signs) with subset --

    @pytest.mark.legacy_only
    @pytest.mark.parametrize("sign", [LESS_EQUAL, GREATER_EQUAL, EQUAL])
    def test_var_comparison_subset_fills_nan(
        self, v: Variable, subset: xr.DataArray, sign: str
    ) -> None:
        if sign == LESS_EQUAL:
            con = v <= subset
        elif sign == GREATER_EQUAL:
            con = v >= subset
        else:
            con = v == subset
        assert con.sizes["dim_2"] == v.sizes["dim_2"]
        assert con.rhs.sel(dim_2=1).item() == 10.0
        assert np.isnan(con.rhs.sel(dim_2=0).item())

    @pytest.mark.v1_only
    @pytest.mark.parametrize("sign", [LESS_EQUAL, GREATER_EQUAL, EQUAL])
    def test_var_comparison_subset_raises(self, v: Variable, sign: str) -> None:
        subset = xr.DataArray([10.0, 30.0], dims=["dim_2"], coords={"dim_2": [1, 3]})
        with pytest.raises(ValueError, match="exact"):
            if sign == LESS_EQUAL:
                v <= subset
            elif sign == GREATER_EQUAL:
                v >= subset
            else:
                v == subset

    @pytest.mark.v1_only
    @pytest.mark.parametrize("sign", [LESS_EQUAL, GREATER_EQUAL, EQUAL])
    def test_var_comparison_subset_join_left(self, v: Variable, sign: str) -> None:
        subset = xr.DataArray([10.0, 30.0], dims=["dim_2"], coords={"dim_2": [1, 3]})
        expr = v.to_linexpr()
        if sign == LESS_EQUAL:
            con = expr.le(subset, join="left")
        elif sign == GREATER_EQUAL:
            con = expr.ge(subset, join="left")
        else:
            con = expr.eq(subset, join="left")
        assert con.sizes["dim_2"] == v.sizes["dim_2"]
        assert con.rhs.sel(dim_2=1).item() == 10.0
        assert np.isnan(con.rhs.sel(dim_2=0).item())

    @pytest.mark.v1_only
    def test_var_comparison_subset_assign_coords(self, v: Variable) -> None:
        """V1 pattern: use assign_coords to align before comparing."""
        target_coords = v.coords["dim_2"][:2]
        subset = xr.DataArray(
            [10.0, 30.0], dims=["dim_2"], coords={"dim_2": target_coords}
        )
        con = v.loc[:1] <= subset
        assert con.sizes["dim_2"] == 2
        assert con.rhs.sel(dim_2=0).item() == 10.0
        assert con.rhs.sel(dim_2=1).item() == 30.0

    # -- expr <= subset --

    @pytest.mark.legacy_only
    def test_expr_le_subset_fills_nan(self, v: Variable, subset: xr.DataArray) -> None:
        expr = v + 5
        con = expr <= subset
        assert con.sizes["dim_2"] == v.sizes["dim_2"]
        assert con.rhs.sel(dim_2=1).item() == pytest.approx(5.0)
        assert con.rhs.sel(dim_2=3).item() == pytest.approx(25.0)
        assert np.isnan(con.rhs.sel(dim_2=0).item())

    @pytest.mark.v1_only
    def test_expr_le_subset_raises(self, v: Variable) -> None:
        subset = xr.DataArray([10.0, 30.0], dims=["dim_2"], coords={"dim_2": [1, 3]})
        expr = v + 5
        with pytest.raises(ValueError, match="exact"):
            expr <= subset

    @pytest.mark.v1_only
    def test_expr_le_subset_join_left(self, v: Variable) -> None:
        subset = xr.DataArray([10.0, 30.0], dims=["dim_2"], coords={"dim_2": [1, 3]})
        expr = v.to_linexpr() + 5
        con = expr.le(subset, join="left")
        assert con.sizes["dim_2"] == v.sizes["dim_2"]
        assert con.rhs.sel(dim_2=1).item() == pytest.approx(5.0)
        assert con.rhs.sel(dim_2=3).item() == pytest.approx(25.0)
        assert np.isnan(con.rhs.sel(dim_2=0).item())

    # -- subset comparison var (reverse) --

    @pytest.mark.legacy_only
    @pytest.mark.parametrize("sign", [LESS_EQUAL, GREATER_EQUAL, EQUAL])
    def test_subset_comparison_var_fills_nan(
        self, v: Variable, subset: xr.DataArray, sign: str
    ) -> None:
        if sign == LESS_EQUAL:
            con = subset <= v
        elif sign == GREATER_EQUAL:
            con = subset >= v
        else:
            con = subset == v
        assert con.sizes["dim_2"] == v.sizes["dim_2"]
        assert np.isnan(con.rhs.sel(dim_2=0).item())
        assert con.rhs.sel(dim_2=1).item() == pytest.approx(10.0)

    @pytest.mark.v1_only
    @pytest.mark.parametrize("sign", [LESS_EQUAL, GREATER_EQUAL, EQUAL])
    def test_subset_comparison_var_raises(self, v: Variable, sign: str) -> None:
        subset = xr.DataArray([10.0, 30.0], dims=["dim_2"], coords={"dim_2": [1, 3]})
        with pytest.raises(ValueError, match="exact"):
            if sign == LESS_EQUAL:
                subset <= v
            elif sign == GREATER_EQUAL:
                subset >= v
            else:
                subset == v

    # -- superset comparison var --

    @pytest.mark.legacy_only
    @pytest.mark.parametrize("sign", [LESS_EQUAL, GREATER_EQUAL])
    def test_superset_comparison_no_nan(
        self, v: Variable, superset: xr.DataArray, sign: str
    ) -> None:
        if sign == LESS_EQUAL:
            con = superset <= v
        else:
            con = superset >= v
        assert con.sizes["dim_2"] == v.sizes["dim_2"]
        assert not np.isnan(con.lhs.coeffs.values).any()
        assert not np.isnan(con.rhs.values).any()

    @pytest.mark.v1_only
    @pytest.mark.parametrize("sign", [LESS_EQUAL, GREATER_EQUAL])
    def test_superset_comparison_var_raises(self, v: Variable, sign: str) -> None:
        superset = xr.DataArray(
            np.arange(25, dtype=float), dims=["dim_2"], coords={"dim_2": range(25)}
        )
        with pytest.raises(ValueError, match="exact"):
            if sign == LESS_EQUAL:
                superset <= v
            else:
                superset >= v

    @pytest.mark.v1_only
    def test_superset_comparison_join_inner(self, v: Variable) -> None:
        superset = xr.DataArray(
            np.arange(25, dtype=float), dims=["dim_2"], coords={"dim_2": range(25)}
        )
        con = v.to_linexpr().le(superset, join="inner")
        assert con.sizes["dim_2"] == v.sizes["dim_2"]
        assert not np.isnan(con.rhs.values).any()

    # -- extra dims --

    @pytest.mark.legacy_only
    def test_rhs_extra_dims_broadcasts(self, v: Variable) -> None:
        rhs = xr.DataArray(
            [[1.0, 2.0]],
            dims=["extra", "dim_2"],
            coords={"dim_2": [0, 1]},
        )
        c = v <= rhs
        assert "extra" in c.dims

    @pytest.mark.v1_only
    def test_rhs_extra_dims_matching_broadcasts(self, v: Variable) -> None:
        rhs = xr.DataArray(
            np.ones((2, 20)), dims=["extra", "dim_2"], coords={"dim_2": range(20)}
        )
        c = v <= rhs
        assert "extra" in c.dims

    @pytest.mark.v1_only
    def test_rhs_extra_dims_mismatched_raises(self, v: Variable) -> None:
        rhs = xr.DataArray(
            [[1.0, 2.0]], dims=["extra", "dim_2"], coords={"dim_2": [0, 1]}
        )
        with pytest.raises(ValueError, match="exact"):
            v <= rhs

    @pytest.mark.v1_only
    def test_rhs_higher_dim_dataarray_matching_broadcasts(self) -> None:
        """V1: DataArray RHS with extra dims broadcasts if shared dim coords match."""
        m = Model()
        x = m.add_variables(coords=[range(5)], name="x")
        rhs = xr.DataArray(
            np.ones((5, 3)),
            dims=["dim_0", "extra"],
            coords={"dim_0": range(5)},
        )
        c = m.add_constraints(x >= rhs)
        assert c.shape == (5, 3)

    @pytest.mark.v1_only
    def test_rhs_higher_dim_dataarray_mismatched_raises(self) -> None:
        """V1: DataArray RHS with mismatched shared dim coords raises."""
        m = Model()
        x = m.add_variables(coords=[range(5)], name="x")
        rhs = xr.DataArray(
            np.ones((3, 3)),
            dims=["dim_0", "extra"],
            coords={"dim_0": [10, 11, 12]},
        )
        with pytest.raises(ValueError, match="exact"):
            m.add_constraints(x >= rhs)

    # -- solver integration --

    @pytest.mark.legacy_only
    def test_subset_constraint_solve_implicit(self) -> None:
        if not available_solvers:
            pytest.skip("No solver available")
        solver = "highs" if "highs" in available_solvers else available_solvers[0]
        m = Model()
        coords = pd.RangeIndex(5, name="i")
        x = m.add_variables(lower=0, upper=100, coords=[coords], name="x")
        subset_ub = xr.DataArray([10.0, 20.0], dims=["i"], coords={"i": [1, 3]})
        m.add_constraints(x <= subset_ub, name="subset_ub")
        m.add_objective(x.sum(), sense="max")
        m.solve(solver_name=solver)
        sol = m.solution["x"]
        assert sol.sel(i=1).item() == pytest.approx(10.0)
        assert sol.sel(i=3).item() == pytest.approx(20.0)
        assert sol.sel(i=0).item() == pytest.approx(100.0)
        assert sol.sel(i=2).item() == pytest.approx(100.0)
        assert sol.sel(i=4).item() == pytest.approx(100.0)

    @pytest.mark.v1_only
    def test_subset_constraint_solve_explicit_join(self) -> None:
        if not available_solvers:
            pytest.skip("No solver available")
        solver = "highs" if "highs" in available_solvers else available_solvers[0]
        m = Model()
        coords = pd.RangeIndex(5, name="i")
        x = m.add_variables(lower=0, upper=100, coords=[coords], name="x")
        subset_ub = xr.DataArray([10.0, 20.0], dims=["i"], coords={"i": [1, 3]})
        # exact default raises — use explicit join="left" (NaN = no constraint)
        m.add_constraints(x.to_linexpr().le(subset_ub, join="left"), name="subset_ub")
        m.add_objective(x.sum(), sense="max")
        m.solve(solver_name=solver)
        sol = m.solution["x"]
        assert sol.sel(i=1).item() == pytest.approx(10.0)
        assert sol.sel(i=3).item() == pytest.approx(20.0)
        assert sol.sel(i=0).item() == pytest.approx(100.0)
        assert sol.sel(i=2).item() == pytest.approx(100.0)
        assert sol.sel(i=4).item() == pytest.approx(100.0)
