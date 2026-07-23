#!/usr/bin/env python3
"""
Created on Thu Mar 18 09:03:35 2021.

@author: fabian
"""

import importlib.util
import json
import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from linopy import LESS_EQUAL, Model, available_solvers, read_netcdf
from linopy.io import (
    read_pips_files,
    read_pips_problem,
    read_pips_solution,
    signed_number,
)
from linopy.testing import assert_model_equal

HAS_NETCDF4 = importlib.util.find_spec("netCDF4") is not None


@pytest.fixture
def model() -> Model:
    m = Model()

    x = m.add_variables(4, pd.Series([8, 10]), name="x")
    y = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4]]), name="y")

    m.add_constraints(x + y, LESS_EQUAL, 10)

    m.add_objective(2 * x + 3 * y)

    m.parameters["param"] = xr.DataArray([1, 2, 3, 4], dims=["x"])

    return m


@pytest.fixture
def model_with_dash_names() -> Model:
    m = Model()

    x = m.add_variables(4, pd.Series([8, 10]), name="x-var")
    x = m.add_variables(4, pd.Series([8, 10]), name="x-var-2")
    y = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4]]), name="y-var")

    m.add_constraints(x + y, LESS_EQUAL, 10, name="constraint-1")

    m.add_objective(2 * x + 3 * y)

    return m


@pytest.fixture
def model_with_multiindex() -> Model:
    m = Model()

    index = pd.MultiIndex.from_tuples(
        [(1, "a"), (1, "b"), (2, "a"), (2, "b")], names=["first", "second"]
    )
    x = m.add_variables(4, pd.Series([8, 10, 12, 14], index=index), name="x-var")
    y = m.add_variables(
        0, pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], index=index), name="y-var"
    )

    m.add_constraints(x + y, LESS_EQUAL, 10, name="constraint-1")

    m.add_objective(2 * x + 3 * y)

    return m


def test_model_to_netcdf(model: Model, tmp_path: Path) -> None:
    m = model
    fn = tmp_path / "test.nc"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    assert_model_equal(m, p)


def test_model_to_netcdf_frozen_constraint(tmp_path: Path) -> None:
    from linopy.constraints import CSRConstraint

    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(5, name="i")], name="x")
    m.add_constraints(x >= 1, name="c", freeze=True)

    assert isinstance(m.constraints["c"], CSRConstraint)

    fn = tmp_path / "test_frozen.nc"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    assert isinstance(p.constraints["c"], CSRConstraint)
    assert_model_equal(m, p)


def test_model_to_netcdf_mixed_sign_constraint(tmp_path: Path) -> None:
    from linopy.constraints import CSRConstraint

    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(4, name="i")], name="x")

    def bound(m: Model, i: int) -> object:
        if i % 2:
            return x.at[i] >= i
        return x.at[i] == 0.0

    m.add_constraints(bound, coords=[pd.RangeIndex(4, name="i")], name="c", freeze=True)
    assert isinstance(m.constraints["c"], CSRConstraint)

    fn = tmp_path / "test_mixed_sign.nc"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    assert isinstance(p.constraints["c"], CSRConstraint)
    import numpy as np

    np.testing.assert_array_equal(m.constraints["c"]._sign, p.constraints["c"]._sign)
    assert_model_equal(m, p)


def test_model_to_netcdf_with_sense(model: Model, tmp_path: Path) -> None:
    m = model
    m.objective.sense = "max"
    fn = tmp_path / "test.nc"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    assert_model_equal(m, p)


def test_model_to_netcdf_with_dash_names(
    model_with_dash_names: Model, tmp_path: Path
) -> None:
    m = model_with_dash_names
    fn = tmp_path / "test.nc"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    assert_model_equal(m, p)


def test_model_to_netcdf_with_status_and_condition(
    model_with_dash_names: Model, tmp_path: Path
) -> None:
    m = model_with_dash_names
    fn = tmp_path / "test.nc"
    m._status = "ok"
    m._termination_condition = "optimal"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    assert_model_equal(m, p)


def test_pickle_model(model_with_dash_names: Model, tmp_path: Path) -> None:
    m = model_with_dash_names
    fn = tmp_path / "test.nc"
    m._status = "ok"
    m._termination_condition = "optimal"

    with open(fn, "wb") as f:
        pickle.dump(m, f)

    with open(fn, "rb") as f:
        p = pickle.load(f)

    assert_model_equal(m, p)


def test_model_to_netcdf_with_multiindex(
    model_with_multiindex: Model, tmp_path: Path
) -> None:
    m = model_with_multiindex
    fn = tmp_path / "test.nc"
    m.to_netcdf(fn)
    p = read_netcdf(fn)

    assert_model_equal(m, p)


# Regression for https://github.com/PyPSA/linopy/issues/525.
def test_model_to_netcdf_with_multiindex_scipy_engine(
    model_with_multiindex: Model, tmp_path: Path
) -> None:
    m = model_with_multiindex
    fn = tmp_path / "test.nc"
    m.to_netcdf(fn, engine="scipy")

    raw_attrs = xr.load_dataset(fn).attrs
    multiindex_attrs = {k: v for k, v in raw_attrs.items() if k.endswith("_multiindex")}
    assert multiindex_attrs
    for k, v in multiindex_attrs.items():
        assert isinstance(v, str), f"{k!r}: {v!r}"

    assert_model_equal(m, read_netcdf(fn))


@pytest.mark.skipif(not HAS_NETCDF4, reason="legacy format requires netCDF4 backend")
def test_read_netcdf_with_multiindex_legacy_list_attr(
    model_with_multiindex: Model, tmp_path: Path
) -> None:
    # Older linopy stored multiindex names as a Python list (netCDF4-only).
    m = model_with_multiindex
    fn = tmp_path / "test.nc"
    m.to_netcdf(fn, engine="netcdf4")

    ds = xr.load_dataset(fn, engine="netcdf4").load()
    ds.attrs = {
        k: (json.loads(v) if k.endswith("_multiindex") and isinstance(v, str) else v)
        for k, v in ds.attrs.items()
    }
    fn_legacy = tmp_path / "legacy.nc"
    ds.to_netcdf(fn_legacy, engine="netcdf4")

    assert_model_equal(m, read_netcdf(fn_legacy))


def test_read_netcdf_without_version_stamp(model: Model, tmp_path: Path) -> None:
    from linopy.io import NETCDF_VERSION_ATTR

    fn = tmp_path / "test.nc"
    model.to_netcdf(fn)

    ds = xr.load_dataset(fn).load()
    del ds.attrs[NETCDF_VERSION_ATTR]
    fn_legacy = tmp_path / "legacy.nc"
    ds.to_netcdf(fn_legacy)

    assert_model_equal(model, read_netcdf(fn_legacy))


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_file_lp(model: Model, tmp_path: Path) -> None:
    import gurobipy

    fn = tmp_path / "test.lp"
    model.to_file(fn)

    gurobipy.read(str(fn))


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_file_lp_explicit_coordinate_names(model: Model, tmp_path: Path) -> None:
    import gurobipy

    fn = tmp_path / "test.lp"
    model.to_file(fn, io_api="lp", explicit_coordinate_names=True)

    gurobipy.read(str(fn))


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_file_lp_None(model: Model) -> None:
    import gurobipy

    fn: str | None = None
    model.to_file(fn)

    fn_path = model.get_problem_file()
    gurobipy.read(str(fn_path))


@pytest.mark.skipif(
    not {"gurobi", "highs"}.issubset(available_solvers),
    reason="Gurobipy of highspy not installed",
)
def test_to_file_mps(model: Model, tmp_path: Path) -> None:
    import gurobipy

    fn = tmp_path / "test.mps"
    model.to_file(fn)

    gurobipy.read(str(fn))


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_file_invalid(model: Model, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        fn = tmp_path / "test.failedtype"
        model.to_file(fn)


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_gurobipy(model: Model) -> None:
    gm = model.to_gurobipy()
    assert gm.NumVars > 0


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_gurobipy_no_names(model: Model) -> None:
    m_with = model.to_gurobipy(set_names=True)
    m_without = model.to_gurobipy(set_names=False)
    names_with = [v.VarName for v in m_with.getVars()]
    names_without = [v.VarName for v in m_without.getVars()]
    assert names_with != names_without


@pytest.mark.skipif("highs" not in available_solvers, reason="Highspy not installed")
def test_to_highspy(model: Model) -> None:
    h = model.to_highspy()
    assert h.getLp().num_col_ > 0


@pytest.mark.skipif("highs" not in available_solvers, reason="Highspy not installed")
def test_to_highspy_no_names(model: Model) -> None:
    h = model.to_highspy(set_names=False)
    lp = h.getLp()
    assert len(lp.col_names_) == 0
    assert len(lp.row_names_) == 0


@pytest.mark.skipif("mosek" not in available_solvers, reason="Mosek not installed")
def test_to_mosek(model: Model) -> None:
    task = model.to_mosek()
    assert task.getnumvar() > 0


@pytest.mark.skipif("xpress" not in available_solvers, reason="Xpress not installed")
def test_to_xpress(model: Model) -> None:
    p = model.to_xpress()
    assert p.attributes.cols > 0
    assert p.attributes.rows > 0


@pytest.mark.skipif("xpress" not in available_solvers, reason="Xpress not installed")
def test_to_xpress_no_names(model: Model) -> None:
    p_with = model.to_xpress(set_names=True)
    p_without = model.to_xpress(set_names=False)
    names_with = [v.name for v in p_with.getVariable()]
    names_without = [v.name for v in p_without.getVariable()]
    assert names_with != names_without


@pytest.mark.skipif("cupdlpx" not in available_solvers, reason="cuPDLPx not installed")
def test_to_cupdlpx(model: Model) -> None:
    cu = model.to_cupdlpx()
    assert cu is not None


def test_model_set_names_in_solver_io_default() -> None:
    assert Model().set_names_in_solver_io is True


@pytest.mark.skipif("highs" not in available_solvers, reason="Highspy not installed")
def test_model_set_names_in_solver_io(model: Model) -> None:
    model.solve(solver_name="highs", io_api="direct")
    expected_obj = model.objective.value

    model.set_names_in_solver_io = False
    status, _ = model.solve(solver_name="highs", io_api="direct")
    assert status == "ok"
    assert model.objective.value == pytest.approx(expected_obj)


def test_to_blocks(tmp_path: Path) -> None:
    m: Model = Model()

    lower: pd.Series = pd.Series(range(20))
    upper: pd.Series = pd.Series(range(30, 50))
    x = m.add_variables(lower, upper, name="x")
    y = m.add_variables(lower, upper, name="y")

    m.add_constraints(x + y, LESS_EQUAL, 10)

    m.add_objective(2 * x + 3 * y)

    m.blocks = xr.DataArray([1] * 10 + [2] * 10)

    m.to_block_files(tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "block0",
        "block1",
        "block2",
        "block3",
    ]

    def read(block: int, suffix: str) -> np.ndarray:
        return np.fromfile(tmp_path / f"block{block}" / suffix, sep="\n")

    labels = np.concatenate([read(n, "x") for n in range(3)]).astype(int)
    assert sorted(labels) == list(range(40))

    lowers = np.concatenate([read(n, "xl") for n in range(3)])
    upper_bounds = np.concatenate([read(n, "xu") for n in range(3)])
    assert sorted(lowers) == sorted(list(range(20)) * 2)
    assert sorted(upper_bounds) == sorted(list(range(30, 50)) * 2)

    coeffs = np.concatenate([read(n, "c") for n in range(3)])
    x_labels = m.variables["x"].labels.values
    is_x = np.isin(labels, x_labels)
    assert (coeffs[is_x] == 2).all()
    assert (coeffs[~is_x] == 3).all()


def _pips_time_plant_model(masked: bool) -> Model:
    n_time, n_plant, n_blocks = 12, 3, 2
    m = Model()
    time = pd.Index(range(n_time), name="time")
    plant = pd.Index(range(n_plant), name="plant")
    demand = pd.Series(3 + np.sin(np.pi / 24 * time), index=time)
    m.blocks = xr.DataArray(
        np.repeat(np.arange(1, n_blocks + 1), n_time // n_blocks), [time]
    )
    x = m.add_variables(lower=0, coords=[time, plant], name="x")
    y = m.add_variables(lower=0, coords=[plant], name="y")
    m.add_constraints(y.sum() >= 0, name="total_capacity")
    m.add_constraints(y - x >= 0, name="capacity")
    m.add_constraints(x.sum("plant") == demand, name="demand")
    ramp = (x - x.shift(time=1)).isel(time=slice(1, None))
    m.add_constraints(ramp <= 10, name="ramplimit")
    m.add_constraints(x.sum() <= 50, name="co2limit")
    if masked:
        mask = xr.DataArray(np.arange(n_time) < n_time // 2, [time])
        m.add_constraints(x.sum("plant") <= 20, name="cap", mask=mask)
    m.add_objective((2 * x).sum() + y.sum())
    return m


def _trivial_pips_model() -> Model:
    m = Model()
    idx = pd.Index(range(4), name="i")
    m.blocks = xr.DataArray([1, 1, 2, 2], [idx])
    x = m.add_variables(lower=0, upper=10, coords=[idx], name="x")
    g = m.add_variables(lower=0, upper=5, name="g")
    m.add_constraints(x.sum() + g <= 20, name="lim")
    m.add_constraints(x >= 1, name="floor")
    m.add_objective(x.sum() + 2 * g, sense="max")
    return m


PIPS_MODELS = {
    "time-plant": lambda: _pips_time_plant_model(masked=False),
    "time-plant-masked": lambda: _pips_time_plant_model(masked=True),
    "trivial": _trivial_pips_model,
}


@pytest.mark.parametrize("builder", PIPS_MODELS.values(), ids=PIPS_MODELS.keys())
def test_pips_problem_matches_matrices(
    builder: Callable[[], Model], tmp_path: Path
) -> None:
    m = builder()
    m.to_pips_files(tmp_path)
    prob = read_pips_problem(tmp_path)
    mats = m.matrices

    order_prob = np.argsort(prob.x_labels)
    order_orig = np.argsort(mats.vlabels)
    assert np.array_equal(prob.x_labels[order_prob], mats.vlabels[order_orig])
    assert np.allclose(prob.lb[order_prob], mats.lb[order_orig])
    assert np.allclose(prob.ub[order_prob], mats.ub[order_orig])
    assert np.allclose(prob.c[order_prob], mats.c[order_orig])
    assert prob.A_full.nnz == mats.A.nnz
    assert prob.A_full.shape == mats.A.shape
    assert sorted(prob.senses.tolist()) == sorted(mats.sense.tolist())
    assert prob.objective_sense == m.objective.sense


@pytest.mark.skipif("highs" not in available_solvers, reason="highs not installed")
@pytest.mark.parametrize("builder", PIPS_MODELS.values(), ids=PIPS_MODELS.keys())
def test_pips_roundtrip_solve(builder: Callable[[], Model], tmp_path: Path) -> None:
    m = builder()
    m.to_pips_files(tmp_path)
    m2 = read_pips_files(tmp_path)

    assert m2.objective.sense == m.objective.sense
    m.solve(solver_name="highs")
    m2.solve(solver_name="highs")
    assert m2.objective.value == pytest.approx(m.objective.value)
    assert set(m2.matrices.vlabels.tolist()) == set(m.matrices.vlabels.tolist())


@pytest.mark.skipif("highs" not in available_solvers, reason="highs not installed")
def test_pips_solution_roundtrip(tmp_path: Path) -> None:
    m = _pips_time_plant_model(masked=True)
    m.to_pips_files(tmp_path)
    m2 = read_pips_files(tmp_path)
    m.solve(solver_name="highs")

    lookup = dict(zip(m.matrices.vlabels.tolist(), m.matrices.sol.tolist()))
    manifest = json.loads((tmp_path / "pips.json").read_text())
    for k in range(manifest["n_blocks"] + 1):
        cl = np.fromfile(
            tmp_path / f"block{k}" / "col_labels", dtype=np.int64, sep="\n"
        )
        np.array([lookup[int(label)] for label in cl]).tofile(
            tmp_path / f"block{k}" / "x_sol", sep="\n"
        )
    np.array([m.objective.value]).tofile(tmp_path / "objective", sep="\n")
    (tmp_path / "status").write_text("optimal")

    solution = read_pips_solution(tmp_path, model=m2)
    assert solution.status == "optimal"
    assert solution.objective == pytest.approx(m.objective.value)
    assert sorted(solution.primal) == pytest.approx(sorted(m.matrices.sol))
    assert m2.objective.value == pytest.approx(m.objective.value)


class TestSignedNumberExpr:
    """Test the signed_number helper function for LP file formatting."""

    def test_positive_numbers(self) -> None:
        """Positive numbers should get a '+' prefix."""
        df = pl.DataFrame({"value": [1.0, 2.5, 100.0]})
        result = df.select(pl.concat_str(signed_number(pl.col("value"))))
        values = result.to_series().to_list()
        assert values == ["+1.0", "+2.5", "+100.0"]

    def test_negative_numbers(self) -> None:
        """Negative numbers should not get a '+' prefix (already have '-')."""
        df = pl.DataFrame({"value": [-1.0, -2.5, -100.0]})
        result = df.select(pl.concat_str(signed_number(pl.col("value"))))
        values = result.to_series().to_list()
        assert values == ["-1.0", "-2.5", "-100.0"]

    def test_positive_zero(self) -> None:
        """Positive zero should get a '+' prefix."""
        df = pl.DataFrame({"value": [0.0]})
        result = df.select(pl.concat_str(signed_number(pl.col("value"))))
        values = result.to_series().to_list()
        assert values == ["+0.0"]

    def test_negative_zero(self) -> None:
        """Negative zero is normalized to +0.0 - this is the bug fix."""
        # Create negative zero using numpy
        neg_zero = np.float64(-0.0)
        df = pl.DataFrame({"value": [neg_zero]})
        result = df.select(pl.concat_str(signed_number(pl.col("value"))))
        values = result.to_series().to_list()
        # The key assertion: should NOT be "+-0.0", -0.0 is normalized to +0.0
        assert values == ["+0.0"]
        assert "+-" not in values[0]

    def test_mixed_values_including_negative_zero(self) -> None:
        """Test a mix of positive, negative, and zero values."""
        neg_zero = np.float64(-0.0)
        df = pl.DataFrame({"value": [1.0, -1.0, 0.0, neg_zero, 2.5, -2.5]})
        result = df.select(pl.concat_str(signed_number(pl.col("value"))))
        values = result.to_series().to_list()
        # -0.0 is normalized to +0.0
        assert values == ["+1.0", "-1.0", "+0.0", "+0.0", "+2.5", "-2.5"]
        # No value should contain "+-"
        for v in values:
            assert "+-" not in v


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_file_lp_with_negative_zero_bounds(tmp_path: Path) -> None:
    """
    Test that LP files with negative zero bounds are valid.

    This is a regression test for the bug where -0.0 bounds would produce
    invalid LP file syntax like "+-0.0 <= x1 <= +0.0".

    See: https://github.com/PyPSA/linopy/issues/XXX
    """
    import gurobipy

    m = Model()

    # Create bounds that could produce -0.0
    # Using numpy to ensure we can create actual negative zeros
    lower = pd.Series([np.float64(-0.0), np.float64(0.0), np.float64(-0.0)])
    upper = pd.Series([np.float64(0.0), np.float64(-0.0), np.float64(1.0)])

    m.add_variables(lower, upper, name="x")
    m.add_objective(m.variables["x"].sum())

    fn = tmp_path / "test_neg_zero.lp"
    m.to_file(fn)

    # Read the LP file content and verify no "+-" appears
    with open(fn) as f:
        content = f.read()
    assert "+-" not in content, f"Found invalid '+-' in LP file: {content}"

    # Verify Gurobi can read it without errors
    gurobipy.read(str(fn))


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_file_lp_with_negative_zero_coefficients(tmp_path: Path) -> None:
    """
    Test that LP files with negative zero coefficients are valid.

    Coefficients can also potentially be -0.0 due to floating point arithmetic.
    """
    import gurobipy

    m = Model()

    x = m.add_variables(name="x", lower=0, upper=10)
    y = m.add_variables(name="y", lower=0, upper=10)

    # Create an expression where coefficients could become -0.0
    # through arithmetic operations
    coeff = np.float64(-0.0)
    expr = coeff * x + 1 * y

    m.add_constraints(expr <= 5)
    m.add_objective(x + y)

    fn = tmp_path / "test_neg_zero_coeffs.lp"
    m.to_file(fn)

    # Read the LP file content and verify no "+-" appears
    with open(fn) as f:
        content = f.read()
    assert "+-" not in content, f"Found invalid '+-' in LP file: {content}"

    # Verify Gurobi can read it without errors
    gurobipy.read(str(fn))


def test_to_file_lp_same_sign_constraints(tmp_path: Path) -> None:
    """Test LP writing when all constraints have the same sign operator."""
    m = Model()
    N = np.arange(5)
    x = m.add_variables(coords=[N], name="x")
    # All constraints use <=
    m.add_constraints(x <= 10, name="upper")
    m.add_constraints(x <= 20, name="upper2")
    m.add_objective(x.sum())

    fn = tmp_path / "same_sign.lp"
    m.to_file(fn)
    content = fn.read_text()
    assert "s.t." in content
    assert "<=" in content


def test_to_file_lp_mixed_sign_constraints(tmp_path: Path) -> None:
    """Test LP writing when constraints have different sign operators."""
    m = Model()
    N = np.arange(5)
    x = m.add_variables(coords=[N], name="x")
    # Mix of <= and >= constraints in the same container
    m.add_constraints(x <= 10, name="upper")
    m.add_constraints(x >= 1, name="lower")
    m.add_constraints(2 * x == 8, name="eq")
    m.add_objective(x.sum())

    fn = tmp_path / "mixed_sign.lp"
    m.to_file(fn)
    content = fn.read_text()
    assert "s.t." in content
    assert "<=" in content
    assert ">=" in content
    assert "=" in content


class TestLPBinaryBounds:
    """LP export honors binary bounds tightened below [0, 1] (#776)."""

    @pytest.fixture
    def make_tightened_model(self) -> Callable[[], Model]:
        def build() -> Model:
            m = Model()
            x = m.add_variables(
                binary=True, coords=[pd.RangeIndex(4, name="t")], name="x"
            )
            x.upper = pd.Series([1, 1, 0, 0], index=pd.RangeIndex(4, name="t"))
            m.add_constraints(x.sum() >= 2, name="atleast2")
            m.add_objective(-1 * x.sum())
            return m

        return build

    def test_default_bounds_omitted(self, tmp_path: Path) -> None:
        """A binary with the implied [0, 1] bounds gets no bounds section."""
        m = Model()
        b = m.add_variables(binary=True, coords=[pd.RangeIndex(3, name="t")], name="b")
        m.add_constraints(b.sum() >= 1, name="c")
        m.add_objective(b.sum())

        fn = tmp_path / "binary_default.lp"
        m.to_file(fn)
        assert "bounds" not in fn.read_text()

    def test_tightened_bounds_written(
        self, make_tightened_model: Callable[[], Model], tmp_path: Path
    ) -> None:
        """Per-element bounds tighter than [0, 1] reach the LP `bounds` section."""
        m = make_tightened_model()
        fn = tmp_path / "binary_tightened.lp"
        m.to_file(fn)

        bounds_section = fn.read_text().split("bounds")[1].split("binary")[0]
        for label in m.variables["x"].labels.values[2:]:
            assert f"x{label} <= +0.0" in bounds_section

    @pytest.mark.skipif(not available_solvers, reason="No solver installed")
    def test_lp_and_direct_agree(
        self, make_tightened_model: Callable[[], Model]
    ) -> None:
        """LP and direct paths see the same feasible set for tightened binaries."""
        solver = available_solvers[0]

        m_direct = make_tightened_model()
        m_direct.solve(solver_name=solver, io_api="direct")

        m_lp = make_tightened_model()
        m_lp.solve(solver_name=solver, io_api="lp")

        assert m_direct.objective.value == m_lp.objective.value == -2


def test_to_file_lp_frozen_vs_mutable(tmp_path: Path) -> None:
    """Test that frozen and mutable constraints produce identical LP output."""
    m_frozen = Model()
    N = np.arange(5)
    x = m_frozen.add_variables(coords=[N], name="x")
    y = m_frozen.add_variables(coords=[N], name="y")
    m_frozen.add_constraints(x + y <= 10, name="upper")
    m_frozen.add_constraints(x >= 1, name="lower")
    m_frozen.add_constraints(2 * x + y == 8, name="eq")
    m_frozen.add_objective(x.sum() + 2 * y.sum())

    m_mutable = Model()
    x2 = m_mutable.add_variables(coords=[N], name="x")
    y2 = m_mutable.add_variables(coords=[N], name="y")
    m_mutable.add_constraints(x2 + y2 <= 10, name="upper", freeze=False)
    m_mutable.add_constraints(x2 >= 1, name="lower", freeze=False)
    m_mutable.add_constraints(2 * x2 + y2 == 8, name="eq", freeze=False)
    m_mutable.add_objective(x2.sum() + 2 * y2.sum())

    fn_frozen = tmp_path / "frozen.lp"
    fn_mutable = tmp_path / "mutable.lp"
    m_frozen.to_file(fn_frozen)
    m_mutable.to_file(fn_mutable)

    assert fn_frozen.read_text() == fn_mutable.read_text()


def test_to_file_lp_frozen_mixed_sign(tmp_path: Path) -> None:
    """Test LP writing for frozen constraint with per-row signs."""
    m_frozen = Model()
    N = pd.RangeIndex(4, name="i")
    x = m_frozen.add_variables(coords=[N], name="x")

    def bound(m: Model, i: int) -> object:
        if i % 2:
            return x.at[i] >= i
        return x.at[i] <= 10

    m_frozen.add_constraints(bound, coords=[N], name="mixed", freeze=True)
    m_frozen.add_objective(x.sum())

    m_mutable = Model()
    x2 = m_mutable.add_variables(coords=[N], name="x")

    def bound2(m: Model, i: int) -> object:
        if i % 2:
            return x2.at[i] >= i
        return x2.at[i] <= 10

    m_mutable.add_constraints(bound2, coords=[N], name="mixed", freeze=False)
    m_mutable.add_objective(x2.sum())

    fn_frozen = tmp_path / "frozen_mixed.lp"
    fn_mutable = tmp_path / "mutable_mixed.lp"
    m_frozen.to_file(fn_frozen)
    m_mutable.to_file(fn_mutable)

    assert fn_frozen.read_text() == fn_mutable.read_text()
