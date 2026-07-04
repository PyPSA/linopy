from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model, read_netcdf
from linopy.constants import Result, Solution, Status
from linopy.solvers import available_solvers


def test_variable_constraint_and_objective_scaling_in_matrices() -> None:
    m = Model()
    i = pd.Index(["a", "b"], name="i")
    scaling = xr.DataArray([10.0, 100.0], coords=[i])

    x = m.add_variables(
        lower=xr.DataArray([1.0, 2.0], coords=[i]),
        upper=xr.DataArray([3.0, 4.0], coords=[i]),
        name="x",
        scaling=scaling,
    )
    b = m.add_variables(binary=True, name="b", scaling=50.0)

    row_scaling = xr.DataArray([2.0, 4.0], coords=[i])
    m.add_constraints(
        2 * x + 3 * b,
        ">=",
        xr.DataArray([20.0, 40.0], coords=[i]),
        name="c",
        scaling=row_scaling,
    )
    m.add_objective((5 * x).sum() + 7 * b, scaling=10.0)

    matrices = m.matrices

    np.testing.assert_allclose(matrices.lb, [10.0, 200.0, 0.0])
    np.testing.assert_allclose(matrices.ub, [30.0, 400.0, 1.0])
    np.testing.assert_allclose(matrices.b, [10.0, 10.0])
    np.testing.assert_allclose(
        matrices.A.toarray(),
        [
            [2 / 10 / 2, 0.0, 3 / 2],
            [0.0, 2 / 100 / 4, 3 / 4],
        ],
    )
    np.testing.assert_allclose(matrices.c, [5 / 10 / 10, 5 / 100 / 10, 7 / 10])

    assert float(b.scaling) == 50.0


def test_quadratic_objective_scaling_in_matrix() -> None:
    m = Model()
    i = pd.Index([0, 1], name="i")
    x = m.add_variables(coords=[i], name="x", scaling=[2.0, 4.0])

    m.add_objective((x * x).sum(), scaling=10.0)

    np.testing.assert_allclose(
        m.matrices.Q.toarray(),
        np.diag([2 / 2 / 2 / 10, 2 / 4 / 4 / 10]),
    )


def test_indicator_constraint_scaling_in_matrix() -> None:
    m = Model()
    i = pd.Index(["a", "b"], name="i")
    x = m.add_variables(coords=[i], name="x", scaling=[10.0, 100.0])
    b = m.add_variables(binary=True, name="b")

    rhs = xr.DataArray([20.0, 40.0], coords=[i])
    scaling = xr.DataArray([2.0, 4.0], coords=[i])
    con = m.add_indicator_constraints(
        b,
        1,
        2 * x,
        "<=",
        rhs,
        name="ic",
        scaling=scaling,
    )

    np.testing.assert_allclose(con.scaling.values, [2.0, 4.0])
    np.testing.assert_allclose(m.matrices.indicator_b, [10.0, 10.0])
    np.testing.assert_allclose(
        m.matrices.indicator_A.toarray(),
        [
            [2 / 10 / 2, 0.0, 0.0],
            [0.0, 2 / 100 / 4, 0.0],
        ],
    )


def test_assign_result_unscales_solution_objective_and_dual() -> None:
    m = Model()
    i = pd.Index(["a", "b"], name="i")
    x = m.add_variables(coords=[i], name="x", scaling=[10.0, 100.0])
    b = m.add_variables(binary=True, name="b", scaling=50.0)

    m.add_constraints(x + b >= 1, name="c", scaling=[2.0, 4.0])
    m.add_objective(x.sum() + b, scaling=10.0)

    primal = np.full(m._xCounter, np.nan)
    primal[x.labels.values.ravel()] = [20.0, 300.0]
    primal[int(b.labels)] = 1.0
    dual = np.full(m._cCounter, np.nan)
    dual[m.constraints["c"].labels.values.ravel()] = [4.0, 8.0]
    result = Result(
        status=Status.from_termination_condition("optimal"),
        solution=Solution(primal=primal, dual=dual, objective=12.3),
        solver_name="mock",
    )

    m.assign_result(result)

    np.testing.assert_allclose(x.solution.values, [2.0, 3.0])
    assert float(b.solution) == 1.0
    assert m.objective.value == pytest.approx(123.0)
    np.testing.assert_allclose(m.constraints["c"].dual.values, [20.0, 20.0])


def test_scaling_is_preserved_in_netcdf(tmp_path) -> None:
    m = Model()
    i = pd.Index(["a", "b"], name="i")
    x = m.add_variables(coords=[i], name="x", scaling=[10.0, 100.0])
    m.add_constraints(x >= 1, name="c", scaling=[2.0, 4.0], freeze=True)
    m.add_objective(x.sum(), scaling=5.0)

    path = tmp_path / "scaled.nc"
    m.to_netcdf(path)
    restored = read_netcdf(path)

    np.testing.assert_allclose(restored.variables["x"].scaling.values, [10.0, 100.0])
    np.testing.assert_allclose(restored.constraints["c"].scaling.values, [2.0, 4.0])
    assert restored.objective.scaling == 5.0


def test_lp_export_uses_scaled_values(tmp_path) -> None:
    m = Model()
    x = m.add_variables(lower=0, upper=10, name="x", scaling=10.0)
    m.add_constraints(2 * x >= 20, name="c", scaling=2.0)
    m.add_objective(5 * x, scaling=5.0)

    path = tmp_path / "scaled.lp"
    m.to_file(path, io_api="lp", progress=False)
    text = path.read_text()

    assert "0.1 x0" in text
    assert ">= 10.0" in text
    assert "<= +100.0" in text


def test_scaling_validation() -> None:
    m = Model()
    with pytest.raises(ValueError, match="finite positive"):
        m.add_variables(name="x", scaling=0)

    x = m.add_variables(name="x")
    with pytest.raises(ValueError, match="finite positive"):
        m.add_constraints(x >= 1, scaling=-1)

    with pytest.raises(ValueError, match="finite and positive"):
        m.add_objective(x, scaling=np.inf)


def test_constraint_scaling_setter_broadcasts_to_rows() -> None:
    m = Model()
    i = pd.Index(["a", "b"], name="i")
    x = m.add_variables(coords=[i], name="x")
    con = m.add_constraints(x >= 1, name="c")

    con.scaling = 2.0

    np.testing.assert_allclose(con.scaling.values, [2.0, 2.0])
    np.testing.assert_allclose(m.matrices.b, [0.5, 0.5])


@pytest.mark.skipif("highs" not in available_solvers, reason="HiGHS is not installed")
def test_scaled_solve_preserves_user_units() -> None:
    reference = Model()
    rx = reference.add_variables(lower=0, name="x")
    reference.add_constraints(rx >= 3)
    reference.add_objective(rx)
    reference.solve("highs", io_api="direct", output_flag=False)

    scaled = Model()
    sx = scaled.add_variables(lower=0, name="x", scaling=10.0)
    scaled.add_constraints(sx >= 3, scaling=2.0)
    scaled.add_objective(sx, scaling=5.0)
    scaled.solve("highs", io_api="direct", output_flag=False)

    assert float(sx.solution) == pytest.approx(float(rx.solution))
    assert scaled.objective.value == pytest.approx(reference.objective.value)
