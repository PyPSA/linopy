from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model, available_solvers


def test_add_sos_constraints_registers_variable() -> None:
    m = Model()
    locations = pd.Index([0, 1, 2], name="locations")
    build = m.add_variables(coords=[locations], name="build")

    m.add_sos_constraints(build, sos_type=1, sos_dim="locations")

    assert build.attrs["sos_type"] == 1
    assert build.attrs["sos_dim"] == "locations"
    assert list(m.variables.sos) == ["build"]

    m.remove_sos_constraints(build)
    assert "sos_type" not in build.attrs
    assert "sos_dim" not in build.attrs


def test_add_sos_constraints_validation() -> None:
    m = Model()
    strings = pd.Index(["a", "b"], name="strings")
    with pytest.raises(ValueError, match="sos_type"):
        m.add_sos_constraints(m.add_variables(name="x"), sos_type=3, sos_dim="i")  # type: ignore[arg-type]

    variable = m.add_variables(coords=[strings], name="string_var")

    with pytest.raises(ValueError, match="dimension"):
        m.add_sos_constraints(variable, sos_type=1, sos_dim="missing")

    with pytest.raises(ValueError, match="numeric"):
        m.add_sos_constraints(variable, sos_type=1, sos_dim="strings")

    numeric = m.add_variables(coords=[pd.Index([0, 1], name="dim")], name="num")
    m.add_sos_constraints(numeric, sos_type=1, sos_dim="dim")
    with pytest.raises(ValueError, match="already has"):
        m.add_sos_constraints(numeric, sos_type=1, sos_dim="dim")


def test_sos_constraints_written_to_lp(tmp_path: Path) -> None:
    m = Model()
    breakpoints = pd.Index([0.0, 1.5, 3.5], name="bp")
    lambdas = m.add_variables(coords=[breakpoints], name="lambda")
    m.add_sos_constraints(lambdas, sos_type=2, sos_dim="bp")

    fn = tmp_path / "sos.lp"
    m.to_file(fn, io_api="lp")
    content = fn.read_text()

    assert "\nsos\n" in content
    assert "S2 ::" in content
    assert "3.5" in content


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobipy not installed")
def test_to_gurobipy_emits_sos_constraints() -> None:
    gurobipy = pytest.importorskip("gurobipy")

    m = Model()
    segments = pd.Index([0.0, 0.5, 1.0], name="seg")
    var = m.add_variables(coords=[segments], name="lambda")
    m.add_sos_constraints(var, sos_type=1, sos_dim="seg")

    try:
        model = m.to_gurobipy()
    except gurobipy.GurobiError as exc:  # pragma: no cover - depends on license setup
        pytest.skip(f"Gurobi environment unavailable: {exc}")

    assert model.NumSOS == 1


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobi not installed")
def test_sos1_binary_maximize_lp_polars() -> None:
    gurobipy = pytest.importorskip("gurobipy")

    m = Model()
    locations = pd.Index([0, 1, 2], name="locations")
    build = m.add_variables(coords=[locations], name="build", binary=True)
    m.add_sos_constraints(build, sos_type=1, sos_dim="locations")
    m.add_objective(build * np.array([1, 2, 3]), sense="max")

    try:
        m.solve(solver_name="gurobi", io_api="lp-polars")
    except gurobipy.GurobiError as exc:  # pragma: no cover - depends on license setup
        pytest.skip(f"Gurobi environment unavailable: {exc}")

    assert np.isclose(build.solution.values, [0, 0, 1]).all()
    assert m.objective.value is not None
    assert np.isclose(m.objective.value, 3)


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobi not installed")
def test_sos2_binary_maximize_direct() -> None:
    gurobipy = pytest.importorskip("gurobipy")

    m = Model()
    locations = pd.Index([0, 1, 2], name="locations")
    build = m.add_variables(coords=[locations], name="build", binary=True)
    m.add_sos_constraints(build, sos_type=2, sos_dim="locations")
    m.add_objective(build * np.array([1, 2, 3]), sense="max")

    try:
        m.solve(solver_name="gurobi", io_api="direct")
    except gurobipy.GurobiError as exc:  # pragma: no cover - depends on license setup
        pytest.skip(f"Gurobi environment unavailable: {exc}")

    assert np.isclose(build.solution.values, [0, 1, 1]).all()
    assert m.objective.value is not None
    assert np.isclose(m.objective.value, 5)


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobi not installed")
def test_sos2_binary_maximize_different_coeffs() -> None:
    gurobipy = pytest.importorskip("gurobipy")

    m = Model()
    locations = pd.Index([0, 1, 2], name="locations")
    build = m.add_variables(coords=[locations], name="build", binary=True)
    m.add_sos_constraints(build, sos_type=2, sos_dim="locations")
    m.add_objective(build * np.array([2, 1, 3]), sense="max")

    try:
        m.solve(solver_name="gurobi", io_api="direct")
    except gurobipy.GurobiError as exc:  # pragma: no cover - depends on license setup
        pytest.skip(f"Gurobi environment unavailable: {exc}")

    assert np.isclose(build.solution.values, [0, 1, 1]).all()
    assert m.objective.value is not None
    assert np.isclose(m.objective.value, 4)


@pytest.mark.skipif("xpress" not in available_solvers, reason="Xpress not installed")
def test_to_xpress_emits_sos_constraints() -> None:
    m = Model()
    segments = pd.Index([0.0, 0.5, 1.0], name="seg")
    var = m.add_variables(coords=[segments], name="lambda")
    m.add_sos_constraints(var, sos_type=1, sos_dim="seg")
    m.add_objective(var.sum())

    problem = m.to_xpress()
    assert problem.attributes.sets == 1


@pytest.mark.skipif("xpress" not in available_solvers, reason="Xpress not installed")
def test_to_xpress_emits_grouped_sos_constraints() -> None:
    m = Model()
    groups = pd.Index(["a", "b"], name="group")
    segments = pd.Index([0.0, 0.5, 1.0], name="seg")
    var = m.add_variables(coords=[groups, segments], name="lambda")
    m.add_sos_constraints(var, sos_type=1, sos_dim="seg")
    m.add_objective(var.sum())

    problem = m.to_xpress()
    assert problem.attributes.sets == len(groups)


@pytest.mark.skipif("xpress" not in available_solvers, reason="Xpress not installed")
def test_sos2_xpress_direct() -> None:
    m = Model()
    locations = pd.Index([0, 1, 2], name="locations")
    build = m.add_variables(coords=[locations], name="build", binary=True)
    m.add_sos_constraints(build, sos_type=2, sos_dim="locations")
    m.add_objective(build * np.array([1, 2, 3]), sense="max")

    m.solve(solver_name="xpress", io_api="direct")

    assert np.isclose(build.solution.values, [0, 1, 1]).all()
    assert m.objective.value is not None
    assert np.isclose(m.objective.value, 5)


@pytest.mark.skipif("xpress" not in available_solvers, reason="Xpress not installed")
def test_qp_sos1_xpress_direct() -> None:
    m = Model()
    seg = pd.Index([0, 1, 2], name="seg")
    x = m.add_variables(lower=0, upper=10, coords=[seg], name="x")
    m.add_sos_constraints(x, sos_type=1, sos_dim="seg")
    m.add_constraints(x.sum() >= 5)

    linear_coeffs = xr.DataArray([0.0, -10.0, 0.0], coords=[seg])
    m.add_objective((x * x).sum() + (linear_coeffs * x).sum(), sense="min")

    m.solve(solver_name="xpress", io_api="direct")

    assert np.isclose(x.solution.values, [0, 5, 0]).all()
    assert m.objective.value is not None
    assert np.isclose(m.objective.value, -25)


@pytest.mark.skipif("gurobi" not in available_solvers, reason="Gurobi not installed")
def test_reformulate_sos_true_reformulates_on_native_solver(tmp_path: Path) -> None:
    """
    ``reformulate_sos=True`` must reformulate even when the solver supports SOS.

    Asserted against the artifacts ``reformulate_sos_constraints`` writes into
    the LP file (the auxiliary binary + cardinality constraint, no ``sos``
    section). The reformulation is undone after solve, so the model itself
    looks unchanged — the LP snapshot is the durable evidence.
    """
    m = Model()
    idx = pd.Index([0, 1, 2], name="i")
    x = m.add_variables(lower=0, upper=1, coords=[idx], name="x")
    m.add_sos_constraints(x, sos_type=1, sos_dim="i")
    m.add_objective(x.sum())

    problem_fn = tmp_path / "problem.lp"
    m.solve(
        solver_name="gurobi",
        io_api="lp",
        reformulate_sos=True,
        problem_fn=problem_fn,
        keep_files=True,
        explicit_coordinate_names=True,
    )

    content = problem_fn.read_text()
    # SOS got rewritten to binary + linear: no `sos` section, the auxiliary
    # binary indicator and cardinality constraint appear instead.
    assert "\nsos\n" not in content
    assert "_sos_reform_x_y" in content
    assert "_sos_reform_x_card" in content


def test_unsupported_solver_raises_error() -> None:
    m = Model()
    locations = pd.Index([0, 1, 2], name="locations")
    build = m.add_variables(coords=[locations], name="build", binary=True)
    m.add_sos_constraints(build, sos_type=1, sos_dim="locations")
    m.add_objective(build * np.array([1, 2, 3]), sense="max")

    for solver in ["glpk", "mosek", "mindopt", "highs"]:
        if solver in available_solvers:
            with pytest.raises(ValueError, match="does not support SOS constraints"):
                m.solve(solver_name=solver)


def test_to_highspy_raises_when_sos_present() -> None:
    pytest.importorskip("highspy")

    m = Model()
    locations = pd.Index([0, 1, 2], name="locations")
    build = m.add_variables(coords=[locations], name="build", binary=True)
    m.add_sos_constraints(build, sos_type=1, sos_dim="locations")

    with pytest.raises(ValueError, match="does not support SOS constraints"):
        m.to_highspy()
