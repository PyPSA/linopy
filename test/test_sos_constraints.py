from __future__ import annotations

import pandas as pd
import pytest

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
        m.add_sos_constraints(m.add_variables(name="x"), sos_type=3, sos_dim="i")

    variable = m.add_variables(coords=[strings], name="string_var")

    with pytest.raises(ValueError, match="dimension"):
        m.add_sos_constraints(variable, sos_type=1, sos_dim="missing")

    with pytest.raises(ValueError, match="numeric"):
        m.add_sos_constraints(variable, sos_type=1, sos_dim="strings")

    numeric = m.add_variables(coords=[pd.Index([0, 1], name="dim")], name="num")
    m.add_sos_constraints(numeric, sos_type=1, sos_dim="dim")
    with pytest.raises(ValueError, match="already has"):
        m.add_sos_constraints(numeric, sos_type=1, sos_dim="dim")


def test_sos_constraints_written_to_lp(tmp_path) -> None:
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
