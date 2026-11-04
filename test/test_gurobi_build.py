"""
The Gurobi direct build loads the model in one ``gurobipy.loadModel`` call.

It must build the same model as the ``addMVar``/``addMConstr`` path it
replaces, which is kept as the fallback for gurobipy without ``loadModel``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from linopy import Model, licensed_solvers
from linopy.io import get_printers_scalar
from linopy.solvers import Gurobi, _iter_sos_sets

pytestmark = pytest.mark.skipif(
    "gurobi" not in licensed_solvers, reason="Gurobi not licensed"
)

gurobipy = pytest.importorskip("gurobipy")
if not hasattr(gurobipy, "loadModel"):  # pragma: no cover - gurobipy < 12
    pytest.skip("gurobipy has no loadModel", allow_module_level=True)


def _lp(sense: str = "min") -> Model:
    m = Model()
    x = m.add_variables(lower=-2, upper=pd.Series([4.0, 5, 6]), name="x")
    y = m.add_variables(
        lower=0, upper=10, coords=[pd.RangeIndex(2, name="j")], name="y"
    )
    m.add_constraints(x.sum() + 2 * y.sum() >= 1, name="c1")
    m.add_constraints(x.loc[[0, 1]].rename(dim_0="j") - 3 * y <= 7, name="c2")
    m.add_constraints(x.loc[2] + y.loc[0] == 2, name="c3")
    m.add_objective(1.5 * x.sum() - y.sum(), sense=sense)
    return m


def _milp() -> Model:
    m = _lp()
    m.add_variables(name="b", binary=True)
    m.add_variables(lower=0, upper=9, name="i", integer=True)
    m.add_variables(lower=1, upper=10, name="s", semi_continuous=True)
    m.add_constraints(m["b"] + m["i"] + m["s"] >= 2, name="c4")
    m.objective += m["i"] + m["s"]
    return m


def _qp() -> Model:
    m = _lp(sense="max")
    x, y = m["x"], m["y"]
    cross = x.isel(dim_0=0, drop=True) * x.isel(dim_0=2, drop=True)
    m.objective = 2 * x.sum() - (x * x).sum() - cross - (y * y).sum()
    m.objective.sense = "max"
    return m


def _no_constraints() -> Model:
    m = Model()
    x = m.add_variables(lower=0, upper=3, coords=[pd.RangeIndex(4)], name="x")
    m.add_objective(-x.sum())
    return m


def _sos_and_indicator() -> Model:
    m = _lp()
    seg = m.add_variables(coords=[pd.Index([0.0, 0.5, 1.0], name="seg")], name="l")
    m.add_sos_constraints(seg, sos_type=2, sos_dim="seg")
    b = m.add_variables(name="b", binary=True)
    m.add_indicator_constraints(b, 1, m["x"].sum(), "<=", 5, name="ic")
    return m


MODELS = {
    "lp": _lp,
    "lp-max": lambda: _lp("max"),
    "milp": _milp,
    "qp": _qp,
    "no-constraints": _no_constraints,
    "sos-indicator": _sos_and_indicator,
}


def _snapshot(gm: Any) -> dict[str, Any]:
    """Everything the build sets, read back from the gurobipy model."""
    gm.update()
    vars_, cons = gm.getVars(), gm.getConstrs()
    Q = gm.getQ().toarray() if gm.IsQP else np.zeros((gm.NumVars, gm.NumVars))
    return {
        "ModelSense": gm.ModelSense,
        "ObjCon": gm.ObjCon,
        "A": gm.getA().toarray(),
        # Gurobi stores x_i x_j and x_j x_i as one term; compare symmetrised
        "Q": Q + Q.T,
        **{
            attr: gm.getAttr(attr, vars_)
            for attr in ("Obj", "LB", "UB", "VType", "VarName")
        },
        **{attr: gm.getAttr(attr, cons) for attr in ("RHS", "Sense", "ConstrName")},
        "NumSOS": gm.NumSOS,
        "NumGenConstrs": gm.NumGenConstrs,
    }


@pytest.fixture(scope="module")
def env() -> Any:
    with gurobipy.Env(params={"OutputFlag": 0}) as e:
        yield e


@pytest.mark.parametrize("set_names", [False, True])
@pytest.mark.parametrize("name", list(MODELS))
def test_load_model_matches_mvar_build(env: Any, name: str, set_names: bool) -> None:
    m = MODELS[name]()
    M = m.matrices
    names = get_printers_scalar(m) if set_names else None
    sos = list(_iter_sos_sets(m))

    with (
        Gurobi._build_solver_model(M, env, names=names, sos=sos) as new,
        Gurobi._build_solver_model_mvar(M, env, names=names, sos=sos) as old,
    ):
        got, expected = _snapshot(new), _snapshot(old)
        assert got.keys() == expected.keys()
        for key in expected:
            np.testing.assert_array_equal(got[key], expected[key], err_msg=key)
        if set_names:
            assert got["VarName"][0].startswith("x")
            assert got["ConstrName"] == [] or got["ConstrName"][0].startswith("c")


@pytest.mark.parametrize("name", ["lp", "lp-max", "milp", "qp", "sos-indicator"])
def test_solution_matches_mvar_build(env: Any, name: str) -> None:
    m = MODELS[name]()
    M = m.matrices
    sos = list(_iter_sos_sets(m))
    objectives = []
    for build in (Gurobi._build_solver_model, Gurobi._build_solver_model_mvar):
        with build(M, env, names=None, sos=sos) as gm:
            gm.optimize()
            assert gm.Status == gurobipy.GRB.OPTIMAL
            objectives.append(gm.ObjVal)
    assert objectives[0] == pytest.approx(objectives[1])


@pytest.mark.parametrize("name", ["lp", "milp", "qp"])
def test_solve_reads_solution_and_duals(name: str) -> None:
    m = MODELS[name]()
    status, condition = m.solve("gurobi", io_api="direct", OutputFlag=0)
    assert status == "ok"
    ref = MODELS[name]()
    ref.solve("gurobi", io_api="lp", OutputFlag=0)
    assert m.objective.value == pytest.approx(ref.objective.value)
    for v in m.variables:
        np.testing.assert_allclose(
            m.variables[v].solution, ref.variables[v].solution, atol=1e-6
        )
    if name != "milp":  # Gurobi has no duals for a MILP
        for c in m.constraints:
            np.testing.assert_allclose(
                m.constraints[c].dual, ref.constraints[c].dual, atol=1e-6
            )
