import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd

import linopy
from linopy.solvers import Gurobi


class _FakeGurobiError(Exception):
    pass


class _FakeEnv:
    def setParam(self, *args: object) -> None:
        pass

    def __enter__(self) -> "_FakeEnv":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class _FakeGurobiModel:
    def __init__(self, *, sol_count: int, status: int = 9) -> None:
        self.SolCount = sol_count
        self.status = status
        self.Runtime = 0.0
        self.MIPGap = float("inf")
        self.ObjBound = 0.0
        self._vars: list[SimpleNamespace] = []
        self.x_attr_calls = 0

    def setParam(self, name: str, value: object) -> None:
        pass

    def addMVar(
        self, shape: tuple[int, ...], *args: object, **kwargs: object
    ) -> np.ndarray:
        self._vars = [SimpleNamespace(VarName=f"x{i}") for i in range(shape[0])]
        return np.zeros(shape[0])

    def addMConstr(self, *args: object, **kwargs: object) -> None:
        pass

    def setObjective(self, *args: object, **kwargs: object) -> None:
        pass

    def update(self) -> None:
        pass

    def optimize(self) -> None:
        pass

    def getVars(self) -> list[SimpleNamespace]:
        return self._vars

    def getConstrs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace()]

    @property
    def ObjVal(self) -> float:
        if self.SolCount == 0:
            raise _FakeGurobiError("Unable to retrieve attribute 'ObjVal'")
        return 1.5

    def getAttr(self, attr: str, objs: list[SimpleNamespace]) -> list[object]:
        if attr == "X":
            self.x_attr_calls += 1
            if self.SolCount == 0:
                raise _FakeGurobiError("Unable to retrieve attribute 'X'")
            return [1.0 for _ in objs]
        if attr == "VarName":
            return [v.VarName for v in objs]
        if attr == "Pi":
            raise _FakeGurobiError("Unable to retrieve attribute 'Pi'")
        return []

    def dispose(self) -> None:
        pass

    def __enter__(self) -> "_FakeGurobiModel":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def _install_fake_gurobipy(monkeypatch, model: _FakeGurobiModel) -> None:
    fake_gurobipy = types.ModuleType("gurobipy")
    fake_gurobipy.GurobiError = _FakeGurobiError
    fake_gurobipy.GRB = SimpleNamespace(
        BINARY="B",
        INTEGER="I",
        CONTINUOUS="C",
        MINIMIZE=1,
        MAXIMIZE=-1,
    )
    fake_gurobipy.Env = lambda *a, **kw: _FakeEnv()
    fake_gurobipy.Model = lambda *a, **kw: model
    monkeypatch.setitem(sys.modules, "gurobipy", fake_gurobipy)
    monkeypatch.setattr(linopy.solvers, "gurobipy", fake_gurobipy)
    monkeypatch.setattr(Gurobi, "is_available", classmethod(lambda cls: True))


def test_gurobi_time_limit_without_incumbent_returns_empty_solution(
    monkeypatch,
) -> None:
    model = _FakeGurobiModel(sol_count=0, status=9)
    _install_fake_gurobipy(monkeypatch, model)

    m = linopy.Model()
    x = m.add_variables(binary=True, coords=[pd.RangeIndex(3, name="item")])
    m.add_constraints(x.sum() <= 2)
    m.add_objective(-x.sum())

    status, cond = m.solve(solver_name="gurobi", io_api="direct")
    assert status == "ok"
    assert cond == "time_limit"
    assert model.x_attr_calls == 0
    assert "solution" not in x.data


def test_gurobi_time_limit_with_incumbent_returns_solution(monkeypatch) -> None:
    model = _FakeGurobiModel(sol_count=1, status=9)
    _install_fake_gurobipy(monkeypatch, model)

    m = linopy.Model()
    x = m.add_variables(binary=True, coords=[pd.RangeIndex(2, name="item")])
    m.add_constraints(x.sum() <= 2)
    m.add_objective(-x.sum())

    status, cond = m.solve(solver_name="gurobi", io_api="direct")
    assert status == "ok"
    assert cond == "time_limit"
    assert model.x_attr_calls == 1
