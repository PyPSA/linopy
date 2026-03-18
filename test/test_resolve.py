from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pytest

from linopy import Model
from linopy.constants import Result, Solution, Status, TerminationCondition
from linopy.solvers import Highs, SolverName


@pytest.fixture
def model() -> Model:
    m = Model()
    x = m.add_variables(lower=0, upper=10, name="x")
    y = m.add_variables(lower=0, upper=10, name="y")
    m.add_constraints(x + y >= 8, name="c1")
    m.add_constraints(x + y <= 15, name="c2")
    m.objective = 2 * x + 3 * y
    return m


def test_apply_result_basic(model: Model) -> None:
    model.solve(solver_name="highs")

    status = Status.from_termination_condition(TerminationCondition.optimal)
    solution = Solution(objective=42.0)
    result = Result(status=status, solution=solution)

    s, tc = model.apply_result(result, solver_name="test")
    assert s == "ok"
    assert model.objective.value == 42.0
    assert model.solver_name == "test"


def test_apply_result_infeasible(model: Model) -> None:
    status = Status.from_termination_condition(TerminationCondition.infeasible)
    result = Result(status=status, solution=Solution(objective=np.nan))

    s, tc = model.apply_result(result)
    assert s == "warning"
    assert tc == "infeasible"
    assert model.solver_name == "unknown"


def test_apply_result_none_solution(model: Model) -> None:
    status = Status.from_termination_condition(TerminationCondition.infeasible)
    result = Result(status=status, solution=None)

    s, tc = model.apply_result(result)
    assert s == "warning"
    assert tc == "infeasible"


def test_resolve_highs(model: Model) -> None:
    pytest.importorskip("highspy")
    model.solve(solver_name="highs")
    obj1 = model.objective.value

    h = model.solver_model
    n_cols = h.getNumCol()
    new_costs = np.array([10.0, 1.0], dtype=float)
    h.changeColsCost(n_cols, np.arange(n_cols, dtype=np.int32), new_costs)

    solver = Highs()
    sense = cast(Literal["min", "max"], model.sense)
    result = solver.resolve(h, sense=sense)
    model.apply_result(result, solver_name="highs")

    assert model.status == "ok"
    assert model.objective.value != obj1
    assert model.solution["y"].values.item() >= model.solution["x"].values.item()


def test_resolve_not_supported() -> None:
    from linopy.solvers import Solver

    class DummySolver(Solver[None]):
        def solve_problem_from_model(
            self,
            model: Model,
            solution_fn: Path | None = None,
            log_fn: Path | None = None,
            warmstart_fn: Path | None = None,
            basis_fn: Path | None = None,
            env: Any = None,
            explicit_coordinate_names: bool = False,
        ) -> Result:
            raise NotImplementedError

        def solve_problem_from_file(
            self,
            problem_fn: Path,
            solution_fn: Path | None = None,
            log_fn: Path | None = None,
            warmstart_fn: Path | None = None,
            basis_fn: Path | None = None,
            env: Any = None,
        ) -> Result:
            raise NotImplementedError

        @property
        def solver_name(self) -> SolverName:
            return SolverName.Highs

    solver = DummySolver.__new__(DummySolver)
    solver.solver_options = {}
    with pytest.raises(NotImplementedError, match="does not support resolve"):
        solver.resolve(None)
