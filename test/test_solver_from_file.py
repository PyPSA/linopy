"""Tests for solving existing problem files without a linopy Model."""

import warnings
from pathlib import Path

import pytest

from linopy.solvers import Highs, Solver

pytestmark = pytest.mark.skipif(
    not Highs.is_available(), reason="HiGHS is not installed"
)

LP = """Minimize
 obj: x0
Subject To
 c0: x0 >= 2
Bounds
 0 <= x0 <= 10
End
"""

MPS = """NAME          SIMPLE
ROWS
 N  obj
 G  c0
COLUMNS
    x0        obj       1
    x0        c0        1
RHS
    rhs       c0        2
BOUNDS
 UP bnd       x0        10
ENDATA
"""


@pytest.mark.parametrize("suffix,content", [("lp", LP), ("mps", MPS)])
@pytest.mark.parametrize("as_string", [False, True])
def test_from_file_solves_existing_input(
    tmp_path: Path, suffix: str, content: str, as_string: bool
) -> None:
    problem = tmp_path / f"problem.{suffix}"
    problem.write_text(content)
    original = problem.read_bytes()
    options = {"time_limit": 60.0}
    log = tmp_path / "solver.log"

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        solver = Solver.from_file(
            "highs", str(problem) if as_string else problem, options=options
        )
        try:
            assert solver.model is None
            assert solver.solver_model is None

            result = solver.solve(log_fn=log)

            assert result.status.status.value == "ok"
            assert result.status.termination_condition.value == "optimal"
            assert result.solution.objective == pytest.approx(2.0)
            native_solution = result.solver_model.getSolution()
            assert native_solution.col_value[0] == pytest.approx(2.0)
            assert result.solver_name == "highs"
            _, time_limit = result.solver_model.getOptionValue("time_limit")
            assert time_limit == 60.0
            assert log.exists()
            assert options == {"time_limit": 60.0}
        finally:
            solver.close()

    assert problem.read_bytes() == original


def test_from_file_unknown_solver(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown solver"):
        Solver.from_file("not_a_real_solver", tmp_path / "problem.lp")


def test_from_file_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Solver.from_file("highs", tmp_path / "missing.lp")


def test_from_file_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="regular file"):
        Solver.from_file("highs", tmp_path)


def test_from_file_rejects_unknown_format(tmp_path: Path) -> None:
    problem = tmp_path / "problem.txt"
    problem.write_text(LP)
    with pytest.raises(ValueError, match=r"\.lp or \.mps"):
        Solver.from_file("highs", problem)


@pytest.mark.parametrize("suffix,content", [("lp", LP), ("mps", MPS)])
@pytest.mark.parametrize("external_names", [False, True])
def test_from_file_matches_legacy_api(
    tmp_path: Path, suffix: str, content: str, external_names: bool
) -> None:
    if external_names:
        content = content.replace("x0", "generation").replace("c0", "demand")
    problem = tmp_path / f"problem.{suffix}"
    problem.write_text(content)
    original = problem.read_bytes()

    legacy = Highs()
    current = Solver.from_file("highs", problem)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            expected = legacy.solve_problem(problem_fn=problem)

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            actual = current.solve()

        assert actual.status.status == expected.status.status
        assert (
            actual.status.termination_condition == expected.status.termination_condition
        )
        assert actual.solution.objective == pytest.approx(expected.solution.objective)
        assert actual.solver_model.getSolution().col_value == pytest.approx(
            expected.solver_model.getSolution().col_value
        )
        assert actual.solution.primal == pytest.approx(expected.solution.primal)
        assert actual.solution.dual == pytest.approx(expected.solution.dual)
    finally:
        current.close()
        legacy.close()

    assert problem.read_bytes() == original
