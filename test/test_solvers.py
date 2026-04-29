#!/usr/bin/env python3
"""
Created on Tue Jan 28 09:03:35 2025.

@author: sid
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from test_io import model  # noqa: F401

from linopy import Model, solvers
from linopy.solver_capabilities import SolverFeature, solver_supports

free_mps_problem = """NAME               sample_mip
ROWS
 N  obj
 G  c1
 L  c2
 E  c3
COLUMNS
    col1        obj       5
    col1        c1        2
    col1        c2        4
    col1        c3        1
    MARK0000  'MARKER'                 'INTORG'
    colu2        obj       3
    colu2        c1        3
    colu2        c2        2
    colu2        c3        1
    col3        obj       7
    col3        c1        4
    col3        c2        3
    col3        c3        1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS_V     c1        12
    RHS_V     c2        15
    RHS_V     c3        6
BOUNDS
 UP BOUND     col1        4
 UI BOUND     colu2        3
 UI BOUND     col3        5
ENDATA
"""

free_lp_problem = """
Maximize
    z: 3 x + 4 y
Subject To
    c1: 2 x + y <= 10
    c2: x + 2 y <= 12
Bounds
    0 <= x
    0 <= y
End
"""


@pytest.mark.parametrize("solver", set(solvers.available_solvers))
def test_free_mps_solution_parsing(solver: str, tmp_path: Path) -> None:
    try:
        solver_enum = solvers.SolverName(solver.lower())
        solver_class = getattr(solvers, solver_enum.name)
    except ValueError:
        raise ValueError(f"Solver '{solver}' is not recognized")

    if not solver_supports(solver, SolverFeature.READ_MODEL_FROM_FILE):
        pytest.skip("Solver does not support reading model from file.")

    # Write the MPS file to the temporary directory
    mps_file = tmp_path / "problem.mps"
    mps_file.write_text(free_mps_problem)

    # Create a solution file path in the temporary directory
    sol_file = tmp_path / "solution.sol"

    s = solver_class()
    result = s.solve_problem(problem_fn=mps_file, solution_fn=sol_file)

    assert result.status.is_ok
    assert result.solution.objective == 30.0


@pytest.mark.skipif(
    "knitro" not in set(solvers.available_solvers), reason="Knitro is not installed"
)
def test_knitro_solver_mps(tmp_path: Path) -> None:
    """Test Knitro solver with a simple MPS problem."""
    knitro = solvers.Knitro()

    mps_file = tmp_path / "problem.mps"
    mps_file.write_text(free_mps_problem)
    sol_file = tmp_path / "solution.sol"

    result = knitro.solve_problem(problem_fn=mps_file, solution_fn=sol_file)

    assert result.status.is_ok
    assert result.solution is not None
    assert result.solution.objective == 30.0


@pytest.mark.skipif(
    "knitro" not in set(solvers.available_solvers), reason="Knitro is not installed"
)
def test_knitro_solver_for_lp(tmp_path: Path) -> None:
    """Test Knitro solver with a simple LP problem."""
    knitro = solvers.Knitro()

    lp_file = tmp_path / "problem.lp"
    lp_file.write_text(free_lp_problem)
    sol_file = tmp_path / "solution.sol"

    result = knitro.solve_problem(problem_fn=lp_file, solution_fn=sol_file)

    assert result.status.is_ok
    assert result.solution is not None
    assert result.solution.objective == pytest.approx(26.666, abs=1e-3)


@pytest.mark.skipif(
    "knitro" not in set(solvers.available_solvers), reason="Knitro is not installed"
)
def test_knitro_solver_with_options(tmp_path: Path) -> None:
    """Test Knitro solver with custom options."""
    knitro = solvers.Knitro(maxit=100, feastol=1e-6)

    mps_file = tmp_path / "problem.mps"
    mps_file.write_text(free_mps_problem)
    sol_file = tmp_path / "solution.sol"
    log_file = tmp_path / "knitro.log"

    result = knitro.solve_problem(
        problem_fn=mps_file, solution_fn=sol_file, log_fn=log_file
    )
    assert result.status.is_ok


@pytest.mark.skipif(
    "knitro" not in set(solvers.available_solvers), reason="Knitro is not installed"
)
def test_knitro_solver_with_model_raises_error(model: Model) -> None:  # noqa: F811
    """Test Knitro solver raises NotImplementedError for model-based solving."""
    knitro = solvers.Knitro()
    with pytest.raises(
        NotImplementedError, match="Direct API not implemented for Knitro"
    ):
        knitro.solve_problem(model=model)


@pytest.mark.skipif(
    "knitro" not in set(solvers.available_solvers), reason="Knitro is not installed"
)
def test_knitro_solver_no_log(tmp_path: Path) -> None:
    """Test Knitro solver without log file."""
    knitro = solvers.Knitro(outlev=0)

    mps_file = tmp_path / "problem.mps"
    mps_file.write_text(free_mps_problem)
    sol_file = tmp_path / "solution.sol"

    result = knitro.solve_problem(problem_fn=mps_file, solution_fn=sol_file)

    assert result.status.is_ok


@pytest.mark.skipif(
    "gurobi" not in set(solvers.available_solvers), reason="Gurobi is not installed"
)
def test_gurobi_environment_with_dict(
    model: Model, tmp_path: Path
) -> None:  # noqa: F811
    gurobi = solvers.Gurobi()

    mps_file = tmp_path / "problem.mps"
    mps_file.write_text(free_mps_problem)
    sol_file = tmp_path / "solution.sol"

    log1_file = tmp_path / "gurobi1.log"
    result = gurobi.solve_problem(
        problem_fn=mps_file, solution_fn=sol_file, env={"LogFile": str(log1_file)}
    )

    assert result.status.is_ok
    assert log1_file.exists()

    log2_file = tmp_path / "gurobi2.log"
    gurobi.solve_problem(
        model=model, solution_fn=sol_file, env={"LogFile": str(log2_file)}
    )
    assert result.status.is_ok
    assert log2_file.exists()


@pytest.mark.skipif(
    "gurobi" not in set(solvers.available_solvers), reason="Gurobi is not installed"
)
def test_gurobi_environment_with_gurobi_env(
    model: Model, tmp_path: Path
) -> None:  # noqa: F811
    import gurobipy as gp

    gurobi = solvers.Gurobi()

    mps_file = tmp_path / "problem.mps"
    mps_file.write_text(free_mps_problem)
    sol_file = tmp_path / "solution.sol"

    log1_file = tmp_path / "gurobi1.log"

    with gp.Env(params={"LogFile": str(log1_file)}) as env:
        result = gurobi.solve_problem(
            problem_fn=mps_file, solution_fn=sol_file, env=env
        )

    assert result.status.is_ok
    assert log1_file.exists()

    log2_file = tmp_path / "gurobi2.log"
    with gp.Env(params={"LogFile": str(log2_file)}) as env:
        gurobi.solve_problem(model=model, solution_fn=sol_file, env=env)
    assert result.status.is_ok
    assert log2_file.exists()


def _make_mosek_task_mock(
    *, bas_solsta=None, itr_solsta=None, itg_solsta=None
) -> MagicMock:
    """Build a ``mosek.Task`` mock with controlled per-soltype statuses."""
    mosek = pytest.importorskip("mosek", reason="Mosek is not installed")

    defined = {
        mosek.soltype.bas: bas_solsta,
        mosek.soltype.itr: itr_solsta,
        mosek.soltype.itg: itg_solsta,
    }

    task = MagicMock()
    task.solutiondef.side_effect = lambda st: defined[st] is not None
    task.getsolsta.side_effect = lambda st: defined[st]
    return task


def test_choose_mosek_solution_prefers_basic_when_itr_is_farkas() -> None:
    """When the IPM ends in a Farkas certificate but crossover is optimal, pick bas."""
    mosek = pytest.importorskip("mosek", reason="Mosek is not installed")
    task = _make_mosek_task_mock(
        bas_solsta=mosek.solsta.optimal,
        itr_solsta=mosek.solsta.dual_infeas_cer,
    )
    assert solvers._choose_mosek_solution(task) is mosek.soltype.bas


def test_choose_mosek_solution_prefers_itr_on_tie() -> None:
    """Both bas and itr optimal: prefer itr to preserve historical default."""
    mosek = pytest.importorskip("mosek", reason="Mosek is not installed")
    task = _make_mosek_task_mock(
        bas_solsta=mosek.solsta.optimal,
        itr_solsta=mosek.solsta.optimal,
    )
    assert solvers._choose_mosek_solution(task) is mosek.soltype.itr


def test_choose_mosek_solution_only_itr_defined() -> None:
    mosek = pytest.importorskip("mosek", reason="Mosek is not installed")
    task = _make_mosek_task_mock(itr_solsta=mosek.solsta.optimal)
    assert solvers._choose_mosek_solution(task) is mosek.soltype.itr


def test_choose_mosek_solution_only_bas_defined() -> None:
    mosek = pytest.importorskip("mosek", reason="Mosek is not installed")
    task = _make_mosek_task_mock(bas_solsta=mosek.solsta.optimal)
    assert solvers._choose_mosek_solution(task) is mosek.soltype.bas


def test_choose_mosek_solution_returns_none_when_nothing_defined() -> None:
    task = _make_mosek_task_mock()
    assert solvers._choose_mosek_solution(task) is None


def test_choose_mosek_solution_returns_itg_for_mip() -> None:
    mosek = pytest.importorskip("mosek", reason="Mosek is not installed")
    task = _make_mosek_task_mock(itg_solsta=mosek.solsta.integer_optimal)
    assert solvers._choose_mosek_solution(task) is mosek.soltype.itg


def test_choose_mosek_solution_itg_wins_over_bas_itr() -> None:
    """If itg is defined we never fall back to continuous solutions."""
    mosek = pytest.importorskip("mosek", reason="Mosek is not installed")
    task = _make_mosek_task_mock(
        bas_solsta=mosek.solsta.optimal,
        itr_solsta=mosek.solsta.optimal,
        itg_solsta=mosek.solsta.integer_optimal,
    )
    assert solvers._choose_mosek_solution(task) is mosek.soltype.itg


def test_choose_mosek_solution_picks_optimal_over_other_defined() -> None:
    """Optimal beats non-optimal defined statuses regardless of iteration order."""
    mosek = pytest.importorskip("mosek", reason="Mosek is not installed")
    task = _make_mosek_task_mock(
        bas_solsta=mosek.solsta.unknown,
        itr_solsta=mosek.solsta.optimal,
    )
    assert solvers._choose_mosek_solution(task) is mosek.soltype.itr

    task = _make_mosek_task_mock(
        bas_solsta=mosek.solsta.optimal,
        itr_solsta=mosek.solsta.unknown,
    )
    assert solvers._choose_mosek_solution(task) is mosek.soltype.bas


def test_choose_mosek_solution_falls_back_to_itr_when_both_non_optimal() -> None:
    """Two defined-but-non-optimal solutions: prefer itr to match prior default."""
    mosek = pytest.importorskip("mosek", reason="Mosek is not installed")
    task = _make_mosek_task_mock(
        bas_solsta=mosek.solsta.prim_infeas_cer,
        itr_solsta=mosek.solsta.dual_infeas_cer,
    )
    assert solvers._choose_mosek_solution(task) is mosek.soltype.itr


@pytest.mark.skipif(
    "mosek" not in set(solvers.available_solvers), reason="Mosek is not installed"
)
def test_mosek_smoke_lp(tmp_path: Path) -> None:
    """End-to-end smoke test: a small bounded LP solves to a finite optimum."""
    mosek_solver = solvers.Mosek()
    lp_file = tmp_path / "problem.lp"
    lp_file.write_text(free_lp_problem)
    sol_file = tmp_path / "solution.sol"

    result = mosek_solver.solve_problem(problem_fn=lp_file, solution_fn=sol_file)

    assert result.status.is_ok
    assert result.solution is not None
    import math

    assert math.isfinite(result.solution.objective)
    assert result.solution.objective == pytest.approx(80.0 / 3.0, abs=1e-3)
