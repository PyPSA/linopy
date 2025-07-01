#!/usr/bin/env python3
"""
Created on Tue Jan 28 09:03:35 2025.

@author: sid
"""

from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from linopy import LESS_EQUAL, Model, solvers

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


@pytest.fixture
def model() -> Model:
    m = Model()

    x = m.add_variables(4, pd.Series([8, 10]), name="x")
    y = m.add_variables(0, pd.DataFrame([[1, 2], [3, 4]]), name="y")

    m.add_constraints(x + y, LESS_EQUAL, 10)

    m.add_objective(2 * x + 3 * y)

    m.parameters["param"] = xr.DataArray([1, 2, 3, 4], dims=["x"])

    return m


@pytest.mark.parametrize("solver", set(solvers.available_solvers))
def test_free_mps_solution_parsing(solver: str, tmp_path: Path) -> None:
    try:
        solver_enum = solvers.SolverName(solver.lower())
        solver_class = getattr(solvers, solver_enum.name)
    except ValueError:
        raise ValueError(f"Solver '{solver}' is not recognized")

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
    "gurobi" not in set(solvers.available_solvers), reason="Gurobi is not installed"
)
def test_gurobi_environment_with_dict(model: Model, tmp_path: Path) -> None:
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
def test_gurobi_environment_with_gurobi_env(model: Model, tmp_path: Path) -> None:
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
