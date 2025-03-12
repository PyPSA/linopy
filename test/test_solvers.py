#!/usr/bin/env python3
"""
Created on Tue Jan 28 09:03:35 2025.

@author: sid
"""

from pathlib import Path

import pytest

from linopy import solvers

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
