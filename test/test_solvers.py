#!/usr/bin/env python3
"""
Created on Tue Jan 28 09:03:35 2025.

@author: sid
"""

from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from linopy import solvers

free_mps_problem = """
NAME        sample_mip
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


@pytest.mark.parametrize("solver", solvers.available_solvers)
def test_free_mps_solution_parsing(solver):
    try:
        solver_enum = solvers.SolverName(solver.lower())
        solver_class = getattr(solvers, solver_enum.name)
    except ValueError:
        raise ValueError(f"Solver '{solver}' is not recognized")

    with NamedTemporaryFile(mode="w", suffix=".mps", delete_on_close=False) as mps_file:
        mps_file.write(free_mps_problem)
        mps_file.close()

        s = solver_class()
        with NamedTemporaryFile(suffix=".sol") as sol_file:
            result = s.solve_problem(
                problem_fn=Path(mps_file.name), solution_fn=Path(sol_file.name)
            )

    assert result.status.is_ok
    assert result.solution.objective == 30.0
