#!/usr/bin/env python3
"""
Created on Tue Jan 28 09:03:35 2025.

@author: sid
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_io import model  # noqa: F401

from linopy import GREATER_EQUAL, Model, solvers
from linopy.constants import Result, Solution, Status
from linopy.solver_capabilities import (
    SOLVER_REGISTRY,
    SolverFeature,
    SolverInfo,
    solver_supports,
)
from linopy.solvers import _installed_version_in


@pytest.fixture
def simple_model() -> Model:
    m = Model(chunk=None)
    x = m.add_variables(name="x")
    y = m.add_variables(name="y")
    m.add_constraints(2 * x + 6 * y, GREATER_EQUAL, 10)
    m.add_constraints(4 * x + 2 * y, GREATER_EQUAL, 3)
    m.add_objective(2 * y + x)
    return m


@pytest.mark.parametrize("solver", sorted(set(solvers.available_solvers)))
def test_solver_instance_attached_after_solve(
    simple_model: Model, solver: str
) -> None:
    simple_model.solve(solver)
    assert isinstance(simple_model.solver, solvers.Solver)
    assert simple_model.solver.status is not None
    assert simple_model.solver.status.is_ok
    assert simple_model.solver.solution is not None
    assert simple_model.solver_model is simple_model.solver.solver_model
    assert simple_model.solver_name == solver


@pytest.mark.parametrize("solver", sorted(set(solvers.available_solvers)))
def test_result_carries_solver_name(simple_model: Model, solver: str) -> None:
    if not solver_supports(solver, SolverFeature.DIRECT_API):
        pytest.skip("Solver does not support direct API.")
    solver_enum = solvers.SolverName(solver.lower())
    solver_class = getattr(solvers, solver_enum.name)
    instance = solver_class()
    result = instance.solve_problem(model=simple_model)
    assert result.solver_name == solver


@pytest.mark.parametrize("solver", sorted(set(solvers.available_solvers)))
def test_prepare_solver_then_run(simple_model: Model, solver: str) -> None:
    if not solver_supports(solver, SolverFeature.DIRECT_API):
        pytest.skip("Solver does not support direct API.")
    simple_model.prepare_solver(solver)
    simple_model.run_solver()

    reference = Model(chunk=None)
    rx = reference.add_variables(name="x")
    ry = reference.add_variables(name="y")
    reference.add_constraints(2 * rx + 6 * ry, GREATER_EQUAL, 10)
    reference.add_constraints(4 * rx + 2 * ry, GREATER_EQUAL, 3)
    reference.add_objective(2 * ry + rx)
    reference.solve(solver, io_api="direct")

    assert simple_model.status == "ok"
    assert np.isclose(simple_model.objective.value, reference.objective.value)


@pytest.mark.parametrize("solver", sorted(set(solvers.available_solvers)))
def test_prepare_solver_set_names_false_run(
    simple_model: Model, solver: str
) -> None:
    if not solver_supports(solver, SolverFeature.DIRECT_API):
        pytest.skip("Solver does not support direct API.")
    simple_model.prepare_solver(solver, set_names=False)
    status, condition = simple_model.run_solver()

    assert status == "ok"
    assert condition == "optimal"
    assert simple_model.objective.value == pytest.approx(3.3)
    assert float(simple_model.variables["x"].solution) == pytest.approx(-0.1)
    assert float(simple_model.variables["y"].solution) == pytest.approx(1.7)


@pytest.mark.skipif(
    "highs" not in set(solvers.available_solvers), reason="HiGHS is not installed"
)
def test_highs_prepare_solver_applies_solver_options(simple_model: Model) -> None:
    highs_model = simple_model.prepare_solver("highs", time_limit=123)

    option_status, time_limit = highs_model.getOptionValue("time_limit")
    assert str(option_status) == "HighsStatus.kOk"
    assert time_limit == 123


@pytest.mark.skipif(
    "highs" not in set(solvers.available_solvers), reason="HiGHS is not installed"
)
def test_solver_state_compatibility_setters(simple_model: Model) -> None:
    simple_model.prepare_solver("highs")
    simple_model.solver_model = None
    assert simple_model.solver is None
    assert simple_model.solver_model is None
    assert simple_model.solver_name is None

    simple_model.prepare_solver("highs")
    simple_model.solver_name = None
    assert simple_model.solver is None
    assert simple_model.solver_model is None
    assert simple_model.solver_name is None

    with pytest.raises(AttributeError, match="managed via model.solver"):
        simple_model.solver_model = object()
    with pytest.raises(AttributeError, match="managed via model.solver"):
        simple_model.solver_name = "highs"


def test_apply_result_explicit(simple_model: Model) -> None:
    x_labels = simple_model.variables["x"].labels.values
    y_labels = simple_model.variables["y"].labels.values
    primal = pd.Series(
        {int(x_labels): 1.5, int(y_labels): 2.0}, dtype=float
    )
    solution = Solution(primal=primal, objective=5.5)
    result = Result(
        status=Status.from_termination_condition("optimal"),
        solution=solution,
        solver_name="mock",
    )
    simple_model.solver = None
    simple_model.apply_result(result)
    assert simple_model.status == "ok"
    assert simple_model.termination_condition == "optimal"
    assert simple_model.objective.value == 5.5
    assert float(simple_model.variables["x"].solution) == 1.5
    assert float(simple_model.variables["y"].solution) == 2.0


@pytest.mark.skipif(
    "gurobi" not in set(solvers.available_solvers), reason="Gurobi is not installed"
)
def test_gurobi_env_persists_after_solve(simple_model: Model) -> None:
    simple_model.solve("gurobi", io_api="direct")
    assert simple_model.solver is not None
    assert simple_model.solver.env is not None
    assert isinstance(simple_model.solver_model.NumVars, int)


@pytest.mark.parametrize("solver", sorted(set(solvers.available_solvers)))
def test_solver_close_releases_state(simple_model: Model, solver: str) -> None:
    simple_model.solve(solver)
    solver_instance = simple_model.solver
    assert solver_instance is not None
    solver_instance.close()
    assert solver_instance.solver_model is None
    assert solver_instance.env is None

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
def test_gurobi_environment_with_dict(model: Model, tmp_path: Path) -> None:  # noqa: F811
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
def test_gurobi_environment_with_gurobi_env(model: Model, tmp_path: Path) -> None:  # noqa: F811
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


@pytest.mark.parametrize(
    "solver_cls, feature, expected",
    [
        (solvers.Gurobi, SolverFeature.SOS_CONSTRAINTS, True),
        (solvers.Gurobi, SolverFeature.GPU_ACCELERATION, False),
        (solvers.Highs, SolverFeature.SOS_CONSTRAINTS, False),
        (solvers.Highs, SolverFeature.SEMI_CONTINUOUS_VARIABLES, True),
        (solvers.CBC, SolverFeature.LP_FILE_NAMES, False),
        (solvers.CBC, SolverFeature.INTEGER_VARIABLES, True),
        (solvers.cuPDLPx, SolverFeature.DIRECT_API, True),
        (solvers.cuPDLPx, SolverFeature.GPU_ACCELERATION, True),
        (solvers.cuPDLPx, SolverFeature.QUADRATIC_OBJECTIVE, False),
        (solvers.PIPS, SolverFeature.INTEGER_VARIABLES, False),
    ],
)
def test_solver_class_supports_feature(
    solver_cls: type, feature: SolverFeature, expected: bool
) -> None:
    assert solver_cls.supports(feature) is expected


def test_solver_instance_supports_matches_class() -> None:
    feature = SolverFeature.QUADRATIC_OBJECTIVE
    assert solvers.Gurobi.supports(feature) is True
    if "gurobi" in solvers.available_solvers:
        assert solvers.Gurobi().supports(feature) is True


@pytest.mark.parametrize("solver_name", [n.value for n in solvers.SolverName])
def test_capability_shim_round_trips(solver_name: str) -> None:
    solver_cls = getattr(solvers, solvers.SolverName(solver_name).name)
    for feature in SolverFeature:
        assert solver_supports(solver_name, feature) == solver_cls.supports(feature)


def test_solver_registry_iter_and_index() -> None:
    names = list(SOLVER_REGISTRY)
    assert "gurobi" in names
    for name in names:
        info = SOLVER_REGISTRY[name]
        assert isinstance(info, SolverInfo)
        assert isinstance(info.features, frozenset)
        assert info.name == name


@pytest.mark.skipif(
    "xpress" not in set(solvers.available_solvers), reason="Xpress is not installed"
)
def test_xpress_gpu_feature_reflects_installed_version() -> None:
    assert solvers.Xpress.supports(
        SolverFeature.GPU_ACCELERATION
    ) == _installed_version_in("xpress", ">=9.8.0")
