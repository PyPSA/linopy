#!/usr/bin/env python3
"""
Created on Tue Jan 28 09:03:35 2025.

@author: sid
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from test_io import model  # noqa: F401

from linopy import GREATER_EQUAL, Model, solvers
from linopy.constants import Result, Solution, Status
from linopy.constraints import CSRConstraint
from linopy.solver_capabilities import (
    SOLVER_REGISTRY,
    SolverFeature,
    SolverInfo,
    solver_supports,
)
from linopy.solvers import _installed_version_in


@pytest.fixture
def lp_only_solver() -> str:
    for name in ("glpk", "cbc"):
        if name in solvers.available_solvers:
            return name
    pytest.skip("Need an LP-only solver (glpk or cbc) installed")


@pytest.fixture
def simple_model() -> Model:
    m = Model(chunk=None)
    x = m.add_variables(name="x")
    y = m.add_variables(name="y")
    m.add_constraints(2 * x + 6 * y, GREATER_EQUAL, 10)
    m.add_constraints(4 * x + 2 * y, GREATER_EQUAL, 3)
    m.add_objective(2 * y + x)
    return m


@pytest.mark.parametrize("solver", sorted(set(solvers.licensed_solvers)))
def test_solver_instance_attached_after_solve(simple_model: Model, solver: str) -> None:
    simple_model.solve(solver)
    assert isinstance(simple_model.solver, solvers.Solver)
    assert simple_model.solver.status is not None
    assert simple_model.solver.status.is_ok
    assert simple_model.solver.solution is not None
    assert simple_model.solver_model is simple_model.solver.solver_model
    assert simple_model.solver_name == solver


@pytest.mark.parametrize("solver", sorted(set(solvers.licensed_solvers)))
def test_result_carries_solver_name(simple_model: Model, solver: str) -> None:
    if not solver_supports(solver, SolverFeature.DIRECT_API):
        pytest.skip("Solver does not support direct API.")
    instance = solvers.Solver.from_name(solver, simple_model, io_api="direct")
    result = instance.solve()
    assert result.solver_name == solver


@pytest.mark.parametrize("solver", sorted(set(solvers.licensed_solvers)))
def test_from_name_then_solve(simple_model: Model, solver: str) -> None:
    if not solver_supports(solver, SolverFeature.DIRECT_API):
        pytest.skip("Solver does not support direct API.")
    built = solvers.Solver.from_name(solver, simple_model, io_api="direct")
    assert built.solver_model is not None
    result = built.solve()
    simple_model.assign_result(result)

    reference = Model(chunk=None)
    rx = reference.add_variables(name="x")
    ry = reference.add_variables(name="y")
    reference.add_constraints(2 * rx + 6 * ry, GREATER_EQUAL, 10)
    reference.add_constraints(4 * rx + 2 * ry, GREATER_EQUAL, 3)
    reference.add_objective(2 * ry + rx)
    reference.solve(solver, io_api="direct")

    assert simple_model.status == "ok"
    assert simple_model.objective.value is not None
    assert reference.objective.value is not None
    assert np.isclose(simple_model.objective.value, reference.objective.value)


@pytest.mark.parametrize("solver", sorted(set(solvers.licensed_solvers)))
def test_from_name_set_names_false(simple_model: Model, solver: str) -> None:
    if not solver_supports(solver, SolverFeature.DIRECT_API):
        pytest.skip("Solver does not support direct API.")
    built = solvers.Solver.from_name(
        solver, simple_model, io_api="direct", set_names=False
    )
    result = built.solve()
    status, condition = simple_model.assign_result(result)

    assert status == "ok"
    assert condition == "optimal"
    assert simple_model.objective.value == pytest.approx(3.3)
    assert float(simple_model.variables["x"].solution) == pytest.approx(-0.1)
    assert float(simple_model.variables["y"].solution) == pytest.approx(1.7)


def test_from_name_unknown_solver_raises(simple_model: Model) -> None:
    with pytest.raises(ValueError, match="unknown solver"):
        solvers.Solver.from_name("not_a_real_solver", simple_model, io_api="direct")


@pytest.mark.skipif(
    "highs" not in set(solvers.licensed_solvers), reason="HiGHS is not installed"
)
def test_from_name_applies_solver_options(simple_model: Model) -> None:
    built = solvers.Solver.from_name(
        "highs", simple_model, io_api="direct", options={"time_limit": 123}
    )
    option_status, time_limit = built.solver_model.getOptionValue("time_limit")
    assert str(option_status) == "HighsStatus.kOk"
    assert time_limit == 123


@pytest.mark.skipif(
    "highs" not in set(solvers.licensed_solvers), reason="HiGHS is not installed"
)
def test_solver_state_compatibility_setters(simple_model: Model) -> None:
    simple_model.solver = solvers.Solver.from_name(
        "highs", simple_model, io_api="direct"
    )
    simple_model.solver_model = None
    assert simple_model.solver is None
    assert simple_model.solver_model is None
    assert simple_model.solver_name is None

    simple_model.solver = solvers.Solver.from_name(
        "highs", simple_model, io_api="direct"
    )
    simple_model.solver_name = None
    assert simple_model.solver is None
    assert simple_model.solver_model is None
    assert simple_model.solver_name is None

    with pytest.raises(AttributeError, match="managed via model.solver"):
        simple_model.solver_model = object()
    with pytest.raises(AttributeError, match="managed via model.solver"):
        simple_model.solver_name = "highs"


def test_assign_result_explicit(simple_model: Model) -> None:
    x_labels = simple_model.variables["x"].labels.values
    y_labels = simple_model.variables["y"].labels.values
    primal = np.full(simple_model._xCounter, np.nan)
    primal[int(x_labels)] = 1.5
    primal[int(y_labels)] = 2.0
    solution = Solution(primal=primal, objective=5.5)
    result = Result(
        status=Status.from_termination_condition("optimal"),
        solution=solution,
        solver_name="mock",
    )
    simple_model.solver = None
    simple_model.assign_result(result)
    assert simple_model.status == "ok"
    assert simple_model.termination_condition == "optimal"
    assert simple_model.objective.value == 5.5
    assert float(simple_model.variables["x"].solution) == 1.5
    assert float(simple_model.variables["y"].solution) == 2.0


def test_assign_result_with_csr_constraints_avoids_data_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m = Model(freeze_constraints=True)
    x = m.add_variables(coords=[range(3)], name="x")
    m.add_constraints(x >= 0, name="c")
    con = m.constraints["c"]
    assert isinstance(con, CSRConstraint)

    primal = np.arange(m._xCounter, dtype=float)
    dual = np.arange(m._cCounter, dtype=float) + 10
    result = Result(
        status=Status.from_termination_condition("optimal"),
        solution=Solution(primal=primal, dual=dual, objective=1.0),
        solver_name="mock",
    )

    def fail_data(self: CSRConstraint) -> None:
        raise AssertionError("CSRConstraint.data was accessed")

    monkeypatch.setattr(CSRConstraint, "data", property(fail_data))
    m.assign_result(result)

    np.testing.assert_array_equal(m.variables["x"].solution.values, primal)
    np.testing.assert_array_equal(m.constraints["c"].dual.values, dual)


@pytest.mark.skipif(
    "gurobi" not in set(solvers.licensed_solvers), reason="Gurobi is not installed"
)
def test_gurobi_env_persists_after_solve(simple_model: Model) -> None:
    simple_model.solve("gurobi", io_api="direct")
    assert simple_model.solver is not None
    assert simple_model.solver.env is not None
    assert isinstance(simple_model.solver_model.NumVars, int)


@pytest.mark.parametrize("solver", sorted(set(solvers.licensed_solvers)))
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


@pytest.mark.parametrize("solver", set(solvers.licensed_solvers))
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
    "knitro" not in set(solvers.licensed_solvers), reason="Knitro is not installed"
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
    "knitro" not in set(solvers.licensed_solvers), reason="Knitro is not installed"
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
    "knitro" not in set(solvers.licensed_solvers), reason="Knitro is not installed"
)
def test_knitro_solver_with_options(tmp_path: Path) -> None:
    """Test Knitro solver with custom options."""
    knitro = solvers.Knitro(options={"maxit": 100, "feastol": 1e-6})

    mps_file = tmp_path / "problem.mps"
    mps_file.write_text(free_mps_problem)
    sol_file = tmp_path / "solution.sol"
    log_file = tmp_path / "knitro.log"

    result = knitro.solve_problem(
        problem_fn=mps_file, solution_fn=sol_file, log_fn=log_file
    )
    assert result.status.is_ok


@pytest.mark.skipif(
    "knitro" not in set(solvers.licensed_solvers), reason="Knitro is not installed"
)
def test_knitro_solver_with_model_raises_error(model: Model) -> None:  # noqa: F811
    """Test Knitro solver raises NotImplementedError for model-based solving."""
    knitro = solvers.Knitro()
    with pytest.raises(
        NotImplementedError, match="Direct API not implemented for knitro"
    ):
        knitro.solve_problem(model=model)


@pytest.mark.skipif(
    "knitro" not in set(solvers.licensed_solvers), reason="Knitro is not installed"
)
def test_knitro_solver_no_log(tmp_path: Path) -> None:
    """Test Knitro solver without log file."""
    knitro = solvers.Knitro(options={"outlev": 0})

    mps_file = tmp_path / "problem.mps"
    mps_file.write_text(free_mps_problem)
    sol_file = tmp_path / "solution.sol"

    result = knitro.solve_problem(problem_fn=mps_file, solution_fn=sol_file)

    assert result.status.is_ok


@pytest.mark.skipif(
    "gurobi" not in set(solvers.licensed_solvers), reason="Gurobi is not installed"
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
    "gurobi" not in set(solvers.licensed_solvers), reason="Gurobi is not installed"
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
        (solvers.cuPDLPx, SolverFeature.GPU_ONLY, True),
        (solvers.cuPDLPx, SolverFeature.QUADRATIC_OBJECTIVE, False),
        (solvers.Gurobi, SolverFeature.GPU_ONLY, False),
        (solvers.Xpress, SolverFeature.GPU_ONLY, False),
        (solvers.PIPS, SolverFeature.INTEGER_VARIABLES, False),
    ],
)
def test_solver_class_supports_feature(
    solver_cls: type[solvers.Solver], feature: SolverFeature, expected: bool
) -> None:
    assert solver_cls.supports(feature) is expected


def test_solver_instance_supports_matches_class() -> None:
    feature = SolverFeature.QUADRATIC_OBJECTIVE
    assert solvers.Gurobi.supports(feature) is True
    if "gurobi" in solvers.licensed_solvers:
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
    "xpress" not in set(solvers.licensed_solvers), reason="Xpress is not installed"
)
def test_xpress_gpu_feature_reflects_installed_version() -> None:
    assert solvers.Xpress.supports(
        SolverFeature.GPU_ACCELERATION
    ) == _installed_version_in("xpress", ">=9.8.0")


class TestValidateModelOnBuild:
    """Solver._build() runs solver-feature checks regardless of entry point."""

    def test_quadratic_without_qp_support_raises(self, lp_only_solver: str) -> None:
        m = Model()
        x = m.add_variables(name="x", lower=0, upper=10)
        m.add_objective(x * x, sense="min")

        with pytest.raises(ValueError, match="does not support quadratic"):
            solvers.Solver.from_name(lp_only_solver, m, io_api="lp")

    def test_semi_continuous_without_support_raises(self, lp_only_solver: str) -> None:
        m = Model()
        x = m.add_variables(name="x", lower=1, upper=10, semi_continuous=True)
        m.add_objective(x)

        with pytest.raises(ValueError, match="does not support semi-continuous"):
            solvers.Solver.from_name(lp_only_solver, m, io_api="lp")

    @pytest.mark.skipif(
        "highs" not in solvers.available_solvers, reason="HiGHS not installed"
    )
    def test_solve_without_objective_raises(self) -> None:
        m = Model()
        m.add_variables(name="x", lower=0, upper=10)
        # No objective added — both entry points should raise the same error.
        with pytest.raises(ValueError, match="No objective has been set"):
            solvers.Solver.from_name("highs", m, io_api="lp").solve()
        with pytest.raises(ValueError, match="No objective has been set"):
            m.solve("highs")


class TestSolverDoesNotMutateModel:
    """Solver.from_model() must not mutate model state (sanitize stays Model-level)."""

    @pytest.mark.skipif(
        "highs" not in solvers.available_solvers, reason="HiGHS not installed"
    )
    def test_from_model_leaves_constraints_untouched(self) -> None:
        m = Model()
        x = m.add_variables(name="x", lower=0, upper=10)
        # Constraint with a near-zero coefficient — would be sanitized away if
        # the Solver path were sanitizing on build.
        m.add_constraints(1e-12 * x + x >= 0, name="c")
        m.add_objective(x)

        before = m.constraints["c"].coeffs.values.copy()
        solvers.Solver.from_name("highs", m, io_api="lp")
        after = m.constraints["c"].coeffs.values

        assert np.allclose(before, after, equal_nan=True), (
            "Solver.from_model() must not mutate model constraints. "
            "Sanitization is a Model-level primitive; call "
            "model.constraints.sanitize_zeros() / .sanitize_infinities() "
            "explicitly before building."
        )


class TestAssignResultWiring:
    """assign_result(result, solver=...) populates model.solver."""

    @pytest.mark.skipif(
        "highs" not in solvers.available_solvers, reason="HiGHS not installed"
    )
    def test_assign_result_with_solver_wires_model_solver(self) -> None:
        m = Model()
        x = m.add_variables(name="x", lower=0, upper=10)
        m.add_objective(x, sense="min")

        assert m.solver is None
        solver = solvers.Solver.from_name("highs", m, io_api="lp")
        result = solver.solve()
        m.assign_result(result, solver=solver)

        assert m.solver is solver
        assert m.solver_model is solver.solver_model

    @pytest.mark.skipif(
        "highs" not in solvers.available_solvers, reason="HiGHS not installed"
    )
    def test_assign_result_without_solver_kwarg_leaves_solver_unset(self) -> None:
        m = Model()
        x = m.add_variables(name="x", lower=0, upper=10)
        m.add_objective(x, sense="min")

        solver = solvers.Solver.from_name("highs", m, io_api="lp")
        result = solver.solve()
        m.assign_result(result)  # no solver kwarg

        assert m.solver is None


mosek_installed = pytest.importorskip("mosek", reason="Mosek is not installed")


class TestMosekChooseSolution:
    @staticmethod
    def _make_task_mock(
        *,
        bas_solsta: object | None = None,
        itr_solsta: object | None = None,
        itg_solsta: object | None = None,
    ) -> MagicMock:
        defined = {
            mosek_installed.soltype.bas: bas_solsta,
            mosek_installed.soltype.itr: itr_solsta,
            mosek_installed.soltype.itg: itg_solsta,
        }
        task = MagicMock()
        task.solutiondef.side_effect = lambda st: defined[st] is not None
        task.getsolsta.side_effect = lambda st: defined[st]
        return task

    @pytest.mark.parametrize(
        "kwargs, expected_soltype",
        [
            pytest.param(
                dict(
                    bas_solsta=mosek_installed.solsta.optimal,
                    itr_solsta=mosek_installed.solsta.dual_infeas_cer,
                ),
                mosek_installed.soltype.bas,
                id="prefers_bas_when_itr_is_farkas",
            ),
            pytest.param(
                dict(
                    bas_solsta=mosek_installed.solsta.optimal,
                    itr_solsta=mosek_installed.solsta.optimal,
                ),
                mosek_installed.soltype.itr,
                id="prefers_itr_on_tie",
            ),
            pytest.param(
                dict(itr_solsta=mosek_installed.solsta.optimal),
                mosek_installed.soltype.itr,
                id="only_itr_defined",
            ),
            pytest.param(
                dict(bas_solsta=mosek_installed.solsta.optimal),
                mosek_installed.soltype.bas,
                id="only_bas_defined",
            ),
            pytest.param(
                dict(),
                None,
                id="nothing_defined",
            ),
            pytest.param(
                dict(itg_solsta=mosek_installed.solsta.integer_optimal),
                mosek_installed.soltype.itg,
                id="itg_for_mip",
            ),
            pytest.param(
                dict(
                    bas_solsta=mosek_installed.solsta.optimal,
                    itr_solsta=mosek_installed.solsta.optimal,
                    itg_solsta=mosek_installed.solsta.integer_optimal,
                ),
                mosek_installed.soltype.itg,
                id="itg_wins_over_bas_itr",
            ),
            pytest.param(
                dict(
                    bas_solsta=mosek_installed.solsta.unknown,
                    itr_solsta=mosek_installed.solsta.optimal,
                ),
                mosek_installed.soltype.itr,
                id="optimal_itr_over_unknown_bas",
            ),
            pytest.param(
                dict(
                    bas_solsta=mosek_installed.solsta.optimal,
                    itr_solsta=mosek_installed.solsta.unknown,
                ),
                mosek_installed.soltype.bas,
                id="optimal_bas_over_unknown_itr",
            ),
            pytest.param(
                dict(
                    bas_solsta=mosek_installed.solsta.prim_infeas_cer,
                    itr_solsta=mosek_installed.solsta.dual_infeas_cer,
                ),
                mosek_installed.soltype.itr,
                id="falls_back_to_itr_when_both_non_optimal",
            ),
        ],
    )
    def test_choose_solution(
        self, kwargs: dict[str, object], expected_soltype: object
    ) -> None:
        task = self._make_task_mock(**kwargs)
        assert solvers.Mosek._choose_solution(task) is expected_soltype

    @pytest.mark.skipif(
        "mosek" not in set(solvers.licensed_solvers),
        reason="Mosek is not licensed",
    )
    def test_smoke_lp(self) -> None:
        import math

        m = Model()
        x = m.add_variables(name="x", lower=0)
        m.add_constraints(2 * x >= 10, name="c1")
        m.add_objective(x)

        result = solvers.Solver.from_name("mosek", m).solve()

        assert result.status.is_ok
        assert result.solution is not None
        assert math.isfinite(result.solution.objective)
        assert result.solution.objective == pytest.approx(5.0, abs=1e-3)
