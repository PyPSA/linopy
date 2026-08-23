"""
Tests for the NVIDIA cuOpt solver.

cuOpt runs through the shared solver suites (``test_optimization.py``,
``test_solvers.py``) under ``--run-gpu``, and its user-facing quirks and
limitations are documented in ``doc/gpu-acceleration.rst``. This module only
pins what neither of those can guard: the places where linopy compensates for
a cuOpt quirk whose failure mode is *silent* -- cuOpt reporting ``Optimal``
with negated duals, a half-scaled Hessian solution or an empty primal -- plus
the process-safety machinery (persistent worker thread, subprocess device
probe) whose regressions only surface as crashes in user processes.

Numeric expectations are baked into the tests: each fixture was designed to
have an analytic optimum (derived in its docstring) and every baked value was
confirmed with HiGHS 1.15.1 before baking. The fixtures are frozen -- changing
one invalidates its expected values.
"""

from __future__ import annotations

import _thread
import contextlib
import os
import select
import signal
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy.sparse import csr_array

import linopy
from linopy import Model, solvers
from linopy.solvers import (
    _cuopt_solve_queue,
    _run_cuopt_with_keyboard_interrupt,
    licensed_solvers,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not os.environ.get("LINOPY_RUN_GPU_TESTS"),
        reason="need --run-gpu option to run GPU tests",
    ),
    pytest.mark.skipif(
        "cuopt" not in licensed_solvers, reason="cuOpt is not installed"
    ),
]

# Tolerances for comparing cuOpt's results against the baked expectations.
# cuOpt is a GPU solver whose own default tolerances are 1e-4, so these are
# chosen from the measured residuals rather than from taste: under linopy's
# default method (Barrier) the duals of the models below agree with the exact
# values to ~2e-09 and the primals to ~6e-08, i.e. every number here has at
# least two orders of margin. A dual *sign* error, in contrast, is 2*|dual|
# -- six orders above them.
CUOPT_OBJ_RTOL: float = 1e-6
CUOPT_OBJ_ATOL: float = 1e-6
CUOPT_PRIMAL_RTOL: float = 1e-6
CUOPT_PRIMAL_ATOL: float = 1e-6
CUOPT_DUAL_RTOL: float = 1e-6
CUOPT_DUAL_ATOL: float = 1e-7
# QP duals of a non-binding row come back as ~1e-10 where the exact value is
# 0, so the comparison needs an absolute tolerance one decade looser than the
# LP one -- a relative tolerance against 0 can never pass.
CUOPT_DUAL_ATOL_QP: float = 1e-6
# cuOpt's own mip_integrality_tolerance default; asserting tighter would test
# the solver's tolerance rather than linopy.
CUOPT_INTEGRALITY_ATOL: float = 1e-5


# ---------------------------------------------------------------------------
# models and helpers
# ---------------------------------------------------------------------------


def sign_matrix_model(sense: str, row_sign: str, **model_kwargs: Any) -> Model:
    """
    Model behind the dual sign matrix.

    ``A = [[1, 2, 1], [3, 1, 1]]``, ``b = [4, 6]``, ``0 <= x <= 10``, with the
    objective coefficients chosen per cell so that the primal optimum is always
    the vertex ``x = (1.6, 1.2, 0)`` with both rows binding, ``x2`` at its
    lower bound and a unique, non-degenerate dual: the basis rows of
    ``A^T y = c`` give ``y = +-(0.4, 0.2)`` and an objective of ``+-2.8``.
    """
    m = Model(chunk=None, **model_kwargs)
    variables = pd.RangeIndex(3, name="i")
    rows = pd.RangeIndex(2, name="row")
    x = m.add_variables(lower=0, upper=10, coords=[variables], name="x")
    A = xr.DataArray([[1.0, 2.0, 1.0], [3.0, 1.0, 1.0]], coords=[rows, variables])
    b = xr.DataArray([4.0, 6.0], coords=[rows])
    m.add_constraints((A * x).sum("i"), row_sign, b, name="con0")
    coeffs = {"<=": [-1.0, -1.0, 0.0], ">=": [1.0, 1.0, 1.0], "=": [1.0, 1.0, 1.0]}[
        row_sign
    ]
    if sense == "max":
        coeffs = [-c for c in coeffs]
    m.add_objective((xr.DataArray(coeffs, coords=[variables]) * x).sum(), sense=sense)
    return m


SIGN_MATRIX_X: np.ndarray = np.array([1.6, 1.2, 0.0])


def square_equality_model(n: int, sense: str) -> Model:
    """
    Model that cuOpt's presolve solves outright (``solved_by == Unset``).

    ``n / 2`` equality rows, each pinning one pair of variables, so the system
    is square and presolve finishes it without calling a method. Handing such a
    model to cuOpt as a maximisation returns negated duals, which is why the
    solver class always minimises instead.

    For ``n = 2`` the optimum is analytic: one row ``x0 + 2 x1 = 10`` with
    positive costs ``c = 1 + rng(5).random(2)`` maximised at ``x = (10, 0)``,
    so the objective is ``10 * c[0]`` and the row's dual is ``c[0]``.
    """
    m = Model(chunk=None)
    rng = np.random.default_rng(5)
    variables = pd.RangeIndex(n, name="i")
    rows = pd.RangeIndex(n // 2, name="row")
    x = m.add_variables(lower=0.0, upper=100.0, coords=[variables], name="x")
    A = np.zeros((n // 2, n))
    for i in range(n // 2):
        A[i, 2 * i] = i + 1.0
        A[i, 2 * i + 1] = i + 2.0
    b = 10.0 + np.arange(n // 2, dtype=float)
    m.add_constraints(
        (xr.DataArray(A, coords=[rows, variables]) * x).sum("i"),
        "=",
        xr.DataArray(b, coords=[rows]),
        name="con0",
    )
    coeffs = xr.DataArray(1.0 + rng.random(n), coords=[variables])
    m.add_objective((coeffs * x).sum(), sense=sense)
    return m


def random_lp(n: int, m_rows: int, seed: int = 0) -> Model:
    """Dense random LP with ``n`` variables and ``m_rows`` ``<=`` rows."""
    rng = np.random.default_rng(seed)
    m = Model(chunk=None)
    variables = pd.RangeIndex(n, name="i")
    rows = pd.RangeIndex(m_rows, name="row")
    x = m.add_variables(lower=0, upper=10, coords=[variables], name="x")
    A = xr.DataArray(rng.random((m_rows, n)), coords=[rows, variables])
    b = xr.DataArray(rng.random(m_rows) * 100 + 100, coords=[rows])
    m.add_constraints((A * x).sum("i"), "<=", b, name="con0")
    coeffs = xr.DataArray(-rng.random(n) - 1.0, coords=[variables])
    m.add_objective((coeffs * x).sum())
    return m


def milp_model(sense: str = "min") -> Model:
    """
    MILP with a continuous and an integer variable block, bounded in both senses.

    Ten independent copies of a two-variable problem. Per copy: ``min`` puts
    the integer at its cap and covers the rest continuously (``y = 9``,
    ``x = 0.5``, cost 10); ``max`` fills the continuous variable first
    (``x = 5``, ``y = 4``, value 14). Objectives: 100 and 140.
    """
    m = Model(chunk=None)
    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, 5, name="x")
    y = m.add_variables(lower, 9, name="y", integer=True)
    m.add_constraints(x + y, "<=" if sense == "max" else ">=", 9.5, name="con0")
    m.add_objective(2 * x + y, sense=sense)
    return m


MILP_OBJECTIVE: dict[str, float] = {"min": 100.0, "max": 140.0}


# The coefficients the three-variable QP below has to produce. cuOpt minimises
# `c'x + x'Qx` and symmetrises Q to `Q + Q'`, while linopy's `M.Q` is the
# Hessian of `0.5 x'Qx`, so the solver class halves it. Pinning both matrices
# is what turns the analytic comparison into a convention guard: if linopy
# ever hands over a different form, the test says so instead of drifting
# quietly.
QP_C: np.ndarray = np.array([-3.0, -1.0, 2.0])
QP_Q: np.ndarray = np.array([[2.0, 1.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
# The analytic optimum of ``three_variable_qp``: the unconstrained minimiser
# (grad f = 0) lies strictly inside the box and inside both rows.
QP_X: np.ndarray = np.array([11 / 7, -1 / 7, -2.0])
QP_OBJECTIVE: float = -30 / 7


def three_variable_qp(sense: str = "min") -> Model:
    """
    Dense-Hessian QP with a cross term, an off-diagonal zero and free signs.

    Its unconstrained optimum ``(11/7, -1/7, -2)`` lies strictly inside the box
    and inside both rows, so the primal is unique, the duals are zero and every
    component is determined. ``max`` negates the whole objective rather than
    only ``c``: maximising the same convex form is non-concave and neither
    solver would accept it.
    """
    m = Model(chunk=None)
    x0 = m.add_variables(lower=-10.0, upper=10.0, name="x0")
    x1 = m.add_variables(lower=-10.0, upper=10.0, name="x1")
    x2 = m.add_variables(lower=-10.0, upper=10.0, name="x2")
    m.add_constraints(x0 + x1 + x2, ">=", -5.0, name="con0")
    m.add_constraints(x0 + 2 * x1, "<=", 8.0, name="con1")
    objective = (
        x0 * x0 + 2 * x1 * x1 + 0.5 * x2 * x2 + x0 * x1 - 3 * x0 - 1 * x1 + 2 * x2
    )
    m.add_objective(objective if sense == "min" else -objective, sense=sense)
    return m


def integer_quadratic_model(mixed: bool) -> Model:
    """MIQP (``mixed``) or pure-integer QP, both of which cuOpt must refuse."""
    m = Model(chunk=None)
    x = m.add_variables(lower=0, upper=10, name="x", integer=True)
    m.add_constraints(1 * x, ">=", 1, name="con0")
    objective = x * x
    if mixed:
        y = m.add_variables(lower=0, upper=10, name="y")
        m.add_constraints(1 * y, ">=", 1, name="con1")
        objective = objective + 1 * y
    m.add_objective(objective)
    return m


def run_in_subprocess(
    script: str, timeout: float = 600, **env: str
) -> subprocess.CompletedProcess[str]:
    """
    Run ``script`` in a fresh interpreter against the same linopy as this test.

    Used wherever a regression would take the test session down with it (a
    segfault on repeated solves) or where process-global state has to start
    clean (the CUDA device probe).
    """
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        p
        for p in (str(Path(linopy.__file__).parents[1]), environment.get("PYTHONPATH"))
        if p
    )
    environment.update(env)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=environment,
    )


def raw_result(model: Model, data_model: Any) -> Any:
    """
    Run a hand-built ``DataModel`` through the solver class.

    Lets a test reach the status branches that linopy's own model translation
    deliberately avoids -- malformed input arrays, an unpadded empty constraint
    matrix -- while still going through the production status mapping.
    """
    instance = solvers.cuOpt.from_model(model, io_api="direct")
    instance.solver_model = data_model
    return instance._run_direct()


# ---------------------------------------------------------------------------
# sign conventions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sense,row_sign,expected_objective,expected_duals",
    [
        ("min", "<=", -2.8, (-0.4, -0.2)),
        ("min", ">=", 2.8, (0.4, 0.2)),
        ("min", "=", 2.8, (0.4, 0.2)),
        ("max", "<=", 2.8, (0.4, 0.2)),
        ("max", ">=", -2.8, (-0.4, -0.2)),
        ("max", "=", -2.8, (-0.4, -0.2)),
    ],
    ids=["min-le", "min-ge", "min-eq", "max-le", "max-ge", "max-eq"],
)
def test_sign_matrix(
    sense: str,
    row_sign: str,
    expected_objective: float,
    expected_duals: tuple[float, float],
) -> None:
    """Duals, primals and objective match the KKT values in all six cells."""
    model = sign_matrix_model(sense, row_sign)
    status, condition = model.solve("cuopt", io_api="direct", log_to_console=False)

    assert status == "ok"
    assert model.objective.value == pytest.approx(
        expected_objective, rel=CUOPT_OBJ_RTOL, abs=CUOPT_OBJ_ATOL
    )
    assert np.allclose(
        np.asarray(model.solution.x),
        SIGN_MATRIX_X,
        rtol=CUOPT_PRIMAL_RTOL,
        atol=CUOPT_PRIMAL_ATOL,
    )
    duals = np.asarray(model.constraints["con0"].dual)
    # A one-row shift from the pad-row slice (see test_model_without_constraints)
    # would misalign every dual, so the shape is part of the contract.
    assert duals.shape == (2,)
    assert np.allclose(
        duals, expected_duals, rtol=CUOPT_DUAL_RTOL, atol=CUOPT_DUAL_ATOL
    )


def test_presolve_max_duals() -> None:
    """
    The case the six-cell matrix misses.

    cuOpt returns *negated* duals, with status ``Optimal`` and a correct
    objective, for a maximisation its presolve solves outright. The solver
    class avoids the branch by always handing cuOpt a minimisation; without
    that, these duals are off by ``2 * |dual|``.
    """
    expected_dual = 1.0 + np.random.default_rng(5).random(2)[0]
    model = square_equality_model(2, "max")
    status, condition = model.solve("cuopt", io_api="direct", log_to_console=False)

    assert status == "ok"
    assert model.objective.value == pytest.approx(
        10 * expected_dual, rel=CUOPT_OBJ_RTOL, abs=CUOPT_OBJ_ATOL
    )
    duals = np.asarray(model.constraints["con0"].dual)
    assert duals.shape == (1,)
    assert np.allclose(
        duals, [expected_dual], rtol=CUOPT_DUAL_RTOL, atol=CUOPT_DUAL_ATOL
    )


def test_presolve_branch_still_reached() -> None:
    """
    Coverage precondition for ``test_presolve_max_duals``, not a correctness claim.

    A red here means cuOpt changed its presolve routing, so the coverage of
    ``test_presolve_max_duals`` has to be re-established on another model. It
    never means the always-minimise transformation should be removed.
    """
    model = square_equality_model(2, "max")
    M = model.matrices
    assert M.A is not None
    A = M.A.tocsr()
    data_model = solvers.cuopt.linear_programming.DataModel()
    data_model.set_csr_constraint_matrix(A.data, A.indices, A.indptr)
    data_model.set_constraint_lower_bounds(M.b)
    data_model.set_constraint_upper_bounds(M.b)
    data_model.set_variable_lower_bounds(M.lb)
    data_model.set_variable_upper_bounds(M.ub)
    data_model.set_objective_coefficients(M.c)
    data_model.set_maximize(True)
    solution = solvers.cuopt.linear_programming.Solve(
        data_model, solvers.cuopt.linear_programming.SolverSettings()
    )

    assert solution.get_termination_reason() == "Optimal"
    assert solution.get_solved_by().name == "Unset"


# ---------------------------------------------------------------------------
# mixed integer problems
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sense", ["min", "max"])
def test_milp_solution_and_bound_report(sense: str) -> None:
    """Objective, integrality, empty duals and a usable MIP bound."""
    model = milp_model(sense)
    status, condition = model.solve("cuopt", io_api="direct", log_to_console=False)

    assert status == "ok"
    assert model.objective.value == pytest.approx(
        MILP_OBJECTIVE[sense], rel=CUOPT_OBJ_RTOL, abs=CUOPT_OBJ_ATOL
    )
    integers = np.asarray(model.solution.y)
    assert np.allclose(integers, np.round(integers), atol=CUOPT_INTEGRALITY_ATOL)

    # cuOpt refuses duals for a mixed integer problem.
    assert model.solver is not None
    assert model.solver.solution is not None
    assert model.solver.solution.dual.size == 0

    report = model.solver.report
    assert report is not None
    assert report.dual_bound is not None
    assert report.mip_gap is not None
    # The bound is a bound: above the objective for a maximisation, below it
    # for a minimisation. Negating it along with the objective is what makes
    # this hold for `max`.
    if sense == "max":
        assert report.dual_bound >= MILP_OBJECTIVE[sense] - CUOPT_OBJ_ATOL
    else:
        assert report.dual_bound <= MILP_OBJECTIVE[sense] + CUOPT_OBJ_ATOL


# ---------------------------------------------------------------------------
# termination status mapping
# ---------------------------------------------------------------------------


def infeasible_lp() -> Model:
    m = Model(chunk=None)
    x = m.add_variables(lower=0, upper=10, name="x")
    m.add_constraints(1 * x, ">=", 2, name="con0")
    m.add_constraints(1 * x, "<=", 1, name="con1")
    m.add_objective(1 * x)
    return m


def unbounded_lp(integer: bool = False) -> Model:
    m = Model(chunk=None)
    x = m.add_variables(lower=0, name="x", integer=integer)
    m.add_constraints(1 * x, ">=", 0, name="con0")
    m.add_objective(-x)
    return m


def infeasible_milp() -> Model:
    m = Model(chunk=None)
    x = m.add_variables(lower=0, upper=10, name="x", integer=True)
    m.add_constraints(2 * x, "=", 1, name="con0")
    m.add_objective(1 * x)
    return m


@pytest.mark.parametrize(
    "build,expected",
    [
        (lambda: sign_matrix_model("min", "<="), "optimal"),
        (infeasible_lp, "infeasible"),
        (unbounded_lp, "infeasible_or_unbounded"),
        (lambda: milp_model("min"), "optimal"),
        (infeasible_milp, "infeasible"),
        (lambda: unbounded_lp(integer=True), "infeasible_or_unbounded"),
    ],
    ids=[
        "lp_optimal",
        "lp_infeasible",
        "lp_unbounded",
        "milp_optimal",
        "milp_infeasible",
        "milp_unbounded",
    ],
)
def test_status_map(build: Callable[[], Model], expected: str) -> None:
    """
    The outcome statuses map onto the right condition for both problem classes.

    LP and MILP rows are separate because cuOpt keeps two termination enums
    whose members collide (``MILP.Infeasible == LP.PrimalInfeasible == 2``), so
    the solver class holds one map per problem category. The limit statuses
    (time, iteration and node limits, ``first_primal_feasible``) were measured
    once and are documented in ``doc/gpu-acceleration.rst``; ``TimeLimit`` is
    still hit live by ``test_time_limit_does_not_scatter_a_partial_primal``,
    and a status linopy does not recognise maps to ``unknown``
    (``test_status_map_unknown_status``).
    """
    model = build()
    status, condition = model.solve("cuopt", io_api="direct", log_to_console=False)
    assert condition == expected


def test_status_map_empty_constraint_matrix() -> None:
    """A constraint matrix without nonzeros is an error, not a silent success."""
    model = sign_matrix_model("min", "<=")
    M = model.matrices
    empty = csr_array((len(M.b), len(M.c)))
    data_model = solvers.cuopt.linear_programming.DataModel()
    data_model.set_csr_constraint_matrix(empty.data, empty.indices, empty.indptr)
    data_model.set_constraint_lower_bounds(np.full(len(M.b), -np.inf))
    data_model.set_constraint_upper_bounds(M.b)
    data_model.set_variable_lower_bounds(M.lb)
    data_model.set_variable_upper_bounds(M.ub)
    data_model.set_objective_coefficients(M.c)

    result = raw_result(model, data_model)

    assert result.status.legacy_status == "NoTermination"
    assert result.status.termination_condition.value == "internal_solver_error"


def test_status_map_mismatched_variable_types() -> None:
    """A malformed input array must not be reported as a solved model."""
    model = milp_model()
    data_model = solvers.cuOpt._build_solver_model(model)
    data_model.set_variable_types(np.array(["I"], dtype="<U1"))

    result = raw_result(model, data_model)

    assert result.status.legacy_status == "NoTermination"
    assert result.status.termination_condition.value == "internal_solver_error"


def test_status_map_numerical_error() -> None:
    """A concave quadratic objective makes cuOpt fail numerically."""
    model = sign_matrix_model("min", "<=")
    data_model = solvers.cuOpt._build_solver_model(model)
    Q = csr_array(np.diag([-1.0, 0.0, 0.0]))
    data_model.set_quadratic_objective_matrix(Q.data, Q.indices, Q.indptr)

    result = raw_result(model, data_model)

    assert result.status.legacy_status == "NumericalError"
    assert result.status.termination_condition.value == "internal_solver_error"


def test_status_map_unknown_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """A status cuOpt adds in a future release maps to ``unknown``."""

    class FakeSolution:
        def get_problem_category(self) -> Any:
            return type("Category", (), {"name": "LP"})()

        def get_termination_reason(self) -> str:
            return "SomeFutureStatus"

        def get_error_status(self) -> int:
            return 0

        def get_primal_solution(self) -> np.ndarray:
            return np.array([])

        def get_solve_time(self) -> float:
            return 0.0

    class FakeLinearProgramming:
        @staticmethod
        def SolverSettings() -> Any:
            return type("Settings", (), {"set_parameter": lambda self, k, v: None})()

        @staticmethod
        def Solve(data_model: Any, settings: Any) -> FakeSolution:
            return FakeSolution()

    model = sign_matrix_model("min", "<=")
    instance = solvers.cuOpt.from_model(model, io_api="direct")
    monkeypatch.setattr(
        solvers,
        "cuopt",
        type("FakeCuopt", (), {"linear_programming": FakeLinearProgramming})(),
    )

    result = instance._run_direct()

    assert result.status.legacy_status == "SomeFutureStatus"
    assert result.status.termination_condition.value == "unknown"


def test_time_limit_does_not_scatter_a_partial_primal() -> None:
    """
    A limit termination is ``ok``, so the primal has to be checked, not trusted.

    cuOpt returns an empty primal for some limit terminations; scattering it
    would misalign every label. Either a full-length solution comes back or an
    empty one does -- never a partial vector, and never an exception.
    """
    model = random_lp(400, 300)
    status, condition = model.solve(
        "cuopt", io_api="direct", log_to_console=False, time_limit=1e-6
    )
    assert status == "ok"
    assert model.solver is not None
    assert model.solver.solution is not None
    primal = model.solver.solution.primal
    assert primal.size in (0, model.nvars)


# ---------------------------------------------------------------------------
# models without constraint nonzeros
# ---------------------------------------------------------------------------


def test_model_without_constraints() -> None:
    """
    A padded constraint row keeps cuOpt happy and must not leak into the result.

    cuOpt rejects a model without constraint nonzeros, so linopy pads one row
    that repeats a variable's own bounds -- not ``(-inf, +inf)``, which makes a
    quadratic objective fail with ``NumericalError`` (the shared
    ``test_quadratic_model_wo_constraint`` exercises that shape). The row
    leaves the feasible set untouched and its dual is sliced off again before
    the solution is built.
    """
    model = Model(chunk=None)
    x = model.add_variables(lower=0, upper=10, name="x")
    model.add_objective(-x)

    status, condition = model.solve("cuopt", io_api="direct", log_to_console=False)

    assert status == "ok"
    assert not len(model.constraints)
    assert model.objective.value == pytest.approx(
        -10.0, rel=CUOPT_OBJ_RTOL, abs=CUOPT_OBJ_ATOL
    )
    assert model.solver is not None
    assert model.solver.solution is not None
    assert model.solver.solution.dual.size == 0


def test_model_without_constraints_or_bounds_raises() -> None:
    """No row can be padded from a model that has no finite bound either."""
    model = Model(chunk=None)
    x = model.add_variables(name="x")
    model.add_objective(1 * x)

    with pytest.raises(NotImplementedError, match="no finite variable bounds"):
        model.solve("cuopt", io_api="direct")


# ---------------------------------------------------------------------------
# quadratic objectives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sense", ["min", "max"])
def test_quadratic_optimum(sense: str) -> None:
    """
    Objective, primal and duals of a determined QP against the analytic optimum.

    This is the guard on the ``0.5 * M.Q`` convention: the naive ``M.Q`` also
    returns ``Optimal``, with half the solution vector and an objective off by
    50% -- see ``test_naive_hessian_changes_the_answer``, which measures
    exactly that.
    """
    sign = 1.0 if sense == "min" else -1.0
    model = three_variable_qp(sense)
    matrices = model.matrices
    assert np.allclose(matrices.c, sign * QP_C)
    assert matrices.Q is not None
    assert np.allclose(matrices.Q.toarray(), sign * QP_Q)

    status, condition = model.solve("cuopt", io_api="direct", log_to_console=False)

    assert status == "ok"
    assert model.objective.value == pytest.approx(
        sign * QP_OBJECTIVE, rel=CUOPT_OBJ_RTOL, abs=CUOPT_OBJ_ATOL
    )
    for name, expected in zip(("x0", "x1", "x2"), QP_X):
        assert float(model.solution[name]) == pytest.approx(
            expected, rel=CUOPT_PRIMAL_RTOL, abs=CUOPT_PRIMAL_ATOL
        )
    # Neither row binds at the interior optimum, so both duals are exactly
    # zero. What is being checked is that a QP returns duals at all, aligned
    # and not scaled: the LP sign matrix covers the sign.
    for name in ("con0", "con1"):
        duals = np.asarray(model.constraints[name].dual)
        assert duals.shape == (1,) or duals.shape == ()
        assert np.allclose(duals, 0.0, atol=CUOPT_DUAL_ATOL_QP)


def test_naive_hessian_changes_the_answer() -> None:
    """
    The deliberate failure that makes the convention guard above a guard.

    Handing cuOpt ``M.Q`` where the solver class hands ``0.5 * M.Q`` is the
    classic factor-of-two error, and cuOpt reports it as ``Optimal``: only the
    comparison against the true optimum catches it. Reproduced here through
    the production status mapping so it is provable that the wrong answer is
    reachable and that the tolerances above would reject it.
    """
    model = three_variable_qp("min")
    data_model = solvers.cuOpt._build_solver_model(model)
    assert model.matrices.Q is not None
    naive = csr_array(model.matrices.Q.tocsr())
    data_model.set_quadratic_objective_matrix(naive.data, naive.indices, naive.indptr)

    result = raw_result(model, data_model)

    assert result.status.termination_condition.value == "optimal"
    assert abs(result.solution.objective - QP_OBJECTIVE) / abs(QP_OBJECTIVE) > 1e-3


@pytest.mark.parametrize("mixed", [False, True], ids=["iqp", "miqp"])
def test_integer_quadratic_is_refused(
    mixed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    MIQP is rejected before cuOpt is touched, because cuOpt does not reject it.

    Handed a quadratic objective with integer variables, cuOpt returns
    ``NoTermination`` with ``obj=0.0`` and an empty solution -- indistinguishable
    from a failed solve. A pre-check is the only honest answer, so this also
    asserts that no call reaches the library.
    """
    calls: list[str] = []

    class Recorder:
        def __getattr__(self, name: str) -> Any:
            calls.append(name)
            raise AssertionError(f"cuOpt was reached ({name}) for an integer QP")

    model = integer_quadratic_model(mixed)
    assert model.type == ("MIQP" if mixed else "IQP")
    monkeypatch.setattr(solvers, "cuopt", Recorder())

    with pytest.raises(NotImplementedError, match="MIQP"):
        model.solve("cuopt", io_api="direct")

    assert calls == []


# ---------------------------------------------------------------------------
# repeated solves, device probe and process safety
# ---------------------------------------------------------------------------


def test_repeated_solves_of_a_medium_model() -> None:
    """
    Twenty sequential solves in one process, above the size that used to crash.

    cuOpt's own default method segfaults on the second or third solve of a
    model with more than about 1300 variables, which is why linopy picks
    Barrier instead. Run in a subprocess so a regression is a failed test
    rather than a dead test session.

    Twenty rather than three because three sits *below* the failure it is
    supposed to catch: the fresh-thread OpenMP abort reported upstream
    (NVIDIA/cuopt#1768) was measured appearing only after 5 to 13 LP solves, so
    a three-solve run would have passed on defective code. Twenty clears that
    measured upper bound of 13 with a margin of 1.54x. The ~135 s this costs is
    the price of the check; shrinking the count to save suite time weakens it.
    """
    result = run_in_subprocess(
        """
        import numpy as np
        import pandas as pd
        import xarray as xr
        from linopy import Model

        rng = np.random.default_rng(7)
        variables = pd.RangeIndex(2000, name="i")
        rows = pd.RangeIndex(1000, name="row")
        for _ in range(20):
            m = Model(chunk=None)
            x = m.add_variables(lower=0, upper=10, coords=[variables], name="x")
            A = xr.DataArray(rng.random((1000, 2000)), coords=[rows, variables])
            b = xr.DataArray(rng.random(1000) * 100 + 100, coords=[rows])
            m.add_constraints((A * x).sum("i"), "<=", b, name="con0")
            coeffs = xr.DataArray(-rng.random(2000) - 1.0, coords=[variables])
            m.add_objective((coeffs * x).sum())
            status, condition = m.solve("cuopt", io_api="direct", log_to_console=False)
            print(condition, flush=True)
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.count("optimal") == 20
    for marker in ("Fatal Python error", "OMP:", "kmp_alloc"):
        assert marker not in result.stdout + result.stderr


def test_unavailable_without_a_cuda_device() -> None:
    """
    Without a usable device cuOpt is simply absent, with a warning that says why.

    ``import cuopt`` succeeds on a machine without a GPU, so availability has
    to be decided by probing the device rather than the import. Only runnable
    on a GPU machine, by masking the devices.
    """
    result = run_in_subprocess(
        """
        import logging
        logging.basicConfig(level=logging.WARNING)
        import linopy
        print("cuopt" in linopy.available_solvers, flush=True)
        """,
        CUDA_VISIBLE_DEVICES="",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("False")
    assert "no usable CUDA device was found" in result.stderr


def test_device_probe_is_fork_safe() -> None:
    """
    Probing the device must not initialise CUDA in the calling process.

    ``available_solvers`` is touched on import in plenty of code bases, and a
    CUDA context created before ``os.fork()`` makes every cuOpt solve in the
    child fail -- which is why ``is_available`` probes the device in a
    subprocess instead of calling ``cudaGetDeviceCount`` in-process.
    """
    result = run_in_subprocess(
        """
        import os
        import sys

        import linopy

        assert "cuopt" in linopy.available_solvers


        def build():
            m = linopy.Model(chunk=None)
            x = m.add_variables(lower=0, upper=10, name="x")
            m.add_constraints(1 * x, "<=", 4, name="con0")
            m.add_objective(-x)
            return m

        pid = os.fork()
        if pid == 0:
            try:
                status, condition = build().solve(
                    "cuopt", io_api="direct", log_to_console=False
                )
                os._exit(0 if condition == "optimal" else 3)
            except BaseException:
                os._exit(4)
        sys.exit(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]))
        """
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# the file io fallback
# ---------------------------------------------------------------------------


def test_bare_solve_falls_back_to_the_direct_api(tmp_path: Path) -> None:
    """
    ``model.solve("cuopt")`` without an ``io_api`` solves through the direct API.

    It must also leave no problem file behind: ``Model.solve`` creates the file
    before building and unlinks only what the solver hands back, so an override
    that keeps quiet about it accumulates an empty file on every bare solve.
    ``tmp_path`` is a fresh directory on purpose -- the default ``solver_dir``
    is shared with every other test.
    """
    model = sign_matrix_model("min", "<=", solver_dir=tmp_path)
    status, condition = model.solve("cuopt")

    assert condition == "optimal"
    assert model.objective.value == pytest.approx(
        -2.8, rel=CUOPT_OBJ_RTOL, abs=CUOPT_OBJ_ATOL
    )
    assert list(Path(model.solver_dir).glob("linopy-problem-*")) == []


# ---------------------------------------------------------------------------
# keyboard interrupt and the persistent worker thread
# ---------------------------------------------------------------------------
# These tests use a dummy solve and need neither cuOpt nor a GPU; they ride
# this module's GPU gate to keep every cuOpt test in one file.


class DummySolve:
    """
    Stand-in for cuOpt's ``Solve``: blocking, and with no cancel API.

    cuOpt defers SIGINT for the whole duration of its C++ solve -- measured at
    52.9 s on a model that takes that long -- so linopy runs the call in a
    worker thread and waits in the main one. The GPU work keeps going in the
    background afterwards, which this dummy reproduces.
    """

    def __init__(self, duration: float = 0.5) -> None:
        self.duration = duration
        self.started = threading.Event()
        self.finished = threading.Event()

    def __call__(self) -> str:
        self.started.set()
        time.sleep(self.duration)
        self.finished.set()
        return "solution"


def test_run_cuopt_interrupt_reaches_the_main_thread() -> None:
    dummy = DummySolve()

    def interrupter() -> None:
        assert dummy.started.wait(timeout=1)
        _thread.interrupt_main()

    threading.Thread(target=interrupter, daemon=True).start()

    start = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        _run_cuopt_with_keyboard_interrupt(dummy)
    elapsed = time.monotonic() - start

    assert elapsed < 1.0
    assert dummy.finished.wait(timeout=5)


def test_run_cuopt_returns_the_solve_result() -> None:
    assert _run_cuopt_with_keyboard_interrupt(lambda: "solution") == "solution"


def test_run_cuopt_reraises_solver_errors() -> None:
    def boom() -> None:
        raise RuntimeError("solver failed")

    with pytest.raises(RuntimeError, match="solver failed"):
        _run_cuopt_with_keyboard_interrupt(boom)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX only")
def test_solve_queue_starts_a_fresh_worker_after_fork() -> None:
    """
    A forked child inherits the cached queue but not its daemon worker.

    Without the at-fork ``cache_clear`` the child hands its job to a queue
    nobody reads and waits on ``job.done`` forever; it then never writes to the
    pipe and the parent's ``select`` timeout below fails the test.
    """
    assert _run_cuopt_with_keyboard_interrupt(lambda: "parent") == "parent"

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child -- never returns
        exit_code = 1
        try:
            os.close(read_fd)
            cleared = _cuopt_solve_queue.cache_info().currsize == 0
            solved = _run_cuopt_with_keyboard_interrupt(lambda: "child") == "child"
            if cleared and solved:
                os.write(write_fd, b"ok")
                exit_code = 0
        finally:
            os._exit(exit_code)

    os.close(write_fd)
    reaped = False
    try:
        ready, _, _ = select.select([read_fd], [], [], 30)
        assert ready, "the forked child never finished its solve"
        assert os.read(read_fd, 8) == b"ok"
        wait_status = os.waitpid(pid, 0)[1]
        reaped = True
        assert os.waitstatus_to_exitcode(wait_status) == 0
    finally:
        os.close(read_fd)
        # Once the child is reaped its pid is free for reuse, so signalling it
        # again could land on an unrelated process.
        if not reaped:
            with contextlib.suppress(ChildProcessError, ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
