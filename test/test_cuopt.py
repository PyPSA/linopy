"""
Tests for the NVIDIA cuOpt solver.

Every numeric expectation in this module comes from solving the identical model
with HiGHS, live and in the same process -- from a deep copy for linear models,
from a second build for quadratic ones (see ``solve_qp_with_both``). No expected
value is copied in from an earlier run: a differential test with a baked-in
expectation is a regression test with an unverified baseline, which is how a
systematic sign error survives a green suite.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap
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
from linopy.solver_capabilities import SOLVER_REGISTRY
from linopy.solvers import SolverName, licensed_solvers, quadratic_solvers

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not os.environ.get("LINOPY_RUN_GPU_TESTS"),
        reason="need --run-gpu option to run GPU tests",
    ),
    pytest.mark.skipif(
        "cuopt" not in licensed_solvers, reason="cuOpt is not installed"
    ),
    pytest.mark.skipif(
        "highs" not in licensed_solvers, reason="HiGHS is not installed"
    ),
]

# Tolerances for the differential comparisons against HiGHS. cuOpt is a GPU
# solver whose own default tolerances are 1e-4, so these are chosen from the
# measured residuals rather than from taste: under linopy's default method
# (Barrier) the duals of the models below agree with HiGHS to ~2e-09 and the
# primals to ~6e-08, i.e. every number here has at least two orders of margin.
# A dual *sign* error, in contrast, is 2*|dual| -- six orders above them.
CUOPT_OBJ_RTOL: float = 1e-6
CUOPT_OBJ_ATOL: float = 1e-6
CUOPT_PRIMAL_RTOL: float = 1e-6
CUOPT_PRIMAL_ATOL: float = 1e-6
# For the singular QP below, where HiGHS itself reports a primal-dual objective
# error of 1e-4: both solvers are only ~1e-4-accurate there, and 1e-4 is also
# cuOpt's own absolute_primal_tolerance default. Not for use anywhere else.
CUOPT_PRIMAL_ATOL_DEGENERATE: float = 1e-4
CUOPT_DUAL_RTOL: float = 1e-6
CUOPT_DUAL_ATOL: float = 1e-7
# QP duals of a non-binding row come back as ~1e-10 where HiGHS returns an
# exact 0, so the comparison needs an absolute tolerance one decade looser
# than the LP one -- a relative tolerance against 0 can never pass.
CUOPT_DUAL_ATOL_QP: float = 1e-6
# cuOpt's own mip_integrality_tolerance default; asserting tighter would test
# the solver's tolerance rather than linopy.
CUOPT_INTEGRALITY_ATOL: float = 1e-5
# The four cuOpt methods disagree with each other by 2.5e-5 relative on a
# 5000x2500 LP at default tolerances; 1e-4 is four times that.
CUOPT_LARGE_OBJ_RTOL: float = 1e-4


# ---------------------------------------------------------------------------
# models and helpers
# ---------------------------------------------------------------------------


def sign_matrix_model(sense: str, row_sign: str, **model_kwargs: Any) -> Model:
    """
    Model behind the dual sign matrix.

    ``A = [[1, 2, 1], [3, 1, 1]]``, ``b = [4, 6]``, ``0 <= x <= 10``, with the
    objective coefficients chosen per cell so that the primal optimum is always
    ``x = (1.6, 1.2, 0)`` with both rows binding, ``x2`` at its lower bound and
    a unique, non-degenerate dual.
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


def square_equality_model(n: int, sense: str) -> Model:
    """
    Model that cuOpt's presolve solves outright (``solved_by == Unset``).

    ``n / 2`` equality rows, each pinning one pair of variables, so the system
    is square and presolve finishes it without calling a method. Handing such a
    model to cuOpt as a maximisation returns negated duals, which is why the
    solver class always minimises instead.
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
    """MILP with a continuous and an integer variable block, bounded in both senses."""
    m = Model(chunk=None)
    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, 5, name="x")
    y = m.add_variables(lower, 9, name="y", integer=True)
    m.add_constraints(x + y, "<=" if sense == "max" else ">=", 9.5, name="con0")
    m.add_objective(2 * x + y, sense=sense)
    return m


def semi_continuous_model(capacity: float) -> Model:
    """``max x`` for a semi-continuous ``x`` in ``[1, 10]`` capped at ``capacity``."""
    m = Model(chunk=None)
    x = m.add_variables(lower=1, upper=10, name="x", semi_continuous=True)
    m.add_constraints(1 * x, "<=", capacity, name="con0")
    m.add_objective(1 * x, sense="max")
    return m


# The coefficients the three-variable QP below has to produce. cuOpt minimises
# `c'x + x'Qx` and symmetrises Q to `Q + Q'`, while linopy's `M.Q` is the
# Hessian of `0.5 x'Qx`, so the solver class halves it. Pinning both matrices
# is what turns the differential into a convention guard: if linopy ever hands
# over a different form, the test says so instead of drifting quietly.
QP_C: np.ndarray = np.array([-3.0, -1.0, 2.0])
QP_Q: np.ndarray = np.array([[2.0, 1.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 1.0]])


def three_variable_qp(sense: str = "min") -> Model:
    """
    Dense-Hessian QP with a cross term, an off-diagonal zero and free signs.

    Its unconstrained optimum ``(11/7, -1/7, -2)`` lies strictly inside the box
    and inside both rows, so the primal is unique, the duals are zero and every
    component is determined -- unlike the degenerate fixtures below. ``max``
    negates the whole objective rather than only ``c``: maximising the same
    convex form is non-concave and neither solver would accept it.
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


def degenerate_qp() -> Model:
    """
    ``test_optimization.py``'s ``quadratic_model`` fixture, built locally.

    The objective is ``x * x`` only, so each ``y_i`` has zero cost, zero
    curvature, ``lb = 0`` and ``ub = +inf``: the optimal face is unbounded in
    ``y`` and only the ``x`` block and the objective are determined.
    """
    m = Model(chunk=None)
    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(lower, name="y")
    m.add_constraints(x + y, ">=", 10, name="con0")
    m.add_objective(x * x)
    return m


def cross_terms_qp() -> Model:
    """
    ``test_optimization.py``'s ``quadratic_model_cross_terms`` fixture.

    Here ``y`` carries a ``+1`` objective coefficient, which pins the optimum
    at ``x = 1.5``, ``y = 8.5`` -- so unlike ``degenerate_qp`` the full primal
    is a legitimate quantity to compare against an oracle.
    """
    m = Model(chunk=None)
    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(lower, name="y")
    m.add_constraints(x + y, ">=", 10, name="con0")
    m.add_objective(-2 * x + y + x * x)
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


def solve_with_both(model: Model, **cuopt_options: Any) -> tuple[Model, Model]:
    """
    Solve two deep copies of ``model`` -- one with cuOpt, one with HiGHS.

    Returns both models so the caller can compare solutions, duals and
    objectives against a live oracle rather than a stored number.
    """
    with_highs = model.copy()
    model.solve("cuopt", io_api="direct", log_to_console=False, **cuopt_options)
    with_highs.solve("highs", output_flag=False)
    return model, with_highs


def solve_qp_with_both(
    build: Callable[[], Model], **cuopt_options: Any
) -> tuple[Model, Model]:
    """
    ``solve_with_both`` for quadratic models: build the oracle, never copy it.

    ``Model.copy`` rebuilds the objective as a ``LinearExpression``
    (``linopy/io.py:1270``), so a copied QP silently loses its quadratic term
    and the "oracle" solves the LP relaxation instead -- a wrong expectation
    that no assertion here could distinguish from a solver bug. Building the
    model twice is the only way to compare the same problem.
    """
    with_cuopt = build()
    with_highs = build()
    with_cuopt.solve("cuopt", io_api="direct", log_to_console=False, **cuopt_options)
    with_highs.solve("highs", output_flag=False)
    return with_cuopt, with_highs


def assert_objectives_match(cuopt_model: Model, highs_model: Model) -> None:
    assert cuopt_model.objective.value == pytest.approx(
        highs_model.objective.value, rel=CUOPT_OBJ_RTOL, abs=CUOPT_OBJ_ATOL
    )


def assert_duals_match(cuopt_model: Model, highs_model: Model, name: str) -> None:
    cuopt_dual = np.asarray(cuopt_model.constraints[name].dual)
    highs_dual = np.asarray(highs_model.constraints[name].dual)
    # Guard against a vacuous comparison: on a model with zero duals a sign
    # error would pass unnoticed.
    assert np.abs(highs_dual).min() > 1e-3
    assert np.allclose(
        cuopt_dual, highs_dual, rtol=CUOPT_DUAL_RTOL, atol=CUOPT_DUAL_ATOL
    )


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
# registration
# ---------------------------------------------------------------------------


def test_cuopt_is_available() -> None:
    assert "cuopt" in linopy.available_solvers


def test_capability_shim_reports_declared_features() -> None:
    assert SOLVER_REGISTRY["cuopt"].features == solvers.cuOpt.supported_features()
    assert SOLVER_REGISTRY["cuopt"].display_name == solvers.cuOpt.display_name


def test_enum_member_name_matches_class_name() -> None:
    # The capability shim resolves classes via getattr(solvers, SolverName.name).
    assert SolverName.cuOpt.name == solvers.cuOpt.__name__


def test_routing_module_stays_unimported() -> None:
    # Importing cuopt.routing installs a global sys.excepthook that writes
    # error_log.txt into the working directory.
    model = sign_matrix_model("min", "<=")
    model.solve("cuopt", io_api="direct", log_to_console=False)
    assert "cuopt.routing" not in sys.modules


# ---------------------------------------------------------------------------
# sign conventions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sense,row_sign",
    [
        ("min", "<="),
        ("min", ">="),
        ("min", "="),
        ("max", "<="),
        ("max", ">="),
        ("max", "="),
    ],
    ids=["min-le", "min-ge", "min-eq", "max-le", "max-ge", "max-eq"],
)
def test_sign_matrix(sense: str, row_sign: str) -> None:
    """Duals, primals and objective agree with HiGHS in all six cells."""
    with_cuopt, with_highs = solve_with_both(sign_matrix_model(sense, row_sign))

    assert with_cuopt.status == "ok"
    assert_objectives_match(with_cuopt, with_highs)
    assert np.allclose(
        np.asarray(with_cuopt.solution.x),
        np.asarray(with_highs.solution.x),
        rtol=CUOPT_PRIMAL_RTOL,
        atol=CUOPT_PRIMAL_ATOL,
    )
    assert_duals_match(with_cuopt, with_highs, "con0")


@pytest.mark.parametrize("n", [2, 10])
def test_presolve_max_duals(n: int) -> None:
    """
    The case the six-cell matrix misses.

    cuOpt returns *negated* duals, with status ``Optimal`` and a correct
    objective, for a maximisation its presolve solves outright. The solver
    class avoids the branch by always handing cuOpt a minimisation; without
    that, these duals are off by ``2 * |dual|``.
    """
    with_cuopt, with_highs = solve_with_both(square_equality_model(n, "max"))

    assert with_cuopt.status == "ok"
    assert_objectives_match(with_cuopt, with_highs)
    assert_duals_match(with_cuopt, with_highs, "con0")


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
def test_milp_differential(sense: str) -> None:
    """Objective, integrality, empty duals and a usable MIP bound."""
    with_cuopt, with_highs = solve_with_both(milp_model(sense))

    assert with_cuopt.status == "ok"
    assert_objectives_match(with_cuopt, with_highs)
    integers = np.asarray(with_cuopt.solution.y)
    assert np.allclose(integers, np.round(integers), atol=CUOPT_INTEGRALITY_ATOL)

    # cuOpt refuses duals for a mixed integer problem.
    assert with_cuopt.solver is not None
    assert with_cuopt.solver.solution is not None
    assert with_cuopt.solver.solution.dual.size == 0

    report = with_cuopt.solver.report
    assert report is not None
    assert report.dual_bound is not None
    assert report.mip_gap is not None
    # The bound is a bound: above the objective for a maximisation, below it
    # for a minimisation. Negating it along with the objective is what makes
    # this hold for `max`.
    objective = with_cuopt.objective.value or 0.0
    if sense == "max":
        assert report.dual_bound >= objective - CUOPT_OBJ_ATOL
    else:
        assert report.dual_bound <= objective + CUOPT_OBJ_ATOL


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


def knapsack(n: int = 60, seed: int = 4) -> Model:
    m = Model(chunk=None)
    rng = np.random.default_rng(seed)
    items = pd.RangeIndex(n, name="i")
    x = m.add_variables(coords=[items], name="x", binary=True)
    value = xr.DataArray(rng.integers(10, 100, n).astype(float), coords=[items])
    weight = xr.DataArray(rng.integers(10, 100, n).astype(float), coords=[items])
    m.add_constraints((weight * x).sum(), "<=", float(weight.sum()) / 2, name="con0")
    m.add_objective(-(value * x).sum())
    return m


def market_split() -> Model:
    """
    Six-row market-split MILP over 50 binaries, hard enough to exhaust a limit.

    The instance is fingerprinted below because the time limit it provokes was
    measured on exactly these numbers: a change in ``default_rng``'s stream
    would silently swap in a different instance of unknown difficulty, and the
    test would then assert a status nobody has ever measured.
    """
    rng = np.random.default_rng(3)
    A = rng.integers(0, 100, size=(6, 50)).astype(float)
    assert np.array_equal(A[0, :6], [81, 8, 17, 23, 18, 80])
    assert A.sum() == 14702.0
    rhs = np.floor(A.sum(axis=1) / 2) + 0.5

    m = Model(chunk=None)
    variables = pd.RangeIndex(50, name="i")
    rows = pd.RangeIndex(6, name="row")
    x = m.add_variables(lower=0, upper=1, coords=[variables], name="x", integer=True)
    coeffs = xr.DataArray(A, coords=[rows, variables])
    m.add_constraints(
        (coeffs * x).sum("i"), "=", xr.DataArray(rhs, coords=[rows]), name="con0"
    )
    m.add_objective(-x.sum())
    return m


@pytest.mark.parametrize(
    "build,options,expected,primal_size",
    [
        (lambda: sign_matrix_model("min", "<="), {}, "optimal", None),
        (infeasible_lp, {}, "infeasible", None),
        (unbounded_lp, {}, "infeasible_or_unbounded", None),
        (lambda: random_lp(400, 300), {"iteration_limit": 1}, "iteration_limit", None),
        (lambda: random_lp(400, 300), {"time_limit": 1e-6}, "time_limit", None),
        (
            lambda: random_lp(400, 300),
            {"first_primal_feasible": True, "method": 1},
            "suboptimal",
            None,
        ),
        (lambda: milp_model("min"), {}, "optimal", None),
        (infeasible_milp, {}, "infeasible", None),
        (lambda: unbounded_lp(integer=True), {}, "infeasible_or_unbounded", None),
        (knapsack, {"node_limit": 1}, "suboptimal", None),
        (market_split, {"time_limit": 2.0}, "time_limit", 0),
    ],
    ids=[
        "lp_optimal",
        "lp_infeasible",
        "lp_unbounded",
        "lp_iteration_limit",
        "lp_time_limit",
        "lp_primal_feasible",
        "milp_optimal",
        "milp_infeasible",
        "milp_unbounded",
        "milp_feasible_found",
        "milp_time_limit",
    ],
)
def test_status_map(
    build: Callable[[], Model],
    options: dict[str, Any],
    expected: str,
    primal_size: int | None,
) -> None:
    """
    Each reachable cuOpt termination status maps onto the right condition.

    Only the condition is asserted for the limit and incumbent cases: the
    objective of an unproven incumbent is not reproducible. A limit *setting*
    also never implies a limit *status* -- ``iteration_limit=1`` on a knapsack
    was measured returning ``Optimal`` -- so nothing here is inferred from the
    options that were passed.

    Where ``primal_size`` is given, the shape of the returned solution is
    asserted too. The market-split MILP finds nothing within its two seconds,
    so cuOpt hands back zero primal values for 50 variables; a limit
    termination is ``ok``, so that mismatch has to be caught and turned into an
    empty ``Solution`` instead of being scattered over the labels.
    """
    model = build()
    status, condition = model.solve(
        "cuopt", io_api="direct", log_to_console=False, **options
    )
    assert condition == expected
    if primal_size is not None:
        assert model.solver is not None
        assert model.solver.solution is not None
        assert model.solver.solution.primal.size == primal_size


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

    The padded row repeats a variable's own bounds, which leaves the feasible
    set untouched, and its dual is sliced off again before the solution is
    built.
    """
    model = Model(chunk=None)
    x = model.add_variables(lower=0, upper=10, name="x")
    model.add_objective(-x)

    with_cuopt, with_highs = solve_with_both(model)

    assert with_cuopt.status == "ok"
    assert not len(with_cuopt.constraints)
    assert_objectives_match(with_cuopt, with_highs)
    assert with_cuopt.solver is not None
    assert with_cuopt.solver.solution is not None
    assert with_cuopt.solver.solution.dual.size == 0


def test_model_without_constraints_or_bounds_raises() -> None:
    """No row can be padded from a model that has no finite bound either."""
    model = Model(chunk=None)
    x = model.add_variables(name="x")
    model.add_objective(1 * x)

    with pytest.raises(NotImplementedError, match="no finite variable bounds"):
        model.solve("cuopt", io_api="direct")


def test_duals_stay_aligned_with_the_constraints() -> None:
    """A one-row shift from the pad slice would move every dual silently."""
    with_cuopt, with_highs = solve_with_both(sign_matrix_model("min", "<="))

    constraint = with_cuopt.constraints["con0"]
    assert constraint.dual.size == constraint.labels.size
    assert_duals_match(with_cuopt, with_highs, "con0")


# ---------------------------------------------------------------------------
# semi-continuous variables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capacity", [0.5, 5.0])
def test_semi_continuous_differential(capacity: float) -> None:
    """
    A semi-continuous variable is off or above its lower bound, never between.

    ``test_semi_continuous.py`` has no solver parametrisation, so this is the
    only coverage the declared feature gets.
    """
    with_cuopt, with_highs = solve_with_both(semi_continuous_model(capacity))

    assert with_cuopt.status == "ok"
    assert_objectives_match(with_cuopt, with_highs)
    assert float(with_cuopt.solution.x) == pytest.approx(
        float(with_highs.solution.x), rel=CUOPT_PRIMAL_RTOL, abs=CUOPT_PRIMAL_ATOL
    )


# ---------------------------------------------------------------------------
# quadratic objectives
# ---------------------------------------------------------------------------


def test_quadratic_objective_is_registered() -> None:
    assert solvers.cuOpt.supports(solvers.SolverFeature.QUADRATIC_OBJECTIVE)
    assert "cuopt" in quadratic_solvers


@pytest.mark.parametrize("sense", ["min", "max"])
def test_quadratic_differential(sense: str) -> None:
    """
    Objective, primal and duals of a determined QP, against live HiGHS.

    This is the guard on the ``0.5 * M.Q`` convention: the naive ``M.Q`` also
    returns ``Optimal``, with half the solution vector and an objective off by
    50% -- see ``test_naive_hessian_changes_the_answer``, which measures
    exactly that.
    """
    sign = 1.0 if sense == "min" else -1.0
    matrices = three_variable_qp(sense).matrices
    assert np.allclose(matrices.c, sign * QP_C)
    assert matrices.Q is not None
    assert np.allclose(matrices.Q.toarray(), sign * QP_Q)

    with_cuopt, with_highs = solve_qp_with_both(lambda: three_variable_qp(sense))

    assert with_cuopt.status == "ok"
    assert_objectives_match(with_cuopt, with_highs)
    for name in ("x0", "x1", "x2"):
        assert float(with_cuopt.solution[name]) == pytest.approx(
            float(with_highs.solution[name]),
            rel=CUOPT_PRIMAL_RTOL,
            abs=CUOPT_PRIMAL_ATOL,
        )
    # Neither row binds at the optimum, so both duals are zero and
    # ``assert_duals_match``'s non-vacuity guard cannot apply here. What is
    # being checked is that a QP returns duals at all, aligned and not scaled:
    # the LP sign matrix covers the sign.
    for name in ("con0", "con1"):
        assert np.allclose(
            np.asarray(with_cuopt.constraints[name].dual),
            np.asarray(with_highs.constraints[name].dual),
            rtol=CUOPT_DUAL_RTOL,
            atol=CUOPT_DUAL_ATOL_QP,
        )


def test_naive_hessian_changes_the_answer() -> None:
    """
    The deliberate failure that makes the convention guard above a guard.

    Handing cuOpt ``M.Q`` where the solver class hands ``0.5 * M.Q`` is the
    classic factor-of-two error, and cuOpt reports it as ``Optimal``: only a
    differential catches it. Reproduced here through the production status
    mapping so it is provable that the wrong answer is reachable and that the
    tolerances above would reject it.
    """
    reference = three_variable_qp("min")
    reference.solve("cuopt", io_api="direct", log_to_console=False)
    expected = reference.objective.value
    assert expected is not None

    model = three_variable_qp("min")
    data_model = solvers.cuOpt._build_solver_model(model)
    assert model.matrices.Q is not None
    naive = csr_array(model.matrices.Q.tocsr())
    data_model.set_quadratic_objective_matrix(naive.data, naive.indices, naive.indptr)

    result = raw_result(model, data_model)

    assert result.status.termination_condition.value == "optimal"
    assert abs(result.solution.objective - expected) / abs(expected) > 1e-3


def test_degenerate_quadratic_differential() -> None:
    """
    ``quadratic_model``: the objective and the ``x`` block only.

    The 10 ``y`` variables have zero cost, zero curvature and no upper bound,
    so the optimal face is unbounded in ``y``: cuOpt returns ~156 and HiGHS ~10
    and both are optimal. Comparing them against an oracle would assert an
    underdetermined quantity, which is why the shared suite asserts an
    inequality on ``y`` instead. On the ``x`` block HiGHS itself is only
    ~1e-4-accurate on this singular problem, hence the looser tolerance.
    """
    with_cuopt, with_highs = solve_qp_with_both(degenerate_qp)

    assert with_cuopt.status == "ok"
    assert_objectives_match(with_cuopt, with_highs)
    assert np.allclose(
        np.asarray(with_cuopt.solution.x),
        np.asarray(with_highs.solution.x),
        rtol=CUOPT_PRIMAL_RTOL,
        atol=CUOPT_PRIMAL_ATOL_DEGENERATE,
    )


def test_cross_terms_quadratic_differential() -> None:
    """``quadratic_model_cross_terms`` is determined, so the full primal counts."""
    with_cuopt, with_highs = solve_qp_with_both(cross_terms_qp)

    assert with_cuopt.status == "ok"
    assert_objectives_match(with_cuopt, with_highs)
    for name in ("x", "y"):
        assert np.allclose(
            np.asarray(with_cuopt.solution[name]),
            np.asarray(with_highs.solution[name]),
            rtol=CUOPT_PRIMAL_RTOL,
            atol=CUOPT_PRIMAL_ATOL,
        )


def test_quadratic_model_without_constraint() -> None:
    """
    A QP whose only row was removed, i.e. the pad row carries the whole matrix.

    This is the shape of ``test_quadratic_model_wo_constraint``, and the reason
    the pad row repeats a variable's own bounds rather than spanning
    ``(-inf, +inf)``: a doubly infinite pad row makes a quadratic objective
    fail with ``NumericalError``. As in ``degenerate_qp`` the ``y`` block is
    underdetermined -- here the optimal face is all of ``[0, inf)^10`` -- so
    only the objective and the ``x`` block are compared.
    """

    def build() -> Model:
        model = degenerate_qp()
        model.constraints.remove("con0")
        return model

    with_cuopt, with_highs = solve_qp_with_both(build)

    assert with_cuopt.termination_condition == "optimal"
    assert (with_cuopt.solution.x.round(3) == 0).all()
    assert round(with_cuopt.objective.value or 0, 3) == 0
    assert_objectives_match(with_cuopt, with_highs)
    assert np.allclose(
        np.asarray(with_cuopt.solution.x),
        np.asarray(with_highs.solution.x),
        rtol=CUOPT_PRIMAL_RTOL,
        atol=CUOPT_PRIMAL_ATOL_DEGENERATE,
    )


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


def test_medium_random_lp() -> None:
    """The default method and tolerances stay usable beyond toy sizes."""
    with_cuopt, with_highs = solve_with_both(random_lp(2000, 1000, seed=7))

    assert with_cuopt.status == "ok"
    assert with_cuopt.objective.value == pytest.approx(
        with_highs.objective.value, rel=CUOPT_LARGE_OBJ_RTOL
    )


def test_unavailable_without_a_cuda_device() -> None:
    """
    Without a usable device cuOpt is simply absent, with a warning that says why.

    ``import cuopt`` succeeds on a machine without a GPU, so availability has
    to be decided by probing the device rather than the import.
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
    assert result.stderr.count("no usable CUDA device was found") == 1
    assert "525.60.13" in result.stderr


FORK_SCRIPT = """
import os
import sys

import linopy

{probe}


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


def test_device_probe_is_fork_safe() -> None:
    """
    Probing the device must not initialise CUDA in the calling process.

    ``available_solvers`` is touched on import in plenty of code bases, and a
    CUDA context created before ``os.fork()`` makes every cuOpt solve in the
    child fail. The second half of this test runs the same harness with the
    naive in-process probe, to show that the check can actually fail.
    """
    with_probe = run_in_subprocess(
        FORK_SCRIPT.format(probe='assert "cuopt" in linopy.available_solvers')
    )
    assert with_probe.returncode == 0, with_probe.stderr

    naive = run_in_subprocess(
        FORK_SCRIPT.format(
            probe=(
                "from cuda.bindings import runtime\n"
                "error, count = runtime.cudaGetDeviceCount()\n"
                "assert int(error) == 0 and count"
            )
        )
    )
    assert naive.returncode != 0


# ---------------------------------------------------------------------------
# solver options
# ---------------------------------------------------------------------------


def test_unknown_option_names_the_offending_parameter() -> None:
    # cuOpt's own message does not say which parameter it rejected.
    model = sign_matrix_model("min", "<=")
    with pytest.raises(ValueError, match="TimeLimit"):
        model.solve("cuopt", io_api="direct", TimeLimit=1)


def test_boolean_option_is_accepted() -> None:
    # cuOpt types most parameters as int and rejects a Python bool for them.
    model = sign_matrix_model("min", "<=")
    status, condition = model.solve(
        "cuopt", io_api="direct", log_to_console=False, presolve=False
    )
    assert condition == "optimal"


def test_log_fn_writes_the_solver_log(tmp_path: Path) -> None:
    model = sign_matrix_model("min", "<=")
    log_fn = tmp_path / "cuopt.log"
    model.solve("cuopt", io_api="direct", log_fn=log_fn)

    assert log_fn.stat().st_size > 0
    assert "cuOpt version" in log_fn.read_text()


def test_log_fn_overrides_a_user_log_file(tmp_path: Path) -> None:
    model = sign_matrix_model("min", "<=")
    log_fn = tmp_path / "linopy.log"
    user_log = tmp_path / "user.log"
    model.solve("cuopt", io_api="direct", log_fn=log_fn, log_file=str(user_log))

    assert log_fn.exists()
    assert not user_log.exists()


# ---------------------------------------------------------------------------
# unsupported surfaces
# ---------------------------------------------------------------------------


def test_warmstart_is_refused(tmp_path: Path) -> None:
    model = sign_matrix_model("min", "<=")
    with pytest.raises(NotImplementedError, match="[Ww]armstart"):
        model.solve("cuopt", io_api="direct", warmstart_fn=tmp_path / "basis.bas")


def test_solution_file_is_refused(tmp_path: Path) -> None:
    model = sign_matrix_model("min", "<=")
    with pytest.raises(NotImplementedError, match="[Ss]olution file"):
        model.solve("cuopt", io_api="direct", solution_fn=tmp_path / "solution.sol")


def test_basis_file_is_ignored_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    model = sign_matrix_model("min", "<=")
    with caplog.at_level(logging.WARNING, logger="linopy.solvers"):
        status, condition = model.solve(
            "cuopt",
            io_api="direct",
            log_to_console=False,
            basis_fn=tmp_path / "basis.bas",
        )
    assert condition == "optimal"
    assert sum("Basis files" in record.message for record in caplog.records) == 1


def test_sos_constraints_are_refused() -> None:
    model = Model(chunk=None)
    items = pd.RangeIndex(3, name="i")
    x = model.add_variables(lower=0, upper=10, coords=[items], name="x")
    model.add_constraints((1 * x).sum(), "<=", 4, name="con0")
    model.add_objective(-(1 * x).sum())
    model.add_sos_constraints(x, sos_type=1, sos_dim="i")

    with pytest.raises(ValueError, match="SOS"):
        model.solve("cuopt", io_api="direct")


def test_indicator_constraints_are_refused() -> None:
    model = Model(chunk=None)
    b = model.add_variables(name="b", binary=True)
    x = model.add_variables(lower=0, upper=10, name="x")
    model.add_constraints(1 * x, "<=", 4, name="con0")
    model.add_indicator_constraints(b, 1, x, "<=", 5, name="indcon0")
    model.add_objective(-x)

    with pytest.raises(ValueError, match="indicator"):
        model.solve("cuopt", io_api="direct")


# ---------------------------------------------------------------------------
# the file io fallback
# ---------------------------------------------------------------------------


def test_bare_solve_falls_back_to_the_direct_api(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``model.solve("cuopt")`` without an ``io_api`` solves, and says so once."""
    model = sign_matrix_model("min", "<=", solver_dir=tmp_path)
    oracle = model.copy()

    with caplog.at_level(logging.WARNING, logger="linopy.solvers"):
        status, condition = model.solve("cuopt")
    oracle.solve("highs", output_flag=False)

    assert condition == "optimal"
    assert_objectives_match(model, oracle)
    assert (
        sum("does not support file IO" in record.message for record in caplog.records)
        == 1
    )


def test_bare_solve_leaves_no_problem_file(tmp_path: Path) -> None:
    """
    The unused problem file goes back to linopy for unlinking.

    ``Model.solve`` creates the file before building and unlinks only what the
    solver hands back, so an override that keeps quiet about it accumulates an
    empty file on every bare solve. ``tmp_path`` is a fresh directory on
    purpose -- the default ``solver_dir`` is shared with every other test.
    """
    model = sign_matrix_model("min", "<=", solver_dir=tmp_path)
    model.solve("cuopt")

    assert list(Path(model.solver_dir).glob("linopy-problem-*")) == []
