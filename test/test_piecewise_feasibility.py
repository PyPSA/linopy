"""
Strategic feasibility-region equivalence tests for PWL inequality.

Stress-tests the documented claim that ``add_piecewise_formulation(sign="<=")``
(or ``">="``) yields the **same feasible region** for ``(x, y)`` regardless
of which method (``lp`` / ``sos2`` / ``incremental``) dispatches the
formulation, on curves where all three are applicable.

The strong test is :class:`TestRotatedObjective`: for every rotation
``(α, β)``, the support function ``min α·x + β·y`` under the PWL must match
a vertex-enumeration oracle.  Equal support functions across enough
directions imply equal (convex) feasible regions.

:class:`TestDomainBoundary` and :class:`TestPointwiseInfeasibility` add
targeted sanity checks for cases that rotated objectives don't directly
probe (domain-bound enforcement, numerical precision of the curve bound).

:class:`TestNVariableInequality` covers 3-variable inequality (LP does not
support it — this is SOS2 vs incremental only) and verifies the split:
bounded first tuple, equality on the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import pytest

from linopy import Model, available_solvers
from linopy.solver_capabilities import (
    SolverFeature,
    get_available_solvers_with_feature,
)
from linopy.variables import Variable

Sign: TypeAlias = Literal["<=", ">="]
Method: TypeAlias = Literal["lp", "sos2", "incremental"]

TOL = 1e-5
X_LO, X_HI = -100.0, 100.0
Y_LO, Y_HI = -100.0, 100.0

_sos2_solvers = get_available_solvers_with_feature(
    SolverFeature.SOS_CONSTRAINTS, available_solvers
)
_any_solvers = [
    s for s in ["highs", "gurobi", "glpk", "cplex"] if s in available_solvers
]

pytestmark = pytest.mark.skipif(
    not (_sos2_solvers and _any_solvers),
    reason="need an SOS2-capable LP/MIP solver",
)


# ---------------------------------------------------------------------------
# Curve definition + oracle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Curve:
    """A piecewise-linear curve + the sign of the bound it carries."""

    name: str
    x_pts: tuple[float, ...]
    y_pts: tuple[float, ...]
    sign: Sign

    def f(self, x: float) -> float:
        """Linear interpolation of ``y`` at ``x`` (ground truth)."""
        return float(np.interp(x, self.x_pts, self.y_pts))

    def vertices(
        self, y_lo: float = Y_LO, y_hi: float = Y_HI
    ) -> list[tuple[float, float]]:
        """
        Vertices of the feasible polygon — used by the oracle.

        The feasible region for ``sign="<="`` is
        ``{(x,y) : x_0 ≤ x ≤ x_n, y_lo ≤ y ≤ f(x)}`` — a polygon whose
        vertices are the breakpoints (top edges) plus two bottom corners.
        For ``sign=">="`` it is the mirror image clipped to ``y_hi``.
        """
        verts = list(zip(self.x_pts, self.y_pts))
        bottom_y = y_lo if self.sign == "<=" else y_hi
        verts.append((self.x_pts[0], bottom_y))
        verts.append((self.x_pts[-1], bottom_y))
        return verts


CURVES: list[Curve] = [
    Curve("concave-smooth", (0, 1, 2, 3, 4), (0, 1.75, 3, 3.75, 4), "<="),
    Curve("concave-shifted", (-2, 0, 5, 10), (-5, 0, 3, 4), "<="),
    Curve("convex-steep", (0, 1, 2, 3, 4), (0, 1, 4, 9, 16), ">="),
    Curve("linear-lte", (0, 1, 2, 3, 4), (10, 12, 14, 16, 18), "<="),
    Curve("linear-gte", (0, 1, 2, 3, 4), (10, 12, 14, 16, 18), ">="),
    Curve("two-segment", (0, 10, 20), (0, 15, 20), "<="),
]


# ---------------------------------------------------------------------------
# Primitives: build a model, solve, assert infeasibility
# ---------------------------------------------------------------------------


def build_model(curve: Curve, method: Method) -> tuple[Model, Variable, Variable]:
    """Build a fresh model with bounded x, y linked by the PWL formulation."""
    m = Model()
    x = m.add_variables(lower=X_LO, upper=X_HI, name="x")
    y = m.add_variables(lower=Y_LO, upper=Y_HI, name="y")
    m.add_piecewise_formulation(
        (y, list(curve.y_pts), curve.sign),
        (x, list(curve.x_pts)),
        method=method,
    )
    return m, x, y


def solve_support(
    curve: Curve, method: Method, alpha: float, beta: float
) -> tuple[float, float, float]:
    """
    Solve ``min α·x + β·y``; return ``(x_sol, y_sol, objective)``.

    The attained *point* is returned alongside the objective because
    the point usually reveals the bug (wrong segment, clipped domain,
    etc.) more clearly than the objective value alone.
    """
    m, x, y = build_model(curve, method)
    m.add_objective(alpha * x + beta * y)
    status, _ = m.solve()
    assert status == "ok", f"{method}/{curve.name}: solve failed at ({alpha}, {beta})"
    x_sol = float(m.solution["x"])
    y_sol = float(m.solution["y"])
    return x_sol, y_sol, alpha * x_sol + beta * y_sol


def oracle_support(curve: Curve, alpha: float, beta: float) -> float:
    """Ground truth ``min α·x + β·y`` over the feasible polygon (vertex min)."""
    return min(alpha * vx + beta * vy for vx, vy in curve.vertices())


def assert_infeasible(m: Model, x: Variable, msg: str) -> None:
    """Solve with a trivial objective; any non-'ok' status counts as infeasible."""
    m.add_objective(x)  # objective is irrelevant — just needs to be set
    status, _ = m.solve()
    assert status != "ok", msg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=CURVES, ids=lambda c: c.name)
def curve(request: pytest.FixtureRequest) -> Curve:
    return request.param


@pytest.fixture(params=["lp", "sos2", "incremental"])
def method(request: pytest.FixtureRequest) -> Method:
    return request.param


# ---------------------------------------------------------------------------
# Rotated objective — the strong test
# ---------------------------------------------------------------------------


_N_DIRECTIONS = 16
_DIRECTIONS = [
    pytest.param(
        float(np.cos(2 * np.pi * i / _N_DIRECTIONS)),
        float(np.sin(2 * np.pi * i / _N_DIRECTIONS)),
        id=f"{round(360 * i / _N_DIRECTIONS):03d}deg",
    )
    for i in range(_N_DIRECTIONS)
]


class TestRotatedObjective:
    """
    Support-function equivalence: ``min α·x + β·y`` under the PWL
    matches the vertex-enumeration oracle for every direction.

    Equal support functions over a dense enough set of directions imply
    equal convex feasible regions — the strongest region-identity check.
    """

    @pytest.mark.parametrize("alpha, beta", _DIRECTIONS)
    def test_support_matches_oracle(
        self, curve: Curve, method: Method, alpha: float, beta: float
    ) -> None:
        x_sol, y_sol, got = solve_support(curve, method, alpha, beta)
        want = oracle_support(curve, alpha, beta)
        assert abs(got - want) < TOL, (
            f"\n  curve: {curve.name}   sign: {curve.sign}   method: {method}"
            f"\n  direction: (α={alpha:+.3f}, β={beta:+.3f})"
            f"\n  attained point: (x={x_sol:+.6f}, y={y_sol:+.6f})"
            f"\n  attained obj:   {got:+.6f}"
            f"\n  oracle obj:     {want:+.6f}"
            f"\n  diff:           {got - want:+.3e}  (TOL={TOL:.1e})"
        )


# ---------------------------------------------------------------------------
# Domain boundary — direct probe that x cannot escape [x_min, x_max]
# ---------------------------------------------------------------------------


class TestDomainBoundary:
    """
    ``x`` outside ``[x_min, x_max]`` is infeasible under all methods.

    LP enforces this with an explicit constraint; SOS2/incremental enforce
    it implicitly via ``sum(λ) = 1`` (or the delta ladder).  Worth a direct
    probe because the two paths are very different implementations.
    """

    def test_below_x_min(self, curve: Curve, method: Method) -> None:
        m, x, _ = build_model(curve, method)
        m.add_constraints(x == curve.x_pts[0] - 1.0)
        assert_infeasible(
            m, x, f"{method}/{curve.name}: x < x_min should be infeasible"
        )

    def test_above_x_max(self, curve: Curve, method: Method) -> None:
        m, x, _ = build_model(curve, method)
        m.add_constraints(x == curve.x_pts[-1] + 1.0)
        assert_infeasible(
            m, x, f"{method}/{curve.name}: x > x_max should be infeasible"
        )


# ---------------------------------------------------------------------------
# Pointwise infeasibility — sanity check that (x, f(x) ± ε) is excluded
# ---------------------------------------------------------------------------


class TestPointwiseInfeasibility:
    """
    ``y`` pushed past ``f(x)`` in the sign direction is infeasible.

    Rotated objectives probe extremes; this targeted check makes sure the
    curve bound is actually a strict inequality at a representative
    interior point (catches 'off by one segment' or NaN-mask bugs that
    might accidentally allow a small slack).
    """

    def test_just_past_curve(self, curve: Curve, method: Method) -> None:
        x_mid = 0.5 * (curve.x_pts[0] + curve.x_pts[-1])
        fx = curve.f(x_mid)
        # nudge y past the bound in the forbidden direction
        y_bad = fx + 0.01 if curve.sign == "<=" else fx - 0.01
        m, x, y = build_model(curve, method)
        m.add_constraints(x == x_mid)
        m.add_constraints(y == y_bad)
        assert_infeasible(
            m,
            x,
            f"{method}/{curve.name}: (x={x_mid}, y={y_bad}) beyond "
            f"f(x)={fx} in direction {curve.sign} should be infeasible",
        )


# ---------------------------------------------------------------------------
# Hand-computed anchors — sanity-check the oracle itself
# ---------------------------------------------------------------------------


class TestHandComputedAnchors:
    """
    A handful of pinpoint tests with hand-calculable expected values.

    The parameterised tests compare the solver against a vertex-enumeration
    oracle — if that oracle or ``np.interp`` ever drifted, the tests could
    continue to pass in false agreement with a broken oracle.  These
    anchors assert *concrete numbers* a reader can verify with a
    calculator in ten seconds, so any oracle drift would surface here.

    Every curve below is arithmetically trivial.  Each expected value has
    a one-line comment showing the arithmetic.
    """

    # y = 2x on [0, 5] — linear, trivial.
    LINEAR = Curve("y_eq_2x", (0, 1, 2, 3, 4, 5), (0, 2, 4, 6, 8, 10), "<=")

    # concave: (0,0) (1,1) (2,1.5) (3,1.75) — slopes 1, 0.5, 0.25 (classic
    # diminishing returns)
    CONCAVE = Curve("dim_returns", (0, 1, 2, 3), (0, 1, 1.5, 1.75), "<=")

    # convex y = x² sampled at 0..3 — slopes 1, 3, 5
    CONVEX = Curve("y_eq_x2", (0, 1, 2, 3), (0, 1, 4, 9), ">=")

    # ---- 2-variable ----------------------------------------------------

    @pytest.mark.parametrize("method", ["lp", "sos2", "incremental"])
    def test_linear_at_midsegment(self, method: Method) -> None:
        """Y ≤ 2x at x=2.5: max y = 5.0 (halfway between (2, 4) and (3, 6))."""
        m, x, y = build_model(self.LINEAR, method)
        m.add_constraints(x == 2.5)
        m.add_objective(-y)
        m.solve()
        assert float(m.solution["y"]) == pytest.approx(5.0, abs=TOL)

    @pytest.mark.parametrize("method", ["lp", "sos2", "incremental"])
    def test_linear_at_breakpoint(self, method: Method) -> None:
        """Y ≤ 2x at x=3 (exact breakpoint): max y = 6.0."""
        m, x, y = build_model(self.LINEAR, method)
        m.add_constraints(x == 3.0)
        m.add_objective(-y)
        m.solve()
        assert float(m.solution["y"]) == pytest.approx(6.0, abs=TOL)

    @pytest.mark.parametrize("method", ["lp", "sos2", "incremental"])
    def test_linear_at_x_min(self, method: Method) -> None:
        """Y ≤ 2x at x=0 (domain lower bound): max y = 0.0."""
        m, x, y = build_model(self.LINEAR, method)
        m.add_constraints(x == 0.0)
        m.add_objective(-y)
        m.solve()
        assert float(m.solution["y"]) == pytest.approx(0.0, abs=TOL)

    @pytest.mark.parametrize("method", ["lp", "sos2", "incremental"])
    def test_linear_at_x_max(self, method: Method) -> None:
        """Y ≤ 2x at x=5 (domain upper bound): max y = 10.0."""
        m, x, y = build_model(self.LINEAR, method)
        m.add_constraints(x == 5.0)
        m.add_objective(-y)
        m.solve()
        assert float(m.solution["y"]) == pytest.approx(10.0, abs=TOL)

    @pytest.mark.parametrize("method", ["lp", "sos2", "incremental"])
    def test_concave_at_midsegment(self, method: Method) -> None:
        """Y ≤ f(x) concave at x=1.5: max y = (1 + 1.5)/2 = 1.25."""
        m, x, y = build_model(self.CONCAVE, method)
        m.add_constraints(x == 1.5)
        m.add_objective(-y)
        m.solve()
        assert float(m.solution["y"]) == pytest.approx(1.25, abs=TOL)

    @pytest.mark.parametrize("method", ["lp", "sos2", "incremental"])
    def test_convex_ge_at_midsegment(self, method: Method) -> None:
        """Y ≥ f(x) convex at x=1.5: min y = (1 + 4)/2 = 2.5."""
        m, x, y = build_model(self.CONVEX, method)
        m.add_constraints(x == 1.5)
        m.add_objective(y)  # minimise — pushes y against the lower bound (curve)
        m.solve()
        assert float(m.solution["y"]) == pytest.approx(2.5, abs=TOL)
