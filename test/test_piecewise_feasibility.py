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
MethodND: TypeAlias = Literal["sos2", "incremental"]  # LP doesn't support N > 2

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
        (y, list(curve.y_pts)),
        (x, list(curve.x_pts)),
        sign=curve.sign,
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
# 3-variable inequality: sign='<=' splits bounded output from equality inputs
# ---------------------------------------------------------------------------


class TestNVariableInequality:
    """
    3-variable ``sign="<="``: the first tuple (output) is bounded above,
    the remaining tuples (inputs) are pinned on the curve — equality-linked.

    LP does not support ``N > 2``, so this is SOS2 vs incremental only.
    The feasible region is a "ribbon" along the fuel axis parameterised
    by the curve's ``(power, heat)`` trajectory:

        { (fuel, power, heat) : ∃ λ SOS2 with Σλ=1,
            power = Σλ·p_i,   heat = Σλ·h_i,   FUEL_LO ≤ fuel ≤ Σλ·f_i }

    Tests probe this region from several angles: a vertex-enumeration
    oracle for rotated objectives, plus targeted feasible/infeasible
    point checks.
    """

    BP = {
        "power": (0, 30, 60, 100),
        "fuel": (0, 40, 85, 160),  # bounded output (first tuple)
        "heat": (0, 25, 55, 95),  # input, forced to equality
    }
    FUEL_LO, FUEL_HI = 0.0, 200.0
    POWER_LO, POWER_HI = 0.0, 100.0
    HEAT_LO, HEAT_HI = 0.0, 100.0

    @pytest.fixture(params=["sos2", "incremental"])
    def method_3var(self, request: pytest.FixtureRequest) -> MethodND:
        return request.param

    # ---- helpers --------------------------------------------------------

    def _build(self, method: MethodND) -> tuple[Model, Variable, Variable, Variable]:
        """CHP model with sign='<=': fuel bounded, power/heat equality-linked."""
        m = Model()
        power = m.add_variables(lower=self.POWER_LO, upper=self.POWER_HI, name="power")
        fuel = m.add_variables(lower=self.FUEL_LO, upper=self.FUEL_HI, name="fuel")
        heat = m.add_variables(lower=self.HEAT_LO, upper=self.HEAT_HI, name="heat")
        m.add_piecewise_formulation(
            (fuel, list(self.BP["fuel"])),
            (power, list(self.BP["power"])),
            (heat, list(self.BP["heat"])),
            sign="<=",
            method=method,
        )
        return m, fuel, power, heat

    def _oracle_support_3d(
        self, alpha_f: float, alpha_p: float, alpha_h: float
    ) -> float:
        """
        Ground-truth ``min α_f·fuel + α_p·power + α_h·heat`` over the region.

        The region is a convex polytope with vertices at each breakpoint
        in two "layers": the top ``(f_i, p_i, h_i)`` and the bottom
        ``(FUEL_LO, p_i, h_i)`` — linear objective extrema are at vertices.
        """
        fuels = self.BP["fuel"]
        powers = self.BP["power"]
        heats = self.BP["heat"]
        top = [
            alpha_f * f + alpha_p * p + alpha_h * h
            for f, p, h in zip(fuels, powers, heats)
        ]
        bot = [
            alpha_f * self.FUEL_LO + alpha_p * p + alpha_h * h
            for p, h in zip(powers, heats)
        ]
        return min(top + bot)

    # ---- existing test: fuel pushed against its upper bound -------------

    @pytest.mark.parametrize("power_fix", [0, 15, 30, 45, 60, 80, 100])
    def test_first_tuple_bounded_rest_equal(
        self, method_3var: MethodND, power_fix: float
    ) -> None:
        m, fuel, power, heat = self._build(method_3var)
        m.add_constraints(power == power_fix)
        m.add_objective(-fuel)  # push fuel against its bound
        status, _ = m.solve()
        assert status == "ok"

        expect_fuel = float(np.interp(power_fix, self.BP["power"], self.BP["fuel"]))
        expect_heat = float(np.interp(power_fix, self.BP["power"], self.BP["heat"]))

        assert abs(float(m.solution["fuel"]) - expect_fuel) < TOL, (
            f"{method_3var}: fuel at power={power_fix} should hit "
            f"f(x)={expect_fuel}, got {float(m.solution['fuel'])}"
        )
        assert abs(float(m.solution["heat"]) - expect_heat) < TOL, (
            f"{method_3var}: heat at power={power_fix} must equal "
            f"f(x)={expect_heat}, got {float(m.solution['heat'])}"
        )

    # ---- new: heat drifting off the curve is infeasible -----------------

    @pytest.mark.parametrize("power_fix", [15, 45, 80])
    def test_heat_off_curve_is_infeasible(
        self, method_3var: MethodND, power_fix: float
    ) -> None:
        """
        Heat is equality-linked.  Pinning heat away from ``f_heat(power)``
        must make the model infeasible under both methods.
        """
        expect_heat = float(np.interp(power_fix, self.BP["power"], self.BP["heat"]))
        m, fuel, power, heat = self._build(method_3var)
        m.add_constraints(power == power_fix)
        m.add_constraints(heat == expect_heat + 5.0)  # nudge off the curve
        m.add_objective(fuel)
        status, _ = m.solve()
        assert status != "ok", (
            f"{method_3var}: heat={expect_heat + 5} at power={power_fix} "
            f"should be infeasible (curve has heat={expect_heat})"
        )

    # ---- new: interior point is feasible --------------------------------

    @pytest.mark.parametrize("power_fix", [15, 45, 80])
    def test_interior_point_is_feasible(
        self, method_3var: MethodND, power_fix: float
    ) -> None:
        """
        With power/heat on the curve and fuel well below its upper
        bound, the point is interior to the ribbon — must be feasible.
        """
        expect_heat = float(np.interp(power_fix, self.BP["power"], self.BP["heat"]))
        expect_fuel = float(np.interp(power_fix, self.BP["power"], self.BP["fuel"]))
        m, fuel, power, heat = self._build(method_3var)
        m.add_constraints(power == power_fix)
        m.add_constraints(heat == expect_heat)
        m.add_constraints(fuel == expect_fuel - 10.0)  # below the bound
        m.add_objective(fuel)
        status, _ = m.solve()
        assert status == "ok", (
            f"{method_3var}: interior point (power={power_fix}, "
            f"heat={expect_heat}, fuel={expect_fuel - 10}) should be feasible"
        )

    # ---- new: rotated objective in 3D -----------------------------------

    DIRECTIONS_3D = [
        pytest.param(-1.0, 0.0, 0.0, id="maxfuel"),
        pytest.param(+1.0, 0.0, 0.0, id="minfuel"),
        pytest.param(0.0, -1.0, 0.0, id="maxpower"),
        pytest.param(0.0, +1.0, 0.0, id="minpower"),
        pytest.param(0.0, 0.0, -1.0, id="maxheat"),
        pytest.param(0.0, 0.0, +1.0, id="minheat"),
        pytest.param(-1.0, -1.0, -1.0, id="maxall"),
        pytest.param(+1.0, +1.0, +1.0, id="minall"),
    ]

    @pytest.mark.parametrize("alpha_f, alpha_p, alpha_h", DIRECTIONS_3D)
    def test_rotated_support_matches_oracle(
        self,
        method_3var: MethodND,
        alpha_f: float,
        alpha_p: float,
        alpha_h: float,
    ) -> None:
        """
        Support function equivalence in 3-space: each method lands at
        the same vertex as the vertex-enumeration oracle.
        """
        m, fuel, power, heat = self._build(method_3var)
        m.add_objective(alpha_f * fuel + alpha_p * power + alpha_h * heat)
        status, _ = m.solve()
        assert status == "ok", (
            f"{method_3var}: solve failed at ({alpha_f},{alpha_p},{alpha_h})"
        )
        fs = float(m.solution["fuel"])
        ps = float(m.solution["power"])
        hs = float(m.solution["heat"])
        got = alpha_f * fs + alpha_p * ps + alpha_h * hs
        want = self._oracle_support_3d(alpha_f, alpha_p, alpha_h)
        assert abs(got - want) < TOL, (
            f"\n  method: {method_3var}"
            f"\n  direction: (α_fuel={alpha_f:+}, α_power={alpha_p:+}, α_heat={alpha_h:+})"
            f"\n  attained: fuel={fs:+.6f}, power={ps:+.6f}, heat={hs:+.6f}"
            f"\n  attained obj: {got:+.6f}   oracle obj: {want:+.6f}"
            f"\n  diff: {got - want:+.3e}  (TOL={TOL:.1e})"
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

    # ---- 3-variable CHP ------------------------------------------------

    @pytest.mark.parametrize("method_3var", ["sos2", "incremental"])
    def test_chp_at_breakpoint(self, method_3var: MethodND) -> None:
        """CHP at power=60 (exact breakpoint 2): max fuel=85, heat=55."""
        m = Model()
        power = m.add_variables(lower=0, upper=100, name="power")
        fuel = m.add_variables(lower=0, upper=200, name="fuel")
        heat = m.add_variables(lower=0, upper=100, name="heat")
        m.add_piecewise_formulation(
            (fuel, [0, 40, 85, 160]),
            (power, [0, 30, 60, 100]),
            (heat, [0, 25, 55, 95]),
            sign="<=",
            method=method_3var,
        )
        m.add_constraints(power == 60.0)
        m.add_objective(-fuel)
        m.solve()
        assert float(m.solution["fuel"]) == pytest.approx(85.0, abs=TOL)
        assert float(m.solution["heat"]) == pytest.approx(55.0, abs=TOL)

    @pytest.mark.parametrize("method_3var", ["sos2", "incremental"])
    def test_chp_at_midsegment(self, method_3var: MethodND) -> None:
        """
        CHP at power=45 (midway between bp1=30 and bp2=60):
        fuel = (40 + 85)/2 = 62.5,   heat = (25 + 55)/2 = 40.0.
        """
        m = Model()
        power = m.add_variables(lower=0, upper=100, name="power")
        fuel = m.add_variables(lower=0, upper=200, name="fuel")
        heat = m.add_variables(lower=0, upper=100, name="heat")
        m.add_piecewise_formulation(
            (fuel, [0, 40, 85, 160]),
            (power, [0, 30, 60, 100]),
            (heat, [0, 25, 55, 95]),
            sign="<=",
            method=method_3var,
        )
        m.add_constraints(power == 45.0)
        m.add_objective(-fuel)
        m.solve()
        assert float(m.solution["fuel"]) == pytest.approx(62.5, abs=TOL)
        assert float(m.solution["heat"]) == pytest.approx(40.0, abs=TOL)
