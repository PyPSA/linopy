"""
Tests for the partial-``active`` gate and the ``active_gate`` helper (#796).

Kept in a dedicated module because ``active_gate`` is a temporary legacy
stopgap (see ``linopy/_active_gate.py``): once the v1 arithmetic semantics
(#717) land, the helper is expected to be deprecated and these tests removed
or rewritten against the bare ``reindex().fillna()`` idiom.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
import pytest

from linopy import Model, active_gate, available_solvers, segments
from linopy.solver_capabilities import (
    SolverFeature,
    get_available_solvers_with_feature,
)

Method: TypeAlias = Literal["sos2", "incremental", "lp", "auto"]
GateBuilder: TypeAlias = Callable[[Model], Any]

_any_solvers = [
    s for s in ["highs", "gurobi", "glpk", "cplex"] if s in available_solvers
]
_sos2_solvers = get_available_solvers_with_feature(
    SolverFeature.SOS_CONSTRAINTS, available_solvers
)


# ``active`` is meaningful only for the committable subset {a, c}; "b" stays
# ungated. The partial-gate shapes below all leave "b" as the gap.
_PWL_GENS = pd.Index(["a", "b", "c"], name="gen")
_COMMITTABLE = pd.Index(["a", "c"], name="gen")


def _subset_gate(m: Model) -> Any:
    """``active`` indexed over a strict subset of the formulation's dim."""
    return m.add_variables(binary=True, coords=[_COMMITTABLE], name="u")


def _masked_gate(m: Model) -> Any:
    """``active`` over the full dim but masked where it does not apply."""
    mask = pd.Series([True, False, True], index=_PWL_GENS)
    return m.add_variables(binary=True, coords=[_PWL_GENS], name="u", mask=mask)


_PARTIAL_GATES = [
    pytest.param(_subset_gate, id="strict-subset"),
    pytest.param(_masked_gate, id="masked"),
]


def _full_gate(m: Model) -> Any:
    return m.add_variables(binary=True, coords=[_PWL_GENS], name="u")


def _scalar_gate(m: Model) -> Any:
    return m.add_variables(binary=True, name="u")


def _padded(make: GateBuilder, fill_value: float = 1) -> GateBuilder:
    return lambda m: active_gate(make(m), {"gen": _PWL_GENS}, fill_value)


# (builder, should_raise): raw partial gates are rejected; padded/full/scalar ok.
_COVERAGE_CASES = [
    pytest.param(_subset_gate, True, id="strict-subset-raises"),
    pytest.param(_masked_gate, True, id="masked-raises"),
    pytest.param(_padded(_subset_gate), False, id="padded-subset-ok"),
    pytest.param(_padded(_masked_gate), False, id="padded-masked-ok"),
    pytest.param(_padded(_subset_gate, 0), False, id="padded-off-ok"),
    pytest.param(_full_gate, False, id="full-ok"),
    pytest.param(_scalar_gate, False, id="scalar-ok"),
]


def _solve_partial_gate(
    solver_name: str,
    make_active: GateBuilder,
    *,
    method: Method,
    disjunctive: bool = False,
) -> None:
    """Pad a partial gate, force the committable units off, demand "b" runs."""
    m = Model()
    x = m.add_variables(lower=0, upper=100, coords=[_PWL_GENS], name="x")
    y = m.add_variables(lower=0, coords=[_PWL_GENS], name="y")
    u = make_active(m)
    gate = active_gate(u, {"gen": _PWL_GENS})
    if disjunctive:
        m.add_piecewise_formulation(
            (x, segments([[0.0, 50.0], [50.0, 100.0]])),
            (y, segments([[0.0, 10.0], [10.0, 50.0]])),
            active=gate,
        )
    else:
        m.add_piecewise_formulation(
            (x, [0, 50, 100]), (y, [0, 10, 50]), active=gate, method=method
        )
    m.add_constraints(u <= 0, name="force_off")
    m.add_constraints(x.sel(gen="b") >= 50, name="demand")
    m.add_objective(y.sum(), sense="min")
    status, _ = m.solve(solver_name=solver_name)
    assert status == "ok"
    np.testing.assert_allclose(float(x.solution.sel(gen="a")), 0, atol=1e-4)
    np.testing.assert_allclose(float(x.solution.sel(gen="c")), 0, atol=1e-4)
    np.testing.assert_allclose(float(x.solution.sel(gen="b")), 50, atol=1e-4)
    np.testing.assert_allclose(float(y.solution.sel(gen="b")), 10, atol=1e-4)


class TestActiveGateHelper:
    """``active_gate`` pads a partial gate; gaps -> ``fill_value``."""

    @pytest.mark.parametrize("fill_value", [1, 0])
    @pytest.mark.parametrize(
        "make_active",
        [*_PARTIAL_GATES, pytest.param(lambda m: 2 * _subset_gate(m), id="linexpr")],
    )
    def test_fills_gap(self, make_active: GateBuilder, fill_value: float) -> None:
        gate = active_gate(make_active(Model()), {"gen": _PWL_GENS}, fill_value)
        assert gate.const.sel(gen="b").item() == fill_value
        assert bool((gate.vars.sel(gen="b") < 0).all())
        assert bool((gate.vars.sel(gen="a") >= 0).any())


class TestPartialActiveValidation:
    """``add_piecewise_formulation`` rejects an under-defined ``active`` (#796)."""

    @pytest.mark.parametrize("make_active, should_raise", _COVERAGE_CASES)
    def test_coverage(self, make_active: GateBuilder, should_raise: bool) -> None:
        m = Model()
        x = m.add_variables(lower=0, upper=100, coords=[_PWL_GENS], name="x")
        y = m.add_variables(lower=0, coords=[_PWL_GENS], name="y")

        def build() -> None:
            m.add_piecewise_formulation(
                (x, [0, 50, 100]),
                (y, [0, 10, 50]),
                active=make_active(m),
                method="incremental",
            )

        if should_raise:
            with pytest.raises(ValueError, match="active_gate"):
                build()
        else:
            build()

    def test_lower_dimensional_active_broadcasts(self) -> None:
        """A gate missing an entire dim broadcasts and must not be rejected."""
        ts = pd.Index([0, 1], name="t")
        m = Model()
        x = m.add_variables(lower=0, upper=100, coords=[_PWL_GENS, ts], name="x")
        y = m.add_variables(lower=0, coords=[_PWL_GENS, ts], name="y")
        u = m.add_variables(binary=True, coords=[_PWL_GENS], name="u")
        m.add_piecewise_formulation(
            (x, [0, 50, 100]), (y, [0, 10, 50]), active=u, method="incremental"
        )


@pytest.mark.skipif(len(_any_solvers) == 0, reason="No solver available")
class TestSolverPartialActiveGate:
    """End-to-end: a padded partial gate leaves ungated units free (#796)."""

    @pytest.fixture(params=_any_solvers)
    def solver_name(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.mark.parametrize("make_active", _PARTIAL_GATES)
    def test_incremental(self, solver_name: str, make_active: GateBuilder) -> None:
        _solve_partial_gate(solver_name, make_active, method="incremental")


@pytest.mark.skipif(len(_sos2_solvers) == 0, reason="No SOS2-capable solver")
class TestSolverPartialActiveGateSOS2:
    @pytest.fixture(params=_sos2_solvers)
    def solver_name(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.mark.parametrize("make_active", _PARTIAL_GATES)
    @pytest.mark.parametrize(
        "method, disjunctive",
        [
            pytest.param("sos2", False, id="sos2"),
            pytest.param("auto", True, id="disjunctive"),
        ],
    )
    def test_solves(
        self,
        solver_name: str,
        make_active: GateBuilder,
        method: Method,
        disjunctive: bool,
    ) -> None:
        _solve_partial_gate(
            solver_name, make_active, method=method, disjunctive=disjunctive
        )
