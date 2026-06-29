"""
Tests for the ``active_fill`` parameter of ``add_piecewise_formulation`` (#796).

``active_fill`` is a transitional convenience: it pads a partial ``active``
gate (a subset of the indexed dimension, or a masked gate) to full coverage.
It is slated for removal once the v1 arithmetic semantics (#717) make
``active.reindex(coords).fillna(value)`` correct on its own, so these tests
live in a dedicated module that can be dropped with the parameter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model, available_solvers, segments
from linopy.piecewise import _resolve_active
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


def _full_gate(m: Model) -> Any:
    return m.add_variables(binary=True, coords=[_PWL_GENS], name="u")


def _scalar_gate(m: Model) -> Any:
    return m.add_variables(binary=True, name="u")


_PARTIAL_GATES = [
    pytest.param(_subset_gate, id="strict-subset"),
    pytest.param(_masked_gate, id="masked"),
]

# (builder, active_fill, should_raise): partial gates raise unless active_fill
# is set; full/scalar gates are always fine.
_COVERAGE_CASES = [
    pytest.param(_subset_gate, None, True, id="subset-None-raises"),
    pytest.param(_masked_gate, None, True, id="masked-None-raises"),
    pytest.param(_subset_gate, 1, False, id="subset-fill1-ok"),
    pytest.param(_masked_gate, 1, False, id="masked-fill1-ok"),
    pytest.param(_subset_gate, 0, False, id="subset-fill0-ok"),
    pytest.param(_full_gate, None, False, id="full-ok"),
    pytest.param(_scalar_gate, None, False, id="scalar-ok"),
]


def _solve_partial_gate(
    solver_name: str,
    make_active: GateBuilder,
    *,
    method: Method,
    disjunctive: bool = False,
) -> None:
    """Fill a partial gate, force the committable units off, demand "b" runs."""
    m = Model()
    x = m.add_variables(lower=0, upper=100, coords=[_PWL_GENS], name="x")
    y = m.add_variables(lower=0, coords=[_PWL_GENS], name="y")
    u = make_active(m)
    if disjunctive:
        m.add_piecewise_formulation(
            (x, segments([[0.0, 50.0], [50.0, 100.0]])),
            (y, segments([[0.0, 10.0], [10.0, 50.0]])),
            active=u,
            active_fill=1,
        )
    else:
        m.add_piecewise_formulation(
            (x, [0, 50, 100]),
            (y, [0, 10, 50]),
            active=u,
            active_fill=1,
            method=method,
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


class TestResolveActiveFill:
    """The private ``_resolve_active`` fills gaps with ``active_fill``."""

    @pytest.mark.parametrize("fill_value", [1, 0])
    @pytest.mark.parametrize("make_active", _PARTIAL_GATES)
    def test_fills_gap(self, make_active: GateBuilder, fill_value: int) -> None:
        reference = xr.DataArray(np.zeros(len(_PWL_GENS)), coords=[_PWL_GENS])
        gate = _resolve_active(1 * make_active(Model()), reference, fill_value)
        assert gate.const.sel(gen="b").item() == fill_value
        assert bool((gate.vars.sel(gen="b") < 0).all())  # no variable at "b"
        assert bool((gate.vars.sel(gen="a") >= 0).any())  # variable kept at "a"


class TestActiveFillValidation:
    """``add_piecewise_formulation`` gates a partial ``active`` via ``active_fill``."""

    @pytest.mark.parametrize("make_active, active_fill, should_raise", _COVERAGE_CASES)
    def test_coverage(
        self,
        make_active: GateBuilder,
        active_fill: int | None,
        should_raise: bool,
    ) -> None:
        m = Model()
        x = m.add_variables(lower=0, upper=100, coords=[_PWL_GENS], name="x")
        y = m.add_variables(lower=0, coords=[_PWL_GENS], name="y")

        def build() -> None:
            m.add_piecewise_formulation(
                (x, [0, 50, 100]),
                (y, [0, 10, 50]),
                active=make_active(m),
                active_fill=active_fill,
                method="incremental",
            )

        if should_raise:
            with pytest.raises(ValueError, match="active_fill"):
                build()
        else:
            build()

    def test_active_fill_without_active_raises(self) -> None:
        m = Model()
        x = m.add_variables(lower=0, upper=100, coords=[_PWL_GENS], name="x")
        y = m.add_variables(lower=0, coords=[_PWL_GENS], name="y")
        with pytest.raises(ValueError, match="without `active`"):
            m.add_piecewise_formulation(
                (x, [0, 50, 100]),
                (y, [0, 10, 50]),
                active_fill=1,
                method="incremental",
            )

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
class TestSolverActiveFill:
    """End-to-end: ``active_fill`` leaves ungated units free (#796)."""

    @pytest.fixture(params=_any_solvers)
    def solver_name(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.mark.parametrize("make_active", _PARTIAL_GATES)
    def test_incremental(self, solver_name: str, make_active: GateBuilder) -> None:
        _solve_partial_gate(solver_name, make_active, method="incremental")


@pytest.mark.skipif(len(_sos2_solvers) == 0, reason="No SOS2-capable solver")
class TestSolverActiveFillSOS2:
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
