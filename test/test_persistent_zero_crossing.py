"""
Persistent coefficient updates must handle zero crossings.

Failure modes: crossing zero causes an unnecessary rebuild, a removed term
remains active in the solver, a restored term is not inserted, same-model
updates miss a dirty coefficient, or a fully zero row retains old coefficients.
"""

from __future__ import annotations

import numpy as np
import pytest
from linopy import Model
from linopy.solvers import Solver


def _activation_model(ratio: float, frozen: bool) -> Model:
    model = Model()
    activation = model.add_variables(0, 1, name="activation")
    reservation = model.add_variables(0, 1, name="reservation")
    model.add_constraints(
        activation <= ratio * reservation, name="limit", freeze=frozen
    )
    model.add_objective(-activation)
    return model


@pytest.mark.parametrize("ratios", [(1.0, 0.0, 1.0), (0.0, 1.0, 0.0)])
@pytest.mark.parametrize(
    "frozen,same_model", [(False, False), (True, False), (False, True)]
)
def test_zero_crossing_updates_match_fresh_solve(
    ratios: tuple[float, ...],
    frozen: bool,
    same_model: bool,
) -> None:
    solver = Solver.from_name(
        "highs",
        model=_activation_model(ratios[0], frozen),
        io_api="direct",
        track_updates=True,
        set_names=False,
    )
    solver.solve(assign=True)
    for ratio in ratios[1:]:
        if same_model:
            updated = solver.model
            activation = updated.variables["activation"]
            reservation = updated.variables["reservation"]
            updated.constraints["limit"].update(lhs=activation - ratio * reservation)
        else:
            updated = _activation_model(ratio, frozen)
        solver.update(updated)
        solver.solve(assign=True)
        fresh = _activation_model(ratio, frozen)
        fresh.solve(solver_name="highs", io_api="direct")
        np.testing.assert_allclose(updated.objective.value, fresh.objective.value)
        np.testing.assert_allclose(updated.objective.value, -ratio)
        assert solver._rebuilds == 0
    solver.close()


@pytest.mark.parametrize("frozen", [False, True])
def test_entire_row_zero_crossing_updates_in_place(frozen: bool) -> None:
    def build(ratio: float) -> Model:
        model = Model()
        x = model.add_variables(0, 1, name="x")
        model.add_constraints(ratio * x <= 0, name="limit", freeze=frozen)
        model.add_objective(-x)
        return model

    solver = Solver.from_name(
        "highs",
        model=build(1.0),
        io_api="direct",
        track_updates=True,
    )
    solver.solve(assign=True)
    for ratio in (0.0, 1.0):
        model = build(ratio)
        solver.solve(model, assign=True)
        np.testing.assert_allclose(model.objective.value, ratio - 1)
        assert solver._rebuilds == 0
    solver.close()
