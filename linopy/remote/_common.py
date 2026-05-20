"""
Shared helpers for the standalone remote-handler classes (``Oetc``, ``SSH``).

These handlers do not inherit from :class:`linopy.solvers.Solver` — they're
a parallel concept. The helpers here cover the two pieces of plumbing
both handlers need: validating the inner-solver string locally, and
mapping a round-tripped solved :class:`~linopy.model.Model` back onto
the source model's label space.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from linopy.constants import Solution

if TYPE_CHECKING:
    from linopy.model import Model


def _validate_inner_solver(inner_solver_name: str, model: Model) -> None:
    """
    Check that the inner-solver string is locally known and
    that the inner solver's feature set covers the model.

    Local installation is *not* required — feature flags are class-level
    metadata. We only need the class to introspect ``supports(...)``.
    Unknown solver names raise so typos fail fast instead of incurring a
    round-trip to the worker.
    """
    # Imported here to avoid a circular import at module load.
    from linopy.solvers import SolverFeature, SolverName, _solver_class_for

    cls = _solver_class_for(inner_solver_name)
    if cls is None:
        valid = ", ".join(sorted(n.value for n in SolverName))
        raise ValueError(
            f"Unknown solver name {inner_solver_name!r}. Pick one of: {valid}."
        )
    if model.is_quadratic and not cls.supports(SolverFeature.QUADRATIC_OBJECTIVE):
        raise ValueError(
            f"Solver {inner_solver_name!r} does not support quadratic problems."
        )
    if model.variables.semi_continuous and not cls.supports(
        SolverFeature.SEMI_CONTINUOUS_VARIABLES
    ):
        raise ValueError(
            f"Solver {inner_solver_name!r} does not support semi-continuous "
            "variables. Use a solver that supports them (gurobi, cplex, highs)."
        )
    if model.variables.sos and not cls.supports(SolverFeature.SOS_CONSTRAINTS):
        raise ValueError(
            f"Solver {inner_solver_name!r} does not support SOS constraints. "
            "Reformulate first via `Model.solve(reformulate_sos=True)` or "
            "`model.apply_sos_reformulation()`, or pick a solver that supports SOS."
        )


def _scatter_solution_from_solved_model(
    local_model: Model, solved: Model, n_vars: int, n_cons: int
) -> Solution:
    """
    Build a label-indexed :class:`~linopy.constants.Solution` from a
    round-tripped solved model.

    The labels on ``solved`` match ``local_model`` because both sides
    serialize/load with the same linopy version; we use the local labels
    as the index. Missing slots stay ``NaN``; constraints without
    ``dual`` are skipped.
    """
    primal = np.full(n_vars, np.nan, dtype=float)
    dual = np.full(n_cons, np.nan, dtype=float)
    for name, var in local_model.variables.items():
        sol = solved.variables[name].solution
        primal[var.labels.values.ravel()] = sol.values.ravel()
    for name, con in local_model.constraints.items():
        if "dual" not in solved.constraints[name]:
            continue
        dual[con.labels.values.ravel()] = solved.constraints[name].dual.values.ravel()

    objective_value = solved.objective.value
    objective = float(objective_value) if objective_value is not None else float("nan")
    return Solution(primal=primal, dual=dual, objective=objective)
