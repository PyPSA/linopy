"""
Shared helper for the standalone remote classes (``Oetc``, ``SSH``).

These classes do not inherit from :class:`linopy.solvers.Solver` — they're
a parallel concept. The helper here validates the solver string locally
before the round-trip to the worker, so an unknown name or an unsupported
feature fails fast instead of after the upload.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
