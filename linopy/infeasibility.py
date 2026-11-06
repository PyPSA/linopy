"""
Irreducible infeasible subsystems (IIS) of a solver's native model.

Shared by :meth:`linopy.Model.compute_infeasibilities` and
:meth:`linopy.restrict.WindowModel.compute_infeasibilities`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from linopy.solvers import SolverFeature, _names_to_labels, available_solvers

if TYPE_CHECKING:
    from linopy.solvers import Solver

__all__ = ["compute_infeasibilities"]


def compute_infeasibilities(solver: Solver | None, clabels: np.ndarray) -> list[int]:
    """
    Sorted constraint labels of an IIS of the solver's native model.

    ``clabels`` maps the native model's row positions to constraint labels. For
    a model read from a file, Gurobi and HiGHS parse the labels from the row
    names instead.
    """
    if solver is None:
        raise NotImplementedError(
            "Computing infeasibilities is not supported for 'unknown' solver. "
            "Only Gurobi, Xpress and HiGHS solvers support IIS computation."
        )
    solver_model = solver.solver_model

    if "gurobi" in available_solvers:
        import gurobipy

        if isinstance(solver_model, gurobipy.Model):
            return _gurobi(solver_model, clabels, solver.io_api == "direct")

    if "xpress" in available_solvers:
        import xpress

        if isinstance(solver_model, xpress.problem):
            return _xpress(solver_model, clabels)

    if "highs" in available_solvers:
        import highspy

        if isinstance(solver_model, highspy.Highs):
            return _highs(solver_model, clabels, solver)

    if solver_model is not None:
        raise NotImplementedError(
            "Computing infeasibilities is only supported for Gurobi, Xpress and HiGHS solvers. "
            f"Current solver model type: {type(solver_model).__name__}"
        )
    if solver.supports(SolverFeature.IIS_COMPUTATION):
        raise ValueError(
            "No solver model available. The model must be solved first with "
            "a solver that supports IIS computation and the result must be infeasible."
        )
    raise NotImplementedError(
        f"Computing infeasibilities is not supported for '{solver.solver_name.value}' "
        "solver. Only Gurobi, Xpress and HiGHS solvers support IIS computation."
    )


def _gurobi(solver_model: Any, clabels: np.ndarray, direct: bool) -> list[int]:
    solver_model.computeIIS()
    constrs = solver_model.getConstrs()
    in_iis = np.asarray(solver_model.getAttr("IISConstr", constrs), dtype=bool)
    if not direct:
        clabels = _names_to_labels([c.ConstrName for c in constrs])
    return sorted(int(label) for label in clabels[in_iis] if label >= 0)


def _xpress(solver_model: Any, clabels: np.ndarray) -> list[int]:
    """
    A single IIS, found with Xpress' fast search (mode 2) over minimality.

    Rows are mapped by position, so masked constraints with gaps in their labels
    are handled.
    """
    try:
        solver_model.firstIIS(2)
    except AttributeError:
        solver_model.iisfirst(2)

    if solver_model.attributes.numiis == 0:
        return []

    position_to_label = {
        constraint: clabels[position]
        for position, constraint in enumerate(solver_model.getConstraint())
        if position < len(clabels) and clabels[position] >= 0
    }
    return sorted(
        {
            int(position_to_label[constraint])
            for constraint in _xpress_iis_constraints(solver_model, 1)
            if constraint in position_to_label
        }
    )


def _xpress_iis_constraints(solver_model: Any, iis_num: int) -> list[Any]:
    """The xpress.constraint objects of IIS number ``iis_num`` (1-indexed)."""
    try:
        rows = solver_model.getIISData(iis_num)[0]
    except AttributeError:
        constraints: list[Any] = []
        solver_model.getiisdata(iis_num, constraints, *([] for _ in range(7)))
        return constraints
    return [solver_model.getConstraint(row) for row in rows]


def _highs(solver_model: Any, clabels: np.ndarray, solver: Solver) -> list[int]:
    import highspy

    if "iis_strategy" not in solver.solver_options:
        from_lp = int(highspy.IisStrategy.kIisStrategyFromLp)
        irreducible = int(highspy.IisStrategy.kIisStrategyIrreducible)
        solver_model.setOptionValue("iis_strategy", from_lp | irreducible)

    status, iis = solver_model.getIis()
    if status == highspy.HighsStatus.kError or not iis.valid_:
        raise RuntimeError(
            "HiGHS failed to compute an irreducible infeasible subsystem (IIS)."
        )

    row_index = np.asarray(iis.row_index_, dtype=np.intp)
    if not len(row_index):
        return []

    if solver.io_api != "direct":
        clabels = _names_to_labels(solver_model.getLp().row_names_)

    return sorted({int(clabels[pos]) for pos in row_index if clabels[pos] >= 0})
