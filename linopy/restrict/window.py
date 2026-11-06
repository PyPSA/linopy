"""
A matrix-backed window of a parent model.

:class:`WindowModel` is what :meth:`Model.restrict` returns. It exposes the
window's flattened matrices directly, so solving never rebuilds a container, and
materialises variables and constraints only when their values are read.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from xarray import DataArray

from linopy.restrict.matrix import MatrixWindow, RestrictionError

if TYPE_CHECKING:
    from linopy.constants import Result, Solution
    from linopy.matrices import Matrices
    from linopy.model import Model
    from linopy.solvers import Solver

__all__ = ["WindowModel", "WindowObjective", "restrict_model"]


class WindowObjective:
    """The window's objective value, as read off the last solve."""

    def __init__(self, window: WindowModel) -> None:
        self._window = window

    @property
    def value(self) -> float | None:
        """The objective value, ``None`` unless the last solve returned a solution."""
        solution = self._window._solution()
        if solution is None:
            return None
        return solution.objective / self._window.matrices.objective_scaling

    @property
    def sense(self) -> str:
        return self._window.matrices.objective_sense


class WindowModel:
    """
    A window of a parent model, backed by its flattened matrices.

    ``matrices`` is available immediately. ``solution``, ``dual`` and access to a
    named variable materialise that container from the flat result vectors, so a
    caller pays only for what it reads. ``status`` and ``termination_condition``
    report the outcome of the last solve, like on :class:`~linopy.Model`.
    """

    def __init__(
        self,
        parent: Model,
        selection: Mapping[str, Any],
        fix: Mapping[str, Any] | None = None,
    ) -> None:
        self._parent = parent
        self._matrix_window = MatrixWindow(parent)
        self._selection = dict(selection)
        self._fix = fix
        self._matrices, self._variables, self._constraints = self._matrix_window.select(
            self._selection, fix
        )
        self.objective = WindowObjective(self)
        self._result: Result | None = None
        self._solver: Solver | None = None

    @property
    def matrices(self) -> Matrices:
        return self._matrices

    @property
    def status(self) -> str:
        """Solver status of the last solve, ``"initialized"`` before any solve."""
        if self._result is None:
            return "initialized"
        return self._result.status.status.value

    @property
    def termination_condition(self) -> str:
        """Termination condition of the last solve, ``""`` before any solve."""
        if self._result is None:
            return ""
        return self._result.status.termination_condition.value

    def advance(
        self, fix: Mapping[str, Any] | None = None, **selection: Any
    ) -> WindowModel:
        """Move the window, recomputing the matrices for the new selection."""
        if selection:
            if selection.keys() != self._selection.keys():
                raise RestrictionError(
                    "Cannot advance model: the selection must name exactly the "
                    f"dimensions {list(self._selection)} of the previous selection, "
                    f"got {list(selection)}."
                )
            self._selection = selection
        self._fix = fix if fix is not None else self._fix
        self._matrices, self._variables, self._constraints = self._matrix_window.select(
            self._selection, self._fix
        )
        self._result = None
        return self

    def solve(
        self,
        solver_name: str | None = None,
        disallow_rebuild: bool = False,
        warm_start: bool = False,
        **options: Any,
    ) -> WindowModel:
        """
        Solve the window, updating an existing solver model in place when possible.

        By default each solve starts from scratch: after :meth:`advance` the
        previous window's basis is usually a poor starting point and slows the
        solver down. Pass ``warm_start=True`` to keep it, e.g. when re-solving
        the same window with slightly changed fixed values.

        Check :attr:`status` and :attr:`termination_condition` afterwards: an
        infeasible or otherwise failed window does not raise, but reading its
        values does.
        """
        from linopy.solvers import Solver

        if self._solver is None:
            self._solver = Solver.from_name(
                solver_name or "highs", io_api="direct", options=options
            )
        elif options:
            self._solver.solver_options.update(options)
        self._result = None  # a solve that raises leaves no stale result behind
        result = self._solver.solve_window(
            self._matrices, disallow_rebuild=disallow_rebuild, warm_start=warm_start
        )
        self._result = result
        _log_in_model_units(result, self._matrices.objective_scaling)
        return self

    def to_file(self, fn: str | Path, explicit_coordinate_names: bool = False) -> Path:
        """
        Write the window to an LP or MPS file, depending on the suffix of ``fn``.

        Columns and rows carry the parent's variable and constraint names, like
        :meth:`Model.to_file <linopy.model.Model.to_file>` writes them, and the
        coefficients are in solver-scaled units, as the solver sees them.
        """
        from linopy.io import get_printers_scalar, matrices_to_file

        fn = Path(fn)
        if fn.suffix not in (".lp", ".mps"):
            raise ValueError(f"Cannot write '{fn}': expected an '.lp' or '.mps' file.")
        names = get_printers_scalar(self._parent, explicit_coordinate_names)
        matrices_to_file(self._matrices, fn, names, sos=[])
        return fn

    def compute_infeasibilities(self) -> list[int]:
        """
        Parent constraint labels of an irreducible infeasible subset of the window.

        Requires the last :meth:`solve` to have found the window infeasible with
        ``gurobi``, ``xpress`` or ``highs``. Format the labels with the parent's
        :meth:`Constraints.format_labels <linopy.constraints.Constraints.format_labels>`.
        """
        from linopy.infeasibility import compute_infeasibilities

        if self._result is None:
            raise ValueError(
                "The window has not been solved. Call `solve()` before computing "
                "infeasibilities."
            )
        return compute_infeasibilities(self._solver, self._matrices.clabels)

    @property
    def solution(self) -> dict[str, DataArray]:
        """
        The window's variable values, one array per variable.

        A mapping rather than a ``Dataset``, so that variables sharing a
        dimension name with different coordinates keep their own coordinates.
        """
        return self._primal(self._variables)

    @property
    def dual(self) -> dict[str, DataArray]:
        """The window's constraint duals, one array per constraint, see :attr:`solution`."""
        dual = self._solved().dual
        mat = self._matrices
        if dual.size != mat.clabels.size:
            raise AttributeError("The last solve returned no dual values.")
        values = dual * mat.con_scaling / mat.objective_scaling
        size = self._parent._cCounter
        return _scatter(self._constraints, mat.clabels, values, size)

    def __getitem__(self, name: str) -> DataArray:
        """A variable of the window, materialised on demand."""
        return self._primal({name: self._variables[name]})[name]

    def _primal(self, variables: dict[str, DataArray]) -> dict[str, DataArray]:
        """Primal values in model units, laid out like the given labels."""
        mat = self._matrices
        values = self._solved().primal / mat.var_scaling
        return _scatter(variables, mat.vlabels, values, self._parent._xCounter)

    def _solution(self) -> Solution | None:
        """The last solve's solution, ``None`` unless it succeeded with values."""
        result = self._result
        if result is None or not result.status.is_ok:
            return None
        solution = result.solution
        if solution is None or solution.primal.size != self._matrices.vlabels.size:
            return None
        return solution

    def _solved(self) -> Solution:
        if self._result is None:
            raise AttributeError(
                "The window has not been solved. Call `solve()` before reading values."
            )
        solution = self._solution()
        if solution is None:
            raise AttributeError(
                "The last solve returned no solution (status "
                f"'{self.status}', termination condition "
                f"'{self.termination_condition}')."
            )
        return solution


def _scatter(
    containers: dict[str, DataArray], labels: np.ndarray, values: np.ndarray, size: int
) -> dict[str, DataArray]:
    """Flat values by label, laid out like each container's windowed labels."""
    by_label = np.full(size, np.nan)
    by_label[labels] = values
    return {
        name: raw.copy(data=np.where(raw.values != -1, by_label[raw.values], np.nan))
        for name, raw in containers.items()
    }


def restrict_model(
    model: Model,
    fix: Mapping[str, Any] | None,
    selection: Mapping[str, Any],
) -> WindowModel:
    """Build a matrix-backed window of ``model``, see :meth:`linopy.Model.restrict`."""
    return WindowModel(model, selection, fix)


def _log_in_model_units(result: Result, objective_scaling: float) -> None:
    """Log *result* like :meth:`Model.solve`, with the objective in model units."""
    solution, report = result.solution, result.report
    if solution is not None:
        solution = replace(solution, objective=solution.objective / objective_scaling)
    if report is not None and report.dual_bound is not None:
        report = replace(report, dual_bound=report.dual_bound / objective_scaling)
    replace(result, solution=solution, report=report).info()
