"""
A matrix-backed window of a parent model.

:class:`WindowModel` is what :meth:`Model.restrict` returns. It exposes the
window's flattened matrices directly, so solving never rebuilds a container, and
materialises variables and constraints only when their values are read.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from xarray import DataArray, Dataset

from linopy.restrict.matrix import MatrixWindow, RestrictionError

if TYPE_CHECKING:
    from linopy.constants import Solution
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
        result = self._window._result
        if result is None:
            return None
        return result.objective / self._window.matrices.objective_scaling

    @property
    def sense(self) -> str:
        return self._window.matrices.objective_sense


class WindowModel:
    """
    A window of a parent model, backed by its flattened matrices.

    ``matrices`` is available immediately. ``solution``, ``dual`` and access to a
    named variable materialise that container from the flat result vectors, so a
    caller pays only for what it reads.
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
        self._result: Solution | None = None
        self._solver: Solver | None = None

    @property
    def matrices(self) -> Matrices:
        return self._matrices

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
        **options: Any,
    ) -> WindowModel:
        """Solve the window, updating an existing solver model in place when possible."""
        from linopy.solvers import Solver

        if self._solver is None:
            self._solver = Solver.from_name(
                solver_name or "highs", io_api="direct", options=options
            )
        elif options:
            self._solver.solver_options.update(options)
        self._result = self._solver.solve_window(
            self._matrices, disallow_rebuild=disallow_rebuild
        )
        return self

    @property
    def solution(self) -> Dataset:
        """The window's variable values, one array per variable."""
        return Dataset(self._primal(self._variables))

    @property
    def dual(self) -> Dataset:
        """The window's constraint duals, one array per constraint."""
        dual = self._solved().dual
        mat = self._matrices
        if dual.size != mat.clabels.size:
            raise AttributeError("The last solve returned no dual values.")
        values = dual * mat.con_scaling / mat.objective_scaling
        size = self._parent._cCounter
        return Dataset(_scatter(self._constraints, mat.clabels, values, size))

    def __getitem__(self, name: str) -> DataArray:
        """A variable of the window, materialised on demand."""
        return self._primal({name: self._variables[name]})[name]

    def _primal(self, variables: dict[str, DataArray]) -> dict[str, DataArray]:
        """Primal values in model units, laid out like the given labels."""
        mat = self._matrices
        values = self._solved().primal / mat.var_scaling
        return _scatter(variables, mat.vlabels, values, self._parent._xCounter)

    def _solved(self) -> Solution:
        if self._result is None:
            raise AttributeError(
                "The window has not been solved. Call `solve()` before reading values."
            )
        return self._result


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
