"""
A matrix-backed window of a parent model.

:class:`WindowModel` is what :meth:`Model.restrict` returns. It exposes the
window's flattened matrices directly, so solving never rebuilds a container, and
materialises variables and constraints only when their values are read.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from xarray import DataArray, Dataset

from linopy.restrict.matrix import (
    MatrixWindow,
    RestrictionError,
    WindowMatrices,
    WindowResult,
)

if TYPE_CHECKING:
    from linopy.model import Model
    from linopy.solvers import Solver

__all__ = ["WindowModel", "WindowModelView", "WindowObjective", "restrict_model"]


class _LabelIndex:
    """
    Dense positional labels.

    A window's result vectors are positional, so labels are ``0..n-1``. That
    keeps the solver's label-indexed solution mapping aligned with the flat
    arrays instead of allocating an array as large as the parent's label space.
    """

    def __init__(self, size: int) -> None:
        self.vlabels = self.clabels = self.label_to_pos = np.arange(size)
        self.n_active_vars = self.n_active_cons = size


class _KindSet:
    """Variables of one kind, with the ``labels`` attribute builders read."""

    def __init__(self, labels: np.ndarray) -> None:
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.size)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.labels)


class _VariablesView:
    """The ``model.variables`` surface a solver's direct builder reads."""

    def __init__(self, semi_continuous: _KindSet, size: int) -> None:
        self.sos: dict = {}
        self.semi_continuous = semi_continuous
        self.label_index = _LabelIndex(size)


class _ConstraintsView:
    """The ``model.constraints`` surface a builder reads."""

    def __init__(self, containers: int, size: int) -> None:
        self._containers = containers
        self.label_index = _LabelIndex(size)

    def sanitize_missings(self) -> None:
        return None

    def sanitize_zeros(self) -> None:
        return None

    def sanitize_infinities(self) -> None:
        return None

    def __len__(self) -> int:
        return self._containers


class WindowModelView:
    """
    The slice of :class:`~linopy.Model` that a solver's direct builder reads.

    Passing this to ``Solver._build`` lets every solver's existing build path
    consume a window unchanged, instead of each backend growing its own
    build-from-arrays routine.
    """

    set_names_in_solver_io = False
    is_quadratic = False
    indicator_constraints = ()

    def __init__(self, window: WindowModel) -> None:
        mat = window.matrices
        self.matrices = mat
        self.objective = window.objective
        self.sense = mat.objective_sense
        self.nvars = self._xCounter = mat.vlabels.size
        self.ncons = self._cCounter = mat.clabels.size
        kinds = {kind: _KindSet(np.flatnonzero(mat.vtypes == kind)) for kind in "BISC"}
        self.binaries = kinds["B"]
        self.integers = kinds["I"]
        self.semi_continuous = kinds["S"]
        self.continuous = kinds["C"]
        self.variables = _VariablesView(self.semi_continuous, self.nvars)
        self.constraints = _ConstraintsView(len(mat.constraints), self.ncons)

    @property
    def type(self) -> str:
        if not (self.binaries or self.integers or self.semi_continuous):
            return "LP"
        return "MILP" if self.continuous else "ILP"


class WindowObjective:
    """The window's objective value, as read off the last solve."""

    def __init__(self, window: WindowModel) -> None:
        self._window = window

    @property
    def value(self) -> float | None:
        result = self._window._result
        return None if result is None else result.objective

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
        self._matrices = self._matrix_window.select(self._selection, fix)
        self.objective = WindowObjective(self)
        self._result: WindowResult | None = None
        self._solver: Solver | None = None
        self._previous: WindowMatrices | None = None

    @property
    def matrices(self) -> WindowMatrices:
        return self._matrices

    def as_model_view(self) -> WindowModelView:
        """The window as the slice of ``Model`` a solver's builder consumes."""
        return WindowModelView(self)

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
        self._matrices = self._matrix_window.select(self._selection, self._fix)
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
            self, self._previous, disallow_rebuild=disallow_rebuild
        )
        self._previous = self._matrices
        return self

    @property
    def solution(self) -> Dataset:
        """The window's variable values, one array per variable."""
        return Dataset(self._primal(self._matrices.variables))

    @property
    def dual(self) -> Dataset:
        """The window's constraint duals, one array per constraint."""
        dual = self._solved().dual
        mat = self._matrices
        if dual.size != mat.clabels.size:
            raise AttributeError("The last solve returned no dual values.")
        values = dual * mat.con_scaling / mat.objective_scaling
        size = self._parent._cCounter
        return Dataset(_scatter(mat.constraints, mat.clabels, values, size))

    def __getitem__(self, name: str) -> DataArray:
        """A variable of the window, materialised on demand."""
        return self._primal({name: self._matrices.variables[name]})[name]

    def _primal(self, variables: dict[str, DataArray]) -> dict[str, DataArray]:
        """Primal values in model units, laid out like the given labels."""
        mat = self._matrices
        values = self._solved().primal / mat.var_scaling
        return _scatter(variables, mat.vlabels, values, self._parent._xCounter)

    def _solved(self) -> WindowResult:
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
