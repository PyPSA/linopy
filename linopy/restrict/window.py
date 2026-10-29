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

from linopy.restrict.matrix import MatrixWindow, RestrictionError, WindowMatrices

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
        labels = np.arange(size)
        self.vlabels = labels
        self.clabels = labels
        self.label_to_pos = labels
        self.n_active_vars = size
        self.n_active_cons = size


class _KindSet:
    """Variables of one kind, with the ``labels`` attribute builders read."""

    def __init__(self, labels: np.ndarray) -> None:
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.size)

    def __bool__(self) -> bool:
        return bool(self.labels.size)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.labels)


class _VariablesView:
    """The ``model.variables`` surface a solver's direct builder reads."""

    def __init__(self, view: WindowModelView) -> None:
        self._view = view

    @property
    def sos(self) -> dict:
        return {}

    @property
    def semi_continuous(self) -> _KindSet:
        return self._view.semi_continuous

    @property
    def label_index(self) -> _LabelIndex:
        return _LabelIndex(len(self._view.matrices.vlabels))


class _ConstraintsView:
    """The ``model.constraints`` surface a builder reads."""

    def __init__(self, window: WindowModel) -> None:
        self._window = window

    def sanitize_missings(self) -> None:
        return None

    def sanitize_zeros(self) -> None:
        return None

    def sanitize_infinities(self) -> None:
        return None

    def __len__(self) -> int:
        return len(self._window.matrices.constraints)

    @property
    def label_index(self) -> _LabelIndex:
        return _LabelIndex(len(self._window.matrices.clabels))


class WindowModelView:
    """
    The slice of :class:`~linopy.Model` that a solver's direct builder reads.

    Passing this to ``Solver._build`` lets every solver's existing build path
    consume a window unchanged, instead of each backend growing its own
    build-from-arrays routine.
    """

    def __init__(self, window: WindowModel) -> None:
        self._window = window
        self.variables = _VariablesView(self)
        self.constraints = _ConstraintsView(window)

    @property
    def matrices(self) -> WindowMatrices:
        return self._window.matrices

    @property
    def objective(self) -> WindowObjective:
        return self._window.objective

    @property
    def sense(self) -> str:
        return self.matrices.objective_sense

    @property
    def set_names_in_solver_io(self) -> bool:
        return False

    @property
    def is_quadratic(self) -> bool:
        return False

    @property
    def nvars(self) -> int:
        return len(self.matrices.vlabels)

    @property
    def ncons(self) -> int:
        return len(self.matrices.clabels)

    @property
    def _xCounter(self) -> int:
        return len(self.matrices.vlabels)

    @property
    def _cCounter(self) -> int:
        return len(self.matrices.clabels)

    @property
    def indicator_constraints(self) -> tuple:
        return ()

    def _of_kind(self, kind: str) -> _KindSet:
        return _KindSet(np.flatnonzero(self.matrices.vtypes == kind))

    @property
    def binaries(self) -> _KindSet:
        return self._of_kind("B")

    @property
    def integers(self) -> _KindSet:
        return self._of_kind("I")

    @property
    def semi_continuous(self) -> _KindSet:
        return self._of_kind("S")

    @property
    def continuous(self) -> _KindSet:
        return self._of_kind("C")

    @property
    def type(self) -> str:
        discrete = bool(self.binaries or self.integers or self.semi_continuous)
        if not discrete:
            return "LP"
        return "MILP" if self.continuous else "ILP"


class WindowObjective:
    """The window's objective value, as read off the last solve."""

    def __init__(self, window: WindowModel) -> None:
        self._window = window

    @property
    def value(self) -> float | None:
        return self._window._objective_value

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
        self._view = MatrixWindow(parent)
        self._fix = fix
        self._selection = dict(selection)
        self._matrices = self._view.select(self._selection, fix)
        self._objective = WindowObjective(self)
        self._objective_value: float | None = None
        self._primal: np.ndarray | None = None
        self._dual: np.ndarray | None = None
        self._solver: Solver | None = None
        self._previous: WindowMatrices | None = None

    @property
    def objective(self) -> WindowObjective:
        return self._objective

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
        self._matrices = self._view.select(self._selection, self._fix)
        self._objective_value = None
        self._primal = None
        self._dual = None
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
        result = self._solver.solve_window(
            self, self._previous, disallow_rebuild=disallow_rebuild
        )
        self._objective_value = result.objective
        self._primal = result.primal
        self._dual = result.dual
        self._previous = self._matrices
        return self

    @property
    def solution(self) -> Dataset:
        """The window's variable values, one array per variable."""
        by_label = self._primal_by_label()
        return Dataset(
            {
                name: _scatter(labels, by_label)
                for name, labels in self._matrices.variables.items()
            }
        )

    @property
    def dual(self) -> Dataset:
        """The window's constraint duals, one array per constraint."""
        self._require_solve()
        mat = self._matrices
        if self._dual is None or self._dual.size != mat.clabels.size:
            raise AttributeError("The last solve returned no dual values.")
        values = self._dual * mat.con_scaling / mat.objective_scaling
        by_label = _by_label(mat.clabels, values, self._parent._cCounter)
        return Dataset(
            {
                name: _scatter(labels, by_label)
                for name, labels in mat.constraints.items()
            }
        )

    def __getitem__(self, name: str) -> DataArray:
        """A variable of the window, materialised on demand."""
        if name not in self._matrices.variables:
            raise KeyError(name)
        return _scatter(self._matrices.variables[name], self._primal_by_label())

    def _primal_by_label(self) -> np.ndarray:
        """Primal values in model units by parent label."""
        mat = self._matrices
        values = self._require_solve() / mat.var_scaling
        return _by_label(mat.vlabels, values, self._parent._xCounter)

    def _require_solve(self) -> np.ndarray:
        if self._primal is None:
            raise AttributeError(
                "The window has not been solved. Call `solve()` before reading values."
            )
        return self._primal


def _by_label(labels: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    """Flat values indexed by label, NaN for labels outside the window."""
    by_label = np.full(size, np.nan)
    by_label[labels] = values
    return by_label


def _scatter(labels: DataArray, by_label: np.ndarray) -> DataArray:
    """Values of one container, laid out like its windowed labels."""
    raw = labels.values
    return labels.copy(data=np.where(raw != -1, by_label[raw], np.nan))


def restrict_model(
    model: Model,
    fix: Mapping[str, Any] | None,
    selection: Mapping[str, Any],
) -> WindowModel:
    """Build a matrix-backed window of ``model``, see :meth:`linopy.Model.restrict`."""
    return WindowModel(model, selection, fix)
