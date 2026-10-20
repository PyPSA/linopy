"""
A matrix-backed window of a parent model.

:class:`WindowModel` is what :meth:`Model.restrict` returns. It exposes the
window's flattened matrices directly, so solving never rebuilds a container, and
materialises variables and constraints only when their values are read.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from xarray import DataArray, Dataset

from linopy.restrict.errors import RestrictionError
from linopy.restrict.matrix import MatrixWindow, WindowMatrices

if TYPE_CHECKING:
    from linopy.model import Model
    from linopy.solvers import Solver

__all__ = ["WindowModel", "WindowObjective"]


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
        return len(self._window.matrices.con_blocks)

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
    named variable or constraint materialise that container from the flat result
    vectors, so a caller pays only for what it reads.
    """

    def __init__(
        self,
        parent: Model,
        dim: str,
        selection: dict[str, Any],
        fix: dict[str, Any] | None = None,
    ) -> None:
        self._parent = parent
        self._dim = dim
        self._view = MatrixWindow(parent, dim)
        self._fix = fix
        self._selection = dict(selection)
        self._matrices = self._view.select(pd.Index(selection[dim]), fix)
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
        self, fix: dict[str, Any] | None = None, **selection: Any
    ) -> WindowModel:
        """Move the window, recomputing the matrices for the new selection."""
        if selection:
            missing = [dim for dim in self._selection if dim not in selection]
            if missing:
                raise RestrictionError(
                    "Cannot advance model: the selection must name all dimensions "
                    f"of the previous selection, missing {missing}."
                )
            self._selection = {dim: selection[dim] for dim in self._selection}
        self._fix = fix if fix is not None else self._fix
        self._previous = self._matrices
        self._matrices = self._view.select(
            pd.Index(self._selection[self._dim]), self._fix
        )
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

    # --- materialisation -----------------------------------------------------

    @property
    def solution(self) -> Dataset:
        """The window's variable values, one array per variable."""
        self._require_solve()
        assert self._primal is not None
        data = {}
        for name, var in self._parent.variables.items():
            start, stop = self._matrices.var_blocks[name]
            # the solver works in scaled columns; report the model's units
            values = self._primal[start:stop] / self._matrices.var_scaling[start:stop]
            data[name] = self._reshape(var, values, self._matrices.var_active[name])
        return Dataset(data)

    @property
    def dual(self) -> Dataset:
        """The window's constraint duals, one array per constraint."""
        self._require_solve()
        expected = sum(
            stop - start for start, stop in self._matrices.con_blocks.values()
        )
        if self._dual is None or self._dual.size != expected:
            raise AttributeError("The last solve returned no dual values.")
        data = {}
        for name, con in self._parent.constraints.items():
            start, stop = self._matrices.con_blocks[name]
            # undo the row and objective scaling the solver applied
            values = (
                self._dual[start:stop]
                * self._matrices.con_scaling[start:stop]
                / self._matrices.objective_scaling
            )
            data[name] = self._reshape(
                con,
                values,
                self._matrices.con_active[name],
                dims=tuple(con.coord_dims),
            )
        return Dataset(data)

    def __getitem__(self, name: str) -> Any:
        """A variable of the window, materialised on demand."""
        if name in self._parent.variables:
            var = self._parent.variables[name]
            start, stop = self._matrices.var_blocks[name]
            values = None
            if self._primal is not None:
                values = (
                    self._primal[start:stop] / self._matrices.var_scaling[start:stop]
                )
            return self._reshape(var, values, self._matrices.var_active[name])
        raise KeyError(name)

    def _reshape(
        self,
        container: Any,
        values: np.ndarray | None,
        active: np.ndarray,
        dims: tuple[Hashable, ...] | None = None,
    ) -> DataArray:
        """Values of one container, scattered to its shape and labelled for the window."""
        dims = tuple(container.dims) if dims is None else dims
        sizes = {dim: container.shape[dims.index(dim)] for dim in dims}
        coords = {}
        for dim in dims:
            if dim != self._dim:
                coords[dim] = container.indexes[dim]
                continue
            # a carrier may hold a subset of the window's labels, so size and
            # coordinates come from the intersection
            index = container.indexes.get(dim)
            window = self._matrices.coords[self._dim]
            coords[dim] = window if index is None else index[index.isin(window)]
            sizes[dim] = coords[dim].size
        full = np.full(active.size, np.nan)
        if values is not None:
            full[active] = values
        return DataArray(
            full.reshape(tuple(sizes[dim] for dim in dims)),
            coords=coords,
            dims=list(dims),
        )

    def _require_solve(self) -> None:
        if self._primal is None:
            raise AttributeError(
                "The window has not been solved. Call `solve()` before reading values."
            )
