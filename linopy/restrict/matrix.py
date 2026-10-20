"""
Window a model's flattened matrices.

A rolling-horizon window is a contiguous run of one coordinate dimension, which
on the flattened matrices is a row slice of the parent CSR plus a column range.
:class:`MatrixWindow` is built once from a parent model, and
:meth:`MatrixWindow.select` returns the window's ``A``, ``b``, ``sense``, ``c``,
``lb`` and ``ub`` together with the parent labels of its rows and columns,
without rebuilding a single container.

Restricting and advancing are the same operation here: a selection. There is no
per-container term pattern to freeze, so a window never rebuilds on a
data-dependent coefficient change.

Limitations, all raised rather than silently approximated:

* one selected dimension,
* every carrier of that dimension must share its index,
* no indicator constraints, quadratic objective, SOS or piecewise formulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.sparse

from linopy.restrict.errors import RestrictionError

if TYPE_CHECKING:
    from linopy.model import Model

__all__ = ["MatrixWindow", "WindowMatrices"]


@dataclass(frozen=True)
class WindowMatrices:
    """
    One window of a parent model, as flat arrays.

    Columns follow the parent variable labels in ``vlabels``; rows follow the
    parent constraint labels in ``clabels``. Out-of-window objective terms are
    dropped, matching :meth:`Model.restrict`, so the objective value differs
    from the parent's by a constant.
    """

    A: scipy.sparse.csr_array
    b: np.ndarray
    sense: np.ndarray
    c: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    vlabels: np.ndarray
    clabels: np.ndarray
    objective_sense: str
    #: Per-container ``(start, stop)`` in the window's flat column and row space,
    #: so a solution vector can be handed back container by container.
    var_blocks: dict[str, tuple[int, int]]
    con_blocks: dict[str, tuple[int, int]]
    #: Coordinate values of the selected dimension in this window.
    coords: dict[str, pd.Index]
    #: Per-container active masks, in the window's grid order, so a flat result
    #: vector can be scattered back into each container's shape.
    var_active: dict[str, np.ndarray]
    con_active: dict[str, np.ndarray]
    #: Variable kinds (``C``, ``B``, ``I``, ``S``) and the scaling applied to
    #: columns and rows. Results must be unscaled with them before reporting.
    vtypes: np.ndarray
    var_scaling: np.ndarray
    con_scaling: np.ndarray
    objective_scaling: float

    @property
    def Q(self) -> None:
        """No quadratic objective: matrix windows are linear."""
        return None

    @property
    def indicator_A(self) -> None:
        """No indicator constraints: matrix windows reject them."""
        return None


@dataclass(frozen=True)
class WindowResult:
    """Flat result of solving one window."""

    objective: float
    primal: np.ndarray
    dual: np.ndarray


@dataclass(frozen=True)
class _Layout:
    """One container's opening position in the flat space, and its active mask."""

    start: int
    active: np.ndarray
    index: pd.Index | None
    labels: Any
    dim: str
    rank: np.ndarray

    @property
    def size(self) -> int:
        return int(self.active.sum())

    def select(self, window: pd.Index) -> tuple[np.ndarray, np.ndarray]:
        """
        Flat positions of the window's entries, and the window's active mask.

        The mask is laid out in the window's own grid order, so a flat result
        vector can be scattered back into the container's shape.
        """
        if not self.size:
            return np.empty(0, dtype=np.int64), np.zeros(0, dtype=bool)
        if self.index is None:
            # the container does not carry the selected dimension: kept whole
            return self.start + np.arange(self.size), np.asarray(self.active)
        lookup = self.index.get_indexer(window)
        if (lookup < 0).any():
            missing = window[lookup < 0]
            raise RestrictionError(
                f"Cannot window the matrices: labels {missing.tolist()} are not "
                "carried by every container of the selected dimension."
            )
        # Membership from the positional lookup rather than `xarray.isin`, which
        # compares tz-aware timestamps as objects and is slow.
        present = np.zeros(self.index.size, dtype=bool)
        present[lookup] = True
        shape = [1] * self.labels.ndim
        shape[self.labels.dims.index(self.dim)] = present.size
        flat = np.broadcast_to(present.reshape(shape), self.labels.shape).ravel()
        window_entries = np.flatnonzero(flat)
        window_active = np.asarray(self.active)[window_entries]
        return self.start + self.rank[window_entries[window_active]], window_active


def _layouts(containers: Any, dim: str) -> dict[str, _Layout]:
    """The flat layout of every container along ``dim``."""
    layouts: dict[str, _Layout] = {}
    position = 0
    for name, item in containers.items():
        labels = item.labels
        active = np.asarray(labels.values).ravel() != -1
        layouts[name] = _Layout(
            position, active, item.indexes.get(dim), labels, dim, np.cumsum(active) - 1
        )
        position += int(active.sum())
    return layouts


def _blocks(
    positions: dict[str, np.ndarray], keep: np.ndarray | None = None
) -> dict[str, tuple[int, int]]:
    """
    Per-container ``(start, stop)`` in the concatenated flat space.

    ``keep`` masks the concatenated rows; the returned blocks index the rows that
    survive it, which is the order of the window arrays.
    """
    blocks: dict[str, tuple[int, int]] = {}
    cursor = 0
    size = 0
    for name, values in positions.items():
        count = int(values.size)
        selected = count if keep is None else int(keep[cursor : cursor + count].sum())
        blocks[name] = (size, size + selected)
        size += selected
        cursor += count
    return blocks


def _row_owners(
    rows: np.ndarray, selection: dict[str, tuple[np.ndarray, np.ndarray]]
) -> list[str]:
    """Names of the containers owning the given concatenated row positions."""
    owners = []
    cursor = 0
    for name, (positions, _) in selection.items():
        count = int(positions.size)
        if np.any((rows >= cursor) & (rows < cursor + count)):
            owners.append(name)
        cursor += count
    return owners


def _filter_masks(
    selection: dict[str, tuple[np.ndarray, np.ndarray]], keep: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-container active masks, with the dropped rows removed."""
    masks: dict[str, np.ndarray] = {}
    cursor = 0
    for name, (positions, mask) in selection.items():
        count = int(positions.size)
        # `keep` is indexed over active rows, one per entry of `positions`; the
        # mask covers the whole window grid, so only its active slots move.
        filtered = mask.copy()
        filtered[np.flatnonzero(mask)] = keep[cursor : cursor + count]
        masks[name] = filtered
        cursor += count
    return masks


class MatrixWindow:
    """
    A matrix-level view of a model, windowed along one dimension.

    Build once, then call :meth:`select` per window. The parent model is read
    only; every selection allocates its own arrays.
    """

    def __init__(self, model: Model, dim: str) -> None:
        if model.variables.sos:
            # Piecewise formulations and reformulated SOS are ordinary variables and
            # constraints, so they window like anything else. Native SOS sets are not
            # in the matrices, so they must be reformulated first.
            raise NotImplementedError(
                "Matrix windows do not support unreformulated SOS constraints: "
                "call `model.apply_sos_reformulation()` first."
            )
        matrices = model.matrices
        if matrices.Q is not None:
            raise NotImplementedError(
                "Matrix windows do not support quadratic objectives."
            )
        if matrices.indicator_A is not None and matrices.indicator_A.nnz:
            raise NotImplementedError(
                "Matrix windows do not support indicator constraints."
            )
        if matrices.A is None:
            raise NotImplementedError("Matrix windows require at least one constraint.")
        self._model = model
        self._matrices = matrices
        self._A = matrices.A.tocsr()
        self._dim = dim
        self._var_layout = _layouts(model.variables, dim)
        self._con_layout = _layouts(model.constraints, dim)

    def select(
        self, window: pd.Index, fix: dict[str, Any] | None = None
    ) -> WindowMatrices:
        """
        The window's matrices.

        ``fix`` maps variable names to values carrying dangling boundary terms,
        exactly like :meth:`Model.restrict`. Out-of-window terms are folded into
        the right-hand side at their actual coefficient; a row that loses every
        term is dropped when its folded right-hand side satisfies its sign.
        """
        con_selection = {
            name: layout.select(window) for name, layout in self._con_layout.items()
        }
        var_selection = {
            name: layout.select(window) for name, layout in self._var_layout.items()
        }
        rows = np.concatenate([positions for positions, _ in con_selection.values()])
        cols = np.concatenate([positions for positions, _ in var_selection.values()])
        fix_by_position = self._fix_by_position(fix)

        A = self._A[rows]
        n_rows = A.shape[0]
        in_window = np.zeros(A.shape[1], dtype=bool)
        in_window[cols] = True

        row_of_entry = np.repeat(np.arange(n_rows), np.diff(A.indptr))
        dangling = ~in_window[A.indices] & (A.data != 0)
        unfixed = dangling & np.isnan(fix_by_position[A.indices])
        if unfixed.any():
            owners = _row_owners(np.unique(row_of_entry[unfixed]), con_selection)
            raise RestrictionError(
                f"Cannot restrict model: constraint(s) {owners} have dangling "
                "terms on variables outside the selection without fixed values. "
                "Pass their values via `fix`."
            )
        contributions = A.data[dangling] * fix_by_position[A.indices[dangling]]
        rhs = self._matrices.b[rows] - np.bincount(
            row_of_entry[dangling], weights=contributions, minlength=n_rows
        )

        kept_entries = in_window[A.indices]
        keep = np.bincount(row_of_entry[kept_entries], minlength=n_rows) > 0
        self._check_dropped_rows(rhs, row_of_entry[dangling], contributions, keep)

        sense = self._matrices.sense[rows]
        return WindowMatrices(
            A=A[keep][:, cols],
            b=rhs[keep],
            sense=sense[keep],
            c=self._matrices.c[cols],
            lb=self._matrices.lb[cols],
            ub=self._matrices.ub[cols],
            vlabels=self._matrices.vlabels[cols],
            clabels=self._matrices.clabels[rows][keep],
            objective_sense=self._model.objective.sense,
            var_blocks=_blocks({name: p for name, (p, _) in var_selection.items()}),
            con_blocks=_blocks(
                {name: p for name, (p, _) in con_selection.items()}, keep
            ),
            coords={self._dim: window},
            var_active={name: mask for name, (_, mask) in var_selection.items()},
            con_active=_filter_masks(con_selection, keep),
            vtypes=self._matrices.vtypes[cols],
            var_scaling=self._matrices.var_scaling[cols],
            con_scaling=self._matrices.con_scaling[rows][keep],
            objective_scaling=float(self._model.objective.scaling),
        )

    def _fix_by_position(self, fix: dict[str, Any] | None) -> np.ndarray:
        """Fixed values on the flat parent column space, NaN where unfixed."""
        n_cols = len(self._matrices.vlabels)
        values = np.full(n_cols, np.nan)
        if not fix:
            return values
        for name, given in fix.items():
            if name not in self._var_layout:
                raise RestrictionError(
                    f"Cannot fix {name!r}: it does not name a variable of the model."
                )
            layout = self._var_layout[name]
            flat = np.asarray(given).ravel()
            if flat.size != layout.active.size:
                raise RestrictionError(
                    f"Fixed values for variable '{name}' do not match its "
                    f"dimensions: got {flat.size} values, expected {layout.active.size}."
                )
            values[layout.start : layout.start + layout.size] = flat[layout.active]
        return values

    def _check_dropped_rows(
        self,
        rhs: np.ndarray,
        dangling_rows: np.ndarray,
        contributions: np.ndarray,
        keep: np.ndarray,
    ) -> None:
        """Raise when a row that lost every term violates its sign."""
        dropped = np.flatnonzero(~keep)
        if not dropped.size:
            return
        magnitude = np.bincount(
            dangling_rows, weights=np.abs(contributions), minlength=len(keep)
        )
        tolerance = 1e-9 * np.maximum(1.0, magnitude[dropped])
        residual = rhs[dropped]
        sense = self._matrices.sense
        signs = (
            np.full(dropped.size, sense) if isinstance(sense, str) else sense[dropped]
        )
        violated = (
            ((signs == "=") & (np.abs(residual) > tolerance))
            | ((signs == "<=") & (residual < -tolerance))
            | ((signs == ">=") & (residual > tolerance))
        )
        if violated.any():
            raise RestrictionError(
                "Cannot window the matrices: folding the dangling terms makes "
                f"rows {dropped[violated].tolist()} unsatisfiable "
                f"(folded rhs {residual[violated].tolist()} against sign "
                f"{signs[violated].tolist()})."
            )
