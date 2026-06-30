"""
Piecewise linear constraint formulations.

Provides SOS2, incremental, pure LP, and disjunctive piecewise linear
constraint methods for use with linopy.Model.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Literal, TypeAlias, TypeGuard

import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray

from linopy.constants import (
    BREAKPOINT_DIM,
    EQUAL,
    GREATER_EQUAL,
    HELPER_DIMS,
    LESS_EQUAL,
    LP_PIECE_DIM,
    PWL_ACTIVE_BOUND_SUFFIX,
    PWL_BINARY_ORDER_SUFFIX,
    PWL_CHORD_SUFFIX,
    PWL_CONVEX_SUFFIX,
    PWL_CONVEXITY,
    PWL_DELTA_BOUND_SUFFIX,
    PWL_DELTA_SUFFIX,
    PWL_DOMAIN_HI_SUFFIX,
    PWL_DOMAIN_LO_SUFFIX,
    PWL_FILL_ORDER_SUFFIX,
    PWL_LAMBDA_SUFFIX,
    PWL_LINK_DIM,
    PWL_LINK_SUFFIX,
    PWL_METHOD,
    PWL_METHODS,
    PWL_ORDER_BINARY_SUFFIX,
    PWL_OUTPUT_LINK_SUFFIX,
    PWL_SEGMENT_BINARY_SUFFIX,
    PWL_SELECT_SUFFIX,
    SEGMENT_DIM,
    SIGNS,
    EvolvingAPIWarning,
    sign_replace_dict,
)

if TYPE_CHECKING:
    from linopy.constraints import Constraint, Constraints
    from linopy.expressions import LinearExpression
    from linopy.model import Model
    from linopy.types import LinExprLike
    from linopy.variables import Variables

logger = logging.getLogger(__name__)

# Each user-facing piecewise entry point fires its EvolvingAPIWarning at
# most once per process.  Without dedup, a single model build emits the
# verbose warning hundreds of times and drowns out other output.
_EvolvingApiKey: TypeAlias = Literal[
    "tangent_lines", "add_piecewise_formulation", "Slopes"
]
_emitted_evolving_warnings: set[_EvolvingApiKey] = set()


def _warn_evolving_api(key: _EvolvingApiKey, message: str, stacklevel: int = 3) -> None:
    """
    Emit an :class:`EvolvingAPIWarning` at most once per session per ``key``.

    ``stacklevel`` defaults to 3 (helper → entry-point function → user
    code).  Pass a larger value when called from one frame deeper than
    a function — e.g. from a dataclass ``__post_init__``, which is
    itself invoked by an auto-generated ``__init__``.
    """
    if key in _emitted_evolving_warnings:
        return
    _emitted_evolving_warnings.add(key)
    warnings.warn(message, category=EvolvingAPIWarning, stacklevel=stacklevel)


# Accepted input types for breakpoint-like data
BreaksLike: TypeAlias = (
    Sequence[float]
    | np.ndarray
    | DataArray
    | pd.Series
    | pd.DataFrame
    | dict[str, Sequence[float]]
)

# Accepted input types for segment-like data (2D: segments × breakpoints)
SegmentsLike: TypeAlias = (
    Sequence[Sequence[float]]
    | np.ndarray
    | DataArray
    | pd.DataFrame
    | dict[str, Sequence[Sequence[float]]]
)


# ---------------------------------------------------------------------------
# Deferred slopes spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class Slopes:
    """
    Per-piece slopes + initial y-value, deferred until an x grid is known.

    Used as the second element of a tuple in
    :func:`add_piecewise_formulation`.  When any :class:`Slopes` tuple is
    present, **exactly one** other tuple must carry explicit breakpoints —
    that tuple's values are the x grid against which all :class:`Slopes`
    are integrated::

        m.add_piecewise_formulation(
            (power, [0, 30, 60, 100]),               # the x grid
            (fuel,  Slopes([1.2, 1.4, 1.7], y0=0)),  # integrated against power
        )

    With two or more non-:class:`Slopes` tuples there is no canonical x
    axis, and the call raises :class:`ValueError`.  Resolve the
    :class:`Slopes` explicitly via :meth:`to_breakpoints` in that case,
    or for any standalone use::

        bp = Slopes([1.2, 1.4, 1.7], y0=0).to_breakpoints([0, 30, 60, 100])

    Parameters
    ----------
    values : BreaksLike
        Per-piece slopes.  1D for shared breakpoints; 2D (DataFrame /
        dict / DataArray with entity dim) for per-entity slopes.
    y0 : float, dict, pd.Series, or DataArray, default 0.0
        y-value at the first breakpoint.  Scalar broadcasts to all
        entities; dict/Series/DataArray provides per-entity values.
    align : {"pieces", "leading"}, default "pieces"
        Alignment of ``values`` relative to the x grid.

        - ``"pieces"``: ``len(values) == len(x_points) - 1``;
          ``values[i]`` is the slope between ``x[i]`` and ``x[i+1]``.
        - ``"leading"``: ``len(values) == len(x_points)``; ``values[0]``
          must be NaN and is dropped, ``values[i]`` for ``i>=1`` is the
          slope between ``x[i-1]`` and ``x[i]``.  Useful when a marginal
          value is tabulated alongside each breakpoint with the first
          row's marginal undefined.
    dim : str, optional
        Entity dimension name.  Required when ``values`` is a
        ``pd.DataFrame`` or ``dict``.

    Warns
    -----
    EvolvingAPIWarning
        :class:`Slopes` is part of the newly-added piecewise API.  Its
        constructor signature and dispatch semantics may be refined.
        Silence with ``warnings.filterwarnings("ignore",
        category=linopy.EvolvingAPIWarning)``.
    """

    values: BreaksLike
    y0: Real | dict[str, Real] | pd.Series | DataArray = 0.0
    align: Literal["pieces", "leading"] = "pieces"
    dim: str | None = None

    def __post_init__(self) -> None:
        # ``stacklevel=4``: warn → _warn_evolving_api → __post_init__ →
        # dataclass-generated ``__init__`` → user code.
        _warn_evolving_api(
            "Slopes",
            "piecewise: Slopes is a new API; the constructor signature and "
            "the dispatch rules for inheriting an x grid from sibling tuples "
            "may be refined in minor releases.",
            stacklevel=4,
        )

    def to_breakpoints(self, x_points: BreaksLike) -> DataArray:
        """
        Resolve to a breakpoint :class:`xarray.DataArray`, given an x grid.

        Rarely called directly — typically you pass the :class:`Slopes`
        instance to :func:`add_piecewise_formulation` and the x grid is
        inherited from a sibling tuple.  Use this method for inspection
        or when building breakpoints outside the formulation pipeline.
        """
        return _breakpoints_from_slopes(
            self.values, x_points, self.y0, self.dim, self.align
        )

    def __repr__(self) -> str:
        bits = [_summarise_breakslike(self.values), f"y0={self.y0!r}"]
        if self.align != "pieces":
            bits.append(f"align={self.align!r}")
        if self.dim is not None:
            bits.append(f"dim={self.dim!r}")
        return f"Slopes({', '.join(bits)})"

    def __eq__(self, other: object) -> bool:
        """
        Value-equality across the field types accepted by the constructor.

        Two ``Slopes`` are equal iff every field matches:

        * ``align`` and ``dim`` compare with ``==`` (str / None).
        * ``y0`` and ``values`` dispatch on type via :func:`_values_equal`:
          numeric scalars compare by value across types (``int 0 ==
          float 0.0 == np.float64(0)``); ``list`` and ``tuple`` are
          promoted to ndarray so NaN content compares element-wise
          regardless of which NaN object was used; ndarrays use
          ``np.array_equal(equal_nan=True)`` (with a fallback for
          non-numeric dtypes); ``pd.Series`` / ``pd.DataFrame`` /
          ``DataArray`` use ``.equals``; ``dict`` recurses on matching
          keys.

        Non-``Slopes`` operands return ``NotImplemented`` per Python
        convention.

        Caveats
        -------
        * ``Series.equals`` / ``DataFrame.equals`` / ``DataArray.equals``
          are *order-sensitive*: two frames with the same content but
          reordered rows / columns / coords compare unequal.
        * Cross-container coercion is limited to ``list``/``tuple`` →
          ndarray.  A ``dict`` and a ``DataFrame`` describing the same
          per-entity slopes still compare unequal.

        ``__hash__`` is set to ``None`` (unhashable) since the inner
        ``values`` may be a mutable container.
        """
        if not isinstance(other, Slopes):
            return NotImplemented
        return (
            self.align == other.align
            and self.dim == other.dim
            and _values_equal(self.y0, other.y0)
            and _values_equal(self.values, other.values)
        )

    __hash__ = None  # type: ignore[assignment]


def _is_numeric_scalar(x: object) -> TypeGuard[Real]:
    return isinstance(x, Real) and not isinstance(x, bool)


def _values_equal(a: object, b: object) -> bool:
    """
    Type-dispatched equality for ``Slopes`` field values (NaN-safe).

    Numeric scalars compare by value across types (``int 0 == float 0.0 ==
    np.float64(0)``); ``bool`` is excluded.  Lists / tuples are promoted
    to ndarray so in-place ``float('nan')`` content compares NaN-safe.
    Non-numeric ndarray dtypes fall back to ``np.array_equal`` without
    ``equal_nan``.  ``DataFrame`` / ``Series`` / ``DataArray`` use
    ``.equals``; ``dict`` recurses on matching keys.
    """
    if _is_numeric_scalar(a) and _is_numeric_scalar(b):
        af, bf = float(a), float(b)
        return af == bf or (af != af and bf != bf)

    if isinstance(a, list | tuple):
        a = np.asarray(a)
    if isinstance(b, list | tuple):
        b = np.asarray(b)

    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.shape != b.shape:
            return False
        try:
            return bool(np.array_equal(a, b, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(a, b))

    if isinstance(a, pd.DataFrame):
        return isinstance(b, pd.DataFrame) and bool(a.equals(b))
    if isinstance(a, pd.Series):
        return isinstance(b, pd.Series) and bool(a.equals(b))
    if isinstance(a, DataArray):
        return isinstance(b, DataArray) and bool(a.equals(b))

    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(_values_equal(a[k], b[k]) for k in a)
        )

    return type(a) is type(b) and bool(a == b)


def _summarise_breakslike(v: BreaksLike) -> str:
    """Compact one-line summary of a BreaksLike value for use in reprs."""
    if isinstance(v, DataArray):
        sizes = ", ".join(f"{d}: {s}" for d, s in v.sizes.items())
        return f"<DataArray {sizes}>"
    if isinstance(v, pd.DataFrame):
        return f"<DataFrame shape={v.shape}>"
    if isinstance(v, pd.Series):
        return f"<Series len={len(v)}>"
    if isinstance(v, dict):
        return f"<dict {len(v)} entries>"

    arr = np.asarray(v)
    if arr.ndim > 1:
        return f"<ndarray shape={arr.shape}>"
    seq: list = arr.tolist()
    if len(seq) <= 8:
        return "[" + ", ".join(_short_num(x) for x in seq) + "]"
    head = ", ".join(_short_num(x) for x in seq[:3])
    tail = ", ".join(_short_num(x) for x in seq[-2:])
    return f"[{head}, ..., {tail}] ({len(seq)} items)"


def _short_num(x: object) -> str:
    """Compact number formatting for repr — ``g`` for floats, ``repr`` else."""
    if isinstance(x, float):
        return f"{x:g}"
    return repr(x)


# Tuple element type covering both eager (DataArray etc.) and deferred (Slopes) bps.
BreaksOrSlopes: TypeAlias = BreaksLike | Slopes


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(slots=True, repr=False)
class PiecewiseFormulation:
    """
    Result of ``add_piecewise_formulation``.

    Groups all auxiliary variables and constraints created by a single
    piecewise formulation. Stores only names internally; ``variables``
    and ``constraints`` properties return live views from the model.

    Attributes
    ----------
    name : str
        Formulation name (used as prefix for auxiliary variables and
        constraints).
    method : PWL_METHOD
        Resolved method actually used. Never ``"auto"``; if the caller
        passed ``method="auto"``, this holds the method that was chosen.
    convexity : PWL_CONVEXITY or None
        Shape of the piecewise curve along the breakpoint axis when it is
        well-defined (exactly two expressions, non-disjunctive, strictly
        monotonic ``x`` breakpoints).  ``None`` otherwise.
    """

    name: str
    method: PWL_METHOD
    """Resolved formulation method (see :data:`~linopy.constants.PWL_METHOD`)."""
    variable_names: list[str]
    constraint_names: list[str]
    model: Model
    convexity: PWL_CONVEXITY | None = None
    """Shape of the piecewise curve when well-defined (see :data:`~linopy.constants.PWL_CONVEXITY`), else ``None``."""

    @property
    def variables(self) -> Variables:
        """View of the auxiliary variables in this formulation."""
        return self.model.variables[self.variable_names]

    @property
    def constraints(self) -> Constraints:
        """View of the auxiliary constraints in this formulation."""
        return self.model.constraints[self.constraint_names]

    def _user_dims_with_sizes(self) -> dict[str, int]:
        """
        User-facing dims across the formulation's variables, with sizes.

        Skips internal ``_``-prefixed dims (e.g. ``_pwl_var``).  Insertion
        order is preserved, so callers can use the keys as a stable
        ordered list.
        """
        dims: dict[str, int] = {}
        for var in self.variables.data.values():
            for d in var.coords:
                ds = str(d)
                if not ds.startswith("_") and ds not in dims:
                    dims[ds] = var.data.sizes[d]
        return dims

    def _user_dims(self) -> list[str]:
        """User-facing dim names across this formulation's auxiliary variables."""
        return list(self._user_dims_with_sizes())

    def __repr__(self) -> str:
        user_dims = self._user_dims_with_sizes()
        dims_str = ", ".join(f"{d}: {s}" for d, s in user_dims.items())
        header = f"PiecewiseFormulation `{self.name}`"
        if dims_str:
            header += f" [{dims_str}]"
        suffix: str = self.method
        if self.convexity is not None:
            suffix += f", {self.convexity}"
        r = f"{header} — {suffix}\n"
        r += "  Variables:\n"
        for vname, var in self.variables.items():
            dims = ", ".join(str(d) for d in var.coords) if var.coords else ""
            r += f"    * {vname} ({dims})\n" if dims else f"    * {vname}\n"
        r += "  Constraints:\n"
        for cname, con in self.constraints.items():
            dims = ", ".join(str(d) for d in con.coords) if con.coords else ""
            r += f"    * {cname} ({dims})\n" if dims else f"    * {cname}\n"
        return r


def _get_piecewise_groups(model: Model) -> tuple[set[str], set[str]]:
    """
    Names of auxiliary variables/constraints that belong to a piecewise
    formulation.  Returned as separate sets because variables and
    constraints live in independent namespaces in the model.
    """
    var_names: set[str] = set()
    con_names: set[str] = set()
    for pwl in model._piecewise_formulations.values():
        var_names.update(pwl.variable_names)
        con_names.update(pwl.constraint_names)
    return var_names, con_names


def _repr_summary(model: Model) -> str:
    """
    Render the model-level summary of all piecewise formulations.

    Returns the empty string when the model has no formulations so the
    caller can unconditionally concatenate.
    """
    if not model._piecewise_formulations:
        return ""
    r = "\nPiecewise Formulations:\n----------------------\n"
    for pwl in model._piecewise_formulations.values():
        n_vars = len(pwl.variable_names)
        n_cons = len(pwl.constraint_names)
        user_dims = pwl._user_dims()
        dims_str = f" ({', '.join(user_dims)})" if user_dims else ""
        r += f" * {pwl.name}{dims_str} — {pwl.method}, {n_vars} vars, {n_cons} cons\n"
    return r


# ---------------------------------------------------------------------------
# DataArray construction helpers
# ---------------------------------------------------------------------------


def _strip_nan(vals: Sequence[float] | np.ndarray) -> list[float]:
    """Remove NaN values from a sequence."""
    arr = np.asarray(vals, dtype=float)
    return arr[~np.isnan(arr)].tolist()


def _rename_to_pieces(da: DataArray, piece_index: np.ndarray) -> DataArray:
    """Rename breakpoint dim to piece dim and reassign coordinates."""
    da = da.rename({BREAKPOINT_DIM: LP_PIECE_DIM})
    da[LP_PIECE_DIM] = piece_index
    return da


def _sequence_to_array(values: Sequence[float] | np.ndarray | pd.Series) -> DataArray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(
            f"Expected a 1D sequence of numeric values, got shape {arr.shape}"
        )
    return DataArray(
        arr, dims=[BREAKPOINT_DIM], coords={BREAKPOINT_DIM: np.arange(len(arr))}
    )


def _dict_to_array(d: dict[str, Sequence[float]], dim: str) -> DataArray:
    """Convert a dict of ragged sequences to a NaN-padded 2D DataArray."""
    max_len = max(len(v) for v in d.values())
    keys = list(d.keys())
    data = np.full((len(keys), max_len), np.nan)
    for i, k in enumerate(keys):
        vals = d[k]
        data[i, : len(vals)] = vals
    return DataArray(
        data,
        dims=[dim, BREAKPOINT_DIM],
        coords={dim: keys, BREAKPOINT_DIM: np.arange(max_len)},
    )


def _dataframe_to_array(df: pd.DataFrame, dim: str) -> DataArray:
    # rows = entities (index), columns = breakpoints
    data = np.asarray(df.values, dtype=float)
    return DataArray(
        data,
        dims=[dim, BREAKPOINT_DIM],
        coords={dim: list(df.index), BREAKPOINT_DIM: np.arange(df.shape[1])},
    )


def _coerce_breaks(values: BreaksLike, dim: str | None = None) -> DataArray:
    """Convert any BreaksLike input to a DataArray with BREAKPOINT_DIM."""
    if isinstance(values, DataArray):
        if BREAKPOINT_DIM not in values.dims:
            raise ValueError(
                f"DataArray must have a '{BREAKPOINT_DIM}' dimension, "
                f"got dims {list(values.dims)}"
            )
        return values
    if isinstance(values, pd.DataFrame):
        if dim is None:
            raise ValueError("'dim' is required when input is a DataFrame")
        return _dataframe_to_array(values, dim)
    if isinstance(values, pd.Series):
        return _sequence_to_array(values)
    if isinstance(values, dict):
        if dim is None:
            raise ValueError("'dim' is required when input is a dict")
        return _dict_to_array(values, dim)
    # Sequence (list, tuple, etc.)
    return _sequence_to_array(values)


def _segments_list_to_array(values: Sequence[Sequence[float]]) -> DataArray:
    max_len = max(len(seg) for seg in values)
    data = np.full((len(values), max_len), np.nan)
    for i, seg in enumerate(values):
        data[i, : len(seg)] = seg
    return DataArray(
        data,
        dims=[SEGMENT_DIM, BREAKPOINT_DIM],
        coords={
            SEGMENT_DIM: np.arange(len(values)),
            BREAKPOINT_DIM: np.arange(max_len),
        },
    )


def _dict_segments_to_array(
    d: dict[str, Sequence[Sequence[float]]], dim: str
) -> DataArray:
    parts = []
    for key, seg_list in d.items():
        arr = _segments_list_to_array(seg_list)
        parts.append(arr.expand_dims({dim: [key]}))
    combined = xr.concat(parts, dim=dim, coords="minimal")
    max_bp = max(max(len(seg) for seg in sl) for sl in d.values())
    max_seg = max(len(sl) for sl in d.values())
    if combined.sizes[BREAKPOINT_DIM] < max_bp or combined.sizes[SEGMENT_DIM] < max_seg:
        combined = combined.reindex(
            {BREAKPOINT_DIM: np.arange(max_bp), SEGMENT_DIM: np.arange(max_seg)},
            fill_value=np.nan,
        )
    return combined


def _breakpoints_from_slopes(
    slopes: BreaksLike,
    x_points: BreaksLike,
    y0: Real | dict[str, Real] | pd.Series | DataArray,
    dim: str | None,
    slopes_align: Literal["pieces", "leading"] = "pieces",
) -> DataArray:
    """Convert slopes + x_points + y0 into a breakpoint DataArray."""
    slopes_arr = _coerce_breaks(slopes, dim)
    xp_arr = _coerce_breaks(x_points, dim)

    if slopes_align == "leading":
        if slopes_arr.sizes[BREAKPOINT_DIM] == 0:
            raise ValueError("slopes_align='leading' requires at least one slope entry")
        first_slope = slopes_arr.isel({BREAKPOINT_DIM: 0})
        if not bool(first_slope.isnull().all()):
            raise ValueError(
                "slopes_align='leading' requires the first slope of each "
                "entity to be NaN"
            )
        slopes_arr = slopes_arr.isel({BREAKPOINT_DIM: slice(1, None)})

    # 1D case: single set of breakpoints
    if slopes_arr.ndim == 1:
        if not isinstance(y0, Real):
            raise TypeError("When 'slopes' is 1D, 'y0' must be a scalar float")
        pts = _slopes_to_points(list(xp_arr.values), list(slopes_arr.values), float(y0))
        return _sequence_to_array(pts)

    # Multi-dim case: per-entity slopes
    entity_dims = [d for d in slopes_arr.dims if d != BREAKPOINT_DIM]
    if len(entity_dims) != 1:
        raise ValueError(
            f"Expected exactly one entity dimension in slopes, got {entity_dims}"
        )
    entity_dim = str(entity_dims[0])
    entity_keys = slopes_arr.coords[entity_dim].values

    # Resolve y0 per entity
    if isinstance(y0, Real):
        y0_map: dict[str, float] = {str(k): float(y0) for k in entity_keys}
    elif isinstance(y0, dict):
        y0_map = {str(k): float(y0[k]) for k in entity_keys}
    elif isinstance(y0, pd.Series):
        y0_map = {str(k): float(y0[k]) for k in entity_keys}
    elif isinstance(y0, DataArray):
        y0_map = {str(k): float(y0.sel({entity_dim: k}).item()) for k in entity_keys}
    else:
        raise TypeError(
            f"'y0' must be a float, Series, DataArray, or dict, got {type(y0)}"
        )

    computed: dict[str, Sequence[float]] = {}
    for key in entity_keys:
        sk = str(key)
        sl = _strip_nan(slopes_arr.sel({entity_dim: key}).values)
        if entity_dim in xp_arr.dims:
            xp = _strip_nan(xp_arr.sel({entity_dim: key}).values)
        else:
            xp = _strip_nan(xp_arr.values)
        computed[sk] = _slopes_to_points(xp, sl, y0_map[sk])

    return _dict_to_array(computed, entity_dim)


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def _slopes_to_points(
    x_points: list[float], slopes: list[float], y0: float
) -> list[float]:
    """
    Convert per-piece slopes + initial y-value to y-coordinates at each breakpoint.

    Internal primitive used by ``Slopes.to_breakpoints``.  Public callers
    should use :class:`Slopes` (DataArray output) instead.
    """
    if len(slopes) != len(x_points) - 1:
        raise ValueError(
            f"len(slopes) must be len(x_points) - 1, "
            f"got {len(slopes)} slopes and {len(x_points)} x_points"
        )
    y_points: list[float] = [y0]
    for i, s in enumerate(slopes):
        y_points.append(y_points[-1] + s * (x_points[i + 1] - x_points[i]))
    return y_points


def breakpoints(
    values: BreaksLike,
    *,
    dim: str | None = None,
) -> DataArray:
    """
    Create a breakpoint DataArray for piecewise linear constraints.

    Parameters
    ----------
    values : BreaksLike
        Breakpoint values. Accepted types: ``Sequence[float]``,
        ``pd.Series``, ``pd.DataFrame``, or ``xr.DataArray``.
        A 1D input (list, Series) creates 1D breakpoints.
        A 2D input (DataFrame, multi-dim DataArray) creates per-entity
        breakpoints (``dim`` is required for DataFrame).
    dim : str, optional
        Entity dimension name. Required when ``values`` is a
        ``pd.DataFrame`` or ``dict``.

    Returns
    -------
    DataArray

    See Also
    --------
    Slopes : per-piece slopes + ``y0`` (deferred or standalone via
        :meth:`Slopes.to_breakpoints`).
    """
    return _coerce_breaks(values, dim)


def _coerce_segments(values: SegmentsLike, dim: str | None = None) -> DataArray:
    """Convert any SegmentsLike input to a DataArray with SEGMENT_DIM and BREAKPOINT_DIM."""
    if isinstance(values, DataArray):
        if SEGMENT_DIM not in values.dims or BREAKPOINT_DIM not in values.dims:
            raise ValueError(
                f"DataArray must have both '{SEGMENT_DIM}' and '{BREAKPOINT_DIM}' "
                f"dimensions, got dims {list(values.dims)}"
            )
        return values
    if isinstance(values, pd.DataFrame):
        data = np.asarray(values.values, dtype=float)
        return DataArray(
            data,
            dims=[SEGMENT_DIM, BREAKPOINT_DIM],
            coords={
                SEGMENT_DIM: np.arange(data.shape[0]),
                BREAKPOINT_DIM: np.arange(data.shape[1]),
            },
        )
    if isinstance(values, dict):
        if dim is None:
            raise ValueError("'dim' is required when 'values' is a dict")
        return _dict_segments_to_array(values, dim)
    # Sequence[Sequence[float]]
    return _segments_list_to_array(list(values))


def segments(
    values: SegmentsLike,
    *,
    dim: str | None = None,
) -> DataArray:
    """
    Create a segmented breakpoint DataArray for disjunctive piecewise constraints.

    Parameters
    ----------
    values : SegmentsLike
        Segment breakpoints. Accepted types: ``Sequence[Sequence[float]]``,
        ``pd.DataFrame`` (rows=segments, columns=breakpoints),
        ``xr.DataArray`` (must have ``SEGMENT_DIM`` and ``BREAKPOINT_DIM``),
        or ``dict[str, Sequence[Sequence[float]]]`` (requires ``dim``).
    dim : str, optional
        Entity dimension name. Required when ``values`` is a dict.

    Returns
    -------
    DataArray
    """
    return _coerce_segments(values, dim)


def _tangent_lines_impl(
    x: LinExprLike,
    x_points: BreaksLike,
    y_points: BreaksLike,
) -> LinearExpression:
    """
    Chord-expression math — the body of ``tangent_lines`` without the
    :class:`EvolvingAPIWarning`.  Called internally by ``_add_lp`` so a
    single ``add_piecewise_formulation((y, y_pts, "<="), (x, x_pts))``
    emits exactly one warning, not two.
    """
    from linopy.expressions import LinearExpression as LinExpr
    from linopy.variables import Variable

    x_points = _coerce_breaks(x_points)
    y_points = _coerce_breaks(y_points)

    dx = x_points.diff(BREAKPOINT_DIM)
    dy = y_points.diff(BREAKPOINT_DIM)
    piece_index = np.arange(dx.sizes[BREAKPOINT_DIM])

    slopes = _rename_to_pieces(dy / dx, piece_index)
    x_base = _rename_to_pieces(
        x_points.isel({BREAKPOINT_DIM: slice(None, -1)}), piece_index
    )
    y_base = _rename_to_pieces(
        y_points.isel({BREAKPOINT_DIM: slice(None, -1)}), piece_index
    )

    intercepts = y_base - slopes * x_base

    if not isinstance(x, Variable | LinExpr):
        raise TypeError(f"x must be a Variable or LinearExpression, got {type(x)}")

    return slopes * _to_linexpr(x) + intercepts


def tangent_lines(
    x: LinExprLike,
    x_points: BreaksLike,
    y_points: BreaksLike,
) -> LinearExpression:
    r"""
    Compute tangent-line (chord) expressions for a piecewise linear function.

    Low-level helper returning a :class:`~linopy.expressions.LinearExpression`
    with an extra piece dimension.  Each element along the piece dimension
    is the chord of one piece: :math:`m_k \cdot x + c_k`.  No auxiliary
    variables are created.

    For most users: prefer :func:`add_piecewise_formulation` with a
    bounded tuple ``(y, y_pts, "<=")`` / ``(y, y_pts, ">=")`` — it builds
    on this helper and adds the ``x ∈ [x_min, x_max]`` domain bound plus
    a curvature-vs-sign check that catches the "wrong region" case.  Use
    ``tangent_lines`` directly only when you need to compose the chord
    expressions manually (e.g. with other linear terms, or without the
    domain bound).

    .. code-block:: python

        t = tangent_lines(power, x_pts, y_pts)
        m.add_constraints(fuel <= t)  # upper bound (concave f)
        m.add_constraints(fuel >= t)  # lower bound (convex f)

    Parameters
    ----------
    x : Variable or LinearExpression
        The input expression.
    x_points : BreaksLike
        Breakpoint x-coordinates (must be strictly monotonic; both
        ascending and descending are accepted).
    y_points : BreaksLike
        Breakpoint y-coordinates.

    Returns
    -------
    LinearExpression
        Expression with an additional ``_breakpoint_piece`` dimension
        (one entry per piece).

    Warns
    -----
    EvolvingAPIWarning
        ``tangent_lines`` is part of the newly-added piecewise API; the
        returned expression shape and piece-dim name may be refined.
        Silence with ``warnings.filterwarnings("ignore",
        category=linopy.EvolvingAPIWarning)``.
    """
    _warn_evolving_api(
        "tangent_lines",
        "piecewise: tangent_lines is a new API; the returned expression "
        "shape and the piece-dim name may be refined in minor releases. "
        "Please share your use cases or concerns at "
        "https://github.com/PyPSA/linopy/issues — your feedback shapes "
        "what stabilises.  This warning fires once per session; silence "
        "entirely with "
        '`warnings.filterwarnings("ignore", category=linopy.EvolvingAPIWarning)`.',
    )
    return _tangent_lines_impl(x, x_points, y_points)


# ---------------------------------------------------------------------------
# Internal validation and utility functions
# ---------------------------------------------------------------------------


def _resolve_active(
    active: LinearExpression, reference: DataArray, active_fill: int | None
) -> LinearExpression:
    """
    Resolve a possibly-partial ``active`` gate against the formulation.

    A gate defined over only a subset of the indexed dimension (or with
    masked entries) would otherwise be gated as if ``active=0`` and forced
    to zero. With ``active_fill is None`` such a gate is rejected; otherwise
    the gaps are filled with ``active_fill`` (``1`` = always active, ``0`` =
    always off). Dimensions absent from ``active`` broadcast and are left
    untouched.
    """
    skip = {BREAKPOINT_DIM, SEGMENT_DIM} | set(HELPER_DIMS)
    indexers = {
        d: reference.indexes[d]
        for d in active.coord_dims
        if d in reference.indexes and d not in skip
    }
    aligned = active.reindex(indexers) if indexers else active

    if active_fill is not None:
        return aligned.where(aligned.has_terms, active_fill)

    term_dims = [d for d in aligned.vars.dims if d not in aligned.coord_dims]
    dangling = ((aligned.vars < 0) & aligned.coeffs.notnull()).any(term_dims)
    covered = aligned.has_terms | (aligned.const.notnull() & ~dangling)
    if not bool(covered.all()):
        raise ValueError(
            "`active` is not defined over the full coordinate of the "
            "piecewise formulation: it is missing labels (a subset of the "
            "coordinate) or has masked entries, which would be gated to "
            "zero. Pass `active_fill=1` to treat those entries as always "
            "active (or `0` as always off), or pass a fully-defined `active`."
        )
    return active


def _validate_breakpoint_shapes(bp_a: DataArray, bp_b: DataArray) -> bool:
    """
    Validate that two breakpoint arrays have compatible shapes.

    Returns whether the formulation is disjunctive (has segment dimension).
    """
    for bp in (bp_a, bp_b):
        if BREAKPOINT_DIM not in bp.dims:
            raise ValueError(
                f"Breakpoints are missing the '{BREAKPOINT_DIM}' dimension, "
                f"got dims {list(bp.dims)}. "
                "Use the breakpoints() or segments() factory."
            )

    if bp_a.sizes[BREAKPOINT_DIM] != bp_b.sizes[BREAKPOINT_DIM]:
        raise ValueError(
            f"Breakpoints must have same size along '{BREAKPOINT_DIM}', "
            f"got {bp_a.sizes[BREAKPOINT_DIM]} and "
            f"{bp_b.sizes[BREAKPOINT_DIM]}"
        )

    a_has_seg = SEGMENT_DIM in bp_a.dims
    b_has_seg = SEGMENT_DIM in bp_b.dims
    if a_has_seg != b_has_seg:
        raise ValueError(
            "If one breakpoint array has a segment dimension, "
            f"both must. Got dims: {list(bp_a.dims)} and {list(bp_b.dims)}."
        )
    if a_has_seg and bp_a.sizes[SEGMENT_DIM] != bp_b.sizes[SEGMENT_DIM]:
        raise ValueError(f"Breakpoints must have same size along '{SEGMENT_DIM}'")

    return a_has_seg


def _validate_numeric_breakpoint_coords(bp: DataArray) -> None:
    coord = bp.coords[BREAKPOINT_DIM]
    if not pd.api.types.is_numeric_dtype(coord):
        raise ValueError(
            f"Breakpoint dimension '{BREAKPOINT_DIM}' must have numeric coordinates "
            f"for SOS2 weights, but got {coord.dtype}"
        )
    values = np.asarray(coord.values)
    if len(values) > 1 and not bool(np.all(np.diff(values) > 0)):
        raise ValueError(
            f"Breakpoint dimension '{BREAKPOINT_DIM}' coordinates must be "
            "strictly increasing for SOS2 weights."
        )


def _check_strict_monotonicity(bp: DataArray) -> bool:
    """Check if breakpoints are strictly monotonic along BREAKPOINT_DIM (ignoring NaN)."""
    diffs = bp.diff(BREAKPOINT_DIM)
    pos = (diffs > 0) | diffs.isnull()
    neg = (diffs < 0) | diffs.isnull()
    all_pos_per_slice = pos.all(BREAKPOINT_DIM)
    all_neg_per_slice = neg.all(BREAKPOINT_DIM)
    has_non_nan = (~diffs.isnull()).any(BREAKPOINT_DIM)
    monotonic = (all_pos_per_slice | all_neg_per_slice) & has_non_nan
    return bool(monotonic.all())


def _detect_convexity(x_points: DataArray, y_points: DataArray) -> PWL_CONVEXITY:
    """
    Classify the shape of a single piecewise curve ``y = f(x)``.

    Invariant to whether breakpoints are listed ascending or descending in
    x — same graph, same label.  Multi-entity inputs are aggregated across
    entities; to classify per entity, iterate at the call site (see
    :data:`PWL_CONVEXITIES` for the possible labels).  Callers must
    enforce strict x-monotonicity per slice upstream.
    """
    dx = x_points.diff(BREAKPOINT_DIM)
    slopes = y_points.diff(BREAKPOINT_DIM) / dx
    # Flip sign when x descends so the classification matches the
    # ascending-x traversal.  All dx in a strictly-monotonic slice share
    # a sign, so the sum resolves direction per entity.
    sd = slopes.diff(BREAKPOINT_DIM) * np.sign(dx.sum(BREAKPOINT_DIM))

    if int((~sd.isnull()).sum()) == 0:
        return "linear"
    tol = 1e-10
    nonneg = bool(((sd >= -tol) | sd.isnull()).all())
    nonpos = bool(((sd <= tol) | sd.isnull()).all())
    if nonneg and nonpos:
        return "linear"
    if nonneg:
        return "convex"
    if nonpos:
        return "concave"
    return "mixed"


def _has_trailing_nan_only(bp: DataArray) -> bool:
    """Check that NaN values only appear as trailing entries along BREAKPOINT_DIM."""
    valid = ~bp.isnull()
    cummin = np.minimum.accumulate(valid.values, axis=valid.dims.index(BREAKPOINT_DIM))
    cummin_da = DataArray(cummin, coords=valid.coords, dims=valid.dims)
    return not bool((valid & ~cummin_da).any())


def _paired_valid_points(*points: DataArray) -> DataArray:
    invalid = points[0].isnull()
    for point in points[1:]:
        invalid = invalid | point.isnull()
    return points[0].where(~invalid)


def _validate_shared_coords(points: Sequence[DataArray]) -> None:
    skip = {BREAKPOINT_DIM, SEGMENT_DIM} | set(HELPER_DIMS)
    for i, left in enumerate(points):
        for right in points[i + 1 :]:
            for dim in (set(left.dims) & set(right.dims)) - skip:
                left_index = pd.Index(left.coords[dim].values)
                right_index = pd.Index(right.coords[dim].values)
                if not left_index.equals(right_index):
                    raise ValueError(
                        f"Breakpoint coordinates for dimension '{dim}' must match."
                    )


def _validate_expr_coords(
    points: Sequence[DataArray], exprs: Sequence[LinearExpression]
) -> None:
    skip = {BREAKPOINT_DIM, SEGMENT_DIM} | set(HELPER_DIMS)
    for point in points:
        for expr in exprs:
            for dim in (set(point.dims) & set(expr.coord_dims)) - skip:
                point_index = pd.Index(point.coords[dim].values)
                expr_index = pd.Index(expr.coords[dim].values)
                if not point_index.equals(expr_index):
                    raise ValueError(
                        f"Breakpoint coordinates for dimension '{dim}' must match "
                        "the expression coordinates."
                    )


def _to_linexpr(expr: LinExprLike) -> LinearExpression:
    from linopy.expressions import LinearExpression

    if isinstance(expr, LinearExpression):
        return expr
    return expr.to_linexpr()


def _var_coords_from(
    points: DataArray, exclude: set[str] | None = None
) -> list[pd.Index]:
    """Extract pd.Index coords from points, excluding specified dimensions."""
    excluded = exclude or set()
    return [
        pd.Index(points.coords[d].values, name=d)
        for d in points.dims
        if d not in excluded
    ]


def _broadcast_points(
    points: DataArray,
    *exprs: LinExprLike,
    disjunctive: bool = False,
) -> DataArray:
    """Broadcast points to cover all dimensions from exprs."""
    skip: set[str] = {BREAKPOINT_DIM} | set(HELPER_DIMS)
    if disjunctive:
        skip.add(SEGMENT_DIM)

    lin_exprs = [_to_linexpr(e) for e in exprs]

    point_dims = {str(d) for d in points.dims}

    # Iterate exprs/dims in order; a set would give a hash-dependent,
    # run-varying expanded dimension order.
    expand_map: dict[str, list] = {}
    for le in lin_exprs:
        for dim in le.coord_dims:
            d = str(dim)
            if d in skip or d in point_dims or d in expand_map:
                continue
            if d in le.coords:
                expand_map[d] = list(le.coords[d].values)

    if expand_map:
        points = points.expand_dims(expand_map)
    return points


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def add_piecewise_formulation(
    model: Model,
    *pairs: tuple[LinExprLike, BreaksOrSlopes]
    | tuple[LinExprLike, BreaksOrSlopes, Literal["==", "<=", ">="]],
    method: PWL_METHOD = "auto",
    active: LinExprLike | None = None,
    active_fill: int | None = None,
    name: str | None = None,
) -> PiecewiseFormulation:
    r"""
    Add piecewise linear constraints.

    Each positional argument is a ``(expression, breakpoints)`` tuple, or
    ``(expression, breakpoints, sign)`` to mark that expression as bounded
    by the piecewise curve rather than pinned to it.  All expressions are
    linked through shared interpolation weights so that every operating
    point lies on the same piece of the piecewise curve.

    Example — 2 variables (joint equality, the default)::

        m.add_piecewise_formulation(
            (power, [0, 30, 60, 100]),
            (fuel,  [0, 36, 84, 170]),
        )

    Example — 3 variables, CHP plant (joint equality)::

        m.add_piecewise_formulation(
            (power, [0, 30, 60, 100]),
            (fuel,  [0, 40, 85, 160]),
            (heat,  [0, 25, 55, 95]),
        )

    **Per-tuple sign — inequality bounds:**

    Add ``"<="`` or ``">="`` as a third tuple element to mark a single
    expression as bounded by the curve instead of pinned to it.  The
    remaining tuples are still forced to equality (input on the curve).
    Reads directly as the relation it encodes:

    .. code-block:: python

        # fuel <= f(power) — concave curve, bounded above
        m.add_piecewise_formulation(
            (fuel, y_pts, "<="),
            (power, x_pts),
        )

        # cost >= g(load) — convex curve, bounded below
        m.add_piecewise_formulation(
            (cost, y_pts, ">="),
            (load, x_pts),
        )

    For 2-variable inequality on convex/concave curves, ``method="auto"``
    automatically selects a pure-LP tangent-line formulation (no auxiliary
    variables).  Non-convex curves fall back to SOS2/incremental with the
    sign applied to the bounded tuple's link constraint.

    **Restrictions on per-tuple sign:**

    - At most one tuple may carry a non-equality sign.  All other tuples
      default to ``"=="``.
    - With **3 or more** tuples, all signs must be ``"=="``.

    Multi-bounded and N≥3-inequality use cases aren't supported yet.  If
    you have a concrete use case, please open an issue at
    https://github.com/PyPSA/linopy/issues so we can scope it properly.

    Parameters
    ----------
    *pairs : tuple of (expression, breakpoints) or (expression, breakpoints, sign)
        Each pair links an expression (Variable or LinearExpression) to
        its breakpoint values.  An optional third element ``"<="`` or
        ``">="`` marks that expression as bounded by the curve; if
        omitted, the expression is pinned (``"=="``).  At least two pairs
        are required; at most one may carry a non-equality sign; with
        3+ pairs all signs must be ``"=="``.
    method : {"auto", "sos2", "incremental", "lp"}, default "auto"
        Formulation method.
        ``"lp"`` uses tangent lines (pure LP, no variables) and requires
        exactly one tuple with ``"<="`` or ``">="`` plus a matching-curvature
        curve with exactly two tuples.
        ``"auto"`` picks ``"lp"`` when applicable, otherwise
        ``"incremental"`` (monotonic breakpoints) or ``"sos2"``.
    active : Variable or LinearExpression, optional
        Binary variable that gates the piecewise function.  When
        ``active=0``, all auxiliary variables are forced to zero.
        Not supported with ``method="lp"``.

        ``active`` must cover the formulation's full coordinate.  A
        *partial* gate — one defined over only a subset of the coordinate's
        labels, or carrying masked entries — is rejected unless
        ``active_fill`` is set (see below).

        With all-equality tuples (the default), the output is then pinned
        to ``0``.  With a bounded tuple (``"<="`` / ``">="``), deactivation
        only pushes the signed bound to ``0`` (the output is ≤ 0 or ≥ 0
        respectively) — the complementary bound still comes from the
        output variable's own lower/upper.  In the common case where
        the output is naturally non-negative (fuel, cost, heat, …),
        just set ``lower=0`` on that variable: combined with the
        ``y ≤ 0`` constraint from deactivation, this forces ``y = 0``
        automatically.  For outputs that genuinely need both signs you
        must add the complementary bound yourself (e.g., a big-M
        coupling ``y`` with ``active``).
    active_fill : int, optional
        Fill value for the gap entries of a partial ``active`` — those where
        ``active`` has no label (a subset of the coordinate) or is masked:
        ``1`` treats them as always active (ungated), ``0`` as always off.
        When ``None`` (the default) a partial ``active`` is rejected instead.
        Useful when one formulation mixes gated and ungated entities (e.g.
        committable and non-committable units sharing a ``status``).
        Transitional convenience: under v1 semantics, pad ``active``
        explicitly with ``active.reindex(coords).fillna(value)`` instead —
        this parameter is slated for removal then.
    name : str, optional
        Base name for generated variables/constraints.

    Returns
    -------
    PiecewiseFormulation

    Warns
    -----
    EvolvingAPIWarning
        ``add_piecewise_formulation`` is a newly-added API; details such
        as the per-tuple sign convention and ``active`` + non-equality
        sign semantics may be refined based on user feedback.  Silence
        with ``warnings.filterwarnings("ignore",
        category=linopy.EvolvingAPIWarning)``.
    """
    _warn_evolving_api(
        "add_piecewise_formulation",
        "piecewise: add_piecewise_formulation is a new API; some details "
        "(e.g. the per-tuple sign convention, active+sign semantics) "
        "may be refined in minor releases.  Please share your use cases "
        "or concerns at https://github.com/PyPSA/linopy/issues — your "
        "feedback shapes what stabilises.  This warning fires once per "
        "session; silence entirely with "
        '`warnings.filterwarnings("ignore", category=linopy.EvolvingAPIWarning)`.',
    )

    if method not in PWL_METHODS:
        raise ValueError(f"method must be one of {sorted(PWL_METHODS)}, got '{method}'")

    if len(pairs) < 2:
        raise TypeError(
            "add_piecewise_formulation() requires at least 2 "
            "(expression, breakpoints[, sign]) pairs."
        )

    # Parse and normalise per-tuple signs.  Each pair is either
    # (expr, bp) — sign defaults to "==" — or (expr, bp, sign).
    parsed: list[tuple[LinExprLike, BreaksOrSlopes, str]] = []
    for i, pair in enumerate(pairs):
        if not isinstance(pair, tuple) or len(pair) not in (2, 3):
            raise TypeError(
                f"Argument {i + 1} must be a (expression, breakpoints) "
                f"or (expression, breakpoints, sign) tuple, got {pair!r}."
            )
        if len(pair) == 2:
            expr, bp = pair
            tuple_sign: str = EQUAL
        else:
            expr, bp, raw_sign = pair
            tuple_sign = sign_replace_dict.get(raw_sign, raw_sign)
            if tuple_sign not in SIGNS:
                raise ValueError(
                    f"Argument {i + 1}: sign must be one of "
                    f"{sorted(SIGNS)}, got {raw_sign!r}."
                )
        parsed.append((expr, bp, tuple_sign))

    slopes_set = {i for i, p in enumerate(parsed) if isinstance(p[1], Slopes)}
    if slopes_set:
        non_slopes_idx = [i for i in range(len(parsed)) if i not in slopes_set]
        if not non_slopes_idx:
            raise ValueError(
                "All tuples are Slopes; at least one tuple must carry an "
                "explicit x grid.  Pass the x grid via a regular tuple "
                "or call Slopes(...).to_breakpoints(x_pts) explicitly."
            )
        if len(non_slopes_idx) > 1:
            raise ValueError(
                f"Slopes tuples present at positions {sorted(slopes_set)}, "
                f"but {len(non_slopes_idx)} non-Slopes tuples carry their "
                f"own breakpoint values (positions {non_slopes_idx}).  "
                "There is no canonical x grid for the Slopes to integrate "
                "against — borrowing from any one of them would silently "
                "depend on tuple order.  Either reduce to a single non-Slopes "
                "tuple, or resolve the Slopes explicitly by calling "
                "Slopes(...).to_breakpoints(x_pts) before passing it in."
            )
        x_grid = parsed[non_slopes_idx[0]][1]
        parsed = [
            (expr, bp.to_breakpoints(x_grid), sign)
            if isinstance(bp, Slopes)
            else (expr, bp, sign)
            for expr, bp, sign in parsed
        ]

    # At most one non-equality sign; with 3+ tuples, none.
    bounded_positions = [i for i, p in enumerate(parsed) if p[2] != EQUAL]
    if len(bounded_positions) > 1:
        raise ValueError(
            "At most one tuple may carry a non-equality sign; got "
            f"{len(bounded_positions)} (positions {bounded_positions})."
        )
    if len(parsed) >= 3 and bounded_positions:
        raise ValueError(
            "Non-equality signs are not supported with 3+ tuples. "
            "Use sign='==' on all tuples (the default), or reduce to 2 tuples. "
            "If you have a concrete use case, please open an issue at "
            "https://github.com/PyPSA/linopy/issues."
        )

    signed_idx: int | None
    if bounded_positions:
        bidx = bounded_positions[0]
        signed_idx = bidx
        sign: str = parsed[bidx][2]
    else:
        signed_idx = None
        sign = EQUAL

    if method == "lp" and sign == EQUAL:
        raise ValueError(
            "method='lp' requires exactly one tuple with sign='<=' or '>='."
        )

    coerced_bps: list[DataArray] = []
    for _, bp, _s in parsed:
        if not isinstance(bp, DataArray):
            bp = _coerce_breaks(bp)
        scalar_coords = [c for c in bp.coords if c not in bp.dims]
        if scalar_coords:
            bp = bp.drop_vars(scalar_coords)
        coerced_bps.append(bp)

    disjunctive = SEGMENT_DIM in coerced_bps[0].dims
    for i in range(1, len(coerced_bps)):
        _validate_breakpoint_shapes(coerced_bps[0], coerced_bps[i])

    raw_exprs = [expr for expr, _, _ in parsed]
    lin_exprs = [_to_linexpr(expr) for expr in raw_exprs]
    bp_list = [
        _broadcast_points(bp, *raw_exprs, disjunctive=disjunctive) for bp in coerced_bps
    ]
    _validate_shared_coords(bp_list)
    _validate_expr_coords(bp_list, lin_exprs)

    combined_null = bp_list[0].isnull()
    for bp in bp_list[1:]:
        combined_null = combined_null | bp.isnull()
    bp_mask = ~combined_null if bool(combined_null.any()) else None

    if name is None:
        name = f"pwl{model._pwlCounter}"
        model._pwlCounter += 1

    from linopy.variables import Variable

    link_coords: list[str] = []
    for i, expr in enumerate(raw_exprs):
        if isinstance(expr, Variable) and expr.name:
            link_coords.append(expr.name)
        else:
            # Internal-prefixed fallback so a user variable named e.g. "1"
            # can't collide with the synthetic coord for an unnamed expr.
            link_coords.append(f"_pwl_{i}")

    if active is None:
        if active_fill is not None:
            raise ValueError("`active_fill` has no effect without `active`.")
        active_expr = None
    else:
        active_expr = _resolve_active(_to_linexpr(active), bp_list[0], active_fill)

    if signed_idx is None:
        inputs = _PwlInputs(
            pinned_exprs=lin_exprs,
            pinned_bps=bp_list,
            pinned_coords=link_coords,
            bounded_expr=None,
            bounded_bp=None,
            bounded_coord=None,
            bounded_sign=EQUAL,
            bp_mask=bp_mask,
        )
    else:
        inputs = _PwlInputs(
            pinned_exprs=[e for j, e in enumerate(lin_exprs) if j != signed_idx],
            pinned_bps=[b for j, b in enumerate(bp_list) if j != signed_idx],
            pinned_coords=[c for j, c in enumerate(link_coords) if j != signed_idx],
            bounded_expr=lin_exprs[signed_idx],
            bounded_bp=bp_list[signed_idx],
            bounded_coord=link_coords[signed_idx],
            bounded_sign=sign,
            bp_mask=bp_mask,
        )

    vars_before = set(model.variables)
    cons_before = set(model.constraints)

    if disjunctive:
        if method == "incremental":
            raise ValueError(
                "Incremental method is not supported for disjunctive constraints"
            )
        if method == "lp":
            raise ValueError(
                "method='lp' is not supported for disjunctive (segment) breakpoints"
            )
        _add_disjunctive(model, name, inputs, active_expr)
        resolved_method: PWL_METHOD = "sos2"
    else:
        resolved_method = _add_continuous(model, name, inputs, method, active_expr)

    new_vars = [n for n in model.variables if n not in vars_before]
    new_cons = [n for n in model.constraints if n not in cons_before]

    if method == "auto":
        logger.info(
            "piecewise formulation '%s': auto selected method='%s' "
            "(sign='%s', %d pair%s)",
            name,
            resolved_method,
            sign,
            inputs.n_tuples,
            "" if inputs.n_tuples == 1 else "s",
        )

    convexity: PWL_CONVEXITY | None = None
    if inputs.n_tuples == 2 and not disjunctive:
        if inputs.is_equality:
            x_pts = inputs.pinned_bps[1]
            y_pts: DataArray = inputs.pinned_bps[0]
        else:
            assert inputs.bounded_bp is not None
            x_pts = inputs.pinned_bps[0]
            y_pts = inputs.bounded_bp
        if _check_strict_monotonicity(x_pts):
            convexity = _detect_convexity(x_pts, y_pts)

    result = PiecewiseFormulation(
        name=name,
        method=resolved_method,
        variable_names=new_vars,
        constraint_names=new_cons,
        model=model,
        convexity=convexity,
    )
    model._piecewise_formulations[name] = result
    return result


def _stack_along_link(
    items: Sequence[DataArray | xr.Dataset],
    link_coords: list[str],
    link_dim: str,
) -> DataArray:
    """Expand and concatenate DataArrays/Datasets along a new link dimension."""
    expanded = [
        item.expand_dims({link_dim: [c]}) for item, c in zip(items, link_coords)
    ]
    return xr.concat(expanded, dim=link_dim, coords="minimal")  # type: ignore


@dataclass
class _PwlInputs:
    """
    Categorised piecewise inputs (post-coercion, post-broadcast).

    ``pinned_*`` are the equality tuples in the user's original order.
    ``bounded_*`` is the single non-equality tuple, or ``None``.
    ``bounded_sign`` is ``EQUAL`` iff ``bounded_expr is None``.
    """

    pinned_exprs: list[LinearExpression]
    pinned_bps: list[DataArray]
    pinned_coords: list[str]
    bounded_expr: LinearExpression | None
    bounded_bp: DataArray | None
    bounded_coord: str | None
    bounded_sign: str
    bp_mask: DataArray | None
    link_dim: str = PWL_LINK_DIM

    @property
    def is_equality(self) -> bool:
        return self.bounded_expr is None

    @property
    def n_tuples(self) -> int:
        return len(self.pinned_exprs) + (0 if self.is_equality else 1)

    def all_bps(self) -> list[DataArray]:
        if self.bounded_bp is None:
            return list(self.pinned_bps)
        return [self.bounded_bp, *self.pinned_bps]

    def all_coords(self) -> list[str]:
        if self.bounded_coord is None:
            return list(self.pinned_coords)
        return [self.bounded_coord, *self.pinned_coords]

    def all_exprs(self) -> list[LinearExpression]:
        if self.bounded_expr is None:
            return list(self.pinned_exprs)
        return [self.bounded_expr, *self.pinned_exprs]


def _lp_eligibility(
    inputs: _PwlInputs,
    active: LinearExpression | None,
) -> tuple[bool, str]:
    """
    Check whether LP tangent-lines dispatch is applicable.

    Returns ``(True, "")`` if LP is applicable, else ``(False, reason)``.
    """
    if inputs.n_tuples != 2:
        return False, f"{inputs.n_tuples} expressions (LP supports only 2)"
    if inputs.is_equality:
        return False, "all tuples are equality (LP needs one bounded tuple)"
    if active is not None:
        return False, "active=... is not supported by LP"
    assert inputs.bounded_bp is not None  # narrowed by is_equality check
    x_pts = inputs.pinned_bps[0]
    y_pts = inputs.bounded_bp
    paired_x = _paired_valid_points(x_pts, y_pts)
    if not _check_strict_monotonicity(paired_x):
        return False, "paired x breakpoints are not strictly monotonic"
    if not _has_trailing_nan_only(paired_x):
        return False, "paired breakpoints contain non-trailing NaN"
    convexity = _detect_convexity(x_pts, y_pts)
    sign = inputs.bounded_sign
    if sign == LESS_EQUAL and convexity not in ("concave", "linear"):
        return False, f"sign='<=' needs concave/linear curvature, got '{convexity}'"
    if sign == GREATER_EQUAL and convexity not in ("convex", "linear"):
        return False, f"sign='>=' needs convex/linear curvature, got '{convexity}'"
    return True, ""


@dataclass
class _PwlLinks:
    """
    Stacked link expressions consumed by SOS2/incremental/disjunctive builders.
    """

    stacked_bp: DataArray
    link_dim: str
    bp_mask: DataArray | None
    sign: str
    eq_expr: LinearExpression | None
    eq_bp: DataArray | None
    signed_expr: LinearExpression | None
    signed_bp: DataArray | None


def _build_links(model: Model, inputs: _PwlInputs) -> _PwlLinks:
    """Stack ``inputs`` into the link representation."""
    from linopy.expressions import LinearExpression

    stacked_bp = _stack_along_link(
        inputs.all_bps(), inputs.all_coords(), inputs.link_dim
    )

    if inputs.is_equality:
        eq_data = _stack_along_link(
            [e.data for e in inputs.pinned_exprs],
            inputs.pinned_coords,
            inputs.link_dim,
        )
        return _PwlLinks(
            stacked_bp=stacked_bp,
            link_dim=inputs.link_dim,
            bp_mask=inputs.bp_mask,
            sign=EQUAL,
            eq_expr=LinearExpression(eq_data, model),
            eq_bp=stacked_bp,
            signed_expr=None,
            signed_bp=None,
        )

    if inputs.pinned_exprs:
        eq_data = _stack_along_link(
            [e.data for e in inputs.pinned_exprs],
            inputs.pinned_coords,
            inputs.link_dim,
        )
        eq_expr: LinearExpression | None = LinearExpression(eq_data, model)
        eq_bp: DataArray | None = _stack_along_link(
            inputs.pinned_bps, inputs.pinned_coords, inputs.link_dim
        )
    else:
        eq_expr = None
        eq_bp = None

    return _PwlLinks(
        stacked_bp=stacked_bp,
        link_dim=inputs.link_dim,
        bp_mask=inputs.bp_mask,
        sign=inputs.bounded_sign,
        eq_expr=eq_expr,
        eq_bp=eq_bp,
        signed_expr=inputs.bounded_expr,
        signed_bp=inputs.bounded_bp,
    )


def _try_lp(
    model: Model,
    name: str,
    inputs: _PwlInputs,
    method: str,
    active: LinearExpression | None,
) -> bool:
    """Dispatch the LP formulation if requested or eligible."""
    if method not in ("lp", "auto"):
        return False
    if method == "auto" and inputs.is_equality:
        return False

    ok, reason = _lp_eligibility(inputs, active)
    if not ok:
        if method == "lp":
            raise ValueError(
                f"method='lp' is not applicable: {reason}. Use method='auto'."
            )
        logger.info(
            "piecewise formulation '%s': LP not applicable (%s); "
            "will use SOS2/incremental instead",
            name,
            reason,
        )
        return False

    assert inputs.bounded_expr is not None
    assert inputs.bounded_bp is not None
    _add_lp(
        model,
        name,
        inputs.pinned_exprs[0],
        inputs.bounded_expr,
        inputs.pinned_bps[0],
        inputs.bounded_bp,
        inputs.bounded_sign,
    )
    return True


def _resolve_sos2_vs_incremental(
    method: str, stacked_bp: DataArray
) -> Literal["incremental", "sos2"]:
    """
    Validate and (for ``method="auto"``) pick between SOS2 and
    incremental based on monotonicity and NaN layout.
    """
    trailing_nan_only = _has_trailing_nan_only(stacked_bp)
    is_monotonic = _check_strict_monotonicity(stacked_bp)

    if method == "auto":
        if not trailing_nan_only:
            raise ValueError(
                "SOS2 method does not support non-trailing NaN breakpoints."
            )
        return "incremental" if is_monotonic else "sos2"

    if method == "incremental":
        if not is_monotonic:
            raise ValueError(
                "Incremental method requires strictly monotonic breakpoints."
            )
        if not trailing_nan_only:
            raise ValueError(
                "Incremental method does not support non-trailing NaN breakpoints."
            )
        return "incremental"

    assert method == "sos2"
    _validate_numeric_breakpoint_coords(stacked_bp)
    if not trailing_nan_only:
        raise ValueError("SOS2 method does not support non-trailing NaN breakpoints.")
    return "sos2"


def _add_continuous(
    model: Model,
    name: str,
    inputs: _PwlInputs,
    method: str,
    active: LinearExpression | None = None,
) -> PWL_METHOD:
    """Returns the resolved method name (``"lp"``, ``"sos2"``, ``"incremental"``)."""
    if _try_lp(model, name, inputs, method, active):
        return "lp"

    links = _build_links(model, inputs)
    resolved = _resolve_sos2_vs_incremental(method, links.stacked_bp)

    if resolved == "sos2":
        rhs = active if active is not None else 1
        _add_sos2(model, name, links, rhs)
    else:
        _add_incremental(model, name, links, active)
    return resolved


def _add_sos2(
    model: Model,
    name: str,
    links: _PwlLinks,
    rhs: LinearExpression | int,
) -> None:
    """
    SOS2 formulation.  ``links.eq_expr`` is the equality side;
    ``links.signed_expr`` (if any) is the output-side link.
    """
    dim = BREAKPOINT_DIM
    stacked_bp = links.stacked_bp
    extra = _var_coords_from(stacked_bp, exclude={dim, links.link_dim})
    lambda_coords = extra + [pd.Index(stacked_bp.coords[dim].values, name=dim)]

    lambda_var = model.add_variables(
        lower=0,
        upper=1,
        coords=lambda_coords,
        name=f"{name}{PWL_LAMBDA_SUFFIX}",
        mask=links.bp_mask,
    )
    model.add_sos_constraints(lambda_var, sos_type=2, sos_dim=dim)
    model.add_constraints(
        lambda_var.sum(dim=dim) == rhs, name=f"{name}{PWL_CONVEX_SUFFIX}"
    )

    if links.eq_expr is not None and links.eq_bp is not None:
        input_weighted = (lambda_var * links.eq_bp).sum(dim=dim)
        model.add_constraints(
            links.eq_expr == input_weighted, name=f"{name}{PWL_LINK_SUFFIX}"
        )

    if links.signed_expr is not None and links.signed_bp is not None:
        output_weighted = (lambda_var * links.signed_bp).sum(dim=dim)
        _add_signed_link(
            model,
            links.signed_expr,
            output_weighted,
            links.sign,
            f"{name}{PWL_OUTPUT_LINK_SUFFIX}",
        )


def _add_incremental(
    model: Model,
    name: str,
    links: _PwlLinks,
    active: LinearExpression | None,
) -> None:
    """
    Incremental formulation.  ``links.eq_expr`` is the equality side;
    ``links.signed_expr`` (if any) is the output-side link.
    """
    dim = BREAKPOINT_DIM
    stacked_bp = links.stacked_bp
    extra = _var_coords_from(stacked_bp, exclude={dim, links.link_dim})

    n_pieces = stacked_bp.sizes[dim] - 1
    piece_dim = LP_PIECE_DIM
    piece_index = pd.Index(range(n_pieces), name=piece_dim)
    delta_coords = extra + [piece_index]

    if links.bp_mask is not None:
        mask_lo = links.bp_mask.isel({dim: slice(None, -1)}).rename({dim: piece_dim})
        mask_hi = links.bp_mask.isel({dim: slice(1, None)}).rename({dim: piece_dim})
        mask_lo[piece_dim] = piece_index
        mask_hi[piece_dim] = piece_index
        delta_mask: DataArray | None = mask_lo & mask_hi
    else:
        delta_mask = None

    delta_var = model.add_variables(
        lower=0,
        upper=1,
        coords=delta_coords,
        name=f"{name}{PWL_DELTA_SUFFIX}",
        mask=delta_mask,
    )

    if active is not None:
        model.add_constraints(
            delta_var <= active, name=f"{name}{PWL_ACTIVE_BOUND_SUFFIX}"
        )

    binary_var = model.add_variables(
        binary=True,
        coords=delta_coords,
        name=f"{name}{PWL_ORDER_BINARY_SUFFIX}",
        mask=delta_mask,
    )
    model.add_constraints(
        delta_var <= binary_var, name=f"{name}{PWL_DELTA_BOUND_SUFFIX}"
    )

    if n_pieces >= 2:
        delta_lo = delta_var.isel({piece_dim: slice(None, -1)}, drop=True)
        delta_hi = delta_var.isel({piece_dim: slice(1, None)}, drop=True)
        model.add_constraints(
            delta_hi <= delta_lo, name=f"{name}{PWL_FILL_ORDER_SUFFIX}"
        )
        binary_hi = binary_var.isel({piece_dim: slice(1, None)}, drop=True)
        model.add_constraints(
            binary_hi <= delta_lo, name=f"{name}{PWL_BINARY_ORDER_SUFFIX}"
        )

    def _incremental_weighted(bp: DataArray) -> LinearExpression:
        steps = bp.diff(dim).rename({dim: piece_dim})
        steps[piece_dim] = piece_index
        bp0 = bp.isel({dim: 0})
        bp0_term: DataArray | LinearExpression = bp0
        if active is not None:
            bp0_term = bp0 * active
        return (delta_var * steps).sum(dim=piece_dim) + bp0_term

    if links.eq_expr is not None and links.eq_bp is not None:
        model.add_constraints(
            links.eq_expr == _incremental_weighted(links.eq_bp),
            name=f"{name}{PWL_LINK_SUFFIX}",
        )

    if links.signed_expr is not None and links.signed_bp is not None:
        _add_signed_link(
            model,
            links.signed_expr,
            _incremental_weighted(links.signed_bp),
            links.sign,
            f"{name}{PWL_OUTPUT_LINK_SUFFIX}",
        )


def _add_disjunctive(
    model: Model,
    name: str,
    inputs: _PwlInputs,
    active: LinearExpression | None = None,
) -> None:
    """Disjunctive SOS2 formulation."""
    link_dim = inputs.link_dim
    links = _build_links(model, inputs)
    stacked_bp = links.stacked_bp
    bp_mask = inputs.bp_mask

    _validate_numeric_breakpoint_coords(stacked_bp)
    if not _has_trailing_nan_only(stacked_bp):
        raise ValueError(
            "Disjunctive SOS2 does not support non-trailing NaN breakpoints. "
            "NaN values must only appear at the end of the breakpoint sequence."
        )

    dim = BREAKPOINT_DIM
    extra = _var_coords_from(stacked_bp, exclude={dim, SEGMENT_DIM, link_dim})
    lambda_coords = extra + [
        pd.Index(stacked_bp.coords[SEGMENT_DIM].values, name=SEGMENT_DIM),
        pd.Index(stacked_bp.coords[dim].values, name=dim),
    ]
    binary_coords = extra + [
        pd.Index(stacked_bp.coords[SEGMENT_DIM].values, name=SEGMENT_DIM),
    ]
    binary_mask = bp_mask.any(dim=dim) if bp_mask is not None else None

    binary_var = model.add_variables(
        binary=True,
        coords=binary_coords,
        name=f"{name}{PWL_SEGMENT_BINARY_SUFFIX}",
        mask=binary_mask,
    )
    rhs = active if active is not None else 1
    model.add_constraints(
        binary_var.sum(dim=SEGMENT_DIM) == rhs,
        name=f"{name}{PWL_SELECT_SUFFIX}",
    )

    lambda_var = model.add_variables(
        lower=0,
        upper=1,
        coords=lambda_coords,
        name=f"{name}{PWL_LAMBDA_SUFFIX}",
        mask=bp_mask,
    )
    model.add_sos_constraints(lambda_var, sos_type=2, sos_dim=dim)
    model.add_constraints(
        lambda_var.sum(dim=dim) == binary_var,
        name=f"{name}{PWL_CONVEX_SUFFIX}",
    )

    if links.eq_expr is not None and links.eq_bp is not None:
        input_weighted = (lambda_var * links.eq_bp).sum(dim=[SEGMENT_DIM, dim])
        model.add_constraints(
            links.eq_expr == input_weighted, name=f"{name}{PWL_LINK_SUFFIX}"
        )

    if links.signed_expr is not None and links.signed_bp is not None:
        output_weighted = (lambda_var * links.signed_bp).sum(dim=[SEGMENT_DIM, dim])
        _add_signed_link(
            model,
            links.signed_expr,
            output_weighted,
            links.sign,
            f"{name}{PWL_OUTPUT_LINK_SUFFIX}",
        )


def _add_signed_link(
    model: Model,
    lhs: LinearExpression,
    rhs: LinearExpression,
    sign: str,
    name: str,
    mask: DataArray | None = None,
) -> Constraint:
    """Add a link constraint with the requested sign."""
    if sign == EQUAL:
        return model.add_constraints(lhs == rhs, name=name, mask=mask)
    elif sign == LESS_EQUAL:
        return model.add_constraints(lhs <= rhs, name=name, mask=mask)
    else:  # ">="
        return model.add_constraints(lhs >= rhs, name=name, mask=mask)


def _add_lp(
    model: Model,
    name: str,
    x_expr: LinearExpression,
    y_expr: LinearExpression,
    x_points: DataArray,
    y_points: DataArray,
    sign: str,
) -> None:
    """
    LP tangent-line formulation (no auxiliary variables).

    Adds one chord constraint per piece plus domain bounds on x.
    Trailing-NaN pieces (per-entity short curves) are masked out so
    they do not contribute spurious ``y ≤ 0`` constraints.
    """
    # Per-piece validity: both endpoints must be non-NaN.
    bp_valid = ~(x_points.isnull() | y_points.isnull())
    piece_count = x_points.sizes[BREAKPOINT_DIM] - 1
    piece_index = np.arange(piece_count)
    full_mask = _rename_to_pieces(
        bp_valid.isel({BREAKPOINT_DIM: slice(None, -1)})
        & bp_valid.isel({BREAKPOINT_DIM: slice(1, None)}).values,
        piece_index,
    )
    piece_mask: DataArray | None = None if bool(full_mask.all()) else full_mask

    # Use the internal impl so we don't fire a second EvolvingAPIWarning —
    # ``add_piecewise_formulation`` already warned on entry.
    tangents = _tangent_lines_impl(x_expr, x_points, y_points)
    _add_signed_link(
        model,
        y_expr,
        tangents,
        sign,
        f"{name}{PWL_CHORD_SUFFIX}",
        mask=piece_mask,
    )

    # Domain bounds: x ∈ [x_min, x_max] over paired-valid breakpoints.
    paired_x_points = x_points.where(bp_valid)
    x_min = paired_x_points.min(dim=BREAKPOINT_DIM)
    x_max = paired_x_points.max(dim=BREAKPOINT_DIM)
    model.add_constraints(x_expr >= x_min, name=f"{name}{PWL_DOMAIN_LO_SUFFIX}")
    model.add_constraints(x_expr <= x_max, name=f"{name}{PWL_DOMAIN_HI_SUFFIX}")
