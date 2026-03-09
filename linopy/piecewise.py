"""
Piecewise linear constraint formulations.

Provides SOS2, incremental, pure LP, and disjunctive piecewise linear
constraint methods for use with linopy.Model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray

from linopy.constants import (
    BREAKPOINT_DIM,
    HELPER_DIMS,
    LP_SEG_DIM,
    PWL_AUX_SUFFIX,
    PWL_BINARY_SUFFIX,
    PWL_CONVEX_SUFFIX,
    PWL_DELTA_SUFFIX,
    PWL_FILL_SUFFIX,
    PWL_INC_BINARY_SUFFIX,
    PWL_INC_LINK_SUFFIX,
    PWL_INC_ORDER_SUFFIX,
    PWL_LAMBDA_SUFFIX,
    PWL_LP_DOMAIN_SUFFIX,
    PWL_LP_SUFFIX,
    PWL_SELECT_SUFFIX,
    PWL_X_LINK_SUFFIX,
    PWL_Y_LINK_SUFFIX,
    SEGMENT_DIM,
)

if TYPE_CHECKING:
    from linopy.constraints import Constraint
    from linopy.expressions import LinearExpression
    from linopy.model import Model
    from linopy.types import LinExprLike

# Accepted input types for breakpoint-like data
BreaksLike: TypeAlias = (
    Sequence[float] | DataArray | pd.Series | pd.DataFrame | dict[str, Sequence[float]]
)

# Accepted input types for segment-like data (2D: segments × breakpoints)
SegmentsLike: TypeAlias = (
    Sequence[Sequence[float]]
    | DataArray
    | pd.DataFrame
    | dict[str, Sequence[Sequence[float]]]
)


# ---------------------------------------------------------------------------
# DataArray construction helpers
# ---------------------------------------------------------------------------


def _sequence_to_array(values: Sequence[float]) -> DataArray:
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
    combined = xr.concat(parts, dim=dim)
    max_bp = max(max(len(seg) for seg in sl) for sl in d.values())
    max_seg = max(len(sl) for sl in d.values())
    if combined.sizes[BREAKPOINT_DIM] < max_bp or combined.sizes[SEGMENT_DIM] < max_seg:
        combined = combined.reindex(
            {BREAKPOINT_DIM: np.arange(max_bp), SEGMENT_DIM: np.arange(max_seg)},
            fill_value=np.nan,
        )
    return combined


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def slopes_to_points(
    x_points: list[float], slopes: list[float], y0: float
) -> list[float]:
    """
    Convert segment slopes + initial y-value to y-coordinates at each breakpoint.

    Parameters
    ----------
    x_points : list[float]
        Breakpoint x-coordinates (length n).
    slopes : list[float]
        Slope of each segment (length n-1).
    y0 : float
        y-value at the first breakpoint.

    Returns
    -------
    list[float]
        y-coordinates at each breakpoint (length n).

    Raises
    ------
    ValueError
        If ``len(slopes) != len(x_points) - 1``.
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
    values: BreaksLike | None = None,
    *,
    slopes: BreaksLike | None = None,
    x_points: BreaksLike | None = None,
    y0: float | dict[str, float] | pd.Series | DataArray | None = None,
    dim: str | None = None,
) -> DataArray:
    """
    Create a breakpoint DataArray for piecewise linear constraints.

    Two modes (mutually exclusive):

    **Points mode**: ``breakpoints(values, ...)``

    **Slopes mode**: ``breakpoints(slopes=..., x_points=..., y0=...)``

    Parameters
    ----------
    values : BreaksLike, optional
        Breakpoint values. Accepted types: ``Sequence[float]``,
        ``pd.Series``, ``pd.DataFrame``, or ``xr.DataArray``.
        A 1D input (list, Series) creates 1D breakpoints.
        A 2D input (DataFrame, multi-dim DataArray) creates per-entity
        breakpoints (``dim`` is required for DataFrame).
    slopes : BreaksLike, optional
        Segment slopes. Mutually exclusive with ``values``.
    x_points : BreaksLike, optional
        Breakpoint x-coordinates. Required with ``slopes``.
    y0 : float, dict, pd.Series, or DataArray, optional
        Initial y-value. Required with ``slopes``. A scalar broadcasts to
        all entities. A dict/Series/DataArray provides per-entity values.
    dim : str, optional
        Entity dimension name. Required when ``values`` or ``slopes`` is a
        ``pd.DataFrame`` or ``dict``.

    Returns
    -------
    DataArray
    """
    # Validate mutual exclusivity
    if values is not None and slopes is not None:
        raise ValueError("'values' and 'slopes' are mutually exclusive")
    if values is not None and (x_points is not None or y0 is not None):
        raise ValueError("'x_points' and 'y0' are forbidden when 'values' is given")
    if slopes is not None:
        if x_points is None or y0 is None:
            raise ValueError("'slopes' requires both 'x_points' and 'y0'")

    # Slopes mode: convert to points, then fall through to coerce
    if slopes is not None:
        if x_points is None or y0 is None:
            raise ValueError("'slopes' requires both 'x_points' and 'y0'")
        slopes_arr = _coerce_breaks(slopes, dim)
        xp_arr = _coerce_breaks(x_points, dim)

        # 1D case: single set of breakpoints
        if slopes_arr.ndim == 1:
            if not isinstance(y0, Real):
                raise TypeError("When 'slopes' is 1D, 'y0' must be a scalar float")
            pts = slopes_to_points(
                list(xp_arr.values), list(slopes_arr.values), float(y0)
            )
            return _sequence_to_array(pts)

        # Multi-dim case: per-entity slopes
        # Identify the entity dimension (not BREAKPOINT_DIM)
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
            y0_map = {
                str(k): float(y0.sel({entity_dim: k}).item()) for k in entity_keys
            }
        else:
            raise TypeError(
                f"'y0' must be a float, Series, DataArray, or dict, got {type(y0)}"
            )

        # Compute points per entity
        computed: dict[str, Sequence[float]] = {}
        for key in entity_keys:
            sk = str(key)
            sl = list(slopes_arr.sel({entity_dim: key}).values)
            # Remove trailing NaN from slopes
            sl = [v for v in sl if not np.isnan(v)]
            if entity_dim in xp_arr.dims:
                xp = list(xp_arr.sel({entity_dim: key}).values)
                xp = [v for v in xp if not np.isnan(v)]
            else:
                xp = [v for v in xp_arr.values if not np.isnan(v)]
            computed[sk] = slopes_to_points(xp, sl, y0_map[sk])

        return _dict_to_array(computed, entity_dim)

    # Points mode
    if values is None:
        raise ValueError("Must pass either 'values' or 'slopes'")

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


# ---------------------------------------------------------------------------
# Piecewise expression and descriptor types
# ---------------------------------------------------------------------------


class PiecewiseExpression:
    """
    Lazy descriptor representing a piecewise linear function of an expression.

    Created by :func:`piecewise`. Supports comparison operators so that
    ``piecewise(x, ...) >= y`` produces a
    :class:`PiecewiseConstraintDescriptor`.
    """

    __slots__ = ("active", "disjunctive", "expr", "x_points", "y_points")

    def __init__(
        self,
        expr: LinExprLike,
        x_points: DataArray,
        y_points: DataArray,
        disjunctive: bool,
        active: LinExprLike | None = None,
    ) -> None:
        self.expr = expr
        self.x_points = x_points
        self.y_points = y_points
        self.disjunctive = disjunctive
        self.active = active

    # y <= pw  →  Python tries y.__le__(pw) → NotImplemented → pw.__ge__(y)
    def __ge__(self, other: LinExprLike) -> PiecewiseConstraintDescriptor:
        return PiecewiseConstraintDescriptor(lhs=other, sign="<=", piecewise_func=self)

    # y >= pw  →  Python tries y.__ge__(pw) → NotImplemented → pw.__le__(y)
    def __le__(self, other: LinExprLike) -> PiecewiseConstraintDescriptor:
        return PiecewiseConstraintDescriptor(lhs=other, sign=">=", piecewise_func=self)

    # y == pw  →  Python tries y.__eq__(pw) → NotImplemented → pw.__eq__(y)
    def __eq__(self, other: object) -> PiecewiseConstraintDescriptor:  # type: ignore[override]
        from linopy.expressions import LinearExpression
        from linopy.variables import Variable

        if not isinstance(other, Variable | LinearExpression):
            return NotImplemented
        return PiecewiseConstraintDescriptor(lhs=other, sign="==", piecewise_func=self)


@dataclass
class PiecewiseConstraintDescriptor:
    """Holds all information needed to add a piecewise constraint to a model."""

    lhs: LinExprLike
    sign: str  # "<=", ">=", "=="
    piecewise_func: PiecewiseExpression


def _detect_disjunctive(x_points: DataArray, y_points: DataArray) -> bool:
    """
    Detect whether point arrays represent a disjunctive formulation.

    Both ``x_points`` and ``y_points`` **must** use the well-known dimension
    names ``BREAKPOINT_DIM`` and, for disjunctive formulations,
    ``SEGMENT_DIM``.  Use the :func:`breakpoints` / :func:`segments` factory
    helpers to build arrays with the correct dimension names.
    """
    x_has_bp = BREAKPOINT_DIM in x_points.dims
    y_has_bp = BREAKPOINT_DIM in y_points.dims
    if not x_has_bp and not y_has_bp:
        raise ValueError(
            "x_points and y_points must have a breakpoint dimension. "
            f"Got x_points dims {list(x_points.dims)} and y_points dims "
            f"{list(y_points.dims)}. Use the breakpoints() or segments() "
            f"factory to create correctly-dimensioned arrays."
        )
    if not x_has_bp:
        raise ValueError(
            "x_points is missing the breakpoint dimension, "
            f"got dims {list(x_points.dims)}. "
            "Use the breakpoints() or segments() factory."
        )
    if not y_has_bp:
        raise ValueError(
            "y_points is missing the breakpoint dimension, "
            f"got dims {list(y_points.dims)}. "
            "Use the breakpoints() or segments() factory."
        )

    x_has_seg = SEGMENT_DIM in x_points.dims
    y_has_seg = SEGMENT_DIM in y_points.dims
    if x_has_seg != y_has_seg:
        raise ValueError(
            "If one of x_points/y_points has a segment dimension, "
            f"both must. x_points dims: {list(x_points.dims)}, "
            f"y_points dims: {list(y_points.dims)}."
        )

    return x_has_seg


def piecewise(
    expr: LinExprLike,
    x_points: BreaksLike,
    y_points: BreaksLike,
    active: LinExprLike | None = None,
) -> PiecewiseExpression:
    """
    Create a piecewise linear function descriptor.

    Parameters
    ----------
    expr : Variable or LinearExpression
        The "x" side expression.
    x_points : BreaksLike
        Breakpoint x-coordinates.
    y_points : BreaksLike
        Breakpoint y-coordinates.
    active : Variable or LinearExpression, optional
        Binary variable that scales the piecewise function. When
        ``active=0``, all auxiliary variables are forced to zero, which
        in turn forces the reconstructed x and y to zero. When
        ``active=1``, the normal piecewise domain ``[x₀, xₙ]`` is
        active. This is the only behavior the linear formulation
        supports — selectively *relaxing* the constraint (letting x and
        y float freely when off) would require big-M or indicator
        constraints.

    Returns
    -------
    PiecewiseExpression
    """
    if not isinstance(x_points, DataArray):
        x_points = _coerce_breaks(x_points)
    if not isinstance(y_points, DataArray):
        y_points = _coerce_breaks(y_points)

    disjunctive = _detect_disjunctive(x_points, y_points)

    # Validate compatible shapes along breakpoint dimension
    if x_points.sizes[BREAKPOINT_DIM] != y_points.sizes[BREAKPOINT_DIM]:
        raise ValueError(
            f"x_points and y_points must have same size along '{BREAKPOINT_DIM}', "
            f"got {x_points.sizes[BREAKPOINT_DIM]} and "
            f"{y_points.sizes[BREAKPOINT_DIM]}"
        )

    # Validate compatible shapes along segment dimension
    if disjunctive:
        if x_points.sizes[SEGMENT_DIM] != y_points.sizes[SEGMENT_DIM]:
            raise ValueError(
                f"x_points and y_points must have same size along '{SEGMENT_DIM}'"
            )

    return PiecewiseExpression(expr, x_points, y_points, disjunctive, active)


# ---------------------------------------------------------------------------
# Internal validation and utility functions
# ---------------------------------------------------------------------------


def _validate_numeric_breakpoint_coords(bp: DataArray) -> None:
    if not pd.api.types.is_numeric_dtype(bp.coords[BREAKPOINT_DIM]):
        raise ValueError(
            f"Breakpoint dimension '{BREAKPOINT_DIM}' must have numeric coordinates "
            f"for SOS2 weights, but got {bp.coords[BREAKPOINT_DIM].dtype}"
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


def _check_strict_increasing(bp: DataArray) -> bool:
    """Check if breakpoints are strictly increasing along BREAKPOINT_DIM."""
    diffs = bp.diff(BREAKPOINT_DIM)
    pos = (diffs > 0) | diffs.isnull()
    has_non_nan = (~diffs.isnull()).any(BREAKPOINT_DIM)
    increasing = pos.all(BREAKPOINT_DIM) & has_non_nan
    return bool(increasing.all())


def _has_trailing_nan_only(bp: DataArray) -> bool:
    """Check that NaN values only appear as trailing entries along BREAKPOINT_DIM."""
    valid = ~bp.isnull()
    cummin = np.minimum.accumulate(valid.values, axis=valid.dims.index(BREAKPOINT_DIM))
    cummin_da = DataArray(cummin, coords=valid.coords, dims=valid.dims)
    return not bool((valid & ~cummin_da).any())


def _to_linexpr(expr: LinExprLike) -> LinearExpression:
    from linopy.expressions import LinearExpression

    if isinstance(expr, LinearExpression):
        return expr
    return expr.to_linexpr()


def _extra_coords(points: DataArray, *exclude_dims: str | None) -> list[pd.Index]:
    excluded = {d for d in exclude_dims if d is not None}
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

    target_dims: set[str] = set()
    for e in exprs:
        le = _to_linexpr(e)
        target_dims.update(str(d) for d in le.coord_dims)

    missing = target_dims - skip - {str(d) for d in points.dims}
    if not missing:
        return points

    expand_map: dict[str, list] = {}
    for d in missing:
        for e in exprs:
            le = _to_linexpr(e)
            if d in le.coords:
                expand_map[str(d)] = list(le.coords[d].values)
                break

    if expand_map:
        points = points.expand_dims(expand_map)
    return points


def _compute_combined_mask(
    x_points: DataArray,
    y_points: DataArray,
    skip_nan_check: bool,
) -> DataArray | None:
    if skip_nan_check:
        if bool(x_points.isnull().any()) or bool(y_points.isnull().any()):
            raise ValueError(
                "skip_nan_check=True but breakpoints contain NaN. "
                "Either remove NaN values or set skip_nan_check=False."
            )
        return None
    return ~(x_points.isnull() | y_points.isnull())


def _detect_convexity(
    x_points: DataArray,
    y_points: DataArray,
) -> Literal["convex", "concave", "linear", "mixed"]:
    """
    Detect convexity of the piecewise function.

    Requires strictly increasing x breakpoints and computes slopes and
    second differences in the given order.
    """
    if not _check_strict_increasing(x_points):
        raise ValueError(
            "Convexity detection requires strictly increasing x_points. "
            "Pass breakpoints in increasing x-order or use method='sos2'."
        )

    dx = x_points.diff(BREAKPOINT_DIM)
    dy = y_points.diff(BREAKPOINT_DIM)

    valid = ~(dx.isnull() | dy.isnull() | (dx == 0))
    slopes = dy / dx

    if slopes.sizes[BREAKPOINT_DIM] < 2:
        return "linear"

    slope_diffs = slopes.diff(BREAKPOINT_DIM)

    valid_diffs = valid.isel({BREAKPOINT_DIM: slice(None, -1)})
    valid_diffs_hi = valid.isel({BREAKPOINT_DIM: slice(1, None)})
    valid_diffs_combined = valid_diffs.values & valid_diffs_hi.values

    sd_values = slope_diffs.values
    if valid_diffs_combined.size == 0 or not valid_diffs_combined.any():
        return "linear"

    valid_sd = sd_values[valid_diffs_combined]
    all_nonneg = bool(np.all(valid_sd >= -1e-10))
    all_nonpos = bool(np.all(valid_sd <= 1e-10))

    if all_nonneg and all_nonpos:
        return "linear"
    if all_nonneg:
        return "convex"
    if all_nonpos:
        return "concave"
    return "mixed"


# ---------------------------------------------------------------------------
# Internal formulation functions
# ---------------------------------------------------------------------------


def _add_pwl_lp(
    model: Model,
    name: str,
    x_expr: LinearExpression,
    y_expr: LinearExpression,
    sign: str,
    x_points: DataArray,
    y_points: DataArray,
) -> Constraint:
    """Add pure LP tangent-line constraints."""
    dx = x_points.diff(BREAKPOINT_DIM)
    dy = y_points.diff(BREAKPOINT_DIM)
    slopes = dy / dx

    slopes = slopes.rename({BREAKPOINT_DIM: LP_SEG_DIM})
    n_seg = slopes.sizes[LP_SEG_DIM]
    slopes[LP_SEG_DIM] = np.arange(n_seg)

    x_base = x_points.isel({BREAKPOINT_DIM: slice(None, -1)})
    y_base = y_points.isel({BREAKPOINT_DIM: slice(None, -1)})
    x_base = x_base.rename({BREAKPOINT_DIM: LP_SEG_DIM})
    y_base = y_base.rename({BREAKPOINT_DIM: LP_SEG_DIM})
    x_base[LP_SEG_DIM] = np.arange(n_seg)
    y_base[LP_SEG_DIM] = np.arange(n_seg)

    rhs = y_base - slopes * x_base
    lhs = y_expr - slopes * x_expr

    if sign == "<=":
        con = model.add_constraints(lhs <= rhs, name=f"{name}{PWL_LP_SUFFIX}")
    else:
        con = model.add_constraints(lhs >= rhs, name=f"{name}{PWL_LP_SUFFIX}")

    # Domain bound constraints to keep x within [x_min, x_max]
    x_lo = x_points.min(dim=BREAKPOINT_DIM)
    x_hi = x_points.max(dim=BREAKPOINT_DIM)
    model.add_constraints(x_expr >= x_lo, name=f"{name}{PWL_LP_DOMAIN_SUFFIX}_lo")
    model.add_constraints(x_expr <= x_hi, name=f"{name}{PWL_LP_DOMAIN_SUFFIX}_hi")

    return con


def _add_pwl_sos2_core(
    model: Model,
    name: str,
    x_expr: LinearExpression,
    target_expr: LinearExpression,
    x_points: DataArray,
    y_points: DataArray,
    lambda_mask: DataArray | None,
    active: LinearExpression | None = None,
) -> Constraint:
    """
    Core SOS2 formulation linking x_expr and target_expr via breakpoints.

    Creates lambda variables, SOS2 constraint, convexity constraint,
    and linking constraints for both x and target.

    When ``active`` is provided, the convexity constraint becomes
    ``sum(lambda) == active`` instead of ``== 1``, forcing all lambda
    (and thus x, y) to zero when ``active=0``.
    """
    extra = _extra_coords(x_points, BREAKPOINT_DIM)
    lambda_coords = extra + [
        pd.Index(x_points.coords[BREAKPOINT_DIM].values, name=BREAKPOINT_DIM)
    ]

    lambda_name = f"{name}{PWL_LAMBDA_SUFFIX}"
    convex_name = f"{name}{PWL_CONVEX_SUFFIX}"
    x_link_name = f"{name}{PWL_X_LINK_SUFFIX}"
    y_link_name = f"{name}{PWL_Y_LINK_SUFFIX}"

    lambda_var = model.add_variables(
        lower=0, upper=1, coords=lambda_coords, name=lambda_name, mask=lambda_mask
    )

    model.add_sos_constraints(lambda_var, sos_type=2, sos_dim=BREAKPOINT_DIM)

    # Convexity constraint: sum(lambda) == 1 or sum(lambda) == active
    rhs = active if active is not None else 1
    convex_con = model.add_constraints(
        lambda_var.sum(dim=BREAKPOINT_DIM) == rhs, name=convex_name
    )

    x_weighted = (lambda_var * x_points).sum(dim=BREAKPOINT_DIM)
    model.add_constraints(x_expr == x_weighted, name=x_link_name)

    y_weighted = (lambda_var * y_points).sum(dim=BREAKPOINT_DIM)
    model.add_constraints(target_expr == y_weighted, name=y_link_name)

    return convex_con


def _add_pwl_incremental_core(
    model: Model,
    name: str,
    x_expr: LinearExpression,
    target_expr: LinearExpression,
    x_points: DataArray,
    y_points: DataArray,
    bp_mask: DataArray | None,
    active: LinearExpression | None = None,
) -> Constraint:
    """
    Core incremental formulation linking x_expr and target_expr.

    Creates delta variables, fill-order constraints, and x/target link constraints.

    When ``active`` is provided, delta bounds are tightened to
    ``δ_i ≤ active`` and base terms become ``x₀ * active``,
    ``y₀ * active``, forcing x and y to zero when ``active=0``.
    """
    delta_name = f"{name}{PWL_DELTA_SUFFIX}"
    fill_name = f"{name}{PWL_FILL_SUFFIX}"
    x_link_name = f"{name}{PWL_X_LINK_SUFFIX}"
    y_link_name = f"{name}{PWL_Y_LINK_SUFFIX}"

    n_segments = x_points.sizes[BREAKPOINT_DIM] - 1
    seg_index = pd.Index(range(n_segments), name=LP_SEG_DIM)
    extra = _extra_coords(x_points, BREAKPOINT_DIM)
    delta_coords = extra + [seg_index]

    x_steps = x_points.diff(BREAKPOINT_DIM).rename({BREAKPOINT_DIM: LP_SEG_DIM})
    x_steps[LP_SEG_DIM] = seg_index
    y_steps = y_points.diff(BREAKPOINT_DIM).rename({BREAKPOINT_DIM: LP_SEG_DIM})
    y_steps[LP_SEG_DIM] = seg_index

    if bp_mask is not None:
        mask_lo = bp_mask.isel({BREAKPOINT_DIM: slice(None, -1)}).rename(
            {BREAKPOINT_DIM: LP_SEG_DIM}
        )
        mask_hi = bp_mask.isel({BREAKPOINT_DIM: slice(1, None)}).rename(
            {BREAKPOINT_DIM: LP_SEG_DIM}
        )
        mask_lo[LP_SEG_DIM] = seg_index
        mask_hi[LP_SEG_DIM] = seg_index
        delta_mask: DataArray | None = mask_lo & mask_hi
    else:
        delta_mask = None

    # When active is provided, upper bound is active (binary) instead of 1
    delta_upper = 1
    delta_var = model.add_variables(
        lower=0,
        upper=delta_upper,
        coords=delta_coords,
        name=delta_name,
        mask=delta_mask,
    )

    if active is not None:
        # Tighten delta bounds: δ_i ≤ active
        active_bound_name = f"{name}_active_bound"
        model.add_constraints(delta_var <= active, name=active_bound_name)

    # Binary indicator variables: y_i for each segment
    inc_binary_name = f"{name}{PWL_INC_BINARY_SUFFIX}"
    inc_link_name = f"{name}{PWL_INC_LINK_SUFFIX}"
    inc_order_name = f"{name}{PWL_INC_ORDER_SUFFIX}"

    binary_var = model.add_variables(
        binary=True, coords=delta_coords, name=inc_binary_name, mask=delta_mask
    )

    # Link constraints: δ_i ≤ y_i for all segments
    model.add_constraints(delta_var <= binary_var, name=inc_link_name)

    # Order constraints: y_{i+1} ≤ δ_i for i = 0..n-2
    fill_con: Constraint | None = None
    if n_segments >= 2:
        delta_lo = delta_var.isel({LP_SEG_DIM: slice(None, -1)}, drop=True)
        delta_hi = delta_var.isel({LP_SEG_DIM: slice(1, None)}, drop=True)
        # Keep existing fill constraint as LP relaxation tightener
        fill_con = model.add_constraints(delta_hi <= delta_lo, name=fill_name)

        binary_hi = binary_var.isel({LP_SEG_DIM: slice(1, None)}, drop=True)
        model.add_constraints(binary_hi <= delta_lo, name=inc_order_name)

    x0 = x_points.isel({BREAKPOINT_DIM: 0})
    y0 = y_points.isel({BREAKPOINT_DIM: 0})

    # When active is provided, multiply base terms by active
    x_base: DataArray | LinearExpression = x0
    y_base: DataArray | LinearExpression = y0
    if active is not None:
        x_base = x0 * active
        y_base = y0 * active

    x_weighted = (delta_var * x_steps).sum(dim=LP_SEG_DIM) + x_base
    model.add_constraints(x_expr == x_weighted, name=x_link_name)

    y_weighted = (delta_var * y_steps).sum(dim=LP_SEG_DIM) + y_base
    model.add_constraints(target_expr == y_weighted, name=y_link_name)

    return fill_con if fill_con is not None else model.constraints[y_link_name]


def _add_dpwl_sos2_core(
    model: Model,
    name: str,
    x_expr: LinearExpression,
    target_expr: LinearExpression,
    x_points: DataArray,
    y_points: DataArray,
    lambda_mask: DataArray | None,
    active: LinearExpression | None = None,
) -> Constraint:
    """
    Core disjunctive SOS2 formulation with separate x/y points.

    When ``active`` is provided, the segment selection becomes
    ``sum(z_k) == active`` instead of ``== 1``, forcing all segment
    binaries, lambdas, and thus x and y to zero when ``active=0``.
    """
    binary_name = f"{name}{PWL_BINARY_SUFFIX}"
    select_name = f"{name}{PWL_SELECT_SUFFIX}"
    lambda_name = f"{name}{PWL_LAMBDA_SUFFIX}"
    convex_name = f"{name}{PWL_CONVEX_SUFFIX}"
    x_link_name = f"{name}{PWL_X_LINK_SUFFIX}"
    y_link_name = f"{name}{PWL_Y_LINK_SUFFIX}"

    extra = _extra_coords(x_points, BREAKPOINT_DIM, SEGMENT_DIM)
    lambda_coords = extra + [
        pd.Index(x_points.coords[SEGMENT_DIM].values, name=SEGMENT_DIM),
        pd.Index(x_points.coords[BREAKPOINT_DIM].values, name=BREAKPOINT_DIM),
    ]
    binary_coords = extra + [
        pd.Index(x_points.coords[SEGMENT_DIM].values, name=SEGMENT_DIM),
    ]

    binary_mask = (
        lambda_mask.any(dim=BREAKPOINT_DIM) if lambda_mask is not None else None
    )

    binary_var = model.add_variables(
        binary=True, coords=binary_coords, name=binary_name, mask=binary_mask
    )

    # Segment selection: sum(z_k) == 1 or sum(z_k) == active
    rhs = active if active is not None else 1
    select_con = model.add_constraints(
        binary_var.sum(dim=SEGMENT_DIM) == rhs, name=select_name
    )

    lambda_var = model.add_variables(
        lower=0, upper=1, coords=lambda_coords, name=lambda_name, mask=lambda_mask
    )

    model.add_sos_constraints(lambda_var, sos_type=2, sos_dim=BREAKPOINT_DIM)

    model.add_constraints(
        lambda_var.sum(dim=BREAKPOINT_DIM) == binary_var, name=convex_name
    )

    x_weighted = (lambda_var * x_points).sum(dim=[SEGMENT_DIM, BREAKPOINT_DIM])
    model.add_constraints(x_expr == x_weighted, name=x_link_name)

    y_weighted = (lambda_var * y_points).sum(dim=[SEGMENT_DIM, BREAKPOINT_DIM])
    model.add_constraints(target_expr == y_weighted, name=y_link_name)

    return select_con


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def add_piecewise_constraints(
    model: Model,
    descriptor: PiecewiseConstraintDescriptor | Constraint,
    method: Literal["sos2", "incremental", "auto", "lp"] = "auto",
    name: str | None = None,
    skip_nan_check: bool = False,
) -> Constraint:
    """
    Add a piecewise linear constraint from a :class:`PiecewiseConstraintDescriptor`.

    Typically called as::

        m.add_piecewise_constraints(piecewise(x, x_points, y_points) >= y)

    Parameters
    ----------
    model : Model
        The linopy model.
    descriptor : PiecewiseConstraintDescriptor
        Created by comparing a variable/expression with a :class:`PiecewiseExpression`.
    method : {"auto", "sos2", "incremental", "lp"}, default "auto"
        Formulation method.
    name : str, optional
        Base name for generated variables/constraints.
    skip_nan_check : bool, default False
        If True, skip NaN detection.

    Returns
    -------
    Constraint
    """
    if not isinstance(descriptor, PiecewiseConstraintDescriptor):
        raise TypeError(
            f"Expected PiecewiseConstraintDescriptor, got {type(descriptor)}. "
            f"Use: m.add_piecewise_constraints(piecewise(x, x_points, y_points) >= y)"
        )

    if method not in ("sos2", "incremental", "auto", "lp"):
        raise ValueError(
            f"method must be 'sos2', 'incremental', 'auto', or 'lp', got '{method}'"
        )

    pw = descriptor.piecewise_func
    sign = descriptor.sign
    y_lhs = descriptor.lhs
    x_expr_raw = pw.expr
    x_points = pw.x_points
    y_points = pw.y_points
    disjunctive = pw.disjunctive
    active = pw.active

    # Broadcast points to match expression dimensions
    x_points = _broadcast_points(x_points, x_expr_raw, y_lhs, disjunctive=disjunctive)
    y_points = _broadcast_points(y_points, x_expr_raw, y_lhs, disjunctive=disjunctive)

    # Compute mask
    mask = _compute_combined_mask(x_points, y_points, skip_nan_check)

    # Name
    if name is None:
        name = f"pwl{model._pwlCounter}"
        model._pwlCounter += 1

    # Convert to LinearExpressions
    x_expr = _to_linexpr(x_expr_raw)
    y_expr = _to_linexpr(y_lhs)

    # Convert active to LinearExpression if provided
    active_expr = _to_linexpr(active) if active is not None else None

    # Validate: active is not supported with LP method
    if active_expr is not None and method == "lp":
        raise ValueError(
            "The 'active' parameter is not supported with method='lp'. "
            "Use method='incremental' or method='sos2'."
        )

    if disjunctive:
        return _add_disjunctive(
            model,
            name,
            x_expr,
            y_expr,
            sign,
            x_points,
            y_points,
            mask,
            method,
            active_expr,
        )
    else:
        return _add_continuous(
            model,
            name,
            x_expr,
            y_expr,
            sign,
            x_points,
            y_points,
            mask,
            method,
            skip_nan_check,
            active_expr,
        )


def _add_continuous(
    model: Model,
    name: str,
    x_expr: LinearExpression,
    y_expr: LinearExpression,
    sign: str,
    x_points: DataArray,
    y_points: DataArray,
    mask: DataArray | None,
    method: str,
    skip_nan_check: bool,
    active: LinearExpression | None = None,
) -> Constraint:
    """Handle continuous (non-disjunctive) piecewise constraints."""
    convexity: Literal["convex", "concave", "linear", "mixed"] | None = None

    # Determine actual method
    if method == "auto":
        if sign == "==":
            if _check_strict_monotonicity(x_points) and _has_trailing_nan_only(
                x_points
            ):
                method = "incremental"
            else:
                method = "sos2"
        else:
            if not _check_strict_increasing(x_points):
                raise ValueError(
                    "Automatic method selection for piecewise inequalities requires "
                    "strictly increasing x_points. Pass breakpoints in increasing "
                    "x-order or use method='sos2'."
                )
            convexity = _detect_convexity(x_points, y_points)
            if convexity == "linear":
                method = "lp"
            elif (sign == "<=" and convexity == "concave") or (
                sign == ">=" and convexity == "convex"
            ):
                method = "lp"
            else:
                method = "sos2"
    elif method == "lp":
        if sign == "==":
            raise ValueError("Pure LP method is not supported for equality constraints")
        convexity = _detect_convexity(x_points, y_points)
        if convexity != "linear":
            if sign == "<=" and convexity != "concave":
                raise ValueError(
                    f"Pure LP method for '<=' requires concave or linear function, "
                    f"got {convexity}"
                )
            if sign == ">=" and convexity != "convex":
                raise ValueError(
                    f"Pure LP method for '>=' requires convex or linear function, "
                    f"got {convexity}"
                )
    elif method == "incremental":
        if not _check_strict_monotonicity(x_points):
            raise ValueError("Incremental method requires strictly monotonic x_points")
        if not _has_trailing_nan_only(x_points):
            raise ValueError(
                "Incremental method does not support non-trailing NaN breakpoints. "
                "NaN values must only appear at the end of the breakpoint sequence."
            )

    if method == "sos2":
        _validate_numeric_breakpoint_coords(x_points)
        if not _has_trailing_nan_only(x_points):
            raise ValueError(
                "SOS2 method does not support non-trailing NaN breakpoints. "
                "NaN values must only appear at the end of the breakpoint sequence."
            )

    # LP formulation
    if method == "lp":
        if active is not None:
            raise ValueError(
                "The 'active' parameter is not supported with method='lp'. "
                "Use method='incremental' or method='sos2'."
            )
        return _add_pwl_lp(model, name, x_expr, y_expr, sign, x_points, y_points)

    # SOS2 or incremental formulation
    if sign == "==":
        # Direct linking: y = f(x)
        if method == "sos2":
            return _add_pwl_sos2_core(
                model, name, x_expr, y_expr, x_points, y_points, mask, active
            )
        else:  # incremental
            return _add_pwl_incremental_core(
                model, name, x_expr, y_expr, x_points, y_points, mask, active
            )
    else:
        # Inequality: create aux variable z, enforce z = f(x), then y <= z or y >= z
        aux_name = f"{name}{PWL_AUX_SUFFIX}"
        aux_coords = _extra_coords(x_points, BREAKPOINT_DIM)
        z = model.add_variables(coords=aux_coords, name=aux_name)
        z_expr = _to_linexpr(z)

        if method == "sos2":
            result = _add_pwl_sos2_core(
                model, name, x_expr, z_expr, x_points, y_points, mask, active
            )
        else:  # incremental
            result = _add_pwl_incremental_core(
                model, name, x_expr, z_expr, x_points, y_points, mask, active
            )

        # Add inequality
        ineq_name = f"{name}_ineq"
        if sign == "<=":
            model.add_constraints(y_expr <= z_expr, name=ineq_name)
        else:
            model.add_constraints(y_expr >= z_expr, name=ineq_name)

        return result


def _add_disjunctive(
    model: Model,
    name: str,
    x_expr: LinearExpression,
    y_expr: LinearExpression,
    sign: str,
    x_points: DataArray,
    y_points: DataArray,
    mask: DataArray | None,
    method: str,
    active: LinearExpression | None = None,
) -> Constraint:
    """Handle disjunctive piecewise constraints."""
    if method == "lp":
        raise ValueError("Pure LP method is not supported for disjunctive constraints")
    if method == "incremental":
        raise ValueError(
            "Incremental method is not supported for disjunctive constraints"
        )

    _validate_numeric_breakpoint_coords(x_points)
    if not _has_trailing_nan_only(x_points):
        raise ValueError(
            "Disjunctive SOS2 does not support non-trailing NaN breakpoints. "
            "NaN values must only appear at the end of the breakpoint sequence."
        )

    if sign == "==":
        return _add_dpwl_sos2_core(
            model, name, x_expr, y_expr, x_points, y_points, mask, active
        )
    else:
        # Create aux variable z, disjunctive SOS2 for z = f(x), then y <= z or y >= z
        aux_name = f"{name}{PWL_AUX_SUFFIX}"
        aux_coords = _extra_coords(x_points, BREAKPOINT_DIM, SEGMENT_DIM)
        z = model.add_variables(coords=aux_coords, name=aux_name)
        z_expr = _to_linexpr(z)

        result = _add_dpwl_sos2_core(
            model, name, x_expr, z_expr, x_points, y_points, mask, active
        )

        ineq_name = f"{name}_ineq"
        if sign == "<=":
            model.add_constraints(y_expr <= z_expr, name=ineq_name)
        else:
            model.add_constraints(y_expr >= z_expr, name=ineq_name)

        return result
