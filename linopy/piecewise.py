"""
Piecewise linear constraint formulations.

Provides SOS2, incremental, pure LP, and disjunctive piecewise linear
constraint methods for use with linopy.Model.
"""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real
from typing import TYPE_CHECKING, Literal, TypeAlias

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
    LP_SEG_DIM,
    PWL_ACTIVE_BOUND_SUFFIX,
    PWL_BINARY_ORDER_SUFFIX,
    PWL_CONVEX_SUFFIX,
    PWL_DELTA_BOUND_SUFFIX,
    PWL_DELTA_SUFFIX,
    PWL_DOMAIN_HI_SUFFIX,
    PWL_DOMAIN_LO_SUFFIX,
    PWL_FILL_ORDER_SUFFIX,
    PWL_LAMBDA_SUFFIX,
    PWL_LINK_SUFFIX,
    PWL_LP_SUFFIX,
    PWL_ORDER_BINARY_SUFFIX,
    PWL_OUTPUT_LINK_SUFFIX,
    PWL_SEGMENT_BINARY_SUFFIX,
    PWL_SELECT_SUFFIX,
    SEGMENT_DIM,
    SIGNS,
    sign_replace_dict,
)

if TYPE_CHECKING:
    from linopy.constraints import Constraint, Constraints
    from linopy.expressions import LinearExpression
    from linopy.model import Model
    from linopy.types import LinExprLike
    from linopy.variables import Variables

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
# Result type
# ---------------------------------------------------------------------------


class PiecewiseFormulation:
    """
    Result of ``add_piecewise_formulation``.

    Groups all auxiliary variables and constraints created by a single
    piecewise formulation. Stores only names internally; ``variables``
    and ``constraints`` properties return live views from the model.
    """

    __slots__ = ("name", "method", "variable_names", "constraint_names", "_model")

    def __init__(
        self,
        name: str,
        method: str,
        variable_names: list[str],
        constraint_names: list[str],
        model: Model,
    ) -> None:
        self.name = name
        self.method = method
        self.variable_names = variable_names
        self.constraint_names = constraint_names
        self._model = model

    @property
    def variables(self) -> Variables:
        """View of the auxiliary variables in this formulation."""
        return self._model.variables[self.variable_names]

    @property
    def constraints(self) -> Constraints:
        """View of the auxiliary constraints in this formulation."""
        return self._model.constraints[self.constraint_names]

    def __repr__(self) -> str:
        # Collect user-facing dims with sizes (skip internal _ prefixed dims)
        user_dims: dict[str, int] = {}
        for var in self.variables.data.values():
            for d in var.coords:
                ds = str(d)
                if not ds.startswith("_") and ds not in user_dims:
                    user_dims[ds] = var.data.sizes[d]
        dims_str = ", ".join(f"{d}: {s}" for d, s in user_dims.items())
        header = f"PiecewiseFormulation `{self.name}`"
        if dims_str:
            header += f" [{dims_str}]"
        r = f"{header} — {self.method}\n"
        r += "  Variables:\n"
        for vname, var in self.variables.items():
            dims = ", ".join(str(d) for d in var.coords) if var.coords else ""
            r += f"    * {vname} ({dims})\n" if dims else f"    * {vname}\n"
        r += "  Constraints:\n"
        for cname, con in self.constraints.items():
            dims = ", ".join(str(d) for d in con.coords) if con.coords else ""
            r += f"    * {cname} ({dims})\n" if dims else f"    * {cname}\n"
        return r


# ---------------------------------------------------------------------------
# DataArray construction helpers
# ---------------------------------------------------------------------------


def _strip_nan(vals: Sequence[float] | np.ndarray) -> list[float]:
    """Remove NaN values from a sequence."""
    return [v for v in vals if not np.isnan(v)]


def _rename_to_segments(da: DataArray, seg_index: np.ndarray) -> DataArray:
    """Rename breakpoint dim to segment dim and reassign coordinates."""
    da = da.rename({BREAKPOINT_DIM: LP_SEG_DIM})
    da[LP_SEG_DIM] = seg_index
    return da


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
    y0: float | dict[str, float] | pd.Series | DataArray,
    dim: str | None,
) -> DataArray:
    """Convert slopes + x_points + y0 into a breakpoint DataArray."""
    slopes_arr = _coerce_breaks(slopes, dim)
    xp_arr = _coerce_breaks(x_points, dim)

    # 1D case: single set of breakpoints
    if slopes_arr.ndim == 1:
        if not isinstance(y0, Real):
            raise TypeError("When 'slopes' is 1D, 'y0' must be a scalar float")
        pts = slopes_to_points(list(xp_arr.values), list(slopes_arr.values), float(y0))
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
        computed[sk] = slopes_to_points(xp, sl, y0_map[sk])

    return _dict_to_array(computed, entity_dim)


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
        return _breakpoints_from_slopes(slopes, x_points, y0, dim)

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


def tangent_lines(
    x: LinExprLike,
    x_points: BreaksLike,
    y_points: BreaksLike,
) -> LinearExpression:
    r"""
    Compute tangent-line expressions for a piecewise linear function.

    Returns a :class:`~linopy.expressions.LinearExpression` with an extra
    segment dimension.  Each element along the segment dimension is the
    tangent line of one segment: :math:`m_k \cdot x + c_k`.

    Use the result in a regular constraint to create an upper or lower
    bound:

    .. code-block:: python

        t = tangent_lines(power, x_pts, y_pts)
        m.add_constraints(fuel <= t)  # upper bound (concave f)
        m.add_constraints(fuel >= t)  # lower bound (convex f)

    No auxiliary variables are created — the result is purely linear.

    Parameters
    ----------
    x : Variable or LinearExpression
        The input expression.
    x_points : BreaksLike
        Breakpoint x-coordinates (must be strictly increasing).
    y_points : BreaksLike
        Breakpoint y-coordinates.

    Returns
    -------
    LinearExpression
        Expression with an additional ``_breakpoint_seg`` dimension
        (one entry per segment).
    """
    from linopy.expressions import LinearExpression as LinExpr
    from linopy.variables import Variable

    x_points = _coerce_breaks(x_points)
    y_points = _coerce_breaks(y_points)

    dx = x_points.diff(BREAKPOINT_DIM)
    dy = y_points.diff(BREAKPOINT_DIM)
    seg_index = np.arange(dx.sizes[BREAKPOINT_DIM])

    slopes = _rename_to_segments(dy / dx, seg_index)
    x_base = _rename_to_segments(
        x_points.isel({BREAKPOINT_DIM: slice(None, -1)}), seg_index
    )
    y_base = _rename_to_segments(
        y_points.isel({BREAKPOINT_DIM: slice(None, -1)}), seg_index
    )

    intercepts = y_base - slopes * x_base

    if not isinstance(x, Variable | LinExpr):
        raise TypeError(f"x must be a Variable or LinearExpression, got {type(x)}")

    return slopes * _to_linexpr(x) + intercepts


# ---------------------------------------------------------------------------
# Internal validation and utility functions
# ---------------------------------------------------------------------------


def _validate_breakpoint_shapes(bp_a: DataArray, bp_b: DataArray) -> bool:
    """
    Validate that two breakpoint arrays have compatible shapes.

    Returns whether the formulation is disjunctive (has segment dimension).
    """
    if BREAKPOINT_DIM not in bp_a.dims:
        raise ValueError(
            f"Breakpoints are missing the '{BREAKPOINT_DIM}' dimension, "
            f"got dims {list(bp_a.dims)}. "
            "Use the breakpoints() or segments() factory."
        )
    if BREAKPOINT_DIM not in bp_b.dims:
        raise ValueError(
            f"Breakpoints are missing the '{BREAKPOINT_DIM}' dimension, "
            f"got dims {list(bp_b.dims)}. "
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


def _detect_convexity(
    x_points: DataArray, y_points: DataArray
) -> Literal["convex", "concave", "linear", "mixed"]:
    """
    Classify the shape of a piecewise function from its breakpoints.

    Requires strictly increasing ``x_points``. Computes segment slopes and
    checks the sign of second differences.
    """
    dx = x_points.diff(BREAKPOINT_DIM)
    dy = y_points.diff(BREAKPOINT_DIM)
    valid = ~(dx.isnull() | dy.isnull() | (dx == 0))
    slopes = dy / dx

    if slopes.sizes[BREAKPOINT_DIM] < 2:
        return "linear"

    slope_diffs = slopes.diff(BREAKPOINT_DIM)
    valid_lo = valid.isel({BREAKPOINT_DIM: slice(None, -1)})
    valid_hi = valid.isel({BREAKPOINT_DIM: slice(1, None)})
    valid_diffs = valid_lo.values & valid_hi.values
    sd_values = slope_diffs.values
    if valid_diffs.size == 0 or not valid_diffs.any():
        return "linear"
    valid_sd = sd_values[valid_diffs]
    tol = 1e-10
    all_nonneg = bool(np.all(valid_sd >= -tol))
    all_nonpos = bool(np.all(valid_sd <= tol))
    if all_nonneg and all_nonpos:
        return "linear"
    if all_nonneg:
        return "convex"
    if all_nonpos:
        return "concave"
    return "mixed"


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

    target_dims: set[str] = set()
    for le in lin_exprs:
        target_dims.update(str(d) for d in le.coord_dims)

    missing = target_dims - skip - {str(d) for d in points.dims}
    if not missing:
        return points

    expand_map: dict[str, list] = {}
    for d in missing:
        for le in lin_exprs:
            if d in le.coords:
                expand_map[str(d)] = list(le.coords[d].values)
                break

    if expand_map:
        points = points.expand_dims(expand_map)
    return points


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def add_piecewise_formulation(
    model: Model,
    *pairs: tuple[LinExprLike, BreaksLike],
    sign: Literal["==", "<=", ">="] = "==",
    method: Literal["sos2", "incremental", "lp", "auto"] = "auto",
    active: LinExprLike | None = None,
    name: str | None = None,
) -> PiecewiseFormulation:
    r"""
    Add piecewise linear constraints.

    Each positional argument is a ``(expression, breakpoints)`` tuple.
    All expressions are linked through shared interpolation weights so
    that every operating point lies on the same segment of the piecewise
    curve.

    Example — 2 variables::

        m.add_piecewise_formulation(
            (power, [0, 30, 60, 100]),
            (fuel,  [0, 36, 84, 170]),
        )

    Example — 3 variables (CHP plant)::

        m.add_piecewise_formulation(
            (power, [0, 30, 60, 100]),
            (fuel,  [0, 40, 85, 160]),
            (heat,  [0, 25, 55, 95]),
        )

    **Sign — inequality bounds:**

    The ``sign`` parameter follows the *first-tuple convention*:

    - ``sign="=="`` (default): all expressions must lie exactly on the
      piecewise curve (joint equality).
    - ``sign="<="``: the **first** tuple's expression is **bounded above**
      by its interpolated value; all other tuples are forced to equality
      (inputs on the curve).  Reads as *"first expression ≤ f(the rest)"*.
    - ``sign=">="``: same but the first is bounded **below**.

    For 2-variable inequality on convex/concave curves, ``method="auto"``
    automatically selects a pure-LP tangent-line formulation (no auxiliary
    variables).  Non-convex curves fall back to SOS2/incremental with the
    sign applied to the first tuple's link constraint.

    Example — ``fuel ≤ f(power)`` on a concave curve::

        m.add_piecewise_formulation(
            (fuel,  y_pts),    # bounded output, listed first
            (power, x_pts),    # input, always equality
            sign="<=",
        )

    Parameters
    ----------
    *pairs : tuple of (expression, breakpoints)
        Each pair links an expression (Variable or LinearExpression) to
        its breakpoint values.  At least two pairs are required.  With
        ``sign != EQUAL`` the **first** pair is the bounded output; all
        later pairs are treated as inputs forced to equality.
    sign : {"==", "<=", ">="}, default "=="
        Constraint sign applied to the *last* tuple's link constraint.
        Earlier tuples always use equality.  See description above.
    method : {"auto", "sos2", "incremental", "lp"}, default "auto"
        Formulation method.
        ``"lp"`` uses tangent lines (pure LP, no variables) and requires
        ``sign != EQUAL`` plus a matching-convexity curve with exactly
        two tuples.
        ``"auto"`` picks ``"lp"`` when applicable, otherwise
        ``"incremental"`` (monotonic breakpoints) or ``"sos2"``.
    active : Variable or LinearExpression, optional
        Binary variable that gates the piecewise function.  When
        ``active=0``, all auxiliary variables are forced to zero.
        Not supported with ``method="lp"``.
    name : str, optional
        Base name for generated variables/constraints.

    Returns
    -------
    PiecewiseFormulation
    """
    # Normalize sign (accept "==" or "=" for equality, etc.)
    sign = sign_replace_dict.get(sign, sign)
    if sign not in SIGNS:
        raise ValueError(f"sign must be one of {sorted(SIGNS)}, got '{sign}'")
    if method not in ("sos2", "incremental", "auto", "lp"):
        raise ValueError(
            f"method must be 'sos2', 'incremental', 'auto', or 'lp', "
            f"got '{method}'"
        )
    if method == "lp" and sign == EQUAL:
        raise ValueError(
            "method='lp' requires sign='<=' or '>='. "
            "LP tangent lines cannot enforce equality on a piecewise curve."
        )

    if len(pairs) < 2:
        raise TypeError(
            "add_piecewise_formulation() requires at least 2 "
            "(expression, breakpoints) pairs."
        )

    for i, pair in enumerate(pairs):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(
                f"Argument {i + 1} must be a (expression, breakpoints) tuple, "
                f"got {type(pair)}."
            )

    # Coerce all breakpoints.  Drop scalar coordinates (e.g. left over
    # from bp.sel(var="power")) so they don't conflict when stacking.
    coerced: list[tuple[LinExprLike, DataArray]] = []
    for expr, bp in pairs:
        if not isinstance(bp, DataArray):
            bp = _coerce_breaks(bp)
        scalar_coords = [c for c in bp.coords if c not in bp.dims]
        if scalar_coords:
            bp = bp.drop_vars(scalar_coords)
        coerced.append((expr, bp))

    # Check for disjunctive (segment dimension) on first pair
    first_bp = coerced[0][1]
    disjunctive = SEGMENT_DIM in first_bp.dims

    # Validate all breakpoint pairs have compatible shapes.
    # Checking each against the first is sufficient since the shape checks are transitive.
    for i in range(1, len(coerced)):
        _validate_breakpoint_shapes(first_bp, coerced[i][1])

    # Broadcast all breakpoints to match all expression dimensions
    all_exprs = [expr for expr, _ in coerced]
    bp_list = [
        _broadcast_points(bp, *all_exprs, disjunctive=disjunctive) for _, bp in coerced
    ]

    # Compute combined mask from all breakpoints
    combined_null = bp_list[0].isnull()
    for bp in bp_list[1:]:
        combined_null = combined_null | bp.isnull()
    bp_mask = ~combined_null if bool(combined_null.any()) else None

    # Name
    if name is None:
        name = f"pwl{model._pwlCounter}"
        model._pwlCounter += 1

    # Build link dimension coordinates from variable names
    from linopy.variables import Variable

    link_coords: list[str] = []
    for i, expr in enumerate(all_exprs):
        if isinstance(expr, Variable) and expr.name:
            link_coords.append(expr.name)
        else:
            link_coords.append(str(i))

    # Convert expressions to LinearExpressions
    lin_exprs = [_to_linexpr(expr) for expr in all_exprs]
    active_expr = _to_linexpr(active) if active is not None else None

    # Snapshot existing names to detect what the formulation adds
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
        _add_disjunctive(
            model,
            name,
            lin_exprs,
            bp_list,
            link_coords,
            bp_mask,
            sign,
            active_expr,
        )
        resolved_method = "sos2"
    else:
        # Continuous: stack into N-variable formulation
        resolved_method = _add_continuous(
            model,
            name,
            lin_exprs,
            bp_list,
            link_coords,
            bp_mask,
            method,
            sign,
            active_expr,
        )

    # Collect newly created variable and constraint names
    new_vars = [n for n in model.variables if n not in vars_before]
    new_cons = [n for n in model.constraints if n not in cons_before]

    result = PiecewiseFormulation(
        name=name,
        method=resolved_method,
        variable_names=new_vars,
        constraint_names=new_cons,
        model=model,
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


def _add_continuous(
    model: Model,
    name: str,
    lin_exprs: list[LinearExpression],
    bp_list: list[DataArray],
    link_coords: list[str],
    bp_mask: DataArray | None,
    method: str,
    sign: str,
    active: LinearExpression | None = None,
) -> str:
    """
    Dispatch continuous piecewise constraints.

    Returns the resolved method name ("lp", "sos2", or "incremental").
    """
    from linopy.expressions import LinearExpression

    link_dim = "_pwl_var"

    # First tuple is the "output" (bounded by sign), second is the input.
    # For LP dispatch we need 2 tuples and a convex/concave match.

    # Auto-dispatch: try LP for 2-var inequality with matching convexity
    if method == "auto" and sign != EQUAL:
        if len(lin_exprs) == 2:
            y_expr, x_expr = lin_exprs[0], lin_exprs[1]
            y_pts, x_pts = bp_list[0], bp_list[1]
            if _check_strict_monotonicity(x_pts) and _has_trailing_nan_only(x_pts):
                convexity = _detect_convexity(x_pts, y_pts)
                if (sign == LESS_EQUAL and convexity in ("concave", "linear")) or (
                    sign == GREATER_EQUAL and convexity in ("convex", "linear")
                ):
                    if active is None:
                        _add_lp(
                            model, name, x_expr, y_expr,
                            x_pts, y_pts, sign,
                        )
                        return "lp"

    # Explicit LP method
    if method == "lp":
        if len(lin_exprs) != 2:
            raise ValueError(
                "method='lp' requires exactly 2 (expression, breakpoints) pairs."
            )
        if active is not None:
            raise ValueError("method='lp' is not compatible with active=...")
        y_expr, x_expr = lin_exprs[0], lin_exprs[1]
        y_pts, x_pts = bp_list[0], bp_list[1]
        if not _check_strict_monotonicity(x_pts):
            raise ValueError(
                "method='lp' requires strictly monotonic x breakpoints."
            )
        convexity = _detect_convexity(x_pts, y_pts)
        _hint = (
            "Use method='auto' to automatically fall back to SOS2/incremental "
            "(which give an exact bound). If you specifically want the chord "
            "expressions (e.g. for a custom outer approximation), call "
            "linopy.tangent_lines() directly."
        )
        if sign == LESS_EQUAL and convexity not in ("concave", "linear"):
            raise ValueError(
                f"method='lp' with sign='<=' is only valid for a concave or "
                f"linear curve — tangent lines on a '{convexity}' curve "
                f"produce a feasible region that is a strict subset of the "
                f"true hypograph (i.e. wrong, not merely loose). " + _hint
            )
        if sign == GREATER_EQUAL and convexity not in ("convex", "linear"):
            raise ValueError(
                f"method='lp' with sign='>=' is only valid for a convex or "
                f"linear curve — tangent lines on a '{convexity}' curve "
                f"produce a feasible region that is a strict subset of the "
                f"true epigraph (i.e. wrong, not merely loose). " + _hint
            )
        _add_lp(model, name, x_expr, y_expr, x_pts, y_pts, sign)
        return "lp"

    stacked_bp = _stack_along_link(bp_list, link_coords, link_dim)
    trailing_nan_only = _has_trailing_nan_only(stacked_bp)

    # Auto-select SOS2 vs incremental (sign already decided LP or not)
    if method in ("incremental", "auto"):
        is_monotonic = _check_strict_monotonicity(stacked_bp)
        if method == "auto":
            method = "incremental" if (is_monotonic and trailing_nan_only) else "sos2"
        elif not is_monotonic:
            raise ValueError(
                "Incremental method requires strictly monotonic breakpoints."
            )
        if method == "incremental" and not trailing_nan_only:
            raise ValueError(
                "Incremental method does not support non-trailing NaN breakpoints."
            )

    if method == "sos2":
        _validate_numeric_breakpoint_coords(stacked_bp)
        if not trailing_nan_only:
            raise ValueError(
                "SOS2 method does not support non-trailing NaN breakpoints."
            )

    # When sign="==", stack ALL tuples into one link constraint (cheaper, same math).
    # When sign != EQUAL, split: output (FIRST, signed) + inputs (stacked, equality).
    if sign == EQUAL:
        stacked_full_data = _stack_along_link(
            [e.data for e in lin_exprs], link_coords, link_dim
        )
        linked_expr = LinearExpression(stacked_full_data, model)
        linked_bp = stacked_bp
        output_expr = None
        output_bp = None
    else:
        output_expr = lin_exprs[0]
        output_bp = bp_list[0]
        inputs_exprs = lin_exprs[1:]
        inputs_bp_list = bp_list[1:]
        inputs_coords = link_coords[1:]
        linked_expr = None
        linked_bp = None
        if inputs_exprs:
            linked_bp = _stack_along_link(inputs_bp_list, inputs_coords, link_dim)
            input_data = _stack_along_link(
                [e.data for e in inputs_exprs], inputs_coords, link_dim
            )
            linked_expr = LinearExpression(input_data, model)

    rhs = active if active is not None else 1

    if method == "sos2":
        _add_sos2(
            model, name, linked_expr, output_expr,
            stacked_bp, linked_bp, output_bp,
            bp_mask, link_dim, rhs, sign,
        )
        return method
    else:
        _add_incremental(
            model, name, linked_expr, output_expr,
            stacked_bp, linked_bp, output_bp,
            bp_mask, link_dim, rhs, active, sign,
        )
        return method


def _add_sos2(
    model: Model,
    name: str,
    input_expr: LinearExpression | None,
    output_expr: LinearExpression,
    stacked_bp: DataArray,
    input_stacked_bp: DataArray | None,
    output_bp: DataArray,
    bp_mask: DataArray | None,
    link_dim: str,
    rhs: LinearExpression | int,
    sign: str,
) -> None:
    """SOS2 formulation.  Inputs use equality; output uses ``sign``."""
    dim = BREAKPOINT_DIM
    extra = _var_coords_from(stacked_bp, exclude={dim, link_dim})
    lambda_mask = bp_mask
    lambda_coords = extra + [pd.Index(stacked_bp.coords[dim].values, name=dim)]

    lambda_name = f"{name}{PWL_LAMBDA_SUFFIX}"
    convex_name = f"{name}{PWL_CONVEX_SUFFIX}"
    link_name = f"{name}{PWL_LINK_SUFFIX}"
    output_link_name = f"{name}{PWL_OUTPUT_LINK_SUFFIX}"

    lambda_var = model.add_variables(
        lower=0, upper=1, coords=lambda_coords, name=lambda_name, mask=lambda_mask
    )
    model.add_sos_constraints(lambda_var, sos_type=2, sos_dim=dim)
    model.add_constraints(lambda_var.sum(dim=dim) == rhs, name=convex_name)

    # Primary (input) link — always equality, covers all tuples when sign="=="
    if input_expr is not None and input_stacked_bp is not None:
        input_weighted = (lambda_var * input_stacked_bp).sum(dim=dim)
        model.add_constraints(input_expr == input_weighted, name=link_name)

    # Output link — only when sign != EQUAL (otherwise already covered above)
    if output_expr is not None and output_bp is not None:
        output_weighted = (lambda_var * output_bp).sum(dim=dim)
        _add_signed_link(model, output_expr, output_weighted, sign, output_link_name)


def _add_incremental(
    model: Model,
    name: str,
    input_expr: LinearExpression | None,
    output_expr: LinearExpression,
    stacked_bp: DataArray,
    input_stacked_bp: DataArray | None,
    output_bp: DataArray,
    bp_mask: DataArray | None,
    link_dim: str,
    rhs: LinearExpression | int,
    active: LinearExpression | None,
    sign: str,
) -> None:
    """Incremental formulation.  Inputs use equality; output uses ``sign``."""
    dim = BREAKPOINT_DIM
    extra = _var_coords_from(stacked_bp, exclude={dim, link_dim})

    delta_name = f"{name}{PWL_DELTA_SUFFIX}"
    fill_order_name = f"{name}{PWL_FILL_ORDER_SUFFIX}"
    link_name = f"{name}{PWL_LINK_SUFFIX}"
    output_link_name = f"{name}{PWL_OUTPUT_LINK_SUFFIX}"
    order_binary_name = f"{name}{PWL_ORDER_BINARY_SUFFIX}"
    delta_bound_name = f"{name}{PWL_DELTA_BOUND_SUFFIX}"
    binary_order_name = f"{name}{PWL_BINARY_ORDER_SUFFIX}"

    n_segments = stacked_bp.sizes[dim] - 1
    seg_dim = f"{dim}_seg"
    seg_index = pd.Index(range(n_segments), name=seg_dim)
    delta_coords = extra + [seg_index]

    if bp_mask is not None:
        mask_lo = bp_mask.isel({dim: slice(None, -1)}).rename({dim: seg_dim})
        mask_hi = bp_mask.isel({dim: slice(1, None)}).rename({dim: seg_dim})
        mask_lo[seg_dim] = seg_index
        mask_hi[seg_dim] = seg_index
        delta_mask: DataArray | None = mask_lo & mask_hi
    else:
        delta_mask = None

    delta_var = model.add_variables(
        lower=0, upper=1, coords=delta_coords, name=delta_name, mask=delta_mask
    )

    if active is not None:
        active_bound_name = f"{name}{PWL_ACTIVE_BOUND_SUFFIX}"
        model.add_constraints(delta_var <= active, name=active_bound_name)

    binary_var = model.add_variables(
        binary=True, coords=delta_coords, name=order_binary_name, mask=delta_mask
    )
    model.add_constraints(delta_var <= binary_var, name=delta_bound_name)

    if n_segments >= 2:
        delta_lo = delta_var.isel({seg_dim: slice(None, -1)}, drop=True)
        delta_hi = delta_var.isel({seg_dim: slice(1, None)}, drop=True)
        model.add_constraints(delta_hi <= delta_lo, name=fill_order_name)

        binary_hi = binary_var.isel({seg_dim: slice(1, None)}, drop=True)
        model.add_constraints(binary_hi <= delta_lo, name=binary_order_name)

    # Primary (input) link — always equality, covers all tuples when sign="=="
    if input_expr is not None and input_stacked_bp is not None:
        input_steps = input_stacked_bp.diff(dim).rename({dim: seg_dim})
        input_steps[seg_dim] = seg_index
        input_bp0 = input_stacked_bp.isel({dim: 0})
        input_bp0_term: DataArray | LinearExpression = input_bp0
        if active is not None:
            input_bp0_term = input_bp0 * active
        input_weighted = (delta_var * input_steps).sum(dim=seg_dim) + input_bp0_term
        model.add_constraints(input_expr == input_weighted, name=link_name)

    # Output link — only when sign != EQUAL (otherwise already covered above)
    if output_expr is not None and output_bp is not None:
        output_steps = output_bp.diff(dim).rename({dim: seg_dim})
        output_steps[seg_dim] = seg_index
        output_bp0 = output_bp.isel({dim: 0})
        output_bp0_term: DataArray | LinearExpression = output_bp0
        if active is not None:
            output_bp0_term = output_bp0 * active
        output_weighted = (delta_var * output_steps).sum(dim=seg_dim) + output_bp0_term
        _add_signed_link(model, output_expr, output_weighted, sign, output_link_name)


def _add_disjunctive(
    model: Model,
    name: str,
    lin_exprs: list[LinearExpression],
    bp_list: list[DataArray],
    link_coords: list[str],
    bp_mask: DataArray | None,
    sign: str,
    active: LinearExpression | None = None,
) -> None:
    """Disjunctive SOS2 formulation.  Inputs use equality; output uses ``sign``."""
    from linopy.expressions import LinearExpression

    link_dim = "_pwl_var"
    stacked_bp = _stack_along_link(bp_list, link_coords, link_dim)

    _validate_numeric_breakpoint_coords(stacked_bp)
    if not _has_trailing_nan_only(stacked_bp):
        raise ValueError(
            "Disjunctive SOS2 does not support non-trailing NaN breakpoints. "
            "NaN values must only appear at the end of the breakpoint sequence."
        )

    # When sign="==", a single stacked link covers all tuples (cheaper, same math).
    # When sign != EQUAL, split: output (FIRST, signed) + inputs (stacked, equality).
    if sign == EQUAL:
        inputs_exprs = lin_exprs
        inputs_bp = bp_list
        inputs_coords = link_coords
        output_expr = None
        output_bp = None
    else:
        output_expr = lin_exprs[0]
        output_bp = bp_list[0]
        inputs_exprs = lin_exprs[1:]
        inputs_bp = bp_list[1:]
        inputs_coords = link_coords[1:]

    dim = BREAKPOINT_DIM
    extra = _var_coords_from(stacked_bp, exclude={dim, SEGMENT_DIM, link_dim})
    lambda_coords = extra + [
        pd.Index(stacked_bp.coords[SEGMENT_DIM].values, name=SEGMENT_DIM),
        pd.Index(stacked_bp.coords[dim].values, name=dim),
    ]
    binary_coords = extra + [
        pd.Index(stacked_bp.coords[SEGMENT_DIM].values, name=SEGMENT_DIM),
    ]

    lambda_mask = bp_mask
    binary_mask = bp_mask.any(dim=dim) if bp_mask is not None else None

    binary_name = f"{name}{PWL_SEGMENT_BINARY_SUFFIX}"
    select_name = f"{name}{PWL_SELECT_SUFFIX}"
    lambda_name = f"{name}{PWL_LAMBDA_SUFFIX}"
    convex_name = f"{name}{PWL_CONVEX_SUFFIX}"
    link_name = f"{name}{PWL_LINK_SUFFIX}"
    output_link_name = f"{name}{PWL_OUTPUT_LINK_SUFFIX}"

    binary_var = model.add_variables(
        binary=True, coords=binary_coords, name=binary_name, mask=binary_mask
    )
    rhs = active if active is not None else 1
    model.add_constraints(binary_var.sum(dim=SEGMENT_DIM) == rhs, name=select_name)

    lambda_var = model.add_variables(
        lower=0, upper=1, coords=lambda_coords, name=lambda_name, mask=lambda_mask
    )
    model.add_sos_constraints(lambda_var, sos_type=2, sos_dim=dim)
    model.add_constraints(lambda_var.sum(dim=dim) == binary_var, name=convex_name)

    # Input link (equality)
    if inputs_exprs:
        stacked_input_bp = _stack_along_link(inputs_bp, inputs_coords, link_dim)
        stacked_input_data = _stack_along_link(
            [e.data for e in inputs_exprs], inputs_coords, link_dim
        )
        stacked_input_expr = LinearExpression(stacked_input_data, model)
        input_weighted = (lambda_var * stacked_input_bp).sum(dim=[SEGMENT_DIM, dim])
        model.add_constraints(stacked_input_expr == input_weighted, name=link_name)

    # Output link (respects sign) — only when sign != EQUAL
    if output_expr is not None and output_bp is not None:
        output_weighted = (lambda_var * output_bp).sum(dim=[SEGMENT_DIM, dim])
        _add_signed_link(model, output_expr, output_weighted, sign, output_link_name)


def _add_signed_link(
    model: Model,
    lhs: LinearExpression,
    rhs: LinearExpression,
    sign: str,
    name: str,
) -> Constraint:
    """Add a link constraint with the requested sign."""
    if sign == EQUAL:
        return model.add_constraints(lhs == rhs, name=name)
    elif sign == LESS_EQUAL:
        return model.add_constraints(lhs <= rhs, name=name)
    else:  # ">="
        return model.add_constraints(lhs >= rhs, name=name)


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

    Adds one chord constraint per segment plus domain bounds on x.
    """
    dx = x_points.diff(BREAKPOINT_DIM)
    dy = y_points.diff(BREAKPOINT_DIM)
    seg_index = np.arange(dx.sizes[BREAKPOINT_DIM])

    slopes = _rename_to_segments(dy / dx, seg_index)
    x_base = _rename_to_segments(
        x_points.isel({BREAKPOINT_DIM: slice(None, -1)}), seg_index
    )
    y_base = _rename_to_segments(
        y_points.isel({BREAKPOINT_DIM: slice(None, -1)}), seg_index
    )
    intercepts = y_base - slopes * x_base
    tangents = slopes * x_expr + intercepts

    lp_name = f"{name}{PWL_LP_SUFFIX}"
    _add_signed_link(model, y_expr, tangents, sign, lp_name)

    # Domain bounds: x ∈ [x_min, x_max]
    x_min = x_points.min(dim=BREAKPOINT_DIM)
    x_max = x_points.max(dim=BREAKPOINT_DIM)
    model.add_constraints(
        x_expr >= x_min, name=f"{name}{PWL_DOMAIN_LO_SUFFIX}"
    )
    model.add_constraints(
        x_expr <= x_max, name=f"{name}{PWL_DOMAIN_HI_SUFFIX}"
    )
