"""
Piecewise linear constraint formulations.

Provides SOS2, incremental, and disjunctive piecewise linear constraint
methods for use with linopy.Model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray

from linopy.constants import (
    DEFAULT_BREAKPOINT_DIM,
    DEFAULT_LINK_DIM,
    DEFAULT_SEGMENT_DIM,
    HELPER_DIMS,
    PWL_BINARY_SUFFIX,
    PWL_CONVEX_SUFFIX,
    PWL_DELTA_SUFFIX,
    PWL_FILL_SUFFIX,
    PWL_LAMBDA_SUFFIX,
    PWL_LINK_SUFFIX,
    PWL_SELECT_SUFFIX,
)

if TYPE_CHECKING:
    from linopy.constraints import Constraint
    from linopy.expressions import LinearExpression
    from linopy.model import Model
    from linopy.types import LinExprLike


def _list_to_array(values: list[float], bp_dim: str) -> DataArray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D list of numeric values, got shape {arr.shape}")
    return DataArray(arr, dims=[bp_dim], coords={bp_dim: np.arange(len(arr))})


def _dict_to_array(d: dict[str, list[float]], dim: str, bp_dim: str) -> DataArray:
    max_len = max(len(v) for v in d.values())
    keys = list(d.keys())
    data = np.full((len(keys), max_len), np.nan)
    for i, k in enumerate(keys):
        vals = d[k]
        data[i, : len(vals)] = vals
    return DataArray(
        data,
        dims=[dim, bp_dim],
        coords={dim: keys, bp_dim: np.arange(max_len)},
    )


def _segments_list_to_array(
    values: list[Sequence[float]], bp_dim: str, seg_dim: str
) -> DataArray:
    max_len = max(len(seg) for seg in values)
    data = np.full((len(values), max_len), np.nan)
    for i, seg in enumerate(values):
        data[i, : len(seg)] = seg
    return DataArray(
        data,
        dims=[seg_dim, bp_dim],
        coords={seg_dim: np.arange(len(values)), bp_dim: np.arange(max_len)},
    )


def _dict_segments_to_array(
    d: dict[str, list[Sequence[float]]], dim: str, bp_dim: str, seg_dim: str
) -> DataArray:
    parts = []
    for key, seg_list in d.items():
        arr = _segments_list_to_array(seg_list, bp_dim, seg_dim)
        parts.append(arr.expand_dims({dim: [key]}))
    combined = xr.concat(parts, dim=dim)
    max_bp = max(max(len(seg) for seg in sl) for sl in d.values())
    max_seg = max(len(sl) for sl in d.values())
    if combined.sizes[bp_dim] < max_bp or combined.sizes[seg_dim] < max_seg:
        combined = combined.reindex(
            {bp_dim: np.arange(max_bp), seg_dim: np.arange(max_seg)},
            fill_value=np.nan,
        )
    return combined


def _get_entity_keys(
    kwargs: Mapping[str, object],
) -> list[str]:
    first_dict = next(v for v in kwargs.values() if isinstance(v, dict))
    return list(first_dict.keys())


def _validate_factory_args(
    values: list | dict | None,
    kwargs: dict,
) -> None:
    if values is not None and kwargs:
        raise ValueError("Cannot pass both positional 'values' and keyword arguments")
    if values is None and not kwargs:
        raise ValueError("Must pass either positional 'values' or keyword arguments")


def _resolve_kwargs(
    kwargs: dict[str, list[float] | dict[str, list[float]] | DataArray],
    dim: str | None,
    bp_dim: str,
    link_dim: str,
) -> DataArray:
    has_dict = any(isinstance(v, dict) for v in kwargs.values())
    if has_dict and dim is None:
        raise ValueError("'dim' is required when any kwarg value is a dict")

    arrays: dict[str, DataArray] = {}
    for name, val in kwargs.items():
        if isinstance(val, DataArray):
            arrays[name] = val
        elif isinstance(val, dict):
            assert dim is not None
            arrays[name] = _dict_to_array(val, dim, bp_dim)
        elif isinstance(val, list):
            base = _list_to_array(val, bp_dim)
            if has_dict:
                base = base.expand_dims({dim: _get_entity_keys(kwargs)})
            arrays[name] = base
        else:
            raise ValueError(
                f"kwarg '{name}' must be a list, dict, or DataArray, got {type(val)}"
            )

    parts = [arr.expand_dims({link_dim: [name]}) for name, arr in arrays.items()]
    return xr.concat(parts, dim=link_dim)


def _resolve_segment_kwargs(
    kwargs: dict[
        str, list[Sequence[float]] | dict[str, list[Sequence[float]]] | DataArray
    ],
    dim: str | None,
    bp_dim: str,
    seg_dim: str,
    link_dim: str,
) -> DataArray:
    has_dict = any(isinstance(v, dict) for v in kwargs.values())
    if has_dict and dim is None:
        raise ValueError("'dim' is required when any kwarg value is a dict")

    arrays: dict[str, DataArray] = {}
    for name, val in kwargs.items():
        if isinstance(val, DataArray):
            arrays[name] = val
        elif isinstance(val, dict):
            assert dim is not None
            arrays[name] = _dict_segments_to_array(val, dim, bp_dim, seg_dim)
        elif isinstance(val, list):
            base = _segments_list_to_array(val, bp_dim, seg_dim)
            if has_dict:
                base = base.expand_dims({dim: _get_entity_keys(kwargs)})
            arrays[name] = base
        else:
            raise ValueError(
                f"kwarg '{name}' must be a list, dict, or DataArray, got {type(val)}"
            )

    parts = [arr.expand_dims({link_dim: [name]}) for name, arr in arrays.items()]
    combined = xr.concat(parts, dim=link_dim)
    max_bp = max(a.sizes.get(bp_dim, 0) for a in arrays.values())
    max_seg = max(a.sizes.get(seg_dim, 0) for a in arrays.values())
    if (
        combined.sizes.get(bp_dim, 0) < max_bp
        or combined.sizes.get(seg_dim, 0) < max_seg
    ):
        combined = combined.reindex(
            {bp_dim: np.arange(max_bp), seg_dim: np.arange(max_seg)},
            fill_value=np.nan,
        )
    return combined


class _BreakpointFactory:
    """
    Factory for creating breakpoint DataArrays for piecewise linear constraints.

    Use ``linopy.breakpoints(...)`` for continuous breakpoints and
    ``linopy.breakpoints.segments(...)`` for disjunctive (disconnected) segments.
    """

    def __call__(
        self,
        values: list[float] | dict[str, list[float]] | None = None,
        *,
        dim: str | None = None,
        bp_dim: str = DEFAULT_BREAKPOINT_DIM,
        link_dim: str = DEFAULT_LINK_DIM,
        **kwargs: list[float] | dict[str, list[float]] | DataArray,
    ) -> DataArray:
        """
        Create a breakpoint DataArray for piecewise linear constraints.

        Parameters
        ----------
        values : list or dict, optional
            Breakpoint values. A list creates 1D breakpoints. A dict creates
            per-entity breakpoints (requires ``dim``). Cannot be used with kwargs.
        dim : str, optional
            Entity dimension name. Required when ``values`` is a dict.
        bp_dim : str, default "breakpoint"
            Name for the breakpoint dimension.
        link_dim : str, default "var"
            Name for the link dimension when using kwargs.
        **kwargs : list, dict, or DataArray
            Per-variable breakpoints. Each kwarg becomes a coordinate on the
            link dimension.

        Returns
        -------
        DataArray
            Breakpoint array with appropriate dimensions and coordinates.
        """
        _validate_factory_args(values, kwargs)

        if values is not None:
            if isinstance(values, list):
                return _list_to_array(values, bp_dim)
            if isinstance(values, dict):
                if dim is None:
                    raise ValueError("'dim' is required when 'values' is a dict")
                return _dict_to_array(values, dim, bp_dim)
            raise TypeError(f"'values' must be a list or dict, got {type(values)}")

        return _resolve_kwargs(kwargs, dim, bp_dim, link_dim)

    def segments(
        self,
        values: list[Sequence[float]] | dict[str, list[Sequence[float]]] | None = None,
        *,
        dim: str | None = None,
        bp_dim: str = DEFAULT_BREAKPOINT_DIM,
        seg_dim: str = DEFAULT_SEGMENT_DIM,
        link_dim: str = DEFAULT_LINK_DIM,
        **kwargs: list[Sequence[float]] | dict[str, list[Sequence[float]]] | DataArray,
    ) -> DataArray:
        """
        Create a segmented breakpoint DataArray for disjunctive piecewise constraints.

        Parameters
        ----------
        values : list or dict, optional
            Segment breakpoints. A list of lists creates 2D breakpoints
            ``[segment, breakpoint]``. A dict creates per-entity segments
            (requires ``dim``). Cannot be used with kwargs.
        dim : str, optional
            Entity dimension name. Required when ``values`` is a dict.
        bp_dim : str, default "breakpoint"
            Name for the breakpoint dimension.
        seg_dim : str, default "segment"
            Name for the segment dimension.
        link_dim : str, default "var"
            Name for the link dimension when using kwargs.
        **kwargs : list, dict, or DataArray
            Per-variable segment breakpoints.

        Returns
        -------
        DataArray
            Breakpoint array with segment and breakpoint dimensions.
        """
        _validate_factory_args(values, kwargs)

        if values is not None:
            if isinstance(values, list):
                return _segments_list_to_array(values, bp_dim, seg_dim)
            if isinstance(values, dict):
                if dim is None:
                    raise ValueError("'dim' is required when 'values' is a dict")
                return _dict_segments_to_array(values, dim, bp_dim, seg_dim)
            raise TypeError(f"'values' must be a list or dict, got {type(values)}")

        return _resolve_segment_kwargs(kwargs, dim, bp_dim, seg_dim, link_dim)


breakpoints = _BreakpointFactory()


def _auto_broadcast_breakpoints(
    bp: DataArray,
    expr: LinExprLike | dict[str, LinExprLike],
    dim: str,
    link_dim: str | None = None,
    exclude_dims: set[str] | None = None,
) -> DataArray:
    _, target_dims = _validate_piecewise_expr(expr)

    skip = {dim} | set(HELPER_DIMS)
    if link_dim is not None:
        skip.add(link_dim)
    if exclude_dims is not None:
        skip.update(exclude_dims)

    target_dims -= skip
    missing = target_dims - {str(d) for d in bp.dims}

    if not missing:
        return bp

    expand_map: dict[str, list] = {}
    all_exprs = expr.values() if isinstance(expr, dict) else [expr]
    for d in missing:
        for e in all_exprs:
            if d in e.coords:
                expand_map[str(d)] = list(e.coords[d].values)
                break

    if expand_map:
        bp = bp.expand_dims(expand_map)

    return bp


def _extra_coords(breakpoints: DataArray, *exclude_dims: str | None) -> list[pd.Index]:
    excluded = {d for d in exclude_dims if d is not None}
    return [
        pd.Index(breakpoints.coords[d].values, name=d)
        for d in breakpoints.dims
        if d not in excluded
    ]


def _validate_breakpoints(breakpoints: DataArray, dim: str) -> None:
    if dim not in breakpoints.dims:
        raise ValueError(
            f"breakpoints must have dimension '{dim}', "
            f"but only has dimensions {list(breakpoints.dims)}"
        )


def _validate_numeric_breakpoint_coords(breakpoints: DataArray, dim: str) -> None:
    if not pd.api.types.is_numeric_dtype(breakpoints.coords[dim]):
        raise ValueError(
            f"Breakpoint dimension '{dim}' must have numeric coordinates "
            f"for SOS2 weights, but got {breakpoints.coords[dim].dtype}"
        )


def _check_strict_monotonicity(breakpoints: DataArray, dim: str) -> bool:
    """
    Check if breakpoints are strictly monotonic along dim.

    Each slice along non-dim dimensions is checked independently,
    allowing different slices to have opposite directions (e.g., one
    increasing and another decreasing). NaN values are ignored.
    """
    diffs = breakpoints.diff(dim)
    pos = (diffs > 0) | diffs.isnull()
    neg = (diffs < 0) | diffs.isnull()
    all_pos_per_slice = pos.all(dim)
    all_neg_per_slice = neg.all(dim)
    has_non_nan = (~diffs.isnull()).any(dim)
    monotonic = (all_pos_per_slice | all_neg_per_slice) & has_non_nan
    return bool(monotonic.all())


def _has_trailing_nan_only(breakpoints: DataArray, dim: str) -> bool:
    """Check that NaN values in breakpoints only appear as trailing entries along dim."""
    valid = ~breakpoints.isnull()
    cummin = np.minimum.accumulate(valid.values, axis=valid.dims.index(dim))
    cummin_da = DataArray(cummin, coords=valid.coords, dims=valid.dims)
    return not bool((valid & ~cummin_da).any())


def _to_linexpr(expr: LinExprLike) -> LinearExpression:
    from linopy.expressions import LinearExpression

    if isinstance(expr, LinearExpression):
        return expr
    return expr.to_linexpr()


def _validate_piecewise_expr(
    expr: LinExprLike | dict[str, LinExprLike],
) -> tuple[bool, set[str]]:
    from linopy.expressions import LinearExpression
    from linopy.variables import Variable

    _types = (Variable, LinearExpression)

    if isinstance(expr, _types):
        return True, {str(d) for d in expr.coord_dims}

    if isinstance(expr, dict):
        dims: set[str] = set()
        for key, val in expr.items():
            if not isinstance(val, _types):
                raise TypeError(
                    f"dict value for key '{key}' must be a Variable or "
                    f"LinearExpression, got {type(val)}"
                )
            dims.update(str(d) for d in val.coord_dims)
        return False, dims

    raise TypeError(
        f"'expr' must be a Variable, LinearExpression, or dict of these, "
        f"got {type(expr)}"
    )


def _compute_mask(
    mask: DataArray | None,
    breakpoints: DataArray,
    skip_nan_check: bool,
) -> DataArray | None:
    if mask is not None:
        return mask
    if skip_nan_check:
        return None
    return ~breakpoints.isnull()


def _resolve_link_dim(
    breakpoints: DataArray,
    expr_keys: set[str],
    exclude_dims: set[str],
) -> str:
    for d in breakpoints.dims:
        if d in exclude_dims:
            continue
        coord_set = {str(c) for c in breakpoints.coords[d].values}
        if coord_set == expr_keys:
            return str(d)
    raise ValueError(
        "Could not auto-detect linking dimension from breakpoints. "
        "Ensure breakpoints have a dimension whose coordinates match "
        f"the expression dict keys. "
        f"Breakpoint dimensions: {list(breakpoints.dims)}, "
        f"expression keys: {list(expr_keys)}"
    )


def _build_stacked_expr(
    model: Model,
    expr_dict: dict[str, LinExprLike],
    breakpoints: DataArray,
    link_dim: str,
) -> LinearExpression:
    from linopy.expressions import LinearExpression

    link_coords = list(breakpoints.coords[link_dim].values)

    expr_data_list = []
    for k in link_coords:
        e = expr_dict[str(k)]
        linexpr = _to_linexpr(e)
        expr_data_list.append(linexpr.data.expand_dims({link_dim: [k]}))

    stacked_data = xr.concat(expr_data_list, dim=link_dim)
    return LinearExpression(stacked_data, model)


def _resolve_expr(
    model: Model,
    expr: LinExprLike | dict[str, LinExprLike],
    breakpoints: DataArray,
    dim: str,
    mask: DataArray | None,
    skip_nan_check: bool,
    exclude_dims: set[str] | None = None,
) -> tuple[LinearExpression, str | None, DataArray | None, DataArray | None]:
    is_single, _ = _validate_piecewise_expr(expr)

    computed_mask = _compute_mask(mask, breakpoints, skip_nan_check)

    if is_single:
        target_expr = _to_linexpr(expr)  # type: ignore[arg-type]
        return target_expr, None, computed_mask, computed_mask

    expr_dict: dict[str, LinExprLike] = expr  # type: ignore[assignment]
    expr_keys = set(expr_dict.keys())
    all_exclude = {dim} | (exclude_dims or set())
    resolved_link_dim = _resolve_link_dim(breakpoints, expr_keys, all_exclude)
    lambda_mask = None
    if computed_mask is not None:
        if resolved_link_dim not in computed_mask.dims:
            computed_mask = computed_mask.broadcast_like(breakpoints)
        lambda_mask = computed_mask.any(dim=resolved_link_dim)
    target_expr = _build_stacked_expr(model, expr_dict, breakpoints, resolved_link_dim)
    return target_expr, resolved_link_dim, computed_mask, lambda_mask


def _add_pwl_sos2(
    model: Model,
    name: str,
    breakpoints: DataArray,
    dim: str,
    target_expr: LinearExpression,
    lambda_coords: list[pd.Index],
    lambda_mask: DataArray | None,
) -> Constraint:
    lambda_name = f"{name}{PWL_LAMBDA_SUFFIX}"
    convex_name = f"{name}{PWL_CONVEX_SUFFIX}"
    link_name = f"{name}{PWL_LINK_SUFFIX}"

    lambda_var = model.add_variables(
        lower=0, upper=1, coords=lambda_coords, name=lambda_name, mask=lambda_mask
    )

    model.add_sos_constraints(lambda_var, sos_type=2, sos_dim=dim)

    convex_con = model.add_constraints(lambda_var.sum(dim=dim) == 1, name=convex_name)

    weighted_sum = (lambda_var * breakpoints).sum(dim=dim)
    model.add_constraints(target_expr == weighted_sum, name=link_name)

    return convex_con


def _add_pwl_incremental(
    model: Model,
    name: str,
    breakpoints: DataArray,
    dim: str,
    target_expr: LinearExpression,
    extra_coords: list[pd.Index],
    breakpoint_mask: DataArray | None,
    link_dim: str | None,
) -> Constraint:
    delta_name = f"{name}{PWL_DELTA_SUFFIX}"
    fill_name = f"{name}{PWL_FILL_SUFFIX}"
    link_name = f"{name}{PWL_LINK_SUFFIX}"

    n_segments = breakpoints.sizes[dim] - 1
    seg_dim = f"{dim}_seg"
    seg_index = pd.Index(range(n_segments), name=seg_dim)
    delta_coords = extra_coords + [seg_index]

    steps = breakpoints.diff(dim).rename({dim: seg_dim})
    steps[seg_dim] = seg_index

    if breakpoint_mask is not None:
        bp_mask = breakpoint_mask
        if link_dim is not None:
            bp_mask = bp_mask.all(dim=link_dim)
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

    fill_con: Constraint | None = None
    if n_segments >= 2:
        delta_lo = delta_var.isel({seg_dim: slice(None, -1)}, drop=True)
        delta_hi = delta_var.isel({seg_dim: slice(1, None)}, drop=True)
        fill_con = model.add_constraints(delta_hi <= delta_lo, name=fill_name)

    bp0 = breakpoints.isel({dim: 0})
    weighted_sum = (delta_var * steps).sum(dim=seg_dim) + bp0
    link_con = model.add_constraints(target_expr == weighted_sum, name=link_name)

    return fill_con if fill_con is not None else link_con


def _add_dpwl_sos2(
    model: Model,
    name: str,
    breakpoints: DataArray,
    dim: str,
    segment_dim: str,
    target_expr: LinearExpression,
    lambda_coords: list[pd.Index],
    lambda_mask: DataArray | None,
    binary_coords: list[pd.Index],
    binary_mask: DataArray | None,
) -> Constraint:
    binary_name = f"{name}{PWL_BINARY_SUFFIX}"
    select_name = f"{name}{PWL_SELECT_SUFFIX}"
    lambda_name = f"{name}{PWL_LAMBDA_SUFFIX}"
    convex_name = f"{name}{PWL_CONVEX_SUFFIX}"
    link_name = f"{name}{PWL_LINK_SUFFIX}"

    binary_var = model.add_variables(
        binary=True, coords=binary_coords, name=binary_name, mask=binary_mask
    )

    select_con = model.add_constraints(
        binary_var.sum(dim=segment_dim) == 1, name=select_name
    )

    lambda_var = model.add_variables(
        lower=0, upper=1, coords=lambda_coords, name=lambda_name, mask=lambda_mask
    )

    model.add_sos_constraints(lambda_var, sos_type=2, sos_dim=dim)

    model.add_constraints(lambda_var.sum(dim=dim) == binary_var, name=convex_name)

    weighted_sum = (lambda_var * breakpoints).sum(dim=[segment_dim, dim])
    model.add_constraints(target_expr == weighted_sum, name=link_name)

    return select_con


def add_piecewise_constraints(
    model: Model,
    expr: LinExprLike | dict[str, LinExprLike],
    breakpoints: DataArray,
    dim: str = DEFAULT_BREAKPOINT_DIM,
    mask: DataArray | None = None,
    name: str | None = None,
    skip_nan_check: bool = False,
    method: Literal["sos2", "incremental", "auto"] = "sos2",
) -> Constraint:
    """
    Add a piecewise linear constraint using SOS2 or incremental formulation.

    This method creates a piecewise linear constraint that links one or more
    variables/expressions together via a set of breakpoints. It supports two
    formulations:

    - **SOS2** (default): Uses SOS2 (Special Ordered Set of type 2) with lambda
      (interpolation) variables. Works for any breakpoints.
    - **Incremental**: Uses delta variables with filling-order constraints.
      Pure LP formulation (no SOS2 or binary variables), but requires strictly
      monotonic breakpoints.

    Parameters
    ----------
    model : Model
        The linopy model to add the constraint to.
    expr : Variable, LinearExpression, or dict of these
        The variable(s) or expression(s) to be linked by the piecewise constraint.
        - If a single Variable/LinearExpression is passed, the breakpoints
          directly specify the piecewise points for that expression.
        - If a dict is passed, the keys must match coordinates of a dimension
          of the breakpoints, allowing multiple expressions to be linked.
    breakpoints : xr.DataArray
        The breakpoint values defining the piecewise linear function.
        Must have `dim` as one of its dimensions. If `expr` is a dict,
        must also have a dimension with coordinates matching the dict keys.
    dim : str, default "breakpoint"
        The dimension in breakpoints that represents the breakpoint index.
        This dimension's coordinates must be numeric (used as SOS2 weights
        for the SOS2 method).
    mask : xr.DataArray, optional
        Boolean mask indicating which piecewise constraints are valid.
        If None, auto-detected from NaN values in breakpoints (unless
        skip_nan_check is True).
    name : str, optional
        Base name for the generated variables and constraints.
        If None, auto-generates names like "pwl0", "pwl1", etc.
    skip_nan_check : bool, default False
        If True, skip automatic NaN detection in breakpoints. Use this
        when you know breakpoints contain no NaN values for better performance.
    method : Literal["sos2", "incremental", "auto"], default "sos2"
        Formulation method. One of:
        - ``"sos2"``: SOS2 formulation with lambda variables (default).
        - ``"incremental"``: Incremental (delta) formulation. Requires strictly
          monotonic breakpoints. Pure LP, no SOS2 or binary variables.
        - ``"auto"``: Automatically selects ``"incremental"`` if breakpoints are
          strictly monotonic, otherwise falls back to ``"sos2"``.

    Returns
    -------
    Constraint
        For SOS2: the convexity constraint (sum of lambda = 1).
        For incremental: the filling-order constraint (or the link
        constraint if only 2 breakpoints).

    Raises
    ------
    ValueError
        If expr is not a Variable, LinearExpression, or dict of these.
        If breakpoints doesn't have the required dim dimension.
        If the linking dimension cannot be auto-detected when expr is a dict.
        If dim coordinates are not numeric (SOS2 method only).
        If breakpoints are not strictly monotonic (incremental method).
        If method is not one of 'sos2', 'incremental', 'auto'.

    Examples
    --------
    Single variable piecewise constraint:

    >>> from linopy import Model
    >>> import xarray as xr
    >>> m = Model()
    >>> x = m.add_variables(name="x")
    >>> breakpoints = xr.DataArray([0, 10, 50, 100], dims=["bp"])
    >>> _ = m.add_piecewise_constraints(x, breakpoints, dim="bp")

    Notes
    -----
    **SOS2 formulation:**

    1. Lambda variables λ_i with bounds [0, 1] are created for each breakpoint
    2. SOS2 constraint ensures at most two adjacent λ_i can be non-zero
    3. Convexity constraint: Σ λ_i = 1
    4. Linking constraints: expr = Σ λ_i × breakpoint_i (for each expression)

    **Incremental formulation** (for strictly monotonic breakpoints bp₀ < bp₁ < ... < bpₙ):

    1. Delta variables δᵢ ∈ [0, 1] for i = 1, ..., n (one per segment)
    2. Filling-order constraints: δᵢ₊₁ ≤ δᵢ for i = 1, ..., n-1
    3. Linking constraint: expr = bp₀ + Σᵢ δᵢ × (bpᵢ - bpᵢ₋₁)
    """
    if method not in ("sos2", "incremental", "auto"):
        raise ValueError(
            f"method must be 'sos2', 'incremental', or 'auto', got '{method}'"
        )

    _validate_breakpoints(breakpoints, dim)
    breakpoints = _auto_broadcast_breakpoints(breakpoints, expr, dim)

    if method in ("incremental", "auto"):
        is_monotonic = _check_strict_monotonicity(breakpoints, dim)
        trailing_nan_only = _has_trailing_nan_only(breakpoints, dim)
        if method == "auto":
            if is_monotonic and trailing_nan_only:
                method = "incremental"
            else:
                method = "sos2"
        elif not is_monotonic:
            raise ValueError(
                "Incremental method requires strictly monotonic breakpoints "
                "along the breakpoint dimension."
            )
        if method == "incremental" and not trailing_nan_only:
            raise ValueError(
                "Incremental method does not support non-trailing NaN breakpoints. "
                "NaN values must only appear at the end of the breakpoint sequence. "
                "Use method='sos2' for breakpoints with gaps."
            )

    if method == "sos2":
        _validate_numeric_breakpoint_coords(breakpoints, dim)

    if name is None:
        name = f"pwl{model._pwlCounter}"
        model._pwlCounter += 1

    target_expr, resolved_link_dim, computed_mask, lambda_mask = _resolve_expr(
        model, expr, breakpoints, dim, mask, skip_nan_check
    )

    extra_coords = _extra_coords(breakpoints, dim, resolved_link_dim)
    lambda_coords = extra_coords + [pd.Index(breakpoints.coords[dim].values, name=dim)]

    if method == "sos2":
        return _add_pwl_sos2(
            model, name, breakpoints, dim, target_expr, lambda_coords, lambda_mask
        )
    else:
        return _add_pwl_incremental(
            model,
            name,
            breakpoints,
            dim,
            target_expr,
            extra_coords,
            computed_mask,
            resolved_link_dim,
        )


def add_disjunctive_piecewise_constraints(
    model: Model,
    expr: LinExprLike | dict[str, LinExprLike],
    breakpoints: DataArray,
    dim: str = DEFAULT_BREAKPOINT_DIM,
    segment_dim: str = DEFAULT_SEGMENT_DIM,
    mask: DataArray | None = None,
    name: str | None = None,
    skip_nan_check: bool = False,
) -> Constraint:
    """
    Add a disjunctive piecewise linear constraint for disconnected segments.

    Unlike ``add_piecewise_constraints``, which models continuous piecewise
    linear functions (all segments connected end-to-end), this method handles
    **disconnected segments** (with gaps between them). The variable must lie
    on exactly one segment, selected by binary indicator variables.

    Uses the disaggregated convex combination formulation (no big-M needed,
    tight LP relaxation):

    1. Binary ``y_k ∈ {0,1}`` per segment, ``Σ y_k = 1``
    2. Lambda ``λ_{k,i} ∈ [0,1]`` per breakpoint in each segment
    3. Convexity: ``Σ_i λ_{k,i} = y_k``
    4. SOS2 within each segment (along breakpoint dim)
    5. Linking: ``expr = Σ_k Σ_i λ_{k,i} × bp_{k,i}``

    Parameters
    ----------
    model : Model
        The linopy model to add the constraint to.
    expr : Variable, LinearExpression, or dict of these
        The variable(s) or expression(s) to be linked by the piecewise
        constraint.
    breakpoints : xr.DataArray
        Breakpoint values with at least ``dim`` and ``segment_dim``
        dimensions. Each slice along ``segment_dim`` defines one segment.
        Use NaN to pad segments with fewer breakpoints.
    dim : str, default "breakpoint"
        Dimension for breakpoint indices within each segment.
        Must have numeric coordinates.
    segment_dim : str, default "segment"
        Dimension indexing the segments.
    mask : xr.DataArray, optional
        Boolean mask. If None, auto-detected from NaN values.
    name : str, optional
        Base name for generated variables/constraints. Auto-generated
        if None using the shared ``_pwlCounter``.
    skip_nan_check : bool, default False
        If True, skip NaN detection in breakpoints.

    Returns
    -------
    Constraint
        The selection constraint (``Σ y_k = 1``).

    Raises
    ------
    ValueError
        If ``dim`` or ``segment_dim`` not in breakpoints dimensions.
        If ``dim == segment_dim``.
        If ``dim`` coordinates are not numeric.
        If ``expr`` is not a Variable, LinearExpression, or dict.

    Examples
    --------
    Two disconnected segments [0,10] and [50,100]:

    >>> from linopy import Model
    >>> import xarray as xr
    >>> m = Model()
    >>> x = m.add_variables(name="x")
    >>> breakpoints = xr.DataArray(
    ...     [[0, 10], [50, 100]],
    ...     dims=["segment", "breakpoint"],
    ...     coords={"segment": [0, 1], "breakpoint": [0, 1]},
    ... )
    >>> _ = m.add_disjunctive_piecewise_constraints(x, breakpoints)
    """
    _validate_breakpoints(breakpoints, dim)
    if segment_dim not in breakpoints.dims:
        raise ValueError(
            f"breakpoints must have dimension '{segment_dim}', "
            f"but only has dimensions {list(breakpoints.dims)}"
        )
    if dim == segment_dim:
        raise ValueError(f"dim and segment_dim must be different, both are '{dim}'")
    _validate_numeric_breakpoint_coords(breakpoints, dim)
    breakpoints = _auto_broadcast_breakpoints(
        breakpoints, expr, dim, exclude_dims={segment_dim}
    )

    if name is None:
        name = f"pwl{model._pwlCounter}"
        model._pwlCounter += 1

    target_expr, resolved_link_dim, computed_mask, lambda_mask = _resolve_expr(
        model,
        expr,
        breakpoints,
        dim,
        mask,
        skip_nan_check,
        exclude_dims={segment_dim},
    )

    extra_coords = _extra_coords(breakpoints, dim, segment_dim, resolved_link_dim)
    lambda_coords = extra_coords + [
        pd.Index(breakpoints.coords[segment_dim].values, name=segment_dim),
        pd.Index(breakpoints.coords[dim].values, name=dim),
    ]
    binary_coords = extra_coords + [
        pd.Index(breakpoints.coords[segment_dim].values, name=segment_dim),
    ]

    binary_mask = lambda_mask.any(dim=dim) if lambda_mask is not None else None

    return _add_dpwl_sos2(
        model,
        name,
        breakpoints,
        dim,
        segment_dim,
        target_expr,
        lambda_coords,
        lambda_mask,
        binary_coords,
        binary_mask,
    )
