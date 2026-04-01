"""
Linearization utilities for approximating nonlinear functions.

These helpers return regular :class:`~linopy.expressions.LinearExpression`
objects --- no auxiliary variables or special constraint types are created.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from xarray import DataArray

from linopy.constants import BREAKPOINT_DIM, LP_SEG_DIM
from linopy.piecewise import BreaksLike, _coerce_breaks

if TYPE_CHECKING:
    from linopy.expressions import LinearExpression
    from linopy.types import LinExprLike


def piecewise_tangents(
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
    envelope:

    .. code-block:: python

        envelope = piecewise_tangents(power, x_pts, y_pts)
        m.add_constraints(fuel <= envelope)  # upper bound (concave f)
        m.add_constraints(fuel >= envelope)  # lower bound (convex f)

    No auxiliary variables are created --- the result is purely linear.

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
    from linopy.expressions import LinearExpression
    from linopy.variables import Variable

    if not isinstance(x_points, DataArray):
        x_points = _coerce_breaks(x_points)
    if not isinstance(y_points, DataArray):
        y_points = _coerce_breaks(y_points)

    dx = x_points.diff(BREAKPOINT_DIM)
    dy = y_points.diff(BREAKPOINT_DIM)
    slopes = dy / dx

    n_seg = slopes.sizes[BREAKPOINT_DIM]
    seg_index = np.arange(n_seg)

    slopes = slopes.rename({BREAKPOINT_DIM: LP_SEG_DIM})
    slopes[LP_SEG_DIM] = seg_index

    x_base = x_points.isel({BREAKPOINT_DIM: slice(None, -1)}).rename(
        {BREAKPOINT_DIM: LP_SEG_DIM}
    )
    y_base = y_points.isel({BREAKPOINT_DIM: slice(None, -1)}).rename(
        {BREAKPOINT_DIM: LP_SEG_DIM}
    )
    x_base[LP_SEG_DIM] = seg_index
    y_base[LP_SEG_DIM] = seg_index

    # tangent_k(x) = slopes_k * (x - x_base_k) + y_base_k
    #              = slopes_k * x + (y_base_k - slopes_k * x_base_k)
    intercepts = y_base - slopes * x_base

    if isinstance(x, Variable):
        x_expr = x.to_linexpr()
    elif isinstance(x, LinearExpression):
        x_expr = x
    else:
        raise TypeError(f"x must be a Variable or LinearExpression, got {type(x)}")

    return slopes * x_expr + intercepts
