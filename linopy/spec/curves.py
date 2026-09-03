"""
The data-time side of a ``piecewise:`` block.

The language decides a curve's shape and can decide nothing about its
numbers. This module fills the parameters an expansion emitted from the
block's own breakpoints, and checks that the numbers hold what the block's
method rests on: the conditions are the program's :data:`~math_spec.program.Check`
values and :func:`~math_spec.program.check_message` words each refusal.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

import numpy as np
import xarray as xr
from math_spec import program as ms

from linopy.spec.errors import SpecDataError

_C = TypeVar("_C", bound=ms.Check)


def derive(
    derivation: ms.Derivation,
    parameters: Mapping[str, xr.DataArray],
    program: ms.Program,
) -> xr.DataArray:
    """
    An emitted ``bool`` parameter, built from the parameters it hangs off.

    A :class:`~math_spec.program.MaskOf` is true wherever the nominated
    breakpoints have a row; :class:`~math_spec.program.FirstOf` and
    :class:`~math_spec.program.LastOf` mark, per curve, the first and last
    breakpoint the mask admits.
    """
    if isinstance(derivation, ms.MaskOf):
        return parameters[derivation.values].notnull()
    mask = parameters[derivation.mask]
    over = program.piecewise[derivation.block].over
    ordinal = xr.DataArray(np.arange(mask.sizes[over]), dims=[over])
    if isinstance(derivation, ms.FirstOf):
        edge = ordinal.where(mask, np.inf).min(over)
    else:
        edge = ordinal.where(mask, -np.inf).max(over)
    return (mask & (ordinal == edge)).transpose(*mask.dims)


def validate(program: ms.Program, parameters: Mapping[str, xr.DataArray]) -> None:
    """
    Refuse curves the data does not supply everywhere they are built, or that bend against their method.

    Raises:
        SpecDataError: A breakpoint parameter with a hole where the block
            builds a weight, a ``points:`` mask that is not one run per curve,
            breakpoints that do not increase, a one-point curve under
            ``method: lp``, or a curve of the curvature the method is not
            exact for.
    """
    for block, decl in program.piecewise.items():
        run = _one(decl.checks, ms.Contiguous)
        mask = None
        if run is not None:
            mask = parameters[run.mask]
            _check_one_run(block, decl, run, mask)
        for values in decl.breakpoints:
            _check_extent(block, values, parameters[values], mask, run)
        curved = _one(decl.checks, ms.Curved)
        if curved is not None:
            _check_curves(block, decl, curved, parameters, mask)


def _one(checks: tuple[ms.Check, ...], kind: type[_C]) -> _C | None:
    return next((check for check in checks if isinstance(check, kind)), None)


def _check_extent(
    block: str,
    name: str,
    values: xr.DataArray,
    mask: xr.DataArray | None,
    run: ms.Contiguous | None,
) -> None:
    needed = (
        xr.ones_like(values, dtype=bool)
        if mask is None
        else mask.any([d for d in mask.dims if d not in values.dims])
    )
    holes = needed & values.isnull()
    if not bool(holes.any()):
        return
    points = None if run is None else (run.values or run.mask)
    remedy = (
        f"  Shorten it    '{points}' claims this breakpoint, so either it is one row too long "
        f"or the value is missing\n"
        f"  Or supply it  a value everywhere the mask says the curve runs"
        if points
        else (
            "  Say how far   points: a mask over the curve, true up to each one's last "
            "breakpoint\n"
            "  Or supply it  a value at every coordinate of the axis"
        )
    )
    raise SpecDataError(
        f"piecewise '{block}': parameter '{name}' has no value at ({_first(holes)}), and every "
        f"breakpoint the block builds gets a weight, so a missing row is not a shorter "
        f"curve: read as a zero coefficient it is a breakpoint at the origin.\n{remedy}"
    )


def _check_one_run(
    block: str, decl: ms.PiecewiseDeclaration, run: ms.Contiguous, mask: xr.DataArray
) -> None:
    over = decl.over
    ordinal = xr.DataArray(np.arange(mask.sizes[over]), dims=[over])
    marked = mask.sum(over)
    span = (
        ordinal.where(mask, -np.inf).max(over)
        - ordinal.where(mask, np.inf).min(over)
        + 1
    )
    broken = (marked == 0) | (span != marked)
    if not bool(broken.any()):
        return
    message = ms.check_message(block, decl, run)
    if not broken.dims:
        raise SpecDataError(message)
    raise SpecDataError(f"{message}\n  Not so at {_first(broken)}")


def _first(flags: xr.DataArray) -> str:
    """The first coordinate *flags* is true at, written as the reader would look for it."""
    stacked = flags.stack(_at=flags.dims)
    at = stacked["_at"].to_index()[stacked.to_numpy()].tolist()[0]
    return ", ".join(f"{d}={v!r}" for d, v in zip(flags.dims, at))


def _check_curves(
    block: str,
    decl: ms.PiecewiseDeclaration,
    curved: ms.Curved,
    parameters: Mapping[str, xr.DataArray],
    mask: xr.DataArray | None,
) -> None:
    over = decl.over
    xs, ys = xr.broadcast(parameters[curved.x], parameters[curved.y])
    on_curve = xs.notnull() & ys.notnull()
    if mask is not None:
        on_curve = on_curve & mask
    xs, ys, on_curve = xr.broadcast(xs, ys, on_curve)
    frame = [d for d in xs.dims if d != over]
    x = xs.transpose(*frame, over).to_numpy().reshape(-1, xs.sizes[over])
    y = ys.transpose(*frame, over).to_numpy().reshape(-1, xs.sizes[over])
    keep = on_curve.transpose(*frame, over).to_numpy().reshape(-1, xs.sizes[over])
    increasing = _one(decl.checks, ms.Increasing)
    segment = _one(decl.checks, ms.AtLeastTwo)
    for row_x, row_y, row_keep in zip(x, y, keep):
        px, py = row_x[row_keep].astype(float), row_y[row_keep].astype(float)
        if segment is not None and px.size < 2:
            raise SpecDataError(
                f"{ms.check_message(block, decl, segment)}\n  This curve carries {px.size}"
            )
        dx = np.diff(px)
        if increasing is not None and not bool((dx > 0).all()):
            raise SpecDataError(
                f"{ms.check_message(block, decl, increasing)} (got {px.tolist()})"
            )
        if _bends_wrong(dx, np.diff(py), curved.curvature):
            raise SpecDataError(
                f"{ms.check_message(block, decl, curved)} (got {py.tolist()})"
            )


def _bends_wrong(dx: np.ndarray, dy: np.ndarray, curvature: str) -> bool:
    slopes = dy / dx
    bend = np.diff(slopes)
    tol = 1e-9 * float(np.abs(slopes).max(initial=0.0))
    rises, falls = bool((bend > tol).any()), bool((bend < -tol).any())
    if curvature == "either":
        return rises and falls
    return falls if curvature == "convex" else rises
