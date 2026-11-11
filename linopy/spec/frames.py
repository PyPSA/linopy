"""
A value read over another frame of the same plain dimensions, as a ``Reframe`` names it.

A multi-dimension's labels are tuples of its members' labels. Gathering
reads a value over members at each listed combination; scattering spreads a
value over a multi-dimension onto every combination of its members, absent
where the multi-dimension lists none. Both read through
:func:`~linopy.spec.operators.at`, the lookup a relation uses.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import xarray as xr

from linopy.spec import operators
from linopy.spec.context import Array, Context


def reframed(value: Array, target: Iterable[str], ctx: Context) -> Array:
    """*value* read over *target*: each multi-dimension it holds and *target* lacks scattered, then each one *target* holds and it lacks gathered."""
    target = tuple(target)
    held = operators.coord_dims(value)
    for dim in held:
        if dim not in target and _members(dim, ctx):
            value = _scattered(value, dim, ctx)
    for dim in target:
        if dim not in held and _members(dim, ctx):
            value = _gathered(value, dim, ctx)
    return value


def scattered_sum(value: Array, dim: str, summed: Iterable[str], ctx: Context) -> Array:
    """
    *value* over the multi-dimension *dim*, scattered onto its members and summed over *summed*.

    Each combination adds into the group of the members it keeps, so no grid
    over every combination of the members is built. A group no combination
    reaches holds the empty sum, 0, as the sum of the scatter does.
    """
    members = _members(dim, ctx)
    kept = [m for m in members if m not in set(summed)]
    if not kept:
        return operators.sum_over(value, dim)
    index = ctx.coords[dim]
    grouped = {m: _labels(index, members.index(m)) for m in kept}
    onto = {m: ctx.coords[m] for m in kept}
    return operators.grouped_sum(value, {dim: index.to_numpy()}, grouped, onto)


def scattering(
    node_dims: Iterable[str], operand_dims: Iterable[str], ctx: Context
) -> str | None:
    """The one multi-dimension a reading from *operand_dims* to *node_dims* scatters, when it does nothing else."""
    node_dims, operand_dims = set(node_dims), set(operand_dims)
    gone = [d for d in operand_dims - node_dims if _members(d, ctx)]
    come = [d for d in node_dims - operand_dims if _members(d, ctx)]
    return gone[0] if len(gone) == 1 and not come else None


def _members(dim: str, ctx: Context) -> tuple[str, ...]:
    declared = ctx.program.dimensions.get(dim)
    return () if declared is None else declared.over


def _labels(index: pd.Index, position: int) -> np.ndarray:
    """The label of the member at *position* in each combination of *index*."""
    out = np.empty(len(index), dtype=object)
    out[:] = [combination[position] for combination in index]
    return out


def _gathered(value: Array, dim: str, ctx: Context) -> Array:
    """*value* read at the member labels of each combination *dim* lists."""
    index = ctx.coords[dim]
    held = operators.coord_dims(value)
    positions = {
        m: xr.DataArray(value.indexes[m].get_indexer(_labels(index, i)), dims=[dim])
        for i, m in enumerate(_members(dim, ctx))
        if m in held
    }
    present = xr.DataArray(
        np.ones(len(index), dtype=bool), coords={dim: index}, dims=[dim]
    )
    return operators.at(value, positions, present)


def _scattered(value: Array, dim: str, ctx: Context) -> Array:
    """*value* spread onto every combination of the members of *dim*, absent where *dim* lists none."""
    members = _members(dim, ctx)
    index = ctx.coords[dim]
    onto = {m: ctx.coords[m] for m in members}
    shape = tuple(len(i) for i in onto.values())
    found = [onto[m].get_indexer(_labels(index, i)) for i, m in enumerate(members)]
    grid = np.full(int(np.prod(shape)), -1)
    grid[np.ravel_multi_index(found, shape)] = np.arange(len(index))
    grid = grid.reshape(shape)
    present = xr.DataArray(grid >= 0, coords=onto, dims=list(members))
    if not len(index):
        return operators.zeros(value, (dim,), onto).where(present)
    positions = {dim: xr.DataArray(np.maximum(grid, 0), dims=list(members))}
    return operators.at(value, positions, present)
