"""
The language's built-in operators, evaluated on xarray and linopy values.

Each entry point takes an operand that is already a value, a ``DataArray``
for data or a linopy term for anything carrying a variable, and returns the
same kind. Nothing here reads the program or the model: the builder
evaluates the operands and the keywords and calls in.
"""

from __future__ import annotations

import operator
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from functools import reduce
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr

from linopy.expressions import LinearExpression, QuadraticExpression, merge
from linopy.spec.context import Array, as_expression
from linopy.spec.errors import SpecDataError

Amount = int | xr.DataArray
_ROW = "_row"


def _filled(expression: Array, fill: float) -> Array:
    """*expression* with every absence in it standing as *fill*."""
    return as_expression(expression).fillna(fill)


def _vacated(
    shifted: Array, operand: Array, over: str, vacated: xr.DataArray, fill: float
) -> Array:
    """
    *shifted*, with the positions the shift vacated filled, and only those.

    The fill lands where the shift vacated and the operand carries the
    coordinate; every other slot keeps the absence it arrived with, so no row
    is invented at a coordinate the operand never had.
    """
    carried = (~operand.isnull()).any(over)
    keep = carried & (~shifted.isnull() | vacated)
    return _filled(shifted, fill).where(keep)


def sum_over(array: Array, over: str) -> Array:
    """Sum *array* over *over*; a term beside an empty dimension is built as the constant zero."""
    if not isinstance(array, xr.DataArray) and any(
        not array.sizes[dim] for dim in array.coord_dims if dim != over
    ):
        return _zeros(array, (over,))
    return array.sum(over)


def grouped_sum(
    array: Array,
    joined: Mapping[str, np.ndarray],
    grouped: Mapping[str, np.ndarray],
    onto: Mapping[str, pd.Index],
) -> Array:
    """
    Sum *array* through the rows of a relation: join, then group by.

    Row *i* of the relation reads *array* where each dimension of *joined*
    holds its label ``joined[d][i]``, and adds it to the group at the labels
    ``grouped[d][i]``. The result is put onto every label of *onto*, the
    grouped dimensions: a group no row reaches holds the empty sum, which is
    0 and not an absence. A row lands in as many groups as the relation lists
    it, so a bare relation sums as a keyed one does.
    """
    rows = pd.RangeIndex(len(next(iter(joined.values()))), name=_ROW)
    if not len(rows):
        return _zeros(array, tuple(joined), onto)
    expression = as_expression(array)
    positions = {
        d: xr.DataArray(expression.indexes[d].get_indexer(labels), coords=[rows])
        for d, labels in joined.items()
    }
    picked = expression.isel(positions)
    if isinstance(picked, xr.DataArray):
        return _scattered(picked, grouped, onto)
    keyed = picked.assign_coords({d: (_ROW, labels) for d, labels in grouped.items()})
    summed = keyed.groupby(list(grouped)).sum()
    if isinstance(expression, QuadraticExpression):
        summed = QuadraticExpression(summed.data, summed.model)
    return summed.reindex(onto).fillna(0)


def _scattered(
    picked: xr.DataArray,
    grouped: Mapping[str, np.ndarray],
    onto: Mapping[str, pd.Index],
) -> xr.DataArray:
    """The rows of *picked* added into their groups, a hole adding nothing."""
    shape = [len(onto[d]) for d in grouped]
    at = [onto[d].get_indexer(labels) for d, labels in grouped.items()]
    rows = picked.fillna(0).transpose(_ROW, ...)
    others = [str(d) for d in rows.dims[1:]]
    flat = np.zeros((int(np.prod(shape)), *rows.shape[1:]))
    np.add.at(flat, np.ravel_multi_index(at, shape), rows.to_numpy())
    coords = {**{d: onto[d] for d in grouped}, **{d: rows.indexes[d] for d in others}}
    return xr.DataArray(
        flat.reshape(*shape, *rows.shape[1:]), coords=coords, dims=list(coords)
    )


def _zeros(
    array: Array, without: tuple[str, ...], extra: Mapping[str, pd.Index] | None = None
) -> Array:
    """The constant zero over the operand's dimensions but *without*, and *extra*: data, or a term of its model."""
    kept = [d for d in _coord_dims(array) if d not in without]
    coords = {**{d: array.indexes[d] for d in kept}, **(extra or {})}
    zeros = xr.DataArray(
        np.zeros([len(index) for index in coords.values()]),
        coords=coords,
        dims=list(coords),
    )
    if isinstance(array, xr.DataArray):
        return zeros
    return LinearExpression.from_constant(array.model, zeros)


def _coord_dims(array: Array) -> list[str]:
    """The dimensions the operand is labelled over, without a term's own ``_term``."""
    dims = array.dims if isinstance(array, xr.DataArray) else array.coord_dims
    return [str(d) for d in dims]


def at(
    array: Array, positions: Mapping[str, xr.DataArray], present: xr.DataArray
) -> Array:
    """
    Read *array* at *positions*: the adjoint of :func:`grouped_sum`.

    *positions* holds, for each dimension *array* is read along, the position
    to read over the relation's key. A key tuple *present* leaves out reads
    nothing, and its row keeps the operand's own absence rather than a zero.
    """
    picked = as_expression(array).isel(positions)
    joined = [d for d in positions if d not in picked.dims]
    picked = picked.drop_vars(joined).assign_coords(present.coords)
    if bool(present.all()):
        return picked
    return picked.where(present)


def positions_of(labels: xr.DataArray, index: pd.Index) -> xr.DataArray:
    """The position in *index* of each of *labels*, 0 at a hole, which :func:`at` masks."""
    found = index.get_indexer(labels.to_numpy().ravel()).reshape(labels.shape)
    return xr.DataArray(np.maximum(found, 0), dims=labels.dims)


@dataclass(frozen=True)
class Groups:
    """How a relation partitions an axis: the shape every group-wise operator reads."""

    labels: np.ndarray
    grouped: xr.DataArray
    belongs: xr.DataArray
    within: xr.DataArray
    size: xr.DataArray
    roster: np.ndarray
    names: np.ndarray
    counts: np.ndarray


def grouped(over: str, labels: np.ndarray, groups: xr.DataArray) -> Groups:
    """
    How the relation *groups* partitions the axis *over*, groups in order of first appearance.

    A coordinate the relation sends nowhere belongs to no group: its ``within``
    is 0, its ``size`` 1 and its ``grouped`` False.
    """
    codes, names = pd.factorize(groups.sel({over: labels}).to_numpy())
    held = codes >= 0
    belongs = np.where(held, codes, 0)
    within = np.where(held, pd.Series(codes).groupby(codes).cumcount().to_numpy(), 0)
    counts = np.bincount(codes[held], minlength=max(len(names), 1))
    roster = np.zeros((len(counts), max(int(counts.max()), 1)), dtype=int)
    roster[codes[held], within[held]] = np.flatnonzero(held)

    def on_axis(values: np.ndarray) -> xr.DataArray:
        return xr.DataArray(values, coords={over: labels}, dims=[over])

    return Groups(
        labels,
        on_axis(held),
        on_axis(belongs),
        on_axis(within),
        on_axis(np.where(held, counts[belongs], 1)),
        roster,
        np.asarray(names),
        counts,
    )


def shift(
    array: Array,
    *,
    over: str,
    offset: Amount,
    wrap: bool,
    fill: float | None,
    by: xr.DataArray | None = None,
) -> Array:
    """
    Translate *array* along *over*: the value at ``t - offset``.

    *wrap* is cyclic and vacates nothing, *fill* is what the vacated
    positions contribute, and neither leaves them absent. An *offset* that
    is an array differs per entity and is a gather. *by* is the relation
    whose groups the translation stays inside.
    """
    offset, partition = _grouping(array, over, offset, by)
    if partition is not None or (isinstance(offset, xr.DataArray) and offset.ndim):
        return _gather(array, over, offset, partition, wrap, fill)
    amount: dict[Hashable, int] = {over: int(offset)}
    if wrap:
        if isinstance(array, xr.DataArray):
            return array.roll(amount, roll_coords=False)
        return array.roll(amount)
    if isinstance(array, xr.DataArray):
        return array.shift(amount, fill_value=np.nan if fill is None else fill)
    if fill is None:
        return array.shift(amount)
    return _gather_by_offset(array, over, offset, wrap, fill)


def sum_back(
    array: Array,
    *,
    over: str,
    within: Amount,
    wrap: bool,
    by: xr.DataArray | None = None,
) -> Array:
    """
    Sum *array* over a trailing window along *over*: positions ``t - within + 1`` through ``t``.

    A position the window cannot reach contributes a zero; a window that
    reaches nothing keeps no row. *by* stops the window at each group's edge.

    With *wrap* the window continues from the other end of *over*, so a width
    above the axis size still reads every position and is taken as the whole
    axis. Without it such a width asks for positions that do not exist and is
    refused.
    """
    within, partition = _grouping(array, over, within, by)
    asked = _widest(within)
    size = int(array.sizes[over])
    if asked > size and not wrap:
        raise SpecDataError(
            f"a trailing sum over '{over}' asks for a window of {asked} position(s) where "
            f"'{over}' holds {size}. Narrow 'window' to the axis, or write edge='wrap' so the "
            f"window continues from the other end."
        )
    widest = max(1, min(asked, size))
    lagged_terms: list[Array] = []
    reached: list[xr.DataArray] = []
    for lag in range(widest):
        lagged = _gather(array, over, lag, partition, wrap, None)
        live, term = ~lagged.isnull(), _filled(lagged, 0.0)
        if isinstance(within, xr.DataArray):
            live, term = live & (within > lag), term * (within > lag).astype(float)
        lagged_terms.append(term)
        reached.append(live)
    return _merged(lagged_terms).where(reduce(operator.or_, reached))


def _widest(within: Amount) -> int:
    """The widest window the data asks for; a width no member carries is a window of nothing."""
    if not isinstance(within, xr.DataArray):
        return int(within)
    widths = np.asarray(within, dtype=float)
    return 0 if np.isnan(widths).all() else int(np.nanmax(widths))


def _merged(values: list[Array]) -> Array:
    """The sum of *values* in one step: a running sum would re-concatenate the term axis once per lag."""
    if isinstance(values[0], xr.DataArray):
        return reduce(operator.add, values)
    return cast(LinearExpression, merge(cast(list[LinearExpression], values)))


def _grouping(
    array: Array, over: str, amount: Amount, by: xr.DataArray | None
) -> tuple[Amount, Groups | None]:
    """*amount* read per group, beside how *by* partitions *over*; no relation keeps the axis whole."""
    if by is None:
        return amount, None
    return per_group(amount, by), grouped(over, np.asarray(array.indexes[over]), by)


def _gather(
    array: Array,
    over: str,
    offset: Amount,
    partition: Groups | None,
    wrap: bool,
    fill: float | None,
) -> Array:
    """*array* translated by *offset* along *over*, or inside each group of *partition*."""
    if partition is None:
        return _gather_by_offset(array, over, offset, wrap, fill)
    return _gather_in_groups(array, over, offset, partition, wrap, fill)


def _gather_by_offset(
    array: Array, over: str, offset: Amount, wrap: bool, fill: float | None
) -> Array:
    """
    Translate *array* along *over* by an offset that may differ per entity.

    Selection is by label, so a non-integer axis works. Out-of-range
    positions are clipped onto the axis and emptied again, so an edge means
    what it does for a scalar shift.
    """
    card = int(array.sizes[over])
    labels = np.asarray(array.indexes[over])
    ordinal = xr.DataArray(np.arange(card), coords={over: labels}, dims=[over])
    source = (ordinal - offset).astype(int)

    def gathered(ordinals: xr.DataArray) -> Array:
        picked = array.sel({over: _labelled(labels, ordinals)})
        return picked.assign_coords({over: labels})

    if wrap:
        return gathered(source % card)
    inside = ((source >= 0) & (source < card)).assign_coords({over: labels})
    moved = gathered(source.clip(0, card - 1)).where(inside)
    if fill is None:
        return moved
    return _vacated(moved, array, over, ~inside, fill)


def per_group(offset: Amount, groups: xr.DataArray) -> Amount:
    """*offset* at every coordinate where it is declared over the group's own dimension."""
    target = str(groups.name)
    if not isinstance(offset, xr.DataArray) or target not in offset.dims:
        return offset
    positions = {target: positions_of(groups, offset.indexes[target])}
    return cast(xr.DataArray, at(offset, positions, groups.notnull()))


def _gather_in_groups(
    array: Array,
    over: str,
    offset: Amount,
    groups: Groups,
    wrap: bool,
    fill: float | None,
) -> Array:
    """
    Translate *array* inside each group rather than along the axis.

    A coordinate in no group reaches nothing, which is not the same as
    reaching off a group's edge: only the second is what a fill speaks for.
    """
    reached = groups.within - offset
    if wrap:
        reached = reached % groups.size
    inside = groups.grouped & (reached >= 0) & (reached < groups.size)

    def peer(group: np.ndarray, position: np.ndarray) -> np.ndarray:
        return groups.roster[group, position]

    source = xr.apply_ufunc(peer, groups.belongs, reached.where(inside, 0).astype(int))
    labels = groups.labels
    gathered = (
        array.sel({over: _labelled(labels, source)})
        .assign_coords({over: labels})
        .where(inside)
    )
    if fill is None:
        return gathered
    return _vacated(gathered, array, over, groups.grouped & ~inside, fill)


def _labelled(labels: np.ndarray, ordinals: xr.DataArray) -> xr.DataArray:
    """*ordinals* as the labels they stand for, carrying no coordinates of their own."""
    return xr.DataArray(labels[ordinals.values], dims=ordinals.dims)
