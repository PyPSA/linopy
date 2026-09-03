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
from typing import cast, overload

import numpy as np
import pandas as pd
import xarray as xr

from linopy.expressions import LinearExpression
from linopy.spec import terms
from linopy.spec.terms import Array, Term

Amount = int | xr.DataArray


def sum_over(array: Array, over: str) -> Array:
    """Sum *array* over *over*; a term beside an empty dimension is built as the constant zero."""
    if not isinstance(array, xr.DataArray) and any(
        not array.sizes[dim] for dim in array.coord_dims if dim != over
    ):
        kept = [dim for dim in array.coord_dims if dim != over]
        zeros = xr.DataArray(
            np.zeros([array.sizes[dim] for dim in kept]),
            coords={dim: array.indexes[dim] for dim in kept},
            dims=kept,
        )
        return LinearExpression.from_constant(array.model, zeros)
    return array.sum(over)


def grouped_sum(
    array: Array,
    mappings: tuple[xr.DataArray, ...],
    *,
    into: tuple[str, ...],
    labels: Mapping[str, pd.Index],
) -> Array:
    """
    Sum *array* through the lookups *mappings*, replacing their dimension by *into*.

    A member a lookup sends nowhere contributes nowhere. The result is put
    onto every declared label of *into*: a group no member reaches holds the
    empty sum, which is 0 and not an absence.
    """
    mappings = _renamed(mappings, into)
    present = _present(mappings)
    dim = str(mappings[0].dims[0])
    if not bool(present.all()):
        keep = present.to_numpy()
        mappings = tuple(m.isel({dim: keep}) for m in mappings)
        array = array.isel({dim: keep})
    attached = array.assign_coords(
        {target: (dim, m.to_numpy()) for target, m in zip(into, mappings)}
    )
    summed = attached.groupby(list(into)).sum()
    return summed.reindex({d: labels[d] for d in into}).fillna(0.0)


@overload
def at(
    array: xr.DataArray, mappings: tuple[xr.DataArray, ...], *, into: tuple[str, ...]
) -> xr.DataArray: ...


@overload
def at(
    array: Term, mappings: tuple[xr.DataArray, ...], *, into: tuple[str, ...]
) -> Term: ...


def at(
    array: Array, mappings: tuple[xr.DataArray, ...], *, into: tuple[str, ...]
) -> Array:
    """
    Read *array* through the lookups *mappings*: the adjoint of :func:`grouped_sum`.

    A member a lookup sends nowhere reads nothing, and its row keeps the
    operand's own absence rather than a zero.
    """
    mappings = _renamed(mappings, into)
    present = _present(mappings)
    if bool(present.all()):
        return array.sel(dict(zip(into, mappings)))
    dim = str(mappings[0].dims[0])
    keep = present.to_numpy()
    picked = array.sel(dict(zip(into, (m.isel({dim: keep}) for m in mappings))))
    return picked.reindex({dim: mappings[0][dim]})


@dataclass(frozen=True)
class _Edge:
    wrap: bool
    fill: float | None


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
    is an array differs per entity and is a gather. *by* is the lookup whose
    groups the translation stays inside.
    """
    edge = _Edge(wrap, fill)
    if by is not None:
        groups = _grouped(over, np.asarray(array.indexes[over]), by)
        return _gather_in_groups(array, over, _per_group(offset, by), groups, edge)
    if isinstance(offset, xr.DataArray) and offset.ndim:
        return _gather_by_offset(array, over, offset, edge)
    amount: dict[Hashable, int] = {over: int(offset)}
    if wrap:
        if isinstance(array, xr.DataArray):
            return array.roll(amount, roll_coords=False)
        return array.roll(amount)
    if isinstance(array, xr.DataArray):
        return array.shift(amount, fill_value=np.nan if fill is None else fill)
    shifted = array.shift(amount)
    if fill is None:
        return shifted
    return terms.vacated(
        shifted, array, over, _off_the_axis(array, over, amount[over]), fill
    )


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
    """
    if by is not None:
        within = _per_group(within, by)
    asked = (
        int(np.nanmax(np.asarray(within)))
        if isinstance(within, xr.DataArray)
        else int(within)
    )
    widest = max(1, min(asked, int(array.sizes[over])))
    probe = _Edge(wrap=wrap, fill=None)
    groups = None if by is None else _grouped(over, np.asarray(array.indexes[over]), by)
    lagged_terms: list[Array] = []
    reached: list[xr.DataArray] = []
    for lag in range(widest):
        lagged = (
            _gather_by_offset(array, over, lag, probe)
            if groups is None
            else _gather_in_groups(array, over, lag, groups, probe)
        )
        live, term = ~lagged.isnull(), terms.filled(lagged, 0.0)
        if isinstance(within, xr.DataArray):
            live, term = live & (within > lag), term * (within > lag).astype(float)
        lagged_terms.append(term)
        reached.append(live)
    return _merged(lagged_terms).where(reduce(operator.or_, reached))


def _merged(values: list[Array]) -> Array:
    """The sum of *values* in one step: a running sum would re-concatenate the term axis once per lag."""
    data = [value for value in values if isinstance(value, xr.DataArray)]
    if len(data) == len(values):
        return reduce(operator.add, data)
    from linopy import merge

    held = [value for value in values if not isinstance(value, xr.DataArray)]
    return cast(LinearExpression, merge(held))


def _renamed(
    mappings: tuple[xr.DataArray, ...], into: tuple[str, ...]
) -> tuple[xr.DataArray, ...]:
    return tuple(mapping.rename(target) for mapping, target in zip(mappings, into))


def _present(mappings: tuple[xr.DataArray, ...]) -> xr.DataArray:
    return reduce(operator.and_, (m.notnull() for m in mappings))


def _gather_by_offset(array: Array, over: str, offset: Amount, edge: _Edge) -> Array:
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

    if edge.wrap:
        return gathered(source % card)
    inside = ((source >= 0) & (source < card)).assign_coords({over: labels})
    moved = gathered(source.clip(0, card - 1)).where(inside)
    if edge.fill is None:
        return moved
    return terms.vacated(moved, array, over, ~inside, edge.fill)


def _per_group(offset: Amount, groups: xr.DataArray) -> Amount:
    """*offset* at every coordinate where it is declared over the group's own dimension."""
    target = groups.name
    if not isinstance(offset, xr.DataArray) or target not in offset.dims:
        return offset
    return at(offset, (groups,), into=(str(target),)).drop_vars(str(target))


@dataclass(frozen=True)
class _Groups:
    labels: np.ndarray
    grouped: xr.DataArray
    belongs: xr.DataArray
    within: xr.DataArray
    size: xr.DataArray
    roster: np.ndarray
    names: tuple[object, ...]
    counts: tuple[int, ...]


def _grouped(over: str, labels: np.ndarray, groups: xr.DataArray) -> _Groups:
    """
    How the lookup *groups* partitions the axis *over*.

    A coordinate the lookup sends nowhere belongs to no group: its ``within``
    is 0, its ``size`` 1 and its ``grouped`` False.
    """
    keys = np.asarray(groups.sel({over: labels}).values, dtype=object)
    peers: dict[object, list[int]] = {}
    within = np.zeros(len(labels), dtype=int)
    grouped = np.zeros(len(labels), dtype=bool)
    for k, key in enumerate(keys):
        if terms.unmapped(key):
            continue
        grouped[k] = True
        beside = peers.setdefault(key, [])
        within[k] = len(beside)
        beside.append(k)
    order = {key: g for g, key in enumerate(peers)}
    widest = max((len(beside) for beside in peers.values()), default=1)
    roster = np.zeros((max(len(peers), 1), widest), dtype=int)
    for key, beside in peers.items():
        roster[order[key], : len(beside)] = beside
    belongs = np.array([order.get(key, 0) for key in keys], dtype=int)
    span = np.array(
        [len(peers[key]) if held else 1 for key, held in zip(keys, grouped)], dtype=int
    )

    def on_axis(values: np.ndarray) -> xr.DataArray:
        return xr.DataArray(values, coords={over: labels}, dims=[over])

    return _Groups(
        labels,
        on_axis(grouped),
        on_axis(belongs),
        on_axis(within),
        on_axis(span),
        roster,
        tuple(peers),
        tuple(len(beside) for beside in peers.values()),
    )


def _gather_in_groups(
    array: Array, over: str, offset: Amount, groups: _Groups, edge: _Edge
) -> Array:
    """
    Translate *array* inside each group rather than along the axis.

    A coordinate in no group reaches nothing, which is not the same as
    reaching off a group's edge: only the second is what a fill speaks for.
    """
    reached = groups.within - offset
    if edge.wrap:
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
    if edge.fill is None:
        return gathered
    return terms.vacated(gathered, array, over, groups.grouped & ~inside, edge.fill)


def _off_the_axis(array: Array, over: str, offset: int) -> xr.DataArray:
    labels = np.asarray(array.indexes[over])
    source = xr.DataArray(np.arange(len(labels)), coords={over: labels}, dims=[over])
    source = source - offset
    return (source < 0) | (source >= len(labels))


def _labelled(labels: np.ndarray, ordinals: xr.DataArray) -> xr.DataArray:
    """*ordinals* as the labels they stand for, carrying no coordinates of their own."""
    return xr.DataArray(
        labels[ordinals.transpose(*ordinals.dims).values], dims=ordinals.dims
    )
