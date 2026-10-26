"""
The language's built-in operators, evaluated on xarray and linopy values.

Each entry point takes an operand that is already a value, a ``DataArray``
for data or a linopy term for anything carrying a variable, and returns the
same kind. Nothing here reads the program or the model: the builder
evaluates the operands and the keywords and calls in.
"""

from __future__ import annotations

import operator
import warnings
from collections.abc import Hashable
from dataclasses import dataclass
from functools import reduce
from typing import cast

import numpy as np
import pandas as pd
import xarray as xr

from linopy.constants import EvolvingAPIWarning
from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.long import Relation
from linopy.spec.context import Array
from linopy.spec.errors import SpecDataError
from linopy.variables import Variable

Amount = int | xr.DataArray


def filled(expression: Array, fill: float) -> Array:
    """*expression* with every absence in it standing as *fill*."""
    if isinstance(expression, Variable):
        expression = expression.to_linexpr()
    return expression.fillna(fill)


def vacated(
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
    return filled(shifted, fill).where(keep)


def sum_over(array: Array, over: str) -> Array:
    """Sum *array* over *over*; a term beside an empty dimension is built as the constant zero."""
    if not isinstance(array, xr.DataArray) and any(
        not array.sizes[dim] for dim in array.coord_dims if dim != over
    ):
        kept = [dim for dim in _coord_dims(array) if dim != over]
        return _zeros(array, kept, {dim: array.indexes[dim] for dim in kept})
    return array.sum(over)


def grouped_sum(
    array: Array, relation: xr.DataArray, *, into: str, labels: pd.Index
) -> Array:
    """
    Sum *array* through *relation*, replacing the relation's key dimension by *into*.

    A member the relation sends nowhere contributes nowhere. The result is
    put onto every declared label of *into*: a group no member reaches holds
    the empty sum, which is 0 and not an absence. A linear term is grouped through
    :class:`linopy.Relation`, a quadratic term through a membership
    matrix and data through xarray.
    """
    dim = str(relation.dims[0])
    present = relation.notnull()
    if not bool(present.any()):
        return _empty_groups(array, dim, into=into, labels=labels)
    member = dim if dim != into else f"{dim}__member"
    keep = present.to_numpy()
    members = np.asarray(relation.indexes[dim])[keep]
    groups = relation.to_numpy()[keep]
    if isinstance(array, QuadraticExpression):
        if member != dim:
            array = array.rename({dim: member})
        weights = xr.DataArray(
            (groups[:, None] == labels.to_numpy()[None, :]).astype(float),
            coords={member: members, into: labels},
            dims=[member, into],
        )
        return (array.sel({member: members}) * weights).sum(member)
    if isinstance(array, xr.DataArray):
        if member != dim:
            array = array.rename({dim: member})
        attached = array.sel({member: members}).assign_coords({into: (member, groups)})
        summed = attached.groupby(into).sum()
        return summed.reindex({into: labels}).fillna(0.0)
    expression = array.to_linexpr() if isinstance(array, Variable) else array
    if member != dim:
        expression = expression.rename({dim: member})
    frame = pd.DataFrame({member: members, into: groups})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EvolvingAPIWarning)
        by = Relation(frame, member, into, groups=labels)
    return expression.groupby(by).sum()


def _empty_groups(
    array: Array,
    dim: str,
    *,
    into: str,
    labels: pd.Index,
) -> Array:
    """
    The grouped sum of an operand no member of *dim* is mapped out of.

    Every declared group holds the empty sum, which is 0. Grouping cannot say
    so itself: filtering the operand down to its mapped members leaves nothing,
    and an empty dimension is one xarray refuses to group over.
    """
    kept = [d for d in _coord_dims(array) if d != dim]
    coords = {**{d: array.indexes[d] for d in kept}, into: labels}
    return _zeros(array, kept + [into], coords)


def _zeros(array: Array, dims: list[str], coords: dict[str, pd.Index]) -> Array:
    """The constant zero over *dims*, of the kind *array* is: data, or a term of its model."""
    zeros = xr.DataArray(
        np.zeros([len(coords[d]) for d in dims]),
        coords={d: coords[d] for d in dims},
        dims=dims,
    )
    if isinstance(array, xr.DataArray):
        return zeros
    return LinearExpression.from_constant(array.model, zeros)


def _coord_dims(array: Array) -> list[str]:
    """The dimensions the operand is labelled over, without a term's own ``_term``."""
    dims = array.dims if isinstance(array, xr.DataArray) else array.coord_dims
    return [str(d) for d in dims]


def at(array: Array, relation: xr.DataArray, *, into: str) -> Array:
    """
    Read *array* through *relation*: the adjoint of :func:`grouped_sum`.

    A member the relation sends nowhere reads nothing, and its row keeps the
    operand's own absence rather than a zero.
    """
    mapping = relation.rename(into)
    present = mapping.notnull()
    dim = str(mapping.dims[0])
    labels = mapping.indexes[dim]
    if bool(present.all()):
        return array.sel({into: mapping}).assign_coords({dim: labels})
    keep = present.to_numpy()
    picked = array.sel({into: mapping.isel({dim: keep})})
    return picked.assign_coords({dim: labels[keep]}).reindex({dim: labels})


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
    roster = np.zeros((len(counts), max(counts.max(), 1)), dtype=int)
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
    if by is not None:
        partition = grouped(over, np.asarray(array.indexes[over]), by)
        offset = _per_group(offset, by)
        return _gather_in_groups(array, over, offset, partition, wrap, fill)
    if isinstance(offset, xr.DataArray) and offset.ndim:
        return _gather_by_offset(array, over, offset, wrap, fill)
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
    if by is not None:
        within = _per_group(within, by)
    asked = _widest(within)
    size = int(array.sizes[over])
    if asked > size and not wrap:
        raise SpecDataError(
            f"a trailing sum over '{over}' asks for a window of {asked} position(s) where "
            f"'{over}' holds {size}. Narrow 'window' to the axis, or write edge='wrap' so the "
            f"window continues from the other end."
        )
    widest = max(1, min(asked, size))
    partition = (
        None if by is None else grouped(over, np.asarray(array.indexes[over]), by)
    )
    lagged_terms: list[Array] = []
    reached: list[xr.DataArray] = []
    for lag in range(widest):
        lagged = (
            _gather_by_offset(array, over, lag, wrap, None)
            if partition is None
            else _gather_in_groups(array, over, lag, partition, wrap, None)
        )
        live, term = ~lagged.isnull(), filled(lagged, 0.0)
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
    from linopy import merge

    return cast(LinearExpression, merge(cast(list[LinearExpression], values)))


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
    return vacated(moved, array, over, ~inside, fill)


def _per_group(offset: Amount, groups: xr.DataArray) -> Amount:
    """*offset* at every coordinate where it is declared over the group's own dimension."""
    target = str(groups.name)
    if not isinstance(offset, xr.DataArray) or target not in offset.dims:
        return offset
    return cast(xr.DataArray, at(offset, groups, into=target)).drop_vars(target)


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
    return vacated(gathered, array, over, groups.grouped & ~inside, fill)


def _labelled(labels: np.ndarray, ordinals: xr.DataArray) -> xr.DataArray:
    """*ordinals* as the labels they stand for, carrying no coordinates of their own."""
    return xr.DataArray(
        labels[ordinals.transpose(*ordinals.dims).values], dims=ordinals.dims
    )
