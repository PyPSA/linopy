"""
The language's built-in operators, evaluated on xarray and linopy values.

Each entry point takes an operand that is already a value, a ``DataArray``
for data or a linopy term for anything carrying a variable, and returns the
same kind. Nothing here reads the program or the model: the builder
evaluates the operands and the keywords and calls in.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from typing import Any, cast

import numpy as np
import pandas as pd
import scipy.sparse
import xarray as xr

from linopy.constants import TERM_DIM
from linopy.csr import (
    CSRStore,
    Grid,
    coo_to_csr,
    group_order,
    indptr_from_counts,
)
from linopy.expressions import LinearExpression, merge
from linopy.model import Model
from linopy.spec.context import Array, as_expression
from linopy.spec.errors import SpecDataError
from linopy.variables import Variable

Amount = int | xr.DataArray
Weight = tuple[Callable[[Any, Any], Any], float | xr.DataArray]
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
    """
    Sum *array* over *over*; a term beside an empty dimension is built as the constant zero.

    A hole in data adds nothing. An array that can hold one is filled with 0
    before summing, which copies it once, where xarray's NaN-skipping sum
    copies it twice.
    """
    if isinstance(array, xr.DataArray):
        if array.dtype.kind in "fcO":
            array = array.fillna(0)
        return array.sum(over, skipna=False)
    if any(not array.sizes[dim] for dim in array.coord_dims if dim != over):
        return _zeros(array, (over,))
    summed = array.sum(over)
    if not isinstance(array, LinearExpression) or type(summed) is not LinearExpression:
        return summed
    return _const_summed(summed, array.const, over)


def _const_summed(
    summed: LinearExpression, const: xr.DataArray, over: str
) -> LinearExpression:
    """
    *summed* with its constant the sum of *const* over *over*, a hole adding nothing.

    Each cell adds its values up along *over* in coordinate order, on a
    contiguous copy, so the sum does not depend on how the operand's
    constant is laid out: a term and the constant term of the same data
    sum to the same bits.
    """
    store = summed._store
    laid = const.transpose(*store.dims, over).values
    values = np.ascontiguousarray(np.where(np.isnan(laid), 0.0, laid))
    flat = values.reshape(-1, values.shape[-1]).sum(axis=-1)
    return LinearExpression._from_store(store.assign_const(flat), summed.model)


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
    read = {d: array.indexes[d].get_indexer(labels) for d, labels in joined.items()}
    if isinstance(array, Variable) or type(array) is LinearExpression:
        aggregated = _aggregated(array, read, grouped, onto)
        if aggregated is not None:
            return aggregated
    expression = as_expression(array)
    positions = {d: xr.DataArray(at, coords=[rows]) for d, at in read.items()}
    picked = expression.isel(positions)
    if isinstance(picked, xr.DataArray):
        return _scattered(picked, grouped, onto)
    if isinstance(picked, LinearExpression) and not picked.is_sparse:
        return _gathered(picked, grouped, onto)
    keyed = picked.assign_coords({d: (_ROW, labels) for d, labels in grouped.items()})
    return keyed.groupby(list(grouped)).sum().reindex(onto).fillna(0)


def weighted_sum(
    term: Variable | LinearExpression,
    weights: Sequence[Weight],
    live: xr.DataArray,
    onto: Mapping[str, pd.Index],
    joined: Mapping[str, np.ndarray] | None = None,
    grouped: Mapping[str, np.ndarray] | None = None,
) -> LinearExpression | None:
    """
    The sum of *term* scaled by data, built from the cells where the data is *live* alone.

    *weights* hold the operator and the data of each scaling, the innermost
    first, the data a number or an array over dimensions of *live*. A cell
    of the product is read where *live* holds, over every coordinate of the
    dimensions of *term* that *live* lacks; no other cell adds a term. The
    cell lands at its own coordinates on the dimensions of *onto*, the
    result's, and every other dimension is summed away. With *joined*, the
    position each row of a relation reads on the dimensions it joins, a
    cell lands where *grouped* puts its row, and one no row reads lands
    nowhere: a key is read by one row at most. The constant is zero, and a
    term with a zero coefficient is left out. ``None`` where a coefficient
    overflows.
    """
    counts, labels, coeffs, held, shape = _live_terms(term)
    wide = [str(d) for d in live.dims]
    own = [d for d in held if d not in wide]
    size = {**dict(zip(held, shape)), **{d: live.sizes[d] for d in wide}}
    size.update({d: len(index) for d, index in onto.items()})
    cells = np.flatnonzero(live.values)
    spread = math.prod(size[d] for d in own)
    at = np.unravel_index(np.repeat(cells, spread), live.shape)
    positions = dict(zip(wide, at))
    if own:
        beside = np.unravel_index(np.arange(spread), [size[d] for d in own])
        positions.update((d, np.tile(p, len(cells))) for d, p in zip(own, beside))
    if joined is not None and grouped is not None:
        keys = list(joined)
        row = np.full(math.prod(size[d] for d in keys), -1, dtype=np.intp)
        row[_flat(joined, keys, size)] = np.arange(len(joined[keys[0]]))
        read = row[_flat(positions, keys, size)]
        kept = np.flatnonzero(read >= 0)
        positions = {d: p[kept] for d, p in positions.items()}
        positions.update((d, p[read[kept]]) for d, p in grouped.items())
    source = _flat(positions, held, size)
    target = _flat(positions, list(onto), size)
    indptr = indptr_from_counts(counts, np.intp)
    n = counts[source]
    picked = np.repeat(indptr[source] - (np.cumsum(n) - n), n) + np.arange(n.sum())
    values = coeffs[picked]
    with np.errstate(over="ignore"):
        for op, data in weights:
            if isinstance(data, xr.DataArray):
                at_term = tuple(np.repeat(positions[str(d)], n) for d in data.dims)
                values = op(values, data.values[at_term])
            else:
                values = op(values, data)
    if not np.isfinite(values).all():
        return None
    model = term.model
    grid = Grid({d: index.rename(d) for d, index in onto.items()})
    shape = (grid.size, model._xCounter)
    csr = coo_to_csr(values, np.repeat(target, n), labels[picked], shape, model)
    store = CSRStore(csr, np.zeros(grid.size), grid, model)
    return LinearExpression._from_store(
        store if model.sparse else store.to_dense(), model
    )


def _flat(
    positions: Mapping[str, np.ndarray], dims: Sequence[str], size: Mapping[str, int]
) -> np.ndarray:
    """The flat cell over *dims* at *positions*, every array of which is as long."""
    if not dims:
        return np.zeros(len(next(iter(positions.values()))), dtype=np.intp)
    at = [positions[d] for d in dims]
    return np.ravel_multi_index(at, [size[d] for d in dims])


def _live_terms(
    term: Variable | LinearExpression,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], tuple[int, ...]]:
    """The live terms of *term* in cell order: their number per cell, labels and coefficients, and the dimensions and shape of the cells."""
    if isinstance(term, Variable):
        cells = term.labels
        labels = cells.values.reshape(-1)
        present = labels != -1
        dims = [str(d) for d in cells.dims]
        ones = np.ones(np.count_nonzero(present))
        return present.astype(np.intp), labels[present], ones, dims, cells.shape
    store = term._store
    counts, labels, coeffs = store.terms()
    shape = tuple(term.sizes[d] for d in store.dims)
    return counts.astype(np.intp, copy=False), labels, coeffs, list(store.dims), shape


def _aggregated(
    array: Variable | LinearExpression,
    read: Mapping[str, np.ndarray],
    grouped: Mapping[str, np.ndarray],
    onto: Mapping[str, pd.Index],
) -> LinearExpression | None:
    """
    A sparse (CSR) term, or a variable of a sparse model, summed through the rows of a relation in one aggregate.

    Row *i* reads the cells at the positions ``read[d][i]`` and lands at the
    labels ``grouped[d][i]``; the grouped dimensions stand where the first
    joined one stood. The rows are gathered in the order of the cells they
    land in, so the terms are copied once, and a cell adds its rows'
    constants up in the order of the rows. ``None`` for a dense term, where
    a row names a label the coordinates lack, and where the term holds
    more coordinates than its dimensions.
    """
    model = array.model
    if isinstance(array, Variable):
        dims = tuple(str(d) for d in array.labels.dims)
        if not model.sparse or set(array.coords) != set(dims):
            return None
        grid = Grid({d: array.indexes[d].rename(d) for d in dims})
    elif array.is_sparse:
        grid = cast(CSRStore, array._store).grid
    else:
        return None
    lands = {d: onto[d].get_indexer(labels) for d, labels in grouped.items()}
    known = all((at >= 0).all() for at in (*read.values(), *lands.values()))
    others = [d for d in grid.dims if d not in read]
    if grid.aux or not known or set(grouped) & set(others):
        return None
    slot = sum(d in others for d in grid.dims[: min(map(grid.dims.index, read))])
    kept = {d: grid.indexes[d] for d in others}
    framed = [*others[:slot], *grouped, *others[slot:]]
    target = Grid({d: onto[d].rename(d) if d in grouped else kept[d] for d in framed})
    beside = dict(zip(others, (np.arange(len(index)) for index in kept.values())))
    source = _cells(grid.strides, read, beside)
    cells = _cells(target.strides, lands, beside)
    shape = (target.size, model._xCounter)
    if isinstance(array, Variable):
        labels = array.labels.values.reshape(-1)[source]
        held = labels != -1
        ones = np.ones(np.count_nonzero(held))
        csr = coo_to_csr(ones, cells[held], labels[held], shape, model)
        const = np.zeros(target.size)
    else:
        store = cast(CSRStore, array._store)
        order, bounds = group_order(cells, target.size)
        taken = store.csr[source[order]]
        arrays = (taken.data, taken.indices, taken.indptr[bounds])
        csr = scipy.sparse.csr_array(arrays, shape=shape)
        weights = store.const[source]
        weights = np.where(np.isnan(weights), 0.0, weights)
        const = np.bincount(cells, weights=weights, minlength=target.size)
    return LinearExpression._from_store(CSRStore(csr, const, target, model), model)


def _cells(
    strides: Mapping[str, int],
    rows: Mapping[str, np.ndarray],
    beside: Mapping[str, np.ndarray],
) -> np.ndarray:
    """The flat cell of every row at every position *beside* it, the rows varying slowest."""
    at = sum(positions * strides[d] for d, positions in rows.items())
    cells = np.asarray(at, dtype=np.int64)
    for d, positions in beside.items():
        cells = cells[..., None] + positions * strides[d]
    return cells.reshape(-1)


def _gathered(
    picked: LinearExpression,
    grouped: Mapping[str, np.ndarray],
    onto: Mapping[str, pd.Index],
) -> LinearExpression:
    """The rows of *picked* gathered into their groups, each row's terms in row order behind the term slot it fills."""
    shape = [len(onto[d]) for d in grouped]
    at = np.ravel_multi_index(
        [onto[d].get_indexer(labels) for d, labels in grouped.items()], shape
    )
    cells = int(np.prod(shape))
    counts = np.bincount(at, minlength=cells)
    order = np.argsort(at, kind="stable")
    slot = np.empty_like(at)
    slot[order] = np.arange(len(at)) - (np.cumsum(counts) - counts)[at[order]]
    data = picked.data.transpose(_ROW, ..., TERM_DIM)
    others = [str(d) for d in data.coeffs.dims[1:-1]]
    sizes = [data.sizes[d] for d in others]
    nterm, depth = data.sizes[TERM_DIM], int(counts.max())
    k = list(picked.coord_dims).index(_ROW)
    behind = (list(range(len(shape))), list(range(k, k + len(shape))))
    framed = [*others[:k], *grouped, *others[k:]]
    variables: dict[str, tuple[list[str], np.ndarray]] = {}
    for name, fill in (("coeffs", np.nan), ("vars", -1)):
        values = data[name].to_numpy()
        out = np.full((cells, *sizes, nterm, depth), fill, dtype=values.dtype)
        out[at, ..., slot] = values
        out = out.reshape(*shape, *sizes, nterm * depth)
        variables[name] = ([*framed, TERM_DIM], np.moveaxis(out, *behind))
    const = np.zeros((cells, *sizes))
    np.add.at(const, at, np.nan_to_num(data.const.transpose(_ROW, ...).to_numpy()))
    const = np.moveaxis(const.reshape(*shape, *sizes), *behind)
    variables["const"] = (framed, const)
    coords = {d: onto[d] if d in grouped else data.indexes[d] for d in framed}
    return LinearExpression(xr.Dataset(variables, coords=coords), picked.model)


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
    return constant_term(array.model, zeros)


def constant_term(model: Model, constant: xr.DataArray) -> LinearExpression:
    """
    *constant* as a term of *model* without a variable, on the model's store.

    It is laid out the way a term is, so nothing is aligned, which makes it
    cheaper than :meth:`~linopy.expressions.LinearExpression.from_constant`.
    """
    dims = (*constant.dims, TERM_DIM)
    slots = np.empty((*constant.shape, 0))
    labels = slots.astype(model.dtypes["labels"])
    data = constant.to_dataset(name="const").assign(
        coeffs=(dims, slots), vars=(dims, labels)
    )
    term = LinearExpression(data, model)
    if not model.sparse:
        return term
    store = term._store.to_sparse()
    return term if isinstance(store, str) else term._from_store(store, model)


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
    expression = as_expression(array)
    if any(not len(expression.indexes[d]) for d in positions):
        onto = {str(d): present.indexes[d] for d in present.dims}
        return _zeros(expression, tuple(positions), onto).where(present)
    if isinstance(expression, LinearExpression) and expression.is_sparse:
        if set(positions) & set(present.dims):
            return _at_by_rows(expression, positions, present)
    if isinstance(expression, xr.DataArray):
        expression = expression.drop_vars(
            [d for d in positions if d in expression.coords]
        )
    picked = expression.isel(_keyed(positions, present))
    joined = [d for d in positions if d not in picked.dims and d in picked.coords]
    picked = picked.drop_vars(joined).assign_coords(present.coords)
    if bool(present.all()):
        return picked
    return picked.where(present)


def nothing_read(
    array: Array, along: tuple[str, ...], onto: Mapping[str, pd.Index]
) -> Array:
    """:func:`at` of *array* read *along* dimensions through a relation keyed *onto* that has no row: absent throughout."""
    return _zeros(as_expression(array), along, onto).where(xr.DataArray(False))


def _at_by_rows(
    expression: LinearExpression,
    positions: Mapping[str, xr.DataArray],
    present: xr.DataArray,
) -> LinearExpression:
    """
    :func:`at` of a sparse term read along a dimension of its own key, as a sum of one row per key tuple.

    Read in place, the dimension keeps the labels it is read at as a
    coordinate over the whole key, which a sparse (CSR) term cannot hold.
    """
    keys = [str(d) for d in present.dims]
    held = np.nonzero(present.transpose(*keys).to_numpy())
    grouped = {k: present.indexes[k][at_k].to_numpy() for k, at_k in zip(keys, held)}
    joined = {k: grouped[k] for k in keys if k in expression.indexes}
    for d, p in positions.items():
        read = p.broadcast_like(present).transpose(*keys).to_numpy()[held]
        joined[d] = expression.indexes[d][read].to_numpy()
    onto = {k: present.indexes[k] for k in keys}
    summed = cast(LinearExpression, grouped_sum(expression, joined, grouped, onto))
    live = cast(xr.DataArray, at(~expression.isnull(), positions, present)) == 1
    return summed.where(present & live)


def _keyed(
    positions: Mapping[str, xr.DataArray], present: xr.DataArray
) -> dict[str, xr.DataArray]:
    """
    *positions* labelled by the key dimensions they do not read along.

    A read left unlabelled makes dimensions without coordinates, which a
    sparse (CSR) term cannot hold; one read along its own key dimension
    stays unlabelled, as a coordinate there would conflict with the operand's.
    """
    return {
        d: p.assign_coords(
            {k: present.indexes[k] for k in p.dims if k not in positions}
        )
        for d, p in positions.items()
    }


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
    sparse = isinstance(array, LinearExpression) and array.is_sparse
    if sparse and isinstance(offset, xr.DataArray) and set(offset.dims) - {over}:
        return _per_offset(array, over, offset, partition, wrap, fill)
    if partition is None:
        return _gather_by_offset(array, over, offset, wrap, fill)
    return _gather_in_groups(array, over, offset, partition, wrap, fill)


def _per_offset(
    array: Array,
    over: str,
    offset: xr.DataArray,
    partition: Groups | None,
    wrap: bool,
    fill: float | None,
) -> Array:
    """
    *array* gathered once per distinct *offset*, each gather kept where the offset is that amount.

    An offset over other dimensions than *over* gathers along two dimensions
    at once, which a sparse (CSR) term cannot select without densifying.
    """
    amounts = offset.to_series().dropna().unique().tolist() or [0]
    terms: list[Array] = []
    reached: list[xr.DataArray] = []
    for amount in amounts:
        moved = _gather(array, over, int(amount), partition, wrap, fill)
        kept = moved.where(offset == amount)
        terms.append(_filled(kept, 0.0))
        reached.append(~kept.isnull())
    return _merged(terms).where(reduce(operator.or_, reached))


def _gather_by_offset(
    array: Array, over: str, offset: Amount, wrap: bool, fill: float | None
) -> Array:
    """
    Translate *array* along *over* by an offset that may differ per entity.

    Selection is by position, so no label is looked up. Out-of-range
    positions are clipped onto the axis and emptied again, so an edge means
    what it does for a scalar shift.
    """
    card = int(array.sizes[over])
    labels = np.asarray(array.indexes[over])
    ordinal = xr.DataArray(np.arange(card), coords={over: labels}, dims=[over])
    source = (ordinal - offset).astype(int)

    def gathered(ordinals: xr.DataArray) -> Array:
        return _read_at(array, over, labels, ordinals)

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
    gathered = _read_at(array, over, labels, source).where(inside)
    if fill is None:
        return gathered
    return _vacated(gathered, array, over, groups.grouped & ~inside, fill)


def _read_at(
    array: Array, over: str, labels: np.ndarray, ordinals: xr.DataArray
) -> Array:
    """
    *array* read along *over* at the positions *ordinals*, labelled *labels* where they land.

    The axis is numbered while it is read: a gather over several dimensions
    copies the labels it reads into a coordinate over all of them, and dates
    copied that way cost more than the gather.
    """
    numbered = array.assign_coords({over: np.arange(len(labels))})
    positions = xr.DataArray(ordinals.values, dims=ordinals.dims)
    return numbered.isel({over: positions}).assign_coords({over: labels})
