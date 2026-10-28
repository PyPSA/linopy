"""A ``where:`` condition as a boolean array over the coordinates it masks."""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import replace
from typing import assert_never, cast

import numpy as np
import xarray as xr
from mathspec import program as ms

from linopy.spec import operators
from linopy.spec.context import Array, Context
from linopy.spec.errors import SpecDataError

_COMPARISONS: dict[str, Callable[..., xr.DataArray]] = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
}


def evaluate_where(mask: ms.Mask | None, ctx: Context) -> xr.DataArray:
    """The rows *mask* admits, as a boolean array; no mask is a 0-d ``True``."""
    if mask is None:
        return xr.DataArray(True)
    return _node(mask.root, ctx)


def as_linopy_mask(mask: xr.DataArray) -> xr.DataArray | None:
    """*mask* as linopy's ``mask=`` takes it: ``None`` where nothing is masked."""
    if mask.ndim == 0 and bool(mask):
        return None
    return mask


def _node(node: ms.Condition, ctx: Context) -> xr.DataArray:
    """One condition node as a boolean array, each evaluated once per context."""
    return ctx.memo.kept(node, lambda: _condition(node, ctx))


def _condition(node: ms.Condition, ctx: Context) -> xr.DataArray:
    """
    One condition node as a boolean array.

    A masked-out variable coordinate and a comparison at a missing row both
    read as exclusion. A null relation value is excluded explicitly: numpy
    answers ``None != 'north'`` with True, so a ``!=`` would otherwise keep
    exactly the labels that map nowhere; a side of an expression comparison
    that is absent is excluded the same way. A count is one number per
    coordinate the counted condition keeps, a translated condition is false
    where the translation vacates, and a joined one where the relation has
    no row.
    """
    if isinstance(node, ms.BooleanLiteral):
        return xr.DataArray(node.value)
    if isinstance(node, ms.ConditionReference):
        return _node(node.body, ctx)
    if isinstance(node, ms.ParameterDefined):
        return _defined(node.name, ctx)
    if isinstance(node, ms.VariableDefined):
        return ctx.model.variables[node.name].mask
    if isinstance(node, ms.ParameterComparison):
        filled, present = ctx.parameter(node.name)
        compared = _compared(filled, node.op, node.value)
        if isinstance(ctx.program.parameters[node.name].missing, str):
            return compared & present
        return compared
    if isinstance(node, ms.ExpressionComparison):
        return _both(node.op, _side(node.left, ctx), _side(node.right, ctx))
    if isinstance(node, ms.CountComparison):
        count = _node(node.condition.root, ctx).sum(node.over)
        return _COMPARISONS[node.op](count, node.value)
    if isinstance(node, ms.TranslatedCondition):
        shifted = _node(node.operand.root, ctx).shift({node.along: node.offset})
        return _bool(shifted)
    if isinstance(node, ms.JoinedCondition):
        pulled = looked_up(_node(node.operand.root, ctx), node.columns, ctx)
        return _bool(cast(xr.DataArray, pulled))
    if isinstance(node, ms.DimensionComparison):
        labels = ctx.coords[node.name]
        arr = xr.DataArray(labels, coords={node.name: labels}, dims=[node.name])
        return _compared(arr, node.op, node.value)
    if isinstance(node, ms.DimensionPosition):
        return _position(node, ctx)
    if isinstance(node, ms.RelationComparison):
        arr = ctx.relations[node.name].column(node.column)
        return _compared(arr, node.op, node.value) & arr.notnull()
    if isinstance(node, ms.RelationPairComparison):
        left = ctx.relations[node.name].column(node.column)
        return _both(node.op, left, ctx.relations[node.other].column(node.other_column))
    if isinstance(node, ms.RelationDefined):
        return ctx.relations[node.name].defined()
    if isinstance(node, ms.Not):
        return ~_node(node.operand, ctx)
    if isinstance(node, ms.And):
        return _node(node.left, ctx) & _node(node.right, ctx)
    if isinstance(node, ms.Or):
        return _node(node.left, ctx) | _node(node.right, ctx)
    assert_never(node)


def looked_up(arr: Array, columns: ms.JoinColumns, ctx: Context) -> Array:
    """
    *arr* read through the relation *columns* name, one value per key tuple.

    *arr* is read along each dimension *columns* joins on: at the label a
    value column holds, and along a key column it carries at the key's own label.
    """
    table = ctx.relations[columns.name]
    present = table.defined()
    positions = {}
    for role in columns.joined:
        dim = columns.dim(role)
        labels = present[dim] if role in table.declared.key else table.column(role)
        positions[dim] = operators.positions_of(labels, arr.indexes[dim])
    return operators.at(arr, positions, present)


def groups_of(partition: ms.Partition, ctx: Context) -> xr.DataArray:
    """The group of each coordinate along the axis *partition* steps, named for the dimension the group column is over."""
    if partition.joined or len(partition.grouped) != 1:
        raise SpecDataError(
            f"within={partition.name}[...] partitions by {list(partition.grouped)} and joins "
            f"on the key column(s) {list(partition.joined)}, and linopy partitions by one "
            f"value column of a relation keyed by the stepped dimension alone."
        )
    role = partition.grouped[0]
    return ctx.relations[partition.name].column(role).rename(partition.dim(role))


def _defined(name: str, ctx: Context) -> xr.DataArray:
    """Where a parameter is defined: a bool reads its value, a float has a row and a finite value, any other a row."""
    filled, present = ctx.parameter(name)
    dtype = ctx.program.parameters[name].dtype
    if dtype == "bool":
        return _bool(filled)
    if dtype == "float":
        return present & np.isfinite(filled)
    return present


def _bool(arr: xr.DataArray) -> xr.DataArray:
    """*arr* as a boolean mask, a hole reading as exclusion."""
    return arr.fillna(False).astype(bool)


def _both(op: str, left: xr.DataArray, right: xr.DataArray) -> xr.DataArray:
    return _bool(_COMPARISONS[op](left, right) & left.notnull() & right.notnull())


def _side(node: ms.Expression, ctx: Context) -> xr.DataArray:
    """One side of an expression comparison as data, its missing rows kept so they read as exclusion."""
    from linopy.spec.evaluator import evaluate

    return xr.DataArray(evaluate(node, replace(ctx, filled=False)))


def _compared(arr: xr.DataArray, op: str, value: object) -> xr.DataArray:
    """*arr* against a literal spelled the way its axis spells it."""
    return _bool(_COMPARISONS[op](arr, _as_the_axis_spells_it(arr, value)))


def _position(node: ms.DimensionPosition, ctx: Context) -> xr.DataArray:
    labels = ctx.coords[node.name]
    if node.partition is not None:
        arr = _group_offsets(node, node.partition, np.asarray(labels), ctx)
        return _bool(_COMPARISONS[node.op](arr, 0) & arr.notnull())
    at = node.position + len(labels) if node.position < 0 else node.position
    if not 0 <= at < len(labels):
        raise SpecDataError(
            f"where: position({node.name}) {node.op} {node.position} names position {at} of "
            f"'{node.name}', which has {len(labels)} coordinate(s). A boundary that names no "
            f"coordinate leaves the rows it was to seed unseeded."
        )
    arr = xr.DataArray(
        np.arange(len(labels)), coords={node.name: labels}, dims=[node.name]
    )
    return _COMPARISONS[node.op](arr, at)


def _group_offsets(
    node: ms.DimensionPosition, within: ms.Partition, labels: np.ndarray, ctx: Context
) -> xr.DataArray:
    """Each coordinate's distance from the boundary of its own group; NaN where it is in no group."""
    partition = operators.grouped(node.name, labels, groups_of(within, ctx))
    needed = node.position + 1 if node.position >= 0 else -node.position
    short = sorted(
        str(g) for g, n in zip(partition.names, partition.counts) if n < needed
    )
    if short:
        raise SpecDataError(
            f"where: position({node.name}, within={within.name}) {node.op} {node.position} names position "
            f"{node.position} within each group, and {len(short)} of them are shorter than that: "
            f"{short[:5]}. A boundary that names no coordinate leaves the rows it was to seed unseeded."
        )
    target = node.position if node.position >= 0 else partition.size + node.position
    return partition.within.where(partition.grouped) - target


def _as_the_axis_spells_it(arr: xr.DataArray, value: object) -> object:
    """A ``where`` literal in the spelling of the axis it is compared against: a date on a datetime axis is a ``datetime64``."""
    if arr.dtype.kind == "M":
        return np.datetime64(str(value))
    return value
