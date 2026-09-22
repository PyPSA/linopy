"""A ``where:`` predicate as a boolean array over the coordinates it masks."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import assert_never

import numpy as np
import xarray as xr
from math_spec import program as ms

from linopy.spec.context import Context, Term, Value
from linopy.spec.errors import SpecDataError
from linopy.spec.groups import grouped

_PREDICATE_OPS: dict[str, Callable[..., xr.DataArray]] = {
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


def _node(node: ms.Predicate, ctx: Context) -> xr.DataArray:
    """
    One predicate node as a boolean array.

    A masked-out variable coordinate and a comparison over NaN both read as
    exclusion. A null relation value is excluded explicitly: numpy answers
    ``None != 'north'`` with True, so a ``!=`` would otherwise keep exactly
    the labels that map nowhere; a side of an expression comparison that is
    absent is excluded the same way. A count is one number per coordinate
    the counted predicate keeps, and a translated predicate is false where
    the translation vacates.
    """
    if isinstance(node, ms.BooleanLiteral):
        return xr.DataArray(node.value)
    if isinstance(node, ms.ParameterDefined):
        return _defined(
            ctx.parameters[node.name], ctx.program.parameters[node.name].dtype
        )
    if isinstance(node, ms.VariableDefined):
        return ctx.model.variables[node.name].mask
    if isinstance(node, ms.ParameterComparison):
        return _compared(ctx.parameters[node.name], node.op, node.value)
    if isinstance(node, ms.ExpressionComparison):
        left, right = _side(node.left, ctx), _side(node.right, ctx)
        compared = _PREDICATE_OPS[node.op](left, right) & left.notnull()
        return _bool(compared & right.notnull())
    if isinstance(node, ms.ArithmeticComparison):
        raise TypeError(
            "an ArithmeticComparison is rewritten by lowering and never reaches a program"
        )
    if isinstance(node, ms.CountComparison):
        count = _node(node.predicate.root, ctx).sum(node.over)
        return _PREDICATE_OPS[node.op](count, node.value).astype(bool)
    if isinstance(node, ms.TranslatedPredicate):
        shifted = _node(node.operand.root, ctx).shift({node.along: node.offset})
        return _bool(shifted)
    if isinstance(node, ms.DimensionComparison):
        labels = ctx.coords[node.name]
        arr = xr.DataArray(labels, coords={node.name: labels}, dims=[node.name])
        return _compared(arr, node.op, node.value)
    if isinstance(node, ms.DimensionPosition):
        return _position(node, ctx)
    if isinstance(node, ms.RelationComparison):
        arr = ctx.relations[node.name]
        return _bool(_PREDICATE_OPS[node.op](arr, node.value) & arr.notnull())
    if isinstance(node, ms.RelationPairComparison):
        left = ctx.relations[node.name]
        right = ctx.relations[node.other]
        compared = _PREDICATE_OPS[node.op](left, right) & left.notnull()
        return _bool(compared & right.notnull())
    if isinstance(node, ms.RelationDefined):
        return ctx.relations[node.name].notnull()
    if isinstance(node, ms.Not):
        return ~_node(node.operand, ctx)
    if isinstance(node, ms.And):
        return _node(node.left, ctx) & _node(node.right, ctx)
    if isinstance(node, ms.Or):
        return _node(node.left, ctx) | _node(node.right, ctx)
    assert_never(node)


def _bool(arr: xr.DataArray) -> xr.DataArray:
    """*arr* as a boolean mask, a hole reading as exclusion."""
    return arr.fillna(False).astype(bool)


def _side(node: ms.Expression, ctx: Context) -> xr.DataArray:
    """One side of an expression comparison as data, its holes kept so they read as exclusion."""
    from linopy.spec.evaluate import evaluate

    value: Value = evaluate(node, ctx.unfilled)
    if isinstance(value, Term):
        raise TypeError("a side of a where comparison is variable-free")
    return value if isinstance(value, xr.DataArray) else xr.DataArray(value)


def _compared(arr: xr.DataArray, op: str, value: object) -> xr.DataArray:
    """*arr* against a literal spelled the way its axis spells it."""
    return _bool(_PREDICATE_OPS[op](arr, _as_the_axis_spells_it(arr, value)))


def _defined(arr: xr.DataArray, dtype: str) -> xr.DataArray:
    """What a bare parameter name asks: a bool is its own answer, a str is defined where it has a row, a number must be finite too."""
    if dtype == "bool":
        return _bool(arr)
    if dtype == "str":
        return arr.notnull()
    return arr.notnull() & np.isfinite(arr)


def _position(node: ms.DimensionPosition, ctx: Context) -> xr.DataArray:
    labels = ctx.coords[node.name]
    if node.partition is not None:
        groups = ctx.relations[node.partition.name]
        arr = _group_offsets(node, groups, np.asarray(labels))
        return _bool(_PREDICATE_OPS[node.op](arr, 0) & arr.notnull())
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
    return _PREDICATE_OPS[node.op](arr, at).astype(bool)


def _group_offsets(
    node: ms.DimensionPosition, groups: xr.DataArray, labels: np.ndarray
) -> xr.DataArray:
    """Each coordinate's distance from the boundary of its own group; NaN where it is in no group."""
    partition = grouped(node.name, labels, groups)
    needed = node.position + 1 if node.position >= 0 else -node.position
    short = sorted(
        str(g) for g, n in zip(partition.names, partition.counts) if n < needed
    )
    if short:
        raise SpecDataError(
            f"where: position({node.name}, by={groups.name}) {node.op} {node.position} names position "
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
