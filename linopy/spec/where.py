"""A ``where:`` predicate as a boolean array over the coordinates it masks."""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from typing import assert_never

import numpy as np
import xarray as xr
from math_spec import program as ms

from linopy.spec import terms
from linopy.spec.context import Context
from linopy.spec.errors import SpecDataError
from linopy.spec.operators import _grouped

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


def bound_lookup(
    name: str, over: str, lookups: Mapping[str, Mapping[str, xr.DataArray]]
) -> xr.DataArray:
    """The lookup *name* as an array over *over*, NaN where a label is unmapped."""
    return lookups[over][name]


def _node(node: ms.WhereNode, ctx: Context) -> xr.DataArray:
    """
    One predicate node as a boolean array.

    A masked-out variable coordinate and a comparison over NaN both read as
    exclusion. A null lookup value is excluded explicitly: numpy answers
    ``None != 'north'`` with True, so a ``!=`` would otherwise keep exactly
    the labels that map nowhere.
    """
    if isinstance(node, ms.BooleanLiteralNode):
        return xr.DataArray(node.value)
    if isinstance(node, ms.ParameterDefinedNode):
        return _defined(
            ctx.parameters[node.name], ctx.program.parameter(node.name).dtype
        )
    if isinstance(node, ms.VariableDefinedNode):
        return terms.present(ctx.model.variables[node.name])
    if isinstance(node, ms.ParameterComparisonNode):
        arr = ctx.parameters[node.name]
        result = _PREDICATE_OPS[node.op](arr, _as_the_axis_spells_it(arr, node.value))
        return result.fillna(False).astype(bool)
    if isinstance(node, ms.DimensionComparisonNode):
        labels = ctx.coords[node.name]
        arr = xr.DataArray(labels, coords={node.name: labels}, dims=[node.name])
        result = _PREDICATE_OPS[node.op](arr, _as_the_axis_spells_it(arr, node.value))
        return result.fillna(False).astype(bool)
    if isinstance(node, ms.DimensionPositionNode):
        return _position(node, ctx)
    if isinstance(node, ms.LookupComparisonNode):
        arr = bound_lookup(node.name, node.over, ctx.lookups)
        compared = _PREDICATE_OPS[node.op](arr, node.value) & arr.notnull()
        return compared.fillna(False).astype(bool)
    if isinstance(node, ms.LookupPairComparisonNode):
        left = bound_lookup(node.name, node.over, ctx.lookups)
        right = bound_lookup(node.other, node.over, ctx.lookups)
        compared = (
            _PREDICATE_OPS[node.op](left, right) & left.notnull() & right.notnull()
        )
        return compared.fillna(False).astype(bool)
    if isinstance(node, ms.LookupDefinedNode):
        return bound_lookup(node.name, node.over, ctx.lookups).notnull()
    if isinstance(node, ms.NotNode):
        return ~_node(node.operand, ctx)
    if isinstance(node, ms.AndNode):
        return _node(node.left, ctx) & _node(node.right, ctx)
    if isinstance(node, ms.OrNode):
        return _node(node.left, ctx) | _node(node.right, ctx)
    assert_never(node)


def _defined(arr: xr.DataArray, dtype: str) -> xr.DataArray:
    """What a bare parameter name asks: a bool is its own answer, a str is defined where it has a row, a number must be finite too."""
    if dtype == "bool":
        return arr.fillna(False).astype(bool)
    if dtype == "str":
        return arr.notnull()
    return arr.notnull() & np.isfinite(arr)


def _position(node: ms.DimensionPositionNode, ctx: Context) -> xr.DataArray:
    labels = ctx.coords[node.name]
    if node.by is not None:
        groups = bound_lookup(node.by, node.name, ctx.lookups)
        arr = _group_offsets(node, groups, np.asarray(labels))
        compared = _PREDICATE_OPS[node.op](arr, 0) & arr.notnull()
        return compared.fillna(False).astype(bool)
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
    node: ms.DimensionPositionNode, groups: xr.DataArray, labels: np.ndarray
) -> xr.DataArray:
    """Each coordinate's distance from the boundary of its own group; NaN where it is in no group."""
    partition = _grouped(node.name, labels, groups)
    needed = node.position + 1 if node.position >= 0 else -node.position
    short = sorted(
        str(g) for g, n in zip(partition.names, partition.counts) if n < needed
    )
    if short:
        raise SpecDataError(
            f"where: position({node.name}, by={node.by}) {node.op} {node.position} names position "
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
