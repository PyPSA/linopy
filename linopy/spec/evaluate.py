"""The recursive evaluator: one expression node to its linopy term, array or number."""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable
from typing import assert_never, cast

import numpy as np
import xarray as xr
from mathspec import program as ms

from linopy.spec import context, operators
from linopy.spec.context import Array, Context, Term, Value
from linopy.spec.errors import SpecDataError, first_coordinates, unknown
from linopy.spec.where import evaluate_where
from linopy.variables import Variable

_Binary = ms.Add | ms.Multiply | ms.Divide | ms.Power


def evaluate_named(name: str, ctx: Context) -> Value:
    """The named expression *name* as its linopy term, array or number over *ctx*, framed as declared."""
    if name not in ctx.program.expressions:
        raise unknown("named expression", name, ctx.program.expressions)
    declared = ctx.program.expressions[name]
    value = evaluate(declared.expression, ctx)
    if isinstance(value, xr.DataArray):
        return _named(value, name).transpose(*declared.dims, ...)
    return value


def fold(name: str, ctx: Context) -> xr.DataArray:
    """The named expression *name* as data, folded over the solution and the parameters *ctx* holds."""
    value = evaluate_named(name, ctx)
    if isinstance(value, float | int):
        return xr.DataArray(float(value), name=name)
    return cast(xr.DataArray, value)


def _named(value: xr.DataArray, name: str) -> xr.DataArray:
    """*value* with its stray non-dimension coordinates dropped and renamed to *name*."""
    stray = [c for c in value.coords if c not in value.dims]
    return value.drop_vars(stray).rename(name)


def evaluate(node: ms.Expression, ctx: Context) -> Value:
    """One node as a linopy term, an array or a number."""
    if isinstance(node, ms.ExpressionReference):
        return evaluate(node.body, ctx)
    if isinstance(node, ms.Constant):
        return node.value
    if isinstance(node, ms.VariableReference):
        return _variable(node.name, ctx)
    if isinstance(node, ms.Dual):
        return _dual(node.constraint, ctx)
    if isinstance(node, ms.ParameterReference):
        return _parameter(node.name, ctx)
    if isinstance(node, ms.Negate):
        return -evaluate(node.operand, ctx)
    if isinstance(node, ms.Add):
        return _combine(node, operator.add, node.left, node.right, ctx)
    if isinstance(node, ms.Multiply):
        return _combine(node, operator.mul, node.left, node.right, ctx)
    if isinstance(node, ms.Divide):
        return _combine(node, operator.truediv, node.numerator, node.divisor, ctx)
    if isinstance(node, ms.Power):
        return _combine(node, operator.pow, node.base, node.exponent, ctx)
    if isinstance(node, ms.Sum):
        return _sum(node, ctx)
    if isinstance(node, ms.Join):
        return _joined(node, ctx)
    if isinstance(node, ms.Translate):
        return operators.shift(
            _array(evaluate(node.operand, ctx)),
            over=node.along,
            offset=_amount(node.offset, ctx),
            wrap=node.wrap,
            fill=node.fill,
            by=_partition(node.partition, ctx),
        )
    if isinstance(node, ms.WindowSum):
        return operators.sum_back(
            _array(evaluate(node.operand, ctx)),
            over=node.along,
            within=_amount(node.width, ctx),
            wrap=node.wrap,
            by=_partition(node.partition, ctx),
        )
    if isinstance(node, ms.Cases):
        regions = (
            _in_region(evaluate(region.value, ctx), evaluate_where(region.when, ctx))
            for region in node.regions
        )
        return functools.reduce(lambda a, b: _added(a, b), regions)
    assert_never(node)


def _sum(node: ms.Sum, ctx: Context) -> Array:
    """
    A sum over its axes: the join axes a relation opens are closed by grouping
    the operand through the relation, the plain ones by summing them away.
    """
    operand = node.operand
    plain = [axis.dimension for axis in node.over if axis.column is None]
    if isinstance(operand, ms.Join) and operand.columns.axes:
        columns = operand.columns
        into = columns.added_dims[0]
        summed = operators.grouped_sum(
            _array(evaluate(operand.operand, ctx)),
            ctx.relations[columns.name],
            into=into,
            labels=ctx.coords[into],
        )
    else:
        summed = _array(evaluate(operand, ctx))
    for dimension in plain:
        summed = operators.sum_over(summed, dimension)
    return summed


def _joined(node: ms.Join, ctx: Context) -> Array:
    """A lookup: the operand read through the relation, one value per row of the key."""
    columns = node.columns
    return operators.at(
        _array(evaluate(node.operand, ctx)),
        ctx.relations[columns.name],
        into=columns.dropped_dims[0],
    )


def _parameter(name: str, ctx: Context) -> xr.DataArray:
    """The parameter filled as its ``missing:`` says; unfilled, a missing row is a hole that a comparison reads as false."""
    arr = ctx.parameters[name]
    if ctx.filled or not isinstance(ctx.program.parameters[name].missing, str):
        return arr
    return arr.where(ctx.parameters.present(name))


def _variable(name: str, ctx: Context) -> Value:
    variable = ctx.model.variables[name]
    missing = ctx.program.variables[name].missing
    if not ctx.solved:
        return context.variable_term(variable, missing)
    if "solution" not in variable.data:
        raise RuntimeError(
            f"variable '{name}' has no solution yet: solve the model before reading a named expression"
        )
    return context.solution(variable, missing)


def _dual(constraint: str, ctx: Context) -> xr.DataArray:
    """
    The dual of *constraint* as the spec reads it: the rate the objective rises
    with the right side of the written comparison. The builder gathers the
    variables onto the left, flipping a row written with them on the right,
    and such a row's dual is negated back.
    """
    if not ctx.solved:
        raise TypeError(
            f"the dual of constraint '{constraint}' has no symbolic form, read `.solution`"
        )
    built = ctx.model.constraints[constraint]
    if built._dual is None:
        raise RuntimeError(
            f"constraint '{constraint}' has no dual yet: solve the model, with a solver "
            f"and a problem that report duals, before reading one"
        )
    dual = built.dual
    return -dual if flipped(ctx.program.constraints[constraint]) else dual


def flipped(constraint: ms.Constraint) -> bool:
    """Whether the builder gathers the constraint's variables from its right side onto the left."""
    return not ms.carries_variable(constraint.lhs)


def _combine(
    node: _Binary,
    op: Callable[[Value, Value], Value],
    left_node: ms.Expression,
    right_node: ms.Expression,
    ctx: Context,
) -> Value:
    """
    *left* and *right* combined by *op*.

    Two arrays have to agree on their shared coordinates, and the data they
    make has to be finite wherever both are: a NaN made of present data would
    read as an absence and silently drop the row, an infinity would stand in
    the problem. A hole beside a term becomes the term's absence.
    """
    left, right = evaluate(left_node, ctx), evaluate(right_node, ctx)
    kind = type(node).__name__.lower()
    if isinstance(left, Term) or isinstance(right, Term):
        if isinstance(node, ms.Add):
            a, b = constant_of(left), constant_of(right)
            refuse_made(kind, ms.parameters_of(node), a + b, a, b, ctx)
        if isinstance(left, xr.DataArray) and isinstance(right, Term):
            right, left = carried(right, left)
        elif isinstance(right, xr.DataArray) and isinstance(left, Term):
            left, right = carried(left, right)
        return op(left, right)
    if not isinstance(left, xr.DataArray) and not isinstance(right, xr.DataArray):
        return op(left, right)
    a, b = xr.DataArray(left), xr.DataArray(right)
    for dim in set(a.dims) & set(b.dims):
        if not a.indexes[dim].equals(b.indexes[dim]):
            raise SpecDataError(
                f"operands are not aligned on '{dim}': {a.indexes[dim].tolist()[:5]} against "
                f"{b.indexes[dim].tolist()[:5]}. Every operand is read on the master "
                f"coordinates, so the data was attached against other labels than the model was built on."
            )
    made = cast(xr.DataArray, op(a, b))
    refuse_made(kind, ms.parameters_of(node), made, a, b, ctx)
    return made


def _added(left: Value, right: Value) -> Value:
    """*left* plus *right*, a hole beside a term becoming the term's absence."""
    if isinstance(left, xr.DataArray) and isinstance(right, Term):
        right, left = carried(right, left)
    elif isinstance(right, xr.DataArray) and isinstance(left, Term):
        left, right = carried(left, right)
    return left + right


def constant_of(value: Value) -> xr.DataArray:
    """The constant part of *value*: a term's constant, or the data itself."""
    if isinstance(value, Variable):
        return xr.DataArray(0.0)
    if isinstance(value, Term):
        return value.const
    return xr.DataArray(value)


def refuse_made(
    kind: str,
    params: frozenset[str],
    made: xr.DataArray,
    left: xr.DataArray,
    right: xr.DataArray,
    ctx: Context,
) -> None:
    if made.dtype.kind not in "fc":
        return
    nan_made = made.isnull() & left.notnull() & right.notnull()
    bad = nan_made | (np.isinf(made) & _finite(left) & _finite(right))
    if ctx.rows is not None:
        bad = bad & ctx.rows
    if not bool(bad.any()):
        return
    named = ", ".join(f"'{p}'" for p in sorted(params))
    raise SpecDataError(
        f"{ctx.subject}: the {kind} of {named} is not finite at "
        f"{first_coordinates(bad, 3)}, though its operands are present. A NaN made there "
        f"would read as an absence and silently drop the row; an infinity would "
        f"stand in the problem.\n"
        f"  Mask the coordinates out with a where, or supply data the arithmetic "
        f"is defined on."
    )


def _finite(arr: xr.DataArray) -> xr.DataArray:
    if arr.dtype.kind in "fc":
        return arr.notnull() & np.isfinite(arr.fillna(0.0))
    return xr.ones_like(arr, dtype=bool)


def carried(term: Term, data: xr.DataArray) -> tuple[Term, xr.DataArray]:
    """A hole an operator left in *data* is an absence the term takes: the slot leaves the row, and the hole reads as a harmless one."""
    if not bool(data.isnull().any()):
        return term, data
    return term.where(data.notnull()), data.fillna(1.0)


def _array(value: Value) -> Array:
    if isinstance(value, float | int):
        raise TypeError("a shape operator takes an array or a term, not a bare number")
    return value


def _in_region(value: Value, rows: xr.DataArray) -> Value:
    """*value* where the region holds and a hard zero everywhere else: a fill, so absence inside the region stands."""
    if isinstance(value, float | int):
        return rows * value
    if isinstance(value, Variable):
        value = value.to_linexpr()
    return value.where(rows, 0)


def _amount(amount: int | str, ctx: Context) -> operators.Amount:
    if isinstance(amount, str):
        return ctx.parameters[amount]
    return amount


def _partition(partition: ms.Partition | None, ctx: Context) -> xr.DataArray | None:
    """The relation a windowed operator stays inside, named for the dimension its group column is over."""
    if partition is None:
        return None
    grouped = partition.dim(partition.grouped[0])
    return ctx.relations[partition.name].rename(grouped)
