"""The recursive evaluator: one expression node to its linopy term, array or number."""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable
from typing import TypeVar, assert_never, cast

import numpy as np
import xarray as xr
from mathspec import program as ms

from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.spec import operators
from linopy.spec.context import Array, Context, Term, Value, as_expression
from linopy.spec.errors import SpecDataError, first_coordinates
from linopy.spec.where import evaluate_where, looked_up
from linopy.variables import Variable

_Binary = ms.Add | ms.Multiply | ms.Divide | ms.Power
_OPERATORS: dict[type, Callable[[Value, Value], Value]] = {
    ms.Add: operator.add,
    ms.Multiply: operator.mul,
    ms.Divide: operator.truediv,
    ms.Power: operator.pow,
}
_Framed = TypeVar(
    "_Framed", bound=xr.DataArray | LinearExpression | QuadraticExpression
)


def evaluate_named(name: str, ctx: Context) -> Value:
    """The named expression *name* as its linopy term, array or number over *ctx*, framed as declared."""
    declared = ctx.program.expressions[name]
    value = evaluate(declared.expression, ctx)
    if isinstance(value, xr.DataArray):
        return without_strays(value).rename(name).transpose(*declared.dims, ...)
    return value


def fold(name: str, ctx: Context) -> xr.DataArray:
    """The named expression *name* as data, folded over the solution and the parameters *ctx* holds."""
    value = evaluate_named(name, ctx)
    if isinstance(value, float | int):
        return xr.DataArray(float(value), name=name)
    return cast(xr.DataArray, value)


def without_strays(value: _Framed) -> _Framed:
    """*value* without the non-dimension coordinates a lookup leaves beside its dimensions, which no file keeps."""
    stray = [c for c in value.coords if c not in value.dims]
    return cast(_Framed, value.drop_vars(stray)) if stray else value


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
    if isinstance(node, _Binary):
        return _combine(node, ctx)
    if isinstance(node, ms.Sum):
        return _sum(node, ctx)
    if isinstance(node, ms.Join):
        return looked_up(_operand(node.operand, ctx), node.columns, ctx)
    if isinstance(node, ms.Translate):
        return operators.shift(
            _operand(node.operand, ctx),
            over=node.along,
            offset=_amount(node.offset, ctx),
            wrap=node.wrap,
            fill=node.fill,
            by=_partition(node.partition, ctx),
        )
    if isinstance(node, ms.WindowSum):
        return operators.sum_back(
            _operand(node.operand, ctx),
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
        return functools.reduce(lambda a, b: operator.add(*_paired(a, b)), regions)
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
            _operand(operand.operand, ctx),
            ctx.relations[columns.name],
            into=into,
            labels=ctx.coords[into],
        )
    else:
        summed = _operand(operand, ctx)
    for dimension in plain:
        summed = operators.sum_over(summed, dimension)
    return summed


def _parameter(name: str, ctx: Context) -> xr.DataArray:
    """The parameter filled as its ``missing:`` says; unfilled, a missing row is a hole that a comparison reads as false."""
    filled, present = ctx.parameter(name)
    if ctx.filled or not isinstance(ctx.program.parameters[name].missing, str):
        return filled
    return filled.where(present)


def _variable(name: str, ctx: Context) -> Value:
    variable = ctx.model.variables[name]
    neutral = ctx.program.variables[name].missing == "neutral"
    if not ctx.solved:
        return variable.fillna(0) if neutral else variable
    if "solution" not in variable.data:
        raise AttributeError(
            f"variable '{name}' has no solution yet: solve the model before reading a named expression"
        )
    return variable.solution.fillna(0) if neutral else variable.solution


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
        raise AttributeError(
            f"constraint '{constraint}' has no dual yet: solve the model, with a solver "
            f"and a problem that report duals, before reading one"
        )
    dual = built.dual
    return -dual if flipped(ctx.program.constraints[constraint]) else dual


def flipped(constraint: ms.Constraint) -> bool:
    """Whether the builder gathers the constraint's variables from its right side onto the left."""
    return not ms.carries_variable(constraint.lhs)


def _operands(node: _Binary) -> tuple[ms.Expression, ms.Expression]:
    if isinstance(node, ms.Divide):
        return node.numerator, node.divisor
    if isinstance(node, ms.Power):
        return node.base, node.exponent
    return node.left, node.right


def _combine(node: _Binary, ctx: Context) -> Value:
    """
    The operands of *node* combined by its operator.

    Two arrays have to agree on their shared coordinates, and the data they
    make has to be finite wherever both are: a NaN made of present data would
    read as an absence and silently drop the row, an infinity would stand in
    the problem. A hole beside a term becomes the term's absence.
    """
    op = _OPERATORS[type(node)]
    left, right = (evaluate(operand, ctx) for operand in _operands(node))
    kind = type(node).__name__.lower()
    if isinstance(left, Term) or isinstance(right, Term):
        if isinstance(node, ms.Add):
            a, b = constant_of(left), constant_of(right)
            refuse_made(kind, ms.parameters_of(node), operator.add, a, b, ctx)
        return op(*_paired(left, right))
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
    return refuse_made(kind, ms.parameters_of(node), op, a, b, ctx)


def _paired(left: Value, right: Value) -> tuple[Value, Value]:
    """*left* and *right* to combine, a hole in data beside a term becoming the term's absence."""
    if isinstance(left, xr.DataArray) and isinstance(right, Term):
        right, left = carried(right, left)
    elif isinstance(right, xr.DataArray) and isinstance(left, Term):
        left, right = carried(left, right)
    return left, right


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
    op: Callable[[Value, Value], Value],
    left: xr.DataArray,
    right: xr.DataArray,
    ctx: Context,
) -> xr.DataArray:
    """*op* of *left* and *right*, refused wherever it is not finite though both are present."""
    made = cast(xr.DataArray, op(left, right))
    if made.dtype.kind not in "fc":
        return made
    nan_made = made.isnull() & left.notnull() & right.notnull()
    bad = nan_made | (np.isinf(made) & _finite(left) & _finite(right))
    if ctx.rows is not None:
        bad = bad & ctx.rows
    if not bool(bad.any()):
        return made
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


def _operand(node: ms.Expression, ctx: Context) -> Array:
    """An operand the language gives dims, so never a bare number."""
    return cast(Array, evaluate(node, ctx))


def _in_region(value: Value, rows: xr.DataArray) -> Value:
    """*value* where the region holds and a hard zero everywhere else: a fill, so absence inside the region stands."""
    if isinstance(value, float | int):
        return rows * value
    return as_expression(value).where(rows, 0)


def _amount(amount: int | str, ctx: Context) -> operators.Amount:
    if isinstance(amount, str):
        return ctx.parameter(amount).filled
    return amount


def _partition(partition: ms.Partition | None, ctx: Context) -> xr.DataArray | None:
    """The relation a windowed operator stays inside, named for the dimension its group column is over."""
    if partition is None:
        return None
    grouped = partition.dim(partition.grouped[0])
    return ctx.relations[partition.name].rename(grouped)
