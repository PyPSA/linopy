"""The recursive evaluator: one expression node to its linopy term, array or number."""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable
from typing import assert_never, cast

import xarray as xr
from math_spec import program as ms

from linopy.spec import context, operators
from linopy.spec.context import Array, Context, Term, Value
from linopy.spec.coverage import check_kind, obligations_of
from linopy.spec.errors import SpecDataError, unknown
from linopy.spec.where import evaluate_where
from linopy.variables import Variable


def evaluate_named(name: str, ctx: Context) -> Value:
    """The named expression *name* as its linopy term, array or number over *ctx*, its divisors checked first."""
    if name not in ctx.program.expressions:
        raise unknown("named expression", name, ctx.program.expressions)
    body = ctx.program.expressions[name].expression
    found = obligations_of((body,), ctx, None)
    check_kind(f"expression '{name}'", "divisor", found["divisor"], ctx)
    value = evaluate(body, ctx)
    return _named(value, name) if isinstance(value, xr.DataArray) else value


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
    if isinstance(node, ms.Constant):
        return node.value
    if isinstance(node, ms.Variable):
        return _variable(node.name, ctx)
    if isinstance(node, ms.Dual):
        return _dual(node.constraint, ctx)
    if isinstance(node, ms.Parameter):
        parameter = ctx.parameters[node.name]
        return context.coefficient(parameter) if ctx.filled else parameter
    if isinstance(node, ms.Negate):
        return -evaluate(node.operand, ctx)
    if isinstance(node, ms.Add):
        return _combine(
            operator.add, evaluate(node.left, ctx), evaluate(node.right, ctx)
        )
    if isinstance(node, ms.Multiply):
        return _combine(
            operator.mul, evaluate(node.left, ctx), evaluate(node.right, ctx)
        )
    if isinstance(node, ms.Divide):
        return _combine(
            operator.truediv, evaluate(node.numerator, ctx), evaluate(node.divisor, ctx)
        )
    if isinstance(node, ms.Power):
        return _combine(
            operator.pow, evaluate(node.base, ctx), evaluate(node.exponent, ctx)
        )
    if isinstance(node, ms.Sum):
        summed = _array(evaluate(node.operand, ctx))
        for dimension in node.over:
            summed = operators.sum_over(summed, dimension)
        return summed
    if isinstance(node, ms.GroupSum):
        return operators.grouped_sum(
            _array(evaluate(node.operand, ctx)),
            (ctx.relations[node.direction.name],),
            into=node.direction.produced_dims,
            labels=ctx.coords,
        )
    if isinstance(node, ms.Pullback):
        return operators.at(
            _array(evaluate(node.operand, ctx)),
            (ctx.relations[node.direction.name],),
            into=node.direction.consumed_dims,
        )
    if isinstance(node, ms.Translate):
        return operators.shift(
            _array(evaluate(node.operand, ctx)),
            over=node.along,
            offset=_amount(node.offset, ctx),
            wrap=node.wrap,
            fill=node.fill,
            by=_partition(node, ctx),
        )
    if isinstance(node, ms.WindowSum):
        return operators.sum_back(
            _array(evaluate(node.operand, ctx)),
            over=node.along,
            within=_amount(node.width, ctx),
            wrap=node.wrap,
            by=_partition(node, ctx),
        )
    if isinstance(node, ms.Cases):
        regions = (
            _in_region(evaluate(region.value, ctx), evaluate_where(region.when, ctx))
            for region in node.regions
        )
        return functools.reduce(lambda a, b: _combine(operator.add, a, b), regions)
    assert_never(node)


def _variable(name: str, ctx: Context) -> Value:
    variable = ctx.model.variables[name]
    absence = ctx.program.variables[name].absence
    if not ctx.solved:
        return context.variable_term(variable, absence)
    if "solution" not in variable.data:
        raise RuntimeError(
            f"variable '{name}' has no solution yet: solve the model before reading a named expression"
        )
    return context.solution(variable, absence)


def _dual(constraint: str, ctx: Context) -> xr.DataArray:
    if not ctx.solved:
        raise TypeError(
            f"the dual of constraint '{constraint}' has no symbolic form, read `.solution`"
        )
    data = ctx.model.constraints[constraint].data
    if "dual" not in data:
        raise RuntimeError(
            f"constraint '{constraint}' has no dual yet: solve the model, with a solver "
            f"and a problem that report duals, before reading one"
        )
    return data["dual"]


def _combine(op: Callable[[Value, Value], Value], left: Value, right: Value) -> Value:
    """*left* and *right* combined by *op*, once two arrays agree on their shared coordinates and a hole beside a term has become its absence."""
    if isinstance(left, xr.DataArray) and isinstance(right, xr.DataArray):
        for dim in set(left.dims) & set(right.dims):
            if not left.indexes[dim].equals(right.indexes[dim]):
                raise SpecDataError(
                    f"operands are not aligned on '{dim}': {left.indexes[dim].tolist()[:5]} against "
                    f"{right.indexes[dim].tolist()[:5]}. Every operand is read on the master "
                    f"coordinates, so the data was attached against other labels than the model was built on."
                )
    elif isinstance(left, xr.DataArray) and isinstance(right, Term):
        right, left = carried(right, left)
    elif isinstance(right, xr.DataArray) and isinstance(left, Term):
        left, right = carried(left, right)
    return op(left, right)


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
        return context.coefficient(ctx.parameters[amount])
    return amount


def _partition(node: ms.Translate | ms.WindowSum, ctx: Context) -> xr.DataArray | None:
    """The relation a windowed operator stays inside, named for the dimension its group column is over."""
    if node.partition is None:
        return None
    partition = node.partition
    return ctx.relations[partition.name].rename(partition.dim(partition.group[0]))
