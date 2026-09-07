"""The recursive evaluator: one expression node to its linopy term, array or number."""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable
from typing import assert_never

import xarray as xr
from math_spec import did_you_mean
from math_spec import program as ms

from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.spec import operators, terms
from linopy.spec.context import Context
from linopy.spec.coverage import check_divisors, obligations_of
from linopy.spec.errors import SpecDataError
from linopy.spec.terms import Array, Term, Value
from linopy.spec.where import evaluate_where
from linopy.variables import Variable


def evaluate_named(name: str, ctx: Context) -> Value:
    """The named expression *name* as its linopy term, array or number over *ctx*, its divisors checked first."""
    if name not in ctx.program.named_expressions:
        raise KeyError(
            f"unknown named expression '{name}'. "
            + did_you_mean(name, ctx.program.named_expressions)
        )
    body = ctx.program.named_expressions[name]
    found = obligations_of((body,), ctx, None)
    check_divisors(f"expression '{name}'", found.divisors, ctx)
    value = evaluate(body, ctx)
    return _named(value, name) if isinstance(value, xr.DataArray) else value


def fold(name: str, ctx: Context) -> xr.DataArray:
    """The named expression *name* as data, folded over the solution and the parameters *ctx* holds."""
    value = evaluate_named(name, ctx)
    if isinstance(value, xr.DataArray):
        return value
    if isinstance(value, float | int):
        return xr.DataArray(float(value), name=name)
    raise TypeError(
        f"expression '{name}' folded to a {type(value).__name__}, not to data"
    )


def _named(value: xr.DataArray, name: str) -> xr.DataArray:
    """*value* with its stray non-dimension coordinates dropped and renamed to *name*."""
    stray = [c for c in value.coords if c not in value.dims]
    return value.drop_vars(stray).rename(name)


def evaluate(node: ms.ExpressionNode, ctx: Context) -> Value:
    """One node as a linopy term, an array or a number."""
    if isinstance(node, ms.Constant):
        return node.value
    if isinstance(node, ms.Variable):
        return _variable(node.name, ctx)
    if isinstance(node, ms.Parameter):
        return terms.coefficient(ctx.parameters[node.name])
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
            _lookup_arrays(node.over, node.coordinate, ctx),
            into=node.into,
            labels=ctx.coords,
        )
    if isinstance(node, ms.At):
        return operators.at(
            _array(evaluate(node.operand, ctx)),
            _lookup_arrays(node.over, node.coordinate, ctx),
            into=node.into,
        )
    if isinstance(node, ms.Translate):
        return operators.shift(
            _array(evaluate(node.operand, ctx)),
            over=node.dimension,
            offset=_amount(node.offset, ctx),
            wrap=node.wrap,
            fill=node.fill,
            by=_partition(node, ctx),
        )
    if isinstance(node, ms.Window):
        return operators.sum_back(
            _array(evaluate(node.operand, ctx)),
            over=node.dimension,
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
    absence = ctx.program.variable(name).absence
    if not ctx.solved:
        return terms.variable_term(variable, absence)
    if "solution" not in variable.data:
        raise RuntimeError(
            f"variable '{name}' has no solution yet: solve the model before reading a named expression"
        )
    return terms.solution(variable, absence)


def _combine(op: Callable[[Value, Value], Value], left: Value, right: Value) -> Value:
    """*left* and *right* combined by *op*, once two arrays agree on their shared coordinates and a hole beside a term has become its absence."""
    if isinstance(left, xr.DataArray) and isinstance(right, xr.DataArray):
        for dim in set(left.dims) & set(right.dims):
            if not left.indexes[dim].equals(right.indexes[dim]):
                raise SpecDataError(
                    f"operands are not aligned on '{dim}': {left.indexes[dim].tolist()[:5]} against "
                    f"{right.indexes[dim].tolist()[:5]}. Every operand is read on the master "
                    f"coordinates, so the data was bound against other labels than the model was built on."
                )
    elif isinstance(left, xr.DataArray) and isinstance(
        right, Variable | LinearExpression | QuadraticExpression
    ):
        right, left = carried(right, left)
    elif isinstance(right, xr.DataArray) and isinstance(
        left, Variable | LinearExpression | QuadraticExpression
    ):
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
        return terms.coefficient(ctx.parameters[amount])
    return amount


def _partition(node: ms.Translate | ms.Window, ctx: Context) -> xr.DataArray | None:
    """The lookup a windowed operator stays inside, named for the dimension its values are labels of."""
    if node.partition is None:
        return None
    array = ctx.lookup(node.partition, node.dimension)
    return array.rename(ctx.program.dimension(node.dimension).targets[node.partition])


def _lookup_arrays(
    over: str, names: tuple[str, ...], ctx: Context
) -> tuple[xr.DataArray, ...]:
    return tuple(ctx.lookup(name, over) for name in names)
