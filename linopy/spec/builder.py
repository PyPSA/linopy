"""
Program plus bound data to linopy declarations, and a named expression to its value.

One evaluator serves both: a build hands every variable to linopy as its
term, a fold hands it in as its solved values, and every other node reads the
same way. Which linopy call each construct becomes is one branch of
:func:`evaluate` or one section below.
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable
from typing import assert_never

import xarray as xr
from math_spec import did_you_mean
from math_spec import program as ms

from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.model import Model
from linopy.spec import curves, operators, terms
from linopy.spec.binder import Bound
from linopy.spec.context import Context, Parameters
from linopy.spec.coverage import (
    check_bounds_cover,
    check_coefficients_cover,
    check_constant_side_covers,
    check_divisors_cover,
)
from linopy.spec.errors import SpecDataError
from linopy.spec.terms import Array, Term, Value
from linopy.spec.where import as_linopy_mask, bound_lookup, evaluate_where
from linopy.variables import Variable

_SIGN = {"==": "=", "<=": "<=", ">=": ">="}
_FLIPPED = {"==": "==", "<=": ">=", ">=": "<="}
_SENSE = {"minimize": "min", "maximize": "max"}


def build(model: Model, bound: Bound) -> None:
    """
    Add every declaration of the bound program to *model*.

    Variables, special-ordered sets, constraints and the objective, in that
    order; then every named expression is checked for divisor and coefficient
    coverage, so a body that cannot be folded is refused at build rather than
    at read.
    """
    ctx = Context(
        model,
        bound.program,
        bound.coords,
        bound.lookups,
        Parameters(bound.program, bound.parameter),
    )
    curves.validate(ctx.program, ctx.parameters)
    _variables(ctx)
    _sos(ctx)
    _constraints(ctx)
    _objective(ctx)
    for name, body in ctx.program.named_expressions.items():
        check_divisors_cover(f"expression '{name}'", (body,), ctx, None)
        check_coefficients_cover(f"expression '{name}'", (body,), ctx, None)


def fold(name: str, ctx: Context) -> xr.DataArray:
    """The named expression *name* as data, folded over the solution and the parameters *ctx* holds."""
    if name not in ctx.program.named_expressions:
        raise KeyError(
            f"unknown named expression '{name}'. "
            + did_you_mean(name, ctx.program.named_expressions)
        )
    body = ctx.program.named_expressions[name]
    check_divisors_cover(f"expression '{name}'", (body,), ctx, None)
    value = evaluate(body, ctx)
    if isinstance(value, xr.DataArray):
        stray = [c for c in value.coords if c not in value.dims]
        return value.drop_vars(stray).rename(name)
    if isinstance(value, float | int):
        return xr.DataArray(float(value), name=name)
    raise TypeError(
        f"expression '{name}' folded to a {type(value).__name__}, not to data"
    )


# ---------------------------------------------------------------------------
# declarations
# ---------------------------------------------------------------------------


def _variables(ctx: Context) -> None:
    for name, declared in ctx.program.variables.items():
        rows = evaluate_where(declared.where, ctx)
        check_bounds_cover(name, declared, ctx, as_linopy_mask(rows))
        ctx.model.add_variables(
            lower=_bound(declared.lower, ctx),
            upper=_bound(declared.upper, ctx),
            coords={d: ctx.coords[d] for d in declared.dims},
            name=name,
            mask=as_linopy_mask(rows),
            binary=declared.variable_type == "binary",
            integer=declared.variable_type == "integer",
        )


def _bound(node: ms.ExpressionNode, ctx: Context) -> float | xr.DataArray:
    """A bound as linopy takes it, read raw: an uncovered slot stays NaN for :func:`check_bounds_cover`."""
    if isinstance(node, ms.Constant):
        return node.value
    if isinstance(node, ms.Parameter):
        return ctx.parameters[node.name]
    raise TypeError(f"a bound is a number or a parameter, not {type(node).__name__}")


def _sos(ctx: Context) -> None:
    for sos in ctx.program.sos.values():
        ctx.model.add_sos_constraints(
            ctx.model.variables[sos.variable],
            sos_type=sos.sos_type,
            sos_dim=sos.over,
            big_m=sos.big_m,
        )


def _constraints(ctx: Context) -> None:
    for name, row in ctx.program.constraints.items():
        rows = evaluate_where(row.where, ctx)
        mask = as_linopy_mask(rows)
        check_divisors_cover(f"constraint '{name}'", (row.lhs, row.rhs), ctx, mask)
        check_constant_side_covers(name, row, ctx, mask)
        check_coefficients_cover(f"constraint '{name}'", (row.lhs, row.rhs), ctx, mask)
        lhs, rhs = evaluate(row.lhs, ctx), evaluate(row.rhs, ctx)
        if _term_free(lhs) and _term_free(rhs):
            continue
        term, other, sense = _sides(lhs, rhs, row.sense)
        if isinstance(other, xr.DataArray):
            term, other = _carried(term, other)
        ctx.model.add_constraints(term, _SIGN[sense], other, name=name, mask=mask)


def _sides(lhs: Value, rhs: Value, sense: str) -> tuple[Term, Value, str]:
    """The comparison with a term on the left, as linopy takes it; a swap flips the sense."""
    if isinstance(lhs, Variable | LinearExpression | QuadraticExpression):
        return lhs, rhs, sense
    if isinstance(rhs, Variable | LinearExpression | QuadraticExpression):
        return rhs, lhs, _FLIPPED[sense]
    raise TypeError("a constraint needs a variable term on one side")


def _term_free(side: Value) -> bool:
    """Whether *side* has nowhere for a variable term to sit: data, or an expression the data emptied."""
    if isinstance(side, Variable):
        return False
    if isinstance(side, LinearExpression | QuadraticExpression):
        return side.nterm == 0
    return True


def _objective(ctx: Context) -> None:
    declared = ctx.program.objective
    if declared is None:
        return
    check_divisors_cover("the objective", (declared.expression,), ctx, None)
    check_coefficients_cover("the objective", (declared.expression,), ctx, None)
    expr = evaluate(declared.expression, ctx)
    if not isinstance(expr, Variable | LinearExpression | QuadraticExpression):
        raise SpecDataError(
            "the objective carries no variable term once the data is bound, so there is nothing to optimize"
        )
    ctx.model.add_objective(expr, overwrite=True, sense=_SENSE[declared.sense])


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


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
        right, left = _carried(right, left)
    elif isinstance(right, xr.DataArray) and isinstance(
        left, Variable | LinearExpression | QuadraticExpression
    ):
        left, right = _carried(left, right)
    return op(left, right)


def _carried(term: Term, data: xr.DataArray) -> tuple[Term, xr.DataArray]:
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
    array = bound_lookup(node.partition, node.dimension, ctx.lookups)
    return array.rename(ctx.program.dimension(node.dimension).targets[node.partition])


def _lookup_arrays(
    over: str, names: tuple[str, ...], ctx: Context
) -> tuple[xr.DataArray, ...]:
    return tuple(bound_lookup(name, over, ctx.lookups) for name in names)
