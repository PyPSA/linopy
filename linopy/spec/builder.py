"""
Program plus bound data to linopy declarations.

A build hands every variable to linopy as its term, then adds special-ordered
sets, constraints and the objective; which linopy call each construct becomes
is one branch of :func:`linopy.spec.evaluate.evaluate`.
"""

from __future__ import annotations

import xarray as xr
from math_spec import program as ms

from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.model import Model
from linopy.spec import curves
from linopy.spec.binder import Bound
from linopy.spec.context import Context
from linopy.spec.coverage import check_bounds_cover, check_coverage
from linopy.spec.errors import SpecDataError
from linopy.spec.evaluate import carried, evaluate
from linopy.spec.parameters import Parameters
from linopy.spec.terms import Term, Value
from linopy.spec.where import as_linopy_mask, evaluate_where
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
        check_coverage(f"expression '{name}'", (body,), ctx, None)


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
        check_coverage(
            f"constraint '{name}'", (row.lhs, row.rhs), ctx, mask, comparison=True
        )
        lhs, rhs = evaluate(row.lhs, ctx), evaluate(row.rhs, ctx)
        if _term_free(lhs) and _term_free(rhs):
            continue
        term, other, sense = _sides(lhs, rhs, row.sense)
        if isinstance(other, xr.DataArray):
            term, other = carried(term, other)
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
    check_coverage("the objective", (declared.expression,), ctx, None)
    expr = evaluate(declared.expression, ctx)
    if not isinstance(expr, Variable | LinearExpression | QuadraticExpression):
        raise SpecDataError(
            "the objective carries no variable term once the data is bound, so there is nothing to optimize"
        )
    ctx.model.add_objective(expr, overwrite=True, sense=_SENSE[declared.sense])
