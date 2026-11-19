"""
Program plus attached data to linopy declarations.

A build hands every variable to linopy as its term, then adds special-ordered
sets, constraints, the objective and the named expressions that carry a
variable term; which linopy call each construct becomes is one branch of
:func:`linopy.spec.evaluator.evaluate`. Everything built is framed in the
order :meth:`~linopy.spec.context.Context.frame` gives its dims: the dims
``dim_order`` names first, the rest in the name order mathspec sorts every
``dims:`` list into.
"""

from __future__ import annotations

import operator
from dataclasses import replace
from typing import cast

import numpy as np
import xarray as xr
from mathspec import program as ms

from linopy.constants import FACTOR_DIM, TERM_DIM
from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.spec.context import Context, Term, Value, as_expression, is_empty
from linopy.spec.errors import SpecDataError, first_coordinates
from linopy.spec.evaluator import (
    carried,
    constant_of,
    evaluate,
    flipped,
    refuse_holes,
    refuse_made,
    without_strays,
)
from linopy.spec.where import admits_a_row, as_linopy_mask, evaluate_where
from linopy.variables import Variable

_FLIPPED: dict[str, ms.ConstraintSense] = {"==": "==", "<=": ">=", ">=": "<="}
_SENSE = {"minimize": "min", "maximize": "max"}


def build(ctx: Context, expressions: bool = True) -> None:
    """
    Add every declaration of the program to the model of *ctx*.

    Variables, special-ordered sets, constraints, the objective and, with
    *expressions*, the named expressions holding a variable term, in that
    order. A declaration without a row costs next to nothing: a variable
    over a dimension without a coordinate is added with its bounds unread,
    and a constraint over one, or one whose ``where`` admits no row, is
    skipped unread, as is an assumption admitted nowhere or over such a
    dimension. The program's assumptions are checked once the variables
    exist, so a condition may read where one is defined. Each declaration
    refuses the missing ``refused`` rows its live rows read, a named
    expression's whether it is built or left to :func:`linopy.spec.evaluate`.
    """
    _variables(ctx)
    _check_assumptions(ctx)
    _sos(ctx)
    _constraints(ctx)
    _objective(ctx)
    if expressions:
        _expressions(ctx)


def _check_assumptions(ctx: Context) -> None:
    """
    Refuse the data wherever an assumption of the program does not hold.

    The language decides nothing about the numbers, so every fact it needs of
    them, the file's own and the ones a ``piecewise:`` method implies, is an
    ``assumptions:`` entry. One holds at every coordinate its ``where`` admits;
    a missing row on either side of a comparison reads as false there.
    """
    for name, assumption in ctx.program.assumptions.items():
        holds, where = assumption.holds, assumption.where
        if is_empty(holds.dims, ctx) or not admits_a_row(where, ctx):
            continue
        admitted = evaluate_where(where, ctx)
        held = evaluate_where(holds, ctx)
        failing = admitted & ~held
        if bool(failing.any()):
            raise SpecDataError(
                f"{ms.assumption_message(name, assumption)}\n"
                f"  Not so at {first_coordinates(failing, 3)}"
            )


def _variables(ctx: Context) -> None:
    for name, declared in ctx.program.variables.items():
        bounds = [b for b in (declared.lower, declared.upper) if b is not None]
        lower: float | xr.DataArray = -np.inf
        upper: float | xr.DataArray = np.inf
        mask = None
        if is_empty(declared.dims, ctx):
            ctx.memo.skipped(*bounds)
        else:
            rows = evaluate_where(declared.where, ctx)
            for bound in bounds:
                refuse_holes(f"variable '{name}'", [bound], ctx, rows)
            lower = _bound(declared.lower, ctx, -np.inf)
            upper = _bound(declared.upper, ctx, np.inf)
            mask = as_linopy_mask(rows)
        ctx.model.add_variables(
            lower=lower,
            upper=upper,
            coords={d: ctx.coords[d] for d in ctx.frame(declared.dims)},
            name=ctx.rename(name),
            mask=mask,
            binary=declared.domain == "binary",
            integer=declared.domain == "integer",
        )


def _bound(
    node: ms.Expression | None, ctx: Context, open_side: float
) -> float | xr.DataArray:
    """A bound, a number or a parameter, as linopy takes it: an open side, and a missing ``absent`` row, is *open_side*."""
    if node is None:
        return open_side
    value = evaluate(node, ctx)
    if isinstance(value, xr.DataArray):
        return value.fillna(open_side)
    return cast(float, value)


def _sos(ctx: Context) -> None:
    for sos in ctx.program.sos.values():
        ctx.model.add_sos_constraints(
            ctx.model.variables[ctx.rename(sos.variable)],
            sos_type=sos.sos_type,
            sos_dim=sos.along,
        )


def _constraints(ctx: Context) -> None:
    for name, row in ctx.program.constraints.items():
        if is_empty(row.dims, ctx) or not admits_a_row(row.where, ctx):
            ctx.memo.skipped(row.lhs, row.rhs)
            continue
        rows = evaluate_where(row.where, ctx)
        mask = as_linopy_mask(rows)
        scoped = replace(ctx, rows=rows, subject=f"constraint '{name}'")
        refuse_holes(scoped.subject, [row.lhs, row.rhs], ctx, rows)
        lhs, rhs, sense = row.lhs, row.rhs, row.sense
        if flipped(row):
            lhs, rhs, sense = rhs, lhs, _FLIPPED[sense]
        term, other = (
            _linear(name, evaluate(side, scoped), rows) for side in (lhs, rhs)
        )
        term = cast(Term, term)
        if any(term.sizes[d] == 0 for d in term.coord_dims):
            continue
        if not bool((rows & _present(term) & _present(other)).any()):
            continue
        if isinstance(other, xr.DataArray):
            const = constant_of(term)
            refuse_made(
                "comparison", (row.lhs, row.rhs), operator.sub, const, other, scoped
            )
            term, other = carried(term, other)
        ctx.model.add_constraints(
            _framed(term, row.dims, ctx),
            sense,
            other,
            name=ctx.rename(name),
            mask=mask,
        )


def _linear(name: str, side: Value, rows: xr.DataArray) -> Value:
    """
    One *side* of a constraint as a linear expression, refused where the data leaves
    a product of two variables with a nonzero coefficient on a row *rows* keeps.

    A product whose coefficient is zero, or one of whose variables is absent or
    masked, holds nothing to constrain, so the row keeps its linear terms only.
    """
    if not isinstance(side, QuadraticExpression):
        return side
    paired = (side.vars != -1).all(FACTOR_DIM)
    live = rows & (paired & (side.coeffs.fillna(0) != 0)).any(TERM_DIM)
    if bool(live.any()):
        raise NotImplementedError(
            f"constraint '{name}': {int(live.sum())} row(s) multiply two variables with a "
            f"nonzero coefficient once the data is attached, the first at "
            f"{first_coordinates(live, 3)}, and linopy carries a quadratic term in the "
            f"objective only. Move the product into the objective, or write the constraint "
            f"so that at most one side of each product holds a variable."
        )
    vars = side.vars.max(FACTOR_DIM).where(~paired, -1)
    return LinearExpression(side.data.assign(vars=vars), side.model)


def _present(side: Value) -> xr.DataArray:
    """
    Per row, whether *side* is present there.

    Absence spread through arithmetic, a masked variable or a vacated
    translation, takes the row with it. A present row stays, also one the
    data leaves without a variable term: it reaches the solver as
    ``constant sense constant``, like any row core linopy builds.
    """
    if isinstance(side, Variable):
        return side.labels != -1
    return constant_of(side).notnull()


def _objective(ctx: Context) -> None:
    declared = ctx.program.objective
    if declared is None:
        return
    refuse_holes("the objective", [declared.expression], ctx)
    expr = evaluate(declared.expression, ctx)
    if not isinstance(expr, Term):
        raise SpecDataError(
            "the objective carries no variable term once the data is attached, so there is nothing to optimize"
        )
    ctx.model.add_objective(
        _linear_if_no_product(expr), overwrite=True, sense=_SENSE[declared.sense]
    )


def _linear_if_no_product(expr: Term) -> Term:
    """
    A quadratic *expr* whose products the data zeroes, as the linear expression it is.

    A spec states a quadratic cost once, and data whose quadratic coefficients
    are all zero leaves only linear terms; the objective is then linear, so a
    solver without quadratic support takes it and it can be read as a
    constraint's left-hand side.
    """
    if not isinstance(expr, QuadraticExpression):
        return expr
    data = expr.data
    first = data.vars.isel({FACTOR_DIM: 0})
    second = data.vars.isel({FACTOR_DIM: 1})
    product = (first != -1) & (second != -1) & (data.coeffs != 0)
    if bool(product.any()):
        return expr
    vars_ = xr.where(first != -1, first, second)
    coeffs = data.coeffs.where(vars_ != -1, 0)
    linear = xr.Dataset({"coeffs": coeffs, "vars": vars_, "const": data.const})
    return LinearExpression(linear, expr.model)


def _expressions(ctx: Context) -> None:
    """
    Every named expression holding a variable term added to the model.

    A data-only body has no linopy term to hold, and one reading a ``dual``
    needs a solved model: both are left to :func:`linopy.spec.evaluate`.
    """
    for name, declared in ctx.program.expressions.items():
        body = declared.expression
        refuse_holes(f"expression '{name}'", [body], ctx)
        if not ms.carries_variable(body) or any(
            isinstance(n, ms.Dual) for n in ms.walk(body)
        ):
            continue
        reference = ms.ExpressionReference(name, body, dims=declared.dims)
        value = cast(Term, evaluate(reference, ctx))
        ctx.model.add_expressions(_framed(value, declared.dims, ctx), name=name)


def _framed(
    term: Term, dims: tuple[str, ...], ctx: Context
) -> LinearExpression | QuadraticExpression:
    """
    *term* with its dimensions in the order :meth:`Context.frame` gives the
    entry's *dims*, without stray coordinates. A dimension the entry declares
    and the term lacks is what the other side of a comparison brings in, and
    lands after the term's own, as a piecewise breakpoint dim does; one
    ``dim_order`` names is spread onto the term to take its place in front.
    """
    expression = without_strays(as_expression(term))
    order = [
        d for d in ctx.frame(dims) if d in expression.coord_dims or d in ctx.dim_order
    ]
    lacking = {d: ctx.coords[d] for d in order if d not in expression.coord_dims}
    if lacking:
        shape = [len(index) for index in lacking.values()]
        expression = expression + xr.DataArray(np.zeros(shape), coords=lacking)
    if isinstance(expression, LinearExpression) and expression.is_sparse:
        store = expression._store.transpose(order)
        return LinearExpression._from_store(store, expression.model)
    data = expression.data
    if set(data.coords) == set(order) and all(
        v.dims[: len(order)] == tuple(order) for v in data.data_vars.values()
    ):
        return expression
    data = data.transpose(*order, ...)
    framed = xr.Dataset(coords={d: data.indexes[d] for d in order}).assign(
        {str(k): v.variable for k, v in data.data_vars.items()}
    )
    return type(expression)(framed, expression.model)
