"""
Program plus attached data to linopy declarations.

A build hands every variable to linopy as its term, then adds special-ordered
sets, constraints, the objective and the named expressions that carry a
variable term; which linopy call each construct becomes is one branch of
:func:`linopy.spec.evaluator.evaluate`. Everything built is framed in the
order its entry declares its dims.
"""

from __future__ import annotations

import operator
import warnings
from dataclasses import replace
from typing import cast

import numpy as np
import xarray as xr
from mathspec import program as ms

from linopy.constants import FACTOR_DIM, TERM_DIM
from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.spec.context import Context, Term, Value, as_expression
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
from linopy.spec.where import as_linopy_mask, evaluate_where
from linopy.variables import Variable

_FLIPPED: dict[str, ms.ConstraintSense] = {"==": "==", "<=": ">=", ">=": "<="}
_SENSE = {"minimize": "min", "maximize": "max"}


def build(ctx: Context) -> None:
    """
    Add every declaration of the program to the model of *ctx*.

    Variables, special-ordered sets, constraints, the objective and the
    named expressions holding a variable term, in that order. The program's
    assumptions are checked once the variables exist, so a condition may
    read where one is defined. Each declaration refuses the missing
    ``refused`` rows its live rows read, a named expression's whether it is
    built or left to :func:`linopy.spec.evaluate`.
    """
    _variables(ctx)
    _check_assumptions(ctx)
    _sos(ctx)
    _constraints(ctx)
    _objective(ctx)
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
        admitted = evaluate_where(assumption.where, ctx)
        held = evaluate_where(assumption.holds, ctx)
        failing = admitted & ~held
        if bool(failing.any()):
            raise SpecDataError(
                f"{ms.assumption_message(name, assumption)}\n"
                f"  Not so at {first_coordinates(failing, 3)}"
            )


def _variables(ctx: Context) -> None:
    for name, declared in ctx.program.variables.items():
        rows = evaluate_where(declared.where, ctx)
        for bound in (declared.lower, declared.upper):
            if bound is not None:
                refuse_holes(f"variable '{name}'", [bound], ctx, rows)
        ctx.model.add_variables(
            lower=_bound(declared.lower, ctx, -np.inf),
            upper=_bound(declared.upper, ctx, np.inf),
            coords={d: ctx.coords[d] for d in declared.dims},
            name=name,
            mask=as_linopy_mask(rows),
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
            ctx.model.variables[sos.variable],
            sos_type=sos.sos_type,
            sos_dim=sos.along,
        )


def _constraints(ctx: Context) -> None:
    for name, row in ctx.program.constraints.items():
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
        _check_live(name, row, term, other, rows, ctx)
        if isinstance(other, xr.DataArray):
            params = ms.parameters_of(row.lhs, row.rhs)
            const = constant_of(term)
            refuse_made("comparison", params, operator.sub, const, other, scoped)
            term, other = carried(term, other)
        ctx.model.add_constraints(
            _framed(term, row.dims), sense, other, name=name, mask=mask
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


def _check_live(
    name: str,
    declared: ms.Constraint,
    term: Term,
    other: Value,
    rows: xr.DataArray,
    ctx: Context,
) -> None:
    """
    Refuse a row the data emptied of every variable term.

    Such a row reads as ``0 sense rhs``: linopy carries no column there, the
    row leaves the problem and the constraint silently stops binding. Where
    every variable of the row declares ``missing: neutral`` the zero is what
    the math says, so a zero other side is warned about rather than refused.
    An absent row is not emptied: absence spread through arithmetic, a masked
    variable or a vacated translation, takes the row with it. A bare variable
    term is never emptied: a row of it exists exactly where the variable does.
    """
    if isinstance(term, Variable):
        return
    dead = rows & ~(term.has_terms | term.isnull())
    if not bool(dead.any()):
        return
    said = (
        f"constraint '{name}': {int(dead.sum())} row(s) hold no variable term once the data "
        f"is attached, the first at {first_coordinates(dead, 3)}. Nothing is left to constrain there, so "
        f"the row leaves the problem without saying so."
    )
    neutral = all(
        ctx.program.variables[v].missing == "neutral"
        for v in ms.variables_of(declared.lhs, declared.rhs)
    )
    if neutral and not _binds(other, dead):
        warnings.warn(
            f"{said} Every variable there is missing: neutral and the other side is 0, "
            f"so the row is trivially true.",
            UserWarning,
            stacklevel=2,
        )
        return
    supply = (
        "  Supply the rows of the variables, if the row is meant to bind."
        if neutral
        else "  Declare missing: neutral on the variables, if an absent term is a zero there."
    )
    raise SpecDataError(
        f"{said}\n"
        f"  Mask them out with a where on the constraint, if the row should not exist there.\n"
        f"{supply}"
    )


def _binds(other: Value, dead: xr.DataArray) -> bool:
    """Whether the side without the variable term is anything but 0 on a row *dead* names."""
    if isinstance(other, xr.DataArray):
        return bool((other.where(dead, 0.0) != 0).any())
    return other != 0


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
    ctx.model.add_objective(expr, overwrite=True, sense=_SENSE[declared.sense])


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
        value = cast(Term, evaluate(body, ctx))
        ctx.model.add_expressions(_framed(value, declared.dims), name=name)


def _framed(
    term: Term, dims: tuple[str, ...]
) -> LinearExpression | QuadraticExpression:
    """
    *term* with its dimensions in the order the entry declares, without
    stray coordinates. A dimension the entry declares and the term lacks is
    what the other side of a comparison brings in.
    """
    expression = without_strays(as_expression(term))
    order = [d for d in dims if d in expression.coord_dims]
    if isinstance(expression, LinearExpression) and expression.is_sparse:
        store = expression._store.transpose(order)
        return LinearExpression._from_store(store, expression.model)
    data = expression.data.transpose(*order, ...)
    framed = xr.Dataset(coords={d: data.indexes[d] for d in order}).assign(
        {str(k): v.variable for k, v in data.data_vars.items()}
    )
    return type(expression)(framed, expression.model)
