"""
Program plus attached data to linopy declarations.

A build hands every variable to linopy as its term, then adds special-ordered
sets, constraints, the objective and the named expressions that carry a
variable term; which linopy call each construct becomes is one branch of
:func:`linopy.spec.evaluate.evaluate`. Everything built is stamped with the
spec's name and framed in the order its entry declares its dims.
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr
from mathspec import program as ms

from linopy.expressions import LinearExpression, QuadraticExpression
from linopy.model import Model
from linopy.spec.assumptions import check_assumptions
from linopy.spec.attach import Attached
from linopy.spec.context import Context, Parameters, Term, Value
from linopy.spec.errors import SpecDataError, first_coordinates
from linopy.spec.evaluate import (
    carried,
    constant_of,
    evaluate,
    flipped,
    refuse_made,
)
from linopy.spec.where import as_linopy_mask, evaluate_where
from linopy.variables import Variable

_SIGN = {"==": "=", "<=": "<=", ">=": ">="}
_FLIPPED = {"==": "==", "<=": ">=", ">=": "<="}
_SENSE = {"minimize": "min", "maximize": "max"}


def build(model: Model, attached: Attached, name: str) -> None:
    """
    Add every declaration of the attached program to *model* as the spec *name*.

    Variables, special-ordered sets, constraints, the objective and the
    named expressions holding a variable term, in that order. The program's
    assumptions are checked once the variables exist, so a condition may
    read where one is defined.
    """
    check_supported(attached.program)
    ctx = Context(
        model,
        attached.program,
        attached.coords,
        attached.relations,
        Parameters(attached.program, attached.read),
        name,
    )
    _variables(ctx)
    check_assumptions(ctx)
    _sos(ctx)
    _constraints(ctx)
    _objective(ctx)
    _expressions(ctx)


def check_supported(program: ms.Program) -> None:
    """
    Refuse the constructs of *program* linopy cannot build, before any of it is built.

    A product of two variable-carrying operands is a quadratic term, and
    linopy carries one in the objective only: a constraint holding one has no
    linopy form to be built into.
    """
    if "constraint" in program.footprint.quadratic:
        raise NotImplementedError(
            "a constraint of the spec multiplies two variable-carrying operands, and linopy "
            "carries a quadratic term in the objective only. Move the product into the "
            "objective, or write the constraint so that at most one side of each product "
            "holds a variable."
        )


def _variables(ctx: Context) -> None:
    for name, declared in ctx.program.variables.items():
        mask = as_linopy_mask(evaluate_where(declared.where, ctx))
        variable = ctx.model.add_variables(
            lower=_bound(declared.lower, ctx, -np.inf),
            upper=_bound(declared.upper, ctx, np.inf),
            coords={d: ctx.coords[d] for d in declared.dims},
            name=name,
            mask=mask,
            binary=declared.domain == "binary",
            integer=declared.domain == "integer",
        )
        variable.spec = ctx.name


def _bound(
    node: ms.Expression | None, ctx: Context, open_side: float
) -> float | xr.DataArray:
    """A bound as linopy takes it: an open side, and a missing ``absent`` row, is *open_side*."""
    if node is None:
        return open_side
    if isinstance(node, ms.ParameterReference):
        return ctx.parameters[node.name].fillna(open_side)
    if isinstance(node, ms.Constant):
        return node.value
    raise TypeError(f"a bound is a number or a parameter, not {type(node).__name__}")


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
        scoped = ctx.within(rows, f"constraint '{name}'")
        if flipped(row):
            sign = _SIGN[_FLIPPED[row.sense]]
            term, other = evaluate(row.rhs, scoped), evaluate(row.lhs, scoped)
        else:
            sign = _SIGN[row.sense]
            term, other = evaluate(row.lhs, scoped), evaluate(row.rhs, scoped)
        if not isinstance(term, Term):
            raise SpecDataError(
                f"constraint '{name}' holds no variable term once the data is attached, "
                f"so there is nothing to constrain."
            )
        if any(term.sizes[d] == 0 for d in term.coord_dims):
            continue
        _check_live(name, row, term, other, rows, ctx)
        if isinstance(other, xr.DataArray):
            const = constant_of(term)
            params = ms.parameters_of(row.lhs, row.rhs)
            refuse_made("comparison", params, const - other, const, other, scoped)
            term, other = carried(term, other)
        built = ctx.model.add_constraints(
            _framed(term, row.dims), sign, other, name=name, mask=mask
        )
        built.spec = ctx.name


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
    expr = evaluate(declared.expression, ctx)
    if not isinstance(expr, Term):
        raise SpecDataError(
            "the objective carries no variable term once the data is attached, so there is nothing to optimize"
        )
    ctx.model.add_objective(expr, overwrite=True, sense=_SENSE[declared.sense])


def _expressions(ctx: Context) -> None:
    """
    Every named expression holding a variable term added to the model.

    A data-only body has no linopy term to hold and stays on the spec; so
    does one reading a ``dual``, which needs a solved model.
    """
    for name, declared in ctx.program.expressions.items():
        body = declared.expression
        if any(isinstance(n, ms.Dual) for n in ms.walk(body)):
            continue
        value = evaluate(body, ctx)
        if isinstance(value, Term):
            built = ctx.model.add_expressions(_framed(value, declared.dims), name=name)
            built.spec = ctx.name


def _framed(
    term: Term, dims: tuple[str, ...]
) -> LinearExpression | QuadraticExpression:
    """
    *term* with its dimensions in the order the entry declares, without the
    coordinates a lookup leaves beside them, which no file keeps. A dimension
    the entry declares and the term lacks is what the other side of a
    comparison brings in.
    """
    expression = term.to_linexpr() if isinstance(term, Variable) else term
    stray = [c for c in expression.coords if c not in expression.dims]
    if stray:
        expression = expression.drop_vars(stray)
    order = [d for d in dims if d in expression.coord_dims]
    if isinstance(expression, LinearExpression) and expression.is_sparse:
        store = expression._store.transpose(order)
        return LinearExpression._from_store(store, expression.model)
    data = expression.data.transpose(*order, ...)
    framed = xr.Dataset(coords={d: data.indexes[d] for d in order}).assign(
        {str(k): v.variable for k, v in data.data_vars.items()}
    )
    return type(expression)(framed, expression.model)
