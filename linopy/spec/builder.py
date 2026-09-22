"""
Program plus attached data to linopy declarations.

A build hands every variable to linopy as its term, then adds special-ordered
sets, constraints, the objective and the named expressions that carry a
variable term; which linopy call each construct becomes is one branch of
:func:`linopy.spec.evaluate.evaluate`. Everything built is stamped with the
spec's name.
"""

from __future__ import annotations

import warnings
from typing import TypeGuard

import xarray as xr
from math_spec import program as ms
from math_spec.program import walk

from linopy.model import Model
from linopy.spec import curves
from linopy.spec.attach import Attached
from linopy.spec.context import Context, Parameters, Term, Value
from linopy.spec.coverage import check_bounds_cover, check_coverage
from linopy.spec.errors import SpecDataError, first_coordinates
from linopy.spec.evaluate import carried, evaluate
from linopy.spec.where import as_linopy_mask, evaluate_where
from linopy.variables import Variable

_SIGN = {"==": "=", "<=": "<=", ">=": ">="}
_FLIPPED = {"==": "==", "<=": ">=", ">=": "<="}
_SENSE = {"minimize": "min", "maximize": "max"}


def build(
    model: Model, attached: Attached, name: str, build_expressions: bool = True
) -> None:
    """
    Add every declaration of the attached program to *model* as the spec *name*.

    Variables, special-ordered sets, constraints, the objective and, with
    *build_expressions*, the named expressions holding a variable term, in
    that order. Every named expression is checked for divisor and
    coefficient coverage either way, so a body that cannot be folded is
    refused at build rather than at read.
    """
    check_supported(attached.program)
    ctx = Context(
        model,
        attached.program,
        attached.coords,
        attached.relations,
        Parameters(attached.program, attached.parameter),
        name,
    )
    curves.validate(ctx.program, ctx.parameters)
    _variables(ctx)
    _sos(ctx)
    _constraints(ctx)
    _objective(ctx)
    _expressions(ctx, build_expressions)


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
        check_bounds_cover(name, declared, ctx, mask)
        variable = ctx.model.add_variables(
            lower=_bound(declared.lower, ctx),
            upper=_bound(declared.upper, ctx),
            coords={d: ctx.coords[d] for d in declared.dims},
            name=name,
            mask=mask,
            binary=declared.domain == "binary",
            integer=declared.domain == "integer",
        )
        variable.spec = ctx.name


def _bound(node: ms.Expression, ctx: Context) -> float | xr.DataArray:
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
        if _has_term(lhs):
            term, other, sign = lhs, rhs, _SIGN[row.sense]
        elif _has_term(rhs):
            term, other, sign = rhs, lhs, _SIGN[_FLIPPED[row.sense]]
        else:
            continue
        _check_live(name, row, term, other, rows, ctx)
        if isinstance(other, xr.DataArray):
            term, other = carried(term, other)
        built = ctx.model.add_constraints(term, sign, other, name=name, mask=mask)
        built.spec = ctx.name


def _check_live(
    name: str,
    declared: ms.ConstraintDeclaration,
    term: Term,
    other: Value,
    rows: xr.DataArray,
    ctx: Context,
) -> None:
    """
    Refuse a row the data emptied of every variable term.

    Such a row reads as ``0 sense rhs``: linopy carries no column there, the
    row leaves the problem and the constraint silently stops binding. Where
    every variable of the row declares ``absence: zero`` the zero is what the
    math says, so a zero other side is warned about rather than refused.
    """
    live = term.mask if isinstance(term, Variable) else term.has_terms
    dead = rows & ~live
    if not bool(dead.any()):
        return
    said = (
        f"constraint '{name}': {int(dead.sum())} row(s) hold no variable term once the data "
        f"is attached, the first at {first_coordinates(dead, 3)}. Nothing is left to constrain there, so "
        f"the row leaves the problem without saying so."
    )
    zeroed = all(
        ctx.program.variables[v].absence == "zero"
        for v in ms.variables_of(declared.lhs, declared.rhs)
    )
    if zeroed and not _binds(other, dead):
        warnings.warn(
            f"{said} Every variable there is absence: zero and the other side is 0, "
            f"so the row is trivially true.",
            UserWarning,
            stacklevel=2,
        )
        return
    supply = (
        "  Supply the rows of the variables, if the row is meant to bind."
        if zeroed
        else "  Declare absence: zero on the variables, if an absent term is a zero there."
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


def _has_term(side: Value) -> TypeGuard[Term]:
    """Whether *side* holds a variable term: not data, and not an expression the data emptied."""
    if not isinstance(side, Term):
        return False
    return isinstance(side, Variable) or side.nterm > 0


def _objective(ctx: Context) -> None:
    declared = ctx.program.objective
    if declared is None:
        return
    check_coverage("the objective", (declared.expression,), ctx, None)
    expr = evaluate(declared.expression, ctx)
    if not isinstance(expr, Term):
        raise SpecDataError(
            "the objective carries no variable term once the data is attached, so there is nothing to optimize"
        )
    ctx.model.add_objective(expr, overwrite=True, sense=_SENSE[declared.sense])


def _expressions(ctx: Context, build: bool) -> None:
    """
    Every named expression coverage-checked; with *build*, the ones holding a variable term added to the model.

    A data-only body has no linopy term to hold and stays on the spec; so
    does one reading a ``dual``, which needs a solved model.
    """
    for name, declared in ctx.program.expressions.items():
        body = declared.expression
        check_coverage(f"expression '{name}'", (body,), ctx, None)
        if not build or any(isinstance(n, ms.Dual) for n in walk(body)):
            continue
        value = evaluate(body, ctx)
        if isinstance(value, Term):
            ctx.model.add_expressions(value, name=name).spec = ctx.name
