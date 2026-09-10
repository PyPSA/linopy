"""
Is the data there where a declaration needs it? Every position asks.

A parameter row that no source supplies is a hole, and the spec refuses it
wherever the row is used: as a coefficient, where the missing row would
silently drop its term; as a bound, where zero is a bound rather than the
absence of one; as a constant side, where it binds; and as a divisor, where
zero is not a divisor at all. Each is decided against the rows the declaration
actually builds, so a ``where`` that removed the coordinate has already
answered.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import xarray as xr
from math_spec import program as ms

from linopy.spec import terms
from linopy.spec.context import Context
from linopy.spec.errors import SpecDataError
from linopy.spec.nodes import amounts_of, parameters_of
from linopy.spec.where import evaluate_where

Rows = xr.DataArray | None
Obligation = tuple[str, Rows]


@dataclass
class Obligations:
    """Every parameter use under a declaration, each with the rows it has to cover, gathered in one walk."""

    divisors: list[Obligation] = field(default_factory=list)
    constants: list[Obligation] = field(default_factory=list)
    coefficients: list[Obligation] = field(default_factory=list)


def gaps_under(array: xr.DataArray, rows: Rows) -> int:
    """How many slots of *array* are null where *rows* still admits the row; ``None`` narrows nothing."""
    missing = array.isnull()
    if rows is not None:
        missing = missing & rows
    return int(missing.sum())


def check_coverage(
    subject: str,
    expressions: Sequence[ms.ExpressionNode],
    ctx: Context,
    rows: Rows,
    *,
    comparison: bool = False,
) -> None:
    """
    Refuse *subject* if a parameter it reads leaves a row it builds uncovered.

    One walk over *expressions* collects what every parameter has to cover,
    narrowed at each ``cases:`` region; divisors are judged first, then, for a
    *comparison*, the side without a variable term, then every coefficient.
    """
    found = obligations_of(expressions, ctx, rows, comparison=comparison)
    check_divisors(subject, found.divisors, ctx)
    check_constant_sides(subject, found.constants, ctx)
    check_coefficients(subject, found.coefficients, ctx)


def obligations_of(
    expressions: Sequence[ms.ExpressionNode],
    ctx: Context,
    rows: Rows,
    *,
    comparison: bool = False,
) -> Obligations:
    """What the parameters under *expressions* have to cover, a side of a *comparison* without a variable being its constant side."""
    found = Obligations()
    for expression in expressions:
        constant = comparison and not ms.carries_variable(expression)
        _collect(expression, ctx, rows, constant, found)
    return found


def _collect(
    node: ms.ExpressionNode,
    ctx: Context,
    rows: Rows,
    constant: bool,
    into: Obligations,
) -> None:
    if isinstance(node, ms.Divide):
        into.divisors.extend(_divisor_uses(node, ctx, rows))
    if isinstance(node, ms.Parameter):
        if constant:
            into.constants.append((node.name, rows))
        into.coefficients.append((node.name, rows))
    into.coefficients.extend((name, None) for name in amounts_of(node))
    if isinstance(node, ms.Cases):
        for region in node.regions:
            inside = evaluate_where(region.when, ctx)
            narrowed = inside if rows is None else rows & inside
            _collect(region.value, ctx, narrowed, constant, into)
        return
    for child in ms.children(node):
        _collect(child, ctx, rows, constant, into)


def _divisor_uses(quotient: ms.Divide, ctx: Context, rows: Rows) -> list[Obligation]:
    """Each parameter in the divisor, with the rows the quotient is divided over: the region, narrowed by the presence of every numerator variable."""
    params = parameters_of(quotient.divisor)
    if not params:
        return []
    needed = rows
    for variable in sorted(ms.variables_of(quotient.numerator)):
        present = terms.present(ctx.model.variables[variable])
        needed = present if needed is None else needed & present
    return [(param, needed) for param in sorted(params)]


def check_divisors(subject: str, found: Sequence[Obligation], ctx: Context) -> None:
    """
    A divisor must have a value wherever *subject* divides by it.

    Reached before evaluation, the last moment the gap is visible: the
    coefficient fill would turn it into a division by zero.
    """
    for param, needed in found:
        missing = gaps_under(ctx.parameters[param], needed)
        if missing:
            raise SpecDataError(
                f"{subject}: parameter '{param}' is used as a divisor but covers {missing} "
                f"fewer coordinates than it is divided over. A missing row means a zero "
                f"coefficient everywhere else, and zero is not a divisor: the term would drop "
                f"and the row would silently stop constraining.\n"
                f"  Supply the missing rows, or mask the coordinates out with a where."
            )


def check_constant_sides(
    subject: str, found: Sequence[Obligation], ctx: Context
) -> None:
    """A comparison's constant side must have values wherever the row is built, or the zero is the bound."""
    for param, needed in sorted(found, key=lambda pair: pair[0]):
        missing = gaps_under(ctx.parameters[param], needed)
        if missing:
            raise SpecDataError(
                f"{subject}: parameter '{param}' covers {missing} fewer coordinates "
                f"than the rows built here. A missing row is read as 0, and on the constant side "
                f"that zero is a bound rather than an absence: the row still exists, and it binds.\n"
                f"  Supply the missing rows, if the value is what was meant.\n"
                f"  Mask them out with a where, if the row should not exist there."
            )


def check_coefficients(subject: str, found: Sequence[Obligation], ctx: Context) -> None:
    """
    A coefficient parameter must reach every row it is built over.

    A missing coefficient row would otherwise read as a zero, dropping its term
    while the row stays. A shift offset or window width given by name is a
    coefficient too, and stands or falls over its own coordinates.
    """
    for param, needed in found:
        missing = gaps_under(ctx.parameters[param], needed)
        if missing:
            raise SpecDataError(
                f"{subject}: parameter '{param}' is used as a coefficient but leaves "
                f"{missing} of the rows built here uncovered. A missing row reads as a zero "
                f"coefficient, dropping the term while the row stays.\n"
                f"  Supply the missing rows, if a value other than 0 was meant.\n"
                f"  Mask them out with a where, if the row should not exist there."
            )


def check_bounds_cover(
    name: str, declared: ms.VariableDeclaration, ctx: Context, rows: Rows
) -> None:
    """A bound parameter must have a value at every coordinate the variable occupies."""
    names = sorted(parameters_of(declared.lower, declared.upper))
    missing = sum(gaps_under(ctx.parameters[p], rows) for p in names)
    if missing:
        raise SpecDataError(
            f"variable '{name}': {missing} rows have NULL bounds, a bound parameter is missing "
            f"values for some coordinates. The two ways out build different models, so neither "
            f"is picked:\n"
            f"  supply the value           the variable exists there, bounded (`inf` is a value)\n"
            f'  where: "<the parameter>"   the variable does not exist there at all'
        )
