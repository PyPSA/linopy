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

from collections.abc import Iterator

import xarray as xr
from math_spec import program as ms

from linopy.spec import terms
from linopy.spec.context import Context
from linopy.spec.errors import SpecDataError
from linopy.spec.nodes import children, parameters_of
from linopy.spec.where import evaluate_where

Rows = xr.DataArray | None


def gaps_under(array: xr.DataArray, rows: Rows) -> int:
    """How many slots of *array* are null where *rows* still admits the row; ``None`` narrows nothing."""
    missing = array.isnull()
    if rows is not None:
        missing = missing & rows
    return int(missing.sum())


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


def check_constant_side_covers(
    name: str, row: ms.ConstraintDeclaration, ctx: Context, rows: Rows
) -> None:
    """A comparison's constant side must have values wherever the row is built, or the zero is the bound."""
    for side in (row.lhs, row.rhs):
        if ms.carries_variable(side):
            continue
        found = sorted(
            (
                (node.name, narrowed)
                for node, narrowed in _under_regions(side, ctx, rows)
                if isinstance(node, ms.Parameter)
            ),
            key=lambda pair: pair[0],
        )
        for param, narrowed in found:
            missing = gaps_under(ctx.parameters[param], narrowed)
            if missing:
                raise SpecDataError(
                    f"constraint '{name}': parameter '{param}' covers {missing} fewer coordinates "
                    f"than the rows built here. A missing row is read as 0, and on the constant side "
                    f"that zero is a bound rather than an absence: the row still exists, and it binds.\n"
                    f"  Supply the missing rows, if the value is what was meant.\n"
                    f"  Mask them out with a where, if the row should not exist there."
                )


def check_divisors_cover(
    subject: str, expressions: tuple[ms.ExpressionNode, ...], ctx: Context, rows: Rows
) -> None:
    """
    A divisor must have a value wherever *subject* divides by it.

    The rows that ask are the declaration's own, narrowed by the presence of
    every variable in the quotient's numerator and by the region of a
    ``cases:`` block. Reached before evaluation, the last moment the gap is
    visible: the coefficient fill would turn it into a division by zero.
    """
    for expression in expressions:
        for quotient, region in _under_regions(expression, ctx, rows):
            if not isinstance(quotient, ms.Divide):
                continue
            params = parameters_of(quotient.divisor)
            if not params:
                continue
            needed = region
            for variable in sorted(ms.variables_of(quotient.numerator)):
                present = terms.present(ctx.model.variables[variable])
                needed = present if needed is None else needed & present
            for param in sorted(params):
                missing = gaps_under(ctx.parameters[param], needed)
                if missing:
                    raise SpecDataError(
                        f"{subject}: parameter '{param}' is used as a divisor but covers {missing} "
                        f"fewer coordinates than it is divided over. A missing row means a zero "
                        f"coefficient everywhere else, and zero is not a divisor: the term would drop "
                        f"and the row would silently stop constraining.\n"
                        f"  Supply the missing rows, or mask the coordinates out with a where."
                    )


def check_coefficients_cover(
    subject: str, expressions: tuple[ms.ExpressionNode, ...], ctx: Context, rows: Rows
) -> None:
    """
    A coefficient parameter must reach every row it is built over.

    A missing coefficient row would otherwise read as a zero, dropping its term
    while the row stays. Decided against the rows the declaration builds,
    narrowed at each ``cases:`` region exactly as the other checks are, so a
    ``where`` that removed the coordinate has already answered. A shift offset
    or window width given by name is a coefficient too, and stands or falls
    over its own coordinates.
    """
    for expression in expressions:
        for node, region in _under_regions(expression, ctx, rows):
            for param, needed in _coefficient_uses(node, region):
                missing = gaps_under(ctx.parameters[param], needed)
                if missing:
                    raise SpecDataError(
                        f"{subject}: parameter '{param}' is used as a coefficient but leaves "
                        f"{missing} of the rows built here uncovered. A missing row reads as a zero "
                        f"coefficient, dropping the term while the row stays.\n"
                        f"  Supply the missing rows, if a value other than 0 was meant.\n"
                        f"  Mask them out with a where, if the row should not exist there."
                    )


def _coefficient_uses(
    node: ms.ExpressionNode, region: Rows
) -> Iterator[tuple[str, Rows]]:
    """Each parameter *node* uses as a coefficient, with the rows it has to cover."""
    if isinstance(node, ms.Parameter):
        yield node.name, region
    elif isinstance(node, ms.Translate) and isinstance(node.offset, str):
        yield node.offset, None
    elif isinstance(node, ms.Window) and isinstance(node.width, str):
        yield node.width, None


def _under_regions(
    node: ms.ExpressionNode, ctx: Context, rows: Rows
) -> Iterator[tuple[ms.ExpressionNode, Rows]]:
    """Every node under *node* with the rows it has to cover, narrowed at each ``cases:`` region."""
    yield node, rows
    if isinstance(node, ms.Cases):
        for region in node.regions:
            inside = evaluate_where(region.when, ctx)
            yield from _under_regions(
                region.value, ctx, inside if rows is None else rows & inside
            )
        return
    for child in children(node):
        yield from _under_regions(child, ctx, rows)
