"""
What the data has to satisfy before anything is solved on it.

The language decides a program's shape and can decide nothing about its
numbers, so every fact it needs of them is stated as an ``assumptions:``
entry: the file's own, and the ones a ``piecewise:`` method implies, which
the expansion writes beside them. Each is a predicate over the bound data
and is checked here, in the words :func:`~math_spec.program.assumption_message`
gives every consumer.
"""

from __future__ import annotations

from math_spec import program as ms

from linopy.spec.context import Context
from linopy.spec.errors import SpecDataError, first_coordinates
from linopy.spec.where import evaluate_where


def check_assumptions(ctx: Context) -> None:
    """
    Refuse the data wherever an assumption of the program does not hold.

    An assumption holds at every coordinate its ``where`` admits; a hole on
    either side of a comparison reads as false there, as it does in any mask.

    Raises
    ------
    SpecDataError
        The first assumption, in the program's order, the data fails, with
        the first coordinates it fails at.
    """
    for name, assumption in ctx.program.assumptions.items():
        admitted = evaluate_where(assumption.where, ctx)
        failing = admitted & ~evaluate_where(assumption.predicate, ctx)
        if bool(failing.any()):
            raise SpecDataError(
                f"{ms.assumption_message(name, assumption)}\n"
                f"  Not so at {first_coordinates(failing, 3)}"
            )
