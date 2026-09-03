"""Errors raised while binding data to a math-spec program."""

from __future__ import annotations


class SpecDataError(ValueError):
    """
    Data bound to a valid spec is missing, malformed or the wrong shape.

    Every refusal names the symbol, the dimension(s) and the offending labels,
    so the message points back at the ``sources`` entry to fix.
    """
