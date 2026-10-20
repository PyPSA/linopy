"""Errors raised by the rolling-horizon primitives."""

from __future__ import annotations


class RestrictionError(ValueError):
    """
    Raised when a model cannot be windowed.

    A window requires a dimension that exists in the model, carriers that
    contain every selected label, and folded boundary terms whose constant rows
    are satisfiable.
    """
