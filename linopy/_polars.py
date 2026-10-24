#!/usr/bin/env python3
"""
Placeholder used in place of :mod:`polars` when it is not installed.

Polars backs the ``to_polars()`` export methods and the LP file writer, which
the default ``Model.solve()`` uses. It is an optional dependency, so every
module that touches it imports it as::

    try:
        import polars as pl
    except ImportError:
        from linopy._polars import missing_polars as pl

Attribute access on the placeholder raises :class:`ImportError` with an install
hint, so the failure surfaces at the first real use rather than at import time.
"""

from typing import Any

_INSTALL_HINT = (
    "polars is required for this feature. Install it with `pip install linopy[polars]`."
)


class _MissingPolars:
    """Stand-in for the ``polars`` module; every attribute access raises."""

    def __getattr__(self, name: str) -> Any:
        # Tools probe dunders such as ``__wrapped__`` and expect AttributeError.
        # Raising ImportError there breaks pytest's doctest collection.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise ImportError(_INSTALL_HINT)


missing_polars = _MissingPolars()
