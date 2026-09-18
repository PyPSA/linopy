"""Errors raised while attaching data to a math-spec program, and how they spell what they name."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from typing import Any

import xarray as xr
from math_spec import did_you_mean


class SpecDataError(ValueError):
    """
    Data attached to a valid spec is missing, malformed or the wrong shape.

    Every refusal names the symbol, the dimension(s) and the offending labels,
    so the message points back at the ``sources`` entry to fix.
    """


def unknown(kind: str, name: str, known: Iterable[str]) -> KeyError:
    """The error for a *name* no *kind* of the spec carries, with the nearest declared one."""
    return KeyError(f"unknown {kind} '{name}'. " + did_you_mean(name, known))


def shown(labels: Sequence[Any], limit: int = 5) -> str:
    """The first *limit* labels, and how many more there are."""
    head = ", ".join(repr(x) for x in labels[:limit])
    return head + (f" (and {len(labels) - limit} more)" if len(labels) > limit else "")


def coordinate(dims: Sequence[str], key: Hashable) -> str:
    """One coordinate as ``dim=label`` pairs."""
    row = key if isinstance(key, tuple) else (key,)
    return ", ".join(f"{d}={v!r}" for d, v in zip(dims, row))


def coordinates_shown(dims: Sequence[str], rows: Iterable[Hashable]) -> str:
    """Several coordinates, one after another."""
    return "; ".join(coordinate(dims, row) for row in rows)


def first_coordinates(flags: xr.DataArray, n: int) -> str:
    """The first *n* coordinates *flags* is true at, spelled the way every refusal spells them."""
    dims = tuple(str(d) for d in flags.dims)
    if not dims:
        return "the only row"
    stacked = flags.stack(_at=dims)
    return coordinates_shown(dims, stacked.indexes["_at"][stacked.to_numpy()][:n])
