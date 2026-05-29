"""
Single source of truth for *what each benchmark phase does to a model*.

Both drivers import these verbs:

- the pytest ``test_<phase>.py`` files wrap them in ``benchmark(...)``;
- ``memory.py`` wraps them in ``memray.Tracker(...)``.

So the measured operation is defined once. Setup — building the model,
creating scratch files — stays in the caller; only the verb itself
lives here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import linopy
import linopy.io as lio
from benchmarks.registry import TO_GUROBIPY, TO_HIGHSPY, TO_MOSEK, TO_XPRESS
from linopy import read_netcdf

# Re-export so callers can ``from benchmarks.phases import read_netcdf``
# alongside the wrappers.
__all__ = [
    "SOLVER_HANDOFFS",
    "read_netcdf",
    "touch_matrices",
    "write_lp",
    "write_netcdf",
]


def touch_matrices(m: linopy.Model) -> None:
    """Force every matrix block to materialise — the thing we measure."""
    mats = m.matrices
    for attr in ("A", "b", "c", "lb", "ub", "sense", "vlabels", "clabels"):
        getattr(mats, attr)
    if m.is_quadratic:
        mats.Q


def write_lp(m: linopy.Model, path: Path) -> None:
    """
    Write the model as an LP file.

    ``progress=False`` is pinned here so the benchmark stays uniform
    across drivers — the progress bar's overhead would otherwise leak
    into the measurement.
    """
    m.to_file(path, progress=False)


def write_netcdf(m: linopy.Model, path: Path) -> None:
    m.to_netcdf(path)


# (solver_name, registry phase tag, wrapper) — consumed by the pytest
# parametrization in ``test_solver_handoff.py`` and by ``memory.py``,
# which looks up the "highs" entry. Adding a solver here automatically
# extends both drivers.
#
# Each wrapper is fetched via ``getattr`` so the tuple silently drops
# any solver wrapper missing from the installed ``linopy`` — necessary
# for cross-version ``sweep`` runs against older releases (e.g.
# ``to_xpress`` doesn't exist before linopy 0.7.1).
SOLVER_HANDOFFS: tuple[tuple[str, str, Callable[[linopy.Model], object]], ...] = tuple(
    (name, tag, wrapper)
    for name, tag, wrapper in (
        ("highs", TO_HIGHSPY, getattr(lio, "to_highspy", None)),
        ("gurobi", TO_GUROBIPY, getattr(lio, "to_gurobipy", None)),
        ("mosek", TO_MOSEK, getattr(lio, "to_mosek", None)),
        ("xpress", TO_XPRESS, getattr(lio, "to_xpress", None)),
    )
    if wrapper is not None
)
