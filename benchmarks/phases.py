"""
The measured operations — what each benchmark phase *does to a model*.

The ``test_<phase>.py`` drivers wrap these verbs in ``benchmark(...)``; setup
(building the model, scratch files) stays in the driver, only the verb itself
lives here.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

import linopy
import linopy.io as lio
from benchmarks.registry import TO_GUROBIPY, TO_HIGHSPY, TO_MOSEK, TO_XPRESS
from linopy import read_netcdf

# linopy <0.4.1's ``to_file`` doesn't accept ``progress``. Checked once at import
# so the suite stays runnable against older linopy (e.g. cross-version sweeps),
# and the benchmark loop stays branchless.
_TO_FILE_HAS_PROGRESS = "progress" in inspect.signature(linopy.Model.to_file).parameters

# Re-export so a driver can ``from benchmarks.phases import read_netcdf``.
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

    Where supported, ``progress=False`` is pinned so the progress bar's overhead
    doesn't leak into the measurement; linopy <0.4.1 doesn't accept the kwarg.
    """
    if _TO_FILE_HAS_PROGRESS:
        m.to_file(path, progress=False)
    else:
        m.to_file(path)


def write_netcdf(m: linopy.Model, path: Path) -> None:
    m.to_netcdf(path)


# (solver_name, registry phase tag, wrapper) — consumed by test_to_solver.py.
# Each wrapper is fetched via ``getattr`` so the tuple silently drops any wrapper
# missing from the installed linopy (e.g. ``to_xpress`` is absent before linopy
# 0.7.1) — keeping the suite runnable on older releases for cross-version sweeps.
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
