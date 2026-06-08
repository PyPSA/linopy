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

import inspect
import tempfile
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from functools import partial
from pathlib import Path
from typing import NamedTuple

import linopy
import linopy.io as lio
from benchmarks.registry import (
    BUILD,
    FROM_NETCDF,
    MATRICES,
    TO_GUROBIPY,
    TO_HIGHSPY,
    TO_LP,
    TO_MOSEK,
    TO_NETCDF,
    TO_XPRESS,
    BenchSpec,
    iter_params,
    spec_param_id,
)
from linopy import read_netcdf
from linopy.solvers import available_solvers

# linopy <0.4.1's ``to_file`` doesn't accept ``progress``. Check once
# at import so the benchmark loop stays branchless on the hot path.
_TO_FILE_HAS_PROGRESS = "progress" in inspect.signature(linopy.Model.to_file).parameters

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

    Where supported, ``progress=False`` is pinned here so the
    benchmark stays uniform across drivers — the progress bar's
    overhead would otherwise leak into the measurement. linopy <0.4.1
    doesn't accept the kwarg; falls back to the native call.
    """
    if _TO_FILE_HAS_PROGRESS:
        m.to_file(path, progress=False)
    else:
        m.to_file(path)


def write_netcdf(m: linopy.Model, path: Path) -> None:
    m.to_netcdf(path)


# (solver_name, registry phase tag, wrapper) — consumed by the pytest
# parametrization in ``test_to_solver.py`` and by ``memory.py``,
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


Action = Callable[[], object]
CaseFactory = Callable[[], AbstractContextManager[Action]]

PIPELINE = "pipeline"

PHASE_NODE: dict[str, str] = {
    BUILD: "benchmarks/test_build.py::test_build",
    MATRICES: "benchmarks/test_matrices.py::test_matrices",
    TO_LP: "benchmarks/test_to_lp.py::test_to_lp",
    TO_NETCDF: "benchmarks/test_netcdf.py::test_to_netcdf",
    FROM_NETCDF: "benchmarks/test_netcdf.py::test_from_netcdf",
    "to_solver": "benchmarks/test_to_solver.py::test_to_solver",
    PIPELINE: "benchmarks/test_pipeline.py::test_pipeline",
}


class PhaseCase(NamedTuple):
    """One parametrization of a phase — what both drivers consume."""

    spec: BenchSpec
    value: int
    id: str
    run: CaseFactory
    skip: str | None


@contextmanager
def _build_case(spec: BenchSpec, value: int) -> Iterator[Action]:
    yield lambda: spec.build(value)


@contextmanager
def _matrices_case(spec: BenchSpec, value: int) -> Iterator[Action]:
    m = spec.build(value)
    yield lambda: touch_matrices(m)


@contextmanager
def _to_lp_case(spec: BenchSpec, value: int) -> Iterator[Action]:
    m = spec.build(value)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.lp"
        yield lambda: write_lp(m, path)


@contextmanager
def _to_netcdf_case(spec: BenchSpec, value: int) -> Iterator[Action]:
    m = spec.build(value)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.nc"
        yield lambda: write_netcdf(m, path)


@contextmanager
def _from_netcdf_case(spec: BenchSpec, value: int) -> Iterator[Action]:
    m = spec.build(value)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.nc"
        write_netcdf(m, path)
        yield lambda: read_netcdf(path)


@contextmanager
def _solver_case(
    spec: BenchSpec, value: int, wrapper: Callable[[linopy.Model], object]
) -> Iterator[Action]:
    m = spec.build(value)
    yield lambda: wrapper(m)


@contextmanager
def _pipeline_case(spec: BenchSpec, value: int) -> Iterator[Action]:
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.lp"

        def action() -> None:
            m = spec.build(value)
            touch_matrices(m)
            write_lp(m, path)

        yield action


_PHASE_CASE: dict[str, tuple[str, Callable[[BenchSpec, int], AbstractContextManager[Action]]]] = {
    BUILD: (BUILD, _build_case),
    MATRICES: (MATRICES, _matrices_case),
    TO_LP: (TO_LP, _to_lp_case),
    TO_NETCDF: (TO_NETCDF, _to_netcdf_case),
    FROM_NETCDF: (FROM_NETCDF, _from_netcdf_case),
    PIPELINE: (TO_LP, _pipeline_case),
}


def phase_cases(phase: str) -> Iterator[PhaseCase]:
    """
    Yield every ``(spec, value)`` parametrization of one phase as a runnable
    case — the single source of truth for "what runs + its id", shared by the
    pytest drivers and the memray engine.

    ``to_solver`` expands to one case per available solver (the solver in the
    id-suffix); every other phase yields one case per applicable ``(spec,
    value)``. ``skip`` is set for solvers that aren't installed.
    """
    if phase == "to_solver":
        for name, tag, wrapper in SOLVER_HANDOFFS:
            skip = None if name in available_solvers else f"{name} not installed"
            for spec, value in iter_params(tag):
                sfx = f"{name}-{spec_param_id(spec.name, spec.axis, value)}"
                run = partial(_solver_case, spec, value, wrapper)
                yield PhaseCase(spec, value, sfx, run, skip)
        return

    tag, case = _PHASE_CASE[phase]
    for spec, value in iter_params(tag):
        sfx = spec_param_id(spec.name, spec.axis, value)
        yield PhaseCase(spec, value, sfx, partial(case, spec, value), None)
