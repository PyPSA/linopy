"""
Shared app object, types, and helpers for the benchmark CLI.

The command groups (``introspect``, ``run``, ``sweep``, ``compare``,
``plot``) all register onto the single ``app`` defined here, so the
user-facing command surface stays flat (``python -m benchmarks run`` etc.).
Time vs memory is a ``--metric`` flag on ``run``/``sweep``, not a sub-app.

Note on colour: ``typer.secho`` strips colour automatically when stdout isn't
a TTY, so piping any command into ``grep`` still yields plain text.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

import typer

from benchmarks.snapshot import discover_snapshots


class Measure(StrEnum):
    """
    What a measuring command records — orthogonal to the workflow.

    ``time`` runs pytest-benchmark (wall clock); ``memory`` tracks peak RSS
    via memray; ``both`` runs them sequentially (never concurrently — memray's
    overhead would skew the wall-clock numbers).
    """

    time = "time"
    memory = "memory"
    both = "both"


app = typer.Typer(
    help=(
        "Linopy internal benchmark suite — a thin layer over pytest plus "
        "registry introspection and memory snapshots."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)

PhaseName = Literal[
    "build", "matrices", "to_lp", "to_netcdf", "from_netcdf", "to_solver"
]
SpecKind = Literal["all", "models", "patterns"]


_PHASE_TEST_FILE: dict[PhaseName, str] = {
    "build": "benchmarks/test_build.py",
    "matrices": "benchmarks/test_matrices.py",
    "to_lp": "benchmarks/test_to_lp.py",
    "to_netcdf": "benchmarks/test_netcdf.py::test_to_netcdf",
    "from_netcdf": "benchmarks/test_netcdf.py::test_from_netcdf",
    "to_solver": "benchmarks/test_to_solver.py",
}

# pytest args that constitute a "smoke" run — quick sizes, no timings.
# Shared between the top-level ``smoke`` command and ``sweep --smoke`` so
# bumping the definition stays single-source.
_SMOKE_PYTEST_ARGS = ["benchmarks/", "--quick", "--benchmark-disable", "-q"]


def _suggest_snapshots(reason: str) -> None:
    """Print an error + a hint listing whatever snapshots we can find."""
    typer.secho(reason, fg=typer.colors.RED, err=True)
    found = discover_snapshots()
    if found:
        typer.echo("\nAvailable snapshots under .benchmarks/:", err=True)
        for p in found:
            typer.echo(f"  {p}", err=True)
    else:
        typer.echo(
            "\nNo snapshots found under .benchmarks/. Generate one with:\n"
            "  python -m benchmarks run --json .benchmarks/<label>.json",
            err=True,
        )
