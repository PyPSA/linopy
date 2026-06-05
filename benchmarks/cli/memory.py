"""Memory subcommands: ``memory save`` / ``memory sweep`` / ``memory compare``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from benchmarks.cli._base import memory_app
from benchmarks.memory import compare as memory_compare
from benchmarks.memory import save as memory_save
from benchmarks.sweep import run_memory_sweep


@memory_app.command("save")
def memory_save_cmd(
    label: Annotated[
        str, typer.Argument(help="Label to attach to this snapshot, e.g. a git sha.")
    ],
    quick: Annotated[
        bool, typer.Option("--quick", help="Use smaller problem sizes.")
    ] = False,
    phase: Annotated[
        list[str] | None,
        typer.Option(
            "--phase",
            help=(
                "Restrict measurement to these phases. Pass multiple ``--phase`` "
                "to select more than one. Default: all (build, matrices, to_lp,"
                " to_netcdf, from_netcdf, to_solver)."
            ),
        ),
    ] = None,
    repeats: Annotated[
        int,
        typer.Option(
            "--repeats",
            help=(
                "Re-run each measurement N times and keep the min peak. Default "
                "1 (single shot). Memory peaks have ~1–3 %% wobble from GC "
                "timing, lazy-import priming, and netcdf page-cache effects — "
                "min-of-3 tightens that signal."
            ),
        ),
    ] = 1,
    filter_expr: Annotated[
        str | None,
        typer.Option(
            "--filter",
            "-k",
            help=(
                "Keep only specs whose name/id contains this — e.g. "
                "``nodal_balance`` (one spec), ``severity`` (patterns), ``n=`` "
                "(models)."
            ),
        ),
    ] = None,
) -> None:
    """
    Measure peak memory across the registry × phase grid via ``memray.Tracker``.

    Each ``(phase, spec, size)`` runs under its own tracker so setup
    allocations (model construction) are excluded from the peak — only the
    phase work itself is counted. Phases run in separate subprocesses for
    isolation.

    Results land in ``.benchmarks/memory/<label>.json``, keyed by full
    pytest-style test IDs so ``compare`` diffs cleanly across runs that
    selected different subsets.
    """
    from benchmarks.memory import ALL_MEMORY_PHASES

    if phase:
        unknown = [p for p in phase if p not in ALL_MEMORY_PHASES]
        if unknown:
            typer.secho(
                f"unknown phase(s): {unknown}; valid options: {list(ALL_MEMORY_PHASES)}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
    memory_save(
        label, quick=quick, phases=phase, repeats=repeats, filter_expr=filter_expr
    )


@memory_app.command("sweep")
def memory_sweep_cmd(
    versions: Annotated[
        list[str],
        typer.Argument(help="linopy versions, e.g. 0.4.0 0.5.0 (or any pip spec)."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            "-o",
            help="Where to save snapshot JSONs.",
        ),
    ] = Path(".benchmarks/memory"),
    quick: Annotated[
        bool,
        typer.Option("--quick", help="Use only the smallest sizes (faster sweep)."),
    ] = False,
    phase: Annotated[
        list[str] | None,
        typer.Option(
            "--phase",
            help=(
                "Restrict each version's run to these phases. Pass multiple "
                "``--phase`` to select more than one."
            ),
        ),
    ] = None,
    repeats: Annotated[
        int,
        typer.Option(
            "--repeats",
            help="min-of-N peak per measurement (default 1).",
        ),
    ] = 1,
    as_of: Annotated[
        str | None,
        typer.Option(
            "--as-of",
            help=(
                "Freeze every dep's resolution to releases on or before this "
                "date (``YYYY-MM-DD`` or ISO 8601). Same semantics as "
                "``sweep --as-of`` — see that command's help."
            ),
        ),
    ] = None,
) -> None:
    """
    Sweep peak-memory measurements across several linopy versions.

    Mirrors the timing :func:`sweep` but invokes ``memory save`` inside
    each per-version uv venv. Each version's snapshot lands at
    ``<output-dir>/linopy-<version>.json`` and is auto-detected by
    ``plot`` (the ``peak_mib`` key distinguishes memory from timing).

    Memory peaks are much more deterministic than wall time, so
    ``--repeats 1`` (default) is usually plenty. Use ``--repeats 3``
    if you need <5%% regression detection.
    """
    run_memory_sweep(
        versions,
        output_dir=output_dir,
        quick=quick,
        phases=phase,
        repeats=repeats,
        as_of=as_of,
    )


@memory_app.command("compare")
def memory_compare_cmd(
    label_a: Annotated[str, typer.Argument(help="Baseline label (typically master).")],
    label_b: Annotated[str, typer.Argument(help="Candidate label (your branch).")],
) -> None:
    """
    Compare two saved memory snapshots side-by-side.

    Prints a per-test table of label_a vs label_b peak RSS and a percent
    change. Tests present in only one snapshot are shown with ``—`` for
    the missing column.
    """
    memory_compare(label_a, label_b)
