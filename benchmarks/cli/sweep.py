"""Cross-version sweep command: ``sweep``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from benchmarks.cli._base import (
    _PHASE_TEST_FILE,
    _SMOKE_PYTEST_ARGS,
    Measure,
    PhaseName,
    app,
)
from benchmarks.sweep import run_memory_sweep, run_sweep


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def sweep(
    ctx: typer.Context,
    versions: Annotated[
        list[str],
        typer.Argument(help="linopy versions, e.g. 0.4.0 0.5.0 (or any pip spec)."),
    ],
    metric: Annotated[
        Measure,
        typer.Option(
            "--metric",
            help=(
                "What to measure: ``time`` (pytest-benchmark wall clock), "
                "``memory`` (peak RSS via memray), or ``both`` (sequential). "
                "Default: time."
            ),
        ),
    ] = Measure.time,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "-o",
            help=(
                "Where to save snapshot JSONs. Default: ``.benchmarks/<metric>/`` "
                "(``both`` writes ``time`` and ``memory`` subdirs)."
            ),
        ),
    ] = None,
    repeats: Annotated[
        int,
        typer.Option(
            "--repeats",
            help="Memory only: min-of-N peak per measurement (default 1).",
        ),
    ] = 1,
    long: Annotated[
        bool, typer.Option("--long", help="Include the slowest sizes.")
    ] = False,
    quick: Annotated[
        bool,
        typer.Option("--quick", help="Use only the smallest sizes (faster sweep)."),
    ] = False,
    phase: Annotated[
        PhaseName | None,
        typer.Option(help="Restrict each version's run to one phase's test file."),
    ] = None,
    filter_expr: Annotated[
        str | None,
        typer.Option(
            "--filter",
            "-k",
            help=(
                "pytest ``-k`` expression selecting specs by name/id — e.g. "
                "``basic`` (one spec), ``severity`` (patterns)."
            ),
        ),
    ] = None,
    rounds: Annotated[
        int | None,
        typer.Option(
            "--rounds",
            help=(
                "Force pytest-benchmark to run exactly N rounds per test in "
                "every version (uniform measurement across the sweep). "
                "Default: pytest-benchmark auto-tunes per test."
            ),
        ),
    ] = None,
    smoke: Annotated[
        bool,
        typer.Option(
            "--smoke",
            help=(
                "Run the smoke suite in each version's venv instead of the "
                "full timing run. Same pytest invocation as the top-level "
                "``smoke`` command — every model/phase fires once at the "
                "quickest size, no timings, ~20 s per version. Useful before "
                "bumping a perf-sensitive pin to check the combination is "
                "viable across every linopy version you'd sweep against."
            ),
        ),
    ] = False,
    as_of: Annotated[
        str | None,
        typer.Option(
            "--as-of",
            help=(
                "Freeze every dep's resolution to releases on or before this "
                "date (``YYYY-MM-DD`` or ISO 8601). Passes ``--exclude-newer`` "
                "to uv. Use a consistent value across invocations for "
                "cross-time-reproducible sweeps — direct pins alone keep "
                "results stable within one call but transitives can drift "
                "between calls."
            ),
        ),
    ] = None,
) -> None:
    """
    Run the benchmark suite against several linopy versions.

    Uses ``uv`` to build a fresh venv per version (near-instant) and to
    install the benchmark infra + target linopy in a single resolution
    pass. The pytest-benchmark JSON snapshot lands in
    ``<output-dir>/linopy-<version>.json``.

    Versions are accepted in two forms:

    - Plain releases: ``0.4.0``, ``0.5.0a1`` — expanded to ``linopy==X``.
    - Pip specs verbatim: ``git+https://github.com/PyPSA/linopy.git@<sha>``
      or ``linopy @ file:///path/to/checkout``.

    The current (repo-tip) benchmark code runs against each linopy
    version, so the measurement layer is constant. ``_API_AVAILABLE``
    gates in the ``sos`` / ``piecewise`` specs let older linopy versions
    skip those phases gracefully.

    Filter knobs (``--phase``, ``--model``, ``--filter``) mirror ``run``
    and apply to every version's pytest invocation. Trailing arguments
    after ``--`` are forwarded to pytest verbatim:

        python -m benchmarks sweep 0.6.7 --phase build --model basic
        python -m benchmarks sweep 0.6.7 -- --tb=short -x

    Wall-clock: roughly 1-2 minutes per version (venv + install +
    benchmarks). uv's wheel cache makes repeated runs much faster.
    """
    # Timing-only knobs can't apply to a memory run.
    if metric is not Measure.time and (smoke or rounds is not None):
        typer.secho(
            "--smoke / --rounds are timing-only (use --metric time)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    def _timing(out: Path) -> None:
        test_target = _PHASE_TEST_FILE[phase] if phase is not None else "benchmarks/"
        run_sweep(
            versions,
            output_dir=out,
            test_target=test_target,
            smoke_args=_SMOKE_PYTEST_ARGS,
            long=long,
            quick=quick,
            rounds=rounds,
            filter_expr=filter_expr,
            smoke=smoke,
            as_of=as_of,
            extra_args=ctx.args,
        )

    def _memory(out: Path) -> None:
        run_memory_sweep(
            versions,
            output_dir=out,
            quick=quick,
            long=long,
            phases=[phase] if phase is not None else None,
            repeats=repeats,
            as_of=as_of,
        )

    # ``both`` runs sequentially into per-metric subdirs so the two
    # ``linopy-<version>.json`` snapshot sets never collide.
    if metric is Measure.both:
        base = output_dir
        _timing(base / "time" if base else Path(".benchmarks/time"))
        _memory(base / "memory" if base else Path(".benchmarks/memory"))
    elif metric is Measure.memory:
        _memory(output_dir or Path(".benchmarks/memory"))
    else:
        _timing(output_dir or Path(".benchmarks/time"))
