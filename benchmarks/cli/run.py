"""Execution commands: ``smoke``, ``run``, ``notebook``."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
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


def _run_pytest(args: list[str]) -> None:
    """Invoke pytest as a subprocess and propagate its exit code."""
    cmd = [sys.executable, "-m", "pytest", *args]
    typer.secho(f"$ {' '.join(cmd)}", fg=typer.colors.BRIGHT_BLACK)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def smoke(ctx: typer.Context) -> None:
    """
    Quick smoke run — what CI uses on every PR.

    Equivalent to ``pytest benchmarks/ --quick --benchmark-disable -q``.
    Every model builds at one size and every phase fires once, no timings
    recorded. Typical wall-clock: ~20s.

    Any trailing arguments are forwarded to pytest verbatim, e.g.::

        python -m benchmarks smoke -k basic --tb=short
    """
    _run_pytest([*_SMOKE_PYTEST_ARGS, *ctx.args])


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run(
    ctx: typer.Context,
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
    quick: Annotated[
        bool,
        typer.Option("--quick", help="Use each spec's quick subset of sizes."),
    ] = False,
    long: Annotated[
        bool,
        typer.Option(
            "--long",
            help="Include the slowest sizes (each spec's long_sizes).",
        ),
    ] = False,
    phase: Annotated[
        PhaseName | None,
        typer.Option(help="Restrict to one phase's test file."),
    ] = None,
    filter_expr: Annotated[
        str | None,
        typer.Option(
            "--filter",
            "-k",
            help=(
                "Select specs by name/id — a pytest ``-k`` expression for time, "
                "a substring for memory. E.g. ``basic``, ``severity``."
            ),
        ),
    ] = None,
    size: Annotated[
        list[int] | None,
        typer.Option("--size", help="Run only these model sizes (repeatable)."),
    ] = None,
    severity: Annotated[
        list[int] | None,
        typer.Option(
            "--severity", help="Run only these pattern severities (repeatable)."
        ),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option(
            "--json",
            help=(
                "Save the snapshot to this path (pytest-benchmark JSON for time, "
                "peak-RSS JSON for memory). Without it, results are only printed."
            ),
        ),
    ] = None,
    rounds: Annotated[
        int | None,
        typer.Option(
            "--rounds",
            help=(
                "Time only: force pytest-benchmark to run exactly N rounds per "
                "test (``--benchmark-min-rounds=N --benchmark-max-time=0``). "
                "Default: auto-tuned per test."
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
) -> None:
    """
    Single-environment benchmark run — time, memory, or both.

    ``--metric time`` (default) records wall-clock with pytest-benchmark;
    ``--metric memory`` tracks peak RSS via memray; ``--metric both`` runs
    them sequentially. Results print to the terminal; pass ``--json PATH``
    to also save a snapshot (one rule for both metrics).

    Without ``--quick``/``--long``, each spec's ``long_sizes`` (the heaviest)
    are skipped — keeps the wall-clock manageable. ``--size``/``--severity``
    pin exact values on either axis.

    Trailing arguments are forwarded to pytest (time only), e.g.::

        python -m benchmarks run --long -- --tb=short -x
        python -m benchmarks run --metric memory --json mem.json -k basic
    """
    sizes = tuple(size or ())
    severities = tuple(severity or ())

    if metric is not Measure.time and rounds is not None:
        typer.secho("--rounds is timing-only", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    if metric is Measure.both and json_out is not None:
        typer.secho(
            "--json can't be used with --metric both (formats would collide); "
            "run each metric separately to save",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    def _timing() -> None:
        args: list[str] = []
        args.append(_PHASE_TEST_FILE[phase] if phase is not None else "benchmarks/")
        if phase == "pipeline":
            args.append("--pipeline")
        if quick:
            args.append("--quick")
        elif long:
            args.append("--long")
        for s in sizes:
            args.extend(["--size", str(s)])
        for s in severities:
            args.extend(["--severity", str(s)])
        args.append("--benchmark-only")
        if json_out is not None:
            args.extend(["--benchmark-json", str(json_out)])
        if rounds is not None:
            args.extend([f"--benchmark-min-rounds={rounds}", "--benchmark-max-time=0"])
        if filter_expr:
            args.extend(["-k", filter_expr])
        args.extend(ctx.args)
        _run_pytest(args)

    def _memory() -> None:
        from benchmarks import memory as mem
        from benchmarks.snapshot import write_memory_snapshot

        results = mem.measure(
            quick=quick,
            phases=[phase] if phase is not None else None,
            repeats=repeats,
            filter_expr=filter_expr,
            long=long,
            sizes=sizes,
            severities=severities,
        )
        if not results:
            typer.secho("no measurements produced", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        if json_out is not None:
            write_memory_snapshot(json_out, json_out.stem, results)
            typer.secho(
                f"saved {len(results)} measurements to {json_out}",
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho(
                f"{len(results)} measurements (pass --json to save)",
                fg=typer.colors.GREEN,
            )

    if metric in (Measure.time, Measure.both):
        _timing()
    if metric in (Measure.memory, Measure.both):
        _memory()


@app.command()
def notebook(
    build: Annotated[
        bool,
        typer.Option(
            "--build",
            help=(
                "Regenerate ``walkthrough.ipynb`` from the ``.md`` source. "
                "One-way build — the ``.ipynb`` is a throwaway artifact for "
                "opening in any editor (JupyterLab, PyCharm, VSCode), the "
                "``.md`` stays canonical. Re-run after editing the ``.md``. "
                "The ``.ipynb`` is gitignored."
            ),
        ),
    ] = False,
) -> None:
    """
    Execute the walkthrough notebook end-to-end (default) or rebuild the
    ``.ipynb`` artifact for interactive viewing (``--build``).

    The walkthrough is a Jupytext MyST markdown file
    (``benchmarks/walkthrough.md``) — diffs cleanly in git, runs as a
    notebook in Jupyter. The ``.md`` is the source of truth; the paired
    ``.ipynb`` is generated output. Edit the ``.md``, re-run ``--build``,
    open the ``.ipynb`` in your editor of choice.

    CI calls this with no flags to catch doc rot; the executed copy goes
    to a tempdir and is discarded so the source file stays output-free.
    """
    nb = Path("benchmarks/walkthrough.md")
    if not nb.exists():
        typer.secho(f"walkthrough not found: {nb}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if build:
        # ``--to ipynb`` is a one-way conversion (no ``formats`` metadata
        # written into the .md). The generated .ipynb is editor-agnostic;
        # contributors regenerate it after editing the .md.
        cmd = [
            sys.executable,
            "-m",
            "jupytext",
            "--to",
            "ipynb",
            str(nb),
        ]
        typer.secho(f"$ {' '.join(cmd)}", fg=typer.colors.BRIGHT_BLACK)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise typer.Exit(code=result.returncode)
        ipynb = nb.with_suffix(".ipynb")
        typer.secho(f"built: {ipynb}  (regenerable from {nb})", fg=typer.colors.GREEN)
        typer.echo(f"Open it:  jupyter lab {ipynb}    # or PyCharm / VSCode / …")
        return

    with tempfile.TemporaryDirectory() as tmp:
        # Jupytext sets the kernel cwd to the output directory (the tempdir
        # here), so forward the repo root via ``LINOPY_REPO_ROOT`` for the
        # walkthrough's first cell to find ``benchmarks/``.
        env = {**os.environ, "LINOPY_REPO_ROOT": str(Path.cwd().resolve())}
        cmd = [
            sys.executable,
            "-m",
            "jupytext",
            "--to",
            "notebook",
            "--execute",
            "--output",
            str(Path(tmp) / "executed.ipynb"),
            str(nb),
        ]
        typer.secho(f"$ {' '.join(cmd)}", fg=typer.colors.BRIGHT_BLACK)
        result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)
