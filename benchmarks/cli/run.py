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
    long: Annotated[
        bool,
        typer.Option(
            "--long",
            help="Include the slowest sizes (above each spec's long_threshold).",
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
                "pytest ``-k`` expression selecting specs by name/id — e.g. "
                "``basic`` (one spec), ``severity`` (patterns), "
                "``'build and basic'``."
            ),
        ),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option("--json", help="Save pytest-benchmark JSON to this path."),
    ] = None,
    rounds: Annotated[
        int | None,
        typer.Option(
            "--rounds",
            help=(
                "Force pytest-benchmark to run exactly N rounds per test "
                "(passes ``--benchmark-min-rounds=N --benchmark-max-time=0``). "
                "Default: pytest-benchmark auto-tunes per test (5–40+ rounds "
                "depending on cost). Use a fixed N for uniform measurement "
                "across versions in a sweep."
            ),
        ),
    ] = None,
) -> None:
    """
    Default timing run. Records timings with pytest-benchmark.

    Without ``--long``, sizes above each spec's ``long_threshold`` are
    skipped — keeps the wall-clock around 45s instead of several minutes.
    Add ``--long`` for the full sweep including the heaviest sizes
    (knapsack at 1M, basic at 1600, pypsa_scigrid at >50).

    Any trailing arguments are forwarded to pytest verbatim, e.g.::

        python -m benchmarks run --long -- --tb=short -x

    To skip timing entirely (e.g. just verifying everything runs at a
    bigger size), use ``smoke`` instead, or pass ``--benchmark-disable``
    as a trailing arg.
    """
    args: list[str] = []
    args.append(_PHASE_TEST_FILE[phase] if phase is not None else "benchmarks/")
    if long:
        args.append("--long")
    args.append("--benchmark-only")
    if json_out is not None:
        args.extend(["--benchmark-json", str(json_out)])
    if rounds is not None:
        args.extend([f"--benchmark-min-rounds={rounds}", "--benchmark-max-time=0"])

    if filter_expr:
        args.extend(["-k", filter_expr])

    args.extend(ctx.args)
    _run_pytest(args)


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
