"""
linopy benchmark CLI — one entry point for the suite.

Run with::

    python -m benchmarks <command> [options]

The CLI is a thin layer over pytest for the timing / smoke commands, plus
direct dispatch for registry introspection and memory snapshots.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import typer

from benchmarks import (
    REGISTRY,
    filter_by,
    get,
)
from benchmarks.memory import compare as memory_compare
from benchmarks.memory import save as memory_save

app = typer.Typer(
    help=(
        "Linopy internal benchmark suite — a thin layer over pytest plus "
        "registry introspection and memory snapshots."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)

memory_app = typer.Typer(
    help="Peak-RSS memory snapshots (pytest-memray under the hood).",
    no_args_is_help=True,
)
app.add_typer(memory_app, name="memory")


PhaseName = Literal["build", "matrices", "lp_write", "netcdf", "solver_handoff"]

_PHASE_TEST_FILE: dict[PhaseName, str] = {
    "build": "benchmarks/test_build.py",
    "matrices": "benchmarks/test_matrices.py",
    "lp_write": "benchmarks/test_lp_write.py",
    "netcdf": "benchmarks/test_netcdf.py",
    "solver_handoff": "benchmarks/test_solver_handoff.py",
}


# --- Introspection commands ------------------------------------------------


@app.command("list")
def list_(
    details: Annotated[
        bool,
        typer.Option("--details", "-d", help="Show features and size range."),
    ] = False,
) -> None:
    """
    List the registered model specs.

    By default emits one name per line — suitable for piping into other
    tools. Pass ``--details`` for a small table that also shows the
    features tags and the size range.
    """
    if not details:
        for name in sorted(REGISTRY):
            typer.echo(name)
        return

    rows = [
        (
            spec.name,
            ",".join(sorted(spec.features)),
            f"{spec.sizes[0]}..{spec.sizes[-1]}",
        )
        for spec in REGISTRY.values()
    ]
    name_w = max(len(r[0]) for r in rows)
    feat_w = max(len(r[1]) for r in rows)
    typer.echo(f"{'name':<{name_w}}  {'features':<{feat_w}}  sizes")
    typer.echo("-" * (name_w + feat_w + 20))
    for name, feats, sizes in rows:
        typer.echo(f"{name:<{name_w}}  {feats:<{feat_w}}  {sizes}")


@app.command()
def show(
    name: Annotated[str, typer.Argument(help="Spec name (see ``list``).")],
) -> None:
    """
    Print full attributes of one model spec.

    Output includes sizes, feature tags, applicable phases, the quick /
    long size thresholds, and any optional ``requires=`` dependencies the
    spec advertises.
    """
    try:
        spec = get(name)
    except KeyError as exc:
        typer.secho(f"unknown model: {name!r}", fg=typer.colors.RED, err=True)
        typer.echo(f"available: {', '.join(sorted(REGISTRY))}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(repr(spec))
    typer.echo(f"  sizes:           {spec.sizes}")
    typer.echo(f"  features:        {sorted(spec.features)}")
    typer.echo(f"  phases:          {sorted(spec.phases)}")
    typer.echo(f"  quick_threshold: {spec.quick_threshold}")
    typer.echo(f"  long_threshold:  {spec.long_threshold}")
    if spec.requires:
        typer.echo(f"  requires:        {list(spec.requires)}")


@app.command("filter")
def filter_(
    feature: Annotated[
        str | None,
        typer.Option(help="Feature tag, e.g. 'quadratic', 'integer', 'sos'."),
    ] = None,
    phase: Annotated[
        str | None,
        typer.Option(help="Phase tag, e.g. 'to_gurobipy', 'lp_write'."),
    ] = None,
) -> None:
    """
    Filter specs by feature or phase tag.

    Both filters can be combined; the result is the intersection.
    At least one of ``--feature`` / ``--phase`` must be supplied.
    """
    if feature is None and phase is None:
        typer.secho("pass --feature and/or --phase", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    matches = filter_by(has_feature=feature, has_phase=phase)
    for spec in matches:
        typer.echo(repr(spec))


# --- Execution commands ----------------------------------------------------


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
    args = ["benchmarks/", "--quick", "--benchmark-disable", "-q", *ctx.args]
    _run_pytest(args)


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
    model: Annotated[
        str | None,
        typer.Option(help="Restrict to one model (passed as pytest ``-k``)."),
    ] = None,
    filter_expr: Annotated[
        str | None,
        typer.Option(
            "--filter",
            "-k",
            help="Arbitrary pytest ``-k`` expression (AND-ed with ``--model``).",
        ),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option("--json", help="Save pytest-benchmark JSON to this path."),
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

    k_parts = [p for p in (model, filter_expr) if p]
    if k_parts:
        args.extend(["-k", " and ".join(k_parts)])

    args.extend(ctx.args)
    _run_pytest(args)


@app.command()
def notebook() -> None:
    """
    Execute the registry-usage notebook end-to-end.

    Used by CI to catch doc rot — if any cell raises, the workflow fails.
    The executed copy is written to a tempdir and discarded, so the
    in-tree notebook stays output-free (nbstripout doesn't have to chase
    a populated file).
    """
    nb = Path("benchmarks/notebooks/registry_usage.ipynb")
    if not nb.exists():
        typer.secho(f"notebook not found: {nb}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=300",
            "--output-dir",
            tmp,
            "--output",
            "executed.ipynb",
            str(nb),
        ]
        typer.secho(f"$ {' '.join(cmd)}", fg=typer.colors.BRIGHT_BLACK)
        result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


# --- Memory subcommands ----------------------------------------------------


@memory_app.command("save")
def memory_save_cmd(
    label: Annotated[
        str, typer.Argument(help="Label to attach to this snapshot, e.g. a git sha.")
    ],
    quick: Annotated[
        bool, typer.Option("--quick", help="Use smaller problem sizes.")
    ] = False,
    test_path: Annotated[
        list[str] | None,
        typer.Option("--test-path", help="Test file(s) to run; defaults to build."),
    ] = None,
) -> None:
    """
    Run the build phase under pytest-memray and save peak RSS to JSON.

    Results land in ``.benchmarks/memory/<label>.json``. Use ``compare``
    afterwards to diff two snapshots.
    """
    memory_save(label, quick=quick, test_paths=test_path)


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


if __name__ == "__main__":  # pragma: no cover
    app()
