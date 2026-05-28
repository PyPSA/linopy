"""
linopy benchmark CLI — one entry point for the suite.

Run with::

    python -m benchmarks <command> [options]

The CLI is a thin layer over pytest for the timing / smoke commands, plus
direct dispatch for registry introspection and memory snapshots.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
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
from benchmarks.plotting import FacetBy, Metric, PlotView, SortMode

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


def _benchmarks_extra_pins() -> list[str]:
    """
    Return the pins from ``pyproject.toml``'s ``[benchmarks]`` extra.

    Both ``sweep`` and ``memory sweep`` install these into each
    per-version venv. Direct pins are kept in pyproject as the single
    source of truth — bump them there and both sweeps pick up the
    change. Transitive deps resolve fresh per venv; uv's deterministic
    resolution gives identical results across versions within one sweep.
    """
    import tomllib

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    return list(data["project"]["optional-dependencies"]["benchmarks"])


_PHASE_TEST_FILE: dict[PhaseName, str] = {
    "build": "benchmarks/test_build.py",
    "matrices": "benchmarks/test_matrices.py",
    "lp_write": "benchmarks/test_lp_write.py",
    "netcdf": "benchmarks/test_netcdf.py",
    "solver_handoff": "benchmarks/test_solver_handoff.py",
}

# pytest args that constitute a "smoke" run — quick sizes, no timings.
# Shared between the top-level ``smoke`` command and ``sweep --smoke`` so
# bumping the definition stays single-source.
_SMOKE_PYTEST_ARGS = ["benchmarks/", "--quick", "--benchmark-disable", "-q"]


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
    # ``secho`` strips colour automatically when stdout isn't a TTY, so
    # piping ``list --details | grep`` still gets plain text.
    typer.secho(
        f"{'name':<{name_w}}  {'features':<{feat_w}}  sizes",
        dim=True,
    )
    typer.secho("-" * (name_w + feat_w + 20), dim=True)
    for name, feats, sizes in rows:
        typer.secho(f"{name:<{name_w}}", fg=typer.colors.CYAN, nl=False)
        typer.echo(f"  {feats:<{feat_w}}  {sizes}")


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

    def _row(label: str, value: object) -> None:
        # Dim the label so the eye lands on the value first; ``secho``
        # auto-strips colour when stdout isn't a TTY.
        typer.secho(f"  {label:<17}", dim=True, nl=False)
        typer.echo(value)

    _row("sizes:", spec.sizes)
    _row("features:", sorted(spec.features))
    _row("phases:", sorted(spec.phases))
    _row("quick_threshold:", spec.quick_threshold)
    _row("long_threshold:", spec.long_threshold)
    if spec.requires:
        _row("requires:", list(spec.requires))


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

    k_parts = [p for p in (model, filter_expr) if p]
    if k_parts:
        args.extend(["-k", " and ".join(k_parts)])

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
        # Jupytext sets the kernel cwd to the output directory (the
        # tempdir here), so forward the repo root via
        # ``LINOPY_REPO_ROOT`` for the walkthrough's first cell to find
        # ``benchmarks/``.
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


# --- Sweep across linopy versions ------------------------------------------


_PLAIN_VERSION_RE = re.compile(r"^\d+(\.\d+)*([a-z]+\d*)?$")


def _linopy_install_spec(version: str) -> str:
    """Turn ``0.4.0`` → ``linopy==0.4.0``, leave anything URL-y untouched."""
    if _PLAIN_VERSION_RE.match(version):
        return f"linopy=={version}"
    return version


def _venv_python(venv: Path) -> Path:
    return (
        venv / "Scripts" / "python.exe" if os.name == "nt" else venv / "bin" / "python"
    )


@dataclass(frozen=True)
class _ProvisionedVenv:
    """
    One fresh per-version venv from :func:`_provision_venvs`.

    On success, ``python`` and ``env`` are populated and ``failed_at``
    is ``None``. On failure, ``failed_at`` names the step that failed
    (``"venv"`` or ``"install"``); the caller skips its per-version
    action and records the failure.
    """

    version: str
    python: Path | None
    env: dict[str, str] | None
    failed_at: str | None


def _provision_venvs(
    versions: list[str], tmp_prefix: str
) -> Iterator[_ProvisionedVenv]:
    """
    Yield one fresh per-version uv venv for each linopy version.

    Used by both ``sweep`` and ``memory sweep`` so the venv plumbing
    (uv venv → install ``[benchmarks]`` pins + the target linopy →
    set ``PYTHONPATH``) lives in one place. The caller supplies the
    tempdir prefix (so ``ps``/``lsof`` can distinguish concurrent
    runs) and does whatever per-version action it needs.

    Each version's tempdir is cleaned up when the generator advances
    (or exits). The caller can break the loop early — Python's
    generator close protocol fires the ``with`` teardown.
    """
    if shutil.which("uv") is None:
        typer.secho(
            "uv not found on PATH — install via https://docs.astral.sh/uv/",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    repo_root = Path.cwd()
    for version in versions:
        typer.secho(f"\n=== linopy {version} ===", fg=typer.colors.CYAN, bold=True)
        with tempfile.TemporaryDirectory(prefix=tmp_prefix) as tmp:
            venv = Path(tmp) / "venv"

            r = subprocess.run(
                ["uv", "venv", "--python", sys.executable, str(venv)],
                check=False,
            )
            if r.returncode != 0:
                typer.secho(
                    f"venv creation failed: {version}",
                    fg=typer.colors.RED,
                    err=True,
                )
                yield _ProvisionedVenv(version, None, None, "venv")
                continue

            vpy = _venv_python(venv)
            spec = _linopy_install_spec(version)

            # Single install pass: pinned infra from pyproject + linopy.
            # Direct pins in [benchmarks] are sufficient for sweep
            # reproducibility — uv resolves the same input deterministically
            # into each per-version venv.
            install_args = [
                "uv",
                "pip",
                "install",
                "--python",
                str(vpy),
                *_benchmarks_extra_pins(),
                spec,
            ]
            r = subprocess.run(install_args, check=False)
            if r.returncode != 0:
                typer.secho(f"install failed: {version}", fg=typer.colors.RED, err=True)
                yield _ProvisionedVenv(version, None, None, "install")
                continue

            # PYTHONPATH makes ``import benchmarks`` resolve against the
            # local checkout — the venv only provides linopy + test infra.
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root)

            yield _ProvisionedVenv(version, vpy, env, None)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def sweep(
    ctx: typer.Context,
    versions: Annotated[
        list[str],
        typer.Argument(help="linopy versions, e.g. 0.4.0 0.5.0 (or any pip spec)."),
    ],
    output_dir: Annotated[
        Path,
        typer.Option("--output-dir", "-o", help="Where to save snapshot JSONs."),
    ] = Path(".benchmarks/sweep"),
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
    if quick and long:
        typer.secho(
            "--quick and --long are mutually exclusive",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    if smoke and (long or rounds is not None):
        typer.secho(
            "--smoke can't be combined with --long or --rounds "
            "(no timings are recorded in smoke mode).",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    if not smoke:
        output_dir.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    for prov in _provision_venvs(versions, "linopy-bench-"):
        if prov.failed_at:
            failed.append(prov.version)
            continue

        if smoke:
            # Smoke mode: reuse the same pytest args as the top-level
            # ``smoke`` command. No JSON snapshot, return code is the
            # signal.
            pytest_cmd = [str(prov.python), "-m", "pytest", *_SMOKE_PYTEST_ARGS]
            k_parts = [p for p in (model, filter_expr) if p]
            if k_parts:
                pytest_cmd.extend(["-k", " and ".join(k_parts)])
            pytest_cmd.extend(ctx.args)

            typer.secho(f"$ {' '.join(pytest_cmd)}", fg=typer.colors.BRIGHT_BLACK)
            r = subprocess.run(pytest_cmd, env=prov.env, check=False)
            if r.returncode != 0:
                typer.secho(
                    f"smoke failed: {prov.version}", fg=typer.colors.RED, err=True
                )
                failed.append(prov.version)
            else:
                typer.secho(f"smoke ok: {prov.version}", fg=typer.colors.GREEN)
            continue

        snapshot = (output_dir / f"linopy-{prov.version}.json").resolve()
        test_target = _PHASE_TEST_FILE[phase] if phase is not None else "benchmarks/"
        pytest_cmd = [
            str(prov.python),
            "-m",
            "pytest",
            test_target,
            "--benchmark-only",
            "--benchmark-json",
            str(snapshot),
        ]
        if quick:
            pytest_cmd.append("--quick")
        elif long:
            pytest_cmd.append("--long")
        if rounds is not None:
            pytest_cmd.extend(
                [f"--benchmark-min-rounds={rounds}", "--benchmark-max-time=0"]
            )

        k_parts = [p for p in (model, filter_expr) if p]
        if k_parts:
            pytest_cmd.extend(["-k", " and ".join(k_parts)])

        pytest_cmd.extend(ctx.args)

        typer.secho(f"$ {' '.join(pytest_cmd)}", fg=typer.colors.BRIGHT_BLACK)
        subprocess.run(pytest_cmd, env=prov.env, check=False)

        if snapshot.exists():
            typer.secho(f"saved {snapshot}", fg=typer.colors.GREEN)
        else:
            typer.secho(
                f"no snapshot produced for {prov.version}",
                fg=typer.colors.RED,
                err=True,
            )
            failed.append(prov.version)

    if failed:
        typer.secho(f"\nFailed versions: {failed}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


# --- Compare timing snapshots ---------------------------------------------


def _discover_snapshots() -> list[Path]:
    """
    Return JSON snapshot files under the canonical .benchmarks/ tree.

    Paths are relative to cwd so they're easier to copy-paste back into
    the CLI than the absolute form would be.
    """
    root = Path(".benchmarks")
    if not root.exists():
        return []
    return sorted(root.rglob("*.json"))


def _suggest_snapshots(reason: str) -> None:
    """Print an error + a hint listing whatever snapshots we can find."""
    typer.secho(reason, fg=typer.colors.RED, err=True)
    found = _discover_snapshots()
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


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def compare(ctx: typer.Context) -> None:
    """
    Compare timing snapshots side-by-side via ``pytest-benchmark compare``.

    Thin wrapper around the upstream tool so the whole suite stays under
    one entry point. Pass the snapshot paths first, then any pytest-benchmark
    flags::

        python -m benchmarks compare a.json b.json
        python -m benchmarks compare a.json b.json --group-by=name
        python -m benchmarks compare a.json b.json --histogram=plots/cmp

    With no arguments (or missing paths), prints what snapshots exist
    under ``.benchmarks/`` so you can copy-paste the path you want.

    For memory snapshots use ``memory compare`` instead — different format,
    different tool.

    Implementation note: typer/click don't have a clean idiom for "list-typed
    positional + pass-through", so this command parses ``ctx.args`` by hand
    — anything before the first flag is a snapshot path, everything after
    is forwarded.
    """
    # Snapshots come first; once we see a flag (``-x`` / ``--foo``) every
    # subsequent token is forwarded to pytest-benchmark. That way the value
    # of a flag like ``-k "build and basic"`` doesn't get mistaken for a path.
    snapshots: list[Path] = []
    extra: list[str] = []
    seen_flag = False
    for arg in ctx.args:
        if arg.startswith("-"):
            seen_flag = True
        if seen_flag:
            extra.append(arg)
        else:
            snapshots.append(Path(arg))

    if len(snapshots) < 2:
        _suggest_snapshots(
            f"compare needs at least two snapshot paths (got {len(snapshots)})."
        )
        raise typer.Exit(code=2)

    missing = [p for p in snapshots if not p.exists()]
    if missing:
        _suggest_snapshots(f"missing snapshots: {[str(p) for p in missing]}")
        raise typer.Exit(code=2)

    # Sensible defaults — pytest-benchmark's defaults emit 10 columns wide,
    # grouped by parametrize group, which is unreadable for two-snapshot diffs.
    # ``--group-by=fullname`` puts each test's (baseline, candidate) rows in
    # their own mini-table; ``--columns=min,iqr`` shows the lowest observed
    # time (approximates the no-noise floor) plus the spread.
    # Each default is only applied if the user didn't override it.
    if not any(a.startswith("--columns") for a in extra):
        extra.insert(0, "--columns=min,iqr")
    if not any(a.startswith("--sort") for a in extra):
        extra.insert(0, "--sort=min")
    if not any(a.startswith("--group-by") for a in extra):
        extra.insert(0, "--group-by=fullname")

    cmd = [
        sys.executable,
        "-m",
        "pytest_benchmark",
        "compare",
        *[str(p) for p in snapshots],
        *extra,
    ]
    typer.secho(f"$ {' '.join(cmd)}", fg=typer.colors.BRIGHT_BLACK)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)


# --- Plotting --------------------------------------------------------------


@app.command()
def plot(
    snapshots: Annotated[
        list[Path],
        typer.Argument(help="pytest-benchmark JSON snapshot(s)."),
    ],
    view: Annotated[
        PlotView | None,
        typer.Option(
            help=(
                "Which plot to produce. Default: ``scaling`` for 1 input, "
                "``scatter`` for 2, ``sweep`` for 3+. ``compare`` (delta "
                "bar chart) is still available via ``--view compare``."
            )
        ),
    ] = None,
    metric: Annotated[
        Metric,
        typer.Option(
            help=(
                "Stat to drive the plot. ``min`` (default) is closest to "
                "the 'true' cost — noise can only slow things down. ``median``"
                " is more robust to a single fast warmup round."
            )
        ),
    ] = "min",
    sort: Annotated[
        SortMode,
        typer.Option(
            help=(
                "Compare-view sort and bar dimension. ``absolute`` (default) "
                "uses ``b - a`` in seconds so the biggest actual-time impacts "
                "float to the bottom — avoids over-weighting cheap "
                "microsecond tests. ``relative`` uses percent change."
            )
        ),
    ] = "absolute",
    facets: Annotated[
        FacetBy | None,
        typer.Option(
            "--facets",
            help=(
                "Split compare / scatter into subplots by ``phase`` (test "
                "file) or ``model`` (parametrize id). Default: no faceting. "
                "Tests whose ids don't match ``[<model>-n=<size>]`` (e.g. "
                "PyPSA carbon-management) land in an ``other`` facet."
            ),
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help=(
                "Where to write the HTML. Defaults to "
                "``.benchmarks/plots/<view>.html`` (gitignored) so "
                "different views don't clobber each other."
            ),
        ),
    ] = None,
    open_browser: Annotated[
        bool,
        typer.Option("--open/--no-open", help="Open the result in a browser."),
    ] = False,
) -> None:
    """
    Render an interactive HTML plot from one or more snapshots.

    Four views, picked automatically from the snapshot count (compare
    for 2, sweep for 3+, scaling for 1) or set explicitly via ``--view``:

    - **compare** (2 snapshots) — horizontal bar chart of per-test delta,
      sorted by magnitude. The "did this PR regress anything?" picture.
    - **scatter** (2 snapshots) — exploratory two-axis plot: baseline
      cost on log-x, ratio on y, absolute Δ encoded in colour. Tests
      in the top-right are the real regressions (slow tests that got
      slower); top-left = cheap tests with big ratio swings (noise,
      not real change); bottom-right = already-slow-but-unchanged.
      Resolves the absolute-vs-relative tension visually.
    - **sweep** (3+ snapshots) — heatmap of ratio relative to the first
      snapshot, rows = tests, columns = snapshot labels.
    - **scaling** (1 snapshot) — log-log time vs ``n`` for
      size-parametrized tests, faceted by phase.

    Output is an interactive Plotly HTML file. Open it in any browser
    (or pass ``--open``).
    """
    missing = [p for p in snapshots if not p.exists()]
    if missing:
        _suggest_snapshots(f"missing snapshots: {[str(p) for p in missing]}")
        raise typer.Exit(code=2)

    chosen = view or (
        "scaling"
        if len(snapshots) == 1
        else "scatter"
        if len(snapshots) == 2
        else "sweep"
    )
    if chosen == "compare" and len(snapshots) != 2:
        typer.secho(
            "compare view needs exactly 2 snapshots", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=2)
    if chosen == "scatter" and len(snapshots) < 2:
        typer.secho(
            "scatter view needs at least 2 snapshots (baseline + 1)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    if chosen == "scaling" and len(snapshots) != 1:
        typer.secho(
            "scaling view takes exactly 1 snapshot", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=2)

    try:
        from benchmarks.plotting import RENDERERS
    except ImportError as exc:
        typer.secho(
            "plotly is required for ``plot`` — ``pip install plotly``",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2) from exc

    # Default filename: ``.benchmarks/plots/<view>.html``. Matches where
    # snapshots already live (and is gitignored), and the per-view name
    # means consecutive ``plot`` calls don't clobber each other.
    if output is None:
        output = Path(".benchmarks") / "plots" / f"{chosen}.html"

    try:
        fig, n_tests = RENDERERS[chosen](snapshots, metric, sort, facets)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output)

    typer.secho(
        f"{chosen} view ({metric}): {n_tests} tests → {output}",
        fg=typer.colors.GREEN,
    )
    if open_browser:
        import webbrowser

        webbrowser.open(output.resolve().as_uri())


# --- Memory subcommands ----------------------------------------------------


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
                "to select more than one. Default: all (build, matrices, lp_write,"
                " netcdf, solver_handoff)."
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
    from benchmarks.memory import MEMORY_PHASES

    if phase:
        unknown = [p for p in phase if p not in MEMORY_PHASES]
        if unknown:
            typer.secho(
                f"unknown phase(s): {unknown}; valid options: {list(MEMORY_PHASES)}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
    memory_save(label, quick=quick, phases=phase, repeats=repeats)


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
    from benchmarks.memory import MEMORY_PHASES

    if phase:
        unknown = [p for p in phase if p not in MEMORY_PHASES]
        if unknown:
            typer.secho(
                f"unknown phase(s): {unknown}; valid options: {list(MEMORY_PHASES)}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path.cwd()

    failed: list[str] = []
    for prov in _provision_venvs(versions, "linopy-mem-"):
        if prov.failed_at:
            failed.append(prov.version)
            continue

        # ``memory save`` writes to ``.benchmarks/memory/<label>.json``
        # under cwd; we run it with cwd pinned to repo root, then move
        # the file if the user asked for a custom output dir.
        label = f"linopy-{prov.version}"
        mem_cmd = [
            str(prov.python),
            "-m",
            "benchmarks",
            "memory",
            "save",
            label,
        ]
        if quick:
            mem_cmd.append("--quick")
        for ph in phase or []:
            mem_cmd.extend(["--phase", ph])
        if repeats > 1:
            mem_cmd.extend(["--repeats", str(repeats)])

        typer.secho(f"$ {' '.join(mem_cmd)}", fg=typer.colors.BRIGHT_BLACK)
        subprocess.run(mem_cmd, env=prov.env, cwd=str(repo_root), check=False)

        default_path = repo_root / ".benchmarks" / "memory" / f"{label}.json"
        target = output_dir / f"{label}.json"
        if default_path.exists() and default_path.resolve() != target.resolve():
            target.parent.mkdir(parents=True, exist_ok=True)
            default_path.replace(target)

        if target.exists():
            typer.secho(f"saved {target}", fg=typer.colors.GREEN)
        else:
            typer.secho(
                f"no snapshot produced for {prov.version}",
                fg=typer.colors.RED,
                err=True,
            )
            failed.append(prov.version)

    if failed:
        typer.secho(f"\nFailed versions: {failed}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


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
