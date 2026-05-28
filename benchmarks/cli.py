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
    use_lock: Annotated[
        bool,
        typer.Option(
            "--use-lock/--no-use-lock",
            help="Install ``benchmarks/requirements.lock`` in each venv.",
        ),
    ] = True,
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

    if shutil.which("uv") is None:
        typer.secho(
            "uv not found on PATH — install via https://docs.astral.sh/uv/",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    repo_root = Path.cwd()
    lockfile = repo_root / "benchmarks" / "requirements.lock"
    if use_lock and not lockfile.exists():
        typer.secho(
            f"--use-lock set but {lockfile} is missing — "
            "regenerate it via ``uv pip compile`` or pass ``--no-use-lock``.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    output_dir.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    for version in versions:
        typer.secho(f"\n=== linopy {version} ===", fg=typer.colors.CYAN, bold=True)
        with tempfile.TemporaryDirectory(prefix="linopy-bench-") as tmp:
            venv = Path(tmp) / "venv"

            # 1. uv venv — same interpreter that's driving the CLI.
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
                failed.append(version)
                continue

            vpy = _venv_python(venv)
            spec = _linopy_install_spec(version)

            # 2. Single install pass: infra (lockfile or pinned subset) + linopy.
            install_args = ["uv", "pip", "install", "--python", str(vpy)]
            if use_lock:
                install_args += ["-r", str(lockfile)]
            else:
                install_args += [
                    "pytest==9.0.3",
                    "pytest-benchmark==5.2.3",
                    "highspy==1.13.1",
                    "netcdf4==1.7.4",
                ]
            install_args.append(spec)
            r = subprocess.run(install_args, check=False)
            if r.returncode != 0:
                typer.secho(f"install failed: {version}", fg=typer.colors.RED, err=True)
                failed.append(version)
                continue

            # 3. Run the benchmarks. PYTHONPATH makes ``import benchmarks``
            #    resolve against the local checkout — the venv only needs to
            #    provide linopy + the test infra.
            snapshot = (output_dir / f"linopy-{version}.json").resolve()
            env = os.environ.copy()
            env["PYTHONPATH"] = str(repo_root)

            test_target = (
                _PHASE_TEST_FILE[phase] if phase is not None else "benchmarks/"
            )
            pytest_cmd = [
                str(vpy),
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

            k_parts = [p for p in (model, filter_expr) if p]
            if k_parts:
                pytest_cmd.extend(["-k", " and ".join(k_parts)])

            pytest_cmd.extend(ctx.args)

            typer.secho(f"$ {' '.join(pytest_cmd)}", fg=typer.colors.BRIGHT_BLACK)
            subprocess.run(pytest_cmd, env=env, check=False)

            if snapshot.exists():
                typer.secho(f"saved {snapshot}", fg=typer.colors.GREEN)
            else:
                typer.secho(
                    f"no snapshot produced for {version}",
                    fg=typer.colors.RED,
                    err=True,
                )
                failed.append(version)

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
    # their own mini-table; ``--columns=median,iqr`` keeps it narrow.
    # Each default is only applied if the user didn't override it.
    if not any(a.startswith("--columns") for a in extra):
        extra.insert(0, "--columns=median,iqr")
    if not any(a.startswith("--sort") for a in extra):
        extra.insert(0, "--sort=name")
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


PlotView = Literal["compare", "sweep", "scaling"]


def _load_snapshot(path: Path) -> tuple[str, dict[str, float]]:
    """Return (label, {fullname: median_seconds}) for a pytest-benchmark JSON."""
    import json

    data = json.loads(path.read_text())
    medians = {bm["fullname"]: bm["stats"]["median"] for bm in data["benchmarks"]}
    return path.stem, medians


def _plot_compare(snapshots: list[Path], output: Path) -> int:
    """Bar chart of relative median delta per test, sorted by magnitude."""
    import pandas as pd
    import plotly.express as px

    (a_label, a_med), (b_label, b_med) = (
        _load_snapshot(snapshots[0]),
        _load_snapshot(snapshots[1]),
    )
    common = sorted(set(a_med) & set(b_med))
    if not common:
        typer.secho(
            "no tests in common between the two snapshots",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    rows = [
        {
            "test": name,
            "delta_pct": (b_med[name] - a_med[name]) / a_med[name] * 100.0,
        }
        for name in common
    ]
    df = pd.DataFrame(rows)
    df = df.reindex(df["delta_pct"].abs().sort_values(ascending=True).index)

    fig = px.bar(
        df,
        x="delta_pct",
        y="test",
        orientation="h",
        color="delta_pct",
        color_continuous_scale=["green", "white", "red"],
        color_continuous_midpoint=0,
        title=f"Median delta: {a_label} → {b_label} (positive = slower)",
        labels={"delta_pct": "median delta %", "test": ""},
    )
    fig.update_layout(height=max(400, len(df) * 14), showlegend=False)
    fig.write_html(output)
    return len(df)


def _plot_sweep(snapshots: list[Path], output: Path) -> int:
    """Heatmap of per-test median ratio relative to the first snapshot."""
    import pandas as pd
    import plotly.express as px

    loaded = [_load_snapshot(p) for p in snapshots]
    versions = [label for label, _ in loaded]
    baseline = loaded[0][1]
    all_tests = sorted(set().union(*[set(med) for _, med in loaded]))

    matrix: dict[str, list[float | None]] = {}
    for test in all_tests:
        base = baseline.get(test)
        if not base:
            continue
        row = []
        for _, med in loaded:
            t = med.get(test)
            row.append(t / base if t else None)
        matrix[test] = row

    if not matrix:
        typer.secho(
            f"no overlap with baseline snapshot {versions[0]}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    df = pd.DataFrame(matrix, index=versions).T  # rows = tests, cols = versions
    fig = px.imshow(
        df,
        color_continuous_scale=["green", "white", "red"],
        color_continuous_midpoint=1.0,
        aspect="auto",
        title=f"Median ratio relative to baseline ({versions[0]})",
        labels={"x": "version", "y": "test", "color": "ratio"},
    )
    fig.update_layout(height=max(400, len(df) * 14))
    fig.write_html(output)
    return len(df)


_SIZE_RE = re.compile(r"(.*)\[([^\[\]]+?)-n=(\d+)\]")


def _plot_scaling(snapshots: list[Path], output: Path) -> int:
    """Log-log median vs N for size-parametrized tests, faceted by phase."""
    import pandas as pd
    import plotly.express as px

    _, med = _load_snapshot(snapshots[0])
    rows = []
    for name, t in med.items():
        m = _SIZE_RE.match(name)
        if not m:
            continue
        phase_path, model, n = m.groups()
        phase = phase_path.split("::")[-1]
        rows.append({"phase": phase, "model": model, "n": int(n), "median": t})

    if not rows:
        typer.secho(
            "no size-parametrized tests found (expected ``...[<model>-n=<N>]``)",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    df = pd.DataFrame(rows).sort_values(["phase", "model", "n"])
    fig = px.line(
        df,
        x="n",
        y="median",
        color="model",
        facet_col="phase",
        facet_col_wrap=3,
        log_x=True,
        log_y=True,
        markers=True,
        title=f"Scaling: median time vs problem size ({snapshots[0].stem})",
    )
    fig.update_layout(height=max(400, ((df["phase"].nunique() + 2) // 3) * 350))
    fig.write_html(output)
    return len(df)


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
                "``compare`` for 2, ``sweep`` for 3+."
            )
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Where to write the HTML."),
    ] = Path("benchmark-plot.html"),
    open_browser: Annotated[
        bool,
        typer.Option("--open/--no-open", help="Open the result in a browser."),
    ] = False,
) -> None:
    """
    Render an interactive HTML plot from one or more snapshots.

    Three views, picked automatically from the snapshot count or set
    explicitly via ``--view``:

    - **compare** (2 snapshots) — horizontal bar chart of per-test median
      delta, sorted by magnitude, green→red colormap. The "did this PR
      regress anything?" picture in one glance.
    - **sweep** (3+ snapshots) — heatmap of median ratio relative to the
      first snapshot, rows = tests, columns = snapshot labels. Useful
      for cross-version sweeps from ``sweep``.
    - **scaling** (1 snapshot) — log-log median vs ``n`` for
      size-parametrized tests, faceted by phase. Shows whether linopy's
      complexity scales as expected.

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
        else "compare"
        if len(snapshots) == 2
        else "sweep"
    )
    if chosen == "compare" and len(snapshots) != 2:
        typer.secho(
            "compare view needs exactly 2 snapshots", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=2)
    if chosen == "scaling" and len(snapshots) != 1:
        typer.secho(
            "scaling view takes exactly 1 snapshot", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=2)

    try:
        import plotly.express  # noqa: F401
    except ImportError as exc:
        typer.secho(
            "plotly is required for ``plot`` — ``pip install plotly``",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = {
        "compare": _plot_compare,
        "sweep": _plot_sweep,
        "scaling": _plot_scaling,
    }[chosen](snapshots, output)

    typer.secho(f"{chosen} view: {rendered} tests → {output}", fg=typer.colors.GREEN)
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
