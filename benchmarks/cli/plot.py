"""Plotting command: ``plot``."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Annotated

import typer

from benchmarks.cli._base import _suggest_snapshots, app
from benchmarks.plotting import FacetBy, Metric, PlotView, SortMode


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
                "file) or ``spec`` (parametrize id). Default: no faceting. "
                "Tests whose ids don't match ``[<spec>-<axis>=<value>]`` (e.g. "
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
    - **scaling** (1 snapshot) — cost vs the sweep dial, faceted by phase:
      log-log for model ``size``, linear for pattern ``severity`` (0-100).

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

    # RENDERERS imports fine without plotly (lazy inside each), so check the dep.
    if importlib.util.find_spec("plotly") is None:
        typer.secho(
            "plotly is required for ``plot`` — ``pip install plotly``",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    from benchmarks.plotting import RENDERERS

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
