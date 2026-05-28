"""
Interactive plotly views over pytest-benchmark JSON snapshots.

Three opinionated views, all returning the number of tests rendered:

- :func:`plot_compare` (2 snapshots) — sorted-by-delta bar chart.
- :func:`plot_sweep` (3+ snapshots) — heatmap of per-test ratio
  relative to the first snapshot. Useful for cross-version sweeps.
- :func:`plot_scaling` (1 snapshot) — log-log time vs ``n`` for
  size-parametrized tests, faceted by phase.

All three accept a ``metric`` argument selecting which pytest-benchmark
stat drives the plot. Default is ``min`` — for microbenchmarks the
lowest observed time is closest to the "true" cost (noise can only slow
things down). ``median`` is more robust to a single weirdly-fast warmup
round; ``mean`` and ``max`` are also accepted.

plotly is imported lazily by the dispatcher so the rest of the benchmark
suite still works without it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

PlotView = Literal["compare", "scatter", "sweep", "scaling"]
Metric = Literal["min", "median", "mean", "max"]
SortMode = Literal["absolute", "relative"]

_SIZE_RE = re.compile(r"(.*)\[([^\[\]]+?)-n=(\d+)\]")


def _load_snapshot(path: Path, metric: Metric = "min") -> tuple[str, dict[str, float]]:
    """Return ``(label, {fullname: <metric>_seconds})`` for one snapshot."""
    data = json.loads(path.read_text())
    values = {bm["fullname"]: bm["stats"][metric] for bm in data["benchmarks"]}
    return path.stem, values


def plot_compare(
    snapshots: list[Path],
    output: Path,
    metric: Metric = "min",
    sort: SortMode = "absolute",
) -> int:
    """
    Bar chart of delta per test, sorted by magnitude.

    ``sort="absolute"`` (default): bar = (b - a) seconds, sort by the
    largest actual time impact. Best for "what change actually affected
    total runtime?" — avoids over-weighting cheap microsecond tests.

    ``sort="relative"``: bar = (b/a - 1) * 100 %, sort by the largest
    proportional change. Best for "what got proportionally worse?".
    """
    import pandas as pd
    import plotly.express as px

    (a_label, a_vals), (b_label, b_vals) = (
        _load_snapshot(snapshots[0], metric),
        _load_snapshot(snapshots[1], metric),
    )
    common = sorted(set(a_vals) & set(b_vals))
    if not common:
        raise ValueError("no tests in common between the two snapshots")

    rows = [
        {
            "test": name,
            a_label: a_vals[name],
            b_label: b_vals[name],
            "delta_abs": b_vals[name] - a_vals[name],
            "delta_pct": (b_vals[name] - a_vals[name]) / a_vals[name] * 100.0,
        }
        for name in common
    ]
    df = pd.DataFrame(rows)
    x_col = "delta_abs" if sort == "absolute" else "delta_pct"
    df = df.reindex(df[x_col].abs().sort_values(ascending=True).index)

    if sort == "absolute":
        x_label = f"{metric} delta (s)"
        text_fmt = ".2s"
    else:
        x_label = f"{metric} delta %"
        text_fmt = ".1f"

    fig = px.bar(
        df,
        x=x_col,
        y="test",
        orientation="h",
        color=x_col,
        color_continuous_scale=["green", "white", "red"],
        color_continuous_midpoint=0,
        title=f"{metric} delta ({sort}): {a_label} → {b_label} (positive = slower)",
        labels={x_col: x_label, "test": ""},
        text_auto=text_fmt,
        hover_data={
            a_label: ":.4g",
            b_label: ":.4g",
            "delta_abs": ":.4g",
            "delta_pct": ":.2f",
        },
    )
    if sort == "absolute":
        # SI-prefixed time on the x-axis (e.g. 24 ms, 2.4 ms, 240 µs).
        fig.update_xaxes(tickformat=".2s", ticksuffix="s")
    fig.update_layout(height=max(400, len(df) * 14), showlegend=False)
    fig.write_html(output)
    return len(df)


def plot_scatter(
    snapshots: list[Path],
    output: Path,
    metric: Metric = "min",
    sort: SortMode = "absolute",  # noqa: ARG001  (uniform signature, unused here)
) -> int:
    """
    Two-axis scatter — baseline cost on log-x, ratio on y.

    Designed as the single best exploratory plot for regression hunting
    across tests of wildly different magnitudes: a point lights up as
    "fix this" only if it sits in the top-right corner — slow tests
    that got slower. Top-left (big ratio, tiny absolute) reads as
    microbenchmark noise; bottom-right (big absolute, tiny ratio) is
    already-slow-but-unchanged. The combined position resolves the
    tension that pure relative or pure absolute sort each blind-spot.

    A horizontal reference at ``ratio = 1`` makes "no change" trivial
    to see; the colour encodes absolute Δ as a third channel.
    """
    import pandas as pd
    import plotly.express as px

    (a_label, a_vals), (b_label, b_vals) = (
        _load_snapshot(snapshots[0], metric),
        _load_snapshot(snapshots[1], metric),
    )
    common = sorted(set(a_vals) & set(b_vals))
    if not common:
        raise ValueError("no tests in common between the two snapshots")

    rows = []
    for name in common:
        a, b = a_vals[name], b_vals[name]
        if a <= 0:
            continue
        rows.append(
            {
                "test": name,
                "baseline_time": a,
                "ratio": b / a,
                "delta_abs": b - a,
                "delta_pct": (b - a) / a * 100.0,
                a_label: a,
                b_label: b,
            }
        )

    df = pd.DataFrame(rows)
    fig = px.scatter(
        df,
        x="baseline_time",
        y="ratio",
        color="delta_abs",
        color_continuous_scale=["green", "white", "red"],
        color_continuous_midpoint=0,
        log_x=True,
        hover_name="test",
        hover_data={
            a_label: ":.4g",
            b_label: ":.4g",
            "delta_abs": ":.4g",
            "delta_pct": ":.2f",
            "ratio": ":.3f",
            "baseline_time": ":.4g",
        },
        title=(
            f"{metric} scatter: {a_label} → {b_label} "
            "(top-right = slow tests that got slower)"
        ),
        labels={
            "baseline_time": f"baseline {metric} (s, log scale)",
            "ratio": f"{metric} ratio  (candidate / baseline)",
            "delta_abs": "Δ (s)",
        },
    )
    # Reference line at ratio == 1 (no change).
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="grey", annotation_text="no change"
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="DarkSlateGrey")))
    fig.update_layout(height=600)
    fig.write_html(output)
    return len(df)


def plot_sweep(
    snapshots: list[Path],
    output: Path,
    metric: Metric = "min",
    sort: SortMode = "absolute",  # noqa: ARG001  (uniform signature, unused here)
) -> int:
    """Heatmap of per-test ratio relative to the first snapshot."""
    import pandas as pd
    import plotly.express as px

    loaded = [_load_snapshot(p, metric) for p in snapshots]
    versions = [label for label, _ in loaded]
    baseline = loaded[0][1]
    all_tests = sorted(set().union(*[set(vals) for _, vals in loaded]))

    ratios: dict[str, list[float | None]] = {}
    absolutes: dict[str, list[float | None]] = {}
    for test in all_tests:
        base = baseline.get(test)
        if not base:
            continue
        ratios[test] = []
        absolutes[test] = []
        for _, vals in loaded:
            t = vals.get(test)
            ratios[test].append(t / base if t else None)
            absolutes[test].append(t)

    if not ratios:
        raise ValueError(f"no overlap with baseline snapshot {versions[0]}")

    df = pd.DataFrame(ratios, index=versions).T  # rows = tests, cols = versions
    abs_df = pd.DataFrame(absolutes, index=versions).T

    fig = px.imshow(
        df,
        color_continuous_scale=["green", "white", "red"],
        color_continuous_midpoint=1.0,
        aspect="auto",
        title=f"{metric} ratio relative to baseline ({versions[0]})",
        labels={"x": "version", "y": "test", "color": "ratio"},
        text_auto=".2f",
    )
    # Inject absolute values as customdata so hover shows both.
    fig.update_traces(
        customdata=abs_df.values,
        hovertemplate=(
            "test: %{y}<br>"
            "version: %{x}<br>"
            "ratio: %{z:.3f}<br>"
            f"{metric}: %{{customdata:.4g}}s"
            "<extra></extra>"
        ),
    )
    fig.update_layout(height=max(400, len(df) * 14))
    fig.write_html(output)
    return len(df)


def plot_scaling(
    snapshots: list[Path],
    output: Path,
    metric: Metric = "min",
    sort: SortMode = "absolute",  # noqa: ARG001  (uniform signature, unused here)
) -> int:
    """Log-log time vs N for size-parametrized tests, faceted by phase."""
    import pandas as pd
    import plotly.express as px

    _, vals = _load_snapshot(snapshots[0], metric)
    rows = []
    for name, t in vals.items():
        m = _SIZE_RE.match(name)
        if not m:
            continue
        phase_path, model, n = m.groups()
        phase = phase_path.split("::")[-1]
        rows.append({"phase": phase, "model": model, "n": int(n), metric: t})

    if not rows:
        raise ValueError(
            "no size-parametrized tests found (expected ``...[<model>-n=<N>]``)"
        )

    df = pd.DataFrame(rows).sort_values(["phase", "model", "n"])
    fig = px.line(
        df,
        x="n",
        y=metric,
        color="model",
        facet_col="phase",
        facet_col_wrap=3,
        log_x=True,
        log_y=True,
        markers=True,
        title=f"Scaling: {metric} time vs problem size ({snapshots[0].stem})",
    )
    fig.update_layout(height=max(400, ((df["phase"].nunique() + 2) // 3) * 350))
    fig.write_html(output)
    return len(df)


RENDERERS: dict[PlotView, Callable[[list[Path], Path, Metric, SortMode], int]] = {
    "compare": plot_compare,
    "scatter": plot_scatter,
    "sweep": plot_sweep,
    "scaling": plot_scaling,
}
