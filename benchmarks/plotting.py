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
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

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
    metric: Metric = "min",
    sort: SortMode = "absolute",
) -> tuple[Figure, int]:
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
    only_a = sorted(set(a_vals) - set(b_vals))
    only_b = sorted(set(b_vals) - set(a_vals))
    if not common:
        raise ValueError("no tests in common between the two snapshots")
    if only_a or only_b:
        # Surface the mismatch so silent intersection isn't a footgun.
        import sys

        print(
            f"compare: {len(only_a)} test(s) only in {a_label}, "
            f"{len(only_b)} only in {b_label} (intersection: {len(common)}).",
            file=sys.stderr,
        )

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

    title = f"{metric} delta ({sort}): {a_label} → {b_label} (positive = slower)"
    if only_a or only_b:
        title += f"<br><sub>{len(only_a)} only in {a_label}, {len(only_b)} only in {b_label}</sub>"

    fig = px.bar(
        df,
        x=x_col,
        y="test",
        orientation="h",
        color=x_col,
        color_continuous_scale=["green", "white", "red"],
        color_continuous_midpoint=0,
        title=title,
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
    fig.update_layout(height=max(500, len(df) * 22), showlegend=False)
    return fig, len(df)


def plot_scatter(
    snapshots: list[Path],
    metric: Metric = "min",
    sort: SortMode = "absolute",  # noqa: ARG001  (uniform signature, unused here)
) -> tuple[Figure, int]:
    """
    Two-axis scatter — baseline cost on log-x, ratio on y.

    Designed as the single best exploratory plot for regression hunting
    across tests of wildly different magnitudes: a point lights up as
    "fix this" only if it sits in the top-right corner — slow tests
    that got slower. Top-left (big ratio, tiny absolute) reads as
    microbenchmark noise; bottom-right (big absolute, tiny ratio) is
    already-slow-but-unchanged. The combined position resolves the
    tension that pure relative or pure absolute sort each blind-spot.

    The first snapshot is the baseline. With 2 snapshots, a static
    scatter is drawn; with 3+, every subsequent snapshot becomes an
    ``animation_frame`` — use the slider / play button to step through
    versions and watch points drift across releases.

    A horizontal reference at ``ratio = 1`` makes "no change" trivial
    to see; the colour encodes absolute Δ as a third channel.
    """
    import numpy as np
    import pandas as pd
    import plotly.express as px

    if len(snapshots) < 2:
        raise ValueError("scatter needs at least 2 snapshots (baseline + 1)")

    loaded = [_load_snapshot(p, metric) for p in snapshots]
    baseline_label, baseline_vals = loaded[0]

    # Include the baseline itself as the first animation frame (all points
    # at ratio=1, Δ=0). Gives the animation a "before anything happened"
    # anchor and makes the visual drift across frames easier to read.
    rows = []
    for label, vals in loaded:
        common = sorted(set(baseline_vals) & set(vals))
        for name in common:
            a, b = baseline_vals[name], vals[name]
            if a <= 0:
                continue
            rows.append(
                {
                    "test": name,
                    "version": label,
                    "baseline_time": a,
                    "candidate_time": b,
                    "ratio": b / a,
                    "delta_abs": b - a,
                    "delta_pct": (b - a) / a * 100.0,
                }
            )

    if not rows:
        raise ValueError(
            f"no tests in common between baseline ({baseline_label}) "
            "and any of the other snapshots"
        )

    df = pd.DataFrame(rows)
    # Fix the axis ranges so the animation doesn't jitter; pad by a small
    # margin so points on the edges aren't clipped.
    x_lo, x_hi = df["baseline_time"].min(), df["baseline_time"].max()
    # y-range uses min/max but is centered symmetrically around 1.0 (the
    # "no change" line), so regressions above and improvements below are
    # equally readable. Asymmetric data still resolves — the larger side
    # just dictates how wide the symmetric window is.
    y_lo, y_hi = df["ratio"].min(), df["ratio"].max()
    max_dist = max(abs(1.0 - y_lo), abs(y_hi - 1.0), 0.05)
    pad_y = max(0.05, max_dist * 0.05)
    y_range = [1.0 - max_dist - pad_y, 1.0 + max_dist + pad_y]

    # Clip the colour scale to the 95th-percentile absolute Δ so a single
    # huge regression doesn't wash everything else to white. Outliers
    # saturate at the bound, the rest stays readable.
    clip = float(np.percentile(df["delta_abs"].abs(), 95)) if len(df) > 0 else 0.0
    if clip == 0.0:
        max_abs = float(df["delta_abs"].abs().max())
        clip = max_abs if max_abs > 0 else 1e-9

    animate = len(snapshots) >= 3
    extra: dict = {}
    if animate:
        extra["animation_frame"] = "version"
        extra["category_orders"] = {"version": [label for label, _ in loaded]}

    fig = px.scatter(
        df,
        x="baseline_time",
        y="ratio",
        color="delta_abs",
        color_continuous_scale=["green", "white", "red"],
        color_continuous_midpoint=0,
        range_color=[-clip, clip],
        log_x=True,
        range_x=[x_lo * 0.5, x_hi * 2],
        range_y=y_range,
        hover_name="test",
        hover_data={
            "baseline_time": ":.4g",
            "candidate_time": ":.4g",
            "delta_abs": ":.4g",
            "delta_pct": ":.2f",
            "ratio": ":.3f",
            "version": True,
        },
        title=(
            f"{metric} scatter vs baseline ({baseline_label}) — "
            "top-right = slow tests that got slower"
        ),
        labels={
            "baseline_time": f"baseline {metric} (s, log scale)",
            "ratio": f"{metric} ratio  (candidate / baseline)",
            "candidate_time": "candidate",
            "delta_abs": "Δ (s, p95-clipped)",
        },
        **extra,
    )
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="grey", annotation_text="no change"
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="DarkSlateGrey")))
    fig.update_layout(height=600)
    return fig, int(df["test"].nunique())


def plot_sweep(
    snapshots: list[Path],
    metric: Metric = "min",
    sort: SortMode = "absolute",  # noqa: ARG001  (uniform signature, unused here)
) -> tuple[Figure, int]:
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
    fig.update_layout(height=max(500, len(df) * 22))
    return fig, len(df)


def plot_scaling(
    snapshots: list[Path],
    metric: Metric = "min",
    sort: SortMode = "absolute",  # noqa: ARG001  (uniform signature, unused here)
) -> tuple[Figure, int]:
    """Log-log time vs N for size-parametrized tests, faceted by phase."""
    import pandas as pd
    import plotly.express as px

    # Read the raw JSON so we can pull ``params`` per benchmark. ``size``
    # comes from there as a clean int — any future rename of the test id
    # format won't silently produce 0 rows. ``model`` still needs the id
    # regex because spec is stored as an unserializable repr in params.
    data = json.loads(snapshots[0].read_text())
    rows = []
    for bm in data["benchmarks"]:
        name = bm["fullname"]
        t = bm["stats"][metric]
        params = bm.get("params") or {}

        size = params.get("size")
        if not isinstance(size, int):
            # Fall back to the id regex.
            m = _SIZE_RE.match(name)
            if not m:
                continue
            size = int(m.group(3))

        m = _SIZE_RE.match(name)
        if not m:
            continue
        phase = m.group(1).split("::")[-1]
        model = m.group(2)
        rows.append({"phase": phase, "model": model, "n": size, metric: t})

    if not rows:
        raise ValueError(
            "no size-parametrized tests found (expected ``...[<model>-n=<N>]`` "
            "or a ``params.size`` int)"
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
    return fig, len(df)


RENDERERS: dict[
    PlotView, Callable[[list[Path], Metric, SortMode], tuple[Figure, int]]
] = {
    "compare": plot_compare,
    "scatter": plot_scatter,
    "sweep": plot_sweep,
    "scaling": plot_scaling,
}
