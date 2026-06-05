"""
Interactive plotly views over pytest-benchmark JSON snapshots.

Three opinionated views, all returning the number of tests rendered:

- :func:`plot_compare` (2 snapshots) — sorted-by-delta bar chart.
- :func:`plot_sweep` (3+ snapshots) — heatmap of per-test ratio
  relative to the first snapshot. Useful for cross-version sweeps.
- :func:`plot_scaling` (1 snapshot) — cost vs the sweep dial for
  parametrized tests, faceted by phase. Log-log for model ``size``;
  linear for pattern ``severity`` (0–100).

All three accept a ``metric`` argument selecting which pytest-benchmark
stat drives the plot. Default is ``min`` — the fastest observed sample
approximates the no-noise floor (GC, scheduling, cache thrash can only
add time). ``median`` is more robust to a single weirdly-fast warmup
round; ``mean`` and ``max`` are also accepted.

plotly is imported lazily by the dispatcher so the rest of the benchmark
suite still works without it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from benchmarks.snapshot import Metric, load_long_df

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

PlotView = Literal["compare", "scatter", "sweep", "scaling"]
SortMode = Literal["absolute", "relative"]
FacetBy = Literal["phase", "spec"]


def _axis_kwargs(unit: str) -> dict:
    """Return ``update_xaxes`` kwargs for a given unit."""
    if unit == "s":
        return {"tickformat": ".2s", "ticksuffix": "s"}
    return {"ticksuffix": f" {unit}"}


def _hide_non_leftmost_yticks(fig: Figure, wrap: int) -> None:
    """
    Hide y-axis tick labels on every facet except the leftmost column.

    Plotly express lays facets out left-to-right, top-to-bottom: with
    ``facet_col_wrap=N`` the leftmost facets are at indices 0, N, 2N…
    Hiding tick labels on the rest keeps the row labels visible only
    once per row instead of repeating at every subplot's left edge.
    """
    yaxes = []
    fig.for_each_yaxis(lambda y: yaxes.append(y))
    for idx, yaxis in enumerate(yaxes):
        if idx % wrap != 0:
            yaxis.update(showticklabels=False)


def _share_axis_labels(fig: Figure, y_label: str, x_label: str) -> None:
    """
    Replace per-facet axis titles with one shared label per axis.

    Plotly express renders the x/y titles on every facet by default,
    which is noisy when faceting wraps a 5+ subplot grid. This clears
    them and adds two ``paper``-coordinate annotations: one on the
    left (rotated) for ``y_label``, one on the bottom for ``x_label``.
    Leave either blank to skip that side.
    """
    fig.for_each_yaxis(lambda yaxis: yaxis.update(title_text=""))
    fig.for_each_xaxis(lambda xaxis: xaxis.update(title_text=""))
    if y_label:
        fig.add_annotation(
            text=y_label,
            xref="paper",
            yref="paper",
            x=-0.05,
            y=0.5,
            textangle=-90,
            showarrow=False,
            font={"size": 13},
        )
    if x_label:
        fig.add_annotation(
            text=x_label,
            xref="paper",
            yref="paper",
            x=0.5,
            y=-0.08,
            showarrow=False,
            font={"size": 13},
        )
    # Give the annotations room.
    fig.update_layout(margin={"l": 90, "b": 70})


def plot_compare(
    snapshots: list[Path],
    metric: Metric = "min",
    sort: SortMode = "absolute",
    facets: FacetBy | None = None,
) -> tuple[Figure, int]:
    """
    Bar chart of delta per test, in alphabetical test-id order.

    ``sort`` chooses the bar *dimension*: ``absolute`` (default) plots
    ``b - a`` in the data's native unit; ``relative`` plots the percent
    change. Bars are not reordered by magnitude — alphabetical ids keep
    related tests visually grouped. Use the scatter view for hunting
    outliers.

    ``facets`` splits the chart into subplots:

    - ``None`` (default): one flat bar chart.
    - ``"phase"``: facet by the test file (``test_build``,
      ``test_lp_write``, ...). Best for "everything in this phase moved
      together?".
    - ``"spec"``: facet by the spec name (``basic``, ``knapsack``, ...).
      Best for "what happened across all the basic-sized variants?".

    Tests whose IDs don't match the standard ``[<spec>-n=<size>]``
    parametrize shape (e.g. PyPSA carbon-management) land in an
    ``other`` facet.
    """
    import sys

    import plotly.express as px

    df_long, unit = load_long_df(snapshots[:2], metric)
    metric_label = metric if unit == "s" else "peak"

    labels = df_long["snapshot"].drop_duplicates().tolist()
    a_label, b_label = labels[0], labels[1]

    # Pivot to wide: one row per test, baseline + candidate as columns,
    # phase / spec / size carried through. Then compute deltas
    # vectorised — no per-row dict construction.
    wide = (
        df_long.pivot(
            index=["test_id", "phase", "spec", "size", "axis"],
            columns="snapshot",
            values="value",
        )
        .reset_index()
        .rename_axis(columns=None)
    )
    only_a = wide[wide[a_label].notna() & wide[b_label].isna()]
    only_b = wide[wide[a_label].isna() & wide[b_label].notna()]
    df = wide.dropna(subset=[a_label, b_label]).copy()
    if df.empty:
        raise ValueError("no tests in common between the two snapshots")
    if len(only_a) or len(only_b):
        print(
            f"compare: {len(only_a)} test(s) only in {a_label}, "
            f"{len(only_b)} only in {b_label} (intersection: {len(df)}).",
            file=sys.stderr,
        )

    df["delta_abs"] = df[b_label] - df[a_label]
    df["delta_pct"] = (df["delta_abs"] / df[a_label]) * 100.0
    df = df.sort_values("test_id").reset_index(drop=True)
    x_col = "delta_abs" if sort == "absolute" else "delta_pct"

    if sort == "absolute":
        x_label = f"{metric_label} delta ({unit})"
        text_fmt = ".2s" if unit == "s" else ".2f"
    else:
        x_label = f"{metric_label} delta %"
        text_fmt = ".1f"

    direction = "slower" if unit == "s" else "more memory"
    title = (
        f"{metric_label} delta ({sort}): {a_label} → {b_label} (positive = {direction})"
    )
    if len(only_a) or len(only_b):
        title += (
            f"<br><sub>{len(only_a)} only in {a_label}, "
            f"{len(only_b)} only in {b_label}</sub>"
        )

    # Inside a facet the y-axis labels whatever *varies* — drop the
    # facetted dimension from the label, keep the rest. Flat ⇒ the full
    # test_id so each bar is self-identifying.
    facet_kwargs: dict = {}
    if facets is None:
        y_col = "test_id"
    else:
        varying = "spec" if facets == "phase" else "phase"
        size_str = df["size"].astype("Int64").astype(str)
        df["_short"] = df[varying] + "-" + df["axis"] + "=" + size_str
        other_mask = df["phase"] == "other"
        df.loc[other_mask, "_short"] = (
            df.loc[other_mask, "test_id"].str.split("::").str[-1]
        )
        y_col = "_short"
        facet_kwargs = {"facet_col": facets}
        facet_kwargs["facet_col_wrap"] = 2 if facets == "phase" else 3

    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        orientation="h",
        color=x_col,
        color_continuous_scale=["green", "white", "red"],
        color_continuous_midpoint=0,
        title=title,
        labels={x_col: x_label, y_col: ""},
        text_auto=text_fmt,
        hover_data={
            "test_id": True,
            a_label: ":.4g",
            b_label: ":.4g",
            "delta_abs": ":.4g",
            "delta_pct": ":.2f",
        },
        **facet_kwargs,
    )
    if sort == "absolute":
        # SI-prefixed time on the x-axis (e.g. 24 ms, 2.4 ms, 240 µs) for
        # timing snapshots; plain MiB for memory.
        fig.update_xaxes(**_axis_kwargs(unit))
    # Render the value text outside the bar (default is inside) so the
    # number stays readable even when a bar is very short.
    fig.update_traces(textposition="outside", cliponaxis=False)
    if facets is not None:
        # Each facet keeps its own y category list (no shared rows full
        # of empty bars), but we hide tick labels on non-leftmost facets
        # within each row so labels appear once per row.
        fig.update_yaxes(matches=None)
        wrap = facet_kwargs["facet_col_wrap"]
        _hide_non_leftmost_yticks(fig, wrap=wrap)
        _share_axis_labels(fig, y_label="test", x_label=x_label)
        # Keep plotly's default equal-share row layout: shorter facets show
        # empty space below their bars, but the header annotations stay put
        # (a manual ``domain`` override would scramble them).
        rows_per_facet = df.groupby(facets)[y_col].nunique().max()
        n_wrap_rows = (df[facets].nunique() + wrap - 1) // wrap
        height = max(500, int(n_wrap_rows * rows_per_facet * 24) + 100)
    else:
        height = max(500, len(df) * 22)
    fig.update_layout(height=height, showlegend=False)
    return fig, len(df)


def plot_scatter(
    snapshots: list[Path],
    metric: Metric = "min",
    sort: SortMode = "absolute",  # noqa: ARG001  (uniform signature, unused here)
    facets: FacetBy | None = None,
) -> tuple[Figure, int]:
    """
    Two-axis scatter — baseline cost on log-x, ratio on y.

    Designed as the single best exploratory plot for regression hunting
    across tests of wildly different magnitudes: a point lights up as
    "fix this" only if it sits in the top-right corner — slow tests
    that got slower. Top-left (big ratio, tiny absolute) is a cheap
    test with noisy ratio swings — not a real change. Bottom-right (big
    absolute, tiny ratio) is already-slow-but-unchanged. The combined
    position resolves the tension that pure relative or pure absolute
    sort each blind-spot.

    The first snapshot is the baseline. With 2 snapshots, a static
    scatter is drawn; with 3+, every subsequent snapshot becomes an
    ``animation_frame`` — use the slider / play button to step through
    versions and watch points drift across releases.

    A horizontal reference at ``ratio = 1`` makes "no change" trivial
    to see; the colour encodes absolute Δ as a third channel.
    """
    import numpy as np
    import plotly.express as px

    if len(snapshots) < 2:
        raise ValueError("scatter needs at least 2 snapshots (baseline + 1)")

    df_long, unit = load_long_df(snapshots, metric)
    metric_label = metric if unit == "s" else "peak"

    labels = df_long["snapshot"].drop_duplicates().tolist()
    baseline_label = labels[0]

    # Attach the baseline value to every row via a per-test groupby (each
    # test's baseline = its value on the first snapshot). Tests with no
    # baseline row (only in non-baseline snapshots) are dropped. Tests
    # with non-positive baseline are dropped because the ratio is
    # undefined for them.
    baseline_vals = df_long.loc[
        df_long["snapshot"] == baseline_label, ["test_id", "value"]
    ].rename(columns={"value": "baseline_time"})
    df = df_long.merge(baseline_vals, on="test_id", how="inner")
    df = df[df["baseline_time"] > 0].copy()
    if df.empty:
        raise ValueError(
            f"no tests in common between baseline ({baseline_label}) "
            "and any of the other snapshots"
        )

    df = df.rename(columns={"snapshot": "version", "value": "candidate_time"})
    df["ratio"] = df["candidate_time"] / df["baseline_time"]
    df["delta_abs"] = df["candidate_time"] - df["baseline_time"]
    df["delta_pct"] = df["delta_abs"] / df["baseline_time"] * 100.0
    df = df.rename(columns={"test_id": "test"})
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
        extra["category_orders"] = {"version": labels}
    if facets is not None:
        extra["facet_col"] = facets
        extra["facet_col_wrap"] = 2 if facets == "phase" else 3

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
            f"{metric_label} scatter vs baseline ({baseline_label}) — "
            "top-right = the regressed corner"
        ),
        labels={
            "baseline_time": f"baseline {metric_label} ({unit}, log scale)",
            "ratio": f"{metric_label} ratio  (candidate / baseline)",
            "candidate_time": "candidate",
            "delta_abs": f"Δ ({unit}, p95-clipped)",
        },
        **extra,
    )
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="grey", annotation_text="no change"
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color="DarkSlateGrey")))
    if facets is not None:
        _share_axis_labels(
            fig,
            y_label=f"{metric_label} ratio (candidate / baseline)",
            x_label=f"baseline {metric_label} ({unit}, log scale)",
        )
    fig.update_layout(height=600)
    return fig, int(df["test"].nunique())


def plot_sweep(
    snapshots: list[Path],
    metric: Metric = "min",
    sort: SortMode = "absolute",  # noqa: ARG001  (uniform signature, unused here)
    facets: FacetBy | None = None,  # noqa: ARG001  (uniform signature, unused here)
) -> tuple[Figure, int]:
    """Heatmap of per-test ratio relative to the first snapshot."""
    import plotly.express as px

    df_long, unit = load_long_df(snapshots, metric)
    metric_label = metric if unit == "s" else "peak"
    versions = df_long["snapshot"].drop_duplicates().tolist()
    baseline_label = versions[0]

    # Pivot absolutes (rows=tests, cols=versions), then drop tests with
    # no baseline reading and divide every column by the baseline column
    # to get ratios in one shot.
    abs_df = df_long.pivot(index="test_id", columns="snapshot", values="value").reindex(
        columns=versions
    )
    abs_df = abs_df.dropna(subset=[baseline_label])
    if abs_df.empty:
        raise ValueError(f"no overlap with baseline snapshot {baseline_label}")
    df = abs_df.div(abs_df[baseline_label], axis=0)
    abs_df.index.name = "test"
    df.index.name = "test"

    fig = px.imshow(
        df,
        color_continuous_scale=["green", "white", "red"],
        color_continuous_midpoint=1.0,
        aspect="auto",
        title=f"{metric_label} ratio relative to baseline ({versions[0]})",
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
            f"{metric_label}: %{{customdata:.4g}}{unit}"
            "<extra></extra>"
        ),
    )
    fig.update_layout(height=max(500, len(df) * 22))
    return fig, len(df)


# How each sweep axis renders: (x-axis label, log-scaled?). Size is
# multiplicative → log-log; severity is a 0–100 % dial → linear.
_AXIS_DISPLAY: dict[str, tuple[str, bool]] = {
    "n": ("n", True),
    "severity": ("severity (%)", False),
}


def plot_scaling(
    snapshots: list[Path],
    metric: Metric = "min",
    sort: SortMode = "absolute",  # noqa: ARG001  (uniform signature, unused here)
    facets: FacetBy | None = None,  # noqa: ARG001  (uniform signature, unused here)
) -> tuple[Figure, int]:
    """
    Cost vs the sweep dial for parametrized tests, faceted by phase.

    Handles both axes the registry sweeps: model ``size`` (``axis="n"``) and
    pattern ``severity``. Models scale multiplicatively, so a model-only
    snapshot is drawn log-log; a severity sweep is linear (it starts at 0 and
    is a 0–100 percentage, so a log x-axis would be meaningless). A mixed
    snapshot falls back to linear x. The x-axis label is taken from the data's
    ``axis`` column rather than hard-coded.
    """
    import plotly.express as px

    df_long, unit = load_long_df(snapshots[:1], metric)
    metric_label = metric if unit == "s" else "peak"
    df = (
        df_long.dropna(subset=["size"])
        .rename(columns={"value": metric})
        .sort_values(["phase", "spec", "size"])
    )
    if df.empty:
        raise ValueError(
            "no parametrized tests found (expected ``...[<spec>-<axis>=<N>]`` ids)"
        )

    axes = sorted(df["axis"].unique())
    axis = axes[0] if len(axes) == 1 else "sweep value"
    x_label, log_axis = _AXIS_DISPLAY.get(axis, (axis, False))
    # Log only makes sense for multiplicative size sweeps that stay > 0.
    log_x = log_axis and bool((df["size"] > 0).all())

    metric_word = {"min": "minimum", "max": "maximum"}.get(metric, metric)
    fig = px.line(
        df,
        x="size",
        y=metric,
        color="spec",
        facet_col="phase",
        facet_col_wrap=3,
        log_x=log_x,
        log_y=log_x,
        markers=True,
        labels={"size": x_label, metric: f"{unit} ({metric_word})"},
        title=(f"Scaling: {metric_label} ({unit}) vs {x_label} ({snapshots[0].stem})"),
    )
    fig.update_layout(height=max(400, ((df["phase"].nunique() + 2) // 3) * 350))
    return fig, len(df)


RENDERERS: dict[
    PlotView,
    Callable[[list[Path], Metric, SortMode, FacetBy | None], tuple[Figure, int]],
] = {
    "compare": plot_compare,
    "scatter": plot_scatter,
    "sweep": plot_sweep,
    "scaling": plot_scaling,
}
