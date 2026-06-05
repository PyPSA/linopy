"""
Interactive plotly views over pytest-benchmark JSON snapshots.

Four views, each returning the number of tests rendered:

- :func:`plot_compare` (2 snapshots) — bar chart of per-test delta.
- :func:`plot_scatter` (2+) — baseline cost vs ratio; the top-right corner is
  the regressed one. 3+ snapshots animate across versions.
- :func:`plot_sweep` (3+) — heatmap of per-test ratio vs the first snapshot.
- :func:`plot_scaling` (1) — cost vs the sweep dial, faceted by phase.

``metric`` selects the pytest-benchmark stat (default ``min``, the no-noise
floor; ``median``/``mean``/``max`` also accepted). plotly is imported lazily so
the rest of the suite works without it.
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


def _metric_label(metric: Metric, unit: str) -> str:
    """Title/axis label for ``metric`` — ``"peak"`` for memory, else the stat."""
    return metric if unit == "s" else "peak"


def _diverging_kwargs(midpoint: float = 0.0) -> dict:
    """green→white→red continuous colour scale centred on ``midpoint``."""
    return {
        "color_continuous_scale": ["green", "white", "red"],
        "color_continuous_midpoint": midpoint,
    }


def _axis_kwargs(unit: str) -> dict:
    """``update_xaxes`` kwargs for a given unit."""
    if unit == "s":
        return {"tickformat": ".2s", "ticksuffix": "s"}
    return {"ticksuffix": f" {unit}"}


def _hide_non_leftmost_yticks(fig: Figure, wrap: int) -> None:
    """
    Hide y-tick labels on every facet except the leftmost column.

    Plotly lays facets out left-to-right, top-to-bottom; with ``facet_col_wrap=N``
    the leftmost are at indices 0, N, 2N… Hiding the rest shows row labels once
    per row instead of at every subplot's left edge.
    """
    yaxes = []
    fig.for_each_yaxis(lambda y: yaxes.append(y))
    for idx, yaxis in enumerate(yaxes):
        if idx % wrap != 0:
            yaxis.update(showticklabels=False)


def _share_axis_labels(fig: Figure, y_label: str, x_label: str) -> None:
    """
    Replace plotly's per-facet axis titles with one shared label per axis.

    Per-facet titles get noisy across a 5+ subplot grid; this clears them and
    adds two ``paper``-coordinate annotations (left/rotated for y, bottom for x).
    Leave either label blank to skip that side.
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
    fig.update_layout(margin={"l": 90, "b": 70})  # room for the annotations


def plot_compare(
    snapshots: list[Path],
    metric: Metric = "min",
    sort: SortMode = "absolute",
    facets: FacetBy | None = None,
) -> tuple[Figure, int]:
    """
    Bar chart of per-test delta, in alphabetical test-id order.

    ``sort`` picks the bar dimension: ``absolute`` (default) plots ``b - a`` in
    the native unit, ``relative`` plots percent change. Bars stay in id order
    (related tests grouped) — use the scatter view to hunt outliers. ``facets``
    splits into subplots by ``"phase"`` (test file) or ``"spec"`` (problem); ids
    not matching ``[<spec>-n=<size>]`` land in an ``other`` facet.
    """
    import sys

    import plotly.express as px

    df_long, unit = load_long_df(snapshots[:2], metric)
    metric_label = _metric_label(metric, unit)

    labels = df_long["snapshot"].drop_duplicates().tolist()
    a_label, b_label = labels[0], labels[1]

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

    # In a facet the y-axis labels whatever varies; flat ⇒ the full test_id.
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
        **_diverging_kwargs(),
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
        fig.update_xaxes(**_axis_kwargs(unit))
    fig.update_traces(textposition="outside", cliponaxis=False)
    if facets is not None:
        # Per-facet y categories (no shared empty rows); hide non-leftmost tick
        # labels so each row's labels appear once.
        fig.update_yaxes(matches=None)
        wrap = facet_kwargs["facet_col_wrap"]
        _hide_non_leftmost_yticks(fig, wrap=wrap)
        _share_axis_labels(fig, y_label="test", x_label=x_label)
        # Keep plotly's equal-share rows: shorter facets show empty space, but a
        # manual ``domain`` override would scramble the header annotations.
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
    Baseline cost (log-x) vs candidate/baseline ratio (y) — the exploratory
    regression-hunting view.

    A point is "fix this" only in the top-right (slow *and* slower); top-left is
    a cheap test with a noisy ratio, bottom-right is already-slow-but-unchanged.
    The first snapshot is the baseline; with 3+, the rest become an
    ``animation_frame``. A dashed line at ``ratio = 1`` marks no change; colour
    encodes absolute Δ.
    """
    import numpy as np
    import plotly.express as px

    if len(snapshots) < 2:
        raise ValueError("scatter needs at least 2 snapshots (baseline + 1)")

    df_long, unit = load_long_df(snapshots, metric)
    metric_label = _metric_label(metric, unit)

    labels = df_long["snapshot"].drop_duplicates().tolist()
    baseline_label = labels[0]

    # Baseline = each test's value on the first snapshot; tests absent there or
    # with non-positive baseline (undefined ratio) are dropped.
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
    # Fixed ranges so the animation doesn't jitter; pad to avoid edge clipping.
    x_lo, x_hi = df["baseline_time"].min(), df["baseline_time"].max()
    # y-range centred symmetrically on 1.0 so regressions and improvements read
    # equally; the larger side sets the window width.
    y_lo, y_hi = df["ratio"].min(), df["ratio"].max()
    max_dist = max(abs(1.0 - y_lo), abs(y_hi - 1.0), 0.05)
    pad_y = max(0.05, max_dist * 0.05)
    y_range = [1.0 - max_dist - pad_y, 1.0 + max_dist + pad_y]

    # Clip the colour scale to the p95 absolute Δ so one huge regression doesn't
    # wash the rest to white; outliers saturate at the bound.
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
        **_diverging_kwargs(),
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
    metric_label = _metric_label(metric, unit)
    versions = df_long["snapshot"].drop_duplicates().tolist()
    baseline_label = versions[0]

    # Pivot to absolutes, drop tests missing the baseline, divide by it for ratios.
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
        **_diverging_kwargs(1.0),
        aspect="auto",
        title=f"{metric_label} ratio relative to baseline ({versions[0]})",
        labels={"x": "version", "y": "test", "color": "ratio"},
        text_auto=".2f",
    )
    # Absolute values as customdata so hover shows both ratio and value.
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


# Per sweep axis: (x label, log-scaled?). Size is multiplicative → log; severity
# is a 0–100% dial → linear.
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

    Model ``size`` (``axis="n"``) scales multiplicatively → log-log; pattern
    ``severity`` is a 0–100 dial → linear; a mixed snapshot falls back to linear
    x. The x-axis label comes from the data's ``axis`` column.
    """
    import plotly.express as px

    df_long, unit = load_long_df(snapshots[:1], metric)
    metric_label = _metric_label(metric, unit)
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
    # Log only suits multiplicative size sweeps that stay > 0.
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
