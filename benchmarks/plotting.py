"""
Interactive plotly views over pytest-benchmark JSON snapshots.

Three opinionated views, all returning the number of tests rendered:

- :func:`plot_compare` (2 snapshots) — sorted-by-delta bar chart.
- :func:`plot_sweep` (3+ snapshots) — heatmap of per-test median ratio
  relative to the first snapshot. Useful for cross-version sweeps.
- :func:`plot_scaling` (1 snapshot) — log-log median vs ``n`` for
  size-parametrized tests, faceted by phase.

plotly is imported lazily by the dispatcher so the rest of the benchmark
suite still works without it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

PlotView = Literal["compare", "sweep", "scaling"]

_SIZE_RE = re.compile(r"(.*)\[([^\[\]]+?)-n=(\d+)\]")


def _load_snapshot(path: Path) -> tuple[str, dict[str, float]]:
    """Return ``(label, {fullname: median_seconds})`` for one snapshot."""
    data = json.loads(path.read_text())
    medians = {bm["fullname"]: bm["stats"]["median"] for bm in data["benchmarks"]}
    return path.stem, medians


def plot_compare(snapshots: list[Path], output: Path) -> int:
    """Bar chart of relative median delta per test, sorted by magnitude."""
    import pandas as pd
    import plotly.express as px

    (a_label, a_med), (b_label, b_med) = (
        _load_snapshot(snapshots[0]),
        _load_snapshot(snapshots[1]),
    )
    common = sorted(set(a_med) & set(b_med))
    if not common:
        raise ValueError("no tests in common between the two snapshots")

    rows = [
        {
            "test": name,
            a_label: a_med[name],
            b_label: b_med[name],
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
        text_auto=".1f",
        hover_data={
            a_label: ":.4g",
            b_label: ":.4g",
            "delta_pct": ":.2f",
        },
    )
    fig.update_layout(height=max(400, len(df) * 14), showlegend=False)
    fig.write_html(output)
    return len(df)


def plot_sweep(snapshots: list[Path], output: Path) -> int:
    """Heatmap of per-test median ratio relative to the first snapshot."""
    import pandas as pd
    import plotly.express as px

    loaded = [_load_snapshot(p) for p in snapshots]
    versions = [label for label, _ in loaded]
    baseline = loaded[0][1]
    all_tests = sorted(set().union(*[set(med) for _, med in loaded]))

    ratios: dict[str, list[float | None]] = {}
    absolutes: dict[str, list[float | None]] = {}
    for test in all_tests:
        base = baseline.get(test)
        if not base:
            continue
        ratios[test] = []
        absolutes[test] = []
        for _, med in loaded:
            t = med.get(test)
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
        title=f"Median ratio relative to baseline ({versions[0]})",
        labels={"x": "version", "y": "test", "color": "ratio"},
        text_auto=".2f",
    )
    # Inject absolute medians as customdata so hover shows both.
    fig.update_traces(
        customdata=abs_df.values,
        hovertemplate=(
            "test: %{y}<br>"
            "version: %{x}<br>"
            "ratio: %{z:.3f}<br>"
            "median: %{customdata:.4g}s"
            "<extra></extra>"
        ),
    )
    fig.update_layout(height=max(400, len(df) * 14))
    fig.write_html(output)
    return len(df)


def plot_scaling(snapshots: list[Path], output: Path) -> int:
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
        raise ValueError(
            "no size-parametrized tests found (expected ``...[<model>-n=<N>]``)"
        )

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


RENDERERS: dict[PlotView, Callable[[list[Path], Path], int]] = {
    "compare": plot_compare,
    "sweep": plot_sweep,
    "scaling": plot_scaling,
}
