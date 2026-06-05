"""
The benchmark snapshot contract — one owner for the on-disk JSON shapes,
the test-id grammar, and the long-DataFrame loader.

Dependency-free within the package (stdlib plus a lazily-imported
pandas), so every writer (pytest-benchmark via file, :func:`memory.save`,
:mod:`benchmarks.bench`) and every reader (:mod:`benchmarks.plotting`,
:func:`memory.compare`) can sit on it without import cycles.

Two snapshot shapes, auto-detected on load:

- **timing** — ``{"benchmarks": [{"fullname": <id>, "stats": {"min":…,
  "median":…, "mean":…, "max":…}}]}`` → value in **seconds** (the shape
  pytest-benchmark writes).
- **memory** — ``{"label": <str>, "peak_mib": {<id>: <float>}}`` → value
  in **MiB**.

Test ids follow ``…[<spec>-<axis>=<value>]`` where ``<axis>`` is the sweep
dial — ``n`` for a model (size) or ``severity`` for a pattern — and ``<value>``
is the integer swept. :func:`parse_test_id` splits one into
``(phase, spec, value, axis)`` and :func:`synth_test_id` builds one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pandas as pd

Metric = Literal["min", "median", "mean", "max"]

_SIZE_RE = re.compile(r"(.*)\[([^\[\]]+?)-(\w+)=(\d+)\]")


# --- test-id grammar -------------------------------------------------------


def parse_test_id(test_id: str) -> tuple[str, str, int | None, str]:
    """
    Return ``(phase, spec, value, axis)`` for a pytest test id.

    ``value`` is the integer swept along ``axis`` (``"n"`` for a model size,
    ``"severity"`` for a pattern). Falls back to
    ``("other", "other", None, "other")`` for ids that don't match the
    ``…[<spec>-<axis>=<value>]`` parametrize shape (e.g.
    ``test_pypsa_carbon_management``).
    """
    m = _SIZE_RE.match(test_id)
    if m:
        phase = m.group(1).split("::")[-1]
        return phase, m.group(2), int(m.group(4)), m.group(3)
    return "other", "other", None, "other"


def spec_param_id(name: str, axis: str, value: object) -> str:
    """
    The ``<name>-<axis>=<value>`` fragment that fills a test id's ``[...]``.

    The single source of truth for the parametrize-id shape — pytest param ids
    (:func:`benchmarks.registry.param_ids`), the memory grid's test ids, and
    the solver-handoff ids all build on it, and :func:`parse_test_id` reads it
    back. Keep it in lock-step with ``_SIZE_RE``.
    """
    return f"{name}-{axis}={value}"


def synth_test_id(
    label: str,
    *,
    spec: str | None,
    size: int | None,
    phase: str | None,
    axis: str = "n",
) -> str:
    """
    Build a snapshot test id from optional metadata.

    With all of ``spec``/``size``/``phase`` supplied, synthesize
    ``bench::{phase}[{spec}-{axis}={size}]`` — this round-trips through
    :func:`parse_test_id` into the columns (so ``plot --view scaling`` works
    across several sweep values). ``axis`` defaults to ``"n"`` (a model size);
    pass ``axis="severity"`` for a pattern. With none of spec/size/phase
    supplied, fall back to ``label`` verbatim (lands in the ``"other"`` bucket
    — still fine for ``compare``). A partial spec is ambiguous and rejected.
    """
    if spec is not None and size is not None and phase is not None:
        return f"bench::{phase}[{spec_param_id(spec, axis, size)}]"
    if spec is not None or size is not None or phase is not None:
        raise ValueError(
            "spec, size, and phase must be given together (or all omitted)"
        )
    return label


# --- writers ---------------------------------------------------------------


def write_timing_snapshot(
    path: str | Path, entries: list[tuple[str, dict[str, float]]]
) -> Path:
    """Write the pytest-benchmark timing shape (seconds) from ``(id, stats)``."""
    data = {
        "benchmarks": [
            {"fullname": fullname, "stats": dict(stats)} for fullname, stats in entries
        ]
    }
    out = Path(path)
    out.write_text(json.dumps(data, indent=2))
    return out


def write_memory_snapshot(
    path: str | Path, label: str, peaks: dict[str, float]
) -> Path:
    """Write the memory shape (``{id: peak_mib}``)."""
    out = Path(path)
    out.write_text(json.dumps({"label": label, "peak_mib": dict(peaks)}, indent=2))
    return out


# --- readers ---------------------------------------------------------------


def load_snapshot(
    path: Path, metric: Metric = "min"
) -> tuple[str, dict[str, float], str]:
    """
    Return ``(label, {fullname: value}, unit)`` for one snapshot.

    Auto-detects the JSON shape:

    - timing (``{"benchmarks": [{"stats": {...}}]}``) → ``value`` is
      ``stats[metric]`` in **seconds**.
    - memory (``{"peak_mib": {id: float}}``) → ``value`` is the peak in
      **MiB**; ``metric`` is ignored.
    """
    data = json.loads(path.read_text())
    if "peak_mib" in data:
        return path.stem, dict(data["peak_mib"]), "MiB"
    values = {bm["fullname"]: bm["stats"][metric] for bm in data["benchmarks"]}
    return path.stem, values, "s"


def discover_snapshots() -> list[Path]:
    """
    Return JSON snapshot files under the canonical ``.benchmarks/`` tree.

    Paths are relative to cwd so they're easier to copy-paste back into
    the CLI than the absolute form would be. Used by ``compare`` / ``plot``
    to suggest available snapshots when the user passes none.
    """
    root = Path(".benchmarks")
    if not root.exists():
        return []
    return sorted(root.rglob("*.json"))


def _check_same_unit(snapshots: list[tuple[str, dict[str, float], str]]) -> str:
    """Validate that every snapshot has the same unit, return it."""
    units = {u for _, _, u in snapshots}
    if len(units) > 1:
        raise ValueError(
            f"snapshots mix units {units}; can't compare timing and memory"
        )
    return next(iter(units))


def load_long_df(
    snapshots: list[Path], metric: Metric = "min"
) -> tuple[pd.DataFrame, str]:
    """
    Return ``(df, unit)`` — one row per ``(snapshot, test_id)`` pair.

    Columns: ``snapshot``, ``test_id``, ``phase``, ``spec``, ``size``
    (``Int64``-nullable for the "other" bucket), ``axis`` (``"n"`` /
    ``"severity"`` / ``"other"``), ``value``. ``unit`` is the shared unit
    string (``"s"`` for timing, ``"MiB"`` for memory) — every loaded snapshot
    must agree.

    Every plot view downstream pivots or filters this single frame so
    test-id parsing, unit checking, and the "x snapshots, y tests"
    matrix logic all live in one place.
    """
    import pandas as pd

    raw = [load_snapshot(p, metric) for p in snapshots]
    unit = _check_same_unit(raw)
    rows = []
    for label, vals, _ in raw:
        for test_id, value in vals.items():
            phase, spec, size, axis = parse_test_id(test_id)
            rows.append(
                {
                    "snapshot": label,
                    "test_id": test_id,
                    "phase": phase,
                    "spec": spec,
                    "size": size,
                    "axis": axis,
                    "value": value,
                }
            )
    df = pd.DataFrame(rows)
    df["size"] = df["size"].astype("Int64")
    return df, unit
