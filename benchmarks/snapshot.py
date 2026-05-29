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

Test ids follow ``…[<model>-n=<size>]``; :func:`parse_test_id` splits one
into ``(phase, model, size)`` and :func:`synth_test_id` builds one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pandas as pd

Metric = Literal["min", "median", "mean", "max"]

_SIZE_RE = re.compile(r"(.*)\[([^\[\]]+?)-n=(\d+)\]")


# --- test-id grammar -------------------------------------------------------


def parse_test_id(test_id: str) -> tuple[str, str, int | None]:
    """
    Return ``(phase, model, size)`` for a pytest test id.

    Falls back to ``("other", "other", None)`` for ids that don't match
    the ``benchmarks/test_<phase>.py::test_<phase>[<model>-n=<size>]``
    parametrize shape (e.g. ``test_pypsa_carbon_management``).
    """
    m = _SIZE_RE.match(test_id)
    if m:
        phase = m.group(1).split("::")[-1]
        return phase, m.group(2), int(m.group(3))
    return "other", "other", None


def synth_test_id(
    label: str, *, model: str | None, size: int | None, phase: str | None
) -> str:
    """
    Build a snapshot test id from optional metadata.

    With all of ``model``/``size``/``phase`` supplied, synthesize
    ``bench::{phase}[{model}-n={size}]`` — this round-trips through
    :func:`parse_test_id` into the three columns (so ``plot --view
    scaling`` works across several sizes). With none supplied, fall back
    to ``label`` verbatim (lands in the ``"other"`` bucket — still fine
    for ``compare``). A partial spec is ambiguous and rejected.
    """
    given = (model is not None, size is not None, phase is not None)
    if all(given):
        return f"bench::{phase}[{model}-n={size}]"
    if any(given):
        raise ValueError(
            "model, size, and phase must be given together (or all omitted)"
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

    Columns: ``snapshot``, ``test_id``, ``phase``, ``model``, ``size``
    (``Int64``-nullable for the "other" bucket), ``value``. ``unit`` is
    the shared unit string (``"s"`` for timing, ``"MiB"`` for memory)
    — every loaded snapshot must agree.

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
            phase, model, size = parse_test_id(test_id)
            rows.append(
                {
                    "snapshot": label,
                    "test_id": test_id,
                    "phase": phase,
                    "model": model,
                    "size": size,
                    "value": value,
                }
            )
    df = pd.DataFrame(rows)
    df["size"] = df["size"].astype("Int64")
    return df, unit
