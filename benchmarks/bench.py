"""
Ad-hoc benchmarking of arbitrary callables on the *current* linopy tree.

Where the pytest suite measures the fixed registry grid and ``sweep``
measures across installed linopy versions, ``bench`` is for the
interactive middle: time or memory-profile any callable — a registry
builder, a phase verb applied to a model you built by hand, or a one-off
lambda — get a result object back, and either inspect it as a DataFrame
or drop it into a snapshot the existing ``plot`` / ``compare`` machinery
already understands::

    from benchmarks import bench, REGISTRY

    r = bench.time(REGISTRY["basic"].build, 100)
    r                                  # rich repr in a notebook
    r.to_snapshot("a.json", spec="basic", size=100, phase="build")

    bench.compare({"v1": f1, "v2": f2}).to_snapshot("cmp.json")

This plugs into the *output* side of the pipeline (snapshot JSON read by
``snapshot.load_long_df``), not into ``sweep``: a sweep runs pytest inside
per-version venvs as subprocesses, so it can only measure importable
registry models — an in-process callable can't cross that boundary. To
sweep a custom model across versions, promote it to ``benchmarks/models/``.

**Methodology.** Timing is built on :class:`timeit.Timer`: an
``autorange`` calibration picks the inner iteration count (so timer
resolution doesn't dominate fast callables), then the per-iteration time
is sampled across rounds with the suite's min-of-N convention (the
fastest sample approximates the no-noise floor). It is *not*
pytest-benchmark's calibrated timer, so absolute numbers are not
interchangeable with suite snapshots — compare ``bench`` to ``bench`` and
suite to suite.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, stdev
from timeit import Timer
from typing import TYPE_CHECKING, Any, Literal

from benchmarks.snapshot import (
    parse_test_id,
    synth_test_id,
    write_memory_snapshot,
    write_timing_snapshot,
)

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "MemoryResult",
    "ResultSet",
    "TimingResult",
    "compare",
    "memory",
    "time",
]

# Floor / cap on the auto-tuned round count when ``rounds`` is unset.
# The floor guarantees a meaningful min-of-N even for slow callables that
# blow past ``min_time`` in one shot; the cap stops a microsecond callable
# from spinning forever.
_ROUND_FLOOR = 5
_ROUND_CAP = 10_000


def _fn_name(fn: Callable[..., object]) -> str:
    """Best-effort label for a callable (``functools.partial`` has no name)."""
    return getattr(fn, "__name__", None) or repr(fn)


def _row(test_id: str, value: float) -> dict[str, object]:
    """One ``load_long_df``-shaped row for an in-process result."""
    phase, spec, size, axis = parse_test_id(test_id)
    return {
        "snapshot": test_id,
        "test_id": test_id,
        "phase": phase,
        "spec": spec,
        "size": size,
        "axis": axis,
        "value": value,
    }


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a DataFrame with the exact column set/dtype of ``load_long_df``."""
    import pandas as pd

    df = pd.DataFrame(
        rows,
        columns=["snapshot", "test_id", "phase", "spec", "size", "axis", "value"],
    )
    df["size"] = df["size"].astype("Int64")
    return df


# --- Result types ----------------------------------------------------------


@dataclass(frozen=True)
class TimingResult:
    """One timed callable: per-round stats with ``min`` as the headline."""

    label: str
    stats: dict[str, float]
    unit: Literal["s"] = "s"

    def to_snapshot(
        self,
        path: str | Path,
        *,
        spec: str | None = None,
        size: int | None = None,
        phase: str | None = None,
    ) -> Path:
        """Write a pytest-benchmark-shaped timing snapshot (seconds)."""
        test_id = synth_test_id(self.label, spec=spec, size=size, phase=phase)
        return write_timing_snapshot(path, [(test_id, dict(self.stats))])

    def to_df(self) -> pd.DataFrame:
        """``load_long_df``-shaped frame (one row, ``value`` = min seconds)."""
        return _frame([_row(self.label, self.stats["min"])])

    def __repr__(self) -> str:
        return (
            f"TimingResult({self.label!r}, min={self.stats['min']:.4g}s, "
            f"rounds={int(self.stats['rounds'])}x{int(self.stats.get('iterations', 1))})"
        )

    def _repr_html_(self) -> str:
        rows = [
            ("min", f"{self.stats['min']:.4g} s"),
            ("median", f"{self.stats['median']:.4g} s"),
            ("mean", f"{self.stats['mean']:.4g} s"),
            ("max", f"{self.stats['max']:.4g} s"),
            ("stddev", f"{self.stats['stddev']:.4g} s"),
            ("rounds", int(self.stats["rounds"])),
            ("iterations", int(self.stats.get("iterations", 1))),
        ]
        return _html_table("TimingResult", self.label, rows)


@dataclass(frozen=True)
class MemoryResult:
    """One memory-profiled callable: peak RSS in MiB."""

    label: str
    peak_mib: float
    unit: Literal["MiB"] = "MiB"

    def to_snapshot(
        self,
        path: str | Path,
        *,
        spec: str | None = None,
        size: int | None = None,
        phase: str | None = None,
    ) -> Path:
        """Write a memory.py-shaped snapshot (peak MiB)."""
        test_id = synth_test_id(self.label, spec=spec, size=size, phase=phase)
        return write_memory_snapshot(path, self.label, {test_id: self.peak_mib})

    def to_df(self) -> pd.DataFrame:
        """``load_long_df``-shaped frame (one row, ``value`` = peak MiB)."""
        return _frame([_row(self.label, self.peak_mib)])

    def __repr__(self) -> str:
        return f"MemoryResult({self.label!r}, peak={self.peak_mib:.1f} MiB)"

    def _repr_html_(self) -> str:
        return _html_table(
            "MemoryResult", self.label, [("peak", f"{self.peak_mib:.1f} MiB")]
        )


@dataclass(frozen=True)
class ResultSet:
    """
    Several results of one kind (all timing, or all memory).

    ``to_snapshot`` writes every result into a single file keyed by its
    label — the natural "compare these N variants" case. For
    size-parametrized ``scaling`` plots, write each result individually
    with ``spec``/``size``/``phase`` instead.
    """

    results: list[TimingResult | MemoryResult] = field(default_factory=list)
    unit: Literal["s", "MiB"] = "s"

    def to_snapshot(self, path: str | Path) -> Path:
        """Write all results into one snapshot, each keyed by its label."""
        if self.unit == "s":
            return write_timing_snapshot(
                path,
                [
                    (r.label, dict(r.stats))
                    for r in self.results
                    if isinstance(r, TimingResult)
                ],
            )
        peaks = {
            r.label: r.peak_mib for r in self.results if isinstance(r, MemoryResult)
        }
        return write_memory_snapshot(path, "compare", peaks)

    def to_df(self) -> pd.DataFrame:
        """Concatenate the per-result frames (shares ``load_long_df`` columns)."""
        import pandas as pd

        return pd.concat([r.to_df() for r in self.results], ignore_index=True)

    def __repr__(self) -> str:
        labels = ", ".join(r.label for r in self.results)
        return f"ResultSet(unit={self.unit!r}, [{labels}])"

    def _repr_html_(self) -> str:
        rows = [
            (
                r.label,
                f"{r.stats['min']:.4g} s"
                if isinstance(r, TimingResult)
                else f"{r.peak_mib:.1f} MiB",
            )
            for r in self.results
        ]
        return _html_table("ResultSet", self.unit, rows)


def _html_table(kind: str, header: str, rows: Sequence[tuple[str, object]]) -> str:
    """Compact two-column Jupyter table, mirroring ``ModelSpec._repr_html_``."""
    body = "".join(
        f"<tr><th style='text-align:left;padding-right:1em'>{k}</th><td>{v}</td></tr>"
        for k, v in rows
    )
    return (
        f"<b>{kind}</b> <code>{header}</code>"
        f"<table style='font-size:90%'>{body}</table>"
    )


# --- Entry points ----------------------------------------------------------


def time(
    fn: Callable[..., object],
    /,
    *args: object,
    rounds: int | None = None,
    warmup: int = 1,
    min_time: float = 0.5,
    label: str | None = None,
    **kwargs: object,
) -> TimingResult:
    """
    Time ``fn(*args, **kwargs)`` and return a :class:`TimingResult`.

    Built on :class:`timeit.Timer`: an ``autorange`` calibration first
    picks the inner iteration count so timer resolution doesn't dominate
    for fast callables (the bespoke "one call per round" loop this
    replaced was unstable in exactly that regime). Each round then runs
    that many calibrated iterations; the per-iteration time is the
    sample. ``warmup`` rounds are discarded to prime caches.

    With ``rounds`` set, run exactly that many rounds; otherwise
    auto-tune — keep going until cumulative timed wall-clock reaches
    ``min_time`` (floor of 5 rounds, hard cap). The headline number is
    ``stats["min"]``; ``stats["iterations"]`` records the calibrated
    inner count.

    This is *not* pytest-benchmark's calibrated timer — ``bench`` numbers
    are only comparable to other ``bench`` numbers, not to suite
    snapshots.
    """
    timer = Timer(lambda: fn(*args, **kwargs))

    # Calibrate inner iterations so a single round is long enough that
    # ``perf_counter`` granularity is negligible (timeit targets ~0.2 s).
    number, _ = timer.autorange()

    for _ in range(max(0, warmup)):
        timer.timeit(number)

    samples: list[float] = []  # per-iteration seconds
    if rounds is not None:
        samples = [
            t / number for t in timer.repeat(repeat=max(1, rounds), number=number)
        ]
    else:
        total = 0.0
        while True:
            t = timer.timeit(number)
            samples.append(t / number)
            total += t
            if len(samples) >= _ROUND_FLOOR and total >= min_time:
                break
            if len(samples) >= _ROUND_CAP:
                break

    stats = {
        "min": min(samples),
        "max": max(samples),
        "mean": mean(samples),
        "median": median(samples),
        "stddev": stdev(samples) if len(samples) > 1 else 0.0,
        "rounds": float(len(samples)),
        "iterations": float(number),
    }
    return TimingResult(label=label or _fn_name(fn), stats=stats)


def memory(
    fn: Callable[..., object],
    /,
    *args: object,
    repeats: int = 1,
    label: str | None = None,
    **kwargs: object,
) -> MemoryResult:
    """
    Peak-RSS profile ``fn(*args, **kwargs)`` and return a :class:`MemoryResult`.

    Thin wrapper over :func:`benchmarks.memory.measure_peak`; ``repeats > 1``
    keeps the minimum peak. Raises on Windows (no ``memray``).
    """
    from benchmarks.memory import measure_peak

    peak = measure_peak(lambda: fn(*args, **kwargs), repeats=repeats)
    return MemoryResult(label=label or _fn_name(fn), peak_mib=peak)


def compare(
    cases: dict[str, Callable[[], object]],
    *,
    kind: Literal["time", "memory"] = "time",
    **opts: Any,
) -> ResultSet:
    """
    Run each zero-arg callable in ``cases`` and collect a :class:`ResultSet`.

    ``kind`` selects timing (default) or memory; ``opts`` are forwarded to
    :func:`time` / :func:`memory` (e.g. ``rounds=``, ``repeats=``). The
    dict key becomes each case's label.
    """
    if kind == "time":
        results: list[TimingResult | MemoryResult] = [
            time(fn, label=name, **opts) for name, fn in cases.items()
        ]
        return ResultSet(results=results, unit="s")
    if kind == "memory":
        results = [memory(fn, label=name, **opts) for name, fn in cases.items()]
        return ResultSet(results=results, unit="MiB")
    raise ValueError(f"kind must be 'time' or 'memory', got {kind!r}")
