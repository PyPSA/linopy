"""
Measure and compare peak memory across the registry × phase grid.

Each measurement runs the phase work inside ``memray.Tracker`` with model
construction *outside* it, so the peak reflects only the phase::

    m = spec.build(size)            # setup, not tracked
    with memray.Tracker(bin_path):
        wrapper(m)                  # tracked

``save(label, ...)`` / ``compare(a, b)`` back the ``python -m benchmarks memory``
CLI. Results land in ``.benchmarks/memory/`` as JSON keyed by full pytest-style
test ids, so cross-snapshot diffs line up with the timing snapshots.

The per-phase peaks are *marginal* (each tracker sees only its own phase's
allocations), so the end-to-end OOM ceiling can't be recovered from them: the
opt-in ``pipeline`` phase (``--phase pipeline``) instead measures
build → matrices → to_lp under one tracker, keyed by a bare
``pipeline[<spec>-<axis>=<value>]`` id.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.snapshot import spec_param_id, write_memory_snapshot

if TYPE_CHECKING:
    from benchmarks.registry import BenchSpec


def _require_memray() -> None:
    """
    Raise if memory measurement isn't supported on this platform.

    Called at the top of every entry point that actually measures
    (:func:`measure_peak`, :func:`run_phase`, :func:`save`) rather than
    at import time, so the module imports cleanly everywhere — notably
    ``benchmarks.bench`` reuses :func:`measure_peak` and must import on
    Windows. Only *measuring* fails there, with the original message.
    """
    if platform.system() == "Windows":
        raise RuntimeError(
            "memory measurement requires ``memray`` which is not available on "
            "Windows. Run memory benchmarks on Linux or macOS."
        )


RESULTS_DIR = Path(".benchmarks/memory")

# Default phases for ``save`` — each measures one phase's marginal allocation.
MEMORY_PHASES: tuple[str, ...] = (
    "build",
    "matrices",
    "to_lp",
    "to_netcdf",
    "from_netcdf",
    "to_solver",
)

# ``pipeline`` re-runs build→matrices→to_lp in one tracker for the end-to-end
# peak, so it duplicates their work and is *not* in the default set; request it
# standalone with ``--phase pipeline``. This is the full set ``--phase`` accepts.
ALL_MEMORY_PHASES: tuple[str, ...] = (*MEMORY_PHASES, "pipeline")


def _phase_tag(phase: str) -> str:
    """Map a phase name to the registry phase tag used by ``spec.applies_to``."""
    from benchmarks.registry import (
        BUILD,
        FROM_NETCDF,
        MATRICES,
        TO_HIGHSPY,
        TO_LP,
        TO_NETCDF,
    )

    return {
        "build": BUILD,
        "matrices": MATRICES,
        "to_lp": TO_LP,
        "to_netcdf": TO_NETCDF,
        "from_netcdf": FROM_NETCDF,
        "to_solver": TO_HIGHSPY,  # we always measure the highs handoff
        "pipeline": BUILD,
    }[phase]


def measure_peak(action: Callable[[], object], repeats: int = 1) -> float:
    """
    Run ``action()`` under ``memray.Tracker`` and return peak MiB.

    With ``repeats > 1`` the action runs that many times in fresh
    trackers and the *minimum* peak is returned — peak memory is
    noisier than naive expectations (GC timing, lazy-import priming,
    file-system page cache for netcdf) so the min-of-N is the cleanest
    estimate of "the floor this code can hit".
    """
    _require_memray()

    import memray

    peaks: list[float] = []
    for _ in range(max(1, repeats)):
        fd, tmp = tempfile.mkstemp(suffix=".bin")
        Path(tmp).unlink()  # memray needs to create the file itself
        # Close the fd; the path is what matters.
        try:
            from os import close as _close

            _close(fd)
        except OSError:
            pass

        try:
            with memray.Tracker(tmp):
                action()
            peak_bytes = memray.FileReader(tmp).metadata.peak_memory
            peaks.append(round(peak_bytes / (1024**2), 3))
        finally:
            Path(tmp).unlink(missing_ok=True)
        gc.collect()

    return min(peaks)


# Back-compat alias: ``_measure_peak`` was the private name before
# ``benchmarks.bench`` needed to reuse it.
_measure_peak = measure_peak


def _measurements(
    phase: str, spec: BenchSpec, size: int
) -> Iterator[tuple[str, Callable[[], object]]]:
    """
    Yield ``(test_id, action)`` pairs for one ``(phase, spec, size)``.

    ``action`` is a zero-arg callable; the caller runs it inside a tracker.
    For non-build phases, the model is built once up front (outside the
    tracker) and the action closes over it so only the phase work is
    counted. ``size`` is the swept value along ``spec.axis`` (model size or
    pattern severity); the test ids match the shared phase drivers either way.
    """
    name = spec.name
    axis = spec.axis

    if phase == "build":
        yield (
            f"benchmarks/test_build.py::test_build[{spec_param_id(name, axis, size)}]",
            lambda: spec.build(size),
        )
        return

    if phase == "pipeline":
        from benchmarks.phases import touch_matrices, write_lp

        tmpdir = tempfile.TemporaryDirectory()
        lp_path = Path(tmpdir.name) / "m.lp"

        def run_pipeline() -> None:
            built = spec.build(size)
            touch_matrices(built)
            write_lp(built, lp_path)

        try:
            yield (f"pipeline[{spec_param_id(name, axis, size)}]", run_pipeline)
        finally:
            tmpdir.cleanup()
        return

    m = spec.build(size)

    if phase == "matrices":
        from benchmarks.phases import touch_matrices

        yield (
            f"benchmarks/test_matrices.py::test_matrices[{spec_param_id(name, axis, size)}]",
            lambda: touch_matrices(m),
        )

    elif phase == "to_lp":
        from benchmarks.phases import write_lp

        tmpdir = tempfile.TemporaryDirectory()
        lp_path = Path(tmpdir.name) / "m.lp"
        try:
            yield (
                f"benchmarks/test_to_lp.py::test_to_lp[{spec_param_id(name, axis, size)}]",
                lambda: write_lp(m, lp_path),
            )
        finally:
            tmpdir.cleanup()

    elif phase == "to_netcdf":
        from benchmarks.phases import write_netcdf

        tmpdir = tempfile.TemporaryDirectory()
        nc_path = Path(tmpdir.name) / "m.nc"
        try:
            yield (
                f"benchmarks/test_netcdf.py::test_to_netcdf[{spec_param_id(name, axis, size)}]",
                lambda: write_netcdf(m, nc_path),
            )
        finally:
            tmpdir.cleanup()

    elif phase == "from_netcdf":
        from benchmarks.phases import read_netcdf, write_netcdf

        tmpdir = tempfile.TemporaryDirectory()
        nc_path = Path(tmpdir.name) / "m.nc"
        write_netcdf(m, nc_path)  # setup: written outside the tracker
        try:
            yield (
                f"benchmarks/test_netcdf.py::test_from_netcdf[{spec_param_id(name, axis, size)}]",
                lambda: read_netcdf(nc_path),
            )
        finally:
            tmpdir.cleanup()

    elif phase == "to_solver":
        from benchmarks.phases import SOLVER_HANDOFFS

        # Memory currently tracks only HiGHS — look it up by name so a
        # reordering of SOLVER_HANDOFFS doesn't silently swap solvers.
        # Older linopy releases without ``to_highspy`` skip the phase
        # silently rather than emitting an id with no possible match.
        highs = next((w for n, _, w in SOLVER_HANDOFFS if n == "highs"), None)
        if highs is None:
            return

        yield (
            (
                f"benchmarks/test_to_solver.py::test_to_solver"
                f"[highs-{spec_param_id(name, axis, size)}]"
            ),
            lambda: highs(m),
        )

    else:
        raise ValueError(f"unknown phase: {phase!r}")


def run_phase(
    phase: str,
    quick: bool = False,
    repeats: int = 1,
    filter_expr: str | None = None,
    long: bool = False,
    sizes: tuple[int, ...] = (),
    severities: tuple[int, ...] = (),
) -> dict[str, float]:
    """
    Measure peak memory for every applicable ``(spec, size)`` under one phase.

    Returns a ``{test_id: peak_mib}`` mapping. Invoked once per phase as a
    subprocess by :func:`measure` for isolation. ``repeats`` is forwarded to
    :func:`measure_peak` so callers can dial up signal-to-noise. ``filter_expr``
    keeps only specs whose ``<name>-<axis>=<value>`` key contains it — e.g.
    ``"nodal_balance"`` (one spec), ``"severity"`` (patterns), ``"n="`` (models).
    Size selection (``quick`` / ``long`` / ``sizes`` / ``severities``) shares
    :func:`benchmarks.registry.skip_reason` with pytest so the two never drift.
    """
    _require_memray()

    from benchmarks.registry import all_specs, skip_reason

    tag = _phase_tag(phase)
    results: dict[str, float] = {}

    for spec in all_specs():
        if not spec.applies_to(tag):
            continue

        # Optional-dep gate (e.g. pypsa_scigrid needs pypsa).
        for mod in spec.requires:
            try:
                __import__(mod)
            except ImportError:
                break
        else:
            for value in spec.sweep:
                if skip_reason(
                    spec,
                    value,
                    quick=quick,
                    long=long,
                    sizes=sizes,
                    severities=severities,
                ):
                    continue
                key = spec_param_id(spec.name, spec.axis, value)
                if filter_expr and filter_expr not in key:
                    continue
                try:
                    for test_id, action in _measurements(phase, spec, value):
                        try:
                            results[test_id] = _measure_peak(action, repeats=repeats)
                            print(
                                f"  {test_id} → {results[test_id]:.1f} MiB",
                                file=sys.stderr,
                            )
                        except Exception as exc:  # noqa: BLE001
                            print(
                                f"  skip {test_id}: {type(exc).__name__}: {exc}",
                                file=sys.stderr,
                            )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  setup failed {spec.name}/{value}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                gc.collect()

    return results


def measure(
    quick: bool = False,
    phases: list[str] | None = None,
    repeats: int = 1,
    filter_expr: str | None = None,
    long: bool = False,
    sizes: tuple[int, ...] = (),
    severities: tuple[int, ...] = (),
) -> dict[str, float]:
    """
    Run one subprocess per phase and return merged ``{test_id: peak_mib}``.

    Per-phase subprocesses keep allocations from one phase out of another's
    measurement; ``memray.Tracker`` only counts what's allocated inside its
    ``with`` block, but the subprocess boundary makes the isolation total.
    ``filter_expr`` restricts which specs are measured (substring of the
    ``<name>-<axis>=<value>`` key); ``quick``/``long``/``sizes``/``severities``
    select sizes the same way pytest does.
    """
    _require_memray()

    phases = list(phases) if phases else list(MEMORY_PHASES)

    all_results: dict[str, float] = {}
    for phase in phases:
        print(f"\n=== {phase} ===", file=sys.stderr)
        # Worker writes JSON to a sidecar file rather than stdout — HiGHS
        # (and other solvers) print to stdout from C code inside the tracked
        # region, which would pollute the data channel.
        fd, out_tmp = tempfile.mkstemp(suffix=".json", prefix=f"mem-{phase}-")
        from os import close as _close

        _close(fd)
        cmd = [
            sys.executable,
            "-m",
            "benchmarks.memory",
            "_worker",
            phase,
            "--out",
            out_tmp,
        ]
        if quick:
            cmd.append("--quick")
        if long:
            cmd.append("--long")
        for s in sizes:
            cmd.extend(["--size", str(s)])
        for s in severities:
            cmd.extend(["--severity", str(s)])
        if repeats > 1:
            cmd.extend(["--repeats", str(repeats)])
        if filter_expr:
            cmd.extend(["--filter", filter_expr])
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.stderr:
                sys.stderr.write(result.stderr)
            if result.returncode != 0:
                print(
                    f"phase {phase} subprocess failed (exit {result.returncode})",
                    file=sys.stderr,
                )
                continue
            try:
                phase_results = json.loads(Path(out_tmp).read_text())
            except (json.JSONDecodeError, FileNotFoundError) as exc:
                print(f"phase {phase} JSON parse error: {exc}", file=sys.stderr)
                continue
            all_results.update(phase_results)
        finally:
            Path(out_tmp).unlink(missing_ok=True)

    return all_results


def save(
    label: str,
    quick: bool = False,
    phases: list[str] | None = None,
    repeats: int = 1,
    filter_expr: str | None = None,
    long: bool = False,
    sizes: tuple[int, ...] = (),
    severities: tuple[int, ...] = (),
) -> Path:
    """Measure peak memory and write a snapshot to ``RESULTS_DIR/<label>.json``."""
    results = measure(
        quick=quick,
        phases=phases,
        repeats=repeats,
        filter_expr=filter_expr,
        long=long,
        sizes=sizes,
        severities=severities,
    )
    if not results:
        print("No measurements produced.", file=sys.stderr)
        sys.exit(1)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = write_memory_snapshot(RESULTS_DIR / f"{label}.json", label, results)
    print(f"\nSaved {len(results)} measurements to {out_path}", file=sys.stderr)
    return out_path


def compare(label_a: str, label_b: str) -> None:
    """Diff two saved memory snapshots (by label) side-by-side."""
    path_a = RESULTS_DIR / f"{label_a}.json"
    path_b = RESULTS_DIR / f"{label_b}.json"
    for p in (path_a, path_b):
        if not p.exists():
            print(f"Not found: {p}. Run 'save {p.stem}' first.", file=sys.stderr)
            sys.exit(1)
    compare_snapshots(path_a, path_b)


def compare_snapshots(path_a: Path, path_b: Path) -> None:
    """Diff two memory snapshots (by path) side-by-side."""
    label_a, label_b = Path(path_a).stem, Path(path_b).stem
    data_a = json.loads(Path(path_a).read_text())["peak_mib"]
    data_b = json.loads(Path(path_b).read_text())["peak_mib"]

    all_tests = sorted(set(data_a) | set(data_b))

    print(f"\n{'Test':<70} {label_a:>10} {label_b:>10} {'Change':>10}")
    print("-" * 104)

    for test in all_tests:
        a = data_a.get(test)
        b = data_b.get(test)
        a_str = f"{a:.1f}" if a is not None else "—"
        b_str = f"{b:.1f}" if b is not None else "—"
        if a is not None and b is not None and a > 0:
            pct = (b - a) / a * 100
            change = f"{pct:+.1f}%"
        else:
            change = "—"
        short = test.split("::")[-1] if "::" in test else test
        print(f"{short:<70} {a_str:>10} {b_str:>10} {change:>10}")

    print()


# ---- subprocess worker ---------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="memory.py worker")
    parser.add_argument("cmd", choices=["_worker"])
    parser.add_argument("phase")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--long", action="store_true")
    parser.add_argument("--size", action="append", type=int, default=[], dest="sizes")
    parser.add_argument(
        "--severity", action="append", type=int, default=[], dest="severities"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Run each measurement N times and keep the min peak (default 1).",
    )
    parser.add_argument(
        "--filter",
        dest="filter_expr",
        default=None,
        help="Keep only specs whose <name>-<axis>=<value> key contains this.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to write the JSON result to (stdout is reserved for solver chatter).",
    )
    args = parser.parse_args()
    if args.cmd == "_worker":
        out = run_phase(
            args.phase,
            quick=args.quick,
            repeats=args.repeats,
            filter_expr=args.filter_expr,
            long=args.long,
            sizes=tuple(args.sizes),
            severities=tuple(args.severities),
        )
        Path(args.out).write_text(json.dumps(out))
