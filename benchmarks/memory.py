"""
Measure and compare peak memory across the registry × phase grid.

Each measurement uses ``memray.Tracker`` directly so the model construction
(setup) lives *outside* the tracked region and the peak reflects only the
phase work itself::

    m = spec.build(size)            # setup, not tracked
    with memray.Tracker(bin_path):
        wrapper(m)                  # tracked
    peak = FileReader(bin_path).metadata.peak_memory

This module exposes ``save(label, ...)`` and ``compare(label_a, label_b)`` as
plain functions; user-facing invocation goes through the typer CLI::

    python -m benchmarks memory save <label> [--quick] [--phase build] ...
    python -m benchmarks memory compare <a> <b>

Results land in ``.benchmarks/memory/`` as JSON keyed by full pytest-style
test IDs (``benchmarks/test_<phase>.py::test_<phase>[<spec>-n=<size>]``)
so cross-snapshot diffs work uniformly regardless of which phases were run.
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

if platform.system() == "Windows":
    raise RuntimeError(
        "memory measurement requires ``memray`` which is not available on "
        "Windows. Run memory benchmarks on Linux or macOS."
    )

if TYPE_CHECKING:
    from benchmarks.registry import ModelSpec

RESULTS_DIR = Path(".benchmarks/memory")
MEMORY_PHASES: tuple[str, ...] = (
    "build",
    "matrices",
    "lp_write",
    "netcdf",
    "solver_handoff",
)


def _phase_tag(phase: str) -> str:
    """Map a phase name to the registry phase tag used by ``spec.applies_to``."""
    from benchmarks.registry import (
        BUILD,
        LP_WRITE,
        MATRICES,
        NETCDF,
        TO_HIGHSPY,
    )

    return {
        "build": BUILD,
        "matrices": MATRICES,
        "lp_write": LP_WRITE,
        "netcdf": NETCDF,
        "solver_handoff": TO_HIGHSPY,  # we always measure the highs handoff
    }[phase]


def _measure_peak(action: Callable[[], object], repeats: int = 1) -> float:
    """
    Run ``action()`` under ``memray.Tracker`` and return peak MiB.

    With ``repeats > 1`` the action runs that many times in fresh
    trackers and the *minimum* peak is returned — peak memory is
    noisier than naive expectations (GC timing, lazy-import priming,
    file-system page cache for netcdf) so the min-of-N is the cleanest
    estimate of "the floor this code can hit".
    """
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


def _measurements(
    phase: str, spec: ModelSpec, size: int
) -> Iterator[tuple[str, Callable[[], object]]]:
    """
    Yield ``(test_id, action)`` pairs for one ``(phase, spec, size)``.

    ``action`` is a zero-arg callable; the caller runs it inside a tracker.
    For non-build phases, the model is built once up front (outside the
    tracker) and the action closes over it so only the phase work is
    counted.
    """
    name = spec.name

    if phase == "build":
        yield (
            f"benchmarks/test_build.py::test_build[{name}-n={size}]",
            lambda: spec.build(size),
        )
        return

    m = spec.build(size)

    if phase == "matrices":
        from benchmarks.phases import touch_matrices

        yield (
            f"benchmarks/test_matrices.py::test_matrices[{name}-n={size}]",
            lambda: touch_matrices(m),
        )

    elif phase == "lp_write":
        from benchmarks.phases import write_lp

        tmpdir = tempfile.TemporaryDirectory()
        lp_path = Path(tmpdir.name) / "m.lp"
        try:
            yield (
                f"benchmarks/test_lp_write.py::test_lp_write[{name}-n={size}]",
                lambda: write_lp(m, lp_path),
            )
        finally:
            tmpdir.cleanup()

    elif phase == "netcdf":
        from benchmarks.phases import read_netcdf, write_netcdf

        tmpdir = tempfile.TemporaryDirectory()
        nc_path = Path(tmpdir.name) / "m.nc"
        try:
            yield (
                f"benchmarks/test_netcdf.py::test_netcdf_write[{name}-n={size}]",
                lambda: write_netcdf(m, nc_path),
            )
            # ``write_netcdf`` was called by the caller as part of the
            # measurement, so ``nc_path`` now exists for the read.
            yield (
                f"benchmarks/test_netcdf.py::test_netcdf_read[{name}-n={size}]",
                lambda: read_netcdf(nc_path),
            )
        finally:
            tmpdir.cleanup()

    elif phase == "solver_handoff":
        from benchmarks.phases import SOLVER_HANDOFFS

        # Memory currently tracks only HiGHS — look it up by name so a
        # reordering of SOLVER_HANDOFFS doesn't silently swap solvers.
        highs = next(w for n, _, w in SOLVER_HANDOFFS if n == "highs")

        yield (
            (
                f"benchmarks/test_solver_handoff.py::test_solver_handoff"
                f"[highs-{name}-n={size}]"
            ),
            lambda: highs(m),
        )

    else:
        raise ValueError(f"unknown phase: {phase!r}")


def run_phase(phase: str, quick: bool = False, repeats: int = 1) -> dict[str, float]:
    """
    Measure peak memory for every applicable ``(spec, size)`` under one phase.

    Returns a ``{test_id: peak_mib}`` mapping. Invoked once per phase as a
    subprocess by :func:`save` for isolation. ``repeats`` is forwarded to
    :func:`_measure_peak` so callers can dial up signal-to-noise.
    """
    from benchmarks import REGISTRY

    tag = _phase_tag(phase)
    results: dict[str, float] = {}

    for spec in REGISTRY.values():
        if not spec.applies_to(tag):
            continue

        # Optional-dep gate (e.g. pypsa_scigrid needs pypsa).
        for mod in spec.requires:
            try:
                __import__(mod)
            except ImportError:
                break
        else:
            for size in spec.sizes:
                if quick and size > spec.quick_threshold:
                    continue
                try:
                    for test_id, action in _measurements(phase, spec, size):
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
                        f"  setup failed {spec.name}/{size}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                gc.collect()

    return results


def save(
    label: str,
    quick: bool = False,
    phases: list[str] | None = None,
    repeats: int = 1,
) -> Path:
    """
    Run one subprocess per phase and merge the results into ``<label>.json``.

    Per-phase subprocesses keep allocations from one phase out of another's
    measurement; ``memray.Tracker`` only counts what's allocated inside its
    ``with`` block, but the subprocess boundary makes the isolation total.
    """
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
        if repeats > 1:
            cmd.extend(["--repeats", str(repeats)])
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

    if not all_results:
        print("No measurements produced.", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{label}.json"
    out_path.write_text(json.dumps({"label": label, "peak_mib": all_results}, indent=2))
    print(f"\nSaved {len(all_results)} measurements to {out_path}", file=sys.stderr)
    return out_path


def compare(label_a: str, label_b: str) -> None:
    """Diff two saved memory snapshots side-by-side."""
    path_a = RESULTS_DIR / f"{label_a}.json"
    path_b = RESULTS_DIR / f"{label_b}.json"
    for p in (path_a, path_b):
        if not p.exists():
            print(f"Not found: {p}. Run 'save {p.stem}' first.", file=sys.stderr)
            sys.exit(1)

    data_a = json.loads(path_a.read_text())["peak_mib"]
    data_b = json.loads(path_b.read_text())["peak_mib"]

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
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Run each measurement N times and keep the min peak (default 1).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Path to write the JSON result to (stdout is reserved for solver chatter).",
    )
    args = parser.parse_args()
    if args.cmd == "_worker":
        out = run_phase(args.phase, quick=args.quick, repeats=args.repeats)
        Path(args.out).write_text(json.dumps(out))
