#!/usr/bin/env python
"""
Measure and compare peak memory using pytest-memray.

Usage:
    # Save a baseline (on master)
    python benchmarks/memory.py save master

    # Save current branch
    python benchmarks/memory.py save my-feature

    # Compare two saved runs
    python benchmarks/memory.py compare master my-feature

    # Quick mode (smaller sizes)
    python benchmarks/memory.py save master --quick

Results are stored in .benchmarks/memory/.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

if platform.system() == "Windows":
    raise RuntimeError(
        "memory.py requires pytest-memray which is not available on Windows. "
        "Run memory benchmarks on Linux or macOS."
    )

RESULTS_DIR = Path(".benchmarks/memory")
MEMORY_RE = re.compile(
    r"Allocation results for (.+?) at the high watermark\s+"
    r"📦 Total memory allocated: ([\d.]+)(MiB|KiB|GiB|B)",
)
# Only the build phase is measured by default. Unlike timing benchmarks (where
# pytest-benchmark isolates the measured function), memray tracks all allocations
# within a test — including model construction in setup. This means LP write and
# matrix tests would report build + phase memory combined, making the phase-specific
# contribution hard to isolate. Since model construction dominates memory usage,
# measuring build alone gives the most accurate and actionable numbers.
DEFAULT_TEST_PATHS = [
    "benchmarks/test_build.py",
]


def _to_mib(value: float, unit: str) -> float:
    factors = {"B": 1 / 1048576, "KiB": 1 / 1024, "MiB": 1, "GiB": 1024}
    return value * factors[unit]


def _collect_test_ids(test_paths: list[str], quick: bool) -> list[str]:
    """Collect test IDs without running them."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *test_paths,
        "--collect-only",
        "-q",
    ]
    if quick:
        cmd.append("--quick")
    result = subprocess.run(cmd, capture_output=True, text=True)
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if "::" in line and not line.startswith(("=", "-", " "))
    ]


def save(label: str, quick: bool = False, test_paths: list[str] | None = None) -> Path:
    """Run each benchmark in a separate process for accurate memory measurement."""
    if test_paths is None:
        test_paths = DEFAULT_TEST_PATHS
    test_ids = _collect_test_ids(test_paths, quick)
    if not test_ids:
        print("No tests collected.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {len(test_ids)} tests (each in a separate process)...")
    entries = {}
    for i, test_id in enumerate(test_ids, 1):
        short = test_id.split("::")[-1]
        print(f"  [{i}/{len(test_ids)}] {short}...", end=" ", flush=True)

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            test_id,
            "--memray",
            "--benchmark-disable",
            "-v",
            "--tb=short",
            "-q",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr

        match = MEMORY_RE.search(output)
        if match:
            value = float(match.group(2))
            unit = match.group(3)
            mib = round(_to_mib(value, unit), 3)
            entries[test_id] = mib
            print(f"{mib:.1f} MiB")
        elif "SKIPPED" in output or "skipped" in output:
            print("skipped")
        else:
            print(
                "WARNING: no memray data (pytest-memray output format may have changed)",
                file=sys.stderr,
            )

    if not entries:
        print("No memray results found. Is pytest-memray installed?", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{label}.json"
    out_path.write_text(json.dumps({"label": label, "peak_mib": entries}, indent=2))
    print(f"\nSaved {len(entries)} results to {out_path}")
    return out_path


def compare(label_a: str, label_b: str) -> None:
    """Compare two saved memory results."""
    path_a = RESULTS_DIR / f"{label_a}.json"
    path_b = RESULTS_DIR / f"{label_b}.json"
    for p in (path_a, path_b):
        if not p.exists():
            print(f"Not found: {p}. Run 'save {p.stem}' first.", file=sys.stderr)
            sys.exit(1)

    data_a = json.loads(path_a.read_text())["peak_mib"]
    data_b = json.loads(path_b.read_text())["peak_mib"]

    all_tests = sorted(set(data_a) | set(data_b))

    print(f"\n{'Test':<60} {label_a:>10} {label_b:>10} {'Change':>10}")
    print("-" * 94)

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
        # Shorten test name for readability
        short = test.split("::")[-1] if "::" in test else test
        print(f"{short:<60} {a_str:>10} {b_str:>10} {change:>10}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_save = sub.add_parser("save", help="Run benchmarks and save memory results")
    p_save.add_argument(
        "label", help="Label for this run (e.g. 'master', 'my-feature')"
    )
    p_save.add_argument(
        "--quick", action="store_true", help="Use smaller problem sizes"
    )
    p_save.add_argument(
        "--test-path",
        nargs="+",
        default=None,
        help="Test file(s) to run (default: all phases)",
    )

    p_cmp = sub.add_parser("compare", help="Compare two saved runs")
    p_cmp.add_argument("label_a", help="First run label (baseline)")
    p_cmp.add_argument("label_b", help="Second run label")

    args = parser.parse_args()
    if args.cmd == "save":
        save(args.label, quick=args.quick, test_paths=args.test_path)
    elif args.cmd == "compare":
        compare(args.label_a, args.label_b)


if __name__ == "__main__":
    main()
