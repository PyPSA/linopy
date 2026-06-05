"""
Guard test for the timing ↔ memory test-id seam.

``memory.py`` hand-rolls f-strings to label each measurement with the
same node id pytest-benchmark produces (e.g.
``benchmarks/test_matrices.py::test_matrices[basic-n=10]``). If a
benchmark test function gets renamed and the matching f-string in
``memory.py`` isn't updated, ``plot`` would silently end up with
non-overlapping timing and memory sets — no error, just missing data.

This test exercises both sides once and asserts every memory-emitted
id is present in pytest's collection.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from benchmarks.memory import MEMORY_PHASES, _measurements
from benchmarks.registry import REGISTRY


def _collect_benchmark_ids() -> set[str]:
    """Return the set of node ids pytest collects under ``benchmarks/``."""
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "benchmarks/",
            "--collect-only",
            "-q",
            "--no-header",
            "--co",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root,
    )
    # pytest -q --co emits one node id per line; trailing summary lines
    # like "N tests collected" can be ignored.
    return {
        line.strip()
        for line in result.stdout.splitlines()
        if re.match(r"^benchmarks/.*::.*\[.*\]$", line.strip())
    }


def test_memory_node_ids_match_pytest_collection() -> None:
    collected = _collect_benchmark_ids()
    assert collected, "pytest collected zero benchmark node ids — sanity broken"

    # ``basic`` at its smallest size is cheap and declares every default
    # phase, so it exercises every node-id format ``_measurements`` emits.
    spec = REGISTRY["basic"]
    size = spec.sizes[0]

    mem_ids: set[str] = set()
    for phase in MEMORY_PHASES:
        for test_id, _ in _measurements(phase, spec, size):
            mem_ids.add(test_id)

    missing = mem_ids - collected
    assert not missing, (
        "memory.py emits node ids that pytest doesn't collect "
        "(test rename drift?):\n" + "\n".join(f"  {m}" for m in sorted(missing))
    )
