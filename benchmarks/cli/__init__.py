"""
linopy benchmark CLI — one entry point for the suite.

Run with::

    python -m benchmarks <command> [options]

The CLI is a thin layer over pytest for the timing / smoke commands, plus
direct dispatch for registry introspection and memory snapshots. Each command
group lives in its own module and registers onto the shared ``app`` from
``_base``; importing them here (in display order) wires up the flat command
surface.
"""

from __future__ import annotations

from benchmarks.cli._base import app

# Imported for side effect: each module registers its commands onto ``app``.
# Kept in this order — and shielded from isort — because it is the order the
# commands appear in ``--help``.
# isort: off
from benchmarks.cli import introspect  # noqa: F401
from benchmarks.cli import run  # noqa: F401
from benchmarks.cli import sweep  # noqa: F401
from benchmarks.cli import compare  # noqa: F401
from benchmarks.cli import plot  # noqa: F401
from benchmarks.cli import memory  # noqa: F401

# isort: on

__all__ = ["app"]
