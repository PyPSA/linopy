"""Runner registry for benchmarks."""

from __future__ import annotations

from benchmarks.runners import build, lp_write, memory

_RUNNERS = {
    "build": build,
    "memory": memory,
    "lp_write": lp_write,
}


def get_runner(phase: str):
    """Return a runner module by phase name."""
    return _RUNNERS[phase]


def list_phases() -> list[str]:
    """Return sorted list of available phase names."""
    return sorted(_RUNNERS)
