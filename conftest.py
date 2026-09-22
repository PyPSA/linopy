"""Root pytest configuration for ``--doctest-modules`` collection of ``linopy/``."""

from __future__ import annotations

from importlib.util import find_spec

collect_ignore: list[str] = []
if find_spec("math_spec") is None:
    collect_ignore.append("linopy/spec")
