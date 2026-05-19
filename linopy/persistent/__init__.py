"""Persistent-solver snapshot and diff primitives."""

from __future__ import annotations

from linopy.persistent.diff import ModelDiff, RebuildReason, compute_diff
from linopy.persistent.errors import UnsupportedUpdate
from linopy.persistent.snapshot import CoefPattern, ModelSnapshot, StructuralKey

__all__ = [
    "CoefPattern",
    "ModelDiff",
    "ModelSnapshot",
    "RebuildReason",
    "StructuralKey",
    "UnsupportedUpdate",
    "compute_diff",
]
