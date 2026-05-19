"""Persistent-solver snapshot and diff primitives."""

from __future__ import annotations

from linopy.persistent.errors import UnsupportedUpdate
from linopy.persistent.snapshot import CoefPattern, ModelSnapshot, StructuralKey

__all__ = [
    "CoefPattern",
    "ModelSnapshot",
    "StructuralKey",
    "UnsupportedUpdate",
]
