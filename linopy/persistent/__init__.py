"""Persistent-solver snapshot and diff primitives."""

from __future__ import annotations

from linopy.persistent.diff import (
    ContainerRowUpdate,
    ContainerVarUpdate,
    ModelDiff,
    RebuildReason,
    compute_diff,
)
from linopy.persistent.errors import UnsupportedUpdate
from linopy.persistent.snapshot import (
    ContainerConBuffers,
    ContainerVarBuffers,
    ModelSnapshot,
    StructuralKey,
)

__all__ = [
    "ContainerConBuffers",
    "ContainerRowUpdate",
    "ContainerVarBuffers",
    "ContainerVarUpdate",
    "ModelDiff",
    "ModelSnapshot",
    "RebuildReason",
    "StructuralKey",
    "UnsupportedUpdate",
    "compute_diff",
]
