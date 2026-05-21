"""Persistent-solver snapshot and diff primitives."""

from __future__ import annotations

from linopy.persistent.diff import (
    ContainerRowUpdate,
    ContainerVarUpdate,
    ModelDiff,
    RebuildReason,
)
from linopy.persistent.errors import (
    RebuildRequiredError,
    UnsupportedUpdate,
    UpdatesDisabledError,
)
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
    "RebuildRequiredError",
    "StructuralKey",
    "UnsupportedUpdate",
    "UpdatesDisabledError",
]
