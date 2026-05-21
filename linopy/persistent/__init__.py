"""Persistent-solver snapshot and diff primitives."""

from __future__ import annotations

from linopy.persistent.diff import (
    ConSlice,
    ModelDiff,
    RebuildReason,
    VarSlice,
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
    VarKind,
)

__all__ = [
    "ConSlice",
    "ContainerConBuffers",
    "ContainerVarBuffers",
    "ModelDiff",
    "ModelSnapshot",
    "RebuildReason",
    "RebuildRequiredError",
    "StructuralKey",
    "UnsupportedUpdate",
    "UpdatesDisabledError",
    "VarKind",
    "VarSlice",
]
