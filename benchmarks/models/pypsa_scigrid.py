"""PyPSA SciGrid-DE benchmark model."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import linopy

LABEL = "pypsa snapshots={snapshots}"
SIZES = [{"snapshots": s} for s in [10, 50, 100, 200]]
QUICK_SIZES = [{"snapshots": s} for s in [10, 50]]
DESCRIPTION = "Real power system model from PyPSA SciGrid-DE"


def build(snapshots: int = 100) -> linopy.Model | None:
    """Build PyPSA SciGrid model. Returns None if pypsa not installed."""
    try:
        import pypsa
    except ImportError:
        return None

    n = pypsa.examples.scigrid_de()
    n.set_snapshots(n.snapshots[:snapshots])
    n.optimize.create_model()
    return n.model
