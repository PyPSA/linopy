"""PyPSA SciGrid-DE benchmark model."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import linopy

SIZES = [10, 50, 100, 200]


def build_pypsa_scigrid(snapshots: int = 100) -> linopy.Model:
    """Build PyPSA SciGrid model. Requires pypsa to be installed."""
    import pypsa

    n = pypsa.examples.scigrid_de()
    n.set_snapshots(n.snapshots[:snapshots])
    n.optimize.create_model()
    return n.model
