"""PyPSA SciGrid-DE benchmark model (requires pypsa)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarks.registry import BenchSpec, register

if TYPE_CHECKING:
    import linopy

SIZES = (10, 50)  # small networks — PyPSA import already dominates the cost


def build_pypsa_scigrid(snapshots: int = 100) -> linopy.Model:
    """Build PyPSA SciGrid model. Requires pypsa to be installed."""
    import pypsa
    import pytest

    try:
        n = pypsa.examples.scigrid_de()
    except Exception as exc:  # network / example-data drift, not a linopy signal
        pytest.skip(f"pypsa example data unavailable: {exc}")
    n.set_snapshots(n.snapshots[:snapshots])
    n.optimize.create_model()  # the linopy build under benchmark — unguarded
    return n.model


SPEC = register(
    BenchSpec(
        name="pypsa_scigrid",
        build=build_pypsa_scigrid,
        sweep=SIZES,
        requires=("pypsa",),
    )
)
