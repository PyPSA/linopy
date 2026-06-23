"""
Model builders for benchmarks.

Importing this package triggers every submodule's ``register(...)`` call,
populating :data:`benchmarks.registry.REGISTRY`. Each submodule exposes a
``build_<name>(size) -> linopy.Model`` callable and a module-level ``SPEC``
:class:`~benchmarks.registry.BenchSpec`. The documented access path is
``REGISTRY["<name>"]``; submodule re-exports are intentionally not exposed
here so that adding a new model is one new file plus one import below.
"""

# Side-effect imports — each module calls ``register(...)`` at import time.
from benchmarks.models import (  # noqa: F401
    basic,
    expression_arithmetic,
    knapsack,
    masked,
    milp,
    piecewise,
    pypsa_scigrid,
    qp,
    sos,
    sparse_network,
    storage,
)
