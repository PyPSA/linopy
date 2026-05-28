"""
Linopy benchmark suite.

Run with ``pytest benchmarks/`` (use ``--quick`` for smaller sizes).

This package also exposes a **reusable model registry** for any test, profiling
session, or example that wants ready-made linopy models of varying sizes and
features. Each entry exposes a ``build(size) -> linopy.Model`` callable plus
metadata::

    from benchmarks import REGISTRY, QUADRATIC

    # Look up by name
    model = REGISTRY["basic"].build(100)

    # Iterate / filter
    for spec in REGISTRY.values():
        m = spec.build(spec.sizes[0])
        ...

    from benchmarks import filter_by
    qp_specs = filter_by(has_feature=QUADRATIC)
"""

# Importing the models package triggers each module's ``register(...)`` call.
from benchmarks import models  # noqa: F401, E402
from benchmarks.registry import (  # noqa: F401 — re-export
    ALL_FEATURES,
    ALL_PHASES,
    BINARY,
    BUILD,
    CONTINUOUS,
    DEFAULT_PHASES,
    INTEGER,
    LP_WRITE,
    MASKED,
    MATRICES,
    NETCDF,
    PIECEWISE,
    QUADRATIC,
    REGISTRY,
    SOLVER_BUILD,
    SOS,
    TO_GUROBIPY,
    TO_HIGHSPY,
    TO_MOSEK,
    TO_XPRESS,
    ModelSpec,
    filter_by,
    get,
    iter_params,
    param_ids,
    register,
)

__all__ = [
    "ALL_FEATURES",
    "ALL_PHASES",
    "BINARY",
    "BUILD",
    "CONTINUOUS",
    "DEFAULT_PHASES",
    "INTEGER",
    "LP_WRITE",
    "MASKED",
    "MATRICES",
    "ModelSpec",
    "NETCDF",
    "PIECEWISE",
    "QUADRATIC",
    "REGISTRY",
    "SOLVER_BUILD",
    "SOS",
    "TO_GUROBIPY",
    "TO_HIGHSPY",
    "TO_MOSEK",
    "TO_XPRESS",
    "filter_by",
    "get",
    "iter_params",
    "param_ids",
    "register",
]
