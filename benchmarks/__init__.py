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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from benchmarks.snapshot import Metric

# Importing the models / patterns packages triggers each module's
# ``register(...)`` / ``register_pattern(...)`` call at import time.
from benchmarks import bench, models, patterns  # noqa: F401, E402


def load_long_df(
    snapshots: list[Path], metric: Metric = "min"
) -> tuple[pd.DataFrame, str]:
    """
    Load one or more benchmark JSON snapshots into a tidy DataFrame.

    Thin re-export of :func:`benchmarks.snapshot.load_long_df` so callers
    can do their own analysis without importing the plotting module
    (which pulls in plotly). Returns ``(df, unit)`` where ``df`` has one
    row per ``(snapshot, test_id)`` with columns ``snapshot, test_id,
    phase, spec, size, value``, and ``unit`` is ``"s"`` (timing) or
    ``"MiB"`` (memory).
    """
    from benchmarks.snapshot import load_long_df as _impl

    return _impl(snapshots, metric)


from benchmarks.registry import (  # noqa: F401, E402 — re-export
    ALL_FEATURES,
    ALL_PHASES,
    BINARY,
    BUILD,
    CONTINUOUS,
    DEFAULT_PHASES,
    DEFAULT_SEVERITIES,
    INTEGER,
    LP_WRITE,
    MASKED,
    MATRICES,
    NETCDF,
    PATTERNS,
    PIECEWISE,
    QUADRATIC,
    REGISTRY,
    SOS,
    TO_GUROBIPY,
    TO_HIGHSPY,
    TO_MOSEK,
    TO_XPRESS,
    BenchSpec,
    ModelSpec,
    PatternSpec,
    all_specs,
    filter_by,
    get,
    get_pattern,
    iter_params,
    param_ids,
    register,
    register_pattern,
)

__all__ = [
    "ALL_FEATURES",
    "ALL_PHASES",
    "BINARY",
    "BUILD",
    "CONTINUOUS",
    "DEFAULT_PHASES",
    "DEFAULT_SEVERITIES",
    "INTEGER",
    "LP_WRITE",
    "MASKED",
    "MATRICES",
    "BenchSpec",
    "ModelSpec",
    "NETCDF",
    "PATTERNS",
    "PIECEWISE",
    "PatternSpec",
    "QUADRATIC",
    "REGISTRY",
    "SOS",
    "TO_GUROBIPY",
    "TO_HIGHSPY",
    "TO_MOSEK",
    "TO_XPRESS",
    "all_specs",
    "bench",
    "filter_by",
    "get",
    "get_pattern",
    "iter_params",
    "load_long_df",
    "param_ids",
    "register",
    "register_pattern",
]
