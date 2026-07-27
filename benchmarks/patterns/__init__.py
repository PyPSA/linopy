"""
Benchmark *patterns* — realistic modelling idioms swept over a severity dial.

A pattern is a fragment of real modelling code (a balance constraint, a KVL
contraction), not a whole model and not an isolated method call. Each is
measured the same way a model is — time and peak memory, through the shared
phases — but parametrised by ``severity`` (0–100, how pathological the data
shape is) instead of ``size``. See :class:`benchmarks.registry.BenchSpec`.

Importing this package registers every idiom into
:data:`benchmarks.registry.PATTERNS` (mirrors :mod:`benchmarks.models`); adding
a pattern is one new file plus one import below.
"""

# Side-effect imports — each module calls ``register_pattern(...)`` at import.
from benchmarks.patterns import (  # noqa: F401
    cumsum,
    kvl_cycles,
    merge_balance,
    nodal_balance,
    rolling,
)
