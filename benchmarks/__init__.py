"""
Linopy benchmark suite — run with ``pytest benchmarks/``.

The model registry it drives is reusable on its own::

    from benchmarks import REGISTRY
    model = REGISTRY["basic"].build(100)
"""

# Importing the models / patterns packages triggers each module's
# ``register(...)`` / ``register_pattern(...)`` call at import time.
from benchmarks import models, patterns  # noqa: F401
from benchmarks.registry import PATTERNS, REGISTRY

__all__ = ["PATTERNS", "REGISTRY"]
