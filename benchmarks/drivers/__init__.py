"""
Phase drivers — one ``test_<phase>.py`` per measured phase.

Each driver is parametrised over every ``(spec, value)`` the phase runs (via
``benchmarks.conftest.cases``), does untimed setup, then wraps the phase verb
from :mod:`benchmarks.phases` in ``benchmark(...)``. Shared across models and
patterns alike — a pattern is just more rows tagged ``axis="severity"``.
"""
