"""
Registry of benchmark models and patterns.

A :class:`BenchSpec` declares how to build a model and which values (sizes for a
model, ``axis="n"``; severities for a pattern, ``axis="severity"``) and phases
it runs; ``register`` / ``register_pattern`` add it to :data:`REGISTRY` /
:data:`PATTERNS`::

    from benchmarks import REGISTRY
    model = REGISTRY["basic"].build(100)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import linopy

# --- Phase tags -------------------------------------------------------------

BUILD = "build"
MATRICES = "matrices"
TO_LP = "to_lp"
TO_NETCDF = "to_netcdf"
FROM_NETCDF = "from_netcdf"
TO_HIGHSPY = "to_highspy"
TO_GUROBIPY = "to_gurobipy"
TO_MOSEK = "to_mosek"
TO_XPRESS = "to_xpress"

ALL_PHASES = frozenset(
    {
        BUILD,
        MATRICES,
        TO_LP,
        TO_NETCDF,
        FROM_NETCDF,
        TO_HIGHSPY,
        TO_GUROBIPY,
        TO_MOSEK,
        TO_XPRESS,
    }
)

# The default phase set; a spec overrides with a narrower one when the default
# solvers can't ingest it natively (e.g. native SOS for HiGHS).
DEFAULT_PHASES = ALL_PHASES

# The severity sweep every pattern runs (axis "severity").
SEVERITIES: tuple[int, ...] = (0, 50, 100)


@dataclass(frozen=True, repr=False)
class BenchSpec:
    """
    One benchmark spec. A model is swept over ``sweep`` sizes (``axis="n"``); a
    pattern over a 0–100 severity dial (``axis="severity"``). Both build a
    :class:`linopy.Model` from one integer and run the same ``phases`` — the
    model-vs-pattern distinction lives in :func:`register` vs
    :func:`register_pattern` (and the ``models/`` vs ``patterns/`` dirs).
    """

    name: str
    build: Callable[[int], linopy.Model]
    sweep: tuple[int, ...]
    axis: str = "n"
    phases: frozenset[str] = DEFAULT_PHASES
    requires: tuple[str, ...] = ()

    def applies_to(self, phase: str) -> bool:
        return phase in self.phases

    def __repr__(self) -> str:
        return f"BenchSpec({self.name!r}, axis={self.axis!r}, sweep={self.sweep})"


REGISTRY: dict[str, BenchSpec] = {}
PATTERNS: dict[str, BenchSpec] = {}


def _validate(spec: BenchSpec, registry: dict[str, BenchSpec], kind: str) -> None:
    if spec.name in registry:
        raise ValueError(f"{kind} {spec.name!r} already registered")
    unknown = spec.phases - ALL_PHASES
    if unknown:
        raise ValueError(f"{kind} {spec.name!r}: unknown phases {sorted(unknown)}")


def register(spec: BenchSpec) -> BenchSpec:
    """Add a model ``spec`` to :data:`REGISTRY`. Returns it for chaining."""
    _validate(spec, REGISTRY, "model")
    REGISTRY[spec.name] = spec
    return spec


def register_pattern(spec: BenchSpec) -> BenchSpec:
    """Add a pattern ``spec`` (``axis="severity"``) to :data:`PATTERNS`."""
    _validate(spec, PATTERNS, "pattern")
    if spec.axis != "severity" or not all(0 <= s <= 100 for s in spec.sweep):
        raise ValueError(
            f"pattern {spec.name!r}: needs axis='severity' and sweep in [0, 100], "
            f"got axis={spec.axis!r} sweep={spec.sweep}"
        )
    PATTERNS[spec.name] = spec
    return spec


def all_specs() -> list[BenchSpec]:
    """Every spec in the suite — models then patterns."""
    return [*REGISTRY.values(), *PATTERNS.values()]


def iter_params(
    phase: str, specs: Iterable[BenchSpec] | None = None
) -> list[tuple[BenchSpec, int]]:
    """
    Flatten ``(spec, value)`` pairs for one phase — the pytest parametrize
    source. ``specs`` defaults to every model and pattern in the suite.
    """
    specs = all_specs() if specs is None else specs
    return [
        (spec, value)
        for spec in specs
        if spec.applies_to(phase)
        for value in spec.sweep
    ]


def spec_param_id(name: str, axis: str, value: object) -> str:
    """
    The ``<name>-<axis>=<value>`` fragment that fills a test id's ``[...]``.

    Single source of truth for the parametrize-id shape — the pytest param
    ids and the solver-handoff ids all build on it.
    """
    return f"{name}-{axis}={value}"
