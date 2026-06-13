"""
Registry of benchmark models and patterns.

A :class:`ModelSpec` / :class:`PatternSpec` declares how to build a model and
which sizes / phases it runs; ``register(...)`` adds it to :data:`REGISTRY`::

    from benchmarks import REGISTRY
    model = REGISTRY["basic"].build(100)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

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


@dataclass(frozen=True, repr=False)
class ModelSpec:
    """Declarative description of one benchmark model — it runs ``sizes``."""

    name: str
    build: Callable[[int], linopy.Model]
    sizes: tuple[int, ...]
    phases: frozenset[str] = DEFAULT_PHASES
    requires: tuple[str, ...] = ()

    @property
    def sweep(self) -> tuple[int, ...]:
        """Values swept along this spec's axis (see :class:`BenchSpec`)."""
        return self.sizes

    @property
    def axis(self) -> str:
        """Sweep axis label — models scale by size."""
        return "n"

    def applies_to(self, phase: str) -> bool:
        return phase in self.phases

    def __repr__(self) -> str:
        size_range = (
            f"{self.sizes[0]}..{self.sizes[-1]}"
            if len(self.sizes) > 1
            else str(self.sizes[0])
        )
        return f"ModelSpec({self.name!r}, sizes={size_range})"


REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    """Add ``spec`` to the global registry. Returns the spec for chaining."""
    if spec.name in REGISTRY:
        raise ValueError(f"model {spec.name!r} already registered")
    unknown_phases = spec.phases - ALL_PHASES
    if unknown_phases:
        raise ValueError(
            f"model {spec.name!r}: unknown phases {sorted(unknown_phases)}"
        )
    REGISTRY[spec.name] = spec
    return spec


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


# --- Patterns ---------------------------------------------------------------

SEVERITIES: tuple[int, ...] = (0, 50, 100)  # the severity sweep every pattern runs


class BenchSpec(Protocol):
    """
    The contract models and patterns share, for axis-agnostic harness code.

    Both build a :class:`linopy.Model` from one integer dial and run through
    the same phases. They differ only in what that dial *means* — captured by
    ``sweep`` (the values) and ``axis`` (the short label, ``"n"`` vs
    ``"severity"``). Read these instead of branching on the concrete type.
    """

    @property
    def name(self) -> str: ...
    @property
    def phases(self) -> frozenset[str]: ...
    @property
    def requires(self) -> tuple[str, ...]: ...
    @property
    def build(self) -> Callable[[int], linopy.Model]: ...
    @property
    def sweep(self) -> tuple[int, ...]: ...
    @property
    def axis(self) -> str: ...
    def applies_to(self, phase: str) -> bool: ...


@dataclass(frozen=True, repr=False)
class PatternSpec:
    """
    Declarative description of one *user pattern* (modelling idiom).

    ``build(severity)`` constructs a realistic model fragment, where
    ``severity`` (0–100) dials the data shape from benign to worst-case. A
    pattern builds a complete model, so it runs the same ``phases`` as a model
    — the build-vs-export contrast is the point. It runs the ``severities``
    sweep (default ``SEVERITIES``).
    """

    name: str
    build: Callable[[int], linopy.Model]
    severities: tuple[int, ...] = SEVERITIES
    phases: frozenset[str] = DEFAULT_PHASES
    requires: tuple[str, ...] = ()

    @property
    def sweep(self) -> tuple[int, ...]:
        return self.severities

    @property
    def axis(self) -> str:
        return "severity"

    def applies_to(self, phase: str) -> bool:
        return phase in self.phases

    def __repr__(self) -> str:
        sev = ", ".join(str(s) for s in self.severities)
        return f"PatternSpec({self.name!r}, severities=[{sev}])"


PATTERNS: dict[str, PatternSpec] = {}


def register_pattern(spec: PatternSpec) -> PatternSpec:
    """Add ``spec`` to the pattern registry. Returns the spec for chaining."""
    if spec.name in PATTERNS:
        raise ValueError(f"pattern {spec.name!r} already registered")
    unknown_phases = spec.phases - ALL_PHASES
    if unknown_phases:
        raise ValueError(
            f"pattern {spec.name!r}: unknown phases {sorted(unknown_phases)}"
        )
    if not all(0 <= s <= 100 for s in spec.severities):
        raise ValueError(
            f"pattern {spec.name!r}: severities must be ints in [0, 100], "
            f"got {spec.severities}"
        )
    PATTERNS[spec.name] = spec
    return spec


def all_specs() -> list[BenchSpec]:
    """Every spec in the suite — models then patterns."""
    return [*REGISTRY.values(), *PATTERNS.values()]
