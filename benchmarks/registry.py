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

# --- Feature tags -----------------------------------------------------------

CONTINUOUS = "continuous"
BINARY = "binary"
INTEGER = "integer"
QUADRATIC = "quadratic"
SOS = "sos"
PIECEWISE = "piecewise"
MASKED = "masked"

ALL_FEATURES = frozenset(
    {CONTINUOUS, BINARY, INTEGER, QUADRATIC, SOS, PIECEWISE, MASKED}
)

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

# Phases every "well-behaved LP / MILP" can do. Models with features the
# default solvers can't ingest natively (e.g. native SOS for HiGHS) override
# this with a narrower set.
DEFAULT_PHASES = frozenset(
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


def _quick_subset(values: tuple[int, ...]) -> tuple[int, ...]:
    """The ``--quick`` subset of a sweep: first, middle, last (deduped)."""
    if not values:
        return ()
    picks = (values[0], values[len(values) // 2], values[-1])
    return tuple(dict.fromkeys(picks))


@dataclass(frozen=True, repr=False)
class ModelSpec:
    """
    Declarative description of one benchmark model.

    Three size tiers gate run cost (each a subset of ``sizes``): ``--quick``
    runs ``quick_sizes`` (``()`` opts out), the default runs ``sizes`` minus
    ``long_sizes``, and ``--long`` runs every size.
    """

    name: str
    build: Callable[[int], linopy.Model]
    sizes: tuple[int, ...]
    features: frozenset[str] = frozenset({CONTINUOUS})
    phases: frozenset[str] = DEFAULT_PHASES
    quick_sizes: tuple[int, ...] | None = None
    long_sizes: tuple[int, ...] = ()
    requires: tuple[str, ...] = ()
    description: str = ""

    @property
    def sweep(self) -> tuple[int, ...]:
        """Values swept along this spec's axis (see :class:`BenchSpec`)."""
        return self.sizes

    @property
    def axis(self) -> str:
        """Sweep axis label — models scale by size."""
        return "n"

    @property
    def quick_subset(self) -> tuple[int, ...]:
        """
        ``--quick`` sizes — ``quick_sizes`` (``()`` opts out); falls back to
        first/mid/last of ``sizes`` if unset.
        """
        return (
            self.quick_sizes
            if self.quick_sizes is not None
            else _quick_subset(self.sweep)
        )

    def applies_to(self, phase: str) -> bool:
        return phase in self.phases

    def has_feature(self, feature: str) -> bool:
        return feature in self.features

    def __repr__(self) -> str:
        feats = ",".join(sorted(self.features))
        size_range = (
            f"{self.sizes[0]}..{self.sizes[-1]}"
            if len(self.sizes) > 1
            else str(self.sizes[0])
        )
        return f"ModelSpec({self.name!r}, features={{{feats}}}, sizes={size_range})"


REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    """Add ``spec`` to the global registry. Returns the spec for chaining."""
    if spec.name in REGISTRY:
        raise ValueError(f"model {spec.name!r} already registered")
    unknown_features = spec.features - ALL_FEATURES
    if unknown_features:
        raise ValueError(
            f"model {spec.name!r}: unknown features {sorted(unknown_features)}"
        )
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

DEFAULT_SEVERITIES: tuple[int, ...] = (0, 25, 50, 75, 100)  # full sweep / --long
QUICK_SEVERITIES: tuple[int, ...] = (0, 50, 100)  # --quick (per-PR)


class BenchSpec(Protocol):
    """
    The contract models and patterns share, for axis-agnostic harness code.

    Both build a :class:`linopy.Model` from one integer dial and run through
    the same phases. They differ only in what that dial *means* — captured by
    ``sweep`` (the values), ``axis`` (the short label, ``"n"`` vs
    ``"severity"``), and ``description`` (the human one-liner). Read these
    instead of branching on the concrete type.
    """

    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def phases(self) -> frozenset[str]: ...
    @property
    def requires(self) -> tuple[str, ...]: ...
    @property
    def quick_subset(self) -> tuple[int, ...]: ...
    @property
    def long_sizes(self) -> tuple[int, ...]: ...
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
    — the build-vs-export contrast is the point. ``--quick`` keeps
    ``QUICK_SEVERITIES`` ``(0, 50, 100)``; ``description`` documents the dial.
    """

    name: str
    build: Callable[[int], linopy.Model]
    description: str
    severities: tuple[int, ...] = DEFAULT_SEVERITIES
    phases: frozenset[str] = DEFAULT_PHASES
    requires: tuple[str, ...] = ()
    quick_sizes: tuple[int, ...] | None = None
    long_sizes: tuple[int, ...] = ()

    @property
    def sweep(self) -> tuple[int, ...]:
        return self.severities

    @property
    def axis(self) -> str:
        return "severity"

    @property
    def quick_subset(self) -> tuple[int, ...]:
        """
        ``--quick`` severities — ``quick_sizes`` if set, else
        ``QUICK_SEVERITIES`` ``(0, 50, 100)``.
        """
        return self.quick_sizes if self.quick_sizes is not None else QUICK_SEVERITIES

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


def skip_reason(
    spec: BenchSpec,
    value: int,
    *,
    quick: bool = False,
    long: bool = False,
    sizes: tuple[int, ...] = (),
    severities: tuple[int, ...] = (),
) -> str | None:
    """
    Why ``(spec, value)`` is excluded under this selection, or ``None`` to run.

    Single source of truth for size/severity selection, applied by
    ``conftest.maybe_skip``. Precedence, most specific first:

    - a manual axis list (``sizes`` for models, ``severities`` for patterns)
      → run only those values;
    - ``--quick`` → only ``spec.quick_subset``;
    - default → skip values in ``spec.long_sizes`` (the heaviest, held back);
    - ``--long`` → everything.
    """
    manual = severities if spec.axis == "severity" else sizes
    if manual:
        return None if value in manual else f"{spec.axis}={value} not selected"
    if quick:
        if value not in spec.quick_subset:
            return f"--quick: skipping {spec.name} {spec.axis}={value}"
        return None
    if not long and value in spec.long_sizes:
        return f"long sweep needs --long: skipping {spec.name} {spec.axis}={value}"
    return None
