"""
Reusable registry of benchmark models.

A :class:`ModelSpec` captures everything needed to drive a model through the
benchmark suite *and* to use it from any other test or script:

- ``build(size) -> linopy.Model``  the actual builder
- ``sizes``                        canonical sizes the model has been tuned for
- ``features``                     what kinds of variables / constraints it uses
- ``phases``                       which benchmark phases apply (lp_write, to_highspy, ...)
- ``quick_threshold``              max size to keep under ``pytest --quick``
- ``requires``                     extra modules to ``pytest.importorskip``

Pattern for downstream use::

    from benchmarks import REGISTRY
    model = REGISTRY["basic"].build(100)

    # Or pick a subset by feature/phase:
    from benchmarks import filter_by, QUADRATIC
    qp_specs = filter_by(has_feature=QUADRATIC)
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


@dataclass(frozen=True, repr=False)
class ModelSpec:
    """
    Declarative description of one benchmark model.

    Three size tiers gate the cost of a default ``pytest benchmarks/`` run:

    - ``size <= quick_threshold``: included under ``--quick`` (smoke / CI).
    - ``size <= long_threshold``: included by default (medium-cost regression).
    - ``size >  long_threshold``: only included under ``--long`` (full sweep).

    Without explicit values, both thresholds default to "no cap".
    """

    name: str
    build: Callable[[int], linopy.Model]
    sizes: tuple[int, ...]
    features: frozenset[str] = frozenset({CONTINUOUS})
    phases: frozenset[str] = DEFAULT_PHASES
    quick_threshold: int = 10**9
    long_threshold: int = 10**9
    requires: tuple[str, ...] = ()
    description: str = ""

    @property
    def sweep(self) -> tuple[int, ...]:
        """Values swept along this spec's axis (see :class:`BenchSpec`)."""
        return self.sizes

    @property
    def axis(self) -> str:
        """Short x-axis label for the sweep dial: a model scales by size."""
        return "n"

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

    def _repr_html_(self) -> str:
        # Rich rendering for Jupyter — a compact two-column table.
        rows = [
            ("name", self.name),
            ("features", ", ".join(sorted(self.features))),
            ("sizes", ", ".join(str(s) for s in self.sizes)),
            ("phases", ", ".join(sorted(self.phases))),
            ("quick_threshold", self.quick_threshold),
            ("long_threshold", self.long_threshold),
            ("requires", ", ".join(self.requires) or "—"),
        ]
        body = "".join(
            f"<tr><th style='text-align:left;padding-right:1em'>{k}</th>"
            f"<td>{v}</td></tr>"
            for k, v in rows
        )
        return (
            f"<b>ModelSpec</b> <code>{self.name}</code>"
            f"<table style='font-size:90%'>{body}</table>"
        )


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


def get(name: str) -> ModelSpec:
    return REGISTRY[name]


def filter_by(
    *,
    has_feature: str | None = None,
    has_phase: str | None = None,
) -> list[ModelSpec]:
    out = []
    for spec in REGISTRY.values():
        if has_feature is not None and not spec.has_feature(has_feature):
            continue
        if has_phase is not None and not spec.applies_to(has_phase):
            continue
        out.append(spec)
    return out


def iter_params(
    phase: str, specs: Iterable[BenchSpec] | None = None
) -> list[tuple[BenchSpec, int]]:
    """
    Pytest parametrize helper — flatten ``(spec, value)`` pairs for one phase.

    ``specs`` defaults to every spec in the suite — models *and* patterns — so a
    phase driver picks both up automatically. Works over any :class:`BenchSpec`
    via its ``sweep`` axis, so models (size) and patterns (severity) share one
    helper. Pass an explicit collection (e.g. ``PATTERNS.values()``) to narrow.
    """
    specs = all_specs() if specs is None else specs
    return [
        (spec, value)
        for spec in specs
        if spec.applies_to(phase)
        for value in spec.sweep
    ]


def param_ids(params: list[tuple[BenchSpec, int]]) -> list[str]:
    from benchmarks.snapshot import spec_param_id

    return [spec_param_id(spec.name, spec.axis, value) for spec, value in params]


# --- Patterns ---------------------------------------------------------------

DEFAULT_SEVERITIES: tuple[int, ...] = (0, 50, 100)


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
    def quick_threshold(self) -> int: ...
    @property
    def long_threshold(self) -> int: ...
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

    ``build(severity)`` constructs a small, realistic model fragment, where
    ``severity`` is an int in ``[0, 100]`` dialling the data shape from benign
    (0) to worst-case (100). The harness measures the act of building it (time
    + peak memory) across the ``severities`` sweep. ``description`` documents
    the dial — by convention ``"<what it varies> — 0: <benign>, 100: <worst>"``
    — and doubles as the plot caption.

    A pattern builds a complete model, so it runs the same ``phases`` as a model
    by default — the build-vs-export contrast (does the dense-``_term`` bloat
    reach the matrix / LP file, or collapse?) is the point. The full severity
    range runs by default; ``--quick`` keeps everything up to the midpoint
    (``{0, 25, 50}``) so smoke exercises real pathology, not just the benign
    endpoint, while skipping the heaviest builds.
    """

    name: str
    build: Callable[[int], linopy.Model]
    description: str
    severities: tuple[int, ...] = DEFAULT_SEVERITIES
    phases: frozenset[str] = DEFAULT_PHASES
    requires: tuple[str, ...] = ()
    quick_threshold: int = 50
    long_threshold: int = 10**9

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

    def _repr_html_(self) -> str:
        rows = [
            ("name", self.name),
            ("description", self.description),
            ("severities", ", ".join(str(s) for s in self.severities)),
            ("phases", ", ".join(sorted(self.phases))),
            ("requires", ", ".join(self.requires) or "—"),
        ]
        body = "".join(
            f"<tr><th style='text-align:left;padding-right:1em'>{k}</th>"
            f"<td>{v}</td></tr>"
            for k, v in rows
        )
        return (
            f"<b>PatternSpec</b> <code>{self.name}</code>"
            f"<table style='font-size:90%'>{body}</table>"
        )


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


def get_pattern(name: str) -> PatternSpec:
    return PATTERNS[name]


def all_specs() -> list[BenchSpec]:
    """Every spec in the suite — models then patterns."""
    return [*REGISTRY.values(), *PATTERNS.values()]
