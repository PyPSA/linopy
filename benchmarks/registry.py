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

from collections.abc import Callable, Iterator
from dataclasses import dataclass

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
LP_WRITE = "lp_write"
NETCDF = "netcdf"
SOLVER_BUILD = "solver_build"  # generic Solver.from_name(..., io_api="direct")
TO_HIGHSPY = "to_highspy"
TO_GUROBIPY = "to_gurobipy"
TO_MOSEK = "to_mosek"
TO_XPRESS = "to_xpress"

ALL_PHASES = frozenset(
    {
        BUILD,
        MATRICES,
        LP_WRITE,
        NETCDF,
        SOLVER_BUILD,
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
        LP_WRITE,
        NETCDF,
        SOLVER_BUILD,
        TO_HIGHSPY,
        TO_GUROBIPY,
        TO_MOSEK,
        TO_XPRESS,
    }
)


@dataclass(frozen=True)
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

    def applies_to(self, phase: str) -> bool:
        return phase in self.phases

    def has_feature(self, feature: str) -> bool:
        return feature in self.features


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


def iter_params(phase: str) -> list[tuple[ModelSpec, int]]:
    """Pytest parametrize helper — flatten (spec, size) pairs for one phase."""
    return [
        (spec, size)
        for spec in REGISTRY.values()
        if spec.applies_to(phase)
        for size in spec.sizes
    ]


def param_ids(params: list[tuple[ModelSpec, int]]) -> list[str]:
    return [f"{spec.name}-n={size}" for spec, size in params]


def __iter__() -> Iterator[ModelSpec]:  # pragma: no cover - convenience
    return iter(REGISTRY.values())
