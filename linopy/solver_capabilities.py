"""
Back-compat shim for legacy solver-capability imports.

Capability data is declared on each `Solver` subclass in `linopy.solvers`.
Prefer `Solver.features` / `Solver.supports()` over the helpers in this module.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from linopy.solvers import Solver, SolverFeature

__all__ = (
    "SOLVER_REGISTRY",
    "SolverFeature",
    "SolverInfo",
    "get_available_solvers_with_feature",
    "get_solvers_with_feature",
    "solver_supports",
)


def __getattr__(name: str) -> object:
    if name == "SolverFeature":
        from linopy import solvers as _solvers_mod

        return _solvers_mod.SolverFeature
    raise AttributeError(name)


@dataclass(frozen=True)
class SolverInfo:
    """Legacy view of a solver's capabilities. Prefer Solver.features / Solver.supports()."""

    name: str
    features: frozenset[Enum]
    display_name: str = ""

    def __post_init__(self) -> None:
        if not self.display_name:
            object.__setattr__(self, "display_name", self.name.upper())

    def supports(self, feature: Enum) -> bool:
        return feature in self.features


def _solver_class(name: str) -> type[Solver] | None:
    from linopy import solvers as _solvers_mod

    try:
        return getattr(_solvers_mod, _solvers_mod.SolverName(name).name, None)
    except ValueError:
        return None


def solver_supports(solver_name: str, feature: SolverFeature) -> bool:
    cls = _solver_class(solver_name)
    return cls is not None and cls.supports(feature)


def get_solvers_with_feature(feature: SolverFeature) -> list[str]:
    from linopy.solvers import SolverName

    return [n.value for n in SolverName if solver_supports(n.value, feature)]


def get_available_solvers_with_feature(
    feature: SolverFeature, available_solvers: Sequence[str]
) -> list[str]:
    return [s for s in get_solvers_with_feature(feature) if s in available_solvers]


class _LazyRegistry(Mapping[str, SolverInfo]):
    def __getitem__(self, key: str) -> SolverInfo:
        cls = _solver_class(key)
        if cls is None:
            raise KeyError(key)
        return SolverInfo(
            name=key,
            features=cls.supported_features(),
            display_name=cls.display_name,
        )

    def __iter__(self) -> Iterator[str]:
        from linopy.solvers import SolverName

        return (n.value for n in SolverName)

    def __len__(self) -> int:
        from linopy.solvers import SolverName

        return len(SolverName)


SOLVER_REGISTRY: Mapping[str, SolverInfo] = _LazyRegistry()
