"""
Linopy module for solver capability tracking.

This module provides a centralized registry of solver capabilities,
replacing scattered hardcoded checks throughout the codebase.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from enum import Enum, auto
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import TYPE_CHECKING

from packaging.specifiers import SpecifierSet

if TYPE_CHECKING:
    from collections.abc import Sequence


def _xpress_supports_gpu() -> bool:
    """Check if installed xpress version supports GPU acceleration (>=9.8.0)."""
    try:
        return package_version("xpress") in SpecifierSet(">=9.8.0")
    except PackageNotFoundError:
        return False


class SolverFeature(Enum):
    """Enumeration of all solver capabilities tracked by linopy."""

    # Model feature support
    INTEGER_VARIABLES = auto()  # Support for integer variables

    # Objective function support
    QUADRATIC_OBJECTIVE = auto()

    # I/O capabilities
    DIRECT_API = auto()  # Solve directly from Model without writing files
    LP_FILE_NAMES = auto()  # Support for named variables/constraints in LP files
    READ_MODEL_FROM_FILE = auto()  # Ability to read models from file
    SOLUTION_FILE_NOT_NEEDED = auto()  # Solver doesn't need a solution file

    # Advanced features
    GPU_ACCELERATION = auto()  # GPU-accelerated solving
    IIS_COMPUTATION = auto()  # Irreducible Infeasible Set computation

    # Special constraint types
    SOS_CONSTRAINTS = auto()  # Special Ordered Sets (SOS1/SOS2) constraints

    # Solver-specific
    SOLVER_ATTRIBUTE_ACCESS = auto()  # Direct access to solver variable attributes


@dataclass(frozen=True)
class SolverInfo:
    """Information about a solver's capabilities."""

    name: str
    features: frozenset[SolverFeature]
    display_name: str = ""

    def __post_init__(self) -> None:
        if not self.display_name:
            object.__setattr__(self, "display_name", self.name.upper())

    def supports(self, feature: SolverFeature) -> bool:
        """Check if this solver supports a given feature."""
        return feature in self.features


# Define all solver capabilities
SOLVER_REGISTRY: dict[str, SolverInfo] = {
    "gurobi": SolverInfo(
        name="gurobi",
        display_name="Gurobi",
        features=frozenset(
            {
                SolverFeature.INTEGER_VARIABLES,
                SolverFeature.QUADRATIC_OBJECTIVE,
                SolverFeature.DIRECT_API,
                SolverFeature.LP_FILE_NAMES,
                SolverFeature.READ_MODEL_FROM_FILE,
                SolverFeature.SOLUTION_FILE_NOT_NEEDED,
                SolverFeature.IIS_COMPUTATION,
                SolverFeature.SOS_CONSTRAINTS,
                SolverFeature.SOLVER_ATTRIBUTE_ACCESS,
            }
        ),
    ),
    "highs": SolverInfo(
        name="highs",
        display_name="HiGHS",
        features=frozenset(
            {
                SolverFeature.INTEGER_VARIABLES,
                SolverFeature.QUADRATIC_OBJECTIVE,
                SolverFeature.DIRECT_API,
                SolverFeature.LP_FILE_NAMES,
                SolverFeature.READ_MODEL_FROM_FILE,
                SolverFeature.SOLUTION_FILE_NOT_NEEDED,
            }
        ),
    ),
    "glpk": SolverInfo(
        name="glpk",
        display_name="GLPK",
        features=frozenset(
            {
                SolverFeature.INTEGER_VARIABLES,
                SolverFeature.READ_MODEL_FROM_FILE,
            }
        ),  # No LP_FILE_NAMES support
    ),
    "cbc": SolverInfo(
        name="cbc",
        display_name="CBC",
        features=frozenset(
            {
                SolverFeature.INTEGER_VARIABLES,
                SolverFeature.READ_MODEL_FROM_FILE,
            }
        ),  # No LP_FILE_NAMES support
    ),
    "cplex": SolverInfo(
        name="cplex",
        display_name="CPLEX",
        features=frozenset(
            {
                SolverFeature.INTEGER_VARIABLES,
                SolverFeature.QUADRATIC_OBJECTIVE,
                SolverFeature.LP_FILE_NAMES,
                SolverFeature.READ_MODEL_FROM_FILE,
                SolverFeature.SOS_CONSTRAINTS,
            }
        ),
    ),
    "xpress": SolverInfo(
        name="xpress",
        display_name="FICO Xpress",
        features=frozenset(
            {
                SolverFeature.INTEGER_VARIABLES,
                SolverFeature.QUADRATIC_OBJECTIVE,
                SolverFeature.LP_FILE_NAMES,
                SolverFeature.READ_MODEL_FROM_FILE,
                SolverFeature.SOLUTION_FILE_NOT_NEEDED,
                SolverFeature.GPU_ACCELERATION,
                SolverFeature.IIS_COMPUTATION,
            }
            if _xpress_supports_gpu()
            else {
                SolverFeature.INTEGER_VARIABLES,
                SolverFeature.QUADRATIC_OBJECTIVE,
                SolverFeature.LP_FILE_NAMES,
                SolverFeature.READ_MODEL_FROM_FILE,
                SolverFeature.SOLUTION_FILE_NOT_NEEDED,
                SolverFeature.IIS_COMPUTATION,
            }
        ),
    ),
    "scip": SolverInfo(
        name="scip",
        display_name="SCIP",
        features=frozenset(
            {
                SolverFeature.INTEGER_VARIABLES,
                SolverFeature.LP_FILE_NAMES,
                SolverFeature.READ_MODEL_FROM_FILE,
                SolverFeature.SOLUTION_FILE_NOT_NEEDED,
            }
            if platform.system() == "Windows"
            else {
                SolverFeature.INTEGER_VARIABLES,
                SolverFeature.QUADRATIC_OBJECTIVE,
                SolverFeature.LP_FILE_NAMES,
                SolverFeature.READ_MODEL_FROM_FILE,
                SolverFeature.SOLUTION_FILE_NOT_NEEDED,
            }
            # SCIP has a bug with quadratic models on Windows, see:
            # https://github.com/PyPSA/linopy/actions/runs/7615240686/job/20739454099?pr=78
        ),
    ),
    "mosek": SolverInfo(
        name="mosek",
        display_name="MOSEK",
        features=frozenset(
            {
                SolverFeature.INTEGER_VARIABLES,
                SolverFeature.QUADRATIC_OBJECTIVE,
                SolverFeature.DIRECT_API,
                SolverFeature.LP_FILE_NAMES,
                SolverFeature.READ_MODEL_FROM_FILE,
                SolverFeature.SOLUTION_FILE_NOT_NEEDED,
            }
        ),
    ),
    "copt": SolverInfo(
        name="copt",
        display_name="COPT",
        features=frozenset(
            {
                SolverFeature.INTEGER_VARIABLES,
                SolverFeature.QUADRATIC_OBJECTIVE,
                SolverFeature.LP_FILE_NAMES,
                SolverFeature.READ_MODEL_FROM_FILE,
                SolverFeature.SOLUTION_FILE_NOT_NEEDED,
            }
        ),
    ),
    "mindopt": SolverInfo(
        name="mindopt",
        display_name="MindOpt",
        features=frozenset(
            {
                SolverFeature.INTEGER_VARIABLES,
                SolverFeature.QUADRATIC_OBJECTIVE,
                SolverFeature.LP_FILE_NAMES,
                SolverFeature.READ_MODEL_FROM_FILE,
                SolverFeature.SOLUTION_FILE_NOT_NEEDED,
            }
        ),
    ),
    "cupdlpx": SolverInfo(
        name="cupdlpx",
        display_name="cuPDLPx",
        features=frozenset(
            {
                SolverFeature.DIRECT_API,
                SolverFeature.GPU_ACCELERATION,
                SolverFeature.SOLUTION_FILE_NOT_NEEDED,
            }
        ),
    ),
}


def solver_supports(solver_name: str, feature: SolverFeature) -> bool:
    """
    Check if a solver supports a given feature.

    Parameters
    ----------
    solver_name : str
        Name of the solver (e.g., "gurobi", "highs")
    feature : SolverFeature
        The feature to check for

    Returns
    -------
    bool
        True if the solver supports the feature, False otherwise.
        Returns False for unknown solvers.
    """
    if solver_name not in SOLVER_REGISTRY:
        return False
    return SOLVER_REGISTRY[solver_name].supports(feature)


def get_solvers_with_feature(feature: SolverFeature) -> list[str]:
    """
    Get all solvers that support a given feature.

    Parameters
    ----------
    feature : SolverFeature
        The feature to filter by

    Returns
    -------
    list[str]
        List of solver names supporting the feature
    """
    return [name for name, info in SOLVER_REGISTRY.items() if info.supports(feature)]


def get_available_solvers_with_feature(
    feature: SolverFeature, available_solvers: Sequence[str]
) -> list[str]:
    """
    Get installed solvers that support a given feature.

    Parameters
    ----------
    feature : SolverFeature
        The feature to filter by
    available_solvers : Sequence[str]
        List of currently available/installed solvers

    Returns
    -------
    list[str]
        List of installed solver names supporting the feature
    """
    return [s for s in get_solvers_with_feature(feature) if s in available_solvers]
