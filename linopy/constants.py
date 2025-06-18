#!/usr/bin/env python3
"""
Linopy module for defining constant values used within the package.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


EQUAL = "="
GREATER_EQUAL = ">="
LESS_EQUAL = "<="

long_EQUAL = "=="
short_GREATER_EQUAL = ">"
short_LESS_EQUAL = "<"


SIGNS: set[str] = {EQUAL, GREATER_EQUAL, LESS_EQUAL}
SIGNS_alternative: set[str] = {long_EQUAL, short_GREATER_EQUAL, short_LESS_EQUAL}
SIGNS_pretty: dict[str, str] = {EQUAL: "=", GREATER_EQUAL: "≥", LESS_EQUAL: "≤"}

sign_replace_dict: dict[str, str] = {
    long_EQUAL: EQUAL,
    short_GREATER_EQUAL: GREATER_EQUAL,
    short_LESS_EQUAL: LESS_EQUAL,
}

TERM_DIM = "_term"
STACKED_TERM_DIM = "_stacked_term"
GROUPED_TERM_DIM = "_grouped_term"
GROUP_DIM = "_group"
FACTOR_DIM = "_factor"
CONCAT_DIM = "_concat"
HELPER_DIMS: list[str] = [
    TERM_DIM,
    STACKED_TERM_DIM,
    GROUPED_TERM_DIM,
    FACTOR_DIM,
    CONCAT_DIM,
]


class ModelStatus(Enum):
    """
    Model status.

    The set of possible model status is a superset of the solver status
    set.
    """

    ok = "ok"
    warning = "warning"
    error = "error"
    aborted = "aborted"
    unknown = "unknown"
    initialized = "initialized"


class SolverStatus(Enum):
    """
    Solver status.
    """

    ok = "ok"
    warning = "warning"
    error = "error"
    aborted = "aborted"
    unknown = "unknown"

    @classmethod
    def process(cls, status: str) -> "SolverStatus":
        try:
            return cls(status)
        except ValueError:
            return cls("unknown")

    @classmethod
    def from_termination_condition(
        cls, termination_condition: "TerminationCondition"
    ) -> "SolverStatus":
        for status in STATUS_TO_TERMINATION_CONDITION_MAP:
            if termination_condition in STATUS_TO_TERMINATION_CONDITION_MAP[status]:
                return status
        return cls("unknown")


class TerminationCondition(Enum):
    """
    Termination condition of the solver.
    """

    # UNKNOWN
    unknown = "unknown"

    # OK
    optimal = "optimal"
    time_limit = "time_limit"
    iteration_limit = "iteration_limit"
    terminated_by_limit = "terminated_by_limit"
    suboptimal = "suboptimal"
    imprecise = "imprecise"

    # WARNING
    unbounded = "unbounded"
    infeasible = "infeasible"
    infeasible_or_unbounded = "infeasible_or_unbounded"
    other = "other"

    # ERROR
    internal_solver_error = "internal_solver_error"
    error = "error"

    # ABORTED
    user_interrupt = "user_interrupt"
    resource_interrupt = "resource_interrupt"
    licensing_problems = "licensing_problems"

    @classmethod
    def process(
        cls, termination_condition: Union["TerminationCondition", str]
    ) -> "TerminationCondition":
        if isinstance(termination_condition, TerminationCondition):
            termination_condition = termination_condition.value
        try:
            return cls(termination_condition)
        except ValueError:
            return cls("unknown")


STATUS_TO_TERMINATION_CONDITION_MAP: dict[SolverStatus, list[TerminationCondition]] = {
    SolverStatus.ok: [
        TerminationCondition.optimal,
        TerminationCondition.iteration_limit,
        TerminationCondition.time_limit,
        TerminationCondition.terminated_by_limit,
        TerminationCondition.suboptimal,
        TerminationCondition.imprecise,
    ],
    SolverStatus.warning: [
        TerminationCondition.unbounded,
        TerminationCondition.infeasible,
        TerminationCondition.infeasible_or_unbounded,
        TerminationCondition.other,
    ],
    SolverStatus.error: [
        TerminationCondition.internal_solver_error,
        TerminationCondition.error,
    ],
    SolverStatus.aborted: [
        TerminationCondition.user_interrupt,
        TerminationCondition.resource_interrupt,
        TerminationCondition.licensing_problems,
    ],
    SolverStatus.unknown: [TerminationCondition.unknown],
}


@dataclass
class Status:
    """
    Status and termination condition of the solver.
    """

    status: SolverStatus
    termination_condition: TerminationCondition
    legacy_status: tuple[str, str] | str = ""

    @classmethod
    def process(cls, status: str, termination_condition: str) -> "Status":
        return cls(
            status=SolverStatus.process(status),
            termination_condition=TerminationCondition.process(termination_condition),
            legacy_status=(status, termination_condition),
        )

    @classmethod
    def from_termination_condition(
        cls, termination_condition: Union["TerminationCondition", str]
    ) -> "Status":
        termination_condition = TerminationCondition.process(termination_condition)
        solver_status = SolverStatus.from_termination_condition(termination_condition)
        return cls(solver_status, termination_condition)

    @property
    def is_ok(self) -> bool:
        return self.status == SolverStatus.ok


def _pd_series_float() -> pd.Series:
    return pd.Series(dtype=float)


@dataclass
class Solution:
    """
    Solution returned by the solver.
    """

    primal: pd.Series = field(default_factory=_pd_series_float)
    dual: pd.Series = field(default_factory=_pd_series_float)
    objective: float = field(default=np.nan)


@dataclass
class Result:
    """
    Result of the optimization.
    """

    status: Status
    solution: Solution | None = None
    solver_model: Any = None

    def __repr__(self) -> str:
        solver_model_string = (
            "not available" if self.solver_model is None else "available"
        )
        if self.solution is not None:
            solution_string = (
                f"Solution: {len(self.solution.primal)} primals, {len(self.solution.dual)} duals\n"
                f"Objective: {self.solution.objective:.2e}\n"
            )
        else:
            solution_string = "Solution: None\n"
        return (
            f"Status: {self.status.status.value}\n"
            f"Termination condition: {self.status.termination_condition.value}\n"
            + solution_string
            + f"Solver model: {solver_model_string}\n"
            f"Solver message: {self.status.legacy_status}"
        )

    def info(self) -> None:
        status = self.status

        if status.is_ok:
            if status.termination_condition == TerminationCondition.suboptimal:
                logger.warning("Optimization solution is sub-optimal: \n%s\n", self)
            else:
                logger.info(" Optimization successful: \n%s\n", self)
        else:
            logger.warning("Optimization potentially failed: \n%s\n", self)
