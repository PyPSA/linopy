#!/usr/bin/env python3
"""
Linopy module for defining constant values used within the package.
"""

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Self, TypeAlias, get_args

import numpy as np

logger = logging.getLogger(__name__)


EQUAL = "="
GREATER_EQUAL = ">="
LESS_EQUAL = "<="


class PerformanceWarning(UserWarning):
    """Warning raised when an operation triggers expensive Dataset reconstruction."""


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

STASHED_LOWER = "_stashed_lower"
STASHED_UPPER = "_stashed_upper"
STASHED_ATTRS: list[str] = [STASHED_LOWER, STASHED_UPPER]

TERM_DIM = "_term"
STACKED_TERM_DIM = "_stacked_term"

PWL_LAMBDA_SUFFIX = "_lambda"
PWL_CONVEX_SUFFIX = "_convex"
PWL_LINK_SUFFIX = "_link"
PWL_DELTA_SUFFIX = "_delta"
PWL_FILL_ORDER_SUFFIX = "_fill_order"
PWL_SEGMENT_BINARY_SUFFIX = "_segment_binary"
PWL_SELECT_SUFFIX = "_select"
PWL_ORDER_BINARY_SUFFIX = "_order_binary"
PWL_DELTA_BOUND_SUFFIX = "_delta_bound"
PWL_BINARY_ORDER_SUFFIX = "_binary_order"
PWL_ACTIVE_BOUND_SUFFIX = "_active_bound"
PWL_OUTPUT_LINK_SUFFIX = "_output_link"
PWL_CHORD_SUFFIX = "_chord"
PWL_DOMAIN_LO_SUFFIX = "_domain_lo"
PWL_DOMAIN_HI_SUFFIX = "_domain_hi"

PWL_METHOD: TypeAlias = Literal["sos2", "lp", "incremental", "auto"]
"""Allowed values for the ``method`` argument of :meth:`Model.add_piecewise_formulation`."""

PWL_METHODS: frozenset[str] = frozenset(get_args(PWL_METHOD))
"""Set of valid :data:`~linopy.constants.PWL_METHOD` values."""

PWL_CONVEXITY: TypeAlias = Literal["convex", "concave", "linear", "mixed"]
"""Possible values for :attr:`~linopy.piecewise.PiecewiseFormulation.convexity`."""

PWL_CONVEXITIES: frozenset[str] = frozenset(get_args(PWL_CONVEXITY))
"""Set of valid :data:`~linopy.constants.PWL_CONVEXITY` values."""
BREAKPOINT_DIM = "_breakpoint"
SEGMENT_DIM = "_segment"
LP_PIECE_DIM = f"{BREAKPOINT_DIM}_piece"
PWL_LINK_DIM = "_pwl_var"
GROUPED_TERM_DIM = "_grouped_term"
GROUP_DIM = "_group"
FACTOR_DIM = "_factor"
CONCAT_DIM = "_concat"
CV_DIM = "_cv"
HELPER_DIMS: list[str] = [
    TERM_DIM,
    STACKED_TERM_DIM,
    GROUPED_TERM_DIM,
    FACTOR_DIM,
    CONCAT_DIM,
    CV_DIM,
]

# SOS constraint attribute keys
SOS_TYPE_ATTR = "sos_type"
SOS_DIM_ATTR = "sos_dim"
SOS_BIG_M_ATTR = "big_m_upper"

# Indicator constraint attribute keys
INDICATOR_BINARY_VAR_ATTR = "indicator_binary_var"
INDICATOR_BINARY_VAL_ATTR = "indicator_binary_val"


class EvolvingAPIWarning(FutureWarning):
    """
    Signals a newly-added API whose details may evolve in minor releases.

    Subclasses :class:`FutureWarning` so it is visible by default.  Each
    emit prefixes its message with the affected feature (e.g.
    ``"piecewise: ..."``) so message-regex filters can target a single
    feature without hiding warnings from other features.

    Silence globally with::

        import warnings
        import linopy

        warnings.filterwarnings("ignore", category=linopy.EvolvingAPIWarning)

    Or only one feature::

        warnings.filterwarnings(
            "ignore",
            category=linopy.EvolvingAPIWarning,
            message=r"^piecewise:",
        )
    """


class ModelStatus(StrEnum):
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


class SolverStatus(StrEnum):
    """
    Solver status.
    """

    ok = "ok"
    warning = "warning"
    error = "error"
    aborted = "aborted"
    unknown = "unknown"

    @classmethod
    def process(cls, status: str) -> Self:
        try:
            return cls(status)
        except ValueError:
            return cls("unknown")

    @classmethod
    def from_termination_condition(
        cls, termination_condition: "TerminationCondition"
    ) -> Self:
        for status in STATUS_TO_TERMINATION_CONDITION_MAP:
            if termination_condition in STATUS_TO_TERMINATION_CONDITION_MAP[status]:
                return status
        return cls("unknown")


class TerminationCondition(StrEnum):
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
    def process(cls, termination_condition: Self | str) -> Self:
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
    def process(cls, status: str, termination_condition: str) -> Self:
        return cls(
            status=SolverStatus.process(status),
            termination_condition=TerminationCondition.process(termination_condition),
            legacy_status=(status, termination_condition),
        )

    @classmethod
    def from_termination_condition(
        cls, termination_condition: TerminationCondition | str | None
    ) -> Self:
        termination_condition = TerminationCondition.process(
            termination_condition if termination_condition is not None else "unknown"
        )
        solver_status = SolverStatus.from_termination_condition(termination_condition)
        return cls(solver_status, termination_condition)

    @property
    def is_ok(self) -> bool:
        return self.status == SolverStatus.ok


@dataclass
class Solution:
    """
    Solution returned by the solver.

    ``primal`` and ``dual`` are dense float arrays indexed by linopy label:
    ``primal[label]`` is the value for variable ``label``, with ``NaN`` where
    no value is available (masked labels, vars dropped by the solver, etc.).
    Each solver is responsible for emitting arrays in this label-indexed form.
    """

    primal: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    dual: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    objective: float = field(default=np.nan)


@dataclass
class SolverReport:
    """
    Solver-reported performance metrics.
    """

    runtime: float | None = None
    mip_gap: float | None = None
    dual_bound: float | None = None
    barrier_iterations: int | None = None
    simplex_iterations: int | None = None


@dataclass
class Result:
    """
    Result of the optimization.
    """

    status: Status
    solution: Solution | None = None
    solver_model: Any = None
    solver_name: str = ""
    report: SolverReport | None = None

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
        solver_name_string = f"Solver: {self.solver_name}\n" if self.solver_name else ""
        report_string = ""
        if self.report is not None:
            if self.report.runtime is not None:
                report_string += f"Runtime: {self.report.runtime:.2f}s\n"
            if self.report.mip_gap is not None:
                report_string += f"MIP gap: {self.report.mip_gap:.2e}\n"
            if self.report.dual_bound is not None:
                report_string += f"Dual bound: {self.report.dual_bound:.2e}\n"
        return (
            f"Status: {self.status.status.value}\n"
            f"Termination condition: {self.status.termination_condition.value}\n"
            + solution_string
            + solver_name_string
            + report_string
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
