#!/usr/bin/env python3
"""
Linopy module for solving lp files with different solvers.
"""

from __future__ import annotations

import contextlib
import enum
import io
import logging
import os
import re
import subprocess as sub
import sys
import threading
import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Callable, Generator
from enum import Enum, auto
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

import numpy as np
import pandas as pd
import xarray as xr
from packaging.specifiers import SpecifierSet
from packaging.version import parse as parse_version
from scipy.sparse import tril, triu

import linopy.io
from linopy.common import count_initial_letters
from linopy.constants import (
    SOS_DIM_ATTR,
    SOS_TYPE_ATTR,
    Result,
    Solution,
    SolverReport,
    SolverStatus,
    Status,
    TerminationCondition,
)


def _parse_int_label(name: str) -> int:
    """Strip leading non-digits and parse the integer label."""
    s = str(name)
    cutoff = count_initial_letters(s)
    try:
        return int(s[cutoff:])
    except ValueError:
        return int(re.sub(r".*#", "", s))


def _names_to_labels(names: Any) -> np.ndarray:
    """Vectorised conversion of solver-provided names to integer labels."""
    return np.fromiter(
        (_parse_int_label(n) for n in names), dtype=np.int64, count=len(names)
    )


class SolverFeature(Enum):
    """Enumeration of all solver capabilities tracked by linopy."""

    INTEGER_VARIABLES = auto()
    QUADRATIC_OBJECTIVE = auto()
    DIRECT_API = auto()
    LP_FILE_NAMES = auto()
    READ_MODEL_FROM_FILE = auto()
    SOLUTION_FILE_NOT_NEEDED = auto()
    GPU_ACCELERATION = auto()
    IIS_COMPUTATION = auto()
    SOS_CONSTRAINTS = auto()
    SEMI_CONTINUOUS_VARIABLES = auto()
    SOLVER_ATTRIBUTE_ACCESS = auto()


def _installed_version_in(pkg: str, spec: str) -> bool:
    """Check whether the installed version of `pkg` satisfies `spec`."""
    try:
        return package_version(pkg) in SpecifierSet(spec)
    except PackageNotFoundError:
        return False


if TYPE_CHECKING:
    import cupdlpx
    import gurobipy
    import highspy
    import mosek

    from linopy.model import Model

EnvType = TypeVar("EnvType")

FILE_IO_APIS = ["lp", "lp-polars", "mps"]
IO_APIS = FILE_IO_APIS + ["direct"]

available_solvers: list[str] = []

which = "where" if os.name == "nt" else "which"


def _run_highs_with_keyboard_interrupt(h: Any) -> None:
    """
    Run `highspy.Highs.run()` while ensuring Ctrl-C cancels the solve.

    HiGHS can run for a long time inside a C-extension call. Running it in a
    worker thread allows the main thread to reliably receive KeyboardInterrupt
    and signal HiGHS to stop via `cancelSolve()`.
    """

    handle_keyboard_interrupt = getattr(h, "HandleKeyboardInterrupt", None)
    handle_user_interrupt = getattr(h, "HandleUserInterrupt", None)

    old_handle_keyboard_interrupt = (
        handle_keyboard_interrupt if not callable(handle_keyboard_interrupt) else None
    )
    old_handle_user_interrupt = (
        handle_user_interrupt if not callable(handle_user_interrupt) else None
    )

    try:
        if callable(handle_keyboard_interrupt):
            handle_keyboard_interrupt(True)
        elif handle_keyboard_interrupt is not None:
            h.HandleKeyboardInterrupt = True

        if callable(handle_user_interrupt):
            handle_user_interrupt(True)
        elif handle_user_interrupt is not None:
            h.HandleUserInterrupt = True

        finished = threading.Event()
        run_error: BaseException | None = None

        def _target() -> None:
            nonlocal run_error
            try:
                h.run()
            except BaseException as exc:  # pragma: no cover
                run_error = exc
            finally:
                finished.set()

        thread = threading.Thread(target=_target, name="linopy-highs-run", daemon=True)
        thread.start()

        try:
            while not finished.wait(0.1):
                pass
        except KeyboardInterrupt:
            cancel_solve = getattr(h, "cancelSolve", None)
            if callable(cancel_solve):
                with contextlib.suppress(Exception):
                    cancel_solve()
            while not finished.wait(0.1):
                pass
            raise

        if run_error is not None:
            raise run_error
    finally:
        if old_handle_keyboard_interrupt is not None:
            h.HandleKeyboardInterrupt = old_handle_keyboard_interrupt
        if old_handle_user_interrupt is not None:
            h.HandleUserInterrupt = old_handle_user_interrupt


# the first available solver will be the default solver
with contextlib.suppress(ModuleNotFoundError):
    import gurobipy

    available_solvers.append("gurobi")
with contextlib.suppress(ModuleNotFoundError):
    _new_highspy_mps_layout = None
    import highspy

    available_solvers.append("highs")
    from importlib.metadata import version

    if parse_version(version("highspy")) < parse_version("1.7.1"):
        # Fallback if parse_version is not available or version string is invalid
        _new_highspy_mps_layout = False
    else:
        _new_highspy_mps_layout = True

if sub.run([which, "glpsol"], stdout=sub.DEVNULL, stderr=sub.STDOUT).returncode == 0:
    available_solvers.append("glpk")


if sub.run([which, "cbc"], stdout=sub.DEVNULL, stderr=sub.STDOUT).returncode == 0:
    available_solvers.append("cbc")

with contextlib.suppress(ModuleNotFoundError):
    import pyscipopt as scip

    available_solvers.append("scip")

with contextlib.suppress(ModuleNotFoundError):
    import cplex

    available_solvers.append("cplex")

with contextlib.suppress(ModuleNotFoundError, ImportError):
    import xpress

    available_solvers.append("xpress")

    # xpress.Namespaces was added in xpress 9.6
    try:
        from xpress import Namespaces as xpress_Namespaces
    except ImportError:

        class xpress_Namespaces:  # type: ignore[no-redef]
            ROW = 1
            COLUMN = 2
            SET = 3


with contextlib.suppress(ModuleNotFoundError, ImportError):
    import knitro

    with contextlib.suppress(Exception):
        kc = knitro.KN_new()
        knitro.KN_free(kc)
        available_solvers.append("knitro")

with contextlib.suppress(ModuleNotFoundError):
    import mosek

    with contextlib.suppress(mosek.Error):
        t = mosek.Task()
        t.optimize()

        available_solvers.append("mosek")

with contextlib.suppress(ModuleNotFoundError):
    import mindoptpy

    with contextlib.suppress(mindoptpy.MindoptError):
        mindoptpy.Env()

        available_solvers.append("mindopt")

with contextlib.suppress(ModuleNotFoundError):
    import coptpy

    try:
        coptpy.Envr()
        available_solvers.append("copt")
    except coptpy.CoptError:
        pass

with contextlib.suppress(ModuleNotFoundError):
    import cupdlpx

    try:
        cupdlpx.Model(np.array([0.0]), np.array([[0.0]]), None, None)
        available_solvers.append("cupdlpx")
    except ImportError:
        pass


logger = logging.getLogger(__name__)


io_structure = dict(
    lp_file={
        "gurobi",
        "xpress",
        "cbc",
        "glpk",
        "cplex",
        "mosek",
        "mindopt",
    },
    blocks={"pips"},
)


# using enum to match solver subclasses with names
class SolverName(enum.Enum):
    CBC = "cbc"
    GLPK = "glpk"
    Highs = "highs"
    Cplex = "cplex"
    Gurobi = "gurobi"
    SCIP = "scip"
    Xpress = "xpress"
    Knitro = "knitro"
    Mosek = "mosek"
    COPT = "copt"
    MindOpt = "mindopt"
    PIPS = "pips"
    cuPDLPx = "cupdlpx"


def path_to_string(path: Path) -> str:
    """
    Convert a pathlib.Path to a string.
    """
    return str(path.resolve())


def read_sense_from_problem_file(problem_fn: Path | str) -> str:
    with open(problem_fn) as file:
        f = file.read()
    file_format = read_io_api_from_problem_file(problem_fn)
    if file_format == "lp":
        return "min" if "min" in f.lower() else "max"
    elif file_format == "mps":
        return "max" if "OBJSENSE\n  MAX\n" in f else "min"
    else:
        msg = "Unsupported problem file format."
        raise ValueError(msg)


def read_io_api_from_problem_file(problem_fn: Path | str) -> str:
    if isinstance(problem_fn, Path):
        return problem_fn.suffix[1:]
    else:
        return problem_fn.split(".")[-1]


def maybe_adjust_objective_sign(
    solution: Solution, io_api: str | None, sense: str | None
) -> Solution:
    if sense == "min":
        return solution
    if np.isnan(solution.objective):
        return solution
    if io_api == "mps" and not _new_highspy_mps_layout:
        logger.info(
            "Adjusting objective sign due to switched coefficients in MPS file."
        )
        solution.objective *= -1
    return solution


class Solver(ABC, Generic[EnvType]):
    """
    Abstract base class for solving a given linear problem.

    All relevant functions are passed on to the specific solver subclasses.
    Subclasses must implement the `solve_problem_from_model()` and
    `solve_problem_from_file()` methods.
    """

    display_name: ClassVar[str] = ""
    features: ClassVar[frozenset[SolverFeature]] = frozenset()

    @classmethod
    def runtime_features(cls) -> frozenset[SolverFeature]:
        """
        Features whose availability depends on the installed solver version
        or runtime environment. Override in subclasses; the default is empty.
        """
        return frozenset()

    @classmethod
    def supported_features(cls) -> frozenset[SolverFeature]:
        """All features supported by this solver, static plus runtime."""
        return cls.features | cls.runtime_features()

    @classmethod
    def supports(cls, feature: SolverFeature) -> bool:
        """Check if this solver supports a given feature."""
        return feature in cls.features or feature in cls.runtime_features()

    def __init__(
        self,
        **solver_options: Any,
    ) -> None:
        self.options: dict[str, Any] = solver_options
        self.solver_options: dict[str, Any] = solver_options
        self.status: Status | None = None
        self.solution: Solution | None = None
        self.report: SolverReport | None = None
        self.solver_model: Any = None
        self.io_api: str | None = None
        self.sense: str | None = None
        self.env: Any = None
        self._env_stack: contextlib.ExitStack | None = None

        if self.solver_name.value not in available_solvers:
            msg = f"Solver package for '{self.solver_name.value}' is not installed. Please install first to initialize solver instance."
            raise ImportError(msg)

    def to_solver_model(self, model: Model, **kwargs: Any) -> Any:
        raise NotImplementedError

    def update_solver_model(self, model: Model, **kwargs: Any) -> None:
        raise NotImplementedError

    def run(self) -> Result:
        if self.solver_model is None:
            raise RuntimeError("call to_solver_model first")
        if self.sense is None:
            raise RuntimeError("sense not set; call to_solver_model first")
        return self._run()

    def _run(self) -> Result:
        raise NotImplementedError

    def close(self) -> None:
        if self._env_stack is not None:
            self._env_stack.close()
        self.env = None
        self.solver_model = None
        self._env_stack = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def __repr__(self) -> str:
        status = self.status.status.value if self.status is not None else "unsolved"
        parts = [f"name={self.solver_name.value!r}", f"status={status!r}"]
        if self.io_api is not None:
            parts.append(f"io_api={self.io_api!r}")
        if self.solver_model is not None:
            parts.append("solver_model=loaded")
        if self.env is not None:
            parts.append("env=active")
        if self.solution is not None:
            parts.append(f"objective={self.solution.objective:.4g}")
        if self.report is not None and self.report.runtime is not None:
            parts.append(f"runtime={self.report.runtime:.3g}s")
        return f"{type(self).__name__}({', '.join(parts)})"

    def _make_result(
        self,
        status: Status,
        solution: Solution | None,
        solver_model: Any = None,
        report: SolverReport | None = None,
    ) -> Result:
        self.status = status
        self.solution = solution
        self.report = report
        if solver_model is not None:
            self.solver_model = solver_model
        return Result(
            status=status,
            solution=solution,
            solver_model=solver_model,
            solver_name=self.solver_name.value,
            report=report,
        )

    def safe_get_solution(
        self, status: Status, func: Callable[[], Solution]
    ) -> Solution:
        """
        Get solution from function call, if status is unknown still try to run it.
        """
        if status.is_ok:
            return func()
        elif status.status == SolverStatus.unknown:
            try:
                logger.warning("Solution status unknown. Trying to parse solution.")
                sol = func()
                status.status = SolverStatus.ok
                logger.warning("Solution parsed successfully.")
                return sol
            except Exception as e:
                logger.error(f"Failed to parse solution: {e}")
        return Solution()

    @abstractmethod
    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: EnvType | None = None,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> Result:
        """
        Solve a linear problem directly from a linopy model.

        Subclasses that support the direct API translate the model into the
        solver's native representation and run it. Subclasses without direct
        API support must still implement this method and raise NotImplementedError.

        Parameters
        ----------
        model : linopy.Model
            Linopy model for the problem.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        env : EnvType, optional
            Solver-specific environment object (or None when not applicable).
        explicit_coordinate_names : bool, optional
            Transfer variable and constraint coordinate names to the solver
            (default: False).
        set_names : bool, optional
            Whether to set variable and constraint names (default: True).
            Setting to False can significantly speed up model export.

        Returns
        -------
        Result
        """
        pass

    @abstractmethod
    def solve_problem_from_file(
        self,
        problem_fn: Path,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: EnvType | None = None,
    ) -> Result:
        """
        Abstract method to solve a linear problem from a problem file.

        Needs to be implemented in the specific solver subclass. Even if the solver
        does not support solving from a file, this method should be implemented and
        raise a NotImplementedError.
        """
        pass

    def solve_problem(
        self,
        model: Model | None = None,
        problem_fn: Path | None = None,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: EnvType | None = None,
        explicit_coordinate_names: bool = False,
    ) -> Result:
        """
        Solve a linear problem either from a model or a problem file.

        Wraps around `self.solve_problem_from_model()` and
        `self.solve_problem_from_file()` and calls the appropriate method
        based on the input arguments (`model` or `problem_fn`).
        """
        if problem_fn is not None and model is not None:
            msg = "Both problem file and model are given. Please specify only one."
            raise ValueError(msg)
        elif model is not None:
            return self.solve_problem_from_model(
                model=model,
                solution_fn=solution_fn,
                log_fn=log_fn,
                warmstart_fn=warmstart_fn,
                basis_fn=basis_fn,
                env=env,
                explicit_coordinate_names=explicit_coordinate_names,
            )
        elif problem_fn is not None:
            return self.solve_problem_from_file(
                problem_fn=problem_fn,
                solution_fn=solution_fn,
                log_fn=log_fn,
                warmstart_fn=warmstart_fn,
                basis_fn=basis_fn,
                env=env,
            )
        else:
            msg = "No problem file or model specified."
            raise ValueError(msg)

    @property
    def solver_name(self) -> SolverName:
        return SolverName[self.__class__.__name__]


class CBC(Solver[None]):
    """
    Solver subclass for the CBC solver.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "CBC"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.READ_MODEL_FROM_FILE,
        }
    )

    def __init__(
        self,
        **solver_options: Any,
    ) -> None:
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> Result:
        msg = "Direct API not implemented for CBC"
        raise NotImplementedError(msg)

    def solve_problem_from_file(
        self,
        problem_fn: Path,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
    ) -> Result:
        """
        Solve a linear problem from a problem file using the CBC solver.

        The function reads the linear problem file and passes it to the solver.
        If the solution is successful it returns variable solutions
        and constraint dual values.

        Parameters
        ----------
        problem_fn : Path
            Path to the problem file.
        solution_fn : Path
            Path to the solution file. This is necessary for solving with CBC.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        env : None, optional
            Environment for the solver

        Returns
        -------
        Result
        """
        sense = read_sense_from_problem_file(problem_fn)
        io_api = read_io_api_from_problem_file(problem_fn)

        if solution_fn is None:
            msg = "No solution file specified. For solving with CBC this is necessary."
            raise ValueError(msg)

        if io_api == "mps" and sense == "max" and _new_highspy_mps_layout:
            msg = (
                "CBC does not support maximization in MPS format highspy versions "
                " >=1.7.1"
            )
            raise ValueError(msg)

        # printingOptions is about what goes in solution file
        command = f"cbc -printingOptions all -import {problem_fn} "

        if warmstart_fn:
            command += f"-basisI {warmstart_fn} "

        if self.solver_options:
            command += (
                " ".join(
                    "-" + " ".join([k, str(v)]) for k, v in self.solver_options.items()
                )
                + " "
            )
        command += f"-solve -solu {solution_fn} "

        if basis_fn:
            command += f"-basisO {basis_fn} "

        Path(solution_fn).parent.mkdir(exist_ok=True)

        command = command.strip()

        if log_fn is None:
            p = sub.Popen(command.split(" "), stdout=sub.PIPE, stderr=sub.PIPE)

            if p.stdout is None:
                msg = (
                    f"Command `{command}` did not run successfully. Check if cbc is "
                    " installed and in PATH."
                )
                raise ValueError(msg)

            output = ""
            for line in iter(p.stdout.readline, b""):
                output += line.decode()
            logger.info(output)
            p.stdout.close()
            p.wait()
        else:
            with open(log_fn, "w") as log_f:
                p = sub.Popen(command.split(" "), stdout=log_f, stderr=log_f)
                p.wait()

        with open(solution_fn) as f:
            first_line = f.readline()

        if first_line.startswith("Optimal "):
            status = Status.from_termination_condition("optimal")
        elif "Infeasible" in first_line:
            status = Status.from_termination_condition("infeasible")
        else:
            status = Status(SolverStatus.warning, TerminationCondition.unknown)
        status.legacy_status = first_line

        # Use HiGHS to parse the problem file and find the set of variable names, needed to parse solution
        if "highs" not in available_solvers:
            raise ModuleNotFoundError(
                f"highspy is not installed. Please install it to use {self.solver_name.name} solver."
            )
        h = highspy.Highs()
        h.silent()
        h.readModel(path_to_string(problem_fn))
        variables = {v.name for v in h.getVariables()}

        def get_solver_solution() -> Solution:
            m = re.match(r"Optimal.* - objective value (-?\d+\.?\d*)$", first_line)
            if m and len(m.groups()) == 1:
                objective = float(m.group(1))
            else:
                objective = np.nan

            with open(solution_fn, "rb") as f:
                trimmed_sol_fn = re.sub(rb"\*\*\s+", b"", f.read())

            df = pd.read_csv(
                io.BytesIO(trimmed_sol_fn),
                header=None,
                skiprows=[0],
                sep=r"\s+",
                usecols=[1, 2, 3],
                index_col=0,
            )
            variables_b = df.index.isin(variables)

            sol_df = df[variables_b]
            dual_df = df[~variables_b]
            sol = sol_df[2].to_numpy(dtype=float)
            dual = dual_df[3].to_numpy(dtype=float)
            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        # Parse the output and get duality gap and solver runtime
        mip_gap, runtime = None, None
        if log_fn is not None:
            with open(log_fn) as log_f:
                output = "".join(log_f.readlines())
        m = re.search(r"\nGap: +(\d+\.?\d*)\n", output)
        if m and len(m.groups()) == 1:
            mip_gap = float(m.group(1))
        m = re.search(r"\nTime \(Wallclock seconds\): +(\d+\.?\d*)\n", output)
        if m and len(m.groups()) == 1:
            runtime = float(m.group(1))
        CbcModel = namedtuple("CbcModel", ["mip_gap", "runtime"])

        self.io_api = io_api
        return self._make_result(
            status,
            solution,
            solver_model=CbcModel(mip_gap, runtime),
            report=SolverReport(runtime=runtime, mip_gap=mip_gap),
        )


class GLPK(Solver[None]):
    """
    Solver subclass for the GLPK solver.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "GLPK"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.READ_MODEL_FROM_FILE,
        }
    )

    def __init(
        self,
        **solver_options: Any,
    ) -> None:
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> Result:
        msg = "Direct API not implemented for GLPK"
        raise NotImplementedError(msg)

    def solve_problem_from_file(
        self,
        problem_fn: Path,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
    ) -> Result:
        """
        Solve a linear problem from a problem file using the glpk solver.

        This function reads the linear problem file and passes it to the
        glpk solver. If the solution is successful it returns variable solutions
        and constraint dual values.

        For more information on the glpk solver options, see

        https://kam.mff.cuni.cz/~elias/glpk.pdf

        Parameters
        ----------
        problem_fn : Path
            Path to the problem file.
        solution_fn : Path
            Path to the solution file. This is necessary for solving with GLPK.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        env : None, optional
            Environment for the solver

        Returns
        -------
        Result
        """
        CONDITION_MAP = {
            "integer optimal": "optimal",
            "integer undefined": "infeasible_or_unbounded",
            "undefined": "infeasible_or_unbounded",
        }
        sense = read_sense_from_problem_file(problem_fn)
        io_api = read_io_api_from_problem_file(problem_fn)
        if solution_fn is None:
            msg = "No solution file specified. For solving with GLPK this is necessary."
            raise ValueError(msg)

        if io_api == "mps" and sense == "max" and _new_highspy_mps_layout:
            msg = (
                "GLPK does not support maximization in MPS format highspy versions "
                " >=1.7.1"
            )
            raise ValueError(msg)

        Path(solution_fn).parent.mkdir(exist_ok=True)

        # TODO use --nopresol argument for non-optimal solution output
        io_api_arg = "freemps" if io_api == "mps" else io_api
        command = f"glpsol --{io_api_arg} {problem_fn} --output {solution_fn} "
        if log_fn is not None:
            command += f"--log {log_fn} "
        if warmstart_fn:
            command += f"--ini {warmstart_fn} "
        if basis_fn:
            command += f"-w {basis_fn} "
        if self.solver_options:
            command += (
                " ".join(
                    "--" + " ".join([k, str(v)]) for k, v in self.solver_options.items()
                )
                + " "
            )
        command = command.strip()

        p = sub.Popen(command.split(" "), stdout=sub.PIPE, stderr=sub.PIPE)
        if log_fn is None:
            output = ""

            if p.stdout is None:
                msg = (
                    f"Command `{command}` did not run successfully. Check if glpsol is "
                    "installed and in PATH."
                )
                raise ValueError(msg)

            for line in iter(p.stdout.readline, b""):
                output += line.decode()
            logger.info(output)
            p.stdout.close()
            p.wait()
        else:
            p.wait()

        if not os.path.exists(solution_fn):
            status = Status(SolverStatus.warning, TerminationCondition.unknown)
            self.io_api = io_api
            return self._make_result(status, Solution())

        f = open(solution_fn)

        def read_until_break(f: io.TextIOWrapper) -> Generator[str, None, None]:
            while True:
                line = f.readline()
                if line in ["\n", ""]:
                    break
                yield line

        info_io = io.StringIO("".join(read_until_break(f))[:-2])
        info = pd.read_csv(info_io, sep=":", index_col=0, header=None)[1]
        condition = info.Status.lower().strip()
        objective = float(re.sub(r"[^0-9\.\+\-e]+", "", info.Objective))

        termination_condition = CONDITION_MAP.get(condition, condition)
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = condition

        def get_solver_solution() -> Solution:
            dual_io = io.StringIO("".join(read_until_break(f))[:-2])
            dual_ = pd.read_fwf(dual_io)[1:].set_index("Row name")
            if "Marginal" in dual_:
                dual = (
                    pd.to_numeric(dual_["Marginal"], "coerce")
                    .fillna(0)
                    .to_numpy(dtype=float)
                )
            else:
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = np.array([], dtype=float)

            sol_io = io.StringIO("".join(read_until_break(f))[:-2])
            sol_df = pd.read_fwf(sol_io)[1:].set_index("Column name")
            sol = sol_df["Activity"].astype(float).to_numpy()
            f.close()
            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)
        self.io_api = io_api
        return self._make_result(status, solution)


class Highs(Solver[None]):
    """
    Solver subclass for the HiGHS solver. HiGHS must be installed
    for usage. Find the documentation at https://highs.dev/.

    The full list of solver options is documented at https://ergo-code.github.io/HiGHS/stable/options/definitions/.

    Some exemplary options are:

    * presolve : "choose" by default - "on"/"off" are alternatives.
    * solver :"choose" by default - "simplex"/"ipm"/"pdlp" are alternatives. Only "choose" solves MIP / QP!
    * parallel : "choose" by default - "on"/"off" are alternatives.
    * time_limit : inf by default.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "HiGHS"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.DIRECT_API,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.READ_MODEL_FROM_FILE,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
            SolverFeature.SEMI_CONTINUOUS_VARIABLES,
        }
    )

    def __init__(
        self,
        **solver_options: Any,
    ) -> None:
        super().__init__(**solver_options)

    def to_solver_model(
        self,
        model: Model,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
        log_fn: Path | None = None,
        **kwargs: Any,
    ) -> highspy.Highs:
        if self.solver_options.get("solver") in [
            "simplex",
            "ipm",
            "pdlp",
        ] and model.type in [
            "QP",
            "MILP",
        ]:
            logger.warning(
                "The HiGHS solver ignores quadratic terms / integrality if the solver is set to 'simplex', 'ipm' or 'pdlp'. "
                "Drop the solver option or use 'choose' to enable quadratic terms / integrality."
            )

        h = self._build_solver_model(
            model,
            explicit_coordinate_names=explicit_coordinate_names,
            set_names=set_names,
        )
        self._set_solver_params(h, log_fn)
        self.solver_model = h
        self.io_api = "direct"
        self.sense = model.sense
        return h

    @staticmethod
    def _build_solver_model(
        model: Model,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> highspy.Highs:
        """Build a highspy.Highs instance that mirrors the linopy `model`."""
        if model.variables.sos:
            raise NotImplementedError(
                "SOS constraints are not supported by the HiGHS direct API. "
                "Use io_api='lp' instead."
            )

        M = model.matrices
        h = highspy.Highs()
        h.addVars(len(M.vlabels), M.lb, M.ub)
        if (
            len(model.binaries)
            + len(model.integers)
            + len(list(model.variables.semi_continuous))
        ):
            vtypes = M.vtypes
            integrality_map = {"C": 0, "B": 1, "I": 1, "S": 2}
            int_mask = (vtypes == "B") | (vtypes == "I") | (vtypes == "S")
            labels = np.arange(len(vtypes))[int_mask]
            integrality = np.array(
                [integrality_map[v] for v in vtypes[int_mask]], dtype=np.int32
            )
            h.changeColsIntegrality(len(labels), labels, integrality)
            if len(model.binaries):
                labels = np.arange(len(vtypes))[vtypes == "B"]
                n = len(labels)
                h.changeColsBounds(
                    n, labels, np.zeros_like(labels), np.ones_like(labels)
                )

        c = M.c
        h.changeColsCost(len(c), np.arange(len(c), dtype=np.int32), c)

        A = M.A
        if A is not None:
            A = A.tocsr()
            num_cons = A.shape[0]
            lower = np.where(M.sense != "<", M.b, -np.inf)
            upper = np.where(M.sense != ">", M.b, np.inf)
            h.addRows(num_cons, lower, upper, A.nnz, A.indptr, A.indices, A.data)

        if set_names:
            print_variables, print_constraints = linopy.io.get_printers_scalar(
                model, explicit_coordinate_names=explicit_coordinate_names
            )
            lp = h.getLp()
            lp.col_names_ = print_variables(M.vlabels)
            if len(M.clabels):
                lp.row_names_ = print_constraints(M.clabels)
            h.passModel(lp)

        Q = M.Q
        if Q is not None:
            Q = triu(Q).tocsr()
            num_vars = Q.shape[0]
            h.passHessian(num_vars, Q.nnz, 1, Q.indptr, Q.indices, Q.data)

        if model.objective.sense == "max":
            h.changeObjectiveSense(highspy.ObjSense.kMaximize)

        return h

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> Result:
        self.to_solver_model(
            model,
            explicit_coordinate_names=explicit_coordinate_names,
            set_names=set_names,
            log_fn=log_fn,
        )

        return self._solve(
            self.solver_model,
            solution_fn,
            warmstart_fn,
            basis_fn,
            io_api="direct",
            sense=model.sense,
        )

    def _run(self) -> Result:
        return self._solve(self.solver_model, io_api=self.io_api, sense=self.sense)

    def solve_problem_from_file(
        self,
        problem_fn: Path,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
    ) -> Result:
        """
        Solve a linear problem from a problem file using the HiGHS solver.
        Reads a linear problem file and passes it to the HiGHS solver.
        If the solution is feasible the function returns the
        objective, solution and dual constraint variables.

        Parameters
        ----------
        problem_fn : Path
            Path to the problem file.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        env : None, optional
            Environment for the solver

        Returns
        -------
        Result
        """

        problem_fn_ = path_to_string(problem_fn)
        h = highspy.Highs()
        self._set_solver_params(h, log_fn)

        h.readModel(problem_fn_)
        self.solver_model = h
        self.io_api = read_io_api_from_problem_file(problem_fn)

        return self._solve(
            h,
            solution_fn,
            warmstart_fn,
            basis_fn,
            io_api=self.io_api,
            sense=read_sense_from_problem_file(problem_fn),
            from_file=True,
        )

    def _set_solver_params(
        self,
        highs_solver: highspy.Highs,
        log_fn: Path | None = None,
    ) -> None:
        if log_fn is not None:
            self.solver_options["log_file"] = path_to_string(log_fn)
            logger.info(f"Log file at {self.solver_options['log_file']}")

        for k, v in self.solver_options.items():
            highs_solver.setOptionValue(k, v)

    def _solve(
        self,
        h: highspy.Highs,
        solution_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        io_api: str | None = None,
        sense: str | None = None,
        from_file: bool = False,
    ) -> Result:
        """
        Solve a linear problem from a HiGHS object.


        Parameters
        ----------
        h : highspy.Highs
            HiGHS object.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        io_api: str
            io_api of the problem. For direct API from linopy model this is "direct".
        sense: str
            "min" or "max"
        from_file: bool
            ``True`` when ``h`` was populated via ``readModel`` — HiGHS may have
            reordered columns/rows, so values are re-permuted using parsed names.

        Returns
        -------
        Result
        """
        # https://ergo-code.github.io/HiGHS/dev/structures/enums/#HighsModelStatus
        CONDITION_MAP: dict[highspy.HighsModelStatus, TerminationCondition] = {
            highspy.HighsModelStatus.kNotset: TerminationCondition.unknown,
            highspy.HighsModelStatus.kLoadError: TerminationCondition.internal_solver_error,
            highspy.HighsModelStatus.kModelError: TerminationCondition.internal_solver_error,
            highspy.HighsModelStatus.kPresolveError: TerminationCondition.internal_solver_error,
            highspy.HighsModelStatus.kSolveError: TerminationCondition.internal_solver_error,
            highspy.HighsModelStatus.kPostsolveError: TerminationCondition.internal_solver_error,
            highspy.HighsModelStatus.kModelEmpty: TerminationCondition.unknown,
            highspy.HighsModelStatus.kMemoryLimit: TerminationCondition.resource_interrupt,
            highspy.HighsModelStatus.kOptimal: TerminationCondition.optimal,
            highspy.HighsModelStatus.kInfeasible: TerminationCondition.infeasible,
            highspy.HighsModelStatus.kUnboundedOrInfeasible: TerminationCondition.infeasible_or_unbounded,
            highspy.HighsModelStatus.kUnbounded: TerminationCondition.unbounded,
            highspy.HighsModelStatus.kObjectiveBound: TerminationCondition.terminated_by_limit,
            highspy.HighsModelStatus.kObjectiveTarget: TerminationCondition.terminated_by_limit,
            highspy.HighsModelStatus.kTimeLimit: TerminationCondition.time_limit,
            highspy.HighsModelStatus.kIterationLimit: TerminationCondition.iteration_limit,
            highspy.HighsModelStatus.kSolutionLimit: TerminationCondition.terminated_by_limit,
            highspy.HighsModelStatus.kInterrupt: TerminationCondition.user_interrupt,
            highspy.HighsModelStatus.kUnknown: TerminationCondition.unknown,
        }

        if warmstart_fn is not None and warmstart_fn.suffix == ".sol":
            h.readSolution(path_to_string(warmstart_fn), 0)
        elif warmstart_fn:
            h.readBasis(path_to_string(warmstart_fn))

        _run_highs_with_keyboard_interrupt(h)

        condition = h.getModelStatus()
        termination_condition = CONDITION_MAP.get(
            condition, TerminationCondition.unknown
        )
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = h.modelStatusToString(condition)

        if basis_fn:
            h.writeBasis(path_to_string(basis_fn))

        if solution_fn:
            h.writeSolution(path_to_string(solution_fn), 0)

        def get_solver_solution() -> Solution:
            objective = h.getObjectiveValue()
            solution = h.getSolution()
            sol = np.asarray(solution.col_value, dtype=float)
            dual = np.asarray(solution.row_dual, dtype=float)
            if from_file:
                lp = h.getLp()
                if len(lp.col_names_):
                    vlabels = _names_to_labels(lp.col_names_)
                    keep = vlabels >= 0
                    sol = sol[keep][np.argsort(vlabels[keep])]
                if len(lp.row_names_):
                    clabels = _names_to_labels(lp.row_names_)
                    keep = clabels >= 0
                    dual = dual[keep][np.argsort(clabels[keep])]
            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        runtime: float | None = None
        mip_gap: float | None = None
        with contextlib.suppress(Exception):
            runtime = float(h.getRunTime())
        with contextlib.suppress(Exception):
            mip_gap = float(h.getInfo().mip_gap)

        self.io_api = io_api
        return self._make_result(
            status,
            solution,
            solver_model=h,
            report=SolverReport(runtime=runtime, mip_gap=mip_gap),
        )


class Gurobi(Solver["gurobipy.Env | dict[str, Any] | None"]):
    """
    Solver subclass for the gurobi solver.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "Gurobi"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.DIRECT_API,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.READ_MODEL_FROM_FILE,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
            SolverFeature.IIS_COMPUTATION,
            SolverFeature.SOS_CONSTRAINTS,
            SolverFeature.SEMI_CONTINUOUS_VARIABLES,
            SolverFeature.SOLVER_ATTRIBUTE_ACCESS,
        }
    )

    def __init__(
        self,
        **solver_options: Any,
    ) -> None:
        super().__init__(**solver_options)

    def _resolve_env(self, env: gurobipy.Env | dict[str, Any] | None) -> gurobipy.Env:
        self.close()
        self._env_stack = contextlib.ExitStack()
        if env is None:
            resolved = self._env_stack.enter_context(gurobipy.Env())
        elif isinstance(env, dict):
            resolved = self._env_stack.enter_context(gurobipy.Env(params=env))
        else:
            resolved = env
        self.env = resolved
        return resolved

    def to_solver_model(
        self,
        model: Model,
        explicit_coordinate_names: bool = False,
        env: gurobipy.Env | dict[str, Any] | None = None,
        set_names: bool = True,
        **kwargs: Any,
    ) -> gurobipy.Model:
        env_ = self._resolve_env(env)
        m = self._build_solver_model(
            model,
            env=env_,
            explicit_coordinate_names=explicit_coordinate_names,
            set_names=set_names,
        )
        self.solver_model = m
        self.io_api = "direct"
        self.sense = model.sense
        return m

    @staticmethod
    def _build_solver_model(
        model: Model,
        env: gurobipy.Env | None = None,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> gurobipy.Model:
        """Build a gurobipy.Model that mirrors the linopy `model`."""
        model.constraints.sanitize_missings()
        gm = gurobipy.Model(env=env)

        M = model.matrices

        kwargs: dict[str, Any] = {}
        if set_names:
            print_variables, print_constraints = linopy.io.get_printers_scalar(
                model, explicit_coordinate_names=explicit_coordinate_names
            )
            kwargs["name"] = print_variables(M.vlabels)
        if (
            len(model.binaries.labels)
            + len(model.integers.labels)
            + len(list(model.variables.semi_continuous))
        ):
            kwargs["vtype"] = M.vtypes
        x = gm.addMVar(M.vlabels.shape, M.lb, M.ub, **kwargs)

        if model.is_quadratic:
            gm.setObjective(0.5 * x.T @ M.Q @ x + M.c @ x)
        else:
            gm.setObjective(M.c @ x)

        if model.objective.sense == "max":
            gm.ModelSense = -1

        if len(model.constraints):
            c = gm.addMConstr(M.A, x, M.sense, M.b)
            if set_names:
                names = print_constraints(M.clabels)
                c.setAttr("ConstrName", names)

        if model.variables.sos:
            for var_name in model.variables.sos:
                var = model.variables.sos[var_name]
                sos_type: int = var.attrs[SOS_TYPE_ATTR]
                sos_dim: str = var.attrs[SOS_DIM_ATTR]

                def add_sos(s: xr.DataArray, sos_type: int, sos_dim: str) -> None:
                    s = s.squeeze()
                    indices = s.values.flatten().tolist()
                    weights = s.coords[sos_dim].values.tolist()
                    gm.addSOS(sos_type, x[indices].tolist(), weights)

                others = [dim for dim in var.labels.dims if dim != sos_dim]
                if not others:
                    add_sos(var.labels, sos_type, sos_dim)
                else:
                    stacked = var.labels.stack(_sos_group=others)
                    for _, s in stacked.groupby("_sos_group"):
                        add_sos(s.unstack("_sos_group"), sos_type, sos_dim)

        gm.update()
        return gm

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: gurobipy.Env | dict[str, Any] | None = None,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> Result:
        self.to_solver_model(
            model,
            explicit_coordinate_names=explicit_coordinate_names,
            env=env,
            set_names=set_names,
        )
        return self._solve(
            self.solver_model,
            solution_fn=solution_fn,
            log_fn=log_fn,
            warmstart_fn=warmstart_fn,
            basis_fn=basis_fn,
            io_api="direct",
            sense=model.sense,
        )

    def _run(self) -> Result:
        return self._solve(
            self.solver_model,
            solution_fn=None,
            log_fn=None,
            warmstart_fn=None,
            basis_fn=None,
            io_api=self.io_api,
            sense=self.sense,
        )

    def solve_problem_from_file(
        self,
        problem_fn: Path,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: gurobipy.Env | dict[str, Any] | None = None,
    ) -> Result:
        """
        Solve a linear problem from a problem file using the Gurobi solver.
        Reads a problem file and passes it to the Gurobi solver.
        This function communicates with gurobi using the gurobipy package.

        Parameters
        ----------
        problem_fn : Path
            Path to the problem file.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        env : gurobipy.Env or dict, optional
            Gurobi environment for the solver, pass env directly or kwargs for creation.

        Returns
        -------
        Result
        """
        sense = read_sense_from_problem_file(problem_fn)
        io_api = read_io_api_from_problem_file(problem_fn)
        problem_fn_ = path_to_string(problem_fn)

        env_ = self._resolve_env(env)
        m = gurobipy.read(problem_fn_, env=env_)
        self.solver_model = m
        self.io_api = io_api

        return self._solve(
            m,
            solution_fn=solution_fn,
            log_fn=log_fn,
            warmstart_fn=warmstart_fn,
            basis_fn=basis_fn,
            io_api=io_api,
            sense=sense,
            from_file=True,
        )

    def _solve(
        self,
        m: gurobipy.Model,
        solution_fn: Path | None,
        log_fn: Path | None,
        warmstart_fn: Path | None,
        basis_fn: Path | None,
        io_api: str | None,
        sense: str | None,
        from_file: bool = False,
    ) -> Result:
        """
        Solve a linear problem from a Gurobi object.


        Parameters
        ----------
        m
            Gurobi object.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        io_api: str
            io_api of the problem. For direct API from linopy model this is "direct".
        sense: str
            "min" or "max"

        Returns
        -------
        Result
        """
        # see https://www.gurobi.com/documentation/10.0/refman/optimization_status_codes.html
        CONDITION_MAP = {
            1: "unknown",
            2: "optimal",
            3: "infeasible",
            4: "infeasible_or_unbounded",
            5: "unbounded",
            6: "other",
            7: "iteration_limit",
            8: "terminated_by_limit",
            9: "time_limit",
            10: "optimal",
            11: "user_interrupt",
            12: "other",
            13: "suboptimal",
            14: "unknown",
            15: "terminated_by_limit",
            16: "internal_solver_error",
            17: "internal_solver_error",
        }

        if self.solver_options is not None:
            for key, value in self.solver_options.items():
                m.setParam(key, value)
        if log_fn is not None:
            m.setParam("logfile", path_to_string(log_fn))

        if warmstart_fn is not None:
            m.read(path_to_string(warmstart_fn))
        m.optimize()

        if basis_fn is not None:
            try:
                m.write(path_to_string(basis_fn))
            except gurobipy.GurobiError as err:
                logger.info("No model basis stored. Raised error: %s", err)

        if solution_fn is not None and solution_fn.suffix == ".sol":
            try:
                m.write(path_to_string(solution_fn))
            except gurobipy.GurobiError as err:
                logger.info("Unable to save solution file. Raised error: %s", err)

        condition = m.status
        termination_condition = CONDITION_MAP.get(condition, condition)
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = condition

        def get_solver_solution() -> Solution:
            objective = m.ObjVal

            vars_ = m.getVars()
            sol = np.array([v.X for v in vars_], dtype=float)
            if from_file and len(vars_):
                vlabels = _names_to_labels([v.VarName for v in vars_])
                keep = vlabels >= 0
                sol = sol[keep][np.argsort(vlabels[keep])]

            try:
                constrs = m.getConstrs()
                dual = np.array([c.Pi for c in constrs], dtype=float)
                if from_file and len(constrs):
                    clabels = _names_to_labels([c.ConstrName for c in constrs])
                    keep = clabels >= 0
                    dual = dual[keep][np.argsort(clabels[keep])]
            except AttributeError:
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = np.array([], dtype=float)

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        runtime: float | None = None
        mip_gap: float | None = None
        with contextlib.suppress(Exception):
            runtime = float(m.Runtime)
        with contextlib.suppress(Exception):
            mip_gap = float(m.MIPGap)

        self.io_api = io_api
        return self._make_result(
            status,
            solution,
            solver_model=m,
            report=SolverReport(runtime=runtime, mip_gap=mip_gap),
        )


class Cplex(Solver[None]):
    """
    Solver subclass for the Cplex solver.

    Note if you pass additional solver_options, the key can specify deeper
    layered parameters, use a dot as a separator here,
    i.e. `**{'aa.bb.cc' : x}`.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "CPLEX"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.READ_MODEL_FROM_FILE,
            SolverFeature.SOS_CONSTRAINTS,
            SolverFeature.SEMI_CONTINUOUS_VARIABLES,
        }
    )

    def __init__(
        self,
        **solver_options: Any,
    ) -> None:
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> Result:
        msg = "Direct API not implemented for Cplex"
        raise NotImplementedError(msg)

    def solve_problem_from_file(
        self,
        problem_fn: Path,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
    ) -> Result:
        """
        Solve a linear problem from a problem file using the cplex solver.

        This function reads the linear problem file and passes it to the cplex
        solver. If the solution is successful it returns variable solutions and
        constraint dual values. Cplex must be installed for using this function.

        Parameters
        ----------
        problem_fn : Path
            Path to the problem file.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        env : None, optional
            Environment for the solver

        Returns
        -------
        Result
        """
        CONDITION_MAP = {
            "integer optimal solution": "optimal",
            "integer optimal, tolerance": "optimal",
            "integer infeasible": "infeasible",
            "time limit exceeded": "time_limit",
            "time limit exceeded, no integer solution": "infeasible",
            "error termination": "error",
            "error termination, no integer solution": "error",
            "memory limit exceeded": "internal_solver_error",
            "memory limit exceeded, no integer solution": "internal_solver_error",
            "aborted": "user_interrupt",
            "integer unbounded": "unbounded",
            "integer infeasible or unbounded": "infeasible_or_unbounded",
            "Unknown status value": "unknown",
        }
        io_api = read_io_api_from_problem_file(problem_fn)
        sense = read_sense_from_problem_file(problem_fn)

        m = cplex.Cplex()

        if log_fn is not None:
            log_f = open(path_to_string(log_fn), "w")
            m.set_results_stream(log_f)
            m.set_warning_stream(log_f)
            m.set_error_stream(log_f)
            m.set_log_stream(log_f)

        if self.solver_options is not None:
            for key, value in self.solver_options.items():
                param = m.parameters
                for key_layer in key.split("."):
                    param = getattr(param, key_layer)
                param.set(value)

        m.read(path_to_string(problem_fn))

        if warmstart_fn is not None:
            m.start.read_basis(path_to_string(warmstart_fn))

        is_lp = m.problem_type[m.get_problem_type()] == "LP"

        with contextlib.suppress(cplex.exceptions.errors.CplexSolverError):
            m.solve()

        if solution_fn is not None:
            try:
                m.solution.write(path_to_string(solution_fn))
            except cplex.exceptions.errors.CplexSolverError as err:
                logger.info("Unable to save solution file. Raised error: %s", err)

        condition = m.solution.get_status_string()
        termination_condition = CONDITION_MAP.get(condition, condition)
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = condition

        if log_fn is not None:
            log_f.close()

        def get_solver_solution() -> Solution:
            if basis_fn and is_lp:
                try:
                    m.solution.basis.write(path_to_string(basis_fn))
                except cplex.exceptions.errors.CplexSolverError:
                    logger.info("No model basis stored")

            objective = m.solution.get_objective_value()

            solution = np.asarray(m.solution.get_values(), dtype=float)

            try:
                dual = np.asarray(m.solution.get_dual_values(), dtype=float)
            except Exception:
                logger.warning(
                    "Dual values not available (e.g. barrier solution without crossover)"
                )
                dual = np.array([], dtype=float)
            return Solution(solution, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        self.io_api = io_api
        return self._make_result(status, solution, solver_model=m)


class SCIP(Solver[None]):
    """
    Solver subclass for the SCIP solver.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "SCIP"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.READ_MODEL_FROM_FILE,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
        }
    )

    def __init__(
        self,
        **solver_options: Any,
    ) -> None:
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> Result:
        msg = "Direct API not implemented for SCIP"
        raise NotImplementedError(msg)

    def solve_problem_from_file(
        self,
        problem_fn: Path,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
    ) -> Result:
        """
        Solve a linear problem from a problem file using the scip solver.

        This function communicates with scip using the pyscipopt package.

        Parameters
        ----------
        problem_fn : Path
            Path to the problem file.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        env : None, optional
            Environment for the solver

        Returns
        -------
        Result
        """
        CONDITION_MAP: dict[str, TerminationCondition] = {
            # https://github.com/scipopt/scip/blob/b2bac412222296ff2b7f2347bb77d5fc4e05a2a1/src/scip/type_stat.h#L40
            "inforunbd": TerminationCondition.infeasible_or_unbounded,
            "userinterrupt": TerminationCondition.user_interrupt,
            "terminate": TerminationCondition.user_interrupt,
            "nodelimit": TerminationCondition.terminated_by_limit,
            "totalnodelimit": TerminationCondition.terminated_by_limit,
            "stallnodelimit": TerminationCondition.terminated_by_limit,
            "timelimit": TerminationCondition.time_limit,
            "memlimit": TerminationCondition.terminated_by_limit,
            "gaplimit": TerminationCondition.optimal,
            "primallimit": TerminationCondition.terminated_by_limit,
            "duallimit": TerminationCondition.terminated_by_limit,
            "sollimit": TerminationCondition.terminated_by_limit,
            "bestsollimit": TerminationCondition.terminated_by_limit,
            "restartlimit": TerminationCondition.terminated_by_limit,
        }

        io_api = read_io_api_from_problem_file(problem_fn)
        sense = read_sense_from_problem_file(problem_fn)

        m = scip.Model()
        m.readProblem(path_to_string(problem_fn))

        if self.solver_options is not None:
            emphasis = self.solver_options.pop("setEmphasis", None)
            if emphasis is not None:
                m.setEmphasis(getattr(scip.SCIP_PARAMEMPHASIS, emphasis.upper()))

            heuristics = self.solver_options.pop("setHeuristics", None)
            if heuristics is not None:
                m.setEmphasis(getattr(scip.SCIP_PARAMSETTING, heuristics.upper()))

            presolve = self.solver_options.pop("setPresolve", None)
            if presolve is not None:
                m.setEmphasis(getattr(scip.SCIP_PARAMSETTING, presolve.upper()))

            m.setParams(self.solver_options)

        if log_fn is not None:
            m.setLogfile(path_to_string(log_fn))

        if warmstart_fn:
            logger.warning("Warmstart not implemented for SCIP")

        m.optimize()

        if basis_fn:
            logger.warning("Basis not implemented for SCIP")

        if solution_fn:
            try:
                m.writeSol(m.getBestSol(), filename=path_to_string(solution_fn))
            except FileNotFoundError as err:
                logger.warning("Unable to save solution file. Raised error: %s", err)

        condition = m.getStatus()
        termination_condition = CONDITION_MAP.get(condition, condition)
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = condition

        def get_solver_solution() -> Solution:
            objective = m.getObjVal()
            vars_to_ignore = {"quadobjvar", "qmatrixvar", "quadobj", "qmatrix"}

            s = m.getSols()[0]
            sol = np.array(
                [s[v] for v in m.getVars() if v.name not in vars_to_ignore],
                dtype=float,
            )

            cons = m.getConss(False)
            if len(cons) != 0:
                dual = np.array(
                    [
                        m.getDualSolVal(c)
                        for c in cons
                        if c.name not in vars_to_ignore
                    ],
                    dtype=float,
                )
            else:
                logger.warning("Dual values not available (is this an MILP?)")
                dual = np.array([], dtype=float)

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        self.io_api = io_api
        return self._make_result(status, solution, solver_model=m)


class Xpress(Solver[None]):
    """
    Solver subclass for the xpress solver.

    For more information on solver options, see
    https://www.fico.com/fico-xpress-optimization/docs/latest/solver/GUID-ACD7E60C-7852-36B7-A78A-CED0EA291CDD.html

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "FICO Xpress"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.READ_MODEL_FROM_FILE,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
            SolverFeature.IIS_COMPUTATION,
        }
    )

    @classmethod
    def runtime_features(cls) -> frozenset[SolverFeature]:
        if _installed_version_in("xpress", ">=9.8.0"):
            return frozenset({SolverFeature.GPU_ACCELERATION})
        return frozenset()

    def __init__(
        self,
        **solver_options: Any,
    ) -> None:
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> Result:
        msg = "Direct API not implemented for Xpress"
        raise NotImplementedError(msg)

    def solve_problem_from_file(
        self,
        problem_fn: Path,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
    ) -> Result:
        """
        Solve a linear problem from a problem file using the Xpress solver.

        This function reads the linear problem file and passes it to
        the Xpress solver. If the solution is successful it returns
        variable solutions and constraint dual values. The `xpress` module
        must be installed for using this function.

        Parameters
        ----------
        problem_fn : Path
            Path to the problem file.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        env : None, optional
            Environment for the solver

        Returns
        -------
        Result
        """
        CONDITION_MAP = {
            xpress.SolStatus.NOTFOUND: "unknown",
            xpress.SolStatus.OPTIMAL: "optimal",
            xpress.SolStatus.FEASIBLE: "terminated_by_limit",
            xpress.SolStatus.INFEASIBLE: "infeasible",
            xpress.SolStatus.UNBOUNDED: "unbounded",
        }

        io_api = read_io_api_from_problem_file(problem_fn)
        sense = read_sense_from_problem_file(problem_fn)

        m = xpress.problem()

        try:  # Try new API first
            m.readProb(path_to_string(problem_fn))
        except AttributeError:  # Fallback to old API
            m.read(path_to_string(problem_fn))

        # Set solver options - new API uses setControl per option, old API accepts dict
        if self.solver_options is not None:
            m.setControl(self.solver_options)

        if log_fn is not None:
            try:  # Try new API first
                m.setLogFile(path_to_string(log_fn))
            except AttributeError:  # Fallback to old API
                m.setlogfile(path_to_string(log_fn))

        if warmstart_fn is not None:
            try:  # Try new API first
                m.readBasis(path_to_string(warmstart_fn))
            except AttributeError:  # Fallback to old API
                m.readbasis(path_to_string(warmstart_fn))

        m.optimize()

        # if the solver is stopped (timelimit for example), postsolve the problem
        if m.attributes.solvestatus == xpress.enums.SolveStatus.STOPPED:
            try:  # Try new API first
                m.postSolve()
            except AttributeError:  # Fallback to old API
                m.postsolve()

        if basis_fn is not None:
            try:
                try:  # Try new API first
                    m.writeBasis(path_to_string(basis_fn))
                except AttributeError:  # Fallback to old API
                    m.writebasis(path_to_string(basis_fn))
            except (xpress.SolverError, xpress.ModelError) as err:
                logger.info("No model basis stored. Raised error: %s", err)

        if solution_fn is not None:
            try:
                try:  # Try new API first
                    m.writeBinSol(path_to_string(solution_fn))
                except AttributeError:  # Fallback to old API
                    m.writebinsol(path_to_string(solution_fn))
            except (xpress.SolverError, xpress.ModelError) as err:
                logger.info("Unable to save solution file. Raised error: %s", err)

        condition = m.attributes.solstatus
        termination_condition = CONDITION_MAP.get(condition, condition)
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = condition

        def get_solver_solution() -> Solution:
            objective = m.attributes.objval

            sol = np.asarray(m.getSolution(), dtype=float)

            try:
                if m.attributes.rows == 0:
                    dual = np.array([], dtype=float)
                else:
                    try:  # Try new API first
                        _dual = m.getDuals()
                    except AttributeError:  # Fallback to old API
                        _dual = m.getDual()
                    dual = np.asarray(_dual, dtype=float)
            except (xpress.SolverError, xpress.ModelError, SystemError):
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = np.array([], dtype=float)

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        self.io_api = io_api
        return self._make_result(status, solution, solver_model=m)


KnitroResult = namedtuple(
    "KnitroResult",
    "reported_runtime mip_relaxation_bnd mip_number_nodes mip_number_solves mip_rel_gap mip_abs_gap abs_feas_error rel_feas_error abs_opt_error rel_opt_error n_vars n_cons n_integer_vars n_continuous_vars",
)


class Knitro(Solver[None]):
    """
    Solver subclass for the Knitro solver.

    For more information on solver options, see
    https://www.artelys.com/app/docs/knitro/3_referenceManual/knitroPythonReference.html

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "Artelys Knitro"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.READ_MODEL_FROM_FILE,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
        }
    )

    def __init__(
        self,
        **solver_options: Any,
    ) -> None:
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> Result:
        msg = "Direct API not implemented for Knitro"
        raise NotImplementedError(msg)

    @staticmethod
    def _set_option(kc: Any, name: str, value: Any) -> None:
        param_id = knitro.KN_get_param_id(kc, name)

        if isinstance(value, bool):
            value = int(value)

        if isinstance(value, int):
            knitro.KN_set_int_param(kc, param_id, value)
        elif isinstance(value, float):
            knitro.KN_set_double_param(kc, param_id, value)
        elif isinstance(value, str):
            knitro.KN_set_char_param(kc, param_id, value)
        else:
            msg = f"Unsupported Knitro option type for {name!r}: {type(value).__name__}"
            raise TypeError(msg)

    @staticmethod
    def _extract_values(
        kc: Any,
        get_count_fn: Callable[..., Any],
        get_values_fn: Callable[..., Any],
    ) -> np.ndarray:
        n = int(get_count_fn(kc))
        if n == 0:
            return np.array([], dtype=float)

        try:
            # Compatible with KNITRO >= 15
            values = get_values_fn(kc)
        except TypeError:
            # Fallback for older wrappers requiring explicit indices
            values = get_values_fn(kc, list(range(n)))

        return np.asarray(values, dtype=float)

    def solve_problem_from_file(
        self,
        problem_fn: Path,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
    ) -> Result:
        """
        Solve a linear problem from a problem file using the Knitro solver.

        Parameters
        ----------
        problem_fn : Path
            Path to the problem file.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        env : None, optional
            Environment for the solver.

        Returns
        -------
        Result
        """
        CONDITION_MAP: dict[int, TerminationCondition] = {
            0: TerminationCondition.optimal,
            -100: TerminationCondition.suboptimal,
            -101: TerminationCondition.infeasible,
            -102: TerminationCondition.suboptimal,
            -200: TerminationCondition.unbounded,
            -201: TerminationCondition.infeasible_or_unbounded,
            -202: TerminationCondition.iteration_limit,
            -203: TerminationCondition.time_limit,
            -204: TerminationCondition.terminated_by_limit,
            -300: TerminationCondition.unbounded,
            -400: TerminationCondition.iteration_limit,
            -401: TerminationCondition.time_limit,
            -410: TerminationCondition.terminated_by_limit,
            -411: TerminationCondition.terminated_by_limit,
        }

        READ_OPTIONS: dict[str, str] = {".lp": "l", ".mps": "m"}

        io_api = read_io_api_from_problem_file(problem_fn)
        sense = read_sense_from_problem_file(problem_fn)

        suffix = problem_fn.suffix.lower()
        if suffix not in READ_OPTIONS:
            msg = f"Unsupported problem file format: {suffix}"
            raise ValueError(msg)

        kc = knitro.KN_new()
        try:
            knitro.KN_read_problem(
                kc,
                path_to_string(problem_fn),
                read_options=READ_OPTIONS[suffix],
            )

            if log_fn is not None:
                logger.warning("Log file output not implemented for Knitro")

            for k, v in self.solver_options.items():
                self._set_option(kc, k, v)

            ret = int(knitro.KN_solve(kc))

            reported_runtime: float | None = None
            mip_relaxation_bnd: float | None = None
            mip_number_nodes: int | None = None
            mip_number_solves: int | None = None
            mip_rel_gap: float | None = None
            mip_abs_gap: float | None = None
            abs_feas_error: float | None = None
            rel_feas_error: float | None = None
            abs_opt_error: float | None = None
            rel_opt_error: float | None = None
            n_vars: int | None = None
            n_cons: int | None = None
            n_integer_vars: int | None = None
            n_continuous_vars: int | None = None
            with contextlib.suppress(Exception):
                reported_runtime = float(knitro.KN_get_solve_time_real(kc))
                mip_relaxation_bnd = float(knitro.KN_get_mip_relaxation_bnd(kc))
                mip_number_nodes = int(knitro.KN_get_mip_number_nodes(kc))
                mip_number_solves = int(knitro.KN_get_mip_number_solves(kc))
                mip_rel_gap = float(knitro.KN_get_mip_rel_gap(kc))
                mip_abs_gap = float(knitro.KN_get_mip_abs_gap(kc))
                abs_feas_error = float(knitro.KN_get_abs_feas_error(kc))
                rel_feas_error = float(knitro.KN_get_rel_feas_error(kc))
                abs_opt_error = float(knitro.KN_get_abs_opt_error(kc))
                rel_opt_error = float(knitro.KN_get_rel_opt_error(kc))
                n_vars = int(knitro.KN_get_number_vars(kc))
                n_cons = int(knitro.KN_get_number_cons(kc))
                var_types = list(knitro.KN_get_var_types(kc))
                n_integer_vars = int(
                    var_types.count(knitro.KN_VARTYPE_INTEGER)
                    + var_types.count(knitro.KN_VARTYPE_BINARY)
                )
                n_continuous_vars = int(var_types.count(knitro.KN_VARTYPE_CONTINUOUS))

            if ret in CONDITION_MAP:
                termination_condition = CONDITION_MAP[ret]
            elif ret > 0:
                termination_condition = TerminationCondition.internal_solver_error
            else:
                termination_condition = TerminationCondition.unknown

            status = Status.from_termination_condition(termination_condition)
            status.legacy_status = str(ret)

            def get_solver_solution() -> Solution:
                objective = float(knitro.KN_get_obj_value(kc))

                sol = self._extract_values(
                    kc,
                    knitro.KN_get_number_vars,
                    knitro.KN_get_var_primal_values,
                )

                try:
                    dual = self._extract_values(
                        kc,
                        knitro.KN_get_number_cons,
                        knitro.KN_get_con_dual_values,
                    )
                except Exception:
                    logger.warning("Dual values couldn't be parsed")
                    dual = np.array([], dtype=float)

                return Solution(sol, dual, objective)

            solution = self.safe_get_solution(status=status, func=get_solver_solution)
            solution = maybe_adjust_objective_sign(solution, io_api, sense)

            if solution_fn is not None:
                solution_fn.parent.mkdir(exist_ok=True)
                knitro.KN_write_mps_file(kc, path_to_string(solution_fn))

            knitro_model = KnitroResult(
                reported_runtime=reported_runtime,
                mip_relaxation_bnd=mip_relaxation_bnd,
                mip_number_nodes=mip_number_nodes,
                mip_number_solves=mip_number_solves,
                mip_rel_gap=mip_rel_gap,
                mip_abs_gap=mip_abs_gap,
                abs_feas_error=abs_feas_error,
                rel_feas_error=rel_feas_error,
                abs_opt_error=abs_opt_error,
                rel_opt_error=rel_opt_error,
                n_vars=n_vars,
                n_cons=n_cons,
                n_integer_vars=n_integer_vars,
                n_continuous_vars=n_continuous_vars,
            )
            self.io_api = io_api
            return self._make_result(
                status,
                solution,
                solver_model=knitro_model,
                report=SolverReport(runtime=reported_runtime, mip_gap=mip_rel_gap),
            )
        finally:
            with contextlib.suppress(Exception):
                knitro.KN_free(kc)


mosek_bas_re = re.compile(r" (XL|XU)\s+([^ \t]+)\s+([^ \t]+)| (LL|UL|BS)\s+([^ \t]+)")


class Mosek(Solver[None]):
    """
    Solver subclass for the Mosek solver.

    https://www.mosek.com/

    For more information on solver options, see
    https://docs.mosek.com/latest/pythonapi/parameters.html#doc-all-parameter-list


    For remote optimization of smaller problems, which do not require a license,
    set the following solver_options:
    {"MSK_SPAR_REMOTE_OPTSERVER_HOST": "http://solve.mosek.com:30080"}

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "MOSEK"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.DIRECT_API,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.READ_MODEL_FROM_FILE,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
        }
    )

    def __init__(
        self,
        **solver_options: Any,
    ) -> None:
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> Result:
        if env is not None:
            warnings.warn(
                "The 'env' parameter in solve_problem_from_model is deprecated and will be "
                "removed in a future version. MOSEK now uses the global environment "
                "automatically, avoiding unnecessary license checkouts.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.to_solver_model(
            model,
            explicit_coordinate_names=explicit_coordinate_names,
            set_names=set_names,
        )
        return self._solve(
            self.solver_model,
            solution_fn=solution_fn,
            log_fn=log_fn,
            warmstart_fn=warmstart_fn,
            basis_fn=basis_fn,
            io_api="direct",
            sense=model.sense,
        )

    def _run(self) -> Result:
        return self._solve(
            self.solver_model,
            solution_fn=None,
            log_fn=None,
            warmstart_fn=None,
            basis_fn=None,
            io_api=self.io_api,
            sense=self.sense,
        )

    def to_solver_model(
        self,
        model: Model,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
        **kwargs: Any,
    ) -> mosek.Task:
        self.close()
        self._env_stack = contextlib.ExitStack()
        task = self._env_stack.enter_context(mosek.Task())
        m = self._build_solver_model(
            model,
            task,
            explicit_coordinate_names=explicit_coordinate_names,
            set_names=set_names,
        )
        self.solver_model = m
        self.io_api = "direct"
        self.sense = model.sense
        return m

    @staticmethod
    def _build_solver_model(
        model: Model,
        task: mosek.Task,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> mosek.Task:
        """Populate an empty MOSEK task with the contents of `model`."""
        if model.variables.sos:
            raise NotImplementedError("SOS constraints are not supported by MOSEK.")
        if model.variables.semi_continuous:
            raise NotImplementedError(
                "Semi-continuous variables are not supported by MOSEK. "
                "Use a solver that supports them (gurobi, cplex, highs)."
            )

        task.appendvars(model.nvars)
        task.appendcons(model.ncons)

        M = model.matrices

        if set_names:
            print_variables, print_constraints = linopy.io.get_printers_scalar(
                model, explicit_coordinate_names=explicit_coordinate_names
            )
            labels = print_variables(M.vlabels)
            task.generatevarnames(
                np.arange(0, len(labels)), "%0", [len(labels)], None, [0], labels
            )

        bkx = [
            (
                (
                    (mosek.boundkey.ra if lb < ub else mosek.boundkey.fx)
                    if ub < np.inf
                    else mosek.boundkey.lo
                )
                if (lb > -np.inf)
                else (mosek.boundkey.up if (ub < np.inf) else mosek.boundkey.fr)
            )
            for (lb, ub) in zip(M.lb, M.ub)
        ]
        blx = [b if b > -np.inf else 0.0 for b in M.lb]
        bux = [b if b < np.inf else 0.0 for b in M.ub]
        task.putvarboundslice(0, model.nvars, bkx, blx, bux)

        if len(model.binaries.labels) + len(model.integers.labels) > 0:
            idx = [i for (i, v) in enumerate(M.vtypes) if v in ["B", "I"]]
            task.putvartypelist(idx, [mosek.variabletype.type_int] * len(idx))
            if len(model.binaries.labels) > 0:
                bidx = [i for (i, v) in enumerate(M.vtypes) if v == "B"]
                task.putvarboundlistconst(bidx, mosek.boundkey.ra, 0.0, 1.0)

        if len(model.constraints) > 0:
            if set_names:
                names = print_constraints(M.clabels)
                for i, n in enumerate(names):
                    task.putconname(i, n)
            bkc = [
                (
                    (mosek.boundkey.up if b < np.inf else mosek.boundkey.fr)
                    if s == "<"
                    else (
                        (mosek.boundkey.lo if b > -np.inf else mosek.boundkey.up)
                        if s == ">"
                        else mosek.boundkey.fx
                    )
                )
                for s, b in zip(M.sense, M.b)
            ]
            blc = [b if b > -np.inf else 0.0 for b in M.b]
            buc = [b if b < np.inf else 0.0 for b in M.b]
            if M.A is not None:
                A = M.A.tocsr()
                task.putarowslice(
                    0, model.ncons, A.indptr[:-1], A.indptr[1:], A.indices, A.data
                )
                task.putconboundslice(0, model.ncons, bkc, blc, buc)

        if M.Q is not None:
            Q = (0.5 * tril(M.Q + M.Q.transpose())).tocoo()
            task.putqobj(Q.row, Q.col, Q.data)
        task.putclist(list(np.arange(model.nvars)), M.c)

        if model.objective.sense == "max":
            task.putobjsense(mosek.objsense.maximize)
        else:
            task.putobjsense(mosek.objsense.minimize)
        return task

    def solve_problem_from_file(
        self,
        problem_fn: Path,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
    ) -> Result:
        """
        Solve a linear problem from a problem file using the MOSEK solver. Both mps and
        lp files are supported; MPS does not support quadratic terms.

        Parameters
        ----------
        problem_fn : Path
            Path to the problem file.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        env : None, optional, deprecated
            Deprecated. This parameter is ignored. MOSEK now uses the global
            environment automatically. Will be removed in a future version.

        Returns
        -------
        Result
        """
        if env is not None:
            warnings.warn(
                "The 'env' parameter in solve_problem_from_file is deprecated and will be "
                "removed in a future version. MOSEK now uses the global environment "
                "automatically, avoiding unnecessary license checkouts.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.close()
        self._env_stack = contextlib.ExitStack()
        m = self._env_stack.enter_context(mosek.Task())
        sense = read_sense_from_problem_file(problem_fn)
        io_api = read_io_api_from_problem_file(problem_fn)
        problem_fn_ = path_to_string(problem_fn)
        m.readdata(problem_fn_)
        self.solver_model = m
        self.io_api = io_api

        return self._solve(
            m,
            solution_fn=solution_fn,
            log_fn=log_fn,
            warmstart_fn=warmstart_fn,
            basis_fn=basis_fn,
            io_api=io_api,
            sense=sense,
        )

    def _solve(
        self,
        m: mosek.Task,
        solution_fn: Path | None,
        log_fn: Path | None,
        warmstart_fn: Path | None,
        basis_fn: Path | None,
        io_api: str | None,
        sense: str | None,
    ) -> Result:
        """
        Solve a linear problem from a Mosek task object.

        Parameters
        ----------
        m : mosek.Task
            Mosek task object.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        io_api: str
            io_api of the problem. For direct API from linopy model this is "direct".
        sense: str
            "min" or "max"

        Returns
        -------
        Result
        """
        CONDITION_MAP = {
            "solsta.unknown": "unknown",
            "solsta.optimal": "optimal",
            "solsta.integer_optimal": "optimal",
            "solsta.prim_infeas_cer": "infeasible",
            "solsta.dual_infeas_cer": "infeasible_or_unbounded",
        }

        for k, v in self.solver_options.items():
            m.putparam(k, str(v))

        if log_fn is not None:
            m.linkfiletostream(mosek.streamtype.log, path_to_string(log_fn), 0)
        else:
            m.set_Stream(mosek.streamtype.log, sys.stdout.write)

        if warmstart_fn is not None:
            m.putintparam(mosek.iparam.sim_hotstart, mosek.simhotstart.status_keys)
            skx = [mosek.stakey.low] * m.getnumvar()
            skc = [mosek.stakey.bas] * m.getnumcon()

            with open(path_to_string(warmstart_fn)) as f:
                for line in f:
                    if line.startswith("NAME "):
                        break

                for line in f:
                    if line.startswith("ENDATA"):
                        break

                    o = mosek_bas_re.match(line)
                    if o is not None:
                        if o.group(1) is not None:
                            key = o.group(1)
                            try:
                                skx[m.getvarnameindex(o.group(2))] = mosek.stakey.basis
                            except:  # noqa: E722
                                pass
                            try:
                                skc[m.getvarnameindex(o.group(3))] = (
                                    mosek.stakey.low if key == "XL" else "XU"
                                )
                            except:  # noqa: E722
                                pass
                        else:
                            key = o.group(4)
                            name = o.group(5)
                            stakey = (
                                mosek.stakey.low
                                if key == "LL"
                                else (
                                    mosek.stakey.upr
                                    if key == "UL"
                                    else mosek.stakey.bas
                                )
                            )

                            try:
                                skx[m.getvarnameindex(name)] = stakey
                            except:  # noqa: E722
                                try:
                                    skc[m.getvarnameindex(name)] = stakey
                                except:  # noqa: E722
                                    pass
            m.putskc(mosek.soltype.bas, skc)
            m.putskx(mosek.soltype.bas, skx)
        m.optimize()

        m.solutionsummary(mosek.streamtype.log)

        if basis_fn is not None:
            if m.solutiondef(mosek.soltype.bas):
                with open(path_to_string(basis_fn), "w") as f:
                    f.write(f"NAME {basis_fn}\n")

                    skc = [
                        (0 if sk != mosek.stakey.bas else 1, i, sk)
                        for (i, sk) in enumerate(m.getskc(mosek.soltype.bas))
                    ]
                    skx = [
                        (0 if sk == mosek.stakey.bas else 1, j, sk)
                        for (j, sk) in enumerate(m.getskx(mosek.soltype.bas))
                    ]
                    skc.sort()
                    skc.reverse()
                    skx.sort()
                    skx.reverse()
                    while skx and skc and skx[-1][0] == 0 and skc[-1][0] == 0:
                        (_, i, kc) = skc.pop()
                        (_, j, kx) = skx.pop()

                        namex = m.getvarname(j)
                        namec = m.getconname(i)

                        if kc in [mosek.stakey.low, mosek.stakey.fix]:
                            f.write(f" XL {namex} {namec}\n")
                        else:
                            f.write(f" XU {namex} {namec}\n")
                    while skc and skc[-1][0] == 0:
                        (_, i, kc) = skc.pop()
                        namec = m.getconname(i)
                        if kc in [mosek.stakey.low, mosek.stakey.fix]:
                            f.write(f" LL {namex}\n")
                        else:
                            f.write(f" UL {namex}\n")
                    while skx:
                        (_, j, kx) = skx.pop()
                        namex = m.getvarname(j)
                        if kx == mosek.stakey.bas:
                            f.write(f" BS {namex}\n")
                        elif kx in [mosek.stakey.low, mosek.stakey.fix]:
                            f.write(f" LL {namex}\n")
                        elif kx == mosek.stakey.upr:
                            f.write(f" UL {namex}\n")
                    f.write("ENDATA\n")

        soltype = None
        possible_soltypes = [
            mosek.soltype.bas,
            mosek.soltype.itr,
            mosek.soltype.itg,
        ]
        for possible_soltype in possible_soltypes:
            try:
                if m.solutiondef(possible_soltype):
                    soltype = possible_soltype
            except mosek.Error:
                pass

        if solution_fn is not None:
            try:
                m.writesolution(mosek.soltype.bas, path_to_string(solution_fn))
            except mosek.Error as err:
                logger.info("Unable to save solution file. Raised error: %s", err)

        condition = str(m.getsolsta(soltype))
        termination_condition = CONDITION_MAP.get(condition, condition)
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = condition

        def get_solver_solution() -> Solution:
            objective = m.getprimalobj(soltype)

            sol = np.asarray(m.getxx(soltype), dtype=float)

            try:
                dual = np.asarray(m.gety(soltype), dtype=float)
            except (mosek.Error, AttributeError):
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = np.array([], dtype=float)

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        self.io_api = io_api
        return self._make_result(status, solution, solver_model=m)


class COPT(Solver[None]):
    """
    Solver subclass for the COPT solver.

    https://guide.coap.online/copt/en-doc/index.html

    For more information on solver options, see
    https://guide.coap.online/copt/en-doc/parameter.html

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "COPT"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.READ_MODEL_FROM_FILE,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
        }
    )

    def __init(
        self,
        **solver_options: Any,
    ) -> None:
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> Result:
        msg = "Direct API not implemented for COPT"
        raise NotImplementedError(msg)

    def solve_problem_from_file(
        self,
        problem_fn: Path,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
    ) -> Result:
        """
        Solve a linear problem from a problem file using the COPT solver.

        Parameters
        ----------
        problem_fn : Path
            Path to the problem file.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        env : None, optional
            COPT environment for the solver

        Returns
        -------
        Result
        """
        # conditions: https://guide.coap.online/copt/en-doc/constant.html#chapconst-solstatus
        CONDITION_MAP = {
            0: "unstarted",
            1: "optimal",
            2: "infeasible",
            3: "unbounded",
            4: "infeasible_or_unbounded",
            5: "numerical",
            6: "node_limit",
            7: "imprecise",
            8: "time_limit",
            9: "unfinished",
            10: "interrupted",
        }

        io_api = read_io_api_from_problem_file(problem_fn)
        sense = read_sense_from_problem_file(problem_fn)

        if env is None:
            env_ = coptpy.Envr()

        m = env_.createModel()

        m.read(path_to_string(problem_fn))

        if log_fn is not None:
            m.setLogFile(path_to_string(log_fn))

        for k, v in self.solver_options.items():
            m.setParam(k, v)

        if warmstart_fn is not None:
            m.readBasis(path_to_string(warmstart_fn))

        m.solve()

        if basis_fn and m.HasBasis:
            try:
                m.write(path_to_string(basis_fn))
            except coptpy.CoptError as err:
                logger.info("No model basis stored. Raised error: %s", err)

        if solution_fn:
            try:
                m.write(path_to_string(solution_fn))
            except coptpy.CoptError as err:
                logger.info("No model solution stored. Raised error: %s", err)

        # TODO: check if this suffices
        condition = m.MipStatus if m.ismip else m.LpStatus
        termination_condition = CONDITION_MAP.get(condition, str(condition))
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = str(condition)

        def get_solver_solution() -> Solution:
            # TODO: check if this suffices
            objective = m.BestObj if m.ismip else m.LpObjVal

            sol = np.array([v.x for v in m.getVars()], dtype=float)

            try:
                dual = np.array([c.pi for c in m.getConstrs()], dtype=float)
            except (coptpy.CoptError, AttributeError):
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = np.array([], dtype=float)

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        env_.close()

        self.io_api = io_api
        return self._make_result(status, solution, solver_model=m)


class MindOpt(Solver[None]):
    """
    Solver subclass for the MindOpt solver.

    https://solver.damo.alibaba.com/doc/en/html/index.html

    For more information on solver options, see
    https://solver.damo.alibaba.com/doc/en/html/API2/param/index.html

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "MindOpt"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.INTEGER_VARIABLES,
            SolverFeature.QUADRATIC_OBJECTIVE,
            SolverFeature.LP_FILE_NAMES,
            SolverFeature.READ_MODEL_FROM_FILE,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
        }
    )

    def __init(
        self,
        **solver_options: Any,
    ) -> None:
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> Result:
        msg = "Direct API not implemented for MindOpt"
        raise NotImplementedError(msg)

    def solve_problem_from_file(
        self,
        problem_fn: Path,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
    ) -> Result:
        """
        Solve a linear problem from a problem file using the MindOpt solver.

        Parameters
        ----------
        problem_fn : Path
            Path to the problem file.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        env : None, optional
            MindOpt environment for the solver

        Returns
        -------
        Result

        """
        CONDITION_MAP = {
            -1: "error",
            0: "unknown",
            1: "optimal",
            2: "infeasible",
            3: "unbounded",
            4: "infeasible_or_unbounded",
            5: "suboptimal",
        }
        io_api = read_io_api_from_problem_file(problem_fn)
        sense = read_sense_from_problem_file(problem_fn)

        if io_api == "lp":
            # for model type "QP", lp file with have "[" and "]" in objective function
            if "[" in open(problem_fn).read() and "]" in open(problem_fn).read():
                msg = (
                    "MindOpt does not support QP problems in LP format. Use MPS file "
                    "format instead."
                )
                raise ValueError(msg)

        if env is None:
            env_ = mindoptpy.Env(path_to_string(log_fn) if log_fn else "")

        env_.start()

        m = mindoptpy.read(path_to_string(problem_fn), env_)

        for k, v in self.solver_options.items():
            m.setParam(k, v)

        if warmstart_fn:
            try:
                m.read(path_to_string(warmstart_fn))
            except mindoptpy.MindoptError as err:
                logger.info("Model basis could not be read. Raised error: %s", err)

        m.optimize()

        if basis_fn:
            try:
                m.write(path_to_string(basis_fn))
            except mindoptpy.MindoptError as err:
                logger.info("No model basis stored. Raised error: %s", err)

        if solution_fn:
            try:
                m.write(path_to_string(solution_fn))
            except mindoptpy.MindoptError as err:
                logger.info("No model solution stored. Raised error: %s", err)

        condition = m.status
        termination_condition = CONDITION_MAP.get(condition, condition)
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = condition

        def get_solver_solution() -> Solution:
            objective = m.objval

            sol = np.array([v.X for v in m.getVars()], dtype=float)

            try:
                dual = np.array([c.DualSoln for c in m.getConstrs()], dtype=float)
            except (mindoptpy.MindoptError, AttributeError):
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = np.array([], dtype=float)

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        m.dispose()
        env_.dispose()

        self.io_api = io_api
        return self._make_result(status, solution, solver_model=m)


class PIPS(Solver[None]):
    """
    Solver subclass for the PIPS solver.
    """

    def __init__(
        self,
        **solver_options: Any,
    ) -> None:
        super().__init__(**solver_options)
        msg = "The PIPS solver interface is not yet implemented."
        raise NotImplementedError(msg)


class cuPDLPx(Solver[None]):
    """
    Solver subclass for the cuPDLPx solver. cuPDLPx must be installed
    with working GPU support for usage. Find the installation instructions
    at https://github.com/MIT-Lu-Lab/cuPDLPx.

    The full list of solver options provided with the python interface
    is documented at https://github.com/MIT-Lu-Lab/cuPDLPx/tree/main/python.

    Some example options are:
    * LogToConsole : False by default.
    * TimeLimit : 3600.0 by default.
    * IterationLimit : 2147483647 by default.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    display_name: ClassVar[str] = "cuPDLPx"
    features: ClassVar[frozenset[SolverFeature]] = frozenset(
        {
            SolverFeature.DIRECT_API,
            SolverFeature.GPU_ACCELERATION,
            SolverFeature.SOLUTION_FILE_NOT_NEEDED,
        }
    )

    def __init__(
        self,
        **solver_options: Any,
    ) -> None:
        super().__init__(**solver_options)

    def solve_problem_from_file(
        self,
        problem_fn: Path,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: EnvType | None = None,
    ) -> Result:
        """
        Solve a linear problem from a problem file using the solver cuPDLPx.
        cuPDLPx does not currently support its own file IO, so this function
        reads the problem file using linopy (only support netcf files) and
        then passes the model to cuPDLPx for solving.
        If the solution is feasible the function returns the
        objective, solution and dual constraint variables.

        Parameters
        ----------
        problem_fn : Path
            Path to the problem file.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        env : None, optional
            Environment for the solver

        Returns
        -------
        Result
        """
        logger.warning(
            "cuPDLPx doesn't currently support file IO. Building model from file using linopy."
        )
        problem_fn_ = path_to_string(problem_fn)

        if problem_fn_.endswith(".netcdf"):
            model: Model = linopy.io.read_netcdf(problem_fn_)
        else:
            msg = "linopy currently only supports reading models from netcdf files. Try using io_api='direct' instead."
            raise NotImplementedError(msg)

        return self.solve_problem_from_model(
            model,
            solution_fn=solution_fn,
            log_fn=log_fn,
            warmstart_fn=warmstart_fn,
            basis_fn=basis_fn,
            env=env,
        )

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: EnvType | None = None,
        explicit_coordinate_names: bool = False,
        set_names: bool = True,
    ) -> Result:
        self.to_solver_model(model)

        return self._solve(
            self.solver_model,
            solution_fn=solution_fn,
            log_fn=log_fn,
            warmstart_fn=warmstart_fn,
            basis_fn=basis_fn,
            io_api="direct",
            sense=model.sense,
        )

    def to_solver_model(self, model: Model, **kwargs: Any) -> cupdlpx.Model:
        if model.type in ["QP", "MILP"]:
            msg = "cuPDLPx does not currently support QP or MILP problems."
            raise NotImplementedError(msg)
        if kwargs.get("explicit_coordinate_names"):
            warnings.warn(
                "cuPDLPx does not support named variables/constraints. "
                "The explicit_coordinate_names parameter is ignored.",
                UserWarning,
                stacklevel=2,
            )
        cu_model = self._build_solver_model(model)
        self.solver_model = cu_model
        self.io_api = "direct"
        self.sense = model.sense
        return cu_model

    @staticmethod
    def _build_solver_model(model: Model) -> cupdlpx.Model:
        """Build a cupdlpx.Model that mirrors the linopy `model`."""
        if model.variables.semi_continuous:
            raise NotImplementedError(
                "Semi-continuous variables are not supported by cuPDLPx. "
                "Use a solver that supports them (gurobi, cplex, highs)."
            )

        M = model.matrices
        if M.A is None:
            raise ValueError("Model has no constraints, cannot export to cuPDLPx.")
        A = M.A.tocsr()
        lower = np.where(
            np.logical_or(np.equal(M.sense, ">"), np.equal(M.sense, "=")),
            M.b,
            -np.inf,
        )
        upper = np.where(
            np.logical_or(np.equal(M.sense, "<"), np.equal(M.sense, "=")),
            M.b,
            np.inf,
        )

        cu_model = cupdlpx.Model(
            objective_vector=M.c,
            constraint_matrix=A,
            constraint_lower_bound=lower,
            constraint_upper_bound=upper,
            variable_lower_bound=M.lb,
            variable_upper_bound=M.ub,
        )

        if model.objective.sense == "max":
            cu_model.ModelSense = cupdlpx.PDLP.MAXIMIZE

        return cu_model

    def _run(self) -> Result:
        return self._solve(
            self.solver_model,
            io_api=self.io_api,
            sense=self.sense,
        )

    def _solve(
        self,
        cu_model: cupdlpx.Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        io_api: str | None = None,
        sense: str | None = None,
    ) -> Result:
        """
        Solve a linear problem from a cupdlpx.Model object.

        Parameters
        ----------
        cu_model: cupdlpx.Model
            cupdlpx object.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        model : linopy.model, optional
            Linopy model for the problem.
        io_api: str
            io_api of the problem. For direct API from linopy model this is "direct".
        sense: str
            "min" or "max"

        Returns
        -------
        Result
        """

        # see https://github.com/MIT-Lu-Lab/cuPDLPx/blob/main/python/cupdlpx/PDLP.py
        CONDITION_MAP: dict[int, TerminationCondition] = {
            cupdlpx.PDLP.OPTIMAL: TerminationCondition.optimal,
            cupdlpx.PDLP.PRIMAL_INFEASIBLE: TerminationCondition.infeasible,
            cupdlpx.PDLP.DUAL_INFEASIBLE: TerminationCondition.infeasible_or_unbounded,
            cupdlpx.PDLP.TIME_LIMIT: TerminationCondition.time_limit,
            cupdlpx.PDLP.ITERATION_LIMIT: TerminationCondition.iteration_limit,
            cupdlpx.PDLP.UNSPECIFIED: TerminationCondition.unknown,
        }

        self._set_solver_params(cu_model)

        if warmstart_fn is not None:
            # cuPDLPx supports warmstart, but there currently isn't the tooling
            # to read it in from a file
            raise NotImplementedError("Warmstarting not yet implemented for cuPDLPx.")
        else:
            cu_model.clearWarmStart()

        if basis_fn is not None:
            logger.warning("Basis files are not supported by cuPDLPx. Ignoring.")

        if log_fn is not None:
            logger.warning("Log files are not supported by cuPDLPx. Ignoring.")

        # solve
        cu_model.optimize()

        # parse solution and output
        if solution_fn is not None:
            raise NotImplementedError(
                "Solution file output not yet implemented for cuPDLPx."
            )

        termination_condition = CONDITION_MAP.get(
            cu_model.StatusCode, cu_model.StatusCode
        )
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = cu_model.Status  # cuPDLPx status message

        def get_solver_solution() -> Solution:
            objective = cu_model.ObjVal
            sol = np.asarray(cu_model.X, dtype=float)
            dual = np.asarray(cu_model.Pi, dtype=float)

            if cu_model.ModelSense == cupdlpx.PDLP.MAXIMIZE:
                dual = -dual

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        runtime: float | None = None
        with contextlib.suppress(Exception):
            runtime = float(cu_model.Runtime)

        self.io_api = io_api
        return self._make_result(
            status,
            solution,
            solver_model=cu_model,
            report=SolverReport(runtime=runtime),
        )

    def _set_solver_params(self, cu_model: cupdlpx.Model) -> None:
        """
        Set solver options for cuPDLPx model.

        For list of available options, see
        https://github.com/MIT-Lu-Lab/cuPDLPx/tree/main/python#parameters
        """
        for k, v in self.solver_options.items():
            cu_model.setParam(k, v)


def _solver_class_for(name: str) -> type[Solver] | None:
    try:
        return globals().get(SolverName(name).name)
    except ValueError:
        return None


QUADRATIC_SOLVERS = [
    n.value
    for n in SolverName
    if (cls := _solver_class_for(n.value)) is not None
    and cls.supports(SolverFeature.QUADRATIC_OBJECTIVE)
]
NO_SOLUTION_FILE_SOLVERS = [
    n.value
    for n in SolverName
    if (cls := _solver_class_for(n.value)) is not None
    and cls.supports(SolverFeature.SOLUTION_FILE_NOT_NEEDED)
]
quadratic_solvers = [s for s in QUADRATIC_SOLVERS if s in available_solvers]
