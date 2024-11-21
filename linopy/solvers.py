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
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import pandas as pd
from pandas.core.series import Series

from linopy.constants import (
    Result,
    Solution,
    SolverStatus,
    Status,
    TerminationCondition,
)

if TYPE_CHECKING:
    from linopy.model import Model

QUADRATIC_SOLVERS = [
    "gurobi",
    "xpress",
    "cplex",
    "highs",
    "scip",
    "mosek",
    "copt",
    "mindopt",
]

FILE_IO_APIS = ["lp", "lp-polars", "mps"]
IO_APIS = FILE_IO_APIS + ["direct"]

available_solvers = []

which = "where" if os.name == "nt" else "which"

# the first available solver will be the default solver
with contextlib.suppress(ModuleNotFoundError):
    import gurobipy

    available_solvers.append("gurobi")
with contextlib.suppress(ModuleNotFoundError):
    _new_highspy_mps_layout = None
    import highspy

    available_solvers.append("highs")
    from importlib.metadata import version

    if version("highspy") < "1.7.1":
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

with contextlib.suppress(ModuleNotFoundError):
    import xpress

    available_solvers.append("xpress")

with contextlib.suppress(ModuleNotFoundError):
    import mosek

    with contextlib.suppress(mosek.Error):
        with mosek.Env() as m:
            t = m.Task()
            t.optimize()
            m.checkinall()

        available_solvers.append("mosek")

with contextlib.suppress(ModuleNotFoundError):
    import mindoptpy

    available_solvers.append("mindopt")

with contextlib.suppress(ModuleNotFoundError):
    import coptpy

    with contextlib.suppress(coptpy.CoptError):
        coptpy.Envr()

        available_solvers.append("copt")

quadratic_solvers = [s for s in QUADRATIC_SOLVERS if s in available_solvers]
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


def set_int_index(series: Series) -> Series:
    """
    Convert string index to int index.
    """
    series.index = series.index.str[1:].astype(int)
    return series


# using enum to match solver subclasses with names
class SolverName(enum.Enum):
    CBC = "cbc"
    GLPK = "glpk"
    Highs = "highs"
    Cplex = "cplex"
    Gurobi = "gurobi"
    SCIP = "scip"
    Xpress = "xpress"
    Mosek = "mosek"
    COPT = "copt"
    MindOpt = "mindopt"
    PIPS = "pips"


def path_to_string(path: Path) -> str:
    """
    Convert a pathlib.Path to a string.
    """
    return str(path.resolve())


def read_sense_from_problem_file(problem_fn: Path | str):
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


def read_io_api_from_problem_file(problem_fn: Path | str):
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


class Solver(ABC):
    """
    Abstract base class for solving a given linear problem.

    All relevant functions are passed on to the specific solver subclasses.
    Subclasses must implement the `solve_problem_from_model()` and
    `solve_problem_from_file()` methods.
    """

    def __init__(
        self,
        **solver_options,
    ):
        self.solver_options = solver_options

        # Check for the solver to be initialized whether the package is installed or not.
        if self.solver_name.value not in available_solvers:
            msg = f"Solver package for '{self.solver_name.value}' is not installed. Please install first to initialize solver instance."
            raise ImportError(msg)

    def safe_get_solution(self, status: Status, func: Callable) -> Solution:
        """
        Get solution from function call, if status is unknown still try to run it.
        """
        if status.is_ok:
            return func()
        elif status.status == SolverStatus.unknown:
            with contextlib.suppress(Exception):
                logger.warning("Solution status unknown. Trying to parse solution.")
                return func()
        return Solution()

    @abstractmethod
    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
    ) -> Result:
        """
        Abstract method to solve a linear problem from a model.

        Needs to be implemented in the specific solver subclass. Even if the solver
        does not support solving from a model, this method should be implemented and
        raise a NotImplementedError.
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
        env: None = None,
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
        env: None = None,
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


class CBC(Solver):
    """
    Solver subclass for the CBC solver.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    def __init__(
        self,
        **solver_options,
    ):
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
    ):
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
            log_f = open(log_fn, "w")
            p = sub.Popen(command.split(" "), stdout=log_f, stderr=log_f)
            p.wait()

        with open(solution_fn) as f:
            data = f.readline()

        if data.startswith("Optimal - objective value"):
            status = Status.from_termination_condition("optimal")
        elif "Infeasible" in data:
            status = Status.from_termination_condition("infeasible")
        else:
            status = Status(SolverStatus.warning, TerminationCondition.unknown)
        status.legacy_status = data

        def get_solver_solution():
            objective = float(data[len("Optimal - objective value ") :])

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
            variables_b = df.index.str[0] == "x"

            sol = df[variables_b][2].pipe(set_int_index)
            dual = df[~variables_b][3].pipe(set_int_index)
            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        return Result(status, solution)


class GLPK(Solver):
    """
    Solver subclass for the GLPK solver.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    def __init__(
        self,
        **solver_options,
    ):
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
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
        command = f"glpsol --{io_api} {problem_fn} --output {solution_fn} "
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
            return Result(status, Solution())

        f = open(solution_fn)

        def read_until_break(f):
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
                    .pipe(set_int_index)
                )
            else:
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = pd.Series(dtype=float)

            sol_io = io.StringIO("".join(read_until_break(f))[:-2])
            sol = (
                pd.read_fwf(sol_io)[1:]
                .set_index("Column name")["Activity"]
                .astype(float)
                .pipe(set_int_index)
            )
            f.close()
            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)
        return Result(status, solution)


class Highs(Solver):
    """
    Solver subclass for the Highs solver. Highs must be installed
    for usage. Find the documentation at https://www.maths.ed.ac.uk/hall/HiGHS/.

    The full list of solver options is documented at https://www.maths.ed.ac.uk/hall/HiGHS/HighsOptions.set.

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

    def __init__(
        self,
        **solver_options,
    ):
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
    ) -> Result:
        """
        Solve a linear problem directly from a linopy model using the Highs solver.
        Reads a linear problem file and passes it to the highs solver.
        If the solution is feasible the function returns the
        objective, solution and dual constraint variables.

        Parameters
        ----------
        model : linopy.model
            Linopy model for the problem.
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
        # check for Highs solver compatibility
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

        h = model.to_highspy()

        if log_fn is None and model is not None:
            log_fn = model.solver_dir / "highs.log"

        return self._solve(
            h,
            solution_fn,
            log_fn,
            warmstart_fn,
            basis_fn,
            model=model,
            io_api="direct",
            sense=model.sense,
        )

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
        Solve a linear problem from a problem file using the Highs solver.
        Reads a linear problem file and passes it to the highs solver.
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
        h.readModel(problem_fn_)

        return self._solve(
            h,
            solution_fn,
            log_fn,
            warmstart_fn,
            basis_fn,
            io_api=read_io_api_from_problem_file(problem_fn),
            sense=read_sense_from_problem_file(problem_fn),
        )

    def _solve(
        self,
        h: highspy.Highs,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        model: Model | None = None,
        io_api: str | None = None,
        sense: str | None = None,
    ) -> Result:
        """
        Solve a linear problem from a Highs object.


        Parameters
        ----------
        h : highspy.Highs
            Highs object.
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
        CONDITION_MAP: dict[str, str] = {}

        if log_fn is not None:
            self.solver_options["log_file"] = path_to_string(log_fn)
            logger.info(f"Log file at {self.solver_options['log_file']}")

        for k, v in self.solver_options.items():
            h.setOptionValue(k, v)

        if warmstart_fn is not None and warmstart_fn.suffix == ".sol":
            h.readSolution(path_to_string(warmstart_fn), 0)
        elif warmstart_fn:
            h.readBasis(path_to_string(warmstart_fn))

        h.run()

        condition = h.modelStatusToString(h.getModelStatus()).lower()
        termination_condition = CONDITION_MAP.get(condition, condition)
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = condition

        if basis_fn:
            h.writeBasis(path_to_string(basis_fn))

        if solution_fn:
            h.writeSolution(path_to_string(solution_fn), 0)

        def get_solver_solution() -> Solution:
            objective = h.getObjectiveValue()
            solution = h.getSolution()

            if model is not None:
                sol = pd.Series(solution.col_value, model.matrices.vlabels, dtype=float)
                dual = pd.Series(solution.row_dual, model.matrices.clabels, dtype=float)
            else:
                sol = pd.Series(
                    solution.col_value, h.getLp().col_names_, dtype=float
                ).pipe(set_int_index)
                dual = pd.Series(
                    solution.row_dual, h.getLp().row_names_, dtype=float
                ).pipe(set_int_index)

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        return Result(status, solution, h)


class Gurobi(Solver):
    """
    Solver subclass for the gurobi solver.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    def __init__(
        self,
        **solver_options,
    ):
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
    ) -> Result:
        """
        Solve a linear problem directly from a linopy model using the Gurobi solver.
        Reads a problem file and passes it to the Gurobi solver.
        This function communicates with gurobi using the gurobipy package.

        Parameters
        ----------
        model : linopy.model
            Linopy model for the problem.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        env : None, optional
            Gurobi environment for the solver

        Returns
        -------
        Result
        """
        with contextlib.ExitStack() as stack:
            if env is None:
                env_ = stack.enter_context(gurobipy.Env())
            else:
                env_ = env

            m = model.to_gurobipy(env=env_)

            return self._solve(
                m,
                solution_fn=solution_fn,
                log_fn=log_fn,
                warmstart_fn=warmstart_fn,
                basis_fn=basis_fn,
                io_api="direct",
                sense=model.sense,
            )

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
        env : None, optional
            Gurobi environment for the solver

        Returns
        -------
        Result
        """
        sense = read_sense_from_problem_file(problem_fn)
        io_api = read_io_api_from_problem_file(problem_fn)
        problem_fn_ = path_to_string(problem_fn)

        with contextlib.ExitStack() as stack:
            if env is None:
                env_ = stack.enter_context(gurobipy.Env())
            else:
                env_ = env

            m = gurobipy.read(problem_fn_, env=env_)

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
        m,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        io_api: str | None = None,
        sense: str | None = None,
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

            sol = pd.Series({v.VarName: v.x for v in m.getVars()}, dtype=float)
            sol = set_int_index(sol)

            try:
                dual = pd.Series(
                    {c.ConstrName: c.Pi for c in m.getConstrs()}, dtype=float
                )
                dual = set_int_index(dual)
            except AttributeError:
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = pd.Series(dtype=float)

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = solution = maybe_adjust_objective_sign(solution, io_api, sense)

        return Result(status, solution, m)


class Cplex(Solver):
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

    def __init__(
        self,
        **solver_options,
    ):
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
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

            solution = pd.Series(
                m.solution.get_values(), m.variables.get_names(), dtype=float
            )
            solution = set_int_index(solution)

            if is_lp:
                dual = pd.Series(
                    m.solution.get_dual_values(),
                    m.linear_constraints.get_names(),
                    dtype=float,
                )
                dual = set_int_index(dual)
            else:
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = pd.Series(dtype=float)
            return Solution(solution, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        return Result(status, solution, m)


class SCIP(Solver):
    """
    Solver subclass for the SCIP solver.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    def __init__(
        self,
        **solver_options,
    ):
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
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
        CONDITION_MAP: dict[str, str] = {}

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

        # In order to retrieve the dual values, we need to turn off presolve
        m.setPresolve(scip.SCIP_PARAMSETTING.OFF)

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

            s = m.getSols()[0]
            sol = pd.Series({v.name: s[v] for v in m.getVars()})
            sol.drop(
                ["quadobjvar", "qmatrixvar"], errors="ignore", inplace=True, axis=0
            )
            sol = set_int_index(sol)

            cons = m.getConss()
            if len(cons) != 0:
                dual = pd.Series({c.name: m.getDualSolVal(c) for c in cons})
                dual = dual[
                    dual.index.str.startswith("c") & ~dual.index.str.startswith("cf")
                ]
                dual = set_int_index(dual)
            else:
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = pd.Series(dtype=float)

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        return Result(status, solution, m)


class Xpress(Solver):
    """
    Solver subclass for the xpress solver.

    For more information on solver options, see
    https://www.fico.com/fico-xpress-optimization/docs/latest/solver/GUID-ACD7E60C-7852-36B7-A78A-CED0EA291CDD.html

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    def __init__(
        self,
        **solver_options,
    ):
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
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
            "lp_optimal": "optimal",
            "mip_optimal": "optimal",
            "lp_infeasible": "infeasible",
            "lp_infeas": "infeasible",
            "mip_infeasible": "infeasible",
            "lp_unbounded": "unbounded",
            "mip_unbounded": "unbounded",
        }

        io_api = read_io_api_from_problem_file(problem_fn)
        sense = read_sense_from_problem_file(problem_fn)

        m = xpress.problem()

        m.read(path_to_string(problem_fn))
        m.setControl(self.solver_options)

        if log_fn is not None:
            m.setlogfile(path_to_string(log_fn))

        if warmstart_fn is not None:
            m.readbasis(path_to_string(warmstart_fn))

        m.solve()

        if basis_fn is not None:
            try:
                m.writebasis(path_to_string(basis_fn))
            except Exception as err:
                logger.info("No model basis stored. Raised error: %s", err)

        if solution_fn is not None:
            try:
                # TODO: possibly update saving of solution file
                m.writesol(path_to_string(solution_fn))
            except Exception as err:
                logger.info("Unable to save solution file. Raised error: %s", err)

        condition = m.getProbStatusString()
        termination_condition = CONDITION_MAP.get(condition, condition)
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = condition

        def get_solver_solution() -> Solution:
            objective = m.getObjVal()

            var = [str(v) for v in m.getVariable()]

            sol = pd.Series(m.getSolution(var), index=var, dtype=float)
            sol = set_int_index(sol)

            try:
                dual_ = [str(d) for d in m.getConstraint()]
                dual = pd.Series(m.getDual(dual_), index=dual_, dtype=float)
                dual = set_int_index(dual)
            except (xpress.SolverError, xpress.ModelError, SystemError):
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = pd.Series(dtype=float)

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        return Result(status, solution, m)


mosek_bas_re = re.compile(r" (XL|XU)\s+([^ \t]+)\s+([^ \t]+)| (LL|UL|BS)\s+([^ \t]+)")


class Mosek(Solver):
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

    def __init__(
        self,
        **solver_options,
    ):
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
    ) -> Result:
        """
        Solve a linear problem directly from a linopy model using the MOSEK solver.

        Parameters
        ----------
        model : linopy.model
            Linopy model for the problem.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        env : None, optional
            Gurobi environment for the solver

        Returns
        -------
        Result
        """
        with contextlib.ExitStack() as stack:
            if env is None:
                env_ = stack.enter_context(mosek.Env())

            with env_.Task() as m:
                m = model.to_mosek(m)

                return self._solve(
                    m,
                    solution_fn=solution_fn,
                    log_fn=log_fn,
                    warmstart_fn=warmstart_fn,
                    basis_fn=basis_fn,
                    io_api="direct",
                    sense=model.sense,
                )

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
        env : None, optional
            Mosek environment for the solver

        Returns
        -------
        Result
        """
        with contextlib.ExitStack() as stack:
            if env is None:
                env_ = stack.enter_context(mosek.Env())

            with env_.Task() as m:
                # read sense and io_api from problem file
                sense = read_sense_from_problem_file(problem_fn)
                io_api = read_io_api_from_problem_file(problem_fn)
                # for Mosek solver, the path needs to be a string
                problem_fn_ = path_to_string(problem_fn)
                m.readdata(problem_fn_)

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
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        io_api: str | None = None,
        sense: str | None = None,
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

            sol = m.getxx(soltype)
            sol = {m.getvarname(i): sol[i] for i in range(m.getnumvar())}
            sol = pd.Series(sol, dtype=float)
            sol = set_int_index(sol)

            try:
                dual = m.gety(soltype)
                dual = {m.getconname(i): dual[i] for i in range(m.getnumcon())}
                dual = pd.Series(dual, dtype=float)
                dual = set_int_index(dual)
            except (mosek.Error, AttributeError):
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = pd.Series(dtype=float)

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        return Result(status, solution)


class COPT(Solver):
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

    def __init__(
        self,
        **solver_options,
    ):
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
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
        termination_condition = CONDITION_MAP.get(condition, condition)
        status = Status.from_termination_condition(termination_condition)
        status.legacy_status = condition

        def get_solver_solution() -> Solution:
            # TODO: check if this suffices
            objective = m.BestObj if m.ismip else m.LpObjVal

            sol = pd.Series({v.name: v.x for v in m.getVars()}, dtype=float)
            sol = set_int_index(sol)

            try:
                dual = pd.Series({v.name: v.pi for v in m.getConstrs()}, dtype=float)
                dual = set_int_index(dual)
            except (coptpy.CoptError, AttributeError):
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = pd.Series(dtype=float)

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        env_.close()

        return Result(status, solution, m)


class MindOpt(Solver):
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

    def __init__(
        self,
        **solver_options,
    ):
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
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

            sol = pd.Series({v.varname: v.X for v in m.getVars()}, dtype=float)
            sol = set_int_index(sol)

            try:
                dual = pd.Series({c.constrname: c.DualSoln for c in m.getConstrs()})
                dual = set_int_index(dual)
            except (mindoptpy.MindoptError, AttributeError):
                logger.warning("Dual values of MILP couldn't be parsed")
                dual = pd.Series(dtype=float)

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        env_.dispose()

        return Result(status, solution, m)


class PIPS(Solver):
    """
    Solver subclass for the PIPS solver.
    """

    def __init__(
        self,
        **solver_options,
    ):
        super().__init__(**solver_options)
        msg = "The PIPS solver interface is not yet implemented."
        raise NotImplementedError(msg)
