#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linopy module for solving lp files with different solvers.
"""
import io
import logging
import os
import re
import subprocess as sub
from pathlib import Path

import pandas as pd

available_solvers = []

if os.name == "nt":
    which = "where"
else:
    which = "which"

if sub.run([which, "glpsol"], stdout=sub.DEVNULL).returncode == 0:
    available_solvers.append("glpk")

if sub.run([which, "cbc"], stdout=sub.DEVNULL).returncode == 0:
    available_solvers.append("cbc")

try:
    import gurobipy

    available_solvers.append("gurobi")
except (ModuleNotFoundError, ImportError):
    pass

try:
    import highspy

    available_solvers.append("highs")
except (ModuleNotFoundError, ImportError):
    pass


try:
    import cplex

    available_solvers.append("cplex")
except (ModuleNotFoundError, ImportError):
    pass

try:
    import xpress

    available_solvers.append("xpress")
except (ModuleNotFoundError, ImportError):
    pass

logger = logging.getLogger(__name__)


io_structure = dict(
    lp_file={"gurobi", "xpress", "cbc", "glpk", "cplex"}, blocks={"pips"}
)


def set_int_index(series):
    """
    Convert string index to int index.
    """
    series.index = series.index.str[1:].astype(int)
    return series


def maybe_convert_path(path):
    """
    Convert a pathlib.Path to a string.
    """
    return str(path.resolve()) if isinstance(path, Path) else path


def run_cbc(
    Model,
    io_api=None,
    problem_fn=None,
    solution_fn=None,
    log_fn=None,
    warmstart_fn=None,
    basis_fn=None,
    keep_files=False,
    **solver_options,
):
    """
    Solve a linear problem using the cbc solver.

    The function reads the linear problem file and passes it to the cbc
    solver. If the solution is successful it returns variable solutions
    and constraint dual values. For more information on the solver
    options, run 'cbc' in your shell
    """
    if io_api is not None and (io_api != "lp"):
        logger.warning(
            f"IO setting '{io_api}' not available for cbc solver. "
            "Falling back to `lp`."
        )

    problem_fn = Model.to_file(problem_fn)

    # printingOptions is about what goes in solution file
    command = f"cbc -printingOptions all -import {problem_fn} "

    if warmstart_fn:
        command += f"-basisI {warmstart_fn} "

    command += " ".join("-" + " ".join([k, str(v)]) for k, v in solver_options.items())
    command += f"-solve -solu {solution_fn} "

    if basis_fn:
        command += f"-basisO {basis_fn} "

    if not os.path.exists(solution_fn):
        os.mknod(solution_fn)

    if log_fn is None:
        p = sub.Popen(command.split(" "), stdout=sub.PIPE, stderr=sub.PIPE)
        for line in iter(p.stdout.readline, b""):
            print(line.decode(), end="")
        p.stdout.close()
        p.wait()
    else:
        log_f = open(log_fn, "w")
        p = sub.Popen(command.split(" "), stdout=log_f, stderr=log_f)
        p.wait()

    with open(solution_fn, "r") as f:
        data = f.readline()

    if data.startswith("Optimal - objective value"):
        status = "ok"
        termination_condition = "optimal"
    elif "Infeasible" in data:
        status = "warning"
        termination_condition = "infeasible"
    else:
        status = "warning"
        termination_condition = "other"

    if termination_condition != "optimal":
        return dict(status=status, termination_condition=termination_condition)

    objective = float(data[len("Optimal - objective value ") :])

    with open(solution_fn, "rb") as f:
        trimmed_sol_fn = re.sub(rb"\*\*\s+", b"", f.read())

    data = pd.read_csv(
        io.BytesIO(trimmed_sol_fn),
        header=None,
        skiprows=[0],
        sep=r"\s+",
        usecols=[1, 2, 3],
        index_col=0,
    )
    variables_b = data.index.str[0] == "x"

    solution = data[variables_b][2].pipe(set_int_index)
    dual = data[~variables_b][3].pipe(set_int_index)

    return dict(
        status=status,
        termination_condition=termination_condition,
        solution=solution,
        dual=dual,
        objective=objective,
    )


def run_glpk(
    Model,
    io_api=None,
    problem_fn=None,
    solution_fn=None,
    log_fn=None,
    warmstart_fn=None,
    basis_fn=None,
    keep_files=False,
    **solver_options,
):
    """
    Solve a linear problem using the glpk solver.

    This function reads the linear problem file and passes it to the glpk
    solver. If the solution is successful it returns variable solutions and
    constraint dual values.

    For more information on the glpk solver options:
    https://kam.mff.cuni.cz/~elias/glpk.pdf
    """
    if io_api is not None and (io_api != "lp"):
        logger.warning(
            f"IO setting '{io_api}' not available for glpk solver. "
            "Falling back to `lp`."
        )

    problem_fn = Model.to_file(problem_fn)

    # TODO use --nopresol argument for non-optimal solution output
    command = f"glpsol --lp {problem_fn} --output {solution_fn}"
    if log_fn is not None:
        command += f" --log {log_fn}"
    if warmstart_fn:
        command += f" --ini {warmstart_fn}"
    if basis_fn:
        command += f" -w {basis_fn}"
    command += " ".join("-" + " ".join([k, str(v)]) for k, v in solver_options.items())

    p = sub.Popen(command.split(" "), stdout=sub.PIPE, stderr=sub.PIPE)
    if log_fn is None:
        for line in iter(p.stdout.readline, b""):
            print(line.decode(), end="")
        p.stdout.close()
        p.wait()
    else:
        p.wait()

    f = open(solution_fn)

    def read_until_break(f):
        linebreak = False
        while not linebreak:
            line = f.readline()
            linebreak = line == "\n"
            yield line

    info = io.StringIO("".join(read_until_break(f))[:-2])
    info = pd.read_csv(info, sep=":", index_col=0, header=None)[1]
    termination_condition = info.Status.lower().strip()
    objective = float(re.sub(r"[^0-9\.\+\-e]+", "", info.Objective))

    if termination_condition in ["optimal", "integer optimal"]:
        status = "ok"
        termination_condition = "optimal"
    elif termination_condition == "undefined":
        status = "warning"
        termination_condition = "infeasible"
    else:
        status = "warning"

    if termination_condition != "optimal":
        return dict(status=status, termination_condition=termination_condition)

    dual_ = io.StringIO("".join(read_until_break(f))[:-2])
    dual_ = pd.read_fwf(dual_)[1:].set_index("Row name")
    if "Marginal" in dual_:
        dual = pd.to_numeric(dual_["Marginal"], "coerce").fillna(0).pipe(set_int_index)
    else:
        logger.warning("Dual values of MILP couldn't be parsed")
        dual = None

    solution = io.StringIO("".join(read_until_break(f))[:-2])
    solution = (
        pd.read_fwf(solution)[1:]
        .set_index("Column name")["Activity"]
        .astype(float)
        .pipe(set_int_index)
    )
    f.close()

    return dict(
        status=status,
        termination_condition=termination_condition,
        solution=solution,
        dual=dual,
        objective=objective,
    )


def run_highs(
    Model,
    io_api=None,
    problem_fn=None,
    solution_fn=None,
    log_fn=None,
    warmstart_fn=None,
    basis_fn=None,
    keep_files=False,
    **solver_options,
):
    """
    Highs solver function. Reads a linear problem file and passes it to the
    highs solver. If the solution is feasible the function returns the
    objective, solution and dual constraint variables. Highs must be installed
    for usage. Find the documentation at https://www.maths.ed.ac.uk/hall/HiGHS/

    . The full list of solver options is documented at
    https://www.maths.ed.ac.uk/hall/HiGHS/HighsOptions.set .

    Some examplary options are:

        * presolve : "choose" by default - "on"/"off" are alternatives.
        * solver :"choose" by default - "simplex"/"ipm" are alternatives.
        * parallel : "choose" by default - "on"/"off" are alternatives.
        * time_limit : inf by default.

    Returns
    -------
    status : string,
        "ok" or "warning"
    termination_condition : string,
        Contains "optimal", "infeasible",
    variables_sol : series
    constraints_dual : series
    objective : float
    """
    if warmstart_fn:
        logger.warning("Warmstart not available with HiGHS solver. Ignore argument.")

    if io_api is None or (io_api == "lp"):
        Model.to_file(problem_fn)
        h = highspy.Highs()
        h.readModel(maybe_convert_path(problem_fn))
    elif io_api == "direct":
        h = Model.to_highspy()
    else:
        raise ValueError(
            "Keyword argument `io_api` has to be one of `lp`, `direct` or None"
        )

    if log_fn is None:
        log_fn = Model.solver_dir / "highs.log"
    solver_options["log_file"] = maybe_convert_path(log_fn)
    logger.info(f"Log file at {solver_options['log_file']}.")

    for k, v in solver_options.items():
        h.setOptionValue(k, v)

    h.run()

    termination_condition = h.modelStatusToString(h.getModelStatus()).lower()
    status = "ok" if "optimal" in termination_condition else "warning"
    objective = h.getObjectiveValue()
    solution = h.getSolution()

    if io_api == "direct":
        sol = pd.Series(solution.col_value, Model.matrices.vlabels)
        dual = pd.Series(solution.row_value, Model.matrices.clabels)
    else:
        sol = pd.Series(solution.col_value, h.getLp().col_names_).pipe(set_int_index)
        dual = pd.Series(solution.row_value, h.getLp().row_names_).pipe(set_int_index)

    return dict(
        status=status,
        termination_condition=termination_condition,
        solution=sol,
        dual=dual,
        objective=objective,
    )


def run_cplex(
    Model,
    io_api=None,
    problem_fn=None,
    solution_fn=None,
    log_fn=None,
    warmstart_fn=None,
    basis_fn=None,
    keep_files=False,
    **solver_options,
):
    """
    Solve a linear problem using the cplex solver.

    This function reads the linear problem file and passes it to the cplex
    solver. If the solution is successful it returns variable solutions and
    constraint dual values. Cplex must be installed for using this function.

    Note if you pass additional solver_options, the key can specify deeper
    layered parameters, use a dot as a separator here,
    i.e. `**{'aa.bb.cc' : x}`.
    """
    if io_api is not None and (io_api != "lp"):
        logger.warning(
            f"IO setting '{io_api}' not available for cplex solver. "
            "Falling back to `lp`."
        )

    Model.to_file(problem_fn)

    m = cplex.Cplex()

    problem_fn = maybe_convert_path(problem_fn)
    log_fn = maybe_convert_path(log_fn)
    warmstart_fn = maybe_convert_path(warmstart_fn)
    basis_fn = maybe_convert_path(basis_fn)

    if log_fn is not None:
        log_f = open(log_fn, "w")
        m.set_results_stream(log_f)
        m.set_warning_stream(log_f)
        m.set_error_stream(log_f)
        m.set_log_stream(log_f)

    if solver_options is not None:
        for key, value in solver_options.items():
            param = m.parameters
            for key_layer in key.split("."):
                param = getattr(param, key_layer)
            param.set(value)

    m.read(problem_fn)

    if warmstart_fn:
        m.start.read_basis(warmstart_fn)
    m.solve()
    is_lp = m.problem_type[m.get_problem_type()] == "LP"

    if log_fn is not None:
        log_f.close()

    termination_condition = m.solution.get_status_string()
    if "optimal" in termination_condition:
        status = "ok"
        termination_condition = "optimal"
    else:
        status = "warning"
        return dict(status=status, termination_condition=termination_condition)

    if (status == "ok") and basis_fn and is_lp:
        try:
            m.solution.basis.write(basis_fn)
        except cplex.exceptions.errors.CplexSolverError:
            logger.info("No model basis stored")

    objective = m.solution.get_objective_value()

    solution = pd.Series(m.solution.get_values(), m.variables.get_names())
    solution = set_int_index(solution)

    if is_lp:
        dual = pd.Series(m.solution.get_dual_values(), m.linear_constraints.get_names())
        dual = set_int_index(dual)
    else:
        logger.warning("Dual values of MILP couldn't be parsed")
        dual = None

    return dict(
        status=status,
        termination_condition=termination_condition,
        solution=solution,
        dual=dual,
        objective=objective,
        model=m,
    )


def run_gurobi(
    Model,
    io_api=None,
    problem_fn=None,
    solution_fn=None,
    log_fn=None,
    warmstart_fn=None,
    basis_fn=None,
    keep_files=False,
    **solver_options,
):
    """
    Solve a linear problem using the gurobi solver.

    This function communicates with gurobi using the gurubipy package.
    """
    # disable logging for this part, as gurobi output is doubled otherwise
    logging.disable(50)

    log_fn = maybe_convert_path(log_fn)
    warmstart_fn = maybe_convert_path(warmstart_fn)
    basis_fn = maybe_convert_path(basis_fn)

    if io_api is None or (io_api == "lp"):
        problem_fn = Model.to_file(problem_fn)
        problem_fn = maybe_convert_path(problem_fn)
        m = gurobipy.read(problem_fn)
    elif io_api == "direct":
        problem_fn = None
        m = Model.to_gurobipy()
    else:
        raise ValueError(
            "Keyword argument `io_api` has to be one of `lp`, `direct` or None"
        )

    if solver_options is not None:
        for key, value in solver_options.items():
            m.setParam(key, value)
    if log_fn is not None:
        m.setParam("logfile", log_fn)

    if warmstart_fn:
        m.read(warmstart_fn)
    m.optimize()
    logging.disable(1)

    if basis_fn:
        try:
            m.write(basis_fn)
        except gurobipy.GurobiError as err:
            logger.info("No model basis stored. Raised error: ", err)

    Status = gurobipy.GRB.Status
    statusmap = {
        getattr(Status, s): s.lower() for s in Status.__dir__() if not s.startswith("_")
    }
    termination_condition = statusmap[m.status]

    if termination_condition == "optimal":
        status = "ok"
    elif termination_condition == "suboptimal":
        status = "warning"
    elif termination_condition == "inf_or_unbd":
        status = "warning"
        termination_condition = "infeasible or unbounded"
    else:
        status = "warning"

    if termination_condition not in ["optimal", "suboptimal"]:
        return dict(
            status=status,
            termination_condition=termination_condition,
            model=m,
        )

    objective = m.ObjVal

    solution = pd.Series({v.VarName: v.x for v in m.getVars()})
    solution = set_int_index(solution)

    try:
        dual = pd.Series({c.ConstrName: c.Pi for c in m.getConstrs()})
        dual = set_int_index(dual)
    except AttributeError:
        logger.warning("Dual values of MILP couldn't be parsed")
        dual = None

    return dict(
        status=status,
        termination_condition=termination_condition,
        solution=solution,
        dual=dual,
        objective=objective,
        model=m,
    )


def run_xpress(
    Model,
    io_api=None,
    problem_fn=None,
    solution_fn=None,
    log_fn=None,
    warmstart_fn=None,
    basis_fn=None,
    keep_files=False,
    **solver_options,
):
    """
    Solve a linear problem using the xpress solver.

    This function reads the linear problem file and passes it to
    the Xpress solver. If the solution is successful it returns
    variable solutions and constraint dual values. The xpress module
    must be installed for using this function.

    For more information on solver options:
    https://www.fico.com/fico-xpress-optimization/docs/latest/solver/GUID-ACD7E60C-7852-36B7-A78A-CED0EA291CDD.html
    """
    if io_api is not None and (io_api != "lp"):
        logger.warning(
            f"IO setting '{io_api}' not available for xpress solver. "
            "Falling back to `lp`."
        )

    problem_fn = Model.to_file(problem_fn)

    m = xpress.problem()

    problem_fn = maybe_convert_path(problem_fn)
    log_fn = maybe_convert_path(log_fn)
    warmstart_fn = maybe_convert_path(warmstart_fn)
    basis_fn = maybe_convert_path(basis_fn)

    m.read(problem_fn)
    m.setControl(solver_options)

    if log_fn is not None:
        m.setlogfile(log_fn)

    if warmstart_fn:
        m.readbasis(warmstart_fn)

    m.solve()

    if basis_fn:
        try:
            m.writebasis(basis_fn)
        except Exception as err:
            logger.info("No model basis stored. Raised error: ", err)

    termination_condition = m.getProbStatusString()

    if termination_condition == "mip_optimal" or termination_condition == "lp_optimal":
        status = "ok"
        termination_condition = "optimal"
    elif (
        termination_condition == "mip_unbounded"
        or termination_condition == "mip_infeasible"
        or termination_condition == "lp_unbounded"
        or termination_condition == "lp_infeasible"
        or termination_condition == "lp_infeas"
    ):
        status = "warning"
        termination_condition = "infeasible or unbounded"
    else:
        status = "warning"

    if termination_condition not in ["optimal"]:
        return dict(status=status, termination_condition=termination_condition)

    objective = m.getObjVal()

    var = [str(v) for v in m.getVariable()]

    solution = pd.Series(m.getSolution(var), index=var)
    solution = set_int_index(solution)

    try:
        dual = [str(d) for d in m.getConstraint()]
        dual = pd.Series(m.getDual(dual), index=dual)
        dual = set_int_index(dual)
    except xpress.SolverError:
        logger.warning("Dual values of MILP couldn't be parsed")
        dual = None

    return dict(
        status=status,
        termination_condition=termination_condition,
        solution=solution,
        dual=dual,
        objective=objective,
        model=m,
    )


def run_pips(
    Model,
    io_api=None,
    problem_fn=None,
    solution_fn=None,
    log_fn=None,
    warmstart_fn=None,
    basis_fn=None,
    keep_files=False,
    **solver_options,
):
    """
    Solve a linear problem using the PIPS solver.
    """
    raise NotImplementedError("The PIPS++ solver interface is not yet implemented.")
