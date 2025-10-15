#!/usr/bin/env python3
"""
Created on Thu Mar 18 08:49:08 2021.

@author: fabian
"""

from __future__ import annotations

import itertools
import logging
import platform
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.testing import assert_equal

from linopy import GREATER_EQUAL, LESS_EQUAL, Model, solvers
from linopy.common import to_path
from linopy.solvers import _new_highspy_mps_layout, available_solvers, quadratic_solvers

logger = logging.getLogger(__name__)

io_apis: list[str] = ["lp", "lp-polars"]

explicit_coordinate_names = [False, True]

if "highs" in available_solvers:
    # mps io is only supported via highspy
    io_apis.append("mps")


params: list[tuple[str, str, bool]] = list(
    itertools.product(available_solvers, io_apis, explicit_coordinate_names)
)

direct_solvers: list[str] = ["gurobi", "highs", "mosek"]
for solver in direct_solvers:
    if solver in available_solvers:
        params.append((solver, "direct", False))

if "mosek" in available_solvers:
    params.append(("mosek", "lp", False))
    params.append(("mosek", "lp", True))


feasible_quadratic_solvers: list[str] = quadratic_solvers
# There seems to be a bug in scipopt with quadratic models on windows, see
# https://github.com/PyPSA/linopy/actions/runs/7615240686/job/20739454099?pr=78
if platform.system() == "Windows" and "scip" in feasible_quadratic_solvers:
    feasible_quadratic_solvers.remove("scip")


def test_print_solvers(capsys: Any) -> None:
    with capsys.disabled():
        print(
            f"\ntesting solvers: {', '.join(available_solvers)}\n"
            f"testing quadratic solvers: {', '.join(feasible_quadratic_solvers)}"
        )


@pytest.fixture
def model() -> Model:
    m = Model(chunk=None)

    x = m.add_variables(name="x")
    y = m.add_variables(name="y")

    m.add_constraints(2 * x + 6 * y, GREATER_EQUAL, 10)
    m.add_constraints(4 * x + 2 * y, GREATER_EQUAL, 3)

    m.add_objective(2 * y + x)
    return m


@pytest.fixture
def model_anonymous_constraint() -> Model:
    m = Model(chunk=None)

    x = m.add_variables(name="x")
    y = m.add_variables(name="y")

    m.add_constraints(2 * x + 6 * y >= 10)
    m.add_constraints(4 * x + 2 * y >= 3)

    m.add_objective(2 * y + x)
    return m


@pytest.fixture
def model_chunked() -> Model:
    m = Model(chunk="auto")

    x = m.add_variables(name="x")
    y = m.add_variables(name="y")

    m.add_constraints(2 * x + 6 * y, GREATER_EQUAL, 10)
    m.add_constraints(4 * x + 2 * y, GREATER_EQUAL, 3)

    m.add_objective(2 * y + x)
    return m


@pytest.fixture
def model_maximization() -> Model:
    m = Model()

    x = m.add_variables(name="x")
    y = m.add_variables(name="y")

    m.add_constraints(2 * x + 6 * y, LESS_EQUAL, 10)
    m.add_constraints(4 * x + 2 * y, LESS_EQUAL, 3)

    m.add_objective(2 * y + x, sense="max")
    return m


@pytest.fixture
def model_with_inf() -> Model:
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(coords=[lower.index], name="x", binary=True)
    y = m.add_variables(lower, name="y")

    m.add_constraints(x + y, GREATER_EQUAL, 10)
    m.add_constraints(1 * x, "<=", np.inf)

    m.objective = 2 * x + y

    return m


@pytest.fixture
def model_with_duplicated_variables() -> Model:
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(coords=[lower.index], name="x")

    m.add_constraints(x + x, GREATER_EQUAL, 10)
    m.objective = 1 * x

    return m


@pytest.fixture
def model_with_non_aligned_variables() -> Model:
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(lower=lower, coords=[lower.index], name="x")
    lower = pd.Series(0, range(8))
    y = m.add_variables(lower=lower, coords=[lower.index], name="y")

    m.add_constraints(x + y, GREATER_EQUAL, 10.5)
    m.objective = 1 * x + 0.5 * y

    return m


@pytest.fixture
def milp_binary_model() -> Model:
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(coords=x.coords, name="y", binary=True)

    m.add_constraints(x + y, GREATER_EQUAL, 10)

    m.add_objective(2 * x + y)
    return m


@pytest.fixture
def milp_binary_model_r() -> Model:
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(coords=[lower.index], name="x", binary=True)
    y = m.add_variables(lower, name="y")

    m.add_constraints(x + y, GREATER_EQUAL, 10)

    m.add_objective(2 * x + y)
    return m


@pytest.fixture
def milp_model() -> Model:
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(lower, 9, name="y", integer=True)

    m.add_constraints(x + y, GREATER_EQUAL, 9.5)

    m.add_objective(2 * x + y)
    return m


@pytest.fixture
def milp_model_r() -> Model:
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x", integer=True)
    y = m.add_variables(lower, name="y")

    m.add_constraints(x + y, GREATER_EQUAL, 10.99)

    m.add_objective(x + 2 * y)
    return m


@pytest.fixture
def quadratic_model() -> Model:
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(lower, name="y")

    m.add_constraints(x + y, GREATER_EQUAL, 10)

    m.add_objective(x * x)
    return m


@pytest.fixture
def quadratic_model_unbounded() -> Model:
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(lower, name="y")

    m.add_constraints(x + y, GREATER_EQUAL, 10)

    m.add_objective(-x - y + x * x)
    return m


@pytest.fixture
def quadratic_model_cross_terms() -> Model:
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(lower, name="y")

    m.add_constraints(x + y >= 10)

    m.add_objective(-2 * x + y + x * x)
    return m


@pytest.fixture
def modified_model() -> Model:
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(coords=[lower.index], name="x", binary=True)
    y = m.add_variables(lower, name="y")

    c = m.add_constraints(x + y, GREATER_EQUAL, 10)

    y.lower = 9
    c.lhs = 2 * x + y
    m.objective = 2 * x + y

    return m


@pytest.fixture
def masked_variable_model() -> Model:
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x")
    mask = pd.Series([True] * 8 + [False, False])
    y = m.add_variables(lower, name="y", mask=mask)

    m.add_constraints(x + y, GREATER_EQUAL, 10)

    m.add_constraints(y, GREATER_EQUAL, 0)

    m.add_objective(2 * x + y)
    return m


@pytest.fixture
def masked_constraint_model() -> Model:
    m = Model()

    lower = pd.Series(0, range(10))
    x = m.add_variables(lower, name="x")
    y = m.add_variables(lower, name="y")

    mask = pd.Series([True] * 8 + [False, False])
    m.add_constraints(x + y, GREATER_EQUAL, 10, mask=mask)
    # for the last two entries only the following constraint will be active
    m.add_constraints(x + y, GREATER_EQUAL, 5)

    m.add_objective(2 * x + y)
    return m


def test_model_types(
    model: Model,
    model_with_duplicated_variables: Model,
    milp_binary_model: Model,
    milp_binary_model_r: Model,
    milp_model: Model,
    milp_model_r: Model,
    quadratic_model: Model,
    quadratic_model_unbounded: Model,
    quadratic_model_cross_terms: Model,
    modified_model: Model,
    masked_variable_model: Model,
    masked_constraint_model: Model,
) -> None:
    assert model.type == "LP"
    assert model_with_duplicated_variables.type == "LP"
    assert milp_binary_model.type == "MILP"
    assert milp_binary_model_r.type == "MILP"
    assert milp_model.type == "MILP"
    assert milp_model_r.type == "MILP"
    assert quadratic_model.type == "QP"
    assert quadratic_model_unbounded.type == "QP"
    assert quadratic_model_cross_terms.type == "QP"
    assert modified_model.type == "MILP"
    assert masked_variable_model.type == "LP"
    assert masked_constraint_model.type == "LP"


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_default_setting(
    model: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    assert model.objective.sense == "min"
    status, condition = model.solve(
        solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
    )
    assert status == "ok"
    assert np.isclose(model.objective.value or 0, 3.3)
    assert model.solver_name == solver


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_default_setting_sol_and_dual_accessor(
    model: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    status, condition = model.solve(
        solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
    )
    assert status == "ok"
    x = model["x"]
    assert_equal(x.solution, model.solution["x"])
    c = model.constraints["con1"]
    assert_equal(c.dual, model.dual["con1"])
    # squeeze in dual getter in matrix
    assert len(model.matrices.dual) == model.ncons
    assert model.matrices.dual[0] == model.dual["con0"]


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_default_setting_expression_sol_accessor(
    model: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    status, condition = model.solve(
        solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
    )
    assert status == "ok"
    x = model["x"]
    y = model["y"]

    expr = 4 * x
    assert_equal(expr.solution, 4 * x.solution)

    qexpr = 4 * x**2
    assert_equal(qexpr.solution, 4 * x.solution**2)

    qexpr = 4 * (x * y)  # type: ignore
    assert_equal(qexpr.solution, 4 * x.solution * y.solution)


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_anonymous_constraint(
    model: Model,
    model_anonymous_constraint: Model,
    solver: str,
    io_api: str,
    explicit_coordinate_names: bool,
) -> None:
    status, condition = model_anonymous_constraint.solve(
        solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
    )
    assert status == "ok"
    assert np.isclose(model_anonymous_constraint.objective.value or 0, 3.3)

    model.solve(
        solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
    )
    assert_equal(model.solution, model_anonymous_constraint.solution)


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_model_maximization(
    model_maximization: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    m = model_maximization
    assert m.objective.sense == "max"
    assert m.objective.value is None

    if solver in ["cbc", "glpk"] and io_api == "mps" and _new_highspy_mps_layout:
        with pytest.raises(ValueError):
            m.solve(
                solver,
                io_api=io_api,
                explicit_coordinate_names=explicit_coordinate_names,
            )
    else:
        status, condition = m.solve(
            solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
        )
        assert status == "ok"
        assert np.isclose(m.objective.value or 0, 3.3)


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_default_settings_chunked(
    model_chunked: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    status, condition = model_chunked.solve(
        solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
    )
    assert status == "ok"
    assert np.isclose(model_chunked.objective.value or 0, 3.3)


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_default_settings_small_slices(
    model: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    assert model.objective.sense == "min"
    status, condition = model.solve(
        solver,
        io_api=io_api,
        explicit_coordinate_names=explicit_coordinate_names,
        slice_size=2,
    )
    assert status == "ok"
    assert np.isclose(model.objective.value or 0, 3.3)
    assert model.solver_name == solver


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_solver_time_limit_options(
    model: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    time_limit_option = {
        "cbc": {"sec": 1},
        "gurobi": {"TimeLimit": 1},
        "glpk": {"tmlim": 1},
        "cplex": {"timelimit": 1},
        "xpress": {"maxtime": 1},
        "highs": {"time_limit": 1},
        "scip": {"limits/time": 1},
        "mosek": {"MSK_DPAR_OPTIMIZER_MAX_TIME": 1},
        "mindopt": {"MaxTime": 1},
        "copt": {"TimeLimit": 1},
    }
    status, condition = model.solve(
        solver,
        io_api=io_api,
        explicit_coordinate_names=explicit_coordinate_names,
        **time_limit_option[solver],  # type: ignore
    )
    assert status == "ok"


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_solver_method_options(
    model: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    method_options = {
        "highs": {"solver": "ipm", "run_crossover": "off", "parallel": "on"},
    }
    if solver in method_options:
        status, condition = model.solve(solver, io_api=io_api, **method_options[solver])  # type: ignore
        assert status == "ok"
        objective = model.objective.value or 0
        assert np.isclose(objective, 3.3)


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_duplicated_variables(
    model_with_duplicated_variables: Model,
    solver: str,
    io_api: str,
    explicit_coordinate_names: bool,
) -> None:
    status, condition = model_with_duplicated_variables.solve(solver, io_api=io_api)
    assert status == "ok"
    assert all(model_with_duplicated_variables.solution["x"] == 5)


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_non_aligned_variables(
    model_with_non_aligned_variables: Model,
    solver: str,
    io_api: str,
    explicit_coordinate_names: bool,
) -> None:
    status, condition = model_with_non_aligned_variables.solve(
        solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
    )
    assert status == "ok"
    with pytest.warns(UserWarning):
        assert model_with_non_aligned_variables.solution["x"][0] == 0
        assert model_with_non_aligned_variables.solution["x"][-1] == 10.5
        assert model_with_non_aligned_variables.solution["y"][0] == 10.5
        assert np.isnan(model_with_non_aligned_variables.solution["y"][-1])

        for dtype in model_with_non_aligned_variables.solution.dtypes.values():
            assert np.issubdtype(dtype, np.floating)


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_set_files(
    tmp_path: Any,
    model: Model,
    solver: str,
    io_api: str,
    explicit_coordinate_names: bool,
) -> None:
    status, condition = model.solve(
        solver,
        io_api=io_api,
        explicit_coordinate_names=explicit_coordinate_names,
        problem_fn=tmp_path / "problem.lp",
        solution_fn=tmp_path / "solution.sol",
        log_fn=tmp_path / "logging.log",
        keep_files=False,
    )
    assert status == "ok"


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_set_files_and_keep_files(
    tmp_path: Any,
    model: Model,
    solver: str,
    io_api: str,
    explicit_coordinate_names: bool,
) -> None:
    status, condition = model.solve(
        solver,
        io_api=io_api,
        explicit_coordinate_names=explicit_coordinate_names,
        problem_fn=tmp_path / "problem.lp",
        solution_fn=tmp_path / "solution.sol",
        log_fn=tmp_path / "logging.log",
        keep_files=True,
    )
    assert status == "ok"
    if io_api != "direct" and solver != "xpress":
        assert (tmp_path / "problem.lp").exists()
        assert (tmp_path / "solution.sol").exists()
    assert (tmp_path / "logging.log").exists()


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_infeasible_model(
    model: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    model.add_constraints([(1, "x")], "<=", 0)
    model.add_constraints([(1, "y")], "<=", 0)

    status, condition = model.solve(
        solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
    )
    assert status == "warning"
    assert "infeasible" in condition

    if solver in ["gurobi", "xpress"]:
        # ignore deprecated warning
        with pytest.warns(DeprecationWarning):
            model.compute_set_of_infeasible_constraints()
        model.compute_infeasibilities()
        model.print_infeasibilities()
    else:
        with pytest.raises((NotImplementedError, ImportError)):
            model.compute_infeasibilities()


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_model_with_inf(
    model_with_inf: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    status, condition = model_with_inf.solve(solver, io_api=io_api)
    assert condition == "optimal"
    assert (model_with_inf.solution.x == 0).all()
    assert (model_with_inf.solution.y == 10).all()


@pytest.mark.parametrize(
    "solver,io_api,explicit_coordinate_names",
    [p for p in params if p[0] not in ["mindopt"]],
)
def test_milp_binary_model(
    milp_binary_model: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    status, condition = milp_binary_model.solve(
        solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
    )
    assert condition == "optimal"
    assert (
        (milp_binary_model.solution.y == 1) | (milp_binary_model.solution.y == 0)
    ).all()


@pytest.mark.parametrize(
    "solver,io_api,explicit_coordinate_names",
    [p for p in params if p[0] not in ["mindopt"]],
)
def test_milp_binary_model_r(
    milp_binary_model_r: Model,
    solver: str,
    io_api: str,
    explicit_coordinate_names: bool,
) -> None:
    status, condition = milp_binary_model_r.solve(
        solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
    )
    assert condition == "optimal"
    assert (
        (milp_binary_model_r.solution.x == 1) | (milp_binary_model_r.solution.x == 0)
    ).all()


@pytest.mark.parametrize(
    "solver,io_api,explicit_coordinate_names",
    [p for p in params if p[0] not in ["mindopt"]],
)
def test_milp_model(
    milp_model: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    status, condition = milp_model.solve(
        solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
    )
    assert condition == "optimal"
    assert ((milp_model.solution.y == 9) | (milp_model.solution.x == 0.5)).all()


@pytest.mark.parametrize(
    "solver,io_api,explicit_coordinate_names",
    [p for p in params if p[0] not in ["mindopt"]],
)
def test_milp_model_r(
    milp_model_r: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    # MPS format by Highs wrong, see https://github.com/ERGO-Code/HiGHS/issues/1325
    # skip it
    if io_api != "mps":
        status, condition = milp_model_r.solve(
            solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
        )
        assert condition == "optimal"
        assert ((milp_model_r.solution.x == 11) | (milp_model_r.solution.y == 0)).all()


@pytest.mark.parametrize(
    "solver,io_api,explicit_coordinate_names",
    [
        p
        for p in params
        if (p[0], p[1]) not in [("mindopt", "lp"), ("mindopt", "lp-polars")]
    ],
)
def test_quadratic_model(
    quadratic_model: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    if solver in feasible_quadratic_solvers:
        status, condition = quadratic_model.solve(
            solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
        )
        assert condition == "optimal"
        assert (quadratic_model.solution.x.round(3) == 0).all()
        assert (quadratic_model.solution.y.round(3) >= 10).all()
        assert round(quadratic_model.objective.value or 0, 3) == 0
    else:
        with pytest.raises(ValueError):
            quadratic_model.solve(
                solver,
                io_api=io_api,
                explicit_coordinate_names=explicit_coordinate_names,
            )


@pytest.mark.parametrize(
    "solver,io_api,explicit_coordinate_names",
    [
        p
        for p in params
        if (p[0], p[1]) not in [("mindopt", "lp"), ("mindopt", "lp-polars")]
    ],
)
def test_quadratic_model_cross_terms(
    quadratic_model_cross_terms: Model,
    solver: str,
    io_api: str,
    explicit_coordinate_names: bool,
) -> None:
    if solver in feasible_quadratic_solvers:
        status, condition = quadratic_model_cross_terms.solve(
            solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
        )
        assert condition == "optimal"
        assert (quadratic_model_cross_terms.solution.x.round(3) == 1.5).all()
        assert (quadratic_model_cross_terms.solution.y.round(3) == 8.5).all()
        assert round(quadratic_model_cross_terms.objective.value or 0, 3) == 77.5
    else:
        with pytest.raises(ValueError):
            quadratic_model_cross_terms.solve(
                solver,
                io_api=io_api,
                explicit_coordinate_names=explicit_coordinate_names,
            )


@pytest.mark.parametrize(
    "solver,io_api,explicit_coordinate_names",
    [
        p
        for p in params
        if (p[0], p[1]) not in [("mindopt", "lp"), ("mindopt", "lp-polars")]
    ],
)
def test_quadratic_model_wo_constraint(
    quadratic_model: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    quadratic_model.constraints.remove("con0")
    if solver in feasible_quadratic_solvers:
        status, condition = quadratic_model.solve(
            solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
        )
        assert condition == "optimal"
        assert (quadratic_model.solution.x.round(3) == 0).all()
        assert round(quadratic_model.objective.value or 0, 3) == 0
    else:
        with pytest.raises(ValueError):
            quadratic_model.solve(
                solver,
                io_api=io_api,
                explicit_coordinate_names=explicit_coordinate_names,
            )


@pytest.mark.parametrize(
    "solver,io_api,explicit_coordinate_names",
    [
        p
        for p in params
        if (p[0], p[1]) not in [("mindopt", "lp"), ("mindopt", "lp-polars")]
    ],
)
def test_quadratic_model_unbounded(
    quadratic_model_unbounded: Model,
    solver: str,
    io_api: str,
    explicit_coordinate_names: bool,
) -> None:
    if solver in feasible_quadratic_solvers:
        status, condition = quadratic_model_unbounded.solve(
            solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
        )
        assert condition in ["unbounded", "unknown", "infeasible_or_unbounded"]
    else:
        with pytest.raises(ValueError):
            quadratic_model_unbounded.solve(
                solver,
                io_api=io_api,
                explicit_coordinate_names=explicit_coordinate_names,
            )


@pytest.mark.parametrize(
    "solver,io_api,explicit_coordinate_names",
    [p for p in params if p[0] not in ["mindopt"]],
)
def test_modified_model(
    modified_model: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    status, condition = modified_model.solve(
        solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
    )

    assert condition == "optimal"
    assert (modified_model.solution.x == 0).all()
    assert (modified_model.solution.y == 10).all()


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_masked_variable_model(
    masked_variable_model: Model,
    solver: str,
    io_api: str,
    explicit_coordinate_names: bool,
) -> None:
    masked_variable_model.solve(
        solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
    )
    x = masked_variable_model.variables.x
    y = masked_variable_model.variables.y
    assert y.solution[-2:].isnull().all()
    assert y.solution[:-2].notnull().all()
    assert x.solution.notnull().all()
    assert (x.solution[-2:] == 10).all()
    # Squeeze in solution getter for expressions with masked variables
    assert_equal(x.add(y).solution, x.solution + y.solution.fillna(0))


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_masked_constraint_model(
    masked_constraint_model: Model,
    solver: str,
    io_api: str,
    explicit_coordinate_names: bool,
) -> None:
    masked_constraint_model.solve(
        solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
    )
    assert (masked_constraint_model.solution.y[:-2] == 10).all()
    assert (masked_constraint_model.solution.y[-2:] == 5).all()


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_basis_and_warmstart(
    tmp_path: Any,
    model: Model,
    solver: str,
    io_api: str,
    explicit_coordinate_names: bool,
) -> None:
    basis_fn = tmp_path / "basis.bas"
    model.solve(
        solver,
        basis_fn=basis_fn,
        io_api=io_api,
        explicit_coordinate_names=explicit_coordinate_names,
    )
    model.solve(
        solver,
        warmstart_fn=basis_fn,
        io_api=io_api,
        explicit_coordinate_names=explicit_coordinate_names,
    )


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_solution_fn_parent_dir_doesnt_exist(
    model: Model,
    solver: str,
    io_api: str,
    explicit_coordinate_names: bool,
    tmp_path: Any,
) -> None:
    solution_fn = tmp_path / "non_existent_dir" / "non_existent_file"
    status, condition = model.solve(
        solver,
        io_api=io_api,
        explicit_coordinate_names=explicit_coordinate_names,
        solution_fn=solution_fn,
    )
    assert status == "ok"


@pytest.mark.parametrize("solver", available_solvers)
def test_non_supported_solver_io(model: Model, solver: str) -> None:
    with pytest.raises(ValueError):
        model.solve(solver, io_api="non_supported")


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_solver_attribute_getter(
    model: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    model.solve(solver)
    if solver != "gurobi":
        with pytest.raises(NotImplementedError):
            model.variables.get_solver_attribute("RC")
    else:
        rc = model.variables.get_solver_attribute("RC")
        assert isinstance(rc, xr.Dataset)
        assert set(rc) == set(model.variables)


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_model_resolve(
    model: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    status, condition = model.solve(
        solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
    )
    assert status == "ok"
    # x = -0.1, y = 1.7
    assert np.isclose(model.objective.value or 0, 3.3)

    # add another constraint after solve
    model.add_constraints(model.variables.y >= 3)

    status, condition = model.solve(
        solver, io_api=io_api, explicit_coordinate_names=explicit_coordinate_names
    )
    assert status == "ok"
    # x = -0.75, y = 3.0
    assert np.isclose(model.objective.value or 0, 5.25)


@pytest.mark.parametrize(
    "solver,io_api,explicit_coordinate_names", [p for p in params if "direct" not in p]
)
def test_solver_classes_from_problem_file(
    model: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    # first test initialization of super class. Should not be possible to initialize
    with pytest.raises(TypeError):
        solvers.Solver()  # type: ignore

    # initialize the solver as object of solver subclass <solver_class>
    solver_class = getattr(solvers, f"{solvers.SolverName(solver).name}")
    solver_ = solver_class()
    # get problem file for testing
    problem_fn = model.get_problem_file(io_api=io_api)
    model.to_file(
        to_path(problem_fn),
        io_api=io_api,
        explicit_coordinate_names=explicit_coordinate_names,
    )
    solution_fn = model.get_solution_file() if solver in ["glpk", "cbc"] else None
    result = solver_.solve_problem(problem_fn=problem_fn, solution_fn=solution_fn)
    assert result.status.status.value == "ok"
    # x = -0.1, y = 1.7
    assert np.isclose(result.solution.objective, 3.3)

    # test for Value error message if no problem file is given
    with pytest.raises(ValueError):
        solver_.solve_problem(solution_fn=solution_fn)

    # test for Value error message if no solution file is passed to glpk or cbc
    if solver in ["glpk", "cbc"]:
        with pytest.raises(ValueError):
            solver_.solve_problem(problem_fn=problem_fn)

    # test for Value error message if invalid problem file format is given
    with pytest.raises(ValueError):
        solver_.solve_problem(problem_fn=solution_fn)


@pytest.mark.parametrize("solver,io_api,explicit_coordinate_names", params)
def test_solver_classes_direct(
    model: Model, solver: str, io_api: str, explicit_coordinate_names: bool
) -> None:
    # initialize the solver as object of solver subclass <solver_class>
    solver_class = getattr(solvers, f"{solvers.SolverName(solver).name}")
    solver_ = solver_class()
    if io_api == "direct":
        result = solver_.solve_problem(model=model)
        assert result.status.status.value == "ok"
        # x = -0.1, y = 1.7
        assert np.isclose(result.solution.objective, 3.3)
        # test for Value error message if direct is tried without giving model
        with pytest.raises(ValueError):
            solver_.model = None
            solver_.solve_problem()
    elif solver not in direct_solvers:
        with pytest.raises(NotImplementedError):
            solver_.solve_problem(model=model)


# def init_model_large():
#     m = Model()
#     time = pd.Index(range(10), name="time")

#     x = m.add_variables(name="x", lower=0, coords=[time])
#     y = m.add_variables(name="y", lower=0, coords=[time])
#     factor = pd.Series(time, index=time)

#     m.add_constraints(3 * x + 7 * y, GREATER_EQUAL, 10 * factor, name="Constraint1")
#     m.add_constraints(5 * x + 2 * y, GREATER_EQUAL, 3 * factor, name="Constraint2")

#     shifted = (1 * x).shift(time=1)
#     lhs = (x - shifted).sel(time=time[1:])
#     m.add_constraints(lhs, "<=", 0.2, "Limited growth")

#     m.add_objective((x + 2 * y).sum())
#     m.solve()
