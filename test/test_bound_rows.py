"""Constraints lowered to variable bounds with ``as_bound=True``."""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from linopy import Model, read_netcdf
from linopy.persistent.diff import ModelDiff, RebuildReason
from linopy.solvers import Solver, licensed_solvers

SOLVERS = [s for s in ("highs", "gurobi") if s in licensed_solvers]
IO_APIS = ["direct", "lp", "mps"]
ROW_NAMES = ["up", "up_loose", "neg", "fix", "lo", "dem", "tot"]
FOLDED = ["up", "up_loose", "neg", "fix", "lo"]

pytestmark = pytest.mark.skipif(not SOLVERS, reason="needs highs or gurobi")


def build(as_bound: bool, sense: str = "min", scaled: bool = False) -> Model:
    """
    A small LP whose bound rows are binding at nonzero duals on several sides:
    upper (``up``, with a looser second row ``up_loose`` on the same column),
    a negative coefficient (``neg``), an equality (``fix``) and a lower row
    (``lo``).
    """
    m = Model()
    i = pd.RangeIndex(4, name="i")
    x = m.add_variables(lower=0, coords=[i], name="x")
    y = m.add_variables(lower=-10, upper=10, coords=[i], name="y")
    z = m.add_variables(lower=-5, upper=5, name="z")
    if scaled:
        x.scaling = 3.0
        y.scaling = 0.25
        z.scaling = 7.0
    cap = np.array([1.0, 2.0, 3.0, 4.0])
    m.add_constraints(x <= cap, name="up", as_bound=as_bound)
    m.add_constraints(2 * x <= 2 * (cap + 1), name="up_loose", as_bound=as_bound)
    m.add_constraints(-2 * y >= 2.0, name="neg", as_bound=as_bound)  # y <= -1
    m.add_constraints(4 * z == 6.0, name="fix", as_bound=as_bound)  # z = 1.5
    m.add_constraints(y >= -3.0, name="lo", as_bound=as_bound)
    m.add_constraints(
        x + y + z >= np.array([1, 2, 3, -20.0]),
        name="dem",
        scaling=2.0 if scaled else 1.0,
    )
    m.add_constraints(x.sum() <= 7, name="tot")
    obj = (np.array([-1, -2, 3, 4.0]) * x + np.array([-1, -1, 0.5, 5.0]) * y).sum()
    obj = obj + 2 * z
    m.add_objective(obj if sense == "min" else -obj, sense=sense)
    if scaled:
        m.objective.scaling = 0.5
    return m


def solved(model: Model, solver: str, io_api: str) -> Model:
    status, _ = model.solve(solver, io_api=io_api)
    assert status == "ok"
    return model


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("io_api", IO_APIS)
@pytest.mark.parametrize("sense", ["min", "max"])
@pytest.mark.parametrize("scaled", [False, True])
def test_bound_rows_match_ordinary_rows(
    solver: str, io_api: str, sense: str, scaled: bool
) -> None:
    ordinary = solved(build(False, sense, scaled), solver, io_api)
    folded = solved(build(True, sense, scaled), solver, io_api)

    assert np.isclose(folded.objective.value, ordinary.objective.value)
    for name in ["x", "y", "z"]:
        np.testing.assert_allclose(
            folded.solution[name], ordinary.solution[name], atol=1e-7
        )
    for name in ROW_NAMES:
        np.testing.assert_allclose(
            folded.constraints[name].dual.values,
            ordinary.constraints[name].dual.values,
            atol=1e-7,
            err_msg=name,
        )


@pytest.mark.parametrize("solver", SOLVERS)
def test_bound_row_duals_are_informative(solver: str) -> None:
    m = solved(build(True), solver, "direct")
    dual = {name: m.constraints[name].dual.values for name in FOLDED}
    # min problem: binding <= rows have negative duals, >= rows positive
    assert (dual["up"][:2] < 0).all()
    assert (dual["up_loose"] == 0).all()  # never binding next to ``up``
    assert (dual["neg"][:2] > 0).all()  # -2 y >= 2 binds
    assert dual["fix"] != 0
    assert dual["lo"][3] > 0  # y[3] is pushed down to its lower row


@pytest.mark.parametrize("solver", SOLVERS)
def test_bound_rows_milp(solver: str) -> None:
    def milp(as_bound: bool) -> Model:
        m = build(as_bound)
        n = m.add_variables(lower=0, upper=10, integer=True, name="n")
        m.add_constraints(3 * n <= 7.5, name="nmax", as_bound=as_bound)
        m.add_constraints(m.variables["x"].sum() + n >= 5.5, name="ncov")
        m.objective = m.objective.expression - 0.1 * n
        return m

    ordinary = solved(milp(False), solver, "direct")
    folded = solved(milp(True), solver, "direct")
    assert np.isclose(folded.objective.value, ordinary.objective.value)
    assert folded.solution["n"].item() == ordinary.solution["n"].item() == 2


@pytest.mark.parametrize("solver", SOLVERS)
def test_bound_rows_quadratic_objective(solver: str) -> None:
    def qp(as_bound: bool) -> Model:
        m = build(as_bound)
        x = m.variables["x"]
        m.objective = m.objective.expression + (x * x).sum() + 0.5 * x.sum()
        return m

    ordinary = solved(qp(False), solver, "direct")
    folded = solved(qp(True), solver, "direct")
    assert np.isclose(folded.objective.value, ordinary.objective.value)
    for name in ROW_NAMES:
        np.testing.assert_allclose(
            folded.constraints[name].dual.values,
            ordinary.constraints[name].dual.values,
            atol=1e-5,
            err_msg=name,
        )


def test_matrices_fold_rows_into_bounds() -> None:
    ordinary, folded = build(False).matrices, build(True).matrices
    assert len(ordinary.clabels) == 4 + 4 + 4 + 1 + 4 + 4 + 1
    assert len(folded.clabels) == folded.A.shape[0] == 4 + 1
    assert folded.A.nnz == ordinary.A.nnz - (4 + 4 + 4 + 1 + 4)
    np.testing.assert_array_equal(folded.ub[:4], [1, 2, 3, 4])
    np.testing.assert_array_equal(folded.ub[4:8], -1)
    np.testing.assert_array_equal(folded.lb[4:8], -3)
    assert folded.lb[8] == folded.ub[8] == 1.5

    rows = folded.bound_rows
    assert rows is not None
    m = build(True)
    up = m.constraints["up"].labels.values
    np.testing.assert_array_equal(rows.upper_row[:4], up)
    assert rows.lower_row[8] == rows.upper_row[8] == m.constraints["fix"].labels


def test_matrices_scaled_bounds() -> None:
    M = build(True, scaled=True).matrices
    np.testing.assert_allclose(M.ub[:4], 3.0 * np.array([1, 2, 3, 4]))
    assert M.lb[8] == M.ub[8] == 7.0 * 1.5


def test_variable_bound_tighter_than_row() -> None:
    m = Model()
    x = m.add_variables(lower=0, upper=1, name="x")
    m.add_constraints(x <= 2, name="loose", as_bound=True)
    m.add_constraints(x >= -1, name="loose_lower", as_bound=True)
    m.add_objective(-x)
    M = m.matrices
    assert (M.lb[0], M.ub[0]) == (0, 1)
    assert M.bound_rows is not None
    assert M.bound_rows.upper_row[0] == M.bound_rows.lower_row[0] == -1
    m.solve(SOLVERS[0])
    assert m.constraints["loose"].dual.item() == 0
    assert m.constraints["loose_lower"].dual.item() == 0


def test_masked_rows_are_skipped() -> None:
    m = Model()
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(3, name="i")], name="x")
    mask = np.array([True, False, True])
    m.add_constraints(x <= 2, name="up", mask=mask, as_bound=True)
    m.add_objective(-x.sum())
    np.testing.assert_array_equal(m.matrices.ub, [2, np.inf, 2])


def test_take_keeps_folded_bounds() -> None:
    M = build(True).matrices
    taken = M.take(np.array([0]), np.array([1, 2, 8]))
    np.testing.assert_array_equal(taken.ub, [2, 3, 1.5])
    assert taken.bound_rows is not None
    assert len(taken.bound_rows.cols) == 2 + 2 + 1
    assert (taken.bound_rows.cols < 3).all()


def test_multi_term_rows_raise() -> None:
    m = Model()
    x = m.add_variables(coords=[pd.RangeIndex(2, name="i")], name="x")
    y = m.add_variables(name="y")
    with pytest.raises(ValueError, match="exactly one variable term"):
        m.add_constraints(x + y <= 1, name="c", as_bound=True)
    assert "c" not in m.constraints


def test_modified_to_multi_term_raises_at_lowering() -> None:
    m = Model()
    x = m.add_variables(name="x")
    y = m.add_variables(name="y")
    c = m.add_constraints(x <= 1, name="c", as_bound=True)
    c.update(lhs=x + y)
    with pytest.raises(ValueError, match="exactly one variable term"):
        m.matrices


def test_rhs_modified_later_is_folded() -> None:
    m = Model()
    x = m.add_variables(name="x")
    c = m.add_constraints(x <= 1, name="c", as_bound=True)
    c.update(rhs=3)
    assert m.matrices.ub[0] == 3


def test_removed_constraint_no_longer_folds() -> None:
    m = build(True)
    m.remove_constraints("up")
    M = m.matrices
    np.testing.assert_array_equal(M.ub[:4], [2, 3, 4, 5])


def test_penalty_and_soften_rejected() -> None:
    m = Model()
    x = m.add_variables(name="x")
    m.add_objective(x)
    with pytest.raises(ValueError, match="penalty"):
        m.add_constraints(x <= 1, name="c", as_bound=True, penalty=1)
    c = m.add_constraints(x <= 1, name="d", as_bound=True)
    with pytest.raises(ValueError, match="cannot be softened"):
        c.soften(penalty=1)


def test_semi_continuous_rejected() -> None:
    m = Model()
    x = m.add_variables(lower=1, upper=10, semi_continuous=True, name="x")
    m.add_constraints(x <= 5, name="c", as_bound=True)
    with pytest.raises(ValueError, match="semi-continuous"):
        m.matrices


def test_lp_file_writes_bounds_not_rows(tmp_path: Path) -> None:
    m = build(True)
    fn = m.to_file(tmp_path / "model.lp")
    text = fn.read_text()
    labels = m.constraints["up"].labels.values
    assert f"c{labels[0]}:" not in text
    assert f"c{int(m.constraints['dem'].labels.values[0])}:" in text
    assert "x0 <= +1.0\n" in text


def test_lp_file_binary_bound(tmp_path: Path) -> None:
    m = Model()
    b = m.add_variables(binary=True, name="b")
    m.add_constraints(b <= 0, name="off", as_bound=True)
    m.add_objective(-b)
    text = m.to_file(tmp_path / "model.lp").read_text()
    assert "x0 <= +0.0\n" in text
    m.solve(SOLVERS[0], io_api="lp")
    assert m.solution["b"].item() == 0


def test_dual_matrix_accessor_aligned_with_clabels() -> None:
    m = solved(build(True), SOLVERS[0], "direct")
    assert len(m.matrices.dual) == len(m.matrices.clabels)


def test_unverified_solver_gets_nan_duals() -> None:
    m = build(True)
    solver = Solver.from_name(SOLVERS[0], m, io_api="direct")
    result = solver.solve()
    with pytest.warns(UserWarning, match="only recovered"):
        m.assign_result(replace(result, solver_name="cbc"))
    assert np.isnan(m.constraints["up"].dual.values).all()
    assert not np.isnan(m.constraints["dem"].dual.values).any()


def test_persistent_update_rebuilds() -> None:
    m = build(True)
    assert ModelDiff.from_models(m, build(True)) is RebuildReason.BOUND_ROWS
    solver = Solver.from_name(SOLVERS[0], m, io_api="direct", track_updates=True)
    solver.solve(assign=True)
    first = m.constraints["up"].dual.values.copy()
    m.constraints["up"].update(rhs=m.constraints["up"].rhs * 0.9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solver.solve(m, assign=True)
    assert solver._last_rebuild_reason is RebuildReason.BOUND_ROWS
    np.testing.assert_allclose(m.matrices.ub[:4], [0.9, 1.8, 2.7, 3.6])
    np.testing.assert_allclose(m.solution["x"].values[:2], [0.9, 1.8])
    assert first.shape == m.constraints["up"].dual.shape


def test_dualize_treats_bound_rows_as_rows() -> None:
    ordinary = build(False).dualize()
    folded = build(True).dualize()
    assert set(folded.variables) == set(ordinary.variables)
    assert len(folded.matrices.clabels) == len(ordinary.matrices.clabels)
    folded.solve(SOLVERS[0])
    ordinary.solve(SOLVERS[0])
    assert np.isclose(folded.objective.value, ordinary.objective.value)


def test_netcdf_roundtrip_keeps_flag(tmp_path: Path) -> None:
    m = build(True)
    m.to_netcdf(tmp_path / "m.nc")
    back = read_netcdf(tmp_path / "m.nc")
    assert back.constraints["up"].as_bound
    assert not back.constraints["dem"].as_bound
    assert len(back.matrices.clabels) == len(m.matrices.clabels)
    assert list(back.constraints.as_bound) == FOLDED


def test_default_unchanged() -> None:
    m = build(False)
    assert not any(c.as_bound for c in m.constraints.data.values())
    assert m.matrices.bound_rows is None
    assert len(m.constraints.as_bound) == 0
