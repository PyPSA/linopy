import numpy as np
import pandas as pd
import pytest

import linopy
import linopy.constants
from linopy.constraints import Constraint


@pytest.fixture
def model_with_mask() -> linopy.Model:
    m = linopy.Model()
    coords = pd.Index(range(5), name="i")
    mask = np.array([True, False, True, True, False])
    x = m.add_variables(lower=0, coords=[coords], name="x")
    y = m.add_variables(lower=0, coords=[coords], name="y")
    m.add_constraints(x + y >= 1, name="c_xy", mask=mask)
    m.add_constraints(x.sum() + y.sum() <= 100, name="c_sum")
    m.add_objective(x.sum() + 2 * y.sum())
    return m


def test_clabels_parity_with_matrices(model_with_mask: linopy.Model) -> None:
    expected = model_with_mask.matrices.clabels
    actual = model_with_mask.constraints.label_index.clabels
    np.testing.assert_array_equal(actual, expected)


def test_assign_result_does_not_build_matrix(
    monkeypatch: pytest.MonkeyPatch, model_with_mask: linopy.Model
) -> None:
    calls = {"n": 0}
    original = Constraint._matrix_export_data

    def counting(self, label_index):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return original(self, label_index)

    monkeypatch.setattr(Constraint, "_matrix_export_data", counting)

    model_with_mask.solve("highs")

    assert model_with_mask.status == "ok"
    # one build for solver input is fine; the post-solve mapping must not add more
    n_after_solve = calls["n"]
    solver = model_with_mask.solver
    assert solver is not None
    assert solver.status is not None
    result = linopy.constants.Result(
        status=solver.status,
        solution=solver.solution,
        solver_model=solver.solver_model,
        solver_name=solver.solver_name.value,
        report=solver.report,
    )
    model_with_mask.assign_result(result)
    assert calls["n"] == n_after_solve


def test_label_index_invalidated_on_add(model_with_mask: linopy.Model) -> None:
    first = model_with_mask.constraints.label_index.clabels.copy()
    x = model_with_mask.variables["x"]
    model_with_mask.add_constraints(x.sum() >= 0, name="c_extra")
    second = model_with_mask.constraints.label_index.clabels
    assert len(second) == len(first) + 1


def test_label_index_invalidated_on_remove(model_with_mask: linopy.Model) -> None:
    before = len(model_with_mask.constraints.label_index.clabels)
    removed = len(model_with_mask.constraints["c_sum"].active_labels())
    model_with_mask.constraints.remove("c_sum")
    after = len(model_with_mask.constraints.label_index.clabels)
    assert after == before - removed


def test_assign_result_correctness_with_mask(model_with_mask: linopy.Model) -> None:
    model_with_mask.solve("highs")
    assert model_with_mask.status == "ok"
    x_sol = model_with_mask.variables["x"].solution.values
    y_sol = model_with_mask.variables["y"].solution.values
    assert np.isfinite(x_sol).all()
    assert np.isfinite(y_sol).all()
    dual = model_with_mask.constraints["c_xy"].dual.values
    mask = np.array([True, False, True, True, False])
    assert np.isfinite(dual[mask]).all()
    assert np.isnan(dual[~mask]).all()
