from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import linopy
import linopy.constants


@pytest.fixture
def model_with_mask(backend: linopy.Model) -> linopy.Model:
    m = backend
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
    store_cls = type(model_with_mask.constraints["c_xy"].lhs._store)
    original = store_cls.positional_csr

    def counting(self, label_index, active_rows, name):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return original(self, label_index, active_rows, name)

    monkeypatch.setattr(store_cls, "positional_csr", counting)

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


def test_update_with_reordered_lhs_keeps_label_order(backend: linopy.Model) -> None:
    m = backend
    i = pd.RangeIndex(2, name="i")
    j = pd.Index(list("abc"), name="j")
    x = m.add_variables(0, 10, coords=[i, j], name="x")
    y = m.add_variables(0, 10, coords=[j], name="y")
    z = m.add_variables(0, 10, coords=[i], name="z")
    rhs = xr.DataArray(np.arange(6.0).reshape(2, 3), coords=[i, j])
    c = m.add_constraints(x >= rhs, name="c")
    c.update(lhs=y + 0 * z + x)
    assert c.labels.dims == ("i", "j")
    start, end = c.range
    np.testing.assert_array_equal(c.labels.values.ravel(), np.arange(start, end))
    m.add_objective(x.sum() + y.sum() + z.sum())
    m.solve("highs")
    expected = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    np.testing.assert_allclose(c.dual.transpose("i", "j").values, expected, atol=1e-9)


@pytest.mark.parametrize(
    "update",
    [
        lambda c, x: c.update(lhs=x),
        lambda c, x: c.update(rhs=x),
        lambda c, x: c.update(x >= 1),
    ],
    ids=["lhs", "rhs", "constraint"],
)
def test_update_adding_dims_raises(
    backend: linopy.Model,
    update: Callable[[linopy.constraints.Constraint, linopy.Variable], object],
) -> None:
    m = backend
    i = pd.RangeIndex(2, name="i")
    j = pd.Index(list("abc"), name="j")
    x = m.add_variables(0, 10, coords=[j, i], name="x")
    z = m.add_variables(0, 10, coords=[i], name="z")
    c = m.add_constraints(z >= 1, name="c")
    with pytest.raises(
        ValueError, match=r"expression that adds dimensions \['j'\] is unsupported"
    ):
        update(c, x)
