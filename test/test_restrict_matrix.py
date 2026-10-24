"""Matrix-level windows agree with the container-level restriction."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from linopy import Model
from linopy.persistent.diff import ModelDiff
from linopy.restrict import MatrixWindow, RestrictionError, WindowMatrices

TIME = pd.RangeIndex(8, name="time")


def storage_model() -> Model:
    """Two coupled constraints: a shifted balance and an elementwise cap."""
    m = Model()
    soc = m.add_variables(lower=0, upper=10, coords=[TIME], name="soc")
    charge = m.add_variables(lower=0, upper=5, coords=[TIME], name="charge")
    m.add_constraints(soc - 0.9 * soc.shift(time=1) - charge == 0, name="balance")
    m.add_constraints(charge <= 4, name="cap")
    m.add_objective((charge + soc).sum())
    return m


def initial(soc_value: float = 3.0) -> pd.Series:
    return pd.Series(np.full(len(TIME), soc_value), index=TIME)


def assert_same_window(window: WindowMatrices, reference: WindowMatrices) -> None:
    assert window.A.shape == reference.A.shape
    assert np.array_equal(window.A.indptr, reference.A.indptr)
    assert np.array_equal(window.A.indices, reference.A.indices)
    assert np.allclose(window.A.data, reference.A.data)
    assert np.allclose(window.b, reference.b)
    assert np.allclose(window.c, reference.c)
    assert np.allclose(window.lb, reference.lb)
    assert np.allclose(window.ub, reference.ub)
    assert np.array_equal(window.sense, reference.sense)


@pytest.mark.parametrize("start", [0, 1, 3])
def test_select_matches_restrict(start: int) -> None:
    m = storage_model()
    window = TIME[start : start + 3]

    matrix = MatrixWindow(m, "time").select(window, fix={"soc": initial()})
    reference = m.restrict(time=window, fix={"soc": initial()}).matrices

    assert_same_window(matrix, reference)


def test_select_without_fix_matches_restrict() -> None:
    """A model with no cross-window coupling needs no boundary values."""
    m = Model()
    x = m.add_variables(lower=0, coords=[TIME], name="x")
    m.add_constraints(x <= 4, name="cap")
    m.add_objective(x.sum())

    window = TIME[2:5]
    matrix = MatrixWindow(m, "time").select(window)
    reference = m.restrict(time=window).matrices

    assert_same_window(matrix, reference)


def test_window_keeps_parent_labels() -> None:
    m = storage_model()
    window = TIME[1:4]
    selected = MatrixWindow(m, "time").select(window, fix={"soc": initial()})

    assert selected.vlabels.size == selected.A.shape[1]
    assert selected.clabels.size == selected.A.shape[0]
    assert np.isin(selected.vlabels, m.variables.label_index.vlabels).all()


def test_infeasible_folded_row_raises() -> None:
    """A pinned row that folds to a violated constant is rejected."""
    m = Model()
    x = m.add_variables(lower=0, coords=[TIME], name="x")
    m.add_constraints(x == 1, name="eq")
    # an absolute-index row: it folds away entirely outside its own timestep
    m.add_constraints(x.isel(time=0) == 2, name="pin")
    m.add_objective(x.sum())

    window = TIME[1:3]
    fix = {"x": pd.Series(np.zeros(len(TIME)), index=TIME)}
    with pytest.raises(RestrictionError, match="unsatisfiable"):
        MatrixWindow(m, "time").select(window, fix=fix)


def test_binary_integrality_is_preserved() -> None:
    """A discrete column stays discrete through the window."""
    m = Model()
    x = m.add_variables(binary=True, coords=[TIME], name="x")
    m.add_constraints(x >= 0.5, name="half")
    m.add_objective(x.sum(), sense="min")

    window = m.restrict(time=TIME[:3])
    window.solve(solver_name="highs", output_flag=False)

    assert set(np.unique(window.solution["x"].values)) <= {0.0, 1.0}
    assert window.solution["x"].values.min() == 1.0


def test_variable_scaling_is_undone() -> None:
    """A scaled column reports the model's units, not the solver's."""
    m = Model()
    x = m.add_variables(lower=0, upper=100, coords=[TIME], scaling=10, name="x")
    m.add_constraints(x >= 50, name="lb")
    m.add_objective(x.sum(), sense="min")

    window = m.restrict(time=TIME[:3])
    window.solve(solver_name="highs", output_flag=False)

    assert np.allclose(window.solution["x"].values, 50.0)


def test_constraint_scaling_is_undone() -> None:
    """A scaled row reports the model's dual units."""
    m = Model()
    x = m.add_variables(lower=0, coords=[TIME], name="x")
    m.add_constraints(x >= 5, name="lb", scaling=2)
    m.add_objective(x.sum(), sense="min")

    window = m.restrict(time=TIME[:3])
    window.solve(solver_name="highs", output_flag=False)

    assert np.allclose(window.dual["lb"].values, 1.0)


def test_subset_carrier_raises() -> None:
    """Every carrier of the selected dimension must share its index."""
    m = Model()
    x = m.add_variables(coords=[TIME], name="x")
    subset = m.add_variables(coords=[TIME[::2]], name="subset_x")
    m.add_constraints(x <= 1, name="cap")
    m.add_constraints(subset <= 1, name="subset_cap")
    m.add_objective(m.variables["x"].sum() + subset.sum())

    with pytest.raises(RestrictionError, match="not carried by every container"):
        MatrixWindow(m, "time").select(TIME[1:3])


def linear_model() -> Model:
    m = Model()
    x = m.add_variables(lower=0, coords=[TIME], name="x")
    m.add_constraints(x <= 5, name="cap")
    m.add_objective(x.sum(), sense="min")
    return m


def test_restrict_rejects_multiple_dimensions() -> None:
    m = linear_model()
    m.add_variables(coords=[pd.RangeIndex(2, name="other")], name="y")

    with pytest.raises(NotImplementedError, match="one selected dimension"):
        m.restrict(time=TIME[:2], other=[0])


def test_restrict_rejects_unknown_dimension() -> None:
    with pytest.raises(RestrictionError, match="does not exist"):
        linear_model().restrict(period=[0])


def test_window_rejects_quadratic_objective() -> None:
    m = Model()
    x = m.add_variables(lower=0, coords=[TIME], name="x")
    m.add_constraints(x <= 5, name="cap")
    m.add_objective((x * x).sum())

    with pytest.raises(NotImplementedError, match="quadratic"):
        MatrixWindow(m, "time")


def test_window_rejects_indicator_constraints() -> None:
    m = Model()
    x = m.add_variables(lower=0, coords=[TIME], name="x")
    flag = m.add_variables(binary=True, coords=[TIME], name="flag")
    m.add_indicator_constraints(flag, 1, x, "<=", 1, name="ind")
    m.add_objective(x.sum())

    with pytest.raises(NotImplementedError, match="indicator"):
        MatrixWindow(m, "time")


def test_window_requires_a_constraint() -> None:
    m = Model()
    m.add_variables(lower=0, coords=[TIME], name="x")
    m.add_objective(m.variables["x"].sum())

    with pytest.raises(NotImplementedError, match="at least one constraint"):
        MatrixWindow(m, "time")


def test_fix_must_name_a_variable() -> None:
    with pytest.raises(RestrictionError, match="does not name a variable"):
        MatrixWindow(linear_model(), "time").select(TIME[:2], fix={"nope": 1.0})


def test_fix_must_match_the_variable_shape() -> None:
    with pytest.raises(RestrictionError, match="do not match its dimensions"):
        MatrixWindow(linear_model(), "time").select(TIME[:2], fix={"x": [1.0, 2.0]})


def test_advance_must_name_every_selected_dimension() -> None:
    window = linear_model().restrict(time=TIME[:2])
    with pytest.raises(RestrictionError, match="missing"):
        window.advance(other=[0])


def test_advance_without_selection_keeps_the_window() -> None:
    window = linear_model().restrict(time=TIME[:2])
    before = window.matrices.b.copy()

    window.advance()

    assert np.array_equal(window.matrices.b, before)


def test_solve_updates_options_on_later_calls() -> None:
    window = linear_model().restrict(time=TIME[:2])
    window.solve(solver_name="highs", output_flag=False)
    window.solve(output_flag=False)
    window.solve()

    assert window.objective.value == pytest.approx(0.0)


def test_solution_before_solve_raises() -> None:
    window = linear_model().restrict(time=TIME[:2])
    with pytest.raises(AttributeError, match="has not been solved"):
        window.solution


def test_dual_size_mismatch_raises() -> None:
    window = linear_model().restrict(time=TIME[:2])
    window.solve(solver_name="highs", output_flag=False)
    window._dual = np.empty(0)

    with pytest.raises(AttributeError, match="no dual values"):
        window.dual


def test_getitem_before_solve_returns_nan() -> None:
    window = linear_model().restrict(time=TIME[:2])
    values = window["x"]

    assert values.sizes["time"] == 2
    assert np.isnan(values.values).all()


def test_getitem_after_solve_returns_values() -> None:
    window = linear_model().restrict(time=TIME[:2])
    window.solve(solver_name="highs", output_flag=False)

    assert np.allclose(window["x"].values, 0.0)


def test_getitem_unknown_variable_raises() -> None:
    window = linear_model().restrict(time=TIME[:2])
    with pytest.raises(KeyError):
        window["nope"]


def test_model_view_exposes_the_builder_surface() -> None:
    window = linear_model().restrict(time=TIME[:2])
    view = window.as_model_view()

    view.constraints.sanitize_missings()
    view.constraints.sanitize_zeros()
    view.constraints.sanitize_infinities()

    assert view.type == "LP"
    assert view.sense == "min"
    assert view.is_quadratic is False
    assert view.indicator_constraints == ()
    assert view.variables.sos == {}
    assert view.continuous
    assert not view.binaries and not view.integers and not view.semi_continuous
    assert view.variables.label_index.vlabels.size == len(window.matrices.vlabels)
    assert view.constraints.label_index.clabels.size == len(window.matrices.clabels)
    assert view._xCounter == len(window.matrices.vlabels)
    assert view._cCounter == len(window.matrices.clabels)
    assert view.nvars == len(window.matrices.vlabels)
    assert view.ncons == len(window.matrices.clabels)
    assert len(view.constraints) == len(window.matrices.con_blocks)


def test_model_view_type_is_milp_for_mixed_variables() -> None:
    m = Model()
    x = m.add_variables(lower=0, coords=[TIME], name="x")
    flag = m.add_variables(binary=True, coords=[TIME], name="flag")
    m.add_constraints(x + flag <= 5, name="cap")
    m.add_objective(x.sum() + flag.sum())

    assert m.restrict(time=TIME[:2]).as_model_view().type == "MILP"


def test_model_view_type_is_ilp_for_discrete_only() -> None:
    m = Model()
    flag = m.add_variables(binary=True, coords=[TIME], name="flag")
    m.add_constraints(flag <= 1, name="cap")
    m.add_objective(flag.sum())

    assert m.restrict(time=TIME[:2]).as_model_view().type == "ILP"


def _window_matrices() -> WindowMatrices:
    return linear_model().restrict(time=TIME[:2]).matrices


def test_from_windows_reports_coefficient_changes() -> None:
    previous = _window_matrices()
    matrix = previous.A.copy()
    matrix.data = matrix.data + 1.0

    diff = ModelDiff.from_windows(previous, replace(previous, A=matrix))

    assert diff.n_coef_updates > 0


def test_from_windows_reports_variable_type_changes() -> None:
    previous = _window_matrices()
    vtypes = previous.vtypes.copy()
    vtypes[0] = "B"

    diff = ModelDiff.from_windows(previous, replace(previous, vtypes=vtypes))

    assert diff.var_type_positions.tolist() == [0]


def test_from_windows_reports_constraint_sign_changes() -> None:
    previous = _window_matrices()
    sense = previous.sense.copy()
    sense[0] = ">="

    diff = ModelDiff.from_windows(previous, replace(previous, sense=sense))

    assert diff.con_sign_indices.tolist() == [0]


def test_from_windows_reports_objective_sense_changes() -> None:
    previous = _window_matrices()

    diff = ModelDiff.from_windows(previous, replace(previous, objective_sense="max"))

    assert diff.obj_sense == "max"


def test_from_windows_reports_bounds_and_cost_changes() -> None:
    previous = _window_matrices()

    diff = ModelDiff.from_windows(
        previous, replace(previous, lb=previous.lb + 1.0, c=previous.c + 1.0)
    )

    assert diff.var_bounds_indices.size == previous.lb.size
    assert diff.obj_c_indices is not None


def test_select_without_fix_rejects_dangling_terms() -> None:
    """A window that cuts a coupling must be given the boundary values."""
    with pytest.raises(RestrictionError, match="dangling terms"):
        MatrixWindow(storage_model(), "time").select(TIME[1:3])


def test_nan_fix_counts_as_missing() -> None:
    fix = {"soc": pd.Series(np.full(len(TIME), np.nan), index=TIME)}
    with pytest.raises(RestrictionError, match="dangling terms"):
        MatrixWindow(storage_model(), "time").select(TIME[1:3], fix=fix)


WINDOW_SOLVERS = ("highs", "gurobi", "cplex", "scip", "xpress", "mosek")


@pytest.mark.parametrize("solver_name", WINDOW_SOLVERS)
def test_window_solve_across_solvers(solver_name: str) -> None:
    """The window recipe runs on every direct solver available in the environment."""
    from linopy import licensed_solvers

    if solver_name not in licensed_solvers:
        pytest.skip(f"{solver_name} is not available with a license")

    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[TIME], name="x")
    y = m.add_variables(lower=0, coords=[TIME], name="y")
    m.add_constraints(x - 0.9 * x.shift(time=1) - y == 0, name="balance")
    m.add_objective(y.sum())
    fix = {"x": pd.Series(np.full(len(TIME), 1.0), index=TIME)}

    window = m.restrict(time=TIME[:3], fix=fix)
    try:
        window.solve(solver_name=solver_name)
    except NotImplementedError as exc:  # a file-only backend
        pytest.skip(str(exc))
    assert window.objective.value == pytest.approx(0.0)

    window.advance(time=TIME[3:6], fix=fix)
    window.solve(solver_name=solver_name)
    assert window.solution["x"].sizes["time"] == 3


def test_solve_refuses_a_rebuild_when_asked() -> None:
    from linopy.persistent.errors import RebuildRequiredError

    m = linear_model()
    fix = {"x": pd.Series(np.zeros(len(TIME)), index=TIME)}
    window = m.restrict(time=TIME[:3], fix=fix)
    window.solve(solver_name="highs", disallow_rebuild=True)

    window.advance(time=TIME[3:5], fix=fix)  # a different width changes the shape
    with pytest.raises(RebuildRequiredError):
        window.solve(solver_name="highs", disallow_rebuild=True)


def test_solve_rebuilds_a_changed_shape() -> None:
    m = linear_model()
    fix = {"x": pd.Series(np.zeros(len(TIME)), index=TIME)}
    window = m.restrict(time=TIME[:3], fix=fix)
    window.solve(solver_name="highs")

    window.advance(time=TIME[3:5], fix=fix)
    window.solve(solver_name="highs")

    assert window.solution["x"].sizes["time"] == 2


def test_window_rejects_sos_constraints() -> None:
    m = linear_model()
    m.add_sos_constraints(m.variables["x"], 1, "time")

    with pytest.raises(NotImplementedError, match="SOS"):
        MatrixWindow(m, "time")


def test_window_solves_piecewise_formulations() -> None:
    """Piecewise expands into ordinary variables and constraints."""
    from linopy.piecewise import add_piecewise_formulation

    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[TIME], name="x")
    y = m.add_variables(lower=0, coords=[TIME], name="y")
    m.add_constraints(x <= 5, name="cap")
    m.add_objective(y.sum())
    add_piecewise_formulation(m, (x, [0, 5, 10]), (y, [0, 4, 9]))

    window = m.restrict(time=TIME[:2])
    window.solve(solver_name="highs", output_flag=False)

    assert np.allclose(window.solution["y"].values, 0.0)


def test_window_rejects_horizon_wide_sos_rows() -> None:
    """Reformulating SOS adds a horizon-wide cardinality row, which cannot fold."""
    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[TIME], name="x")
    m.add_constraints(x <= 5, name="cap")
    m.add_objective(x.sum())
    m.add_sos_constraints(x, 1, "time")
    m.apply_sos_reformulation()

    with pytest.raises(RestrictionError, match="dangling terms"):
        m.restrict(time=TIME[:2])
