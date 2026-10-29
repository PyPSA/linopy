"""Matrix-level windows agree with the container-level restriction."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import LinearExpression, Model, merge
from linopy.matrices import Matrices
from linopy.persistent.diff import ModelDiff
from linopy.persistent.errors import RebuildRequiredError
from linopy.restrict import MatrixWindow, RestrictionError

TIME = pd.RangeIndex(8, name="time")
BUS = pd.Index(["a", "b"], name="bus")
BOUNDARY = xr.DataArray(np.where(TIME == 0, 7.0, 0.0), coords=[TIME])


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


def assert_same_window(window: Matrices, reference: Matrices) -> None:
    assert window.A is not None and reference.A is not None
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

    matrix = MatrixWindow(m).select({"time": window}, fix={"soc": initial()})[0]
    reference = m.restrict(time=window, fix={"soc": initial()}).matrices

    assert_same_window(matrix, reference)


def test_select_without_fix_matches_restrict() -> None:
    """A model with no cross-window coupling needs no boundary values."""
    m = Model()
    x = m.add_variables(lower=0, coords=[TIME], name="x")
    m.add_constraints(x <= 4, name="cap")
    m.add_objective(x.sum())

    window = TIME[2:5]
    matrix = MatrixWindow(m).select({"time": window})[0]
    reference = m.restrict(time=window).matrices

    assert_same_window(matrix, reference)


def test_window_keeps_parent_labels() -> None:
    m = storage_model()
    window = TIME[1:4]
    selected, _, _ = MatrixWindow(m).select({"time": window}, fix={"soc": initial()})

    assert selected.A is not None
    assert selected.A.shape == (selected.clabels.size, selected.vlabels.size)
    assert np.isin(selected.vlabels, m.variables.label_index.vlabels).all()


@pytest.mark.parametrize(
    ("sign", "rhs", "satisfied"),
    [("==", 2, False), (">=", 2, False), ("<=", -2, False), ("<=", 2, True)],
)
def test_folded_row_is_checked_against_its_sign(
    sign: str, rhs: float, satisfied: bool
) -> None:
    """A row that loses every term is dropped only when its sign holds."""
    m = Model()
    x = m.add_variables(lower=0, coords=[TIME], name="x")
    m.add_constraints(x == 1, name="eq")
    m.add_constraints(x.isel(time=0), sign, rhs, name="pin")
    m.add_objective(x.sum())

    window = MatrixWindow(m)
    fix = {"x": pd.Series(np.zeros(len(TIME)), index=TIME)}
    if satisfied:
        assert window.select({"time": TIME[1:3]}, fix=fix)[0].clabels.size == 2
        return
    with pytest.raises(RestrictionError, match="unsatisfiable"):
        window.select({"time": TIME[1:3]}, fix=fix)


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

    with pytest.raises(RestrictionError, match="does not carry the selected labels"):
        MatrixWindow(m).select({"time": TIME[1:3]})


def linear_model() -> Model:
    m = Model()
    x = m.add_variables(lower=0, coords=[TIME], name="x")
    m.add_constraints(x <= 5, name="cap")
    m.add_objective(x.sum(), sense="min")
    return m


def test_restrict_rejects_unknown_dimension() -> None:
    with pytest.raises(RestrictionError, match="does not exist"):
        linear_model().restrict(period=[0])


def test_window_rejects_quadratic_objective() -> None:
    m = Model()
    x = m.add_variables(lower=0, coords=[TIME], name="x")
    m.add_constraints(x <= 5, name="cap")
    m.add_objective((x * x).sum())

    with pytest.raises(NotImplementedError, match="quadratic"):
        MatrixWindow(m)


def test_window_rejects_indicator_constraints() -> None:
    m = Model()
    x = m.add_variables(lower=0, coords=[TIME], name="x")
    flag = m.add_variables(binary=True, coords=[TIME], name="flag")
    m.add_indicator_constraints(flag, 1, x, "<=", 1, name="ind")
    m.add_objective(x.sum())

    with pytest.raises(NotImplementedError, match="indicator"):
        MatrixWindow(m)


def test_window_requires_a_constraint() -> None:
    m = Model()
    m.add_variables(lower=0, coords=[TIME], name="x")
    m.add_objective(m.variables["x"].sum())

    with pytest.raises(NotImplementedError, match="at least one constraint"):
        MatrixWindow(m)


def test_fix_must_name_a_variable() -> None:
    with pytest.raises(RestrictionError, match="does not name a variable"):
        MatrixWindow(linear_model()).select({"time": TIME[:2]}, fix={"nope": 1.0})


@pytest.mark.parametrize(
    "given",
    [[1.0, 2.0], BOUNDARY.isel(time=slice(0, 4)), BOUNDARY.expand_dims(other=[0])],
    ids=["wrong-length", "missing-coords", "extra-dim"],
)
def test_fix_must_match_the_variable_coordinates(given: object) -> None:
    with pytest.raises(RestrictionError, match="do not match its coordinates"):
        MatrixWindow(linear_model()).select({"time": TIME[:2]}, fix={"x": given})


def ramp_model(scaling: float = 1.0) -> Model:
    """A two-dimensional ramp: every step must not fall below the previous one."""
    m = Model()
    x = m.add_variables(0, 100, coords=[TIME, BUS], scaling=scaling, name="x")
    m.add_constraints(x - x.shift(time=1) >= 0, name="ramp")
    m.add_objective(x.sum())
    return m


def solve_window(m: Model, fix: object, **selection: object) -> xr.DataArray:
    window = m.restrict(fix={"x": fix}, **selection)
    window.solve(solver_name="highs", output_flag=False)
    return window.solution["x"]


@pytest.mark.parametrize("scaling", [1.0, 10.0])
@pytest.mark.parametrize(
    "fix",
    [
        BOUNDARY,
        BOUNDARY.isel(time=slice(None, None, -1)),
        BOUNDARY.broadcast_like(xr.DataArray(BUS, coords=[BUS])).transpose(),
        BOUNDARY.to_series(),
        xr.DataArray(BOUNDARY.values, dims=["time"]),
        7.0,
    ],
    ids=["broadcast", "reversed", "transposed", "series", "positional", "scalar"],
)
def test_fix_is_aligned_by_coordinates(scaling: float, fix: object) -> None:
    """Fixed values align by label, broadcast over missing dims, in model units."""
    solution = solve_window(ramp_model(scaling), fix, time=TIME[1:3])

    assert solution.dims == ("time", "bus")
    assert np.allclose(solution.values, 7.0)


@pytest.mark.parametrize(
    ("selection", "sizes"),
    [
        ({"time": slice(1, 2)}, {"time": 2, "bus": 2}),
        ({"time": [1, 2]}, {"time": 2, "bus": 2}),
        ({"time": slice(1, 2), "bus": ["a"]}, {"time": 2, "bus": 1}),
        ({"time": 1, "bus": "b"}, {}),
    ],
    ids=["slice", "list", "multi-dim", "scalars"],
)
def test_selection_follows_sel_semantics(
    selection: dict[str, object], sizes: dict[str, int]
) -> None:
    solution = solve_window(ramp_model(), 7.0, **selection)

    assert dict(solution.sizes) == sizes
    assert np.allclose(solution.values, 7.0)
    assert solution.coords.to_dataset().equals(
        ramp_model().variables["x"].labels.sel(selection).coords.to_dataset()
    )


def test_slice_selection_matches_label_selection() -> None:
    window = MatrixWindow(storage_model())
    fix = {"soc": initial()}

    assert_same_window(
        window.select({"time": slice(1, 3)}, fix=fix)[0],
        window.select({"time": TIME[1:4]}, fix=fix)[0],
    )


def sparse_coupling_model() -> Model:
    """A coefficient that is zero at one step, so windows differ in sparsity."""
    m = Model()
    x = m.add_variables(0, 100, coords=[TIME[:4]], name="x")
    y = m.add_variables(0, 100, coords=[TIME[:4]], name="y")
    k = xr.DataArray([1.0, 1.0, 0.0, 1.0], coords=[TIME[:4]])
    m.add_constraints(x + k * y >= 3, name="c")
    m.add_objective((x + 2 * y).sum())
    return m


def test_advance_rebuilds_on_a_changed_sparsity_pattern() -> None:
    window = sparse_coupling_model().restrict(time=TIME[:2])
    window.solve(solver_name="highs", output_flag=False)
    window.advance(time=TIME[2:4])

    with pytest.raises(RebuildRequiredError):
        window.solve(disallow_rebuild=True)
    window.solve()

    assert np.allclose(window.solution["x"].values, 3.0)
    assert np.allclose(window.solution["y"].values, 0.0)


def test_from_windows_rejects_a_changed_sparsity_pattern() -> None:
    m = sparse_coupling_model()
    window = MatrixWindow(m)
    first = window.select({"time": TIME[:2]})[0]
    second = window.select({"time": TIME[2:4]})[0]

    assert first.A is not None and second.A is not None
    assert first.A.shape == second.A.shape
    with pytest.raises(ValueError, match="sparsity pattern"):
        ModelDiff.from_windows(first, second)


@pytest.mark.parametrize(
    "selection",
    [{"other": [0]}, {"time": TIME[2:4], "bus": ["a"]}],
    ids=["missing", "extra"],
)
def test_advance_must_name_exactly_the_selected_dimensions(
    selection: dict[str, object],
) -> None:
    window = ramp_model().restrict(fix={"x": 0.0}, time=TIME[:2])
    with pytest.raises(RestrictionError, match="name exactly the dimensions"):
        window.advance(**selection)


@pytest.mark.parametrize(
    ("selection", "match"),
    [
        ({"time": slice(20, 30)}, "is empty"),
        ({"time": [1, 1]}, "repeats labels"),
        ({"time": [1, 2], "bus": ["a", "a"]}, "repeats labels"),
    ],
    ids=["empty", "repeated", "repeated-multi-dim"],
)
def test_select_rejects_degenerate_selections(
    selection: dict[str, object], match: str
) -> None:
    with pytest.raises(RestrictionError, match=match):
        ramp_model().restrict(fix={"x": 0.0}, **selection)


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


@pytest.mark.parametrize(
    "read",
    [lambda w: w.solution, lambda w: w.dual, lambda w: w["x"]],
    ids=["solution", "dual", "getitem"],
)
def test_reading_values_before_solve_raises(read: Any) -> None:
    window = linear_model().restrict(time=TIME[:2])
    with pytest.raises(AttributeError, match="has not been solved"):
        read(window)


def test_dual_size_mismatch_raises() -> None:
    window = linear_model().restrict(time=TIME[:2])
    window.solve(solver_name="highs", output_flag=False)
    assert window._result is not None
    window._result = replace(window._result, dual=np.empty(0))

    with pytest.raises(AttributeError, match="no dual values"):
        window.dual


def test_getitem_after_solve_returns_values() -> None:
    window = linear_model().restrict(time=TIME[:2])
    window.solve(solver_name="highs", output_flag=False)

    assert np.allclose(window["x"].values, 0.0)


def test_getitem_unknown_variable_raises() -> None:
    window = linear_model().restrict(time=TIME[:2])
    with pytest.raises(KeyError):
        window["nope"]


def kinds_model(kinds: str, time: pd.Index) -> Model:
    """Unit demand split across continuous ``C`` and cheaper integer ``I`` supply."""
    m = Model()
    supply = {
        kind: m.add_variables(0, 5, coords=[time], integer=kind == "I", name=kind)
        for kind in kinds
    }
    cost = {"C": 1.0, "I": 0.6}
    lhs = merge([var.to_linexpr() for var in supply.values()], cls=LinearExpression)
    m.add_constraints(lhs >= 1.5, name="demand")
    m.add_objective(
        merge([cost[k] * var.sum() for k, var in supply.items()], cls=LinearExpression)
    )
    return m


@pytest.mark.parametrize(
    "kinds", ["C", "CI", "I"], ids=["continuous", "mixed", "discrete"]
)
def test_window_solve_matches_the_restricted_model(kinds: str) -> None:
    window = kinds_model(kinds, TIME).restrict(time=TIME[:2])
    window.solve(solver_name="highs", output_flag=False)
    reference = kinds_model(kinds, TIME[:2])
    reference.solve(solver_name="highs", output_flag=False)

    assert window.objective.value == pytest.approx(reference.objective.value)
    for kind in kinds:
        assert np.allclose(window[kind].values, reference.solution[kind].values)


def _window_matrices() -> Matrices:
    return linear_model().restrict(time=TIME[:2]).matrices


def test_from_windows_reports_coefficient_changes() -> None:
    previous = _window_matrices()
    assert previous.A is not None
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
        MatrixWindow(storage_model()).select({"time": TIME[1:3]})


def test_nan_fix_counts_as_missing() -> None:
    fix = {"soc": pd.Series(np.full(len(TIME), np.nan), index=TIME)}
    with pytest.raises(RestrictionError, match="dangling terms"):
        MatrixWindow(storage_model()).select({"time": TIME[1:3]}, fix=fix)


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
        MatrixWindow(m)


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
