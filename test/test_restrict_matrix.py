"""Matrix-level windows agree with the container-level restriction."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import highspy
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import LinearExpression, Model, merge
from linopy.matrices import Matrices
from linopy.persistent.diff import ModelDiff, RebuildReason
from linopy.persistent.errors import RebuildRequiredError
from linopy.restrict import MatrixWindow, RestrictionError, WindowModel
from linopy.solvers import Solver

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


@pytest.mark.parametrize(
    "window",
    [TIME[0:3], TIME[1:4], TIME[3:6], [3, 1], [1, 3, 6], [6, 2, 4]],
    ids=["head", "inner", "tail", "reversed", "gapped", "unordered"],
)
def test_select_matches_dense_reference(window: Any) -> None:
    m = storage_model()
    mat = m.matrices
    selection = {"time": window}

    def positions(containers: Any) -> np.ndarray:
        flat = [
            c.labels.sel(selection).values.ravel() for c in containers.data.values()
        ]
        labels = np.concatenate(flat)
        return containers.label_index.label_to_pos[labels[labels != -1]]

    rows, cols = positions(m.constraints), positions(m.variables)
    fixed = np.zeros(mat.vlabels.size)
    fixed[m.variables.label_index.label_to_pos[m.variables["soc"].labels.values]] = 3.0
    fixed[cols] = 0.0
    assert mat.A is not None
    A = mat.A.toarray()[rows]

    window_mat = MatrixWindow(m).select(selection, fix={"soc": initial()})[0]

    assert window_mat.A is not None
    assert np.allclose(window_mat.A.toarray(), A[:, cols])
    assert np.allclose(window_mat.b, mat.b[rows] - A @ fixed)
    assert np.allclose(window_mat.c, mat.c[cols])


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


ZERO_X = {"x": pd.Series(np.zeros(len(TIME)), index=TIME)}


def pinned_model(sign: str, rhs: float) -> Model:
    """Unit values per step, plus a row on the first step only."""
    m = Model()
    x = m.add_variables(lower=0, coords=[TIME], name="x")
    m.add_constraints(x == 1, name="eq")
    m.add_constraints(x.isel(time=0), sign, rhs, name="pin")
    m.add_objective(x.sum())
    return m


@pytest.mark.parametrize(
    ("sign", "rhs", "satisfied"),
    [("==", 2, False), (">=", 2, False), ("<=", -2, False), ("<=", 2, True)],
)
def test_folded_row_is_checked_against_its_sign(
    sign: str, rhs: float, satisfied: bool
) -> None:
    """A row that loses every term is dropped only when its sign holds."""
    window = MatrixWindow(pinned_model(sign, rhs))
    if satisfied:
        assert window.select({"time": TIME[1:3]}, fix=ZERO_X)[0].clabels.size == 2
        return
    with pytest.raises(RestrictionError, match="unsatisfiable"):
        window.select({"time": TIME[1:3]}, fix=ZERO_X)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"tol": 1e-6}, {"drop": {"pin": True}}],
    ids=["default", "loose-tol", "dropped"],
)
def test_folded_residual_passes_within_tol_or_when_dropped(
    kwargs: dict[str, Any],
) -> None:
    m = pinned_model("==", 1e-7)
    if not kwargs:
        with pytest.raises(RestrictionError, match="unsatisfiable"):
            m.restrict(time=TIME[1:3], fix=ZERO_X)
        return
    window = m.restrict(time=TIME[1:3], fix=ZERO_X, **kwargs)

    assert window.matrices.clabels.size == 2


def test_advance_keeps_the_tolerance() -> None:
    window = pinned_model("==", 1e-7).restrict(time=TIME[1:3], fix=ZERO_X, tol=1e-6)
    window.advance(time=TIME[2:4])

    with pytest.raises(RestrictionError, match="unsatisfiable"):
        window.advance(tol=1e-9)


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


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fix": {"nope": 1.0}}, "does not name a variable"),
        ({"drop": {"nope": True}}, "does not name a constraint"),
    ],
    ids=["fix", "drop"],
)
def test_select_rejects_unknown_names(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(RestrictionError, match=match):
        MatrixWindow(linear_model()).select({"time": TIME[:2]}, **kwargs)


def test_dropped_rows_leave_the_window_and_have_no_dual() -> None:
    m = linear_model()
    cap = m.constraints["cap"].labels
    window = m.restrict(time=TIME[:3], drop={"cap": TIME == 1})
    window.solve(solver_name="highs", output_flag=False)

    assert list(window.matrices.clabels) == list(cap.values[[0, 2]])
    assert window.dual["cap"].isnull().values.tolist() == [False, True, False]


def test_advance_updates_the_drop_masks() -> None:
    m = linear_model()
    cap = m.constraints["cap"].labels.values
    window = m.restrict(time=TIME[:3], drop={"cap": TIME == 0})
    assert list(window.matrices.clabels) == list(cap[[1, 2]])

    window.advance(drop={"cap": TIME == 1})
    assert list(window.matrices.clabels) == list(cap[[0, 2]])

    window.advance(time=TIME[1:4])
    assert list(window.matrices.clabels) == list(cap[[2, 3]])

    window.advance(drop={})
    assert list(window.matrices.clabels) == list(cap[1:4])


def target_model() -> Model:
    """A storage that must reach a rising state of charge every second step."""
    m = Model()
    soc = m.add_variables(0, 10, coords=[TIME], name="soc")
    charge = m.add_variables(0, 5, coords=[TIME], name="charge")
    m.add_constraints(soc - soc.shift(time=1) - charge == 0, name="balance")
    target = xr.DataArray(TIME + 1.0, coords=[TIME])
    m.add_constraints(soc == target, mask=target % 2 == 0, name="target")
    m.add_objective(charge.sum() + 0.1 * soc.sum())
    return m


def test_drop_leaves_look_ahead_targets_out_of_rolling_windows() -> None:
    """Each window binds the target of its own block, not of its look-ahead."""
    window = target_model().restrict(time=TIME[:4], drop={"target": TIME > 1})
    window.solve(solver_name="highs", output_flag=False)
    assert np.allclose(window["soc"].values, [2, 2, 2, 2])
    assert np.isnan(window.dual["target"].sel(time=3))

    fix = {"soc": window["soc"].sel(time=TIME[:2]).reindex(time=TIME)}
    window.advance(time=TIME[2:6], fix=fix, drop={"target": TIME > 3})
    window.solve()
    assert np.allclose(window["soc"].values, [2, 4, 4, 4])


@pytest.mark.parametrize(
    "given",
    [
        [1.0, 2.0],
        BOUNDARY.isel(time=slice(0, 4)),
        BOUNDARY.expand_dims(other=[0]),
        xr.DataArray(np.ones(len(TIME) - 1), dims="time"),
    ],
    ids=["wrong-length", "missing-coords", "extra-dim", "coordless-wrong-length"],
)
def test_fix_must_match_the_variable_coordinates(given: object) -> None:
    with pytest.raises(RestrictionError, match="do not match its coordinates"):
        MatrixWindow(linear_model()).select({"time": TIME[:2]}, fix={"x": given})


def ramp_model(scaling: float = 1.0, mask: xr.DataArray | None = None) -> Model:
    """A two-dimensional ramp: every step must not fall below the previous one."""
    m = Model()
    x = m.add_variables(
        0, 100, coords=[TIME, BUS], scaling=scaling, mask=mask, name="x"
    )
    m.add_constraints(x - x.shift(time=1) >= 0, name="ramp")
    m.add_objective(x.sum())
    return m


def solve_window(m: Model, fix: object, **selection: object) -> xr.DataArray:
    window = m.restrict(fix={"x": fix}, **selection)
    window.solve(solver_name="highs", output_flag=False)
    return window.solution["x"]


MASK = xr.DataArray([[True, True]] + [[True, False]] * 7, coords=[TIME, BUS])
ACTIVE = xr.DataArray([[True, True]] * 8, coords=[TIME, BUS])


@pytest.mark.parametrize("mask", [ACTIVE, MASK], ids=["active", "masked"])
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
def test_fix_is_aligned_by_coordinates(
    scaling: float, fix: object, mask: xr.DataArray
) -> None:
    """Fixed values align by label, broadcast over missing dims, in model units."""
    solution = solve_window(ramp_model(scaling, mask), fix, time=TIME[1:3])

    assert solution.dims == ("time", "bus")
    assert solution.where(mask).equals(xr.full_like(solution, 7.0).where(mask))


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


def alternating_sign_model() -> Model:
    """Constraint signs alternate in time, so shifted windows differ in sign."""
    m = Model()
    x = m.add_variables(0, 10, coords=[TIME], name="x")
    sign = xr.DataArray(np.where(TIME % 2 == 0, ">=", "<="), coords=[TIME])
    m.add_constraints(x, sign, 5, name="c")
    m.add_objective(x.sum())
    return m


@pytest.mark.parametrize(
    ("m", "first", "second", "reason", "expected"),
    [
        (
            sparse_coupling_model(),
            TIME[:2],
            TIME[2:4],
            RebuildReason.STRUCTURAL_LABELS,
            {"x": [3, 3], "y": [0, 0]},
        ),
        (
            alternating_sign_model(),
            TIME[:2],
            TIME[1:3],
            RebuildReason.BACKEND_REJECTED,
            {"x": [0, 5]},
        ),
    ],
    ids=["sparsity", "sign"],
)
def test_advance_rebuilds_when_an_update_is_impossible(
    m: Model,
    first: pd.Index,
    second: pd.Index,
    reason: RebuildReason,
    expected: dict[str, list[float]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    window = m.restrict(time=first)
    window.solve(solver_name="highs", output_flag=False)
    window.advance(time=second)

    with pytest.raises(RebuildRequiredError) as excinfo:
        window.solve(disallow_rebuild=True)
    assert excinfo.value.reason is reason
    rejected = reason is RebuildReason.BACKEND_REJECTED
    assert (excinfo.value.__cause__ is not None) is rejected
    with caplog.at_level(logging.WARNING, logger="linopy.solvers"):
        window.solve()
    assert "rebuild required" not in caplog.text

    assert window._solver is not None
    assert window._solver._rebuilds == 1
    assert window._solver._last_rebuild_reason is reason
    for name, values in expected.items():
        assert np.allclose(window.solution[name].values, values)


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
    selection: dict[str, Any],
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


def knapsack_model() -> Model:
    """A binary knapsack that HiGHS does not solve at its first incumbent."""
    items = pd.RangeIndex(12, name="item")
    weight = xr.DataArray(np.arange(12) * 37 % 41 + 5, coords=[items])
    value = xr.DataArray(np.arange(12) * 29 % 43 + 5, coords=[items])
    m = Model()
    x = m.add_variables(binary=True, coords=[items], name="x")
    m.add_constraints((weight * x).sum() <= 60, name="cap")
    m.add_objective(-(value * x).sum())
    return m


def test_limited_mip_solve_returns_its_incumbent() -> None:
    """Options of a later in-place solve apply, and its incumbent is readable."""
    window = knapsack_model().restrict(item=slice(None))
    window.solve(solver_name="highs", output_flag=False)
    assert window.termination_condition == "optimal"
    optimum = window.objective.value

    window.solve(disallow_rebuild=True, mip_max_improving_sols=1)

    assert window.termination_condition == "terminated_by_limit"
    assert set(np.unique(window["x"].values)) <= {0.0, 1.0}
    assert window.objective.value > optimum


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
    assert window._result is not None and window._result.solution is not None
    solution = replace(window._result.solution, dual=np.empty(0))
    window._result = replace(window._result, solution=solution)

    with pytest.raises(AttributeError, match="no dual values"):
        window.dual


def test_status_before_solve() -> None:
    window = linear_model().restrict(time=TIME[:2])

    assert window.status == "initialized"
    assert window.termination_condition == ""
    assert window.objective.value is None


def test_status_after_optimal_solve() -> None:
    window = linear_model().restrict(time=TIME[:2])
    window.solve(solver_name="highs", output_flag=False)

    assert window.status == "ok"
    assert window.termination_condition == "optimal"


def infeasible_window() -> WindowModel:
    sns = pd.RangeIndex(4, name="snapshot")
    m = Model()
    x = m.add_variables(0, 1, coords=[sns], name="x")
    m.add_constraints(x >= 2, name="infeasible")
    m.add_objective(x.sum())
    return m.restrict(snapshot=sns[:2])


@pytest.mark.parametrize(
    "read",
    [lambda w: w.solution, lambda w: w.dual, lambda w: w["x"]],
    ids=["solution", "dual", "getitem"],
)
def test_infeasible_window_reports_status_and_raises_on_read(read: Any) -> None:
    window = infeasible_window()
    window.solve(solver_name="highs", output_flag=False)

    assert window.status == "warning"
    assert window.termination_condition == "infeasible"
    with pytest.raises(AttributeError, match="termination condition 'infeasible'"):
        read(window)


def test_infeasible_window_has_no_objective_value() -> None:
    window = infeasible_window()
    window.solve(solver_name="highs", output_flag=False)

    assert window.objective.value is None


def solved_window() -> WindowModel:
    window = linear_model().restrict(time=TIME[:2])
    window.solve(solver_name="highs", output_flag=False)
    assert window._result is not None
    return window


def test_ok_solve_without_a_solution_raises_on_read() -> None:
    window = solved_window()
    assert window._result is not None
    window._result = replace(window._result, solution=None)

    assert window.objective.value is None
    with pytest.raises(AttributeError, match="returned no solution"):
        window.solution


def test_close_drops_the_solver_and_keeps_the_result() -> None:
    window = solved_window()
    solver = window.solver
    assert solver is not None
    assert window.solver_model is solver.solver_model is not None

    window.close()

    assert window.solver is None
    assert window.solver_model is None
    assert solver.solver_model is None
    assert np.allclose(window["x"].values, 0.0)
    window.solve()
    assert window.solver is not None
    assert window.solver is not solver
    assert window.status == "ok"


@pytest.mark.parametrize("solver_name", ["highs", "gurobi"])
def test_solve_writes_one_log_per_window(solver_name: str, tmp_path: Path) -> None:
    from linopy import licensed_solvers

    if solver_name not in licensed_solvers:
        pytest.skip(f"{solver_name} is not available with a license")

    fix = {"soc": initial()}
    first, second = tmp_path / "first.log", tmp_path / "second.log"
    window = storage_model().restrict(time=TIME[1:4], fix=fix)
    window.solve(solver_name=solver_name, log_fn=first)
    logged = first.read_text()
    window.advance(time=TIME[4:7], fix=fix)
    window.solve(disallow_rebuild=True, log_fn=second)

    assert logged
    assert first.read_text().startswith(logged)
    assert second.read_text()


def test_solve_that_raises_drops_the_previous_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = solved_window()
    assert window._solver is not None

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("solver crashed")

    monkeypatch.setattr(window._solver, "solve_window", broken)
    with pytest.raises(RuntimeError, match="solver crashed"):
        window.solve()

    assert window.status == "initialized"
    assert window.objective.value is None


def test_solve_logs_the_objective_in_model_units(
    caplog: pytest.LogCaptureFixture,
) -> None:
    window = kinds_model("C", TIME, scaling=10.0).restrict(time=TIME[:2])
    with caplog.at_level(logging.INFO, logger="linopy.constants"):
        window.solve(solver_name="highs", output_flag=False)

    assert window.objective.value == pytest.approx(3.0)
    assert "Objective: 3.00e+00" in caplog.text


def test_status_resets_on_advance() -> None:
    window = linear_model().restrict(time=TIME[:2])
    window.solve(solver_name="highs", output_flag=False)
    window.advance(time=TIME[1:3])

    assert window.status == "initialized"
    assert window.termination_condition == ""


def test_solution_keeps_per_variable_coords_on_shared_dims() -> None:
    sns = pd.RangeIndex(2, name="snapshot")
    m = Model()
    a = m.add_variables(
        0, 1, coords=[sns, pd.Index(["g1", "g2"], name="name")], name="a"
    )
    b = m.add_variables(0, 1, coords=[sns, pd.Index(["s1"], name="name")], name="b")
    m.add_constraints(a.sum("name") + b.sum("name") >= 1, name="demand")
    m.add_constraints(a >= 0, name="a_lb")
    m.add_constraints(b >= 0, name="b_lb")
    m.add_objective(a.sum() + 2 * b.sum())
    window = m.restrict(snapshot=sns[:1])
    window.solve(solver_name="highs", output_flag=False)

    solution, dual = window.solution, window.dual
    assert list(solution["a"].indexes["name"]) == ["g1", "g2"]
    assert list(solution["b"].indexes["name"]) == ["s1"]
    assert solution["b"].identical(window["b"])
    assert not solution["b"].isnull().any()
    assert list(dual["a_lb"].indexes["name"]) == ["g1", "g2"]
    assert list(dual["b_lb"].indexes["name"]) == ["s1"]
    assert not dual["b_lb"].isnull().any()


def test_getitem_after_solve_returns_values() -> None:
    window = linear_model().restrict(time=TIME[:2])
    window.solve(solver_name="highs", output_flag=False)

    assert np.allclose(window["x"].values, 0.0)


def test_getitem_unknown_variable_raises() -> None:
    window = linear_model().restrict(time=TIME[:2])
    with pytest.raises(KeyError):
        window["nope"]


def kinds_model(kinds: str, time: pd.Index, scaling: float = 1.0) -> Model:
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
        merge([cost[k] * var.sum() for k, var in supply.items()], cls=LinearExpression),
        scaling=scaling,
    )
    return m


@pytest.mark.parametrize("scaling", [1.0, 10.0])
@pytest.mark.parametrize(
    "kinds", ["C", "CI", "I"], ids=["continuous", "mixed", "discrete"]
)
def test_window_solve_matches_the_restricted_model(kinds: str, scaling: float) -> None:
    window = kinds_model(kinds, TIME, scaling).restrict(time=TIME[:2])
    window.solve(solver_name="highs", output_flag=False)
    reference = kinds_model(kinds, TIME[:2], scaling)
    reference.solve(solver_name="highs", output_flag=False)

    assert window.objective.value == pytest.approx(reference.objective.value)
    for kind in kinds:
        assert np.allclose(window[kind].values, reference.solution[kind].values)


@pytest.mark.parametrize("scaling", [1.0, 10.0])
def test_solve_fixed_lp_reports_the_duals_of_the_fixed_mip(scaling: float) -> None:
    window = kinds_model("CI", TIME, scaling).restrict(time=TIME[:2])
    window.solve(solver_name="highs", output_flag=False)
    window.solve_fixed_lp()
    reference = kinds_model("CI", TIME[:2], scaling)
    reference.solve(solver_name="highs", output_flag=False)
    reference.variables["I"].fix()
    reference.variables["I"].relax()
    reference.solve(solver_name="highs", output_flag=False)

    assert window.matrices.type == "MILP"
    assert np.allclose(window["I"].values, 1.0)
    assert np.allclose(window.dual["demand"].values, reference.dual["demand"].values)
    assert window.objective.value == pytest.approx(reference.objective.value)


def test_solve_after_solve_fixed_lp_restores_integrality() -> None:
    window = kinds_model("CI", TIME).restrict(time=TIME[:2])
    window.solve(solver_name="highs", output_flag=False)
    window.solve_fixed_lp()
    window.solve(disallow_rebuild=True)

    assert window.objective.value == pytest.approx(2 * 1.1)
    assert np.allclose(window["I"].values, 1.0)


def test_solve_fixed_lp_requires_a_solution() -> None:
    with pytest.raises(AttributeError, match="has not been solved"):
        kinds_model("CI", TIME).restrict(time=TIME[:2]).solve_fixed_lp()


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
    """Every available direct solver updates a same-pattern window in place."""
    from linopy import licensed_solvers

    if solver_name not in licensed_solvers:
        pytest.skip(f"{solver_name} is not available with a license")

    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[TIME], name="x")
    y = m.add_variables(lower=0, coords=[TIME], name="y")
    m.add_constraints(x - 0.9 * x.shift(time=1) - y == 0, name="balance")
    m.add_objective((x + y).sum())
    fix = {"x": pd.Series(TIME + 1.0, index=TIME)}
    decay = 0.9 ** np.arange(1, 4)

    window = m.restrict(time=TIME[1:4], fix=fix)
    try:
        window.solve(solver_name=solver_name)
    except NotImplementedError as exc:  # a file-only backend
        pytest.skip(str(exc))
    assert np.allclose(window.solution["x"].values, decay)

    window.advance(time=TIME[4:7], fix=fix)
    window.solve(disallow_rebuild=True, warm_start=True)
    assert window._solver is not None
    assert window._solver._in_place_updates == 1
    assert np.allclose(window.solution["x"].values, 4 * decay)
    assert window.objective.value == pytest.approx(4 * decay.sum())


def simplex_iterations(window: WindowModel) -> int:
    assert window._solver is not None
    return window._solver.solver_model.getInfo().simplex_iteration_count


@pytest.mark.parametrize("warm_start", [None, False, True])
def test_window_solve_discards_the_warm_start_by_default(
    warm_start: bool | None,
) -> None:
    window = storage_model().restrict(time=TIME[1:4], fix={"soc": initial()})
    window.solve(solver_name="highs", presolve="off")
    assert simplex_iterations(window) > 0

    window.advance()
    if warm_start is None:
        window.solve(disallow_rebuild=True)
    else:
        window.solve(disallow_rebuild=True, warm_start=warm_start)
    assert window._solver is not None
    assert window._solver._in_place_updates == 1
    assert (simplex_iterations(window) == 0) == bool(warm_start)
    assert window.status == "ok"


@pytest.mark.parametrize("solver_name", ["highs", "gurobi"])
def test_cold_window_solve_updates_in_place(solver_name: str) -> None:
    from linopy import licensed_solvers

    if solver_name not in licensed_solvers:
        pytest.skip(f"{solver_name} is not available with a license")

    fix = {"soc": initial()}
    window = storage_model().restrict(time=TIME[1:4], fix=fix)
    window.solve(solver_name=solver_name)
    window.advance(time=TIME[4:7], fix=fix)
    window.solve(disallow_rebuild=True)

    assert window._solver is not None
    assert window._solver._in_place_updates == 1
    fresh = storage_model().restrict(time=TIME[4:7], fix=fix).solve(solver_name)
    assert window.objective.value == pytest.approx(fresh.objective.value)


def test_cold_window_solve_rebuilds_without_backend_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linopy.persistent.errors import UnsupportedUpdate
    from linopy.solvers import Highs

    def unsupported(self: Highs) -> None:
        raise UnsupportedUpdate("no reset")

    monkeypatch.setattr(Highs, "_discard_warm_start", unsupported)
    fix = {"soc": initial()}
    window = storage_model().restrict(time=TIME[1:4], fix=fix)
    window.solve(solver_name="highs")
    window.advance(time=TIME[4:7], fix=fix)
    with pytest.raises(RebuildRequiredError):
        window.solve(disallow_rebuild=True)

    window.solve()
    assert window._solver is not None
    assert window._solver._last_rebuild_reason is RebuildReason.BACKEND_REJECTED
    assert window.status == "ok"


def test_window_solve_detaches_the_solver_from_its_model() -> None:
    solver = Solver.from_name("highs", io_api="direct", track_updates=True)
    solver.solve(model=linear_model())
    solver.solve_window(linear_model().restrict(time=TIME[:2]).matrices)

    with pytest.raises(RuntimeError, match="matrix window"):
        solver.update(linear_model())


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


@pytest.mark.parametrize("suffix", [".lp", ".mps"])
@pytest.mark.parametrize(
    "explicit, columns, rows",
    [
        (False, ["x1", "x2", "x9", "x10"], ["c1", "c2", "c9", "c10"]),
        (
            True,
            ["soc(1)#1", "soc(2)#2", "charge(1)#9", "charge(2)#10"],
            ["balance(1)#1", "balance(2)#2", "cap(1)#9", "cap(2)#10"],
        ),
    ],
    ids=["labels", "coordinates"],
)
def test_to_file_writes_the_window_with_parent_names(
    tmp_path: Path, suffix: str, explicit: bool, columns: list[str], rows: list[str]
) -> None:
    window = storage_model().restrict(time=TIME[1:3], fix={"soc": initial()})
    window.solve(solver_name="highs", output_flag=False)

    fn = window.to_file(
        tmp_path / f"window{suffix}", explicit_coordinate_names=explicit
    )
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.readModel(str(fn))
    h.run()

    assert h.getLp().col_names_ == columns
    assert h.getLp().row_names_ == rows
    assert h.getInfo().objective_function_value == pytest.approx(window.objective.value)


def test_to_file_rejects_unknown_formats(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="expected an '.lp' or '.mps' file"):
        linear_model().restrict(time=TIME[:2]).to_file(tmp_path / "window.nc")


def infeasible_boundary_model() -> Model:
    """A ramp from a boundary fixed at zero cannot reach the floor at time 3."""
    m = Model()
    x = m.add_variables(0, 10, coords=[TIME], name="x")
    m.add_constraints(x - x.shift(time=1) <= 1, name="ramp")
    floor = xr.DataArray(np.where(TIME == 3, 5.0, 0.0), coords=[TIME])
    m.add_constraints(x >= floor, name="lower")
    m.add_objective(x.sum())
    return m


@pytest.mark.parametrize("solver_name", ["highs", "gurobi", "xpress"])
def test_compute_infeasibilities_returns_parent_labels(solver_name: str) -> None:
    from linopy import licensed_solvers

    if solver_name not in licensed_solvers:
        pytest.skip(f"{solver_name} is not available with a license")
    m = infeasible_boundary_model()
    window = m.restrict(time=TIME[3:5], fix={"x": pd.Series(0.0, index=TIME)})
    window.solve(solver_name=solver_name)

    labels = window.compute_infeasibilities()

    assert [m.constraints.get_label_position(label) for label in labels] == [
        ("ramp", {"time": 3}),
        ("lower", {"time": 3}),
    ]
    assert m.constraints.format_labels(labels).splitlines() == [
        "ramp[3]: +1 x[3] - 1 x[2] ≤ 1",
        "lower[3]: +1 x[3] ≥ 5",
    ]


@pytest.mark.parametrize("advance", [False, True], ids=["unsolved", "advanced"])
def test_compute_infeasibilities_requires_a_solve(advance: bool) -> None:
    window = infeasible_window()
    if advance:
        window.solve(solver_name="highs", output_flag=False)
        window.advance()

    with pytest.raises(ValueError, match="has not been solved"):
        window.compute_infeasibilities()
