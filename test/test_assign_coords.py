#!/usr/bin/env python3
"""
Test Model.assign_coords and the per-container assign_coords machinery.

Coordinate reassignment on an existing model: values-only replacement with
unchanged shape, dataset variable order preserved, model-wide propagation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model
from linopy.constants import Result, Solution, Status
from linopy.constraints import CSRConstraint

sns0 = pd.date_range("2026-01-01", periods=3, freq="h", name="snapshot")
sns1 = sns0 + pd.Timedelta("1h")


@pytest.fixture
def m() -> Model:
    """Model with a ``snapshot`` dim in every coordinate-carrying container."""
    model = Model()
    x = model.add_variables(coords=[sns0], name="x")
    y = model.add_variables(
        coords=[sns0, ["a", "b"]], dims=["snapshot", "spatial"], name="y"
    )
    model.add_constraints(x >= 0, name="dense_c")
    model.add_constraints(x >= 1, name="frozen_c", freeze=True)
    model.add_expressions(x * 2 + y.sum(), name="e")
    model.add_objective(1.0 * x)
    model.parameters = xr.Dataset(
        {"w": (("snapshot",), [1.0, 2.0, 3.0])}, coords={"snapshot": sns0}
    )
    return model


def test_assign_coords_reaches_all_containers(m: Model) -> None:
    """Reassignment lands on every container type and returns the model."""
    result = m.assign_coords(snapshot=sns1)

    assert result is m
    assert (m.variables["x"].coords["snapshot"].values == sns1.values).all()
    assert (m.variables["y"].coords["snapshot"].values == sns1.values).all()
    assert (m.constraints["dense_c"].coords["snapshot"].values == sns1.values).all()
    assert (m.expressions["e"].coords["snapshot"].values == sns1.values).all()
    assert (m.parameters.coords["snapshot"].values == sns1.values).all()


def test_assign_coords_csr_constraint(m: Model) -> None:
    """
    CSR-backed constraints get the new labels on their grid, not only in
    their reconstructed Dataset view.
    """
    con = m.constraints["frozen_c"]
    assert isinstance(con, CSRConstraint)

    m.assign_coords(snapshot=sns1)

    con = m.constraints["frozen_c"]
    assert (con.coords["snapshot"].values == sns1.values).all()
    # CSR-backed container still reconstructs its Dataset with the new coords
    assert (con.data.coords["snapshot"].values == sns1.values).all()
    assert con.labels.sizes["snapshot"] == 3


def test_assign_coords_preserves_order(m: Model) -> None:
    """
    Coordinate and data-variable order survive the reassignment.

    Plain ``Dataset.assign_coords`` moves the updated coord to the end, which
    breaks dim inference downstream (e.g. in ``Model.assign_result``) — the
    order-safe rebuild must not.
    """
    var_before = m.variables["y"]
    con_before = m.constraints["dense_c"]
    coords_before = list(var_before.coords)
    var_order_before = list(var_before.data.data_vars)
    con_coords_before = list(con_before.coords)
    con_order_before = list(con_before.data.data_vars)

    m.assign_coords(snapshot=sns1)

    assert list(m.variables["y"].coords) == coords_before
    assert list(m.variables["y"].data.data_vars) == var_order_before
    assert list(m.constraints["dense_c"].coords) == con_coords_before
    assert list(m.constraints["dense_c"].data.data_vars) == con_order_before


def test_assign_coords_values_only(m: Model) -> None:
    """Only index labels move: shape, dims and data variables are untouched."""
    y = m.variables["y"]
    dims_before = y.dims
    data_vars_before = list(y.data.data_vars)

    m.assign_coords(snapshot=sns1)

    assert y.shape == (3, 2)
    assert y.dims == dims_before
    # labels and bounds are untouched, only the index labels moved
    assert y.labels.dims == y.dims
    assert list(y.data.data_vars) == data_vars_before


def test_assign_result_after_reassign(m: Model) -> None:
    """
    Solution and dual values land on the *new* labels after reassignment.

    This is the rolling-horizon flow from the issue: reassign the window,
    then write a solver result back — solution/dual arrays must carry the
    updated coordinates, not the stale ones.
    """
    n_labels = 1 + max(int(v.labels.max()) for _, v in m.variables.items())
    n_cons = 1 + max(int(c.labels.max()) for _, c in m.constraints.items())
    result = Result(
        status=Status.process("ok", "optimal"),
        solution=Solution(
            primal=np.arange(n_labels, dtype=float),
            dual=np.arange(n_cons, dtype=float),
            objective=42.0,
        ),
    )

    m.assign_coords(snapshot=sns1)
    m.assign_result(result)

    solution = m.variables["x"].solution
    assert (solution.coords["snapshot"].values == sns1.values).all()
    assert solution.dims == ("snapshot",)

    dual = m.constraints["dense_c"].dual
    assert (dual.coords["snapshot"].values == sns1.values).all()
    assert dual.dims == ("snapshot",)


def test_mock_solve_after_reassign(m: Model) -> None:
    """Mock solving still infers dims correctly from the reordered-safe coords."""
    m.assign_coords(snapshot=sns1)
    m._mock_solve()
    assert (m.variables["x"].solution.coords["snapshot"].values == sns1.values).all()


def test_assign_coords_container_methods_return_self(m: Model) -> None:
    """The private per-container methods mutate in place and chain."""
    assert m.variables["x"]._assign_coords(snapshot=sns1) is m.variables["x"]
    assert (
        m.constraints["dense_c"]._assign_coords(snapshot=sns1)
        is (m.constraints["dense_c"])
    )
    assert m.expressions["e"]._assign_coords(snapshot=sns1) is m.expressions["e"]


def test_assign_coords_rejects_wrong_length(m: Model) -> None:
    """
    New values shorter than the dimension raise, at both API levels.

    Validates the issue's "same length" contract and that a failed call
    leaves the model untouched.
    """
    short = pd.date_range("2026-01-01", periods=2, freq="h", name="snapshot")
    with pytest.raises(ValueError, match="length"):
        m.assign_coords(snapshot=short)

    with pytest.raises(ValueError, match="length"):
        m.variables["x"]._assign_coords(snapshot=short)

    # nothing was mutated
    assert (m.variables["x"].coords["snapshot"].values == sns0.values).all()


def test_assign_coords_rejects_unknown_dim(m: Model) -> None:
    """
    Unknown dimensions raise instead of being added as new coordinates.

    The model-level message reports the dimension as not found in the model;
    the per-container message reports the missing coordinate.
    """
    with pytest.raises(ValueError, match="not found"):
        m.assign_coords(nonexistent=[1, 2, 3])

    with pytest.raises(ValueError, match="missing"):
        m.variables["x"]._assign_coords(nonexistent=[1, 2, 3])


def test_assign_coords_accepts_index_like_values(m: Model) -> None:
    """
    Lists, pandas Index objects and DataArrays are accepted as values.

    A DataArray contributes its values, like ``xarray.assign_coords``; a
    passed Index that is named differently is renamed to the target
    dimension (the keyword key is authoritative).
    """
    m.assign_coords(snapshot=list(range(3)))
    assert (m.variables["x"].coords["snapshot"].values == np.arange(3)).all()

    m.assign_coords(snapshot=pd.Index([9, 8, 7], name="snapshot"))
    assert (m.variables["x"].coords["snapshot"].values == [9, 8, 7]).all()

    # a DataArray contributes its values, like xarray.assign_coords
    da = xr.DataArray([1, 2, 3], coords={"snapshot": sns1})
    m.assign_coords(snapshot=da)
    assert (m.variables["x"].coords["snapshot"].values == [1, 2, 3]).all()


def test_assign_coords_multiple_dims(m: Model) -> None:
    """Several dimensions can be reassigned in one call."""
    m.assign_coords(snapshot=sns1, spatial=["c", "d"])
    assert (m.variables["y"].coords["snapshot"].values == sns1.values).all()
    assert (m.variables["y"].coords["spatial"].values == ["c", "d"]).all()


def test_model_assign_coords_skips_carriers_without_dim(m: Model) -> None:
    """Containers not carrying the dimension are left untouched."""
    m.add_variables(coords=[pd.RangeIndex(2, name="other")], name="scalar_var")

    m.assign_coords(snapshot=sns1)

    assert (m.variables["scalar_var"].coords["other"].values == [0, 1]).all()
    assert (m.variables["x"].coords["snapshot"].values == sns1.values).all()


def test_assign_coords_rejects_scalar_values(m: Model) -> None:
    """Scalars are not index-like and raise instead of broadcasting."""
    with pytest.raises(ValueError, match="index-like"):
        m.assign_coords(snapshot=5)

    with pytest.raises(ValueError, match="index-like"):
        m.variables["x"]._assign_coords(snapshot=5)


def test_csr_assign_coords_rejects_wrong_length(m: Model) -> None:
    """The CSR path validates lengths and rejects unknown coordinates too."""
    short = pd.date_range("2026-01-01", periods=2, freq="h", name="snapshot")
    con = m.constraints["frozen_c"]
    assert isinstance(con, CSRConstraint)

    with pytest.raises(ValueError, match="length"):
        con._assign_coords(snapshot=short)

    with pytest.raises(ValueError, match="missing"):
        con._assign_coords(nonexistent=[1, 2, 3])


def test_assign_coords_reaches_quadratic_expression(m: Model) -> None:
    """Stored quadratic expressions get the new labels like linear ones."""
    x = m.variables["x"]
    m.add_expressions(x * x, name="q")

    m.assign_coords(snapshot=sns1)

    q = m.expressions["q"]
    assert (q.coords["snapshot"].values == sns1.values).all()


def test_assign_coords_skips_uncoordinated_parameters(m: Model) -> None:
    """A parameters dim without an index coord has no labels to move: skip it."""
    m.parameters = xr.Dataset({"w": ("snapshot", [1.0, 2.0, 3.0])})

    m.assign_coords(snapshot=sns1)

    assert (m.variables["x"].coords["snapshot"].values == sns1.values).all()


def test_assign_coords_renames_mismatched_index(m: Model) -> None:
    """A passed Index named differently is renamed to the target dimension."""
    m.assign_coords(snapshot=pd.Index([9, 8, 7], name="timestep"))
    coord = m.variables["x"].coords["snapshot"]
    assert coord.name == "snapshot"
    assert (coord.values == [9, 8, 7]).all()


def test_assign_coords_maps_subset_carriers(m: Model) -> None:
    """
    A container holding a subset of the dimension is mapped by label.

    Case 1 from the design: a piecewise commitment gate ``u`` on a subset of
    ``gen`` must follow the master's relabeling (``a -> g1``, ``c -> g3``),
    preserving the subset relation — not be relabeled positionally.
    """
    from linopy.variables import Variable

    full = pd.Index(["a", "b", "c"], name="gen")
    subset = pd.Index(["a", "c"], name="gen")
    m2 = Model()
    x2 = m2.add_variables(coords=[full], name="x")
    m2.add_variables(binary=True, coords=[subset], name="u")
    m2.add_constraints(x2 >= 0, name="c")
    m2.add_constraints(x2.sel(gen=["a", "c"]) >= 1, name="c_subset")

    m2.assign_coords(gen=["g1", "g2", "g3"])

    assert (m2.variables["x"].coords["gen"].values == ["g1", "g2", "g3"]).all()
    assert (m2.variables["u"].coords["gen"].values == ["g1", "g3"]).all()
    assert (m2.constraints["c_subset"].coords["gen"].values == ["g1", "g3"]).all()

    # a container with labels outside the master's index is corruption: raise
    odd_data = x2.data.isel(gen=slice(1, 2)).assign_coords(gen=["z"])
    m2.variables.data["odd"] = Variable(odd_data, m2, "odd")
    with pytest.raises(ValueError, match="outside the"):
        m2.assign_coords(gen=["h1", "h2", "h3"])


def test_assign_coords_rejects_unmatched_master_length(m: Model) -> None:
    """New values matching no carrier's length raise, listing the lengths."""
    with pytest.raises(ValueError, match="no container carries it"):
        m.assign_coords(snapshot=pd.date_range("2026-01-01", periods=7, freq="h"))


def test_assign_coords_rejects_ambiguous_masters(m: Model) -> None:
    """Same-length containers with different values are ambiguous: raise."""
    m.variables["x"]._assign_coords(snapshot=sns0 + pd.Timedelta("5h"))

    with pytest.raises(ValueError, match="matching length carry different"):
        m.assign_coords(snapshot=sns1)


@pytest.mark.v1
def test_solve_rejects_diverged_coords(m: Model) -> None:
    """
    v1 solving raises when containers carry incompatible labels on a dim.

    Same length, different values — the mislabeling trap no construction-time
    §8 check catches (and which the public API cannot build); the pre-solve
    guard is the backstop. A re-aligned model passes again.
    """
    shifted = sns0 + pd.Timedelta("5h")
    m.variables["x"]._assign_coords(snapshot=shifted)

    with pytest.raises(ValueError, match="incompatible"):
        m._check_coord_consistency()

    # a consistent model passes
    m.variables["x"]._assign_coords(snapshot=sns0)
    m._check_coord_consistency()


@pytest.mark.legacy
def test_check_coord_consistency_legacy_noop(m: Model) -> None:
    """
    Under legacy, non-aligned containers are documented positional
    behavior (convention §8), so the guard is a no-op.
    """
    shifted = sns0 + pd.Timedelta("5h")
    m.variables["x"]._assign_coords(snapshot=shifted)

    m._check_coord_consistency()


@pytest.mark.v1
def test_check_coord_consistency_allows_helper_dim_divergence(m: Model) -> None:
    """
    Internal (underscore-prefixed) dims are exempt from the guard.

    The piecewise machinery labels ``_breakpoint_piece`` differently on each
    container on purpose — that must not trip the guard.
    """
    delta = m.add_variables(
        binary=True,
        coords=[sns0, [0, 1, 2]],
        dims=["snapshot", "_breakpoint_piece"],
        name="delta",
    )
    delta_hi = delta.isel(_breakpoint_piece=slice(1, None), drop=True)
    delta_hi._assign_coords(_breakpoint_piece=[0, 1])

    m._check_coord_consistency()


@pytest.mark.v1
def test_check_coord_consistency_allows_subset_labels(m: Model) -> None:
    """Containers nesting by inclusion (a gate on a gen subset) pass the guard."""
    m.add_variables(
        binary=True,
        coords=[pd.Index(["a", "c"], name="gen")],
        name="u",
    )
    m.add_variables(
        coords=[pd.Index(["a", "b", "c"], name="gen")],
        name="x_gen",
    )

    m._check_coord_consistency()


@pytest.mark.v1
def test_check_coord_consistency_rejects_partial_overlap(m: Model) -> None:
    """Labels neither equal nor nested (partial overlap) raise under v1."""
    m.add_variables(coords=[pd.Index(["a", "c"], name="gen")], name="u")
    m.add_variables(coords=[pd.Index(["b", "c"], name="gen")], name="v")

    with pytest.raises(ValueError, match="incompatible"):
        m._check_coord_consistency()
