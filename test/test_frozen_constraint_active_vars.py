"""
Differential tests for frozen (CSR-backed) constraints: changing the variables.

The invariant under test (stated in PR #630): ``to_matrix`` on both the mutable
``Constraint`` and the frozen ``CSRConstraint`` "always returns a csr matrix of
shape (n_active_constraints, n_active_variables)". A frozen constraint caches
its CSR at freeze time, so any change to the active-variable set *after* freezing
(adding or removing variables) must still yield a matrix consistent with the
mutable path.

Strategy: build the same model twice under an identical construction order --
once with ``freeze_constraints=False`` (mutable, treated as ground truth) and
once with ``freeze_constraints=True`` -- and assert the assembled constraint
matrix, shape, and solved objective agree. Any divergence is a frozen-path bug.

The mechanism under test is ``CSRConstraint._reconciled_csr``: the frozen CSR
records the active-variable basis its column indices refer to and is lazily
reconciled to the model's current basis on access (widened after additions,
remapped through variable labels after removals). Scenarios J-O cover the
consumers beyond matrix assembly (``.data``, ``to_polars``, ``repr``,
``has_variable``, ``iterate_slices``, netcdf round-trip) and the guard against
dangling references.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import linopy
from linopy import Model, read_netcdf
from linopy.constraints import CSRConstraint


def _assemble(freeze: bool) -> Model:
    """
    Return a model built in a fixed order that freezes before adding a variable.

    Order: add x, add a constraint on x, THEN add y, add a constraint on y.
    With freeze_constraints=True the first constraint is frozen while only x
    exists.
    """
    m = Model(freeze_constraints=freeze)
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(3, name="i")], name="x")
    m.add_constraints(x >= 1, name="cx")
    y = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="j")], name="y")
    m.add_constraints(y >= 2, name="cy")
    m.add_objective(x.sum() + y.sum())
    return m


def _dense_A(m: Model) -> np.ndarray:
    A = m.matrices.A
    assert A is not None
    return np.asarray(A.todense())


# ---------------------------------------------------------------------------
# Scenario A: baseline -- all variables added before any frozen constraint.
# This is what upstream fixtures do; should pass with or without the fix.
# ---------------------------------------------------------------------------
def test_A_all_vars_first() -> None:
    m_mut = Model()
    x = m_mut.add_variables(lower=0, coords=[pd.RangeIndex(3, name="i")], name="x")
    y = m_mut.add_variables(lower=0, coords=[pd.RangeIndex(2, name="j")], name="y")
    m_mut.add_constraints(x >= 1, name="cx", freeze=False)
    m_mut.add_constraints(y >= 2, name="cy", freeze=False)

    m_frz = Model()
    x = m_frz.add_variables(lower=0, coords=[pd.RangeIndex(3, name="i")], name="x")
    y = m_frz.add_variables(lower=0, coords=[pd.RangeIndex(2, name="j")], name="y")
    m_frz.add_constraints(x >= 1, name="cx", freeze=True)
    m_frz.add_constraints(y >= 2, name="cy", freeze=True)

    np.testing.assert_array_equal(_dense_A(m_mut), _dense_A(m_frz))


# ---------------------------------------------------------------------------
# Scenario B: interleaved -- variable added AFTER a frozen constraint.
# This is the widening case. Expected to FAIL on unpatched master.
# ---------------------------------------------------------------------------
def test_B_var_added_after_freeze_matrix_shape() -> None:
    m = _assemble(freeze=True)
    n_vars = m.variables.label_index.n_active_vars
    n_cons = m.constraints.label_index.n_active_cons
    A = m.matrices.A
    assert A is not None
    assert A.shape == (n_cons, n_vars)


def test_B_interleaved_matrix_matches_mutable() -> None:
    A_mut = _dense_A(_assemble(freeze=False))
    A_frz = _dense_A(_assemble(freeze=True))
    assert A_mut.shape == A_frz.shape
    np.testing.assert_array_equal(A_mut, A_frz)


def test_B_interleaved_solves_equal() -> None:
    if "highs" not in linopy.available_solvers:
        pytest.skip("highs unavailable")
    m_mut = _assemble(freeze=False)
    m_frz = _assemble(freeze=True)
    m_mut.solve(solver_name="highs")
    m_frz.solve(solver_name="highs")
    obj_mut = m_mut.objective.value
    obj_frz = m_frz.objective.value
    assert obj_mut is not None and obj_frz is not None
    assert obj_frz == pytest.approx(obj_mut)


# ---------------------------------------------------------------------------
# Scenario C: a frozen constraint that links a variable added later, plus an
# extra late variable that widens the index further.
# ---------------------------------------------------------------------------
def _assemble_linking(freeze: bool) -> Model:
    m = Model(freeze_constraints=freeze)
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="x")
    m.add_constraints(x >= 1, name="cx")
    y = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="y")
    # constraint linking x and y, frozen while both exist
    m.add_constraints(x + y >= 5, name="cxy")
    z = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="z")
    m.add_constraints(z >= 3, name="cz")
    m.add_objective(x.sum() + y.sum() + z.sum())
    return m


def test_C_linking_matrix_matches_mutable() -> None:
    A_mut = _dense_A(_assemble_linking(freeze=False))
    A_frz = _dense_A(_assemble_linking(freeze=True))
    assert A_mut.shape == A_frz.shape
    np.testing.assert_array_equal(A_mut, A_frz)


# ---------------------------------------------------------------------------
# Scenario D: variable removal AFTER freezing. Removing a variable renumbers
# dense positions for all later variables. A frozen constraint caches absolute
# positions, so this is the silent-corruption hypothesis.
#
# We remove the FIRST variable (lowest positions) so that a later frozen
# constraint referencing a higher-positioned variable has its cached column
# indices invalidated by the renumbering.
# ---------------------------------------------------------------------------
def _assemble_for_removal(freeze: bool) -> Model:
    m = Model(freeze_constraints=freeze)
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(3, name="i")], name="x")
    m.add_constraints(x >= 1, name="cx")
    y = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="j")], name="y")
    m.add_constraints(y >= 7, name="cy")
    m.add_objective(x.sum() + y.sum())
    return m


def test_D_remove_first_var_matrix_matches_mutable() -> None:
    m_mut = _assemble_for_removal(freeze=False)
    m_frz = _assemble_for_removal(freeze=True)
    # Removing x also removes cx (references x); cy (on y) must survive intact.
    m_mut.remove_variables("x")
    m_frz.remove_variables("x")
    A_mut = _dense_A(m_mut)
    A_frz = _dense_A(m_frz)
    assert A_mut.shape == A_frz.shape
    np.testing.assert_array_equal(A_mut, A_frz)


def test_D_remove_first_var_solves_equal() -> None:
    if "highs" not in linopy.available_solvers:
        pytest.skip("highs unavailable")
    m_mut = _assemble_for_removal(freeze=False)
    m_frz = _assemble_for_removal(freeze=True)
    m_mut.remove_variables("x")
    m_frz.remove_variables("x")
    m_mut.solve(solver_name="highs")
    m_frz.solve(solver_name="highs")
    obj_mut = m_mut.objective.value
    obj_frz = m_frz.objective.value
    assert obj_mut is not None and obj_frz is not None
    assert obj_frz == pytest.approx(obj_mut)


# ---------------------------------------------------------------------------
# Scenario E: a surviving frozen constraint whose terms STRADDLE the removed
# variable (references a variable before it and one after it). Exercises the
# column remap within a single CSR row and the sorted-indices assumption.
# ---------------------------------------------------------------------------
def _assemble_straddle(freeze: bool) -> Model:
    m = Model(freeze_constraints=freeze)
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="x")
    y = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="y")
    z = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="z")
    # constraint links x and z (straddles y), frozen while all three exist
    m.add_constraints(x + z >= 4, name="cxz")
    m.add_objective(x.sum() + y.sum() + z.sum())
    return m


def test_E_straddle_remove_middle_matches_mutable() -> None:
    m_mut = _assemble_straddle(freeze=False)
    m_frz = _assemble_straddle(freeze=True)
    m_mut.remove_variables("y")  # y does not appear in cxz -> cxz survives
    m_frz.remove_variables("y")
    A_mut = _dense_A(m_mut)
    A_frz = _dense_A(m_frz)
    assert A_mut.shape == A_frz.shape
    np.testing.assert_array_equal(A_mut, A_frz)
    # sanity: CSR indices remain sorted within rows after remap
    A_frz_sp = m_frz.matrices.A
    assert A_frz_sp is not None
    dense_before = np.asarray(A_frz_sp.todense())
    A_frz_sp.sort_indices()
    np.testing.assert_array_equal(np.asarray(A_frz_sp.todense()), dense_before)


# ---------------------------------------------------------------------------
# Scenario F: multiple sequential removals, each renumbering again.
# ---------------------------------------------------------------------------
def _assemble_three(freeze: bool) -> Model:
    m = Model(freeze_constraints=freeze)
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="x")
    m.add_constraints(x >= 1, name="cx")
    y = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="y")
    m.add_constraints(y >= 2, name="cy")
    z = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="z")
    m.add_constraints(z >= 3, name="cz")
    m.add_objective(x.sum() + y.sum() + z.sum())
    return m


def test_F_multiple_sequential_removals_match_mutable() -> None:
    m_mut = _assemble_three(freeze=False)
    m_frz = _assemble_three(freeze=True)
    for name in ("x", "y"):  # remove two, leaving only z + cz
        m_mut.remove_variables(name)
        m_frz.remove_variables(name)
        np.testing.assert_array_equal(_dense_A(m_mut), _dense_A(m_frz))


# ---------------------------------------------------------------------------
# Scenario G: interleave removal and addition (remove, then add a new var and
# a new frozen constraint that widens again).
# ---------------------------------------------------------------------------
def test_G_remove_then_add_matches_mutable() -> None:
    def build(freeze: bool) -> Model:
        m = Model(freeze_constraints=freeze)
        x = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="x")
        m.add_constraints(x >= 1, name="cx")
        y = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="y")
        m.add_constraints(y >= 2, name="cy")
        m.remove_variables("x")  # renumber: cy remapped
        w = m.add_variables(lower=0, coords=[pd.RangeIndex(3, name="i")], name="w")
        m.add_constraints(w >= 5, name="cw")  # widen again
        m.add_objective(y.sum() + w.sum())
        return m

    np.testing.assert_array_equal(_dense_A(build(False)), _dense_A(build(True)))


# ---------------------------------------------------------------------------
# Scenario H: removing a variable also drops a frozen constraint referencing it
# even when that constraint references surviving variables too (parity check).
# ---------------------------------------------------------------------------
def test_H_removal_drops_referencing_frozen_constraint() -> None:
    def build(freeze: bool) -> Model:
        m = Model(freeze_constraints=freeze)
        x = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="x")
        y = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="y")
        m.add_constraints(x + y >= 3, name="cxy")
        m.add_constraints(y >= 1, name="cy")
        m.add_objective(x.sum() + y.sum())
        return m

    m_mut = build(False)
    m_frz = build(True)
    m_mut.remove_variables("x")  # drops cxy in both; cy survives
    m_frz.remove_variables("x")
    assert set(m_mut.constraints) == set(m_frz.constraints)
    np.testing.assert_array_equal(_dense_A(m_mut), _dense_A(m_frz))


# ---------------------------------------------------------------------------
# Scenario I: shrink -- remove the LAST-added variable array, so the new basis
# is a proper prefix of the stored one. A pure "widen if too narrow" check
# misses this case, and a naive elementwise basis comparison would compare
# arrays of unequal length.
# ---------------------------------------------------------------------------
def test_I_remove_last_var_shrinks_matrix() -> None:
    def build(freeze: bool) -> Model:
        m = Model(freeze_constraints=freeze)
        x = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="x")
        y = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="y")
        m.add_constraints(x >= 1, name="cx")  # frozen with width covering x and y
        m.add_objective(x.sum() + y.sum())
        return m

    m_mut = build(False)
    m_frz = build(True)
    m_mut.remove_variables("y")  # cx survives; basis shrinks
    m_frz.remove_variables("y")
    A_mut = _dense_A(m_mut)
    A_frz = _dense_A(m_frz)
    assert A_frz.shape == (2, 2)
    np.testing.assert_array_equal(A_mut, A_frz)


# ---------------------------------------------------------------------------
# Scenario J: non-matrix consumers after basis changes. The CSR's column
# indices are interpreted against the current variable basis not only when
# assembling the constraint matrix, but also by ``.data`` (Dataset
# reconstruction), ``to_polars`` (LP export), ``iterate_slices``,
# ``has_variable``, and ``repr``. All must decode to the same variable labels
# as the mutable path after variables were added and removed.
# ---------------------------------------------------------------------------
def _assemble_straddle_then_mutate(freeze: bool) -> Model:
    m = _assemble_straddle(freeze)
    m.remove_variables("y")  # remap: z's positions shift down
    m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="w")  # widen
    return m


def test_J_data_reconstruction_matches_mutable() -> None:
    m_mut = _assemble_straddle_then_mutate(freeze=False)
    m_frz = _assemble_straddle_then_mutate(freeze=True)
    con_mut = m_mut.constraints["cxz"]
    con_frz = m_frz.constraints["cxz"]
    assert isinstance(con_frz, CSRConstraint)
    ds_mut = con_mut.data
    ds_frz = con_frz.data
    np.testing.assert_array_equal(ds_mut.vars.values, ds_frz.vars.values)
    np.testing.assert_array_equal(ds_mut.coeffs.values, ds_frz.coeffs.values)
    np.testing.assert_array_equal(ds_mut.labels.values, ds_frz.labels.values)


def test_J_to_polars_decodes_current_labels() -> None:
    m = _assemble_straddle_then_mutate(freeze=True)
    con = m.constraints["cxz"]
    assert isinstance(con, CSRConstraint)
    df = con.to_polars()
    expected_vars = np.stack(
        [
            m.variables["x"].labels.values,
            m.variables["z"].labels.values,
        ]
    ).T.ravel()  # per row: x[i], z[i]
    np.testing.assert_array_equal(df["vars"].to_numpy(), expected_vars)


def test_J_iterate_slices_decodes_current_labels() -> None:
    m = _assemble_straddle_then_mutate(freeze=True)
    con = m.constraints["cxz"]
    assert isinstance(con, CSRConstraint)
    dfs = [s.to_polars() for s in con.iterate_slices(slice_size=2)]
    assert len(dfs) > 1  # actually sliced
    vars_concat = np.concatenate([df["vars"].to_numpy() for df in dfs])
    expected_vars = np.stack(
        [
            m.variables["x"].labels.values,
            m.variables["z"].labels.values,
        ]
    ).T.ravel()
    np.testing.assert_array_equal(vars_concat, expected_vars)


def test_J_has_variable_after_mutation() -> None:
    m = _assemble_straddle_then_mutate(freeze=True)
    con = m.constraints["cxz"]
    assert isinstance(con, CSRConstraint)
    assert con.has_variable(m.variables["x"])
    assert con.has_variable(m.variables["z"])
    assert not con.has_variable(m.variables["w"])


def test_J_repr_after_mutation_matches_mutable_body() -> None:
    m_mut = _assemble_straddle_then_mutate(freeze=False)
    m_frz = _assemble_straddle_then_mutate(freeze=True)
    r_mut = repr(m_mut.constraints["cxz"])
    r_frz = repr(m_frz.constraints["cxz"])
    for snippet in ("x[0]", "z[0]", "x[1]", "z[1]"):
        assert snippet in r_frz
    # the expression bodies (everything after the header) agree
    assert r_mut.split("\n")[2:] == r_frz.split("\n")[2:]


# ---------------------------------------------------------------------------
# Scenario K: netcdf round-trip after basis changes. Serialization reconciles
# first, so the stored column indices always match the variables stored
# alongside, and the loaded model adopts the load-time basis.
# ---------------------------------------------------------------------------
def test_K_netcdf_roundtrip_after_mutation(tmp_path: Path) -> None:
    m = _assemble_straddle_then_mutate(freeze=True)
    fn = tmp_path / "frozen_mutated.nc"
    m.to_netcdf(fn)
    p = read_netcdf(fn)
    assert isinstance(p.constraints["cxz"], CSRConstraint)
    np.testing.assert_array_equal(_dense_A(m), _dense_A(p))


# ---------------------------------------------------------------------------
# Scenario L: a dangling reference (variable removed behind the model's back,
# bypassing Model.remove_variables) is detected and reported instead of
# producing a silently wrong matrix.
# ---------------------------------------------------------------------------
def test_L_dangling_reference_raises() -> None:
    m = Model(freeze_constraints=True)
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="x")
    m.add_constraints(x >= 1, name="cx")
    m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="y")
    m.variables.remove("x")  # bypasses constraint cleanup on purpose
    con = m.constraints["cx"]
    assert isinstance(con, CSRConstraint)
    with pytest.raises(ValueError, match="no longer part of the model"):
        con.to_matrix()


# ---------------------------------------------------------------------------
# Scenario M: masked variables. Masked entries hold label -1, so dense
# positions and labels diverge; decoding CSR columns as labels without going
# through the basis is wrong for any masked model.
# ---------------------------------------------------------------------------
def _assemble_masked(freeze: bool) -> Model:
    m = Model(freeze_constraints=freeze)
    mask = xr.DataArray([False, True, True], coords=[pd.RangeIndex(3, name="i")])
    x = m.add_variables(
        lower=0, coords=[pd.RangeIndex(3, name="i")], name="x", mask=mask
    )
    m.add_constraints(x >= 1, name="cx")
    y = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="j")], name="y")
    m.add_constraints(y >= 2, name="cy")
    m.add_objective(x.sum() + y.sum())
    return m


def test_M_masked_matrix_matches_mutable() -> None:
    np.testing.assert_array_equal(
        _dense_A(_assemble_masked(False)), _dense_A(_assemble_masked(True))
    )


def test_M_masked_repr_decodes_labels() -> None:
    m = _assemble_masked(freeze=True)
    r = repr(m.constraints["cx"])
    assert "x[1]" in r
    assert "x[2]" in r


# ---------------------------------------------------------------------------
# Scenario N: an empty CSR (all rows masked away) survives additions and
# removals of other variables.
# ---------------------------------------------------------------------------
def test_N_empty_csr_reconciles() -> None:
    def build(freeze: bool) -> Model:
        m = Model(freeze_constraints=freeze)
        x = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="x")
        mask = xr.DataArray([False, False], coords=[pd.RangeIndex(2, name="i")])
        m.add_constraints(x >= 1, name="c_empty", mask=mask)
        y = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="y")
        m.add_constraints(y >= 2, name="cy")
        m.add_objective(x.sum() + y.sum())
        return m

    m_mut = build(False)
    m_frz = build(True)
    m_mut.remove_variables("x")
    m_frz.remove_variables("x")
    np.testing.assert_array_equal(_dense_A(m_mut), _dense_A(m_frz))


# ---------------------------------------------------------------------------
# Scenario O: whitebox -- reconcile caching and copy-on-write semantics.
# ---------------------------------------------------------------------------
def test_O_reconcile_caches_and_rebinds() -> None:
    m = Model(freeze_constraints=True)
    x = m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="x")
    m.add_constraints(x >= 1, name="cx")
    con = m.constraints["cx"]
    assert isinstance(con, CSRConstraint)

    csr1 = con._reconciled_csr()
    assert con._reconciled_csr() is csr1  # fast path: same basis, same object

    m.add_variables(lower=0, coords=[pd.RangeIndex(2, name="i")], name="y")
    csr2 = con._reconciled_csr()
    assert csr2.shape[1] == m.variables.label_index.n_active_vars  # widened
    # copy-on-write: the previously returned csr object is not mutated
    assert csr1.shape[1] == 2
    np.testing.assert_array_equal(csr1.todense(), csr2.todense()[:, :2])
    # basis reference updated -> subsequent accesses take the fast path again
    assert con._vlabels is m.variables.label_index.vlabels
    assert con._reconciled_csr() is csr2
