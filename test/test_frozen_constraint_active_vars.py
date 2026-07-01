"""
Differential tests for frozen (CSR-backed) constraints: adding variables.

The invariant under test (stated in PR #630): ``to_matrix`` on both the mutable
``Constraint`` and the frozen ``CSRConstraint`` "always returns a csr matrix of
shape (n_active_constraints, n_active_variables)". A frozen constraint caches
its CSR at freeze time, so adding variables *after* freezing must still yield a
matrix consistent with the mutable path.

Strategy: build the same model twice under an identical construction order --
once with ``freeze_constraints=False`` (mutable, treated as ground truth) and
once with ``freeze_constraints=True`` -- and assert the assembled constraint
matrix, shape, and solved objective agree. Any divergence is a frozen-path bug.
"""

import numpy as np
import pandas as pd
import pytest

import linopy
from linopy import Model


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
