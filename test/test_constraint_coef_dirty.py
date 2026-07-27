from __future__ import annotations

import pytest

from linopy import Model


@pytest.fixture
def m_with_c() -> tuple[Model, str]:
    m = Model()
    x = m.add_variables(0, 10, coords=[range(3)], name="x")
    y = m.add_variables(0, 10, coords=[range(3)], name="y")
    m.add_constraints(2 * x + y >= 5, name="c")
    return m, "c"


def test_initial_coef_dirty_false(m_with_c: tuple[Model, str]) -> None:
    m, name = m_with_c
    assert m.constraints[name]._coef_dirty is False


def test_update_coeffs_sets_dirty(m_with_c: tuple[Model, str]) -> None:
    m, name = m_with_c
    c = m.constraints[name]
    c.update(coeffs=c.coeffs * 2)
    assert c._coef_dirty is True


def test_update_variables_sets_dirty(m_with_c: tuple[Model, str]) -> None:
    m, name = m_with_c
    c = m.constraints[name]
    x = m.variables["x"]
    c.update(variables=x)
    assert c._coef_dirty is True


def test_update_lhs_sets_dirty(m_with_c: tuple[Model, str]) -> None:
    m, name = m_with_c
    c = m.constraints[name]
    x = m.variables["x"]
    c.update(lhs=3 * x)
    assert c._coef_dirty is True


def test_update_pure_constant_rhs_short_circuits(m_with_c: tuple[Model, str]) -> None:
    m, name = m_with_c
    c = m.constraints[name]
    coeffs_buf = c.data["coeffs"].values
    vars_buf = c.data["vars"].values
    c.update(rhs=9)
    assert c._coef_dirty is False
    assert c.data["coeffs"].values is coeffs_buf
    assert c.data["vars"].values is vars_buf


def test_update_rhs_with_variable_sets_dirty(m_with_c: tuple[Model, str]) -> None:
    m, name = m_with_c
    c = m.constraints[name]
    x = m.variables["x"]
    c.update(rhs=x + 3)
    assert c._coef_dirty is True


def test_update_sign_does_not_set_dirty(m_with_c: tuple[Model, str]) -> None:
    m, name = m_with_c
    c = m.constraints[name]
    c.update(sign="<=")
    assert c._coef_dirty is False


def test_flag_persists_across_container_access(m_with_c: tuple[Model, str]) -> None:
    m, name = m_with_c
    m.constraints[name].update(coeffs=m.constraints[name].coeffs * 2)
    assert m.constraints[name]._coef_dirty is True


def test_update_positional_constraint_sets_dirty(m_with_c: tuple[Model, str]) -> None:
    """Positional ``c.update(expr <= rhs)`` rewrites lhs and must flip the flag."""
    m, name = m_with_c
    c = m.constraints[name]
    x = m.variables["x"]
    c.update(4 * x >= 7)
    assert c._coef_dirty is True


def test_update_noop_does_not_set_dirty(m_with_c: tuple[Model, str]) -> None:
    """``c.update()`` with no args is a no-op and must not flip the flag."""
    m, name = m_with_c
    c = m.constraints[name]
    c.update()
    assert c._coef_dirty is False
