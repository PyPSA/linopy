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


def test_coeffs_setter_sets_dirty(m_with_c: tuple[Model, str]) -> None:
    m, name = m_with_c
    c = m.constraints[name]
    c.coeffs = c.coeffs * 2
    assert c._coef_dirty is True


def test_vars_setter_sets_dirty(m_with_c: tuple[Model, str]) -> None:
    m, name = m_with_c
    c = m.constraints[name]
    c.vars = c.vars
    assert c._coef_dirty is True


def test_lhs_setter_sets_dirty(m_with_c: tuple[Model, str]) -> None:
    m, name = m_with_c
    c = m.constraints[name]
    x = m.variables["x"]
    c.lhs = 3 * x
    assert c._coef_dirty is True


def test_pure_constant_rhs_short_circuits(m_with_c: tuple[Model, str]) -> None:
    m, name = m_with_c
    c = m.constraints[name]
    coeffs_buf = c.data["coeffs"].values
    vars_buf = c.data["vars"].values
    c.rhs = 9
    assert c._coef_dirty is False
    assert c.data["coeffs"].values is coeffs_buf
    assert c.data["vars"].values is vars_buf


def test_rhs_with_variable_sets_dirty(m_with_c: tuple[Model, str]) -> None:
    m, name = m_with_c
    c = m.constraints[name]
    x = m.variables["x"]
    c.rhs = x + 3
    assert c._coef_dirty is True


def test_sign_setter_does_not_set_dirty(m_with_c: tuple[Model, str]) -> None:
    m, name = m_with_c
    c = m.constraints[name]
    c.sign = "<="
    assert c._coef_dirty is False


def test_flag_persists_across_container_access(m_with_c: tuple[Model, str]) -> None:
    m, name = m_with_c
    m.constraints[name].coeffs = m.constraints[name].coeffs * 2
    assert m.constraints[name]._coef_dirty is True
