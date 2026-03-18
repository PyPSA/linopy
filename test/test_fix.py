"""Tests for Variable.fix(), Variable.unfix(), and Variable.fixed."""

import numpy as np
import pandas as pd
import pytest
from xarray import DataArray

from linopy import Model
from linopy.variables import FIX_CONSTRAINT_PREFIX


@pytest.fixture
def model_with_solution():
    """Create a simple model and simulate a solution."""
    m = Model()
    x = m.add_variables(lower=0, upper=10, name="x")
    y = m.add_variables(lower=-5, upper=5, coords=[pd.Index([0, 1])], name="y")
    z = m.add_variables(binary=True, name="z")
    w = m.add_variables(lower=0, upper=100, integer=True, name="w")

    # Simulate solution values
    x.solution = 3.14159265
    y.solution = DataArray([2.71828, -1.41421], dims="dim_0")
    z.solution = 0.9999999997
    w.solution = 41.9999999998
    m._status = "ok"
    m._termination_condition = "optimal"

    return m


class TestVariableFix:
    def test_fix_uses_solution(self, model_with_solution):
        m = model_with_solution
        m.variables["x"].fix()
        assert m.variables["x"].fixed
        assert f"{FIX_CONSTRAINT_PREFIX}x" in m.constraints

    def test_fix_with_explicit_value(self, model_with_solution):
        m = model_with_solution
        m.variables["x"].fix(value=5.0)
        assert m.variables["x"].fixed
        con = m.constraints[f"{FIX_CONSTRAINT_PREFIX}x"]
        np.testing.assert_almost_equal(con.rhs.item(), 5.0)

    def test_fix_rounds_binary(self, model_with_solution):
        m = model_with_solution
        m.variables["z"].fix()
        con = m.constraints[f"{FIX_CONSTRAINT_PREFIX}z"]
        # 0.9999999997 should be rounded to 1.0
        np.testing.assert_equal(con.rhs.item(), 1.0)

    def test_fix_rounds_integer(self, model_with_solution):
        m = model_with_solution
        m.variables["w"].fix()
        con = m.constraints[f"{FIX_CONSTRAINT_PREFIX}w"]
        # 41.9999999998 should be rounded to 42.0
        np.testing.assert_equal(con.rhs.item(), 42.0)

    def test_fix_rounds_continuous(self, model_with_solution):
        m = model_with_solution
        m.variables["x"].fix(decimals=4)
        con = m.constraints[f"{FIX_CONSTRAINT_PREFIX}x"]
        np.testing.assert_almost_equal(con.rhs.item(), 3.1416, decimal=4)

    def test_fix_clips_to_upper_bound(self, model_with_solution):
        m = model_with_solution
        m.variables["x"].fix(value=10.0000001)
        con = m.constraints[f"{FIX_CONSTRAINT_PREFIX}x"]
        np.testing.assert_almost_equal(con.rhs.item(), 10.0)

    def test_fix_clips_to_lower_bound(self, model_with_solution):
        m = model_with_solution
        m.variables["x"].fix(value=-0.0000001)
        con = m.constraints[f"{FIX_CONSTRAINT_PREFIX}x"]
        np.testing.assert_almost_equal(con.rhs.item(), 0.0)

    def test_fix_overwrites_existing(self, model_with_solution):
        m = model_with_solution
        m.variables["x"].fix(value=3.0)
        m.variables["x"].fix(value=5.0)
        con = m.constraints[f"{FIX_CONSTRAINT_PREFIX}x"]
        np.testing.assert_almost_equal(con.rhs.item(), 5.0)

    def test_fix_multidimensional(self, model_with_solution):
        m = model_with_solution
        m.variables["y"].fix()
        assert m.variables["y"].fixed
        con = m.constraints[f"{FIX_CONSTRAINT_PREFIX}y"]
        np.testing.assert_array_almost_equal(con.rhs.values, [2.71828, -1.41421])


class TestVariableUnfix:
    def test_unfix_removes_constraint(self, model_with_solution):
        m = model_with_solution
        m.variables["x"].fix(value=5.0)
        m.variables["x"].unfix()
        assert not m.variables["x"].fixed
        assert f"{FIX_CONSTRAINT_PREFIX}x" not in m.constraints

    def test_unfix_noop_if_not_fixed(self, model_with_solution):
        m = model_with_solution
        # Should not raise
        m.variables["x"].unfix()
        assert not m.variables["x"].fixed


class TestVariableFixRelax:
    def test_fix_relax_binary(self, model_with_solution):
        m = model_with_solution
        m.variables["z"].fix(relax=True)
        # Should be relaxed to continuous
        assert not m.variables["z"].attrs["binary"]
        assert not m.variables["z"].attrs["integer"]
        assert "z" in m._relaxed_registry
        assert m._relaxed_registry["z"] == "binary"

    def test_fix_relax_integer(self, model_with_solution):
        m = model_with_solution
        m.variables["w"].fix(relax=True)
        assert not m.variables["w"].attrs["integer"]
        assert not m.variables["w"].attrs["binary"]
        assert "w" in m._relaxed_registry
        assert m._relaxed_registry["w"] == "integer"

    def test_unfix_restores_binary(self, model_with_solution):
        m = model_with_solution
        m.variables["z"].fix(relax=True)
        m.variables["z"].unfix()
        assert m.variables["z"].attrs["binary"]
        assert "z" not in m._relaxed_registry

    def test_unfix_restores_integer(self, model_with_solution):
        m = model_with_solution
        m.variables["w"].fix(relax=True)
        m.variables["w"].unfix()
        assert m.variables["w"].attrs["integer"]
        assert "w" not in m._relaxed_registry

    def test_fix_relax_continuous_noop(self, model_with_solution):
        m = model_with_solution
        m.variables["x"].fix(relax=True)
        # Continuous variable should not be in registry
        assert "x" not in m._relaxed_registry


class TestVariableFixed:
    def test_fixed_false_initially(self, model_with_solution):
        m = model_with_solution
        assert not m.variables["x"].fixed

    def test_fixed_true_after_fix(self, model_with_solution):
        m = model_with_solution
        m.variables["x"].fix(value=5.0)
        assert m.variables["x"].fixed

    def test_fixed_false_after_unfix(self, model_with_solution):
        m = model_with_solution
        m.variables["x"].fix(value=5.0)
        m.variables["x"].unfix()
        assert not m.variables["x"].fixed


class TestVariablesContainerFixUnfix:
    def test_fix_all(self, model_with_solution):
        m = model_with_solution
        m.variables.fix()
        for name in m.variables:
            assert m.variables[name].fixed

    def test_unfix_all(self, model_with_solution):
        m = model_with_solution
        m.variables.fix()
        m.variables.unfix()
        for name in m.variables:
            assert not m.variables[name].fixed

    def test_fix_integers_only(self, model_with_solution):
        m = model_with_solution
        m.variables.integers.fix()
        assert m.variables["w"].fixed
        assert not m.variables["x"].fixed

    def test_fix_binaries_only(self, model_with_solution):
        m = model_with_solution
        m.variables.binaries.fix()
        assert m.variables["z"].fixed
        assert not m.variables["x"].fixed

    def test_fixed_returns_dict(self, model_with_solution):
        m = model_with_solution
        m.variables["x"].fix(value=5.0)
        result = m.variables.fixed
        assert isinstance(result, dict)
        assert result["x"] is True
        assert result["y"] is False

    def test_fix_relax_integers(self, model_with_solution):
        m = model_with_solution
        m.variables.integers.fix(relax=True)
        assert not m.variables["w"].attrs["integer"]
        m.variables.integers.unfix()
        # After unfix from the integers view, the variable should be restored
        # but we need to unfix from the actual variable since integers view
        # won't contain it anymore after relaxation
        # Let's unfix via the model variables directly
        m.variables["w"].unfix()
        assert m.variables["w"].attrs["integer"]


class TestRemoveVariablesCleansUpFix:
    def test_remove_fixed_variable(self, model_with_solution):
        m = model_with_solution
        m.variables["x"].fix(value=5.0)
        m.remove_variables("x")
        assert f"{FIX_CONSTRAINT_PREFIX}x" not in m.constraints

    def test_remove_relaxed_variable(self, model_with_solution):
        m = model_with_solution
        m.variables["z"].fix(relax=True)
        m.remove_variables("z")
        assert "z" not in m._relaxed_registry
        assert f"{FIX_CONSTRAINT_PREFIX}z" not in m.constraints
