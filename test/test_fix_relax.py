"""Tests for Variable.fix(), Variable.unfix(), and Variable.fixed."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xarray import DataArray

from linopy import Model
from linopy.constants import FIX_CONSTRAINT_PREFIX


@pytest.fixture
def model_with_solution() -> Model:
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


SCALAR_VALUES: list = [
    pytest.param(5, id="int"),
    pytest.param(5.0, id="float"),
    pytest.param(np.float64(5.0), id="np.float64"),
    pytest.param(np.int64(5), id="np.int64"),
    pytest.param(np.array(5.0), id="np.0d-array"),
    pytest.param(DataArray(5.0), id="DataArray"),
]

ARRAY_VALUES: list = [
    pytest.param([2.5, -1.5], id="list"),
    pytest.param(np.array([2.5, -1.5]), id="np.array"),
    pytest.param(DataArray([2.5, -1.5], dims="dim_0"), id="DataArray"),
    pytest.param(pd.Series([2.5, -1.5]), id="pd.Series"),
]


class TestVariableFix:
    @pytest.mark.parametrize("value", SCALAR_VALUES)
    def test_fix_scalar_dtypes(self, model_with_solution: Model, value: object) -> None:
        m = model_with_solution
        m.variables["x"].fix(value=value)
        assert m.variables["x"].fixed
        con = m.constraints[f"{FIX_CONSTRAINT_PREFIX}x"]
        np.testing.assert_almost_equal(con.rhs.item(), 5.0)

    @pytest.mark.parametrize("value", ARRAY_VALUES)
    def test_fix_array_dtypes(self, model_with_solution: Model, value: object) -> None:
        m = model_with_solution
        m.variables["y"].fix(value=value)
        assert m.variables["y"].fixed
        con = m.constraints[f"{FIX_CONSTRAINT_PREFIX}y"]
        np.testing.assert_array_almost_equal(con.rhs.values, [2.5, -1.5])

    def test_fix_uses_solution(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["x"].fix()
        assert m.variables["x"].fixed
        assert f"{FIX_CONSTRAINT_PREFIX}x" in m.constraints

    def test_fix_rounds_binary(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["z"].fix()
        con = m.constraints[f"{FIX_CONSTRAINT_PREFIX}z"]
        np.testing.assert_equal(con.rhs.item(), 1.0)

    def test_fix_rounds_integer(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["w"].fix()
        con = m.constraints[f"{FIX_CONSTRAINT_PREFIX}w"]
        np.testing.assert_equal(con.rhs.item(), 42.0)

    def test_fix_rounds_continuous(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["x"].fix(decimals=4)
        con = m.constraints[f"{FIX_CONSTRAINT_PREFIX}x"]
        np.testing.assert_almost_equal(con.rhs.item(), 3.1416, decimal=4)

    def test_fix_raises_above_upper_bound(self, model_with_solution: Model) -> None:
        m = model_with_solution
        with pytest.raises(ValueError, match="outside the variable bounds"):
            m.variables["x"].fix(value=11.0)

    def test_fix_raises_below_lower_bound(self, model_with_solution: Model) -> None:
        m = model_with_solution
        with pytest.raises(ValueError, match="outside the variable bounds"):
            m.variables["x"].fix(value=-1.0)

    def test_fix_small_overshoot_rounded_within_bounds(
        self, model_with_solution: Model
    ) -> None:
        m = model_with_solution
        m.variables["x"].fix(value=10.0000000001)
        con = m.constraints[f"{FIX_CONSTRAINT_PREFIX}x"]
        np.testing.assert_almost_equal(con.rhs.item(), 10.0)

    def test_fix_raises_if_already_fixed_no_overwrite(
        self, model_with_solution: Model
    ) -> None:
        m = model_with_solution
        m.variables["x"].fix(value=3.0)
        with pytest.raises(ValueError, match="already fixed"):
            m.variables["x"].fix(value=5.0, overwrite=False)

    def test_fix_overwrite_replaces_existing(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["x"].fix(value=3.0)
        m.variables["x"].fix(value=5.0, overwrite=True)
        con = m.constraints[f"{FIX_CONSTRAINT_PREFIX}x"]
        np.testing.assert_almost_equal(con.rhs.item(), 5.0)

    def test_fix_multidimensional(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["y"].fix()
        assert m.variables["y"].fixed
        con = m.constraints[f"{FIX_CONSTRAINT_PREFIX}y"]
        np.testing.assert_array_almost_equal(con.rhs.values, [2.71828, -1.41421])


class TestVariableUnfix:
    def test_unfix_removes_constraint(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["x"].fix(value=5.0)
        m.variables["x"].unfix()
        assert not m.variables["x"].fixed
        assert f"{FIX_CONSTRAINT_PREFIX}x" not in m.constraints

    def test_unfix_noop_if_not_fixed(self, model_with_solution: Model) -> None:
        m = model_with_solution
        # Should not raise
        m.variables["x"].unfix()
        assert not m.variables["x"].fixed


class TestFixThenRelax:
    """Test the combined fix() + relax() workflow (fix first, then relax)."""

    def test_fix_then_relax_binary(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["z"].fix()
        m.variables["z"].relax()
        assert not m.variables["z"].attrs["binary"]
        assert m.variables["z"].fixed
        assert m.variables["z"].relaxed

    def test_unfix_restores_relaxed_binary(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["z"].fix()
        m.variables["z"].relax()
        m.variables["z"].unfix()
        assert m.variables["z"].attrs["binary"]
        assert not m.variables["z"].fixed
        assert not m.variables["z"].relaxed

    def test_fix_then_relax_integer(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["w"].fix()
        m.variables["w"].relax()
        assert not m.variables["w"].attrs["integer"]
        assert m.variables["w"].fixed
        assert m.variables["w"].relaxed

    def test_unfix_restores_relaxed_integer(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["w"].fix()
        m.variables["w"].relax()
        m.variables["w"].unfix()
        assert m.variables["w"].attrs["integer"]
        assert not m.variables["w"].fixed
        assert not m.variables["w"].relaxed


class TestVariableFixed:
    def test_fixed_false_initially(self, model_with_solution: Model) -> None:
        m = model_with_solution
        assert not m.variables["x"].fixed

    def test_fixed_true_after_fix(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["x"].fix(value=5.0)
        assert m.variables["x"].fixed

    def test_fixed_false_after_unfix(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["x"].fix(value=5.0)
        m.variables["x"].unfix()
        assert not m.variables["x"].fixed


class TestVariablesContainerFixUnfix:
    def test_fix_all(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables.fix()
        for name in m.variables:
            assert m.variables[name].fixed

    def test_unfix_all(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables.fix()
        m.variables.unfix()
        for name in m.variables:
            assert not m.variables[name].fixed

    def test_fix_integers_only(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables.integers.fix()
        assert m.variables["w"].fixed
        assert not m.variables["x"].fixed

    def test_fix_binaries_only(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables.binaries.fix()
        assert m.variables["z"].fixed
        assert not m.variables["x"].fixed

    def test_fixed_returns_dict(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["x"].fix(value=5.0)
        result = m.variables.fixed
        assert isinstance(result, dict)
        assert result["x"] is True
        assert result["y"] is False

    def test_fix_then_relax_integers(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables.integers.fix()
        m.variables.integers.relax()
        assert not m.variables["w"].attrs["integer"]
        assert m.variables["w"].fixed
        m.variables["w"].unfix()
        assert m.variables["w"].attrs["integer"]


class TestVariableRelax:
    def test_relax_binary(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["z"].relax()
        assert not m.variables["z"].attrs["binary"]
        assert not m.variables["z"].attrs["integer"]
        assert m.variables["z"].relaxed
        assert m._relaxed_registry["z"] == "binary"

    def test_relax_integer(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["w"].relax()
        assert not m.variables["w"].attrs["integer"]
        assert not m.variables["w"].attrs["binary"]
        assert m.variables["w"].relaxed
        assert m._relaxed_registry["w"] == "integer"

    def test_relax_continuous_noop(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["x"].relax()
        assert "x" not in m._relaxed_registry
        assert not m.variables["x"].relaxed

    def test_relax_semi_continuous_raises(self) -> None:
        m = Model()
        m.add_variables(lower=1, upper=10, semi_continuous=True, name="sc")
        with pytest.raises(NotImplementedError, match="semi-continuous"):
            m.variables["sc"].relax()

    def test_unrelax_binary(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["z"].relax()
        m.variables["z"].unrelax()
        assert m.variables["z"].attrs["binary"]
        assert not m.variables["z"].relaxed
        assert "z" not in m._relaxed_registry

    def test_unrelax_integer(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["w"].relax()
        m.variables["w"].unrelax()
        assert m.variables["w"].attrs["integer"]
        assert not m.variables["w"].relaxed
        assert "w" not in m._relaxed_registry

    def test_unrelax_noop_if_not_relaxed(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["x"].unrelax()
        assert not m.variables["x"].relaxed

    def test_relax_preserves_binary_bounds(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["z"].relax()
        assert float(m.variables["z"].lower) == 0.0
        assert float(m.variables["z"].upper) == 1.0

    def test_relax_preserves_integer_bounds(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["w"].relax()
        assert float(m.variables["w"].lower) == 0.0
        assert float(m.variables["w"].upper) == 100.0


class TestVariablesContainerRelax:
    def test_relax_all(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables.relax()
        assert not m.variables["z"].attrs["binary"]
        assert not m.variables["w"].attrs["integer"]
        assert m.variables["z"].relaxed
        assert m.variables["w"].relaxed
        # Continuous variables unaffected
        assert not m.variables["x"].relaxed

    def test_unrelax_all(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables.relax()
        m.variables.unrelax()
        assert m.variables["z"].attrs["binary"]
        assert m.variables["w"].attrs["integer"]
        assert not m.variables["z"].relaxed
        assert not m.variables["w"].relaxed

    def test_relax_integers_only(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables.integers.relax()
        assert not m.variables["w"].attrs["integer"]
        assert m.variables["w"].relaxed
        # Binary should be untouched
        assert m.variables["z"].attrs["binary"]
        assert not m.variables["z"].relaxed

    def test_relax_binaries_only(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables.binaries.relax()
        assert not m.variables["z"].attrs["binary"]
        assert m.variables["z"].relaxed
        # Integer should be untouched
        assert m.variables["w"].attrs["integer"]
        assert not m.variables["w"].relaxed

    def test_relaxed_returns_dict(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["z"].relax()
        result = m.variables.relaxed
        assert isinstance(result, dict)
        assert result["z"] is True
        assert result["x"] is False

    def test_relax_with_semi_continuous_raises(self) -> None:
        m = Model()
        m.add_variables(lower=0, upper=10, name="x")
        m.add_variables(lower=1, upper=10, semi_continuous=True, name="sc")
        with pytest.raises(NotImplementedError, match="semi-continuous"):
            m.variables.relax()

    def test_relax_all_converts_milp_to_lp(self, model_with_solution: Model) -> None:
        m = model_with_solution
        assert m.type == "MILP"
        m.variables.relax()
        assert m.type == "LP"
        m.variables.unrelax()
        assert m.type == "MILP"


class TestRemoveVariablesCleansUpFix:
    def test_remove_fixed_variable(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["x"].fix(value=5.0)
        m.remove_variables("x")
        assert f"{FIX_CONSTRAINT_PREFIX}x" not in m.constraints

    def test_remove_relaxed_variable(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["z"].fix()
        m.variables["z"].relax()
        m.remove_variables("z")
        assert "z" not in m._relaxed_registry
        assert f"{FIX_CONSTRAINT_PREFIX}z" not in m.constraints


class TestFixIO:
    def test_relaxed_registry_survives_netcdf(
        self, model_with_solution: Model, tmp_path: Path
    ) -> None:
        m = model_with_solution
        m.variables["z"].fix()
        m.variables["z"].relax()
        m.variables["w"].fix()
        m.variables["w"].relax()

        path = tmp_path / "model.nc"
        m.to_netcdf(path)

        from linopy.io import read_netcdf

        m2 = read_netcdf(path)
        assert m2._relaxed_registry == {"z": "binary", "w": "integer"}
        # Fix constraints should also survive
        assert f"{FIX_CONSTRAINT_PREFIX}z" in m2.constraints
        assert f"{FIX_CONSTRAINT_PREFIX}w" in m2.constraints

    def test_empty_registry_netcdf(
        self, model_with_solution: Model, tmp_path: Path
    ) -> None:
        m = model_with_solution
        path = tmp_path / "model.nc"
        m.to_netcdf(path)

        from linopy.io import read_netcdf

        m2 = read_netcdf(path)
        assert m2._relaxed_registry == {}

    def test_unrelax_after_roundtrip(
        self, model_with_solution: Model, tmp_path: Path
    ) -> None:
        m = model_with_solution
        m.variables["z"].relax()

        path = tmp_path / "model.nc"
        m.to_netcdf(path)

        from linopy.io import read_netcdf

        m2 = read_netcdf(path)
        m2.variables["z"].unrelax()
        assert m2.variables["z"].attrs["binary"]
        assert "z" not in m2._relaxed_registry
