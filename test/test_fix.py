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


class TestVariableFixRelax:
    def test_fix_relax_binary(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["z"].fix(relax=True)
        # Should be relaxed to continuous
        assert not m.variables["z"].attrs["binary"]
        assert not m.variables["z"].attrs["integer"]
        assert "z" in m._relaxed_registry
        assert m._relaxed_registry["z"] == "binary"

    def test_fix_relax_integer(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["w"].fix(relax=True)
        assert not m.variables["w"].attrs["integer"]
        assert not m.variables["w"].attrs["binary"]
        assert "w" in m._relaxed_registry
        assert m._relaxed_registry["w"] == "integer"

    def test_unfix_restores_binary(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["z"].fix(relax=True)
        m.variables["z"].unfix()
        assert m.variables["z"].attrs["binary"]
        assert "z" not in m._relaxed_registry

    def test_unfix_restores_integer(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["w"].fix(relax=True)
        m.variables["w"].unfix()
        assert m.variables["w"].attrs["integer"]
        assert "w" not in m._relaxed_registry

    def test_fix_relax_continuous_noop(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["x"].fix(relax=True)
        # Continuous variable should not be in registry
        assert "x" not in m._relaxed_registry


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

    def test_fix_relax_integers(self, model_with_solution: Model) -> None:
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
    def test_remove_fixed_variable(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["x"].fix(value=5.0)
        m.remove_variables("x")
        assert f"{FIX_CONSTRAINT_PREFIX}x" not in m.constraints

    def test_remove_relaxed_variable(self, model_with_solution: Model) -> None:
        m = model_with_solution
        m.variables["z"].fix(relax=True)
        m.remove_variables("z")
        assert "z" not in m._relaxed_registry
        assert f"{FIX_CONSTRAINT_PREFIX}z" not in m.constraints


class TestFixIO:
    def test_relaxed_registry_survives_netcdf(
        self, model_with_solution: Model, tmp_path: Path
    ) -> None:
        m = model_with_solution
        m.variables["z"].fix(relax=True)
        m.variables["w"].fix(relax=True)

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

    def test_unfix_after_roundtrip(
        self, model_with_solution: Model, tmp_path: Path
    ) -> None:
        m = model_with_solution
        m.variables["z"].fix(relax=True)

        path = tmp_path / "model.nc"
        m.to_netcdf(path)

        from linopy.io import read_netcdf

        m2 = read_netcdf(path)
        m2.variables["z"].unfix()
        assert m2.variables["z"].attrs["binary"]
        assert "z" not in m2._relaxed_registry
        assert f"{FIX_CONSTRAINT_PREFIX}z" not in m2.constraints
