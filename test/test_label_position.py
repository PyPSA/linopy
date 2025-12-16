"""
Tests for LabelPositionIndex and get_label_position optimization.

This module tests:
1. Correctness of optimized implementation vs original
2. Cache invalidation when adding/removing variables and constraints
3. Edge cases (single lookups, batch lookups, 2D arrays)
"""

import numpy as np
import pytest

from linopy import Model
from linopy.common import (
    LabelPositionIndex,
    _get_label_position_linear,
)


@pytest.fixture
def model() -> Model:
    """Create a test model with variables and constraints."""
    m = Model()
    x = m.add_variables(lower=0, upper=10, name="x", coords=[range(5), range(3)])
    y = m.add_variables(lower=-5, upper=5, name="y", coords=[range(4)])
    z = m.add_variables(lower=0, upper=100, name="z", coords=[range(2), range(2)])

    m.add_constraints(x.sum() >= 0, name="con_x")
    m.add_constraints(y.sum() <= 10, name="con_y")
    m.add_constraints(z.sum() == 5, name="con_z")

    return m


@pytest.fixture
def large_model() -> Model:
    """Create a larger model for performance-sensitive tests."""
    rng = np.random.default_rng(42)
    m = Model()

    x = m.add_variables(lower=0, upper=100, name="x", coords=[range(20), range(20)])
    y = m.add_variables(lower=-50, upper=50, name="y", coords=[range(20), range(20)])

    for i in range(100):
        idx1, idx2 = rng.integers(0, 20, size=2)
        m.add_constraints(
            x.isel(dim_0=idx1, dim_1=idx2) + y.isel(dim_0=idx1, dim_1=idx2) >= 0,
            name=f"con{i}",
        )

    return m


class TestLabelPositionIndex:
    """Tests for the LabelPositionIndex class."""

    def test_index_creation(self, model: Model) -> None:
        """Test that index can be created from variables/constraints."""
        var_index = LabelPositionIndex(model.variables)
        con_index = LabelPositionIndex(model.constraints)

        # Index should be lazy - not built until first use
        assert not var_index._built
        assert not con_index._built

    def test_index_build_on_first_use(self, model: Model) -> None:
        """Test that index is built on first lookup."""
        var_index = LabelPositionIndex(model.variables)
        assert not var_index._built

        # Trigger build
        var_index.find_single(0)
        assert var_index._built
        assert var_index._starts is not None
        assert var_index._names is not None

    def test_index_invalidation(self, model: Model) -> None:
        """Test that invalidate() clears the index."""
        var_index = LabelPositionIndex(model.variables)
        var_index.find_single(0)  # Build index
        assert var_index._built

        var_index.invalidate()
        assert not var_index._built
        assert var_index._starts is None
        assert var_index._names is None

    def test_find_single_returns_correct_result(self, model: Model) -> None:
        """Test that find_single returns correct name and coordinates."""
        var_index = LabelPositionIndex(model.variables)

        # Test first variable (x)
        name, coord = var_index.find_single(0)
        assert name == "x"
        assert coord is not None
        assert "dim_0" in coord
        assert "dim_1" in coord

        # Verify by checking the actual label
        actual_label = int(model.variables["x"].labels.sel(coord).values)
        assert actual_label == 0

    def test_find_single_with_minus_one(self, model: Model) -> None:
        """Test that -1 returns (None, None)."""
        var_index = LabelPositionIndex(model.variables)
        name, coord = var_index.find_single(-1)
        assert name is None
        assert coord is None

    def test_find_single_invalid_label(self, model: Model) -> None:
        """Test that invalid labels raise ValueError."""
        var_index = LabelPositionIndex(model.variables)
        with pytest.raises(ValueError, match="not existent"):
            var_index.find_single(99999)

    def test_find_single_with_index(self, model: Model) -> None:
        """Test that find_single_with_index returns correct name, coord, and index."""
        var_index = LabelPositionIndex(model.variables)

        # Test first variable (x)
        name, coord, index = var_index.find_single_with_index(0)
        assert name == "x"
        assert coord is not None
        assert "dim_0" in coord
        assert "dim_1" in coord
        assert isinstance(index, tuple)

        # Verify index can be used for direct numpy access
        var = model.variables["x"]
        label_via_index = var.labels.values[index]
        assert label_via_index == 0

        # Verify coord matches index
        label_via_coord = int(var.labels.sel(coord).values)
        assert label_via_coord == label_via_index

    def test_find_single_with_index_minus_one(self, model: Model) -> None:
        """Test that find_single_with_index returns (None, None, None) for -1."""
        var_index = LabelPositionIndex(model.variables)
        name, coord, index = var_index.find_single_with_index(-1)
        assert name is None
        assert coord is None
        assert index is None


class TestGetLabelPositionOptimized:
    """Tests for the get_label_position_optimized function."""

    def test_single_int_lookup(self, model: Model) -> None:
        """Test single integer lookup."""
        result = model.variables.get_label_position(0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        name, coord = result
        assert name in ["x", "y", "z"]
        assert isinstance(coord, dict)

    def test_single_numpy_int_lookup(self, model: Model) -> None:
        """Test single numpy integer lookup."""
        label = np.int64(0)
        result = model.variables.get_label_position(int(label))
        assert isinstance(result, tuple)

    def test_1d_array_lookup(self, model: Model) -> None:
        """Test 1D array lookup."""
        labels = np.array([0, 1, 2, 3, 4])
        results = model.variables.get_label_position(labels)

        assert isinstance(results, list)
        assert len(results) == 5
        for name, coord in results:
            assert name in ["x", "y", "z"]
            assert isinstance(coord, dict)

    def test_2d_array_lookup(self, model: Model) -> None:
        """Test 2D array lookup."""
        labels = np.array([[0, 1], [2, 3], [4, 5]])
        results = model.variables.get_label_position(labels)

        # Results should be list of lists (column-major)
        assert isinstance(results, list)
        assert len(results) == 2  # 2 columns
        for col in results:
            assert isinstance(col, list)
            assert len(col) == 3  # 3 rows

    def test_matches_original_implementation(self, model: Model) -> None:
        """Test that optimized matches original implementation."""
        rng = np.random.default_rng(123)
        max_label = model._xCounter
        labels = rng.integers(0, max_label, size=50)

        for label in labels:
            original = _get_label_position_linear(model.variables, int(label))
            optimized = model.variables.get_label_position(int(label))
            assert original == optimized, f"Mismatch for label {label}"

    def test_matches_original_batch(self, model: Model) -> None:
        """Test that batch lookup matches original."""
        labels = np.arange(min(20, model._xCounter))

        original = _get_label_position_linear(model.variables, labels)
        optimized = model.variables.get_label_position(labels)

        assert original == optimized

    def test_constraint_lookup(self, model: Model) -> None:
        """Test constraint label lookup."""
        max_label = model._cCounter
        for label in range(max_label):
            result = model.constraints.get_label_position(label)
            assert isinstance(result, tuple) and len(result) == 2
            name, coord = result
            assert name is not None and coord is not None
            assert name in ["con_x", "con_y", "con_z"]

            # Verify correctness
            actual_label = int(model.constraints[name].labels.sel(coord).values)
            assert actual_label == label


class TestCacheInvalidation:
    """Tests for cache invalidation when model is modified."""

    def test_cache_invalidated_on_add_variable(self, model: Model) -> None:
        """Test that cache is invalidated when adding a variable."""
        # Trigger cache build
        _ = model.variables.get_label_position(0)
        assert model.variables._label_position_index is not None
        assert model.variables._label_position_index._built

        # Add new variable
        old_max = model._xCounter
        model.add_variables(lower=0, upper=1, name="new_var", coords=[range(3)])

        # Cache should be invalidated
        assert model.variables._label_position_index is not None
        assert not model.variables._label_position_index._built

        # New variable should be findable
        name, coord = model.variables.get_label_position(old_max)
        assert name == "new_var"

    def test_cache_invalidated_on_remove_variable(self, model: Model) -> None:
        """Test that cache is invalidated when removing a variable."""
        # Trigger cache build
        _ = model.variables.get_label_position(0)
        assert model.variables._label_position_index is not None
        assert model.variables._label_position_index._built

        # Remove variable
        model.variables.remove("z")

        # Cache should be invalidated
        assert model.variables._label_position_index is not None
        assert not model.variables._label_position_index._built

    def test_cache_invalidated_on_add_constraint(self, model: Model) -> None:
        """Test that cache is invalidated when adding a constraint."""
        # Trigger cache build
        _ = model.constraints.get_label_position(0)
        assert model.constraints._label_position_index is not None
        assert model.constraints._label_position_index._built

        # Add new constraint
        old_max = model._cCounter
        x = model.variables["x"]
        model.add_constraints(x.isel(dim_0=0, dim_1=0) >= -1, name="new_con")

        # Cache should be invalidated
        assert model.constraints._label_position_index is not None
        assert not model.constraints._label_position_index._built

        # New constraint should be findable
        name, coord = model.constraints.get_label_position(old_max)
        assert name == "new_con"

    def test_cache_invalidated_on_remove_constraint(self, model: Model) -> None:
        """Test that cache is invalidated when removing a constraint."""
        # Trigger cache build
        _ = model.constraints.get_label_position(0)
        assert model.constraints._label_position_index is not None
        assert model.constraints._label_position_index._built

        # Remove constraint
        model.constraints.remove("con_z")

        # Cache should be invalidated
        assert model.constraints._label_position_index is not None
        assert not model.constraints._label_position_index._built

    def test_repeated_add_remove_cycle(self, model: Model) -> None:
        """Test that cache handles repeated add/remove cycles."""
        for i in range(5):
            # Add variable
            model.add_variables(
                lower=0, upper=1, name=f"temp_var_{i}", coords=[range(2)]
            )

            # Lookup should work
            result = model.variables.get_label_position(0)
            assert result is not None

            # Remove variable
            model.variables.remove(f"temp_var_{i}")

            # Lookup should still work
            result = model.variables.get_label_position(0)
            assert result is not None


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_model(self) -> None:
        """Test behavior with no variables/constraints."""
        m = Model()
        # Should not raise, but nothing to look up
        assert len(list(m.variables)) == 0
        assert len(list(m.constraints)) == 0

    def test_single_variable(self) -> None:
        """Test with a single variable."""
        m = Model()
        m.add_variables(lower=0, upper=1, name="x")

        name, coord = m.variables.get_label_position(0)
        assert name == "x"

    def test_single_constraint(self) -> None:
        """Test with a single constraint."""
        m = Model()
        x = m.add_variables(lower=0, upper=1, name="x")
        m.add_constraints(x >= 0, name="con")

        name, coord = m.constraints.get_label_position(0)
        assert name == "con"

    def test_large_model_correctness(self, large_model: Model) -> None:
        """Test correctness on larger model."""
        rng = np.random.default_rng(456)

        # Test variables
        max_var = large_model._xCounter
        for label in rng.integers(0, max_var, size=100):
            original = _get_label_position_linear(large_model.variables, int(label))
            optimized = large_model.variables.get_label_position(int(label))
            assert original == optimized

        # Test constraints
        max_con = large_model._cCounter
        for label in rng.integers(0, max_con, size=100):
            original = _get_label_position_linear(large_model.constraints, int(label))
            optimized = large_model.constraints.get_label_position(int(label))
            assert original == optimized

    def test_scalar_array_input(self, model: Model) -> None:
        """Test with 0-dimensional numpy array."""
        label = np.array(0)  # 0-d array
        result = model.variables.get_label_position(label)
        assert isinstance(result, tuple)
        assert len(result) == 2
