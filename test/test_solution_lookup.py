import numpy as np
from numpy import nan

from linopy.common import lookup_vals, values_to_lookup_array


class TestValuesToLookupArray:
    def test_basic(self) -> None:
        arr = values_to_lookup_array(np.array([10.0, 20.0, 30.0]), np.array([0, 1, 2]))
        np.testing.assert_array_equal(arr, [10.0, 20.0, 30.0])

    def test_negative_labels_skipped(self) -> None:
        arr = values_to_lookup_array(np.array([nan, 10.0, 20.0]), np.array([-1, 0, 2]))
        assert arr[0] == 10.0
        assert np.isnan(arr[1])
        assert arr[2] == 20.0

    def test_sparse_labels(self) -> None:
        arr = values_to_lookup_array(np.array([5.0, 7.0]), np.array([0, 100]))
        assert len(arr) == 101
        assert arr[0] == 5.0
        assert arr[100] == 7.0
        assert np.isnan(arr[50])

    def test_only_negative_labels(self) -> None:
        arr = values_to_lookup_array(np.array([nan]), np.array([-1]))
        assert len(arr) == 0

    def test_explicit_size(self) -> None:
        arr = values_to_lookup_array(np.array([5.0, 7.0]), np.array([0, 2]), size=5)
        assert len(arr) == 5
        assert arr[0] == 5.0
        assert arr[2] == 7.0
        assert np.isnan(arr[1])
        assert np.isnan(arr[3])
        assert np.isnan(arr[4])


class TestLookupVals:
    def test_basic(self) -> None:
        arr = np.array([10.0, 20.0, 30.0])
        idx = np.array([0, 1, 2])
        result = lookup_vals(arr, idx)
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0])

    def test_negative_labels_become_nan(self) -> None:
        arr = np.array([10.0, 20.0])
        idx = np.array([0, -1, 1, -1])
        result = lookup_vals(arr, idx)
        assert result[0] == 10.0
        assert np.isnan(result[1])
        assert result[2] == 20.0
        assert np.isnan(result[3])

    def test_out_of_range_labels_become_nan(self) -> None:
        arr = np.array([10.0, 20.0])
        idx = np.array([0, 1, 999])
        result = lookup_vals(arr, idx)
        assert result[0] == 10.0
        assert result[1] == 20.0
        assert np.isnan(result[2])

    def test_all_negative(self) -> None:
        arr = np.array([10.0])
        idx = np.array([-1, -1, -1])
        result = lookup_vals(arr, idx)
        assert all(np.isnan(result))

    def test_no_mutation_of_source(self) -> None:
        arr = np.array([10.0, 20.0, 30.0])
        idx1 = np.array([-1, 1])
        idx2 = np.array([0, 2])
        lookup_vals(arr, idx1)
        result2 = lookup_vals(arr, idx2)
        np.testing.assert_array_equal(result2, [10.0, 30.0])
        np.testing.assert_array_equal(arr, [10.0, 20.0, 30.0])
