import numpy as np
import pandas as pd
from numpy import nan

from linopy.common import lookup_vals, series_to_lookup_array


class TestSeriesToLookupArray:
    def test_basic(self) -> None:
        s = pd.Series([10.0, 20.0, 30.0], index=pd.Index([0, 1, 2]))
        arr = series_to_lookup_array(s)
        np.testing.assert_array_equal(arr, [10.0, 20.0, 30.0])

    def test_with_negative_index(self) -> None:
        s = pd.Series([nan, 10.0, 20.0], index=pd.Index([-1, 0, 2]))
        arr = series_to_lookup_array(s)
        assert arr[0] == 10.0
        assert np.isnan(arr[1])
        assert arr[2] == 20.0

    def test_sparse_index(self) -> None:
        s = pd.Series([5.0, 7.0], index=pd.Index([0, 100]))
        arr = series_to_lookup_array(s)
        assert len(arr) == 101
        assert arr[0] == 5.0
        assert arr[100] == 7.0
        assert np.isnan(arr[50])

    def test_only_negative_index(self) -> None:
        s = pd.Series([nan], index=pd.Index([-1]))
        arr = series_to_lookup_array(s)
        assert len(arr) == 1
        assert np.isnan(arr[0])


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
