import numpy as np
from numpy import nan

from linopy.common import values_to_lookup_array
from linopy.solvers import _solution_from_names


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


class TestSolutionFromNames:
    def test_default_names(self) -> None:
        arr = _solution_from_names(
            np.array([1.0, 2.0, 3.0]), ["x2", "x0", "x1"], size=4
        )
        np.testing.assert_array_equal(arr[:3], [2.0, 3.0, 1.0])
        assert np.isnan(arr[3])

    def test_explicit_coordinate_names(self) -> None:
        arr = _solution_from_names(
            np.array([1.0, 2.0]), ["power[1]#5", "power[0]#3"], size=7
        )
        assert arr[3] == 2.0
        assert arr[5] == 1.0
        assert np.isnan(arr[4])
