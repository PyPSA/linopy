#!/usr/bin/env python3
"""
Tests for linopy.alignment — conversion, broadcasting, and validation of
user input against coordinates.

Organized by the module's public surface:

- ``TestAsDataarrayFrom*``      — :func:`as_dataarray` (convert only)
- ``TestCoordsToDict``          — the coords-entry naming rules
- ``TestAddVariablesCoords``    — coords/dims → variable dims (end-to-end)
- ``TestBroadcastToCoords``     — ``broadcast_to_coords(strict=False)``
- ``TestMultiIndexProjection``  — implicit MI-level projection (values,
  deprecation warnings, coverage gaps) — the legacy/v1 fork point
- ``TestStrictMode``            — ``broadcast_to_coords(strict=True)``
- ``TestValidateAlignment``     — the validation primitive
- ``TestAlign``                 — symmetric :func:`align`
"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
from xarray import DataArray
from xarray.testing.assertions import assert_equal

from linopy import EvolvingAPIWarning, LinearExpression, Model, Variable
from linopy.alignment import (
    _coords_to_dict,
    align,
    as_dataarray,
    broadcast_to_coords,
    fill_missing_coords,
    validate_alignment,
)
from linopy.testing import assert_linequal, assert_varequal
from linopy.types import CoordsLike

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mi_index() -> pd.MultiIndex:
    """Named (level1, level2) MultiIndex backing the stacked dim 'dim_3'."""
    idx = pd.MultiIndex.from_product([[1, 2], ["a", "b"]], names=("level1", "level2"))
    idx.name = "dim_3"
    return idx


@pytest.fixture
def mi_coords(mi_index: pd.MultiIndex) -> xr.Coordinates:
    """Coordinates of the stacked MultiIndex dim 'dim_3'."""
    return xr.Coordinates.from_pandas_multiindex(mi_index, "dim_3")


@pytest.fixture
def by_level1() -> DataArray:
    """A constant indexed by level1 only — a partial level set."""
    return DataArray([10.0, 20.0], coords={"level1": [1, 2]}, dims=["level1"])


# ---------------------------------------------------------------------------
# as_dataarray — convert only
# ---------------------------------------------------------------------------


class TestAsDataarrayFromPandas:
    """Series / DataFrame conversion: pandas axis names vs the dims argument."""

    @pytest.mark.parametrize(
        ("index", "dims", "expected_dim"),
        [
            pytest.param([0, 1, 2], None, "dim_0", id="default"),
            pytest.param(["a", "b", "c"], ["dim1"], "dim1", id="dims-set"),
            pytest.param(
                pd.Index(["a", "b", "c"], name="dim1"), [], "dim1", id="dims-given"
            ),
            pytest.param(
                pd.Index(["a", "b", "c"], name="dim1"),
                ["other"],
                "dim1",
                id="pandas-name-has-priority",
            ),
            pytest.param(["a", "b", "c"], [], "dim_0", id="dims-subset"),
            pytest.param(
                ["a", "b", "c"], ["dim_a", "other"], "dim_a", id="dims-superset"
            ),
        ],
    )
    def test_series_dim_naming(
        self, index: Any, dims: list[str] | None, expected_dim: str
    ) -> None:
        s = pd.Series([1, 2, 3], index=index)
        da = as_dataarray(s, dims=dims) if dims is not None else as_dataarray(s)
        assert isinstance(da, DataArray)
        assert da.dims == (expected_dim,)
        assert list(da.coords[expected_dim].values) == list(s.index)

    @pytest.mark.parametrize(
        ("index", "columns", "dims", "expected_dims"),
        [
            pytest.param([0, 1], ["A", "B"], None, ("dim_0", "dim_1"), id="default"),
            pytest.param(
                ["a", "b"],
                ["A", "B"],
                ("dim1", "dim2"),
                ("dim1", "dim2"),
                id="dims-set",
            ),
            pytest.param(
                pd.Index(["a", "b"], name="dim1"),
                pd.Index(["A", "B"], name="dim2"),
                [],
                ("dim1", "dim2"),
                id="dims-given",
            ),
            pytest.param(
                pd.Index(["a", "b"], name="dim1"),
                pd.Index(["A", "B"], name="dim2"),
                ["other"],
                ("dim1", "dim2"),
                id="pandas-name-has-priority",
            ),
            pytest.param(
                ["a", "b"], ["A", "B"], [], ("dim_0", "dim_1"), id="dims-subset"
            ),
            pytest.param(
                ["a", "b"],
                ["A", "B"],
                ["dim_a", "dim_b", "other"],
                ("dim_a", "dim_b"),
                id="dims-superset",
            ),
        ],
    )
    def test_dataframe_dim_naming(
        self,
        index: Any,
        columns: Any,
        dims: Any,
        expected_dims: tuple[str, ...],
    ) -> None:
        df = pd.DataFrame([[1, 2], [3, 4]], index=index, columns=columns)
        da = as_dataarray(df, dims=dims) if dims is not None else as_dataarray(df)
        assert isinstance(da, DataArray)
        assert da.dims == expected_dims
        assert list(da.coords[expected_dims[0]].values) == list(df.index)
        assert list(da.coords[expected_dims[1]].values) == list(df.columns)

    @pytest.mark.parametrize(
        "coords",
        [[["a", "b", "c"]], {"dim_0": ["a", "b", "c"]}],
        ids=["list", "dict"],
    )
    def test_series_aligned_coords_do_not_warn(self, coords: Any) -> None:
        """Coords matching the pandas index are accepted silently — no misalignment warning."""
        s = pd.Series([1, 2, 3], index=["a", "b", "c"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            da = as_dataarray(s, coords=coords)
        assert da.dims == ("dim_0",)
        assert list(da.coords["dim_0"].values) == ["a", "b", "c"]

    @pytest.mark.parametrize(
        "coords",
        [[["a", "b"], ["A", "B"]], {"dim_0": ["a", "b"], "dim_1": ["A", "B"]}],
        ids=["list", "dict"],
    )
    def test_dataframe_aligned_coords_do_not_warn(self, coords: Any) -> None:
        """Coords matching the frame's index/columns are accepted silently."""
        df = pd.DataFrame([[1, 2], [3, 4]], index=["a", "b"], columns=["A", "B"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            da = as_dataarray(df, coords=coords)
        assert da.dims == ("dim_0", "dim_1")
        assert list(da.coords["dim_0"].values) == ["a", "b"]
        assert list(da.coords["dim_1"].values) == ["A", "B"]

    def test_polars_series(self) -> None:
        target_dim = "dim_0"
        target_index = [0, 1, 2]
        s = pl.Series([1, 2, 3])
        da = as_dataarray(s)
        assert isinstance(da, DataArray)
        assert da.dims == (target_dim,)
        assert list(da.coords[target_dim].values) == target_index

    def test_series_dims_as_bare_string(self) -> None:
        """Dims may be a single dim name instead of a list."""
        da = as_dataarray(pd.Series([1, 2, 3]), dims="x")
        assert da.dims == ("x",)


class TestAsDataarrayFromNumpy:
    """ndarray conversion: positional labeling from coords / dims."""

    arr = np.array([[1, 2], [3, 4]])

    @pytest.mark.parametrize(
        ("coords", "dims", "expected"),
        [
            pytest.param(
                None, None, {"dim_0": [0, 1], "dim_1": [0, 1]}, id="no-coords-no-dims"
            ),
            pytest.param(
                [["a", "b"], ["A", "B"]],
                None,
                {"dim_0": ["a", "b"], "dim_1": ["A", "B"]},
                id="coords-list",
            ),
            pytest.param(
                [pd.Index(["a", "b"], name="dim1"), pd.Index(["A", "B"], name="dim2")],
                None,
                {"dim1": ["a", "b"], "dim2": ["A", "B"]},
                id="coords-named-indexes",
            ),
            pytest.param(
                {"dim_0": ["a", "b"], "dim_2": ["A", "B"]},
                None,
                {"dim_0": ["a", "b"], "dim_2": ["A", "B"]},
                id="coords-dict",
            ),
            pytest.param(
                [["a", "b"], ["A", "B"]],
                ("dim1", "dim2"),
                {"dim1": ["a", "b"], "dim2": ["A", "B"]},
                id="coords-list-and-dims",
            ),
            pytest.param(
                [["a", "b"], ["A", "B"]],
                ("dim1", "dim2", "dim3"),
                {"dim1": ["a", "b"], "dim2": ["A", "B"]},
                id="dims-superset",
            ),
            pytest.param(
                [["a", "b"], ["A", "B"]],
                ["dim0"],
                {"dim0": ["a", "b"], "dim_1": ["A", "B"]},
                id="dims-subset",
            ),
            pytest.param(
                [pd.Index(["a", "b"], name="dim1"), pd.Index(["A", "B"], name="dim2")],
                ("dim1", "dim2"),
                {"dim1": ["a", "b"], "dim2": ["A", "B"]},
                id="named-indexes-and-matching-dims",
            ),
            pytest.param(
                {"dim_0": ["a", "b"], "dim_1": ["A", "B"]},
                ("dim_0", "dim_1"),
                {"dim_0": ["a", "b"], "dim_1": ["A", "B"]},
                id="coords-dict-and-matching-dims",
            ),
        ],
    )
    def test_labeling(self, coords: Any, dims: Any, expected: dict[str, list]) -> None:
        da = as_dataarray(self.arr, coords=coords, dims=dims)
        assert isinstance(da, DataArray)
        assert da.dims == tuple(expected)
        for dim, values in expected.items():
            assert list(da.coords[dim]) == values

    def test_named_indexes_conflicting_dims_raise(self) -> None:
        coords = [pd.Index(["a", "b"], name="dim1"), pd.Index(["A", "B"], name="dim2")]
        with pytest.raises(ValueError):
            as_dataarray(self.arr, coords=coords, dims=("dim3", "dim4"))

    def test_extra_coord_entries_are_dropped(self) -> None:
        """as_dataarray converts only: dims label the axes, extra coord entries are dropped."""
        target_coords = {"dim_0": ["a", "b"], "dim_2": ["A", "B"]}
        da = as_dataarray(self.arr, coords=target_coords, dims=("dim_0", "dim_1"))
        assert da.dims == ("dim_0", "dim_1")
        assert list(da.coords["dim_0"].values) == ["a", "b"]
        assert "dim_2" not in da.coords

    def test_dims_as_bare_string(self) -> None:
        """Dims may be a single dim name; dict coords are filtered to those dims."""
        da = as_dataarray(np.array([1, 2]), coords={"x": [0, 1], "drop": [9]}, dims="x")
        assert da.dims == ("x",)
        assert list(da.coords["x"].values) == [0, 1]
        assert "drop" not in da.coords

    def test_zero_dim_array_expands_over_dict_coords(self) -> None:
        """A 0-d array converts like a scalar, expanding over dict coords."""
        da = as_dataarray(np.array(5.0), coords={"a": [0, 1]})
        assert da.dims == ("a",)
        assert da.values.tolist() == [5.0, 5.0]


class TestAsDataarrayFromScalar:
    """Scalar conversion: numbers expand over coords when given."""

    @pytest.mark.parametrize(
        "num", [1, np.float64(1)], ids=["python-int", "np-float64"]
    )
    def test_with_dims_and_coords(self, num: Any) -> None:
        da = as_dataarray(num, dims=["dim1"], coords=[["a"]])
        assert isinstance(da, DataArray)
        assert da.dims == ("dim1",)
        assert list(da.coords["dim1"].values) == ["a"]

    def test_default_dims_coords(self) -> None:
        da = as_dataarray(1)
        assert isinstance(da, DataArray)
        assert da.dims == ()
        assert da.coords == {}

    def test_with_named_index_coords(self) -> None:
        da = as_dataarray(1, coords=[pd.RangeIndex(10, name="a")])
        assert isinstance(da, DataArray)
        assert da.dims == ("a",)
        assert list(da.coords["a"].values) == list(range(10))


class TestAsDataarrayFromDataArray:
    """DataArray inputs pass through; unsupported types raise."""

    da_in = DataArray(
        data=[[1, 2], [3, 4]],
        dims=["dim1", "dim2"],
        coords={"dim1": ["a", "b"], "dim2": ["A", "B"]},
    )

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param(
                {"dims": ["dim1", "dim2"], "coords": [["a", "b"], ["A", "B"]]},
                id="matching-dims-and-coords",
            ),
            pytest.param({}, id="default"),
        ],
    )
    def test_passthrough(self, kwargs: dict[str, Any]) -> None:
        da_out = as_dataarray(self.da_in, **kwargs)
        assert isinstance(da_out, DataArray)
        assert da_out.dims == self.da_in.dims
        assert list(da_out.coords["dim1"].values) == list(
            self.da_in.coords["dim1"].values
        )
        assert list(da_out.coords["dim2"].values) == list(
            self.da_in.coords["dim2"].values
        )

    def test_unsupported_type_raises(self) -> None:
        with pytest.raises(TypeError):
            as_dataarray(lambda x: 1, dims=["dim1"], coords=[["a"]])

    def test_fill_missing_coords_rejects_non_xarray(self) -> None:
        with pytest.raises(
            TypeError, match="Expected xarray.DataArray or xarray.Dataset"
        ):
            fill_missing_coords([1, 2, 3])  # type: ignore[call-overload]

    def test_does_not_expand_missing_coord_dims(self) -> None:
        """as_dataarray converts; only broadcast_to_coords expands missing dims."""
        coords = {"a": [0, 1], "b": [10, 20]}
        arr = np.array([1, 2])

        converted = as_dataarray(arr, coords=coords, dims=["a"])
        assert converted.dims == ("a",)

        broadcast = broadcast_to_coords(arr, coords=coords, dims=["a"], strict=False)
        assert broadcast.dims == ("a", "b")


class TestAsDataarrayMultiIndexCoords:
    """MultiIndex coords inputs: level names must not become extra dims."""

    station_mi = pd.MultiIndex.from_tuples(
        [("a", 1), ("b", 2)], names=["letter", "num"]
    )

    @pytest.mark.parametrize(
        ("arr", "expected_values"),
        [
            (np.float64(3.0), [3.0, 3.0]),
            (3, [3, 3]),
            (3.0, [3.0, 3.0]),
            (np.array([10.0, 20.0]), [10.0, 20.0]),
        ],
        ids=["np_number", "python_int", "python_float", "numpy_array"],
    )
    def test_input_types(self, arr: object, expected_values: list[float]) -> None:
        """Level names in multi-index coords must not be treated as extra dims."""
        source = DataArray(
            [1.0, 2.0], coords={"station": self.station_mi}, dims="station"
        )

        da = as_dataarray(arr, coords=source.coords)

        assert da.dims == ("station",)
        assert da.shape == (2,)
        assert set(da.coords.keys()) == {"station", "letter", "num"}
        assert list(da.coords["letter"].values) == ["a", "b"]
        assert list(da.coords["num"].values) == [1, 2]
        assert da.coords["letter"].dims == ("station",)
        assert da.coords["num"].dims == ("station",)
        assert list(da.values) == expected_values

    @pytest.mark.parametrize(
        "coords_factory",
        [
            lambda mi: xr.Coordinates.from_pandas_multiindex(mi, "station"),
            lambda mi: {"station": mi},
            lambda mi: (
                DataArray([1.0, 2.0], coords={"station": mi}, dims="station").coords
            ),
        ],
        ids=["xarray_Coordinates", "plain_dict", "dataarray_coords"],
    )
    def test_coord_input_forms(
        self, coords_factory: Callable[[pd.MultiIndex], CoordsLike]
    ) -> None:
        """Users may pass a MultiIndex via Coordinates, a dict, or another DataArray's coords."""
        coords = coords_factory(self.station_mi)

        da = as_dataarray(3.0, coords=coords)

        assert da.dims == ("station",)
        assert da.shape == (2,)
        assert set(da.coords.keys()) == {"station", "letter", "num"}
        assert da.coords["letter"].dims == ("station",)
        assert da.coords["num"].dims == ("station",)
        assert (da.values == 3.0).all()

    def test_explicit_dims_win_over_inference(self) -> None:
        """Explicit dims must win over any inference from Coordinates."""
        source = DataArray(
            [1.0, 2.0], coords={"station": self.station_mi}, dims="station"
        )

        da = as_dataarray(3.0, coords=source.coords, dims=["station"])
        assert da.dims == ("station",)
        assert da.shape == (2,)
        assert set(da.coords.keys()) == {"station", "letter", "num"}


def _ij_multiindex() -> pd.MultiIndex:
    """Unnamed (i, j) MultiIndex used across the coords-entry tests."""
    return pd.MultiIndex.from_product([[0, 1], ["a", "b"]], names=["i", "j"])


def _named_multiindex(name: str = "multi") -> pd.MultiIndex:
    """:func:`_ij_multiindex` carrying an overall index name."""
    mi = _ij_multiindex()
    mi.name = name
    return mi


# ---------------------------------------------------------------------------
# _coords_to_dict — the coords-entry naming rules
# ---------------------------------------------------------------------------


class TestCoordsToDict:
    """
    Executable spec of ``_coords_to_dict``: how each coords-entry form is
    named or rejected, parameterized by entry form. The end-to-end dim
    assignment these feed lives in :class:`TestAddVariablesCoords`.
    """

    @staticmethod
    def _parse(coords: Any, dims: Any = None) -> dict:
        return _coords_to_dict(coords, dims=dims)

    @pytest.mark.parametrize(
        "coords, dims",
        [
            ([("x", [0, 1, 2])], None),
            ([pd.Index([0, 1, 2], name="x")], None),
            ([pd.Index([0, 1, 2])], ["x"]),
            ([[0, 1, 2]], ["x"]),
            ([range(3)], ["x"]),
            ([np.array([0, 1, 2])], ["x"]),
        ],
        ids=[
            "tuple",
            "named-index",
            "unnamed-index+dims",
            "list+dims",
            "range+dims",
            "ndarray+dims",
        ],
    )
    def test_named_form_parses_to_x(self, coords: Any, dims: Any) -> None:
        """Each naming form parses to {"x": [0, 1, 2]} (tuple = xarray form)."""
        result = self._parse(coords, dims=dims)
        assert set(result) == {"x"}
        assert list(result["x"]) == [0, 1, 2]
        assert result["x"].name == "x"

    @pytest.mark.parametrize(
        "coords, expected",
        [
            (
                xr.Coordinates.from_pandas_multiindex(_ij_multiindex(), "stacked"),
                {"stacked"},
            ),
            ([_named_multiindex()], {"multi"}),
            ([("x", [0, 1, 2], {"units": "m"})], {"x"}),
        ],
        ids=["xarray-coordinates", "named-multiindex", "tuple-with-attrs"],
    )
    def test_other_forms_parse_to_expected_names(
        self, coords: Any, expected: set
    ) -> None:
        assert set(self._parse(coords)) == expected

    def test_mapping_returns_shallow_copy(self) -> None:
        src = {"x": [0, 1, 2], "y": [10, 20]}
        result = self._parse(src)
        assert result == src
        assert result is not src

    @pytest.mark.parametrize(
        "entry", [pd.Index([0, 1, 2]), _ij_multiindex()], ids=["index", "multiindex"]
    )
    def test_unnamed_index_named_from_dims_on_a_copy(self, entry: Any) -> None:
        result = self._parse([entry], dims=["x"])
        assert result["x"].name == "x"
        assert entry.name is None  # caller not mutated

    @pytest.mark.parametrize(
        "entry",
        [[0, 1, 2], range(3), np.array([0, 1, 2]), pd.Index([0, 1, 2])],
        ids=["list", "range", "ndarray", "unnamed-index"],
    )
    def test_unlabeled_without_dims_is_skipped(self, entry: Any) -> None:
        assert self._parse([entry]) == {}

    @pytest.mark.parametrize(
        "coords, dims, match",
        [
            ([_ij_multiindex()], None, r"MultiIndex.*must have \.name set"),
            ([("x",)], None, r"\(dim_name, values\) convention"),
            ([(0, 1, 2)], ["x"], r"\(dim_name, values\) convention"),
            ([("x", 5)], None, r"with array-like values"),
            (
                [DataArray([0, 1, 2], dims=["x"])],
                None,
                r"coords entries must be pd\.Index",
            ),
            ([object()], None, r"coords entries must be pd\.Index"),
        ],
        ids=[
            "unnamed-multiindex",
            "tuple-too-short",
            "tuple-bare-values",
            "tuple-scalar-values",
            "dataarray",
            "unknown-type",
        ],
    )
    def test_invalid_entry_raises_typeerror(
        self, coords: Any, dims: Any, match: str
    ) -> None:
        with pytest.raises(TypeError, match=match):
            self._parse(coords, dims=dims)


# ---------------------------------------------------------------------------
# add_variables — coords / dims map to the variable's dimensions
# ---------------------------------------------------------------------------


class TestAddVariablesCoords:
    """End-to-end: each coords / dims form sets the variable's dimensions."""

    @pytest.mark.parametrize(
        "coords, dims, expected_dims",
        [
            ([("x", [0, 1, 2])], None, ("x",)),
            ([pd.Index([0, 1, 2], name="x")], None, ("x",)),
            ([pd.Index([0, 1, 2])], ["x"], ("x",)),
            ([[0, 1, 2]], ["x"], ("x",)),
            ([range(3)], ["x"], ("x",)),
            ([np.array([0, 1, 2])], ["x"], ("x",)),
            ([[0, 1, 2]], None, ("dim_0",)),
            ([range(3)], None, ("dim_0",)),
            ([np.array([0, 1, 2])], None, ("dim_0",)),
            ([pd.Index([0, 1, 2])], None, ("dim_0",)),
            ([("origin", ["a", "b"]), ("dest", ["x", "y"])], None, ("origin", "dest")),
        ],
        ids=[
            "tuple",
            "named-index",
            "unnamed-index+dims",
            "list+dims",
            "range+dims",
            "ndarray+dims",
            "list",
            "range",
            "ndarray",
            "unnamed-index",
            "multiple-tuples",
        ],
    )
    def test_coords_set_variable_dims(
        self, coords: Any, dims: Any, expected_dims: tuple
    ) -> None:
        m = Model()
        v = m.add_variables(lower=0, coords=coords, dims=dims)
        assert v.dims == expected_dims


# ---------------------------------------------------------------------------
# broadcast_to_coords(strict=False) — broadcast mechanics, mismatches pass
# ---------------------------------------------------------------------------


class TestBroadcastToCoords:
    """strict=False: dims are made to agree; entry mismatches pass through."""

    def test_preserves_extra_dims(self) -> None:
        """Extra dims in the input are not rejected — they broadcast downstream."""
        arr = DataArray(
            [[1, 2], [3, 4], [5, 6]],
            dims=["a", "t"],
            coords={"a": [0, 1, 2], "t": [10, 20]},
        )
        coords = {"a": [0, 1, 2]}
        da = broadcast_to_coords(arr, coords=coords, strict=False)
        assert set(da.dims) == {"a", "t"}
        assert list(da.coords["t"].values) == [10, 20]

    def test_keeps_disjoint_shared_dim_values(self) -> None:
        """Different value sets on a shared dim are passed through (xr.align handles)."""
        arr = DataArray([1, 2, 3, 4, 5], dims=["a"], coords={"a": [0, 1, 2, 3, 4]})
        coords = {"a": [2, 3]}
        da = broadcast_to_coords(arr, coords=coords, strict=False)
        # No exception, no reindex; downstream alignment intersects.
        assert list(da.coords["a"].values) == [0, 1, 2, 3, 4]

    def test_extra_coord_entries_broadcast_in(self) -> None:
        """Coords is source of truth: extra coord entries broadcast into the result."""
        target_coords = {"dim_0": ["a", "b"], "dim_2": ["A", "B"]}
        arr = np.array([[1, 2], [3, 4]])
        da = broadcast_to_coords(
            arr, coords=target_coords, dims=("dim_0", "dim_1"), strict=False
        )
        # dims labels the positional axes; coords adds dim_2 by broadcast.
        assert set(da.dims) == {"dim_0", "dim_1", "dim_2"}
        assert list(da.coords["dim_0"].values) == ["a", "b"]
        assert list(da.coords["dim_2"].values) == ["A", "B"]


# ---------------------------------------------------------------------------
# Implicit MultiIndex-level projection — the legacy/v1 fork point
# ---------------------------------------------------------------------------


class TestBroadcastToCoordsMultiIndexProjection:
    """
    Inputs indexed by levels of a stacked MultiIndex dim are projected onto it.

    Implicit projection is deprecated (scenario B, #732/#737): it warns under
    both modes today and will raise under the v1 convention. Coverage gaps
    raise under strict mode. When #717 lands, the deprecation tests here fork
    into legacy (warn) and v1 (raise) variants.
    """

    def test_broadcasts_single_level(
        self, mi_coords: xr.Coordinates, by_level1: DataArray
    ) -> None:
        """
        A constant indexed by one MultiIndex level broadcasts across the MI dim.

        PyPSA multi-investment multiplies an expression over a (period, timestep)
        'snapshot' MultiIndex by a weighting indexed only by 'period'. Each level
        combination of the MultiIndex must pick up its level's value.
        """
        with pytest.warns(EvolvingAPIWarning, match=r"broadcasting level subset"):
            da = broadcast_to_coords(
                by_level1, coords=mi_coords, dims=["dim_3"], strict=False
            )

        assert da.dims == ("dim_3",)
        assert isinstance(da.indexes["dim_3"], pd.MultiIndex)
        assert da.sel(dim_3=(1, "a")).item() == 10.0
        assert da.sel(dim_3=(1, "b")).item() == 10.0
        assert da.sel(dim_3=(2, "a")).item() == 20.0
        assert da.sel(dim_3=(2, "b")).item() == 20.0

    def test_stacks_full_levels(self, mi_coords: xr.Coordinates) -> None:
        """
        A constant indexed by all MI level names stacks element-wise into the MI dim.

        PyPSA's storage_weightings is a pandas Series over a (period, timestep)
        MultiIndex subset (the last snapshot of each period); it must align onto
        the matching level combinations of the 'snapshot' MultiIndex. Combinations
        the subset does not cover are left as NaN (broadcast path).
        """
        subset = pd.MultiIndex.from_tuples(
            [(1, "a"), (2, "b")], names=["level1", "level2"]
        )
        weights = pd.Series([10.0, 20.0], index=subset)

        with pytest.warns(
            EvolvingAPIWarning, match=r"filling uncovered level combinations"
        ):
            da = broadcast_to_coords(
                weights, coords=mi_coords, dims=["dim_3"], strict=False
            )

        assert da.dims == ("dim_3",)
        assert isinstance(da.indexes["dim_3"], pd.MultiIndex)
        assert da.sel(dim_3=(1, "a")).item() == 10.0
        assert da.sel(dim_3=(2, "b")).item() == 20.0
        assert np.isnan(da.sel(dim_3=(1, "b")).item())
        assert np.isnan(da.sel(dim_3=(2, "a")).item())

    def test_full_coverage_is_silent(
        self, mi_coords: xr.Coordinates, mi_index: pd.MultiIndex
    ) -> None:
        """
        Full-level, fully-covering alignment is convention-clean → no warning.

        Aligning an input that reconstructs the whole MultiIndex onto its dim is
        equivalent to the input already carrying that dim (future §11), so it must
        not emit the EvolvingAPIWarning the partial/gap projections do.
        """
        full = pd.Series([1.0, 2.0, 3.0, 4.0], index=mi_index)

        with warnings.catch_warnings():
            warnings.simplefilter("error", EvolvingAPIWarning)
            da = broadcast_to_coords(
                full, coords=mi_coords, dims=["dim_3"], strict=False
            )

        assert da.dims == ("dim_3",)
        assert da.values.tolist() == [1.0, 2.0, 3.0, 4.0]

    def test_expands_missing_mi_dim_keeps_levels(self) -> None:
        """
        Broadcasting a missing MultiIndex dim must keep its level coords intact.

        expand_dims drops MultiIndex level coords, leaving a degenerate flat
        index that fails to align downstream (PyPSA multi-investment regression).
        """
        midx = pd.MultiIndex.from_tuples(
            [(2020, 0), (2020, 1), (2030, 0), (2030, 1)],
            names=["period", "timestep"],
        )
        midx.name = "snapshot"
        sc = xr.Coordinates.from_pandas_multiindex(midx, "snapshot")
        labels = DataArray(
            [[1], [2], [3], [4]],
            coords={**sc, "name": ["1"]},
            dims=["snapshot", "name"],
        )
        coeff = broadcast_to_coords(
            DataArray([1.0], coords={"name": ["1"]}, dims=["name"]),
            coords=labels.coords,
            dims=labels.dims,
            strict=False,
        )
        assert set(coeff.xindexes) == {"snapshot", "period", "timestep", "name"}
        coeff.reindex_like(labels, fill_value=0)

    def test_ambiguous_level_raises(self) -> None:
        """A level name shared by two MI dims cannot be resolved."""
        a = pd.MultiIndex.from_product([[1, 2], ["a", "b"]], names=("shared", "x"))
        b = pd.MultiIndex.from_product([[1, 2], ["c", "d"]], names=("shared", "y"))
        coords = {
            **xr.Coordinates.from_pandas_multiindex(a, "dimA"),
            **xr.Coordinates.from_pandas_multiindex(b, "dimB"),
        }
        arr = DataArray([1.0, 2.0], coords={"shared": [1, 2]}, dims=["shared"])

        with pytest.raises(ValueError, match=r"shared.*shared by MultiIndex"):
            broadcast_to_coords(arr, coords=coords, strict=False)

    def test_missing_level_value_raises(self, mi_coords: xr.Coordinates) -> None:
        """A level value absent from the input cannot be broadcast."""
        by_level1 = DataArray([10.0, 20.0], coords={"level1": [1, 9]}, dims=["level1"])

        with pytest.raises(ValueError, match=r"Cannot align level.*is missing"):
            broadcast_to_coords(
                by_level1, coords=mi_coords, dims=["dim_3"], strict=False
            )

    def test_unrelated_mi_series_still_unstacks(self) -> None:
        """A MI Series whose levels match no coords MI dim keeps unstacking."""
        sub = pd.MultiIndex.from_product([["p", "q"], [1, 2]], names=["foo", "bar"])
        series = pd.Series([1.0, 2.0, 3.0, 4.0], index=sub)

        da = broadcast_to_coords(series, coords={"time": [0, 1, 2]}, strict=False)

        assert set(da.dims) == {"time", "foo", "bar"}

    def test_partially_named_mi_levels(self) -> None:
        """A None level name in the MultiIndex is skipped during projection."""
        mi = pd.MultiIndex.from_product([[1, 2], ["a", "b"]], names=("level1", None))
        mi.name = "dim_3"
        by_level1 = DataArray([10.0, 20.0], coords={"level1": [1, 2]}, dims=["level1"])

        with pytest.warns(EvolvingAPIWarning, match=r"broadcasting level subset"):
            da = broadcast_to_coords(by_level1, coords={"dim_3": mi}, strict=False)

        assert da.dims == ("dim_3",)
        assert da.values.tolist() == [10.0, 10.0, 20.0, 20.0]

    def test_gap_detection_with_extra_dims(self, mi_coords: xr.Coordinates) -> None:
        """Gaps are detected per level combination even when the input has extra dims."""
        arr = DataArray(
            [[[1.0, np.nan], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]],
            dims=["level1", "level2", "extra"],
            coords={"level1": [1, 2], "level2": ["a", "b"], "extra": [0, 1]},
        )

        with pytest.warns(
            EvolvingAPIWarning, match=r"filling uncovered level combinations"
        ):
            da = broadcast_to_coords(
                arr, coords=mi_coords, dims=["dim_3"], strict=False
            )

        assert set(da.dims) == {"dim_3", "extra"}

    def test_strict_gap_error_truncates_long_missing_list(self) -> None:
        """More than 5 missing combinations are truncated in the error message."""
        idx = pd.MultiIndex.from_product(
            [[1, 2, 3], ["a", "b", "c"]], names=("l1", "l2")
        )
        idx.name = "dim_m"
        coords = xr.Coordinates.from_pandas_multiindex(idx, "dim_m")
        # Diagonal subset: every level value present, 6 of 9 combinations missing.
        diagonal = pd.MultiIndex.from_tuples(
            [(1, "a"), (2, "b"), (3, "c")], names=["l1", "l2"]
        )
        weights = pd.Series([1.0, 2.0, 3.0], index=diagonal)

        with pytest.raises(
            ValueError, match=r"no value for 6 level combination.*in total"
        ):
            broadcast_to_coords(weights, coords, dims=["dim_m"], label="lower bound")

    # --- strict-mode policy on MI projections (deprecation / gaps) ---

    def test_strict_partial_level_warns(
        self, mi_coords: xr.Coordinates, by_level1: DataArray
    ) -> None:
        """
        Per-level bounds broadcast across the MI dim, with the deprecation warning.

        Scenario B (#732 / #737 discussion): implicit MI-level projection is
        deprecated everywhere, including the strict (bounds/mask) path, and will
        raise under the v1 convention.
        """
        with pytest.warns(EvolvingAPIWarning, match=r"broadcasting level subset"):
            da = broadcast_to_coords(
                by_level1, mi_coords, dims=["dim_3"], label="lower bound"
            )

        assert da.sel(dim_3=(1, "b")).item() == 10.0
        assert da.sel(dim_3=(2, "a")).item() == 20.0

    def test_strict_rejects_coverage_gap(self, mi_coords: xr.Coordinates) -> None:
        """A coverage gap warns on the broadcast rung but raises on the strict rung."""
        subset = pd.MultiIndex.from_tuples(
            [(1, "a"), (2, "b")], names=["level1", "level2"]
        )
        weights = pd.Series([10.0, 20.0], index=subset)

        with pytest.warns(
            EvolvingAPIWarning, match=r"filling uncovered level combinations"
        ):
            broadcast_to_coords(weights, coords=mi_coords, dims=["dim_3"], strict=False)

        with pytest.raises(ValueError, match=r"no value for .* level combination"):
            broadcast_to_coords(weights, mi_coords, dims=["dim_3"], label="lower bound")

    def test_strict_rejects_unnamed_mi_mismatch(self) -> None:
        """
        A MultiIndex input with unnamed levels cannot be projected by level name,
        so it keeps its own index under the coords dim. The strict rung must still
        reject it when its level combinations don't cover coords, just as the
        named-level coverage-gap case does.
        """
        idx = pd.MultiIndex.from_product([[2020, 2030], ["t1", "t2"]], names=("p", "t"))
        idx.name = "snapshot"
        coords = xr.Coordinates.from_pandas_multiindex(idx, "snapshot")
        sparse_unnamed = pd.Series({(2020, "t1"): 1.0, (2030, "t2"): 2.0})

        with pytest.raises(ValueError, match=r"MultiIndex for dimension 'snapshot'"):
            broadcast_to_coords(
                sparse_unnamed, coords, dims=["snapshot"], label="lower bound"
            )


# ---------------------------------------------------------------------------
# broadcast_to_coords(strict=True) — the contract
# ---------------------------------------------------------------------------


class TestBroadcastToCoordsStrictMode:
    """strict=True: anything broadcasting can't resolve raises, naming label."""

    def test_extra_dims_pass_loose_fail_strict(self) -> None:
        """Extra dims pass through the broadcast rung but fail the strict rung."""
        arr = DataArray(
            [[1, 2], [3, 4]], dims=["a", "t"], coords={"a": [0, 1], "t": [10, 20]}
        )
        coords = {"a": [0, 1]}

        da = broadcast_to_coords(arr, coords=coords, strict=False)
        assert set(da.dims) == {"a", "t"}

        with pytest.raises(ValueError, match=r"not declared in coords"):
            broadcast_to_coords(arr, coords, label="lower bound")

    def test_requires_label(self) -> None:
        """strict=True without label raises: errors must name their subject."""
        with pytest.raises(TypeError, match=r"requires `label`"):
            broadcast_to_coords(np.array([1, 2]), {"x": [0, 1]})  # type: ignore[call-overload]

    def test_wraps_conversion_errors(self) -> None:
        with pytest.raises(ValueError, match=r"lower bound could not be aligned"):
            broadcast_to_coords(np.array([1, 2]), {"x": [0, 1, 2]}, label="lower bound")

    def test_preserves_type_errors(self) -> None:
        """Unsupported input types stay TypeError (don't become ValueError)."""
        with pytest.raises(TypeError, match=r"lower bound could not be aligned"):
            broadcast_to_coords(lambda x: x, {"x": [0, 1, 2]}, label="lower bound")

    def test_does_not_relabel_coords_errors(self) -> None:
        """Coords-side TypeError carries its own message, not the value label."""
        mi = pd.MultiIndex.from_product([[0, 1], ["a", "b"]], names=["i", "j"])
        with pytest.raises(TypeError, match=r"MultiIndex.*must have \.name set"):
            broadcast_to_coords(np.array([1, 2, 3, 4]), [mi], label="lower bound")


# ---------------------------------------------------------------------------
# validate_alignment — the validation primitive
# ---------------------------------------------------------------------------


class TestValidateAlignment:
    """Raise when arr is incompatible with coords; no-op otherwise."""

    def test_rejects_extra_dims(self) -> None:
        arr = DataArray(
            [[1, 2], [3, 4]], dims=["a", "b"], coords={"a": [0, 1], "b": [0, 1]}
        )
        with pytest.raises(ValueError, match=r"not declared in coords"):
            validate_alignment(arr, {"a": [0, 1]})

    def test_rejects_value_mismatch(self) -> None:
        arr = DataArray([1, 2, 3], dims=["a"], coords={"a": [0, 1, 2]})
        with pytest.raises(ValueError, match="do not match coords"):
            validate_alignment(arr, {"a": [10, 20, 30]})

    def test_allows_subset_dims(self) -> None:
        """arr.dims ⊂ coords.dims is fine (broadcasting fills the missing dim)."""
        arr = DataArray([1, 2, 3], dims=["a"], coords={"a": [0, 1, 2]})
        validate_alignment(arr, {"a": [0, 1, 2], "b": [10, 20]})  # no raise

    def test_unnamed_coords_and_dims(self) -> None:
        """coords=[[...]], dims=[...] enforces the same contract as a named mapping."""
        arr = DataArray([1, 2, 3], dims=["x"], coords={"x": [0, 1, 2]})
        validate_alignment(arr, [[0, 1, 2]], dims=["x"])  # no raise

        bad = DataArray(
            [[1, 2], [3, 4]], dims=["x", "y"], coords={"x": [0, 1], "y": [0, 1]}
        )
        with pytest.raises(ValueError, match=r"not declared in coords"):
            validate_alignment(bad, [[0, 1]], dims=["x"])

    def test_label_in_error(self) -> None:
        arr = DataArray(
            [[1, 2], [3, 4]], dims=["a", "b"], coords={"a": [0, 1], "b": [0, 1]}
        )
        with pytest.raises(ValueError, match=r"lower bound has dimension\(s\) \['b'\]"):
            validate_alignment(arr, {"a": [0, 1]}, label="lower bound")


# ---------------------------------------------------------------------------
# align — the symmetric counterpart (wraps xarray.align)
# ---------------------------------------------------------------------------


class TestAlign:
    """align() conforms multiple linopy / xarray objects to common coords."""

    def test_inner_join_intersects_coords(self, x: Variable) -> None:
        """Default join keeps only the shared coords (x over [0, 1] ∩ alpha over [1, 2])."""
        alpha = xr.DataArray([1, 2], [[1, 2]])

        x_obs, alpha_obs = align(x, alpha)

        assert isinstance(x_obs, Variable)
        assert x_obs.shape == alpha_obs.shape == (1,)
        assert_varequal(x_obs, x.loc[[1]])

    def test_left_join_keeps_left_coords_and_fills(self, x: Variable) -> None:
        """join='left' keeps x's coords; the right operand is reindexed with NaN."""
        alpha = xr.DataArray([1, 2], [[1, 2]])

        x_obs, alpha_obs = align(x, alpha, join="left")

        assert isinstance(x_obs, Variable)
        assert x_obs.shape == alpha_obs.shape == (2,)
        assert_varequal(x_obs, x)
        assert_equal(alpha_obs, DataArray([np.nan, 1], [[0, 1]]))

    def test_inner_join_over_multiindex(self, u: Variable) -> None:
        """Inner join intersects MultiIndex coords element-wise across the stacked dim."""
        beta = xr.DataArray(
            [1, 2, 3],
            [
                (
                    "dim_3",
                    pd.MultiIndex.from_tuples(
                        [(1, "b"), (2, "b"), (1, "c")], names=["level1", "level2"]
                    ),
                )
            ],
        )

        beta_obs, u_obs = align(beta, u)

        assert isinstance(u_obs, Variable)
        assert u_obs.shape == beta_obs.shape == (2,)
        assert_varequal(u_obs, u.loc[[(1, "b"), (2, "b")]])
        assert_equal(beta_obs, beta.loc[[(1, "b"), (2, "b")]])

    def test_aligns_linear_expression(self, x: Variable) -> None:
        """A LinearExpression aligns alongside variables, keeping its _term dim."""
        alpha = xr.DataArray([1, 2], [[1, 2]])
        expr = 20 * x

        x_obs, expr_obs, alpha_obs = align(x, expr, alpha)

        assert isinstance(expr_obs, LinearExpression)
        assert x_obs.shape == alpha_obs.shape == (1,)
        assert expr_obs.shape == (1, 1)  # the trailing 1 is the _term dim
        assert_linequal(expr_obs, expr.loc[[1]])
