#!/usr/bin/env python3
"""
Created on Mon Jun 19 12:11:03 2023

@author: fabian
"""

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from linopy import Model
from linopy.common import (
    assign_fields,
    best_int,
    coords_from_dataset,
    coords_to_dataset_vars,
    get_dims_with_index_levels,
    get_printout_labels,
    is_constant,
    iterate_slices,
    label_coord,
    maybe_group_terms_polars,
    set_data_var,
)


def test_best_int() -> None:
    # Test for int8
    assert best_int(127) == np.int8
    # Test for int16
    assert best_int(128) == np.int16
    assert best_int(32767) == np.int16
    # Test for int32
    assert best_int(32768) == np.int32
    assert best_int(2147483647) == np.int32
    # Test for int64
    assert best_int(2147483648) == np.int64
    assert best_int(9223372036854775807) == np.int64

    # Test for value too large
    with pytest.raises(
        ValueError, match=r"Value 9223372036854775808 is too large for int64."
    ):
        best_int(9223372036854775808)


def test_assign_fields_replaces_without_aligning() -> None:
    ds = xr.Dataset({"vars": ("_term", [0, 1, 2]), "const": 1.0}, attrs={"a": 1})
    result = assign_fields(ds, vars=xr.DataArray([5], dims="_term"), scaling=2.0)
    assert list(result) == ["const", "vars", "scaling"]
    assert result.sizes["_term"] == 1
    assert result.attrs == {"a": 1}


@pytest.mark.parametrize("name", ["solution", "dual"])
def test_set_data_var(name: str) -> None:
    ds = xr.Dataset({"solution": (("a", "b"), np.zeros((2, 3)))})
    out = set_data_var(ds, name, ("a", "b"), np.ones((2, 3)))
    assert out[name].dims == ("a", "b")
    assert (out[name] == 1).all()
    assert (ds["solution"] == 0).all()


def test_set_data_var_shape_mismatch_raises() -> None:
    ds = xr.Dataset({"solution": (("a", "b"), np.zeros((2, 3)))})
    with pytest.raises(ValueError, match="does not match"):
        set_data_var(ds, "solution", ("a", "b"), np.ones((3, 2)))


def test_coords_dataset_vars_roundtrip() -> None:
    """Dash-named coords survive serialization to Dataset vars and back."""
    coords = [pd.Index([1, 2, 3], name="my-simple"), pd.Index(["a"], name="b-c")]
    ds = xr.Dataset(coords_to_dataset_vars(coords))
    assert not any("-" in str(k) for k in ds)
    restored, levels = coords_from_dataset(ds, ["my-simple", "b-c"])
    assert levels == {}
    for got, want in zip(restored, coords):
        assert got.equals(want)
        assert got.name == want.name


def test_coords_from_dataset_reads_name_keyed_format() -> None:
    """Files written before the positional format keyed the variables by name."""
    ds = xr.Dataset({"_coord_simple": ("e", [1, 2, 3])})
    (plain,), _ = coords_from_dataset(ds, ["simple"])
    assert plain.equals(pd.Index([1, 2, 3])) and plain.name == "simple"


@pytest.mark.parametrize(
    "ds",
    [
        pytest.param(
            xr.Dataset(
                {
                    "_coord_d_level_p": ("l0", [1, 2]),
                    "_coord_d_level_q": ("l1", ["a", "b"]),
                    "_coord_d_codes": (("n", "k"), [[0, 0], [0, 1], [1, 0]]),
                }
            ),
            id="name-keyed-0.9",
        ),
        pytest.param(
            xr.Dataset(
                {
                    "_index0_level0": ("l0", [1, 2]),
                    "_index0_level1": ("l1", ["a", "b"]),
                    "_index0_codes": (
                        ("n", "k"),
                        [[0, 0], [0, 1], [1, 0]],
                        {"level_names": '["p", "q"]'},
                    ),
                }
            ),
            id="positional-0.10",
        ),
    ],
)
def test_coords_from_dataset_loads_multiindex_flat(ds: xr.Dataset) -> None:
    """A frozen constraint over a MultiIndex from linopy < 1.0 loads flat."""
    (index,), levels = coords_from_dataset(ds, ["d"])
    assert index.equals(pd.RangeIndex(3)) and index.name == "d"
    assert {n: (d, v.tolist()) for n, (d, v) in levels.items()} == {
        "p": ("d", [1, 1, 2]),
        "q": ("d", ["a", "b", "a"]),
    }


def test_iterate_slices_basic() -> None:
    ds = xr.Dataset(
        {"var": (("x", "y"), np.random.rand(10, 10))},  # noqa: NPY002
        coords={"x": np.arange(10), "y": np.arange(10)},
    )
    slices = list(iterate_slices(ds, slice_size=20))
    assert len(slices) == 5
    for s in slices:
        assert isinstance(s, xr.Dataset)
        assert set(s.dims) == set(ds.dims)


def test_iterate_slices_with_exclude_dims() -> None:
    ds = xr.Dataset(
        {"var": (("x", "y"), np.random.rand(10, 20))},  # noqa: NPY002
        coords={"x": np.arange(10), "y": np.arange(20)},
    )
    slices = list(iterate_slices(ds, slice_size=20, slice_dims=["x"]))
    assert len(slices) == 10
    for s in slices:
        assert isinstance(s, xr.Dataset)
        assert set(s.dims) == set(ds.dims)


def test_iterate_slices_large_max_size() -> None:
    ds = xr.Dataset(
        {"var": (("x", "y"), np.random.rand(10, 10))},  # noqa: NPY002
        coords={"x": np.arange(10), "y": np.arange(10)},
    )
    slices = list(iterate_slices(ds, slice_size=200))
    assert len(slices) == 1
    for s in slices:
        assert isinstance(s, xr.Dataset)
        assert set(s.dims) == set(ds.dims)


def test_iterate_slices_small_max_size() -> None:
    ds = xr.Dataset(
        {"var": (("x", "y"), np.random.rand(10, 20))},  # noqa: NPY002
        coords={"x": np.arange(10), "y": np.arange(20)},
    )
    slices = list(iterate_slices(ds, slice_size=8, slice_dims=["x"]))
    assert (
        len(slices) == 10
    )  # goes to the smallest slice possible which is 1 for the x dimension
    for s in slices:
        assert isinstance(s, xr.Dataset)
        assert set(s.dims) == set(ds.dims)


def test_iterate_slices_slice_size_none() -> None:
    ds = xr.Dataset(
        {"var": (("x", "y"), np.random.rand(10, 10))},  # noqa: NPY002
        coords={"x": np.arange(10), "y": np.arange(10)},
    )
    slices = list(iterate_slices(ds, slice_size=None))
    assert len(slices) == 1
    for s in slices:
        assert ds.equals(s)


def test_iterate_slices_includes_last_slice() -> None:
    ds = xr.Dataset(
        {"var": (("x"), np.random.rand(10))},  # noqa: NPY002
        coords={"x": np.arange(10)},
    )
    slices = list(iterate_slices(ds, slice_size=3, slice_dims=["x"]))
    assert len(slices) == 4  # 10 slices for dimension 'x' with size 10
    total_elements = sum(s.sizes["x"] for s in slices)
    assert total_elements == ds.sizes["x"]  # Ensure all elements are included
    for s in slices:
        assert isinstance(s, xr.Dataset)
        assert set(s.dims) == set(ds.dims)


def test_iterate_slices_empty_slice_dims() -> None:
    ds = xr.Dataset(
        {"var": (("x", "y"), np.random.rand(10, 10))},  # noqa: NPY002
        coords={"x": np.arange(10), "y": np.arange(10)},
    )
    slices = list(iterate_slices(ds, slice_size=50, slice_dims=[]))
    assert len(slices) == 1
    for s in slices:
        assert ds.equals(s)


def test_iterate_slices_invalid_slice_dims() -> None:
    ds = xr.Dataset(
        {"var": (("x", "y"), np.random.rand(10, 10))},  # noqa: NPY002
        coords={"x": np.arange(10), "y": np.arange(10)},
    )
    with pytest.raises(ValueError):
        list(iterate_slices(ds, slice_size=50, slice_dims=["z"]))


def test_iterate_slices_empty_dataset() -> None:
    ds = xr.Dataset(
        {"var": (("x", "y"), np.array([]).reshape(0, 0))}, coords={"x": [], "y": []}
    )
    slices = list(iterate_slices(ds, slice_size=10, slice_dims=["x"]))
    assert len(slices) == 1
    assert ds.equals(slices[0])


def test_iterate_slices_single_element() -> None:
    ds = xr.Dataset({"var": (("x", "y"), np.array([[1]]))}, coords={"x": [0], "y": [0]})
    slices = list(iterate_slices(ds, slice_size=1, slice_dims=["x"]))
    assert len(slices) == 1
    assert ds.equals(slices[0])


def test_get_dims_with_index_levels() -> None:
    # Create test data

    # Case 1: Simple dataset with regular dimensions
    ds1 = xr.Dataset(
        {"temp": (("time", "lat"), np.random.rand(3, 2))},  # noqa: NPY002
        coords={"time": pd.date_range("2024-01-01", periods=3), "lat": [0, 1]},
    )

    # Run tests

    # Test case 1: Regular dimensions
    assert get_dims_with_index_levels(ds1) == ["time", "lat"]

    # Test case 5: Empty dataset
    ds5 = xr.Dataset()
    assert get_dims_with_index_levels(ds5) == []

    # Test case 6: Flat dim with a decomposed multi-index as level coords
    mi = pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=["region", "tech"])
    ds6 = (
        xr.DataArray(np.zeros(4), coords=[mi], dims=["node"])
        .reset_index("node")
        .to_dataset(name="v")
    )
    assert get_dims_with_index_levels(ds6) == ["node (region, tech)"]


def test_get_printout_labels() -> None:
    time = pd.date_range("2024-01-01", periods=3)
    plain = xr.Dataset(coords={"time": time})
    assert list(get_printout_labels(plain, ["time"])[0]) == list(time)

    mi = pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=["region", "tech"])
    flat = (
        xr.DataArray(np.zeros(4), coords=[mi], dims=["node"])
        .reset_index("node")
        .to_dataset(name="v")
    )
    multi = get_printout_labels(flat, ["node"])[0]
    assert multi == [("a", 1), ("a", 2), ("b", 1), ("b", 2)]

    single = flat.drop_vars("tech")
    assert list(get_printout_labels(single, ["node"])[0]) == ["a", "a", "b", "b"]


def test_label_coord() -> None:
    mi = pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=["region", "tech"])
    flat = xr.DataArray(np.zeros(4), coords=[mi], dims=["node"]).reset_index("node")
    assert label_coord(flat, ["node"], [2]) == {"node": ("b", 1)}
    assert label_coord(flat.drop_vars("tech"), ["node"], [2]) == {"node": "b"}

    plain = xr.DataArray(np.zeros(3), coords=[pd.Index([10, 20, 30], name="t")])
    assert label_coord(plain, ["t"], [1]) == {"t": 20}


def test_is_constant() -> None:
    model = Model()
    index = pd.Index(range(10), name="t")
    a = model.add_variables(name="a", coords=[index])
    b = a.sel(t=1)
    c = a * 2
    d = a * a

    non_constant = [a, b, c, d]
    for nc in non_constant:
        assert not is_constant(nc)

    constant_values = [
        5,
        3.14,
        np.int32(7),
        np.float64(2.71),
        pd.Series([1, 2, 3]),
        np.array([4, 5, 6]),
        xr.DataArray([k for k in range(10)], coords=[index]),
    ]
    for cv in constant_values:
        assert is_constant(cv)


def test_maybe_group_terms_polars_no_duplicates() -> None:
    """Fast path: distinct (labels, vars) pairs skip group_by."""
    df = pl.DataFrame({"labels": [0, 0], "vars": [1, 2], "coeffs": [3.0, 4.0]})
    result = maybe_group_terms_polars(df)
    assert result.shape == (2, 3)
    assert result.columns == ["labels", "vars", "coeffs"]
    assert result["coeffs"].to_list() == [3.0, 4.0]


def test_maybe_group_terms_polars_with_duplicates() -> None:
    """Slow path: duplicate (labels, vars) pairs trigger group_by."""
    df = pl.DataFrame({"labels": [0, 0], "vars": [1, 1], "coeffs": [3.0, 4.0]})
    result = maybe_group_terms_polars(df)
    assert result.shape == (1, 3)
    assert result["coeffs"].to_list() == [7.0]
