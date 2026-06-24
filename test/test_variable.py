#!/usr/bin/env python3
"""
Created on Tue Nov  2 22:36:38 2021.

@author: fabian
"""

from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr
import xarray.core.indexes
import xarray.core.utils
from xarray import DataArray
from xarray.testing import assert_equal

import linopy
import linopy.variables
from linopy import Model
from linopy.testing import assert_linequal


@pytest.fixture
def m() -> Model:
    m = Model()
    m.add_variables(coords=[pd.RangeIndex(10, name="first")], name="x")
    m.add_variables(coords=[pd.Index([1, 2, 3], name="second")], name="y")
    m.add_variables(0, 10, name="z")
    return m


@pytest.fixture
def x(m: Model) -> linopy.Variable:
    return m.variables["x"]


@pytest.fixture
def z(m: Model) -> linopy.Variable:
    return m.variables["z"]


def test_variable_repr(x: linopy.Variable) -> None:
    x.__repr__()


def test_variable_inherited_properties(x: linopy.Variable) -> None:
    assert isinstance(x.attrs, dict)
    assert isinstance(x.coords, xr.Coordinates)
    assert isinstance(x.indexes, xarray.core.indexes.Indexes)
    assert isinstance(x.sizes, xarray.core.utils.Frozen)
    assert isinstance(x.shape, tuple)
    assert isinstance(x.size, int)
    assert isinstance(x.dims, tuple)
    assert isinstance(x.ndim, int)


def test_variable_type() -> None:
    m = Model()
    x = m.add_variables(lower=0, upper=10, name="x")
    assert x.type == "Continuous Variable"

    b = m.add_variables(binary=True, name="b")
    assert b.type == "Binary Variable"

    i = m.add_variables(lower=0, upper=10, integer=True, name="i")
    assert i.type == "Integer Variable"

    sc = m.add_variables(lower=1, upper=10, semi_continuous=True, name="sc")
    assert sc.type == "Semi-continuous Variable"


def test_variable_labels(x: linopy.Variable) -> None:
    isinstance(x.labels, xr.DataArray)


def test_variable_data(x: linopy.Variable) -> None:
    isinstance(x.data, xr.DataArray)


def test_wrong_variable_init(m: Model, x: linopy.Variable) -> None:
    # wrong data type
    with pytest.raises(ValueError):
        linopy.Variable(x.labels.values, m, "")  # type: ignore

    # no model
    with pytest.raises(ValueError):
        linopy.Variable(x.labels, None, "")  # type: ignore


def test_variable_getter(x: linopy.Variable, z: linopy.Variable) -> None:
    assert isinstance(x[0], linopy.variables.Variable)

    assert isinstance(x.at[0], linopy.variables.ScalarVariable)


def test_variable_getter_slice(x: linopy.Variable) -> None:
    res = x[:5]
    assert isinstance(res, linopy.Variable)
    assert res.size == 5


def test_variable_getter_slice_with_step(x: linopy.Variable) -> None:
    res = x[::2]
    assert isinstance(res, linopy.Variable)
    assert res.size == 5


def test_variables_getter_list(x: linopy.Variable) -> None:
    res = x[[1, 2, 3]]
    assert isinstance(res, linopy.Variable)
    assert res.size == 3


def test_variable_getter_invalid_shape(x: linopy.Variable) -> None:
    with pytest.raises(AssertionError):
        x.at[0, 0]


def test_variable_loc(x: linopy.Variable) -> None:
    assert isinstance(x.loc[[1, 2, 3]], linopy.Variable)


def test_variable_sel(x: linopy.Variable) -> None:
    assert isinstance(x.sel(first=[1, 2, 3]), linopy.Variable)


def test_variable_isel(x: linopy.Variable) -> None:
    assert isinstance(x.isel(first=[1, 2, 3]), linopy.Variable)
    assert_equal(
        x.isel(first=[0, 1]).labels,
        x.sel(first=[0, 1]).labels,
    )


def test_variable_upper_getter(z: linopy.Variable) -> None:
    assert z.upper.item() == 10


def test_variable_lower_getter(z: linopy.Variable) -> None:
    assert z.lower.item() == 0


def test_variable_upper_setter(z: linopy.Variable) -> None:
    z.upper = 20
    assert z.upper.item() == 20


def test_variable_lower_setter(z: linopy.Variable) -> None:
    z.lower = 8
    assert z.lower == 8


def test_variable_upper_setter_with_array(x: linopy.Variable) -> None:
    idx = pd.RangeIndex(10, name="first")
    upper = pd.Series(range(25, 35), index=idx)
    x.upper = upper
    assert isinstance(x.upper, xr.DataArray)
    assert (x.upper == upper).all()


def test_variable_upper_setter_with_array_invalid_dim(x: linopy.Variable) -> None:
    with pytest.raises(ValueError):
        upper = pd.Series(range(25, 35))
        x.upper = upper


def test_variable_upper_setter_with_non_constant(z: linopy.Variable) -> None:
    with pytest.raises(TypeError):
        z.upper = z


def test_variable_lower_setter_with_array(x: linopy.Variable) -> None:
    idx = pd.RangeIndex(10, name="first")
    lower = pd.Series(range(15, 25), index=idx)
    x.lower = lower
    assert isinstance(x.lower, xr.DataArray)
    assert (x.lower == lower).all()


def test_variable_lower_setter_with_array_invalid_dim(x: linopy.Variable) -> None:
    with pytest.raises(ValueError):
        lower = pd.Series(range(15, 25))
        x.lower = lower


def test_variable_update_bounds(z: linopy.Variable) -> None:
    z.update(lower=2, upper=20)
    assert z.lower.item() == 2
    assert z.upper.item() == 20


def test_variable_update_lower_only(z: linopy.Variable) -> None:
    z.update(lower=3)
    assert z.lower.item() == 3
    assert z.upper.item() == 10  # unchanged from fixture default


def test_variable_update_no_kwargs_is_noop(z: linopy.Variable) -> None:
    old_lower, old_upper = z.lower.item(), z.upper.item()
    z.update()
    assert z.lower.item() == old_lower
    assert z.upper.item() == old_upper


def test_variable_update_rejects_inverted_bounds(z: linopy.Variable) -> None:
    with pytest.raises(ValueError, match="lower > upper"):
        z.update(lower=20, upper=5)


def test_variable_update_rejects_non_constant(z: linopy.Variable) -> None:
    with pytest.raises(TypeError, match="must be a constant"):
        z.update(upper=z)


def test_variable_update_returns_self(z: linopy.Variable) -> None:
    out = z.update(lower=1)
    assert out is z


def test_variable_update_array_invalid_dim(x: linopy.Variable) -> None:
    with pytest.raises(ValueError):
        x.update(lower=pd.Series(range(15, 25)))


def test_variable_update_upper_only(z: linopy.Variable) -> None:
    """upper= alone changes upper; lower untouched."""
    old_lower = z.lower.copy()
    z.update(upper=25)
    assert (z.upper == 25).all()
    assert (z.lower == old_lower).all()


def test_variable_update_with_array(x: linopy.Variable) -> None:
    """Array bound that aligns on the variable's coord is accepted."""
    lower = pd.Series(range(10, 20), index=pd.RangeIndex(10, name="first"))
    x.update(lower=lower)
    np.testing.assert_array_equal(x.lower.values, lower.values)


def test_variable_sum(x: linopy.Variable) -> None:
    res = x.sum()
    assert res.nterm == 10


def test_variable_sum_warn_using_dims(x: linopy.Variable) -> None:
    with pytest.warns(DeprecationWarning):
        x.sum(dims="first")


def test_variable_sum_warn_unknown_kwargs(x: linopy.Variable) -> None:
    with pytest.raises(ValueError):
        x.sum(unknown_kwarg="first")


def test_fill_value() -> None:
    isinstance(linopy.variables.Variable._fill_value, dict)


def test_variable_where(x: linopy.Variable) -> None:
    x = x.where([True] * 4 + [False] * 6)
    assert isinstance(x, linopy.variables.Variable)
    assert x.labels[9] == x._fill_value["labels"]

    x = x.where([True] * 4 + [False] * 6, x.at[0])
    assert isinstance(x, linopy.variables.Variable)
    assert x.labels[9] == x.at[0].label

    x = x.where([True] * 4 + [False] * 6, x.loc[0])
    assert isinstance(x, linopy.variables.Variable)
    assert x.labels[9] == x.at[0].label

    with pytest.raises(ValueError):
        x.where([True] * 4 + [False] * 6, 0)  # type: ignore


def test_variable_where_with_solution(x: linopy.Variable) -> None:
    x.solution = xr.DataArray(np.arange(10.0), coords=x.labels.coords)
    cond = [True] * 4 + [False] * 6
    filtered = x.where(cond)
    assert filtered.labels[9] == x._fill_value["labels"]
    assert filtered.data["solution"][0] == 0.0
    assert np.isnan(filtered.data["solution"][9])


def test_variable_shift(x: linopy.Variable) -> None:
    x = x.shift(first=3)
    assert isinstance(x, linopy.variables.Variable)
    assert x.labels[0] == -1


def test_variable_swap_dims(x: linopy.Variable) -> None:
    x = x.assign_coords({"second": ("first", x.indexes["first"] + 100)})
    x = x.swap_dims({"first": "second"})
    assert isinstance(x, linopy.variables.Variable)
    assert x.dims == ("second",)


def test_variable_set_index(x: linopy.Variable) -> None:
    x = x.assign_coords({"second": ("first", x.indexes["first"] + 100)})
    x = x.set_index({"multi": ["first", "second"]})
    assert isinstance(x, linopy.variables.Variable)
    assert x.dims == ("multi",)
    assert isinstance(x.indexes["multi"], pd.MultiIndex)


def test_isnull(x: linopy.Variable) -> None:
    x = x.where([True] * 4 + [False] * 6)
    assert isinstance(x.isnull(), xr.DataArray)
    assert (x.isnull() == [False] * 4 + [True] * 6).all()


def test_variable_fillna(x: linopy.Variable) -> None:
    x = x.where([True] * 4 + [False] * 6)

    isinstance(x.fillna(x.at[0]), linopy.variables.Variable)


def test_variable_bfill(x: linopy.Variable) -> None:
    x = x.where([False] * 4 + [True] * 6)
    x = x.bfill("first")
    assert isinstance(x, linopy.variables.Variable)
    assert x.labels[2] == x.labels[4]
    assert x.labels[2] != x.labels[5]


def test_variable_broadcast_like(x: linopy.Variable) -> None:
    result = x.broadcast_like(x.labels)
    assert isinstance(result, linopy.variables.Variable)


def test_variable_ffill(x: linopy.Variable) -> None:
    x = x.where([True] * 4 + [False] * 6)
    x = x.ffill("first")
    assert isinstance(x, linopy.variables.Variable)
    assert x.labels[9] == x.labels[3]
    assert x.labels[3] != x.labels[2]


def test_variable_expand_dims(x: linopy.Variable) -> None:
    result = x.expand_dims("new_dim")
    assert isinstance(result, linopy.variables.Variable)
    assert result.dims == ("new_dim", "first")


def test_variable_stack(x: linopy.Variable) -> None:
    result = x.expand_dims("new_dim").stack(new=("new_dim", "first"))
    assert isinstance(result, linopy.variables.Variable)
    assert result.dims == ("new",)


def test_variable_unstack(x: linopy.Variable) -> None:
    result = x.expand_dims("new_dim").stack(new=("new_dim", "first")).unstack("new")
    assert isinstance(result, linopy.variables.Variable)
    assert result.dims == ("new_dim", "first")


def test_variable_flat(x: linopy.Variable) -> None:
    result = x.flat
    assert isinstance(result, pd.DataFrame)
    assert len(result) == x.size


def test_variable_polars(x: linopy.Variable) -> None:
    result = x.to_polars()
    assert isinstance(result, pl.DataFrame)
    assert len(result) == x.size


def test_variable_sanitize(x: linopy.Variable) -> None:
    # convert intentionally to float with nans
    fill_value: dict[str, str | int | float] = {
        "labels": np.nan,
        "lower": np.nan,
        "upper": np.nan,
    }
    x = x.where([True] * 4 + [False] * 6, fill_value)
    x = x.sanitize()
    assert isinstance(x, linopy.variables.Variable)
    assert x.labels[9] == -1


def test_variable_iterate_slices(x: linopy.Variable) -> None:
    slices = x.iterate_slices(slice_size=2)
    for s in slices:
        assert isinstance(s, linopy.variables.Variable)
        assert s.size <= 2


def test_variable_addition(x: linopy.Variable) -> None:
    expr1 = x + 1
    assert isinstance(expr1, linopy.expressions.LinearExpression)
    expr2 = 1 + x
    assert isinstance(expr2, linopy.expressions.LinearExpression)
    assert_linequal(expr1, expr2)

    assert x.__radd__(object()) is NotImplemented
    assert x.__add__(object()) is NotImplemented


def test_variable_subtraction(x: linopy.Variable) -> None:
    expr1 = -x + 1
    assert isinstance(expr1, linopy.expressions.LinearExpression)
    expr2 = 1 - x
    assert isinstance(expr2, linopy.expressions.LinearExpression)
    assert_linequal(expr1, expr2)

    assert x.__rsub__(object()) is NotImplemented
    assert x.__sub__(object()) is NotImplemented


def test_variable_multiplication(x: linopy.Variable) -> None:
    expr1 = x * 2
    assert isinstance(expr1, linopy.expressions.LinearExpression)
    expr2 = 2 * x
    assert isinstance(expr2, linopy.expressions.LinearExpression)
    assert_linequal(expr1, expr2)

    expr3 = x * x
    assert isinstance(expr3, linopy.expressions.QuadraticExpression)

    assert x.__rmul__(object()) is NotImplemented
    assert x.__mul__(object()) is NotImplemented


class TestAddVariablesBoundsWithCoords:
    """Test that add_variables correctly handles all bound types with coords."""

    SEQ_COORDS = [pd.RangeIndex(3, name="x")]
    DICT_COORDS = {"x": [0, 1, 2]}

    @pytest.fixture()
    def model(self) -> "Model":
        return Model()

    # -- All bound types should work with both coord formats ---------------

    @pytest.mark.parametrize(
        "lower",
        [
            pytest.param(0, id="scalar"),
            pytest.param(np.float64(0), id="np.number"),
            pytest.param(np.array(0), id="numpy-0d"),
            pytest.param(np.array([0, 0, 0]), id="numpy-1d"),
            pytest.param(
                pd.Series([0, 0, 0], index=pd.RangeIndex(3, name="x")), id="pandas"
            ),
            pytest.param([0, 0, 0], id="list"),
            pytest.param(
                DataArray([0, 0, 0], dims=["x"], coords={"x": [0, 1, 2]}),
                id="dataarray",
            ),
            pytest.param(DataArray([0, 0, 0], dims=["x"]), id="dataarray-no-coords"),
            pytest.param(xr.DataArray(0), id="dataarray-0d"),
        ],
    )
    @pytest.mark.parametrize(
        "coords",
        [
            pytest.param([pd.RangeIndex(3, name="x")], id="seq-coords"),
            pytest.param({"x": [0, 1, 2]}, id="dict-coords"),
        ],
    )
    def test_bound_types_with_coords(
        self, model: "Model", lower: Any, coords: Any
    ) -> None:
        var = model.add_variables(lower=lower, coords=coords, name="x")
        assert var.shape == (3,)
        assert var.dims == ("x",)
        assert list(var.coords["x"].values) == [0, 1, 2]

    # -- DataArray validation: mismatch and extra dims ---------------------

    @pytest.mark.parametrize(
        "coords",
        [
            pytest.param([pd.RangeIndex(5, name="x")], id="seq-coords"),
            pytest.param({"x": [0, 1, 2, 3, 4]}, id="dict-coords"),
        ],
    )
    def test_dataarray_coord_mismatch(self, model: "Model", coords: Any) -> None:
        lower = DataArray([0, 0, 0], dims=["x"], coords={"x": [0, 1, 2]})
        with pytest.raises(ValueError, match="lower bound.*do not match coords"):
            model.add_variables(lower=lower, coords=coords, name="x")

    def test_dataarray_coord_mismatch_upper(self, model: "Model") -> None:
        upper = DataArray([1, 2, 3], dims=["x"], coords={"x": [10, 20, 30]})
        with pytest.raises(ValueError, match="upper bound.*do not match coords"):
            model.add_variables(upper=upper, coords=self.SEQ_COORDS, name="x")

    def test_dataarray_extra_dims(self, model: "Model") -> None:
        lower = DataArray(
            [[1, 2], [3, 4], [5, 6]], dims=["x", "y"], coords={"x": [0, 1, 2]}
        )
        with pytest.raises(ValueError, match=r"lower bound has dimension\(s\) \['y'\]"):
            model.add_variables(lower=lower, coords=self.DICT_COORDS, name="x")

    def test_mask_extra_dims_with_unnamed_coords_and_dims(self, model: "Model") -> None:
        """Mask is validated against coords + dims= like lower/upper."""
        mask = DataArray(
            [[True, False], [True, False], [False, True]],
            dims=["x", "extra"],
            coords={"x": [0, 1, 2]},
        )
        with pytest.raises(ValueError, match=r"mask has dimension\(s\) \['extra'\]"):
            model.add_variables(
                mask=mask,
                coords=[[0, 1, 2]],
                dims=["x"],
                name="m",
            )

    def test_dataarray_coord_reorder(self, model: "Model") -> None:
        """A bound whose coords differ only in order is reindexed to coords."""
        lower = DataArray([3, 1, 2], dims=["x"], coords={"x": ["c", "a", "b"]})
        var = model.add_variables(
            lower=lower, coords=[pd.Index(["a", "b", "c"], name="x")], name="x"
        )
        assert (var.data.lower == [1, 2, 3]).all()

    def test_positional_bound_aligns_to_coords(self, model: "Model") -> None:
        """
        Numpy / unnamed-pandas bounds align to coords positionally,
        even when the input's auto-generated coord values would not match.
        """
        coords = [pd.Index(list("abc"), name="x")]
        # numpy array — no labels at all, positional alignment.
        v_np = model.add_variables(upper=np.array([1, 2, 3]), coords=coords, name="np")
        assert v_np.dims == ("x",)
        assert (v_np.data.upper.sel(x="a") == 1).all()
        assert (v_np.data.upper.sel(x="c") == 3).all()
        # Unnamed Series — pandas index is auto-generated, ignored in favour
        # of coords (positional alignment, principle: coords is source of truth).
        v_s = model.add_variables(
            upper=pd.Series([10, 20, 30]), coords=coords, name="s"
        )
        assert v_s.dims == ("x",)
        assert (v_s.data.upper.sel(x="a") == 10).all()
        assert (v_s.data.upper.sel(x="c") == 30).all()
        # Unnamed DataFrame — both axes positional.
        v_df = model.add_variables(
            upper=pd.DataFrame([[1, 2], [3, 4], [5, 6]]),
            coords=[pd.Index(list("abc"), name="x"), pd.Index(list("xy"), name="y")],
            name="df",
        )
        assert v_df.dims == ("x", "y")
        assert (v_df.data.upper.sel(x="a", y="x") == 1).all()
        assert (v_df.data.upper.sel(x="c", y="y") == 6).all()

    def test_positional_bound_wrong_size_raises_clear_error(
        self, model: "Model"
    ) -> None:
        """
        Shape mismatch on positional inputs surfaces as a size error,
        not a 'coordinates do not match' error.
        """
        coords = [pd.Index(list("abc"), name="x")]
        with pytest.raises(ValueError, match=r"upper bound could not be aligned"):
            model.add_variables(upper=np.array([1, 2]), coords=coords, name="np_bad")
        with pytest.raises(ValueError, match=r"upper bound could not be aligned"):
            model.add_variables(upper=pd.Series([1, 2]), coords=coords, name="s_bad")

    def test_unnamed_pd_index_is_size_only(self, model: "Model") -> None:
        bound = DataArray([1, 2, 3], dims=["dim_0"])
        var = model.add_variables(upper=bound, coords=[pd.Index([0, 1, 2])], name="x")
        assert (var.upper == [1, 2, 3]).all()

    # -- Broadcasting missing dims -----------------------------------------

    @pytest.mark.parametrize(
        "bound",
        [
            pytest.param(
                DataArray([1, 2, 3], dims=["time"], coords={"time": range(3)}),
                id="DataArray",
            ),
            pytest.param(
                pd.Series(index=pd.RangeIndex(3, name="time"), data=[1, 2, 3]),
                id="Series",
            ),
            pytest.param(
                pd.DataFrame(
                    index=pd.RangeIndex(3, name="time"),
                    columns=pd.Index(["red"], name="colour"),
                    data=[[1], [2], [3]],
                ),
                id="DataFrame",
            ),
            pytest.param(
                pd.Series(
                    index=pd.MultiIndex.from_product(
                        [pd.RangeIndex(3), ["red"]], names=("time", "colour")
                    ),
                    data=[1, 2, 3],
                ),
                id="Series-multiindex",
            ),
            pytest.param(
                pd.DataFrame(
                    index=pd.RangeIndex(3, name="time"),
                    columns=pd.MultiIndex.from_product(
                        [["a", "b"], ["red"]], names=("space", "colour")
                    ),
                    data=[[1, 1], [2, 2], [3, 3]],
                ),
                id="DataFrame-multicolumns",
            ),
            pytest.param(
                pd.DataFrame(
                    index=pd.MultiIndex.from_product(
                        [pd.RangeIndex(3), ["a", "b"]], names=("time", "space")
                    ),
                    columns=pd.Index(["red"], name="colour"),
                    data=[[1], [1], [2], [2], [3], [3]],
                ),
                id="DataFrame-multiindex",
            ),
        ],
    )
    def test_bound_broadcast_missing_dim(
        self, model: "Model", bound: DataArray | pd.Series | pd.DataFrame
    ) -> None:
        """Pandas / DataArray bounds missing dims are broadcast to coords."""
        time = pd.RangeIndex(3, name="time")
        space = pd.Index(["a", "b"], name="space")
        colour = pd.Index(["red"], name="colour")
        var = model.add_variables(
            lower=-bound, upper=bound, coords=[time, space, colour], name="x"
        )
        assert var.dims == ("time", "space", "colour")
        assert var.data.lower.dims == ("time", "space", "colour")
        assert var.data.upper.dims == ("time", "space", "colour")
        assert var.sizes == {"time": 3, "space": 2, "colour": 1}
        assert not var.data.lower.isnull().any()
        assert (var.data.lower.sel(space="a", colour="red") == [-1, -2, -3]).all()
        assert (var.data.lower.sel(space="b", colour="red") == [-1, -2, -3]).all()
        assert (var.data.upper.sel(space="a", colour="red") == [1, 2, 3]).all()

    @pytest.mark.parametrize(
        "lower, upper",
        [
            pytest.param(0, "da", id="scalar-lower+da-upper"),
            pytest.param("da", 1, id="da-lower+scalar-upper"),
            pytest.param("da", "da", id="da-lower+da-upper"),
        ],
    )
    def test_dataarray_broadcast_missing_dim_order(
        self, model: "Model", lower: Any, upper: Any
    ) -> None:
        """Dimension order follows coords, not the type of the bounds (#706)."""
        x = pd.Index(["a", "b", "c"], name="x")
        y = pd.Index(["X", "Y"], name="y")
        full = DataArray(
            np.arange(6).reshape(3, 2), coords={"x": x, "y": y}, dims=["x", "y"]
        )
        # bounds are DataArrays missing the 'y' dimension
        da = full.sum("y")
        lower = da if lower == "da" else lower
        upper = da if upper == "da" else upper
        var = model.add_variables(lower=lower, upper=upper, coords=[x, y], name="x")
        assert var.dims == ("x", "y")
        assert var.data.lower.dims == ("x", "y")
        assert var.data.upper.dims == ("x", "y")

    # -- Special coord formats ---------------------------------------------

    def test_xarray_coordinates_object(self, model: "Model") -> None:
        time = pd.RangeIndex(3, name="time")
        base = model.add_variables(lower=0, coords=[time], name="base")
        lower = DataArray([1, 1, 1], dims=["time"], coords={"time": range(3)})
        var = model.add_variables(lower=lower, coords=base.coords, name="x2")
        assert var.shape == (3,)

    # -- Mixed bound type combinations ------------------------------------

    @pytest.mark.parametrize(
        "lower, upper",
        [
            pytest.param(
                DataArray([0, 0, 0], dims=["x"], coords={"x": [0, 1, 2]}),
                np.array([1, 1, 1]),
                id="da-lower+numpy-upper",
            ),
            pytest.param(
                np.array([0, 0, 0]),
                DataArray([1, 1, 1], dims=["x"], coords={"x": [0, 1, 2]}),
                id="numpy-lower+da-upper",
            ),
            pytest.param(
                DataArray([0, 0, 0], dims=["x"], coords={"x": [0, 1, 2]}),
                DataArray([1, 1, 1], dims=["x"], coords={"x": [0, 1, 2]}),
                id="da-lower+da-upper",
            ),
            pytest.param(
                DataArray([0, 0, 0], dims=["x"], coords={"x": [0, 1, 2]}),
                10,
                id="da-lower+scalar-upper",
            ),
            pytest.param(
                0,
                DataArray([1, 1, 1], dims=["x"], coords={"x": [0, 1, 2]}),
                id="scalar-lower+da-upper",
            ),
            pytest.param(
                DataArray([0, 0, 0], dims=["x"], coords={"x": [0, 1, 2]}),
                xr.DataArray(10),
                id="da-lower+scalar-da-upper",
            ),
        ],
    )
    def test_mixed_bound_types(self, model: "Model", lower: Any, upper: Any) -> None:
        var = model.add_variables(
            lower=lower, upper=upper, coords=self.SEQ_COORDS, name="x"
        )
        assert var.shape == (3,)
        assert var.dims == ("x",)
        assert not var.data.lower.isnull().any()
        assert not var.data.upper.isnull().any()

    def test_both_dataarray_different_dim_subsets(self, model: "Model") -> None:
        """Lower and upper cover different subsets of dims, both broadcast."""
        time = pd.RangeIndex(3, name="time")
        space = pd.Index(["a", "b"], name="space")
        lower = DataArray([0, 0, 0], dims=["time"], coords={"time": range(3)})
        upper = DataArray([10, 20], dims=["space"], coords={"space": ["a", "b"]})
        var = model.add_variables(
            lower=lower, upper=upper, coords=[time, space], name="x"
        )
        assert var.sizes == {"time": 3, "space": 2}
        assert not var.data.lower.isnull().any()
        assert not var.data.upper.isnull().any()
        assert (var.data.upper.sel(time=0) == [10, 20]).all()

    def test_one_dataarray_mismatches_other_ok(self, model: "Model") -> None:
        """Only the mismatched bound should raise, regardless of the other."""
        lower = DataArray([0, 0, 0], dims=["x"], coords={"x": [0, 1, 2]})
        upper = DataArray([1, 1], dims=["x"], coords={"x": [10, 20]})
        with pytest.raises(ValueError, match=r"upper bound.*do not match coords"):
            model.add_variables(
                lower=lower, upper=upper, coords=self.SEQ_COORDS, name="x"
            )

    # -- Coords inferred from bounds (no coords arg) ----------------------

    @pytest.mark.parametrize(
        "lower",
        [
            pytest.param(
                DataArray([0, 0, 0], dims=["x"], coords={"x": [10, 20, 30]}),
                id="dataarray",
            ),
            pytest.param(
                pd.Series([0, 0, 0], index=pd.Index([10, 20, 30], name="x")),
                id="pandas",
            ),
        ],
    )
    def test_coords_inferred_from_bounds(self, model: "Model", lower: Any) -> None:
        """When coords is None, dims/coords are inferred from the bounds."""
        var = model.add_variables(lower=lower, name="x")
        assert var.dims == ("x",)
        assert list(var.coords["x"].values) == [10, 20, 30]

    def test_coords_inferred_multidim(self, model: "Model") -> None:
        lower = DataArray(
            np.zeros((3, 2)),
            dims=["time", "space"],
            coords={"time": [0, 1, 2], "space": ["a", "b"]},
        )
        var = model.add_variables(lower=lower, name="x")
        assert set(var.dims) == {"time", "space"}
        assert var.sizes == {"time": 3, "space": 2}

    # -- Multi-dimensional coords -----------------------------------------

    @pytest.mark.parametrize(
        "coords",
        [
            pytest.param(
                [pd.RangeIndex(3, name="time"), pd.Index(["a", "b"], name="space")],
                id="seq-coords",
            ),
            pytest.param(
                {"time": [0, 1, 2], "space": ["a", "b"]},
                id="dict-coords",
            ),
        ],
    )
    def test_multidim_coords_with_scalar(self, model: "Model", coords: Any) -> None:
        var = model.add_variables(lower=0, upper=1, coords=coords, name="x")
        assert set(var.dims) == {"time", "space"}
        assert var.sizes == {"time": 3, "space": 2}

    def test_multidim_dataarray_with_coords(self, model: "Model") -> None:
        lower = DataArray(
            np.zeros((3, 2)),
            dims=["time", "space"],
            coords={"time": [0, 1, 2], "space": ["a", "b"]},
        )
        coords = [pd.RangeIndex(3, name="time"), pd.Index(["a", "b"], name="space")]
        var = model.add_variables(lower=lower, coords=coords, name="x")
        assert set(var.dims) == {"time", "space"}
        assert var.sizes == {"time": 3, "space": 2}
        assert not var.data.lower.isnull().any()

    def test_bounds_with_different_dim_order(self, model: "Model") -> None:
        """Lower (time, space) and upper (space, time) should align correctly."""
        time = pd.RangeIndex(3, name="time")
        space = pd.Index(["a", "b"], name="space")
        lower = DataArray(
            np.zeros((3, 2)),
            dims=["time", "space"],
            coords={"time": range(3), "space": ["a", "b"]},
        )
        upper = DataArray(
            np.ones((2, 3)),
            dims=["space", "time"],
            coords={"space": ["a", "b"], "time": range(3)},
        )
        var = model.add_variables(
            lower=lower, upper=upper, coords=[time, space], name="x"
        )
        assert var.sizes == {"time": 3, "space": 2}
        assert (var.data.lower.values == 0).all()
        assert (var.data.upper.values == 1).all()

    # -- Reordered coordinates ---------------------------------------------

    def test_reordered_coords_reindexed(self, model: "Model") -> None:
        """Same coord values in different order should reindex, not raise."""
        lower = DataArray([10, 20, 30], dims=["x"], coords={"x": ["c", "a", "b"]})
        var = model.add_variables(lower=lower, coords={"x": ["a", "b", "c"]}, name="x")
        assert list(var.coords["x"].values) == ["a", "b", "c"]
        # Values must follow the reindexed order, not the original
        assert list(var.data.lower.values) == [20, 30, 10]

    def test_reordered_coords_different_values_raises(self, model: "Model") -> None:
        """Overlapping but not identical coord sets must still raise."""
        lower = DataArray([10, 20], dims=["x"], coords={"x": ["a", "b"]})
        with pytest.raises(ValueError, match=r"lower bound.*do not match coords"):
            model.add_variables(lower=lower, coords={"x": ["a", "c"]}, name="x")

    # -- String and datetime coordinates -----------------------------------

    def test_string_coordinates(self, model: "Model") -> None:
        coords = {"region": ["north", "south", "east"]}
        lower = DataArray(
            [0, 0, 0],
            dims=["region"],
            coords={"region": ["north", "south", "east"]},
        )
        var = model.add_variables(lower=lower, coords=coords, name="x")
        assert var.dims == ("region",)
        assert list(var.coords["region"].values) == ["north", "south", "east"]

    def test_datetime_coordinates(self, model: "Model") -> None:
        dates = pd.date_range("2025-01-01", periods=3)
        coords = [dates.rename("time")]
        lower = DataArray([0, 0, 0], dims=["time"], coords={"time": dates})
        var = model.add_variables(lower=lower, coords=coords, name="x")
        assert var.dims == ("time",)
        assert var.shape == (3,)

    def test_string_coords_mismatch(self, model: "Model") -> None:
        lower = DataArray(
            [0, 0], dims=["region"], coords={"region": ["north", "south"]}
        )
        with pytest.raises(ValueError, match=r"lower bound.*do not match coords"):
            model.add_variables(
                lower=lower,
                coords={"region": ["north", "south", "east"]},
                name="x",
            )


class TestAddVariablesMultiIndexCoords:
    """MultiIndex-specific coord handling in add_variables."""

    @pytest.fixture
    def model(self) -> "Model":
        return Model()

    @pytest.fixture
    def midx(self) -> pd.MultiIndex:
        mi = pd.MultiIndex.from_product([[0, 1], ["a", "b"]], names=("l1", "l2"))
        mi.name = "multi"
        return mi

    def test_scalar_bounds(self, model: "Model", midx: pd.MultiIndex) -> None:
        var = model.add_variables(lower=0, upper=1, coords=[midx], name="x")
        assert var.shape == (4,)
        assert var.dims == ("multi",)

    def test_dataarray_bound(self, model: "Model", midx: pd.MultiIndex) -> None:
        bound = DataArray([1, 2, 3, 4], dims=["multi"], coords={"multi": midx})
        var = model.add_variables(upper=bound, coords=[midx], name="x")
        assert var.shape == (4,)
        assert (var.data.upper == [1, 2, 3, 4]).all()

    def test_dataarray_bound_broadcast(
        self, model: "Model", midx: pd.MultiIndex
    ) -> None:
        time = pd.Index([10, 20, 30], name="time")
        bound = DataArray([1, 2, 3, 4], dims=["multi"], coords={"multi": midx})
        var = model.add_variables(
            lower=-bound, upper=bound, coords=[midx, time], name="x"
        )
        assert var.dims == ("multi", "time")
        assert var.shape == (4, 3)
        assert (var.data.upper.sel(time=10) == [1, 2, 3, 4]).all()

    def test_without_name_raises(self, model: "Model") -> None:
        midx = pd.MultiIndex.from_product([[0, 1], ["a", "b"]], names=("l1", "l2"))
        with pytest.raises(TypeError, match="MultiIndex.*must have .name set"):
            model.add_variables(lower=0, upper=1, coords=[midx], name="x")

    def test_mismatched_multiindex_raises(
        self, model: "Model", midx: pd.MultiIndex
    ) -> None:
        other = pd.MultiIndex.from_product([[0, 1], ["x", "y"]], names=("l1", "l2"))
        other.name = "multi"
        bound = DataArray([1, 2, 3, 4], dims=["multi"], coords={"multi": other})
        with pytest.raises(ValueError, match="MultiIndex.*does not match"):
            model.add_variables(upper=bound, coords=[midx], name="x")

    def test_single_level_bound_broadcasts(
        self, model: "Model", midx: pd.MultiIndex
    ) -> None:
        bound = DataArray([5, 6], dims=["l1"], coords={"l1": [0, 1]})
        # Implicit level projection is deprecated (scenario B) — warns until
        # the v1 convention makes it an error.
        with pytest.warns(
            linopy.EvolvingAPIWarning, match=r"broadcasting level subset"
        ):
            var = model.add_variables(upper=bound, coords=[midx], name="x")
        assert var.dims == ("multi",)
        assert (var.data.upper == [5, 5, 6, 6]).all()

    def test_incomplete_level_bound_raises(
        self, model: "Model", midx: pd.MultiIndex
    ) -> None:
        subset = pd.MultiIndex.from_tuples([(0, "a"), (1, "b")], names=("l1", "l2"))
        bound = pd.Series([1, 2], index=subset)
        with pytest.raises(ValueError, match="no value for .* level combination"):
            model.add_variables(upper=bound, coords=[midx], name="x")
