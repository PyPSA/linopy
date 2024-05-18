#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linopy common module.

This module contains commonly used functions.
"""

import operator
import os
from functools import reduce, wraps
from typing import Any, Dict, List, Optional, Union
from warnings import warn

import numpy as np
import pandas as pd
import polars as pl
from numpy import arange
from xarray import DataArray, Dataset, align, apply_ufunc, broadcast
from xarray.core import indexing, utils

from linopy.config import options
from linopy.constants import (
    HELPER_DIMS,
    SIGNS,
    SIGNS_alternative,
    SIGNS_pretty,
    sign_replace_dict,
)


def maybe_replace_sign(sign):
    """
    Replace the sign with an alternative sign if available.

    Parameters:
        sign (str): The sign to be replaced.

    Returns:
        str: The replaced sign.

    Raises:
        ValueError: If the sign is not in the available signs.
    """
    if sign in SIGNS_alternative:
        return sign_replace_dict[sign]
    elif sign in SIGNS:
        return sign
    else:
        raise ValueError(f"Sign {sign} not in {SIGNS} or {SIGNS_alternative}")


def maybe_replace_signs(sign):
    """
    Replace signs with alternative signs if available.

    Parameters:
        sign (np.ndarray): The signs to be replaced.

    Returns:
        np.ndarray: The replaced signs.
    """
    func = np.vectorize(maybe_replace_sign)
    return apply_ufunc(func, sign, dask="parallelized", output_dtypes=[sign.dtype])


def format_string_as_variable_name(name):
    """
    Format a string to a valid python variable name.

    Parameters:
        name (str): The name to be converted.

    Returns:
        str: The formatted name.
    """
    return name.replace(" ", "_").replace("-", "_")


def get_from_list(lst: Optional[list], index: int):
    """
    Returns the element at the specified index of the list, or None if the index
    is out of bounds.
    """
    if lst is None:
        return None
    if not isinstance(lst, list):
        lst = list(lst)
    return lst[index] if 0 <= index < len(lst) else None


def pandas_to_dataarray(
    arr: Union[pd.DataFrame, pd.Series],
    coords: Optional[Union[dict, list]] = None,
    dims: Optional[list] = None,
    **kwargs,
) -> DataArray:
    """
    Convert a pandas DataFrame or Series to a DataArray.

    As pandas objects already have a concept of coordinates, the
    coordinates (index, columns) will be used as coordinates for the DataArray.
    Solely the dimension names can be specified.

    Parameters:
        arr (Union[pd.DataFrame, pd.Series]):
            The input pandas DataFrame or Series.
        coords (Optional[Union[dict, list]]):
            The coordinates for the DataArray. If None, default coordinates will be used.
        dims (Optional[list]):
            The dimensions for the DataArray. If None, the column names of the DataFrame or the index names of the Series will be used.
        **kwargs:
            Additional keyword arguments to be passed to the DataArray constructor.

    Returns:
        DataArray:
            The converted DataArray.
    """
    dims = [
        axis.name or get_from_list(dims, i) or f"dim_{i}"
        for i, axis in enumerate(arr.axes)
    ]
    if coords is not None:
        pandas_coords = dict(zip(dims, arr.axes))
        if isinstance(coords, (list, tuple)):
            coords = dict(zip(dims, coords))
        shared_dims = set(pandas_coords.keys()) & set(coords.keys())
        non_aligned = []
        for dim in shared_dims:
            coord = coords[dim]
            if not isinstance(coord, pd.Index):
                coord = pd.Index(coord)
            if not pandas_coords[dim].equals(coord):
                non_aligned.append(dim)
        if any(non_aligned):
            warn(
                f"coords for dimension(s) {non_aligned} is not aligned with the pandas object. "
                "Previously, the indexes of the pandas were ignored and overwritten in "
                "these cases. Now, the pandas object's coordinates are taken considered"
                " for alignment."
            )

    return DataArray(arr, coords=None, dims=dims, **kwargs)


def numpy_to_dataarray(
    arr: np.ndarray,
    coords: Optional[Union[dict, list]] = None,
    dims: Optional[list] = None,
    **kwargs,
) -> DataArray:
    """
    Convert a numpy array to a DataArray.

    Parameters:
        arr (np.ndarray):
            The input numpy array.
        coords (Optional[Union[dict, list]]):
            The coordinates for the DataArray. If None, default coordinates will be used.
        dims (Optional[list]):
            The dimensions for the DataArray. If None, the dimensions will be automatically generated.
        **kwargs:
            Additional keyword arguments to be passed to the DataArray constructor.

    Returns:
        DataArray:
            The converted DataArray.
    """
    ndim = max(arr.ndim, 0 if coords is None else len(coords))

    dims_given = dims is not None and len(dims)
    if dims_given:
        # fill up dims with default names to match the number of dimensions
        dims = [get_from_list(dims, i) or f"dim_{i}" for i in range(ndim)]

    if isinstance(coords, list) and dims_given:
        coords = dict(zip(dims, coords))

    return DataArray(arr, coords=coords, dims=dims, **kwargs)


def as_dataarray(
    arr,
    coords: Optional[Union[dict, list]] = None,
    dims: Optional[list] = None,
    **kwargs,
) -> DataArray:
    """
    Convert an object to a DataArray.

    Parameters:
        arr:
            The input object.
        coords (Optional[Union[dict, list]]):
            The coordinates for the DataArray. If None, default coordinates will be used.
        dims (Optional[list]):
            The dimensions for the DataArray. If None, the dimensions will be automatically generated.
        **kwargs:
            Additional keyword arguments to be passed to the DataArray constructor.

    Returns:
        DataArray:
            The converted DataArray.
    """
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        arr = pandas_to_dataarray(arr, coords=coords, dims=dims, **kwargs)
    elif isinstance(arr, np.ndarray):
        arr = numpy_to_dataarray(arr, coords=coords, dims=dims, **kwargs)
    elif isinstance(arr, (np.number, int, float, str, bool, list)):
        arr = DataArray(arr, coords=coords, dims=dims, **kwargs)

    elif not isinstance(arr, DataArray):
        supported_types = [
            np.number,
            str,
            bool,
            list,
            pd.Series,
            pd.DataFrame,
            np.ndarray,
            DataArray,
        ]
        supported_types_str = ", ".join([t.__name__ for t in supported_types])
        raise TypeError(
            f"Unsupported type of arr: {type(arr)}. Supported types are: {supported_types_str}"
        )

    arr = fill_missing_coords(arr)
    return arr


# TODO: rename to to_pandas_dataframe
def to_dataframe(ds, mask_func=None):
    """
    Convert an xarray Dataset to a pandas DataFrame.

    This is an memory efficient alternative implementation to the built-in `to_dataframe` method, which
    does not create a multi-indexed DataFrame.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to convert to a DataFrame.
    """
    data = broadcast(ds)[0]
    data = {k: v.values.reshape(-1) for k, v in data.items()}

    if mask_func is not None:
        mask = mask_func(data)
        for k, v in data.items():
            data[k] = v[mask]

    return pd.DataFrame(data, copy=False)


def check_has_nulls(df: pd.DataFrame, name: str):
    any_nan = df.isna().any()
    if any_nan.any():
        fields = ", ".join("`" + df.columns[any_nan] + "`")
        raise ValueError(f"{name} contains nan's in field(s) {fields}")


def infer_schema_polars(ds: Dataset, overwrites: dict[str, pl.DataType]) -> dict:
    """
    Infer the schema for a Polars DataFrame based on the data types of its columns.

    Args:
        ds (polars.DataFrame): The Polars DataFrame for which to infer the schema.

    Returns:
        dict: A dictionary mapping column names to their corresponding Polars data types.
    """
    schema = {}
    for col_name, array in ds.items():
        if col_name in overwrites:
            schema[col_name] = overwrites[col_name]
        elif np.issubdtype(array.dtype, np.integer):
            schema[col_name] = pl.Int32 if os.name == "nt" else pl.Int64
        elif np.issubdtype(array.dtype, np.floating):
            schema[col_name] = pl.Float64
        elif np.issubdtype(array.dtype, np.bool_):
            schema[col_name] = pl.Boolean
        elif np.issubdtype(array.dtype, np.object_):
            schema[col_name] = pl.Object
        else:
            schema[col_name] = pl.Utf8
    return schema


def to_polars(ds: Dataset, **kwargs) -> pl.DataFrame:
    """
    Convert an xarray Dataset to a polars DataFrame.

    This is an memory efficient alternative implementation
    of `to_dataframe`.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to convert to a DataFrame.
    kwargs : dict
        Additional keyword arguments to be passed to the
        DataFrame constructor.
    """
    data = broadcast(ds)[0]
    return pl.LazyFrame({k: v.values.reshape(-1) for k, v in data.items()}, **kwargs)


def check_has_nulls_polars(df: pl.LazyFrame, name: str = "") -> None:
    """
    Checks if the given DataFrame contains any null values and raises a ValueError if it does.

    Args:
        df (pl.DataFrame): The DataFrame to check for null values.
        name (str): The name of the data container being checked.

    Raises:
        ValueError: If the DataFrame contains null values,
        a ValueError is raised with a message indicating the name of the constraint and the fields containing null values.
    """
    has_nulls = df.select(pl.col("*").is_null().any()).collect()
    null_columns = [col for col in has_nulls.columns if has_nulls[col][0]]
    if null_columns:
        raise ValueError(f"{name} contains nan's in field(s) {null_columns}")


def filter_nulls_polars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter out rows containing "empty" values from a polars DataFrame.

    Args:
        df (pl.DataFrame): The DataFrame to filter.

    Returns:
        pl.DataFrame: The filtered DataFrame.
    """
    cond = []
    varcols = [c for c in df.columns if c.startswith("vars")]
    if varcols:
        cond.append(reduce(operator.or_, [pl.col(c).ne(-1) for c in varcols]))
    if "coeffs" in df.columns:
        cond.append(pl.col("coeffs").ne(0))
    if "labels" in df.columns:
        cond.append(pl.col("labels").ne(-1))

    cond = reduce(operator.and_, cond)
    return df.filter(cond)


def group_terms_polars(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Groups terms in a polars DataFrame.

    Args:
        df (pl.DataFrame): The input DataFrame containing the terms.

    Returns:
        pl.DataFrame: The DataFrame with grouped terms.

    """
    varcols = [c for c in df.columns if c.startswith("vars")]
    agg_list = [pl.col("coeffs").sum().alias("coeffs")]
    for col in set(df.columns) - set(["coeffs", "labels", *varcols]):
        agg_list.append(pl.col(col).first().alias(col))

    by = [c for c in ["labels"] + varcols if c in df.columns]
    df = df.group_by(by, maintain_order=True).agg(agg_list)
    return df


def save_join(*dataarrays, integer_dtype=False):
    """
    Join multiple xarray Dataarray's to a Dataset and warn if coordinates are not equal.
    """
    try:
        arrs = align(*dataarrays, join="exact")
    except ValueError:
        warn(
            "Coordinates across variables not equal. Perform outer join.",
            UserWarning,
        )
        arrs = align(*dataarrays, join="outer")
        if integer_dtype:
            arrs = [ds.fillna(-1).astype(int) for ds in arrs]
    return Dataset({ds.name: ds for ds in arrs})


def fill_missing_coords(ds, fill_helper_dims=False):
    """
    Fill coordinates of a xarray Dataset or DataArray with integer coordinates.

    This function fills in the integer coordinates for all dimensions of a
    Dataset or DataArray that have no coordinates assigned yet.

    Parameters
    ----------
    ds : xarray.DataArray or xarray.Dataset
    fill_helper_dims : bool, optional
        Whether to fill in integer coordinates for helper dimensions, by default False.

    """
    ds = ds.copy()
    if not isinstance(ds, (Dataset, DataArray)):
        raise TypeError(f"Expected xarray.DataArray or xarray.Dataset, got {type(ds)}.")

    skip_dims = [] if fill_helper_dims else HELPER_DIMS

    # Fill in missing integer coordinates
    for dim in ds.dims:
        if dim not in ds.coords and dim not in skip_dims:
            ds.coords[dim] = arange(ds.sizes[dim])

    return ds


def _remap(array, mapping):
    return mapping[array.ravel()].reshape(array.shape)


def replace_by_map(ds, mapping):
    """
    Replace values in a DataArray by a one-dimensional mapping.
    """
    return apply_ufunc(
        _remap,
        ds,
        kwargs=dict(mapping=mapping),
        dask="parallelized",
        output_dtypes=[mapping.dtype],
    )


def best_int(max_value):
    """
    Get the minimal int dtype for storing values <= max_value.
    """
    for t in (np.int8, np.int16, np.int32, np.int64):
        if max_value <= np.iinfo(t).max:
            return t


def get_index_map(*arrays):
    """
    Given arrays of hashable objects, create a map from unique combinations to unique integers.
    """
    # Create unique combinations
    unique_combinations = set(zip(*arrays))

    return {combination: i for i, combination in enumerate(unique_combinations)}


def generate_indices_for_printout(dim_sizes, max_lines):
    total_lines = int(np.prod(dim_sizes))
    lines_to_skip = total_lines - max_lines + 1 if total_lines > max_lines else 0
    if lines_to_skip > 0:
        half_lines = max_lines // 2
        for i in range(half_lines):
            yield np.unravel_index(i, dim_sizes)
        yield None
        for i in range(total_lines - half_lines, total_lines):
            yield tuple(np.unravel_index(i, dim_sizes))
    else:
        for i in range(total_lines):
            yield tuple(np.unravel_index(i, dim_sizes))


def align_lines_by_delimiter(lines, delimiter):
    # Determine the maximum position of the delimiter
    if isinstance(delimiter, str):
        delimiter = [delimiter]
    try:
        max_pos = max(line.index(d) for line in lines for d in delimiter if d in line)
    except ValueError:
        return lines

    # Create the formatted lines
    formatted_lines = []
    for line in lines:
        formatted_line = line
        for d in delimiter:
            if d in line:
                parts = line.split(d)
                formatted_line = f"{parts[0]:<{max_pos}}{d} {parts[1].strip()}"
        formatted_lines.append(formatted_line)
    return formatted_lines


def get_label_position(obj, values):
    """
    Get tuple of name and coordinate for variable labels.
    """

    def find_single(value):
        if value == -1:
            return None, None
        for name, val in obj.items():
            labels = val.labels
            start, stop = val.range

            if value >= start and value < stop:
                index = np.unravel_index(value - start, labels.shape)

                # Extract the coordinates from the indices
                coord = {
                    dim: labels.indexes[dim][i] for dim, i in zip(labels.dims, index)
                }

                # Add the name of the DataArray and the coordinates to the result list
                return name, coord

    ndim = np.array(values).ndim
    if ndim == 0:
        return find_single(values)
    elif ndim == 1:
        return [find_single(v) for v in values]
    elif ndim == 2:
        return [[find_single(v) for v in _] for _ in values.T]
    else:
        raise ValueError("Array's with more than two dimensions is not supported")


def print_coord(coord):
    if isinstance(coord, dict):
        coord = coord.values()
    return "[" + ", ".join([str(c) for c in coord]) + "]" if len(coord) else ""


def print_single_variable(model, label):
    if label == -1:
        return "None"

    variables = model.variables
    name, coord = variables.get_label_position(label)

    lower = variables[name].lower.sel(coord).item()
    upper = variables[name].upper.sel(coord).item()

    if variables[name].attrs["binary"]:
        bounds = " ∈ {0, 1}"
    elif variables[name].attrs["integer"]:
        bounds = f" ∈ Z ⋂ [{lower:.4g},...,{upper:.4g}]"
    else:
        bounds = f" ∈ [{lower:.4g}, {upper:.4g}]"

    return f"{name}{print_coord(coord)}{bounds}"


def print_single_expression(c, v, const, model):
    """
    Print a single linear expression based on the coefficients and variables.
    """

    c, v = np.atleast_1d(c), np.atleast_1d(v)

    # catch case that to many terms would be printed
    def print_line(expr, const):
        res = []
        for i, (coeff, var) in enumerate(expr):
            coeff_string = f"{coeff:+.4g}"
            if i:
                # split sign and coefficient
                coeff_string = f"{coeff_string[0]} {coeff_string[1:]}"

            if isinstance(var, list):
                var_string = ""
                for name, coords in var:
                    if name is not None:
                        coord_string = print_coord(coords)
                        var_string += f" {name}{coord_string}"
            else:
                name, coords = var
                coord_string = print_coord(coords)
                var_string = f" {name}{coord_string}"

            res.append(f"{coeff_string}{var_string}")

        if not np.isnan(const) and not (const == 0.0 and len(res) >= 1):
            const_string = f"{const:+.4g}"
            if len(res):
                res.append(f"{const_string[0]} {const_string[1:]}")
            else:
                res.append(const_string)
        return " ".join(res) if len(res) else "None"

    if v.ndim == 1:
        mask = v != -1
        c, v = c[mask], v[mask]
    else:
        # case for quadratic expressions
        mask = (v != -1).any(0)
        c = c[mask]
        v = v[:, mask]

    max_terms = options.get_value("display_max_terms")
    if len(c) > max_terms:
        truncate = max_terms // 2
        positions = model.variables.get_label_position(v[..., :truncate])
        expr = list(zip(c[:truncate], positions))
        res = print_line(expr, const)
        res += " ... "
        expr = list(
            zip(
                c[-truncate:],
                model.variables.get_label_position(v[-truncate:]),
            )
        )
        residual = print_line(expr, const)
        if residual != " None":
            res += residual
        return res
    expr = list(zip(c, model.variables.get_label_position(v)))
    return print_line(expr, const)


def print_single_constraint(model, label):
    constraints = model.constraints
    name, coord = constraints.get_label_position(label)

    coeffs = model.constraints[name].coeffs.sel(coord).values
    vars = model.constraints[name].vars.sel(coord).values
    sign = model.constraints[name].sign.sel(coord).item()
    rhs = model.constraints[name].rhs.sel(coord).item()

    expr = print_single_expression(coeffs, vars, 0, model)
    sign = SIGNS_pretty[sign]

    return f"{name}{print_coord(coord)}: {expr} {sign} {rhs:.12g}"


def has_optimized_model(func):
    """
    Check if a reference model is set.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.model is None:
            raise AttributeError("No reference model set.")
        if self.model.status != "ok":
            raise AttributeError("Underlying model not optimized.")
        return func(self, *args, **kwargs)

    return wrapper


def is_constant(func):
    from linopy import expressions, variables

    @wraps(func)
    def wrapper(self, arg):
        if isinstance(
            arg,
            (
                variables.Variable,
                variables.ScalarVariable,
                expressions.LinearExpression,
            ),
        ):
            raise TypeError(f"Assigned rhs must be a constant, got {type()}).")
        return func(self, arg)

    return wrapper


def forward_as_properties(**routes):
    #
    def add_accessor(cls, item, attr):
        @property
        def get(self):
            return getattr(getattr(self, item), attr)

        setattr(cls, attr, get)

    def deco(cls):
        for item, attrs in routes.items():
            for attr in attrs:
                add_accessor(cls, item, attr)
        return cls

    return deco


def check_common_keys_values(list_of_dicts: List[Dict[str, Any]]) -> bool:
    """
    Check if all common keys among a list of dictionaries have the same value.

    Parameters
    ----------
    list_of_dicts : list of dict
        A list of dictionaries.

    Returns
    -------
    bool
        True if all common keys have the same value across all dictionaries, False otherwise.
    """
    common_keys = set.intersection(*(set(d.keys()) for d in list_of_dicts))
    return all(len({d[k] for d in list_of_dicts if k in d}) == 1 for k in common_keys)


class LocIndexer:
    __slots__ = ("object",)

    def __init__(self, obj):
        self.object = obj

    def __getitem__(self, key) -> Dataset:
        if not utils.is_dict_like(key):
            # expand the indexer so we can handle Ellipsis
            labels = indexing.expanded_indexer(key, self.object.ndim)
            key = dict(zip(self.object.dims, labels))
        return self.object.sel(key)
