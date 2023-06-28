#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linopy common module.

This module contains commonly used functions.
"""

import hashlib
from functools import partialmethod, update_wrapper, wraps
from typing import Any, Dict, List, Optional, Union
from warnings import warn

import numpy as np
import pandas as pd
from numpy import arange, hstack
from xarray import DataArray, Dataset, align, apply_ufunc, merge
from xarray.core import indexing, utils

from linopy.config import options
from linopy.constants import (
    HELPER_DIMS,
    SIGNS,
    TERM_DIM,
    SIGNS_alternative,
    SIGNS_pretty,
    sign_replace_dict,
)


def maybe_replace_sign(sign):
    if sign in SIGNS_alternative:
        return sign_replace_dict[sign]
    elif sign in SIGNS:
        return sign
    else:
        raise ValueError(f"Sign {sign} not in {SIGNS} or {SIGNS_alternative}")


def maybe_replace_signs(sign):
    func = np.vectorize(maybe_replace_sign)
    return apply_ufunc(func, sign, dask="parallelized", output_dtypes=[sign.dtype])


def as_dataarray(
    arr,
    coords: Optional[Union[dict, list]] = None,
    dims: Optional[list] = None,
    **kwargs,
) -> DataArray:
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        if dims is not None:
            dims = [axis.name or list(dims)[i] for i, axis in enumerate(arr.axes)]
        if coords is not None and not isinstance(coords, list):
            coords = {dim: coords[dim] for dim in dims}
        arr = DataArray(arr, coords=coords, dims=dims, **kwargs)

    elif isinstance(arr, np.ndarray):
        ndim = max(arr.ndim, 0 if coords is None else len(coords))

        if dims is not None and len(dims):
            # ensure dims is defined for ndim
            dims = list(dims)[:ndim] if dims else []
            dims = dims + [f"dim_{i}" for i in range(len(dims), ndim)]

        if isinstance(coords, list) and dims is not None and len(dims):
            coords = dict(zip(dims, coords))

        arr = DataArray(arr, coords=coords, dims=dims, **kwargs)

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


def save_join(*dataarrays):
    """
    Join multiple xarray Dataarray's to a Dataset and warn if coordinates are not equal.
    """
    try:
        labels = align(*dataarrays, join="exact")
    except ValueError:
        warn("Coordinates across variables not equal. Perform outer join.", UserWarning)
        labels = align(*dataarrays, join="outer")
        labels = [ds.fillna(-1).astype(int) for ds in labels]
    return Dataset({ds.name: ds for ds in labels})


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

        if const != 0:
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
            zip(c[-truncate:], model.variables.get_label_position(v[-truncate:]))
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
