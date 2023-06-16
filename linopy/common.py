#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linopy common module.

This module contains commonly used functions.
"""

from functools import partialmethod, update_wrapper, wraps
from typing import Union
from warnings import warn

import numpy as np
from numpy import arange, hstack
from xarray import DataArray, Dataset, align, apply_ufunc, merge
from xarray.core import indexing, utils

from linopy.config import options
from linopy.constants import SIGNS, SIGNS_alternative, sign_replace_dict


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


def _merge_inplace(self, attr, da, name, **kwargs):
    """
    Assign a new dataarray to the dataset `attr` by merging.

    This takes care of all coordinate alignments, instead of a direct
    assignment like self.variables[name] = var
    """
    ds = merge([getattr(self, attr), da.rename(name)], **kwargs)
    setattr(self, attr, ds)


def as_dataarray(arr, coords=None):
    """
    Convert an object to a DataArray if it is not already a DataArray.
    """
    if not isinstance(arr, DataArray):
        return DataArray(arr, coords=coords)
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


def dictsel(d, keys):
    "Reduce dictionary to keys that appear in selection."
    return {key: d[key] for key in keys}


def head_tail_range(stop, max_number_of_values=14):
    split_at = max_number_of_values // 2
    if stop > max_number_of_values:
        return hstack([arange(split_at), arange(stop - split_at, stop)])
    else:
        return arange(stop)


def print_coord(coord):
    if isinstance(coord, dict):
        coord = coord.values()
    if len(coord):
        return "[" + ", ".join([str(c) for c in coord]) + "]"
    else:
        return ""


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


def print_single_expression(c, v, model):
    """
    Print a single linear expression based on the coefficients and variables.
    """

    c, v = np.atleast_1d(c), np.atleast_1d(v)

    # catch case that to many terms would be printed
    def print_line(expr):
        res = []
        for i, (coeff, (name, coord)) in enumerate(expr):
            coord_string = print_coord(coord)
            if i:
                # split sign and coefficient
                coeff_string = f"{float(coeff):+.4g}"
                res.append(f"{coeff_string[0]} {coeff_string[1:]} {name}{coord_string}")
            else:
                res.append(f"{float(coeff):+.4g} {name}{coord_string}")
        return " ".join(res) if len(res) else "None"

    if isinstance(c, np.ndarray):
        mask = v != -1
        c, v = c[mask], v[mask]

    max_terms = options.get_value("display_max_terms")
    if len(c) > max_terms:
        truncate = max_terms // 2
        expr = list(zip(c[:truncate], model.variables.get_label_position(v[:truncate])))
        res = print_line(expr)
        res += " ... "
        expr = list(
            zip(c[-truncate:], model.variables.get_label_position(v[-truncate:]))
        )
        residual = print_line(expr)
        if residual != " None":
            res += residual
        return res
    expr = list(zip(c, model.variables.get_label_position(v)))
    return print_line(expr)


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

    #
    @wraps(func)
    def wrapper(self, arg):
        if isinstance(arg, (variables.Variable, expressions.LinearExpression)):
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
