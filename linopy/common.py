#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linopy common module.

This module contains commonly used functions.
"""

from functools import partialmethod, update_wrapper, wraps

import numpy as np
from numpy import arange, hstack
from xarray import DataArray, apply_ufunc, merge

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


def as_dataarray(arr):
    """
    Convert an object to a DataArray if it is not already a DataArray.
    """
    if not isinstance(arr, DataArray):
        return DataArray(arr)
    return arr


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


def head_tail_range(stop, max_number_of_values=14):
    split_at = max_number_of_values // 2
    if stop > max_number_of_values:
        return hstack([arange(split_at), arange(stop - split_at, stop)])
    else:
        return arange(stop)


def print_coord(coord):
    if len(coord):
        return "[" + ", ".join([str(c) for c in coord]) + "]"
    else:
        return ""


def print_single_variable(lower, upper, var, vartype):
    if vartype == "Binary Variable":
        return f"\n {var}"
    else:
        return f"\n{lower} ≤  {var} ≤ {upper}"


def print_single_expression(c, v, model):
    """
    Print a single linear expression based on the coefficients and variables.
    """
    # catch case that to many terms would be printed
    v = model.variables.get_label_position(v)

    def print_line(expr):
        res = ""
        for coeff, (name, coord) in expr:
            coord_string = print_coord(coord)
            res += f" {coeff:+} {name}{coord_string} "
        return res if res else " None"

    if len(c) > 6:
        expr = list(zip(c[:3], v[:3]))
        res = print_line(expr)
        res += "... "
        expr = list(zip(c[-3:], v[-3:]))
        res += print_line(expr)
        return res
    expr = list(zip(c, v))
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

    @wraps(func)
    def wrapper(self, arg):
        if isinstance(arg, (variables.Variable, expressions.LinearExpression)):
            raise TypeError(f"Assigned rhs must be a constant, got {type()}).")
        return func(self, arg)

    return wrapper


def forward_as_properties(**routes):
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
