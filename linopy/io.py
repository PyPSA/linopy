#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module containing all import/export functionalities."""
import logging
import os
import time
from functools import partial, reduce

import numpy as np
import xarray as xr
from numpy import dtype
from xarray import apply_ufunc, full_like, concat

logger = logging.getLogger(__name__)


ufunc_kwargs = dict(dask="parallelized", vectorize=True)
concat_dim = "_concat_dim"
concat_kwargs = dict(dim=concat_dim, coords="minimal")


# IO functions
def to_float_str(da):
    """Convert a float array to a string array with lp like format for coefficients."""
    return apply_ufunc(lambda f: "%+f" % f, da.fillna(0), **ufunc_kwargs)


def to_int_str(da, nonnans=None):
    """Convert a int array to a string array."""
    return xr.apply_ufunc(lambda d: "%d" % d, da.fillna(0), **ufunc_kwargs)


def sum_strings(da, dim=None):
    """
    Sum all values in the terms dimension. This function is needed in order
    to prevent numpy to change the dtype to unicode. This happens when strings
    of a one-dimensional array is concatenated (good question why...).
    """
    assert len(da.dims) > 1, "Dimensions must be more than one."
    if dim is None:
        dim = [dim for dim in da.dims if "_term" in dim].pop()
    return da.reduce(np.sum, dim)


def join_str(*arraylist):
    """Join string array together (elementwise concatenation of strings)."""
    func = partial(np.add, dtype=object)  # np.core.defchararray.add
    return reduce(func, arraylist, "")


def array_to_file(array, fn):
    """Elementwise writing out string values to a file."""
    write = np.vectorize(fn.write)
    write(array.data.ravel())


def objective_to_file(m, f):
    """Write out the objective of a model to a lp file."""
    f.write("min\nobj:\n")
    coef = m.objective.coeffs
    var = m.objective.vars

    nonnans = coef.notnull() & (var != -1)
    objective = [
        to_float_str(coef),
        full_like(coef, " x", dtype=dtype("<U9")),
        to_int_str(var),
        full_like(coef, "\n", dtype=dtype("<U9")),
    ]
    objective = concat(objective, **concat_kwargs).where(nonnans, "")
    objective = objective.transpose(..., concat_dim)
    array_to_file(objective, f)


def constraints_to_file(m, f):
    """Write out the constraints of a model to a lp file."""
    f.write("\n\ns.t.\n\n")
    labels = m.constraints.labels
    vars = m.constraints.vars
    coeffs = m.constraints.coeffs
    sign = m.constraints.sign
    rhs = m.constraints.rhs

    nonnans_terms = coeffs.notnull() & (vars != -1)
    nonnans = (labels != -1) & sign.notnull() & rhs.notnull()

    for k in labels:
        term_dim = f"{k}_term"

        lhs = [
            to_float_str(coeffs[k]).where(nonnans_terms[k], ""),
            full_like(coeffs[k], " x", dtype=dtype("<U9")).where(nonnans_terms[k], ""),
            to_int_str(vars[k]).where(nonnans_terms[k], ""),
            full_like(vars[k], "\n", dtype=dtype("<U9")).where(nonnans_terms[k], ""),
        ]

        lhs = concat(lhs, **concat_kwargs)
        lhs = lhs.stack(_=[term_dim, concat_dim]).drop("_").rename(_=term_dim)

        newline = full_like(labels[k], "\n", dtype=dtype("<U9"))

        constraints = [
            full_like(labels[k], "c", dtype=dtype("<U9")),
            to_int_str(labels[k]),
            full_like(labels[k], ":\n", dtype=dtype("<U9")),
            lhs,
            sign[k],
            newline,
            to_float_str(rhs[k]),
            newline,
            newline,
        ]

        constraints = concat(constraints, term_dim, coords="minimal")
        constraints = constraints.where(nonnans[k], "").transpose(..., term_dim)
        array_to_file(constraints, fn=f)


def bounds_to_file(m, f):
    """Write out variables of a model to a lp file."""
    f.write("\nbounds\n")
    lower = m.variables.lower[m._non_binary_variables]
    labels = m.variables.labels[m._non_binary_variables]
    upper = m.variables.upper[m._non_binary_variables]

    nonnans = lower.notnull() & upper.notnull() & (labels != -1)

    # iterate over dataarray to reduce memory usage
    for k in labels:
        bounds = [
            to_float_str(lower[k]),
            full_like(labels[k], " <= x", dtype=dtype("<U9")),
            to_int_str(labels[k]),
            full_like(labels[k], " <= ", dtype=dtype("<U9")),
            to_float_str(upper[k]),
            full_like(labels[k], "\n", dtype=dtype("<U9")),
        ]

        bounds = concat(bounds, **concat_kwargs).where(nonnans[k], "")
        bounds = bounds.transpose(..., concat_dim)
        array_to_file(bounds, fn=f)


def binaries_to_file(m, f):
    """Write out binaries of a model to a lp file."""
    f.write("\nbinary\n")
    labels = m.binaries.labels

    nonnans = labels != -1

    # iterate over dataarray to reduce memory usage
    for k in labels:
        binaries = [
            full_like(labels[k], "x", dtype=dtype("<U9")),
            to_int_str(labels[k]),
            full_like(labels[k], "\n", dtype=dtype("<U9")),
        ]

        binaries = concat(binaries, **concat_kwargs).where(nonnans[k], "")
        binaries = binaries.transpose(..., concat_dim)
        array_to_file(binaries, fn=f)


def to_file(m, fn):
    """Write out a model to a lp file."""
    if os.path.exists(fn):
        os.remove(fn)  # ensure a clear file

    with open(fn, mode="w") as f:

        start = time.time()

        objective_to_file(m, f)
        constraints_to_file(m, f)
        bounds_to_file(m, f)
        binaries_to_file(m, f)
        f.write("end\n")

        logger.info(f" Writing time: {round(time.time()-start, 2)}s")


def non_bool_dict(d):
    """Convert bool to int for netCDF4 storing"""
    return {k: v if not isinstance(v, bool) else int(v) for k, v in d.items()}


def to_netcdf(m, *args, **kwargs):
    """
    Write out the model to a netcdf file.

    Parameters
    ----------
    m : linopy.Model
        Model to write out.
    *args
        Arguments passed to ``xarray.Dataset.to_netcdf``.
    **kwargs : TYPE
        Keyword arguments passed to ``xarray.Dataset.to_netcdf``.

    """

    def get_and_rename(m, attr, prefix=""):
        ds = getattr(m, attr)
        return ds.rename({v: prefix + attr + "-" + v for v in ds})

    vars = [
        get_and_rename(m.variables, attr, "variables_")
        for attr in m.variables.dataset_attrs
    ]
    cons = [
        get_and_rename(m.constraints, attr, "constraints_")
        for attr in m.constraints.dataset_attrs
    ]
    others = [get_and_rename(m, d) for d in m.dataset_attrs + ["objective"]]
    scalars = {k: getattr(m, k) for k in m.scalar_attrs}
    ds = xr.merge(vars + cons + others).assign_attrs(scalars)

    for k in ds:
        ds[k].attrs = non_bool_dict(ds[k].attrs)

    ds.to_netcdf(*args, **kwargs)


def read_netcdf(path, **kwargs):
    """
    Read in a model from a netcdf file.

    Parameters
    ----------
    path : path_like
        Path of the stored model.
    **kwargs
        Keyword arguments passed to ``xarray.load_dataset``.

    Returns
    -------
    m : linopy.Model

    """
    from linopy.model import Constraints, LinearExpression, Model, Variables

    m = Model()
    all_ds = xr.load_dataset(path, **kwargs)

    def get_and_rename(ds, attr, prefix=""):
        keys = [k for k in ds if k.startswith(prefix + attr + "-")]
        return ds[keys].rename({k: k[len(prefix + attr) + 1 :] for k in keys})

    attrs = Variables.dataset_attrs
    kwargs = {attr: get_and_rename(all_ds, attr, "variables_") for attr in attrs}
    m.variables = Variables(**kwargs, model=m)

    attrs = Constraints.dataset_attrs
    kwargs = {attr: get_and_rename(all_ds, attr, "constraints_") for attr in attrs}
    m.constraints = Constraints(**kwargs, model=m)

    for attr in m.dataset_attrs + ["objective"]:
        setattr(m, attr, get_and_rename(all_ds, attr))
    m.objective = LinearExpression(m.objective)

    for k in m.scalar_attrs:
        setattr(m, k, all_ds.attrs.pop(k))

    return m
