#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module containing all import/export functionalities."""
import logging
import os
import time

import numpy as np
import xarray as xr
from numpy import dtype
from xarray import DataArray, apply_ufunc, concat

logger = logging.getLogger(__name__)


ufunc_kwargs = dict(vectorize=True)
concat_dim = "_concat_dim"
concat_kwargs = dict(dim=concat_dim, coords="minimal")


# IO functions
def to_float_str(da):
    """Convert a float array to a string array with lp like format for coefficients."""
    return apply_ufunc(lambda f: "%+f" % f, da.fillna(0).compute(), **ufunc_kwargs)


def to_int_str(da):
    """Convert a int array to a string array."""
    return xr.apply_ufunc(lambda d: "%d" % d, da.fillna(0).compute(), **ufunc_kwargs)


def array_to_file(array, fn):
    """Elementwise writing out string values to a file."""
    write = np.frompyfunc(fn.write, 1, 1)

    def func(data):
        return write(data.ravel()).reshape(data.shape)

    return xr.apply_ufunc(func, array)


def objective_to_file(m, f):
    """Write out the objective of a model to a lp file."""
    f.write("min\nobj:\n")
    coef = m.objective.coeffs
    var = m.objective.vars

    objective = [
        to_float_str(coef),
        DataArray(" x").astype(dtype("<U9")),
        to_int_str(var),
        DataArray("\n").astype(dtype("<U9")),
    ]
    nonnans = coef.notnull() & (var != -1)
    objective = concat(objective, **concat_kwargs).where(nonnans, "")
    objective = objective.transpose(..., concat_dim)
    array_to_file(objective.compute(), f)


def constraints_to_file(m, f):
    """Write out the constraints of a model to a lp file."""
    f.write("\n\ns.t.\n\n")
    for name, labels in m.constraints.labels.items():

        dim = f"{name}_term"

        lhs = [
            to_float_str(m.constraints.coeffs[name]),
            DataArray(" x").astype(dtype("<U9")),
            to_int_str(m.constraints.vars[name]),
            DataArray("\n").astype(dtype("<U9")),
        ]

        nonnans_terms = m.constraints.vars[name] != -1
        lhs = concat(lhs, **concat_kwargs).where(nonnans_terms, "")
        lhs = lhs.stack(_=[dim, concat_dim]).drop("_").rename(_=dim)

        constraints = [
            DataArray("c").astype(dtype("<U9")),
            to_int_str(labels),
            DataArray(":\n").astype(dtype("<U9")),
            lhs,
            m.constraints.sign[name],
            DataArray("\n").astype(dtype("<U9")),
            to_float_str(m.constraints.rhs[name]),
            DataArray("\n\n").astype(dtype("<U9")),
        ]

        da = concat(constraints, dim=dim, coords="minimal")
        da = da.where(labels != -1, "").transpose(..., dim)
        array_to_file(da.compute(), fn=f)


def bounds_to_file(m, f):
    """Write out variables of a model to a lp file."""
    f.write("\nbounds\n")
    for name, labels in m.variables.labels[m._non_binary_variables].items():

        bounds = [
            to_float_str(m.variables.lower[name]),
            DataArray(" <= x").astype(dtype("<U9")),
            to_int_str(labels),
            DataArray(" <= ").astype(dtype("<U9")),
            to_float_str(m.variables.upper[name]),
            DataArray("\n").astype(dtype("<U9")),
        ]

        bounds = concat(bounds, **concat_kwargs).where(labels != -1, "")
        bounds = bounds.transpose(..., concat_dim)
        array_to_file(bounds.compute(), f)


def binaries_to_file(m, f):
    """Write out binaries of a model to a lp file."""
    f.write("\nbinary\n")
    for name, labels in m.binaries.labels.items():

        binaries = [
            DataArray("x").astype(dtype("<U9")),
            to_int_str(labels),
            DataArray("\n").astype(dtype("<U9")),
        ]

        binaries = concat(binaries, **concat_kwargs).where(labels != -1, "")
        binaries = binaries.transpose(..., concat_dim)
        array_to_file(binaries.compute(), fn=f)


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
