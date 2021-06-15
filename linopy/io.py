#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module containing all import/export functionalities."""
import logging
import os
import time
from functools import reduce

import numpy as np
import xarray as xr
from xarray import apply_ufunc

logger = logging.getLogger(__name__)


ufunc_kwargs = dict(dask="parallelized", vectorize=True, output_dtypes=[object])

# IO functions
def to_float_str(da):
    """Convert a float array to a string array with lp like format for coefficients."""
    return apply_ufunc(lambda f: "%+f" % f, da.fillna(0), **ufunc_kwargs)


def to_int_str(da, nonnans=None):
    """Convert a int array to a string array."""
    return xr.apply_ufunc(lambda d: "%d" % d, da.fillna(0), **ufunc_kwargs)


def join_str_arrays(arraylist):
    """Join string array together (elementwise concatenation of strings)."""
    func = lambda *l: np.add(*l, dtype=object)  # np.core.defchararray.add
    return reduce(func, arraylist, "")


def str_array_to_file(array, fn):
    """Elementwise writing out string values to a file."""
    return xr.apply_ufunc(
        lambda x: fn.write(x),
        array,
        dask="parallelized",
        vectorize=True,
        output_dtypes=[int],
    )


def objective_to_file(m, f):
    """Write out the objective of a model to a lp file."""
    f.write("min\nobj:\n")
    coef = m.objective.coeffs
    var = m.objective.vars

    nonnans = coef.notnull() & var.notnull()
    join = [to_float_str(coef), " x", to_int_str(var), "\n"]
    objective_str = join_str_arrays(join).where(nonnans, "")
    str_array_to_file(objective_str, f).compute()


def constraints_to_file(m, f):
    """Write out the constraints of a model to a lp file."""
    f.write("\n\ns.t.\n\n")
    con = m.constraints
    coef = m.constraints_lhs_coeffs
    var = m.constraints_lhs_vars
    sign = m.constraints_sign
    rhs = m.constraints_rhs

    nonnans = coef.notnull() & var.notnull()
    join = [to_float_str(coef), " x", to_int_str(var), "\n"]
    lhs_str = join_str_arrays(join).where(nonnans, "").reduce(np.sum, "_term")
    # .sum() does not work

    nonnans = nonnans.any("_term") & con.notnull() & sign.notnull() & rhs.notnull()

    join = [
        "c",
        to_int_str(con),
        ": \n",
        lhs_str,
        sign,
        "\n",
        to_float_str(rhs),
        "\n\n",
    ]
    constraints_str = join_str_arrays(join).where(nonnans, "")
    str_array_to_file(constraints_str, f).compute()


def bounds_to_file(m, f):
    """Write out variables of a model to a lp file."""
    f.write("\nbounds\n")
    lb = m.variables_lower_bound
    v = m.variables
    ub = m.variables_upper_bound

    nonnans = lb.notnull() & v.notnull() & ub.notnull()
    join = [to_float_str(lb), " <= x", to_int_str(v), " <= ", to_float_str(ub), "\n"]
    bounds_str = join_str_arrays(join).where(nonnans, "")
    str_array_to_file(bounds_str, f).compute()


def binaries_to_file(m, f):
    """Write out binaries of a model to a lp file."""
    f.write("\nbinary\n")

    binaries_str = join_str_arrays([to_int_str(m.binaries)])
    str_array_to_file(binaries_str, f).compute()
    f.write("end\n")


def to_file(m, fn):
    """Write out a model to a lp file."""
    if os.path.exists(fn):
        os.remove(fn)  # ensure a clear file

    f = open(fn, mode="w")

    start = time.time()

    objective_to_file(m, f)
    constraints_to_file(m, f)
    bounds_to_file(m, f)
    binaries_to_file(m, f)

    f.close()
    logger.info(f" Writing time: {round(time.time()-start, 2)}s")


all_ds_attrs = [
    "variables",
    "variables_lower_bound",
    "variables_upper_bound",
    "binaries",
    "constraints",
    "constraints_lhs_coeffs",
    "constraints_lhs_vars",
    "constraints_sign",
    "constraints_rhs",
    "solution",
    "dual",
    "objective",
]
all_obj_attrs = ["objective_value", "status", "_xCounter", "_cCounter"]


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

    def get_and_rename(m, attr):
        ds = getattr(m, attr)
        return ds.rename({v: attr + "-" + v for v in ds})

    ds = xr.merge([get_and_rename(m, d) for d in all_ds_attrs])
    ds = ds.assign_attrs({k: getattr(m, k) for k in all_obj_attrs})

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
    from .model import LinearExpression, Model  # avoid cyclic imports

    m = Model()
    all_ds = xr.load_dataset(path, **kwargs)

    for attr in all_ds_attrs:
        keys = [k for k in all_ds if k.startswith(attr + "-")]
        ds = all_ds[keys].rename({k: k[len(attr) + 1 :] for k in keys})
        setattr(m, attr, ds)
    m.objective = LinearExpression(m.objective)

    for k in all_obj_attrs:
        setattr(m, k, ds.attrs.pop(k))

    return m
