#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:17:37 2021

@author: fabian
"""
import numpy as np
import xarray as xr
from functools import reduce
from tempfile import mkstemp
from xarray import apply_ufunc
import os
import shutil
import time
import logging
logger = logging.getLogger(__name__)


ufunc_kwargs = dict(dask='parallelized', vectorize=True, output_dtypes=[object])

# IO functions
def to_float_str(da):
    return apply_ufunc(lambda f: '%+f'%f, da.fillna(0), **ufunc_kwargs)


def to_int_str(da, nonnans=None):
    return xr.apply_ufunc(lambda d: '%d'%d, da.fillna(0), **ufunc_kwargs)


def join_str_arrays(arraylist):
    func = lambda *l: np.add(*l, dtype=object) # np.core.defchararray.add
    return reduce(func, arraylist, '')


def str_array_to_file(array, fn):
    return xr.apply_ufunc(lambda x: fn.write(x), array, dask='parallelized',
                          vectorize=True, output_dtypes=[int])


def objective_to_file(m, f):
        f.write('min\nobj:\n')
        coef = m.objective.coeffs
        var = m.objective.vars

        nonnans = coef.notnull() & var.notnull()
        join = [to_float_str(coef), ' x', to_int_str(var), '\n']
        objective_str = join_str_arrays(join).where(nonnans, '')
        str_array_to_file(objective_str, f).compute()


def constraints_to_file(m, f):
        f.write("\n\ns.t.\n\n")
        con = m.constraints
        coef = m.constraints_lhs_coeffs
        var = m.constraints_lhs_vars
        sign = m.constraints_sign
        rhs = m.constraints_rhs

        nonnans = coef.notnull() & var.notnull()
        join = [to_float_str(coef), ' x', to_int_str(var), '\n']
        lhs_str = join_str_arrays(join).where(nonnans, '').reduce(np.sum, 'term_')
        # .sum() does not work

        nonnans = (nonnans.any('term_') & con.notnull() &
                   sign.notnull() & rhs.notnull())

        join = ['c', to_int_str(con), ': \n', lhs_str, sign, '\n',
                to_float_str(rhs), '\n\n']
        constraints_str = join_str_arrays(join).where(nonnans, '')
        str_array_to_file(constraints_str, f).compute()


def bounds_to_file(m, f):
        f.write("\nbounds\n")
        lb = m.variables_lower_bounds
        v = m.variables
        ub = m.variables_upper_bounds

        nonnans = lb.notnull() & v.notnull() & ub.notnull()
        join = [to_float_str(lb), ' <= x', to_int_str(v), ' <= ', to_float_str(ub), '\n']
        bounds_str = join_str_arrays(join).where(nonnans, '')
        str_array_to_file(bounds_str, f).compute()


def binaries_to_file(m, f):
        f.write("\nbinary\n")

        binaries_str = join_str_arrays([to_int_str(m.binaries)])
        str_array_to_file(binaries_str, f).compute()
        f.write("end\n")


def to_file(m, fn):

    if os.path.exists(fn):
        os.remove(fn)  # ensure a clear file

    f = open(fn, mode='w')

    start = time.time()

    objective_to_file(m, f)
    constraints_to_file(m, f)
    bounds_to_file(m, f)
    binaries_to_file(m, f)

    f.close()
    logger.info(f' Writing time: {round(time.time()-start, 2)}s')


