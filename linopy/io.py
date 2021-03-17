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
import os
import shutil
import time
import logging
logger = logging.getLogger(__name__)


# IO functions
def to_float_str(da):
    func = np.vectorize(lambda f: '%+f'%f, otypes=[object])
    return xr.apply_ufunc(func, da, dask='parallelized', output_dtypes=[object])


def to_int_str(da):
    func = np.vectorize(lambda d: '%d'%d, otypes=[object])
    return xr.apply_ufunc(func, da, dask='parallelized', output_dtypes=[object])


def join_str_arrays(*arrays):
    func = lambda *l: np.add(*l, dtype=object) # np.core.defchararray.add
    return reduce(func, arrays, '')


def str_array_to_file(array, fn):
    return xr.apply_ufunc(lambda x: fn.write(x), array, dask='parallelized',
                          vectorize=True, output_dtypes=[int])


def objective_to_file(model, f):
        f.write('\* LOPF *\n\nmin\nobj:\n')
        # breakpoint()
        objective_str = join_str_arrays(
            to_float_str(model.objective.coefficients),
            ' x', to_int_str(model.objective.variables), '\n'
            )
        str_array_to_file(objective_str, f).compute()


def constraints_to_file(model, f):
        f.write("\n\ns.t.\n\n")

        lhs_str = join_str_arrays(
            to_float_str(model.constraints_lhs_coeffs),
            ' x', to_int_str(model.constraints_lhs_vars), '\n'
            ).reduce(np.sum, 'term_') # .sum() does not work

        constraints_str = join_str_arrays(
            'c', to_int_str(model.constraints), ': \n',
            lhs_str,
            model.constraints_sign, '\n',
            to_float_str(model.constraints_rhs), '\n\n')
        str_array_to_file(constraints_str, f).compute()


def bounds_to_file(model, f):
        f.write("\nbounds\n")

        bounds_str = join_str_arrays(
            to_float_str(model.variables_lower_bounds),
            ' <= x',to_int_str(model.variables),
            ' <= ', to_float_str(model.variables_upper_bounds), '\n')
        str_array_to_file(bounds_str, f).compute()


def binaries_to_file(model, f):
        f.write("\nbinary\n")

        binaries_str = join_str_arrays(to_int_str(model.binaries))
        str_array_to_file(binaries_str, f).compute()
        f.write("end\n")


def to_file(model, fn):

    if os.path.exists(fn):
        os.remove(fn)  # ensure a clear file

    f = open(fn, mode='w')

    start = time.time()

    objective_to_file(model, f)
    constraints_to_file(model, f)
    bounds_to_file(model, f)
    binaries_to_file(model, f)

    logger.info(f' Writing time: {round(time.time()-start, 2)}s')


