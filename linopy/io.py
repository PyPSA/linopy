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
    # TODO: sometimes lines are written out two times
    func = np.vectorize(lambda x: fn.write(x))
    return xr.apply_ufunc(func, array, dask='parallelized', output_dtypes=[int])



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



def objective_to_file(model, f):
        f.write('\* LOPF *\n\nmin\nobj:\n')

        objective_str = join_str_arrays(
            to_float_str(model.objective.coefficients),
            ' x', to_int_str(model.objective.variables), '\n'
            ).expand_dims('objective').sum('term_') # .sum() does not work
        str_array_to_file(objective_str, f).compute()



def to_file(model, fn=None, tmp_dir=None, keep_files=False):

    if tmp_dir is None:
        tmp_dir = model.solver_dir

    tmpkwargs = dict(text=True, dir=model.solver_dir)
    if fn is None:
        fdp, fn = mkstemp('.lp', 'linopy-problem-', **tmpkwargs)

    fdo, objective_fn = mkstemp('.txt', 'linopy-objectve-', **tmpkwargs)
    fdc, constraints_fn = mkstemp('.txt', 'linopy-constraints-', **tmpkwargs)
    fdb, bounds_fn = mkstemp('.txt', 'linopy-bounds-', **tmpkwargs)
    fdi, binaries_fn = mkstemp('.txt', 'linopy-binaries-', **tmpkwargs)
    fdp, problem_fn = mkstemp('.lp', 'linopy-problem-', **tmpkwargs)


    objective_f = open(objective_fn, mode='w')
    constraints_f = open(constraints_fn, mode='w')
    bounds_f = open(bounds_fn, mode='w')
    binaries_f = open(binaries_fn, mode='w')


    try:
        bounds_to_file(model, bounds_f)
        constraints_to_file(model, constraints_f)
        objective_to_file(model, objective_f)
        binaries_to_file(model, binaries_f)

        # concat files
        with open(problem_fn, 'wb') as wfd:
            for f in [objective_fn, constraints_fn, bounds_fn, binaries_fn]:
                with open(f,'rb') as fd:
                    shutil.copyfileobj(fd, wfd)
    finally:
        for f in [objective_fn, constraints_fn, bounds_fn, binaries_fn]:
            if not keep_files:
                os.remove(f)

    # logger.info(f'Total preparation time: {round(time.time()-start, 2)}s')
    return fdp, problem_fn


