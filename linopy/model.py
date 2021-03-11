# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import dask
import xarray as xr
import numpy as np
import os
import shutil
from tempfile import mkstemp
from xarray import DataArray
from numpy import inf
from functools import reduce


class Model:


    def __init__(self, solver_dir=None, chunk=100):
        self._xCounter = 1
        self._cCounter = 1
        self.chunk = chunk

        self.variables = xr.Dataset()
        self.variables_lower_bounds = xr.Dataset()
        self.variables_upper_bounds = xr.Dataset()

        self.constraints = xr.Dataset()
        self.constraints_lhs = xr.Dataset()
        self.constraints_sign = xr.Dataset()
        self.constraints_rhs = xr.Dataset()

        self.solver_dir = solver_dir


    def add_variables(self, name, lower=-inf, upper=inf, coords=None):

        assert name not in self.variables
        # TODO: warning if name is like var names (x100).

        lower = DataArray(lower).chunk(self.chunk)
        upper = DataArray(upper).chunk(self.chunk)

        if coords is None:
            # only a lazy calculation for extracting coords, shape and size
            coords = (lower + upper).coords

        reslike = DataArray(coords=coords)

        start = self._xCounter
        var = np.arange(start, start + reslike.size).reshape(reslike.shape)
        self._xCounter += reslike.size
        var = xr.DataArray(var, coords=reslike.coords).chunk(self.chunk)

        self.variables = self.variables.assign({name: var})
        self.variables_lower_bounds = self.variables_lower_bounds.assign({name: lower})
        self.variables_upper_bounds = self.variables_upper_bounds.assign({name: upper})
        return var


    def add_constraints(self, name, lhs, sign, rhs):

        assert name not in self.constraints

        lhs = DataArray(lhs).chunk(self.chunk)
        sign = DataArray(sign).chunk(self.chunk)
        rhs = DataArray(rhs).chunk(self.chunk)

        # only a lazy calculation for extracting coords, shape and size
        reslike = join_exprs(lhs, sign, rhs)

        start = self._cCounter
        con = np.arange(start, start + reslike.size).reshape(reslike.shape)
        self._cCounter += reslike.size
        con = xr.DataArray(con, coords=reslike.coords).chunk(self.chunk)
        self.constraints = self.constraints.assign({name: con})

        # TODO: assert (or warning) when dimensions are not overlapping
        self.constraints_lhs = self.constraints_lhs.assign({name: lhs})
        self.constraints_sign = self.constraints_sign.assign({name: sign})
        self.constraints_rhs = self.constraints_rhs.assign({name: rhs})



    def linexpr(self, *tuples):
        return linexpr(*tuples, model=self)


    def __repr__(self):
        return (f"<Linopy model>\n"
                f"Variables: {', '.join(self.variables)}\n"
                f"Constraints: {', '.join(self.constraints)}\n"
                f"Dimensions: {', '.join(self.variables.indexes)}")


    def to_file(self, keep_files=False):
        tmpkwargs = dict(text=True, dir=self.solver_dir)

        fdo, objective_fn = mkstemp('.txt', 'objectve-', **tmpkwargs)
        fdc, constraints_fn = mkstemp('.txt', 'constraints-', **tmpkwargs)
        fdb, bounds_fn = mkstemp('.txt', 'bounds-', **tmpkwargs)
        fdi, binaries_fn = mkstemp('.txt', 'binaries-', **tmpkwargs)
        fdp, problem_fn = mkstemp('.lp', 'problem-', **tmpkwargs)

        self.objective_f = open(objective_fn, mode='w')
        self.constraints_f = open(constraints_fn, mode='w')
        self.bounds_f = open(bounds_fn, mode='w')
        self.binaries_f = open(binaries_fn, mode='w')

        self.objective_f.write('\* LOPF *\n\nmin\nobj:\n')
        self.constraints_f.write("\n\ns.t.\n\n")
        self.bounds_f.write("\nbounds\n")
        self.binaries_f.write("\nbinary\n")


        # write everything...

        self.binaries_f.write("end\n")

        # explicit closing with file descriptor is necessary for windows machines
        for f, fd in (('bounds_f', fdb), ('constraints_f', fdc),
                      ('objective_f', fdo), ('binaries_f', fdi)):
            getattr(self, f).close(); delattr(self, f); os.close(fd)

        # concat files
        with open(problem_fn, 'wb') as wfd:
            for f in [objective_fn, constraints_fn, bounds_fn, binaries_fn]:
                with open(f,'rb') as fd:
                    shutil.copyfileobj(fd, wfd)
                if not keep_files:
                    os.remove(f)

        logger.info(f'Total preparation time: {round(time.time()-start, 2)}s')
        return fdp, problem_fn



def to_float_str(da):
    func = np.vectorize(lambda f: '%+f'%f, otypes=[object])
    return xr.apply_ufunc(func, da, dask='parallelized', output_dtypes=[object])

def to_int_str(da):
    func = np.vectorize(lambda d: '%d'%d, otypes=[object])
    return xr.apply_ufunc(func, da, dask='parallelized', output_dtypes=[object])



def linexpr(*tuples, model=None, chunk=None):
    expr = xr.DataArray('').astype(object)
    chunk = 100 if model is None else model.chunk
    for coeff, var in tuples:
        if isinstance(var, str) and model is not None:
            var = model.variables[var]
        if not isinstance(coeff, DataArray):
            coeff = DataArray(coeff).chunk(chunk)
        if not isinstance(var, DataArray):
            var = DataArray(var).chunk(chunk)
        expr = join_exprs(expr, to_float_str(coeff), ' x', to_int_str(var), '\n')
    # ensure dtype is object (necessary when vars are only scalars)
    return expr if expr.dtype == object else expr.astype(object)


def join_exprs(*arrays):
    func = lambda *l: np.add(*l, dtype=object) # np.core.defchararray.add
    return reduce(func, arrays, '')



m = Model()

lower = xr.DataArray(np.zeros((10,10)), coords=[range(10), range(10)])
upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
m.add_variables('var1', lower, upper)

m.add_variables('var2')

lhs = m.linexpr((1, 'var1'), (10, 'var2'))

m.linexpr((1, 1), (10, 'var1'))


