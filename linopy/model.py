# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import xarray as xr
import numpy as np
import os
import shutil
from tempfile import mkstemp
from xarray import DataArray, Dataset
from numpy import inf
from functools import reduce
import logging

logger = logging.getLogger(__name__)

class Model:


    def __init__(self, solver_dir=None, chunk=100):
        # TODO maybe allow the model to be non-lazy, perhaps with an attribute
        # self._is_lazy and a maybe_chunk function
        self._xCounter = 1
        self._cCounter = 1
        self.chunk = chunk

        self.variables = Dataset()
        self.variables_lower_bounds = Dataset()
        self.variables_upper_bounds = Dataset()

        self.constraints = Dataset()
        self.constraints_lhs_coeffs = Dataset()
        self.constraints_lhs_vars = Dataset()
        self.constraints_sign = Dataset()
        self.constraints_rhs = Dataset()

        self.objective = None

        self.solver_dir = solver_dir


    def add_variables(self, name, lower=-inf, upper=inf, coords=None):

        assert name not in self.variables
        # TODO: warning if name is like var names (x100).

        lower = DataArray(lower)
        upper = DataArray(upper)

        if coords is None:
            # only a lazy calculation for extracting coords, shape and size
            coords = (lower.chunk() + upper.chunk()).coords

        reslike = DataArray(coords=coords)

        start = self._xCounter
        var = np.arange(start, start + reslike.size).reshape(reslike.shape)
        self._xCounter += reslike.size
        var = xr.DataArray(var, coords=reslike.coords).chunk(self.chunk)
        var = var.assign_attrs(name=name)

        if self.chunk:
            lower = lower.chunk(self.chunk)
            upper = upper.chunk(self.chunk)
            var = var.chunk(self.chunk)

        self.variables[name] = var
        self.variables_lower_bounds[name] = lower
        self.variables_upper_bounds[name] = upper
        return var


    def add_constraints(self, name, lhs, sign, rhs):

        assert name not in self.constraints
       # TODO: warning if name is like con names (c100).
       # TODO: check if rhs is constants (floats, int)

        assert isinstance(lhs, LinearExpression)

        sign = DataArray(sign)
        rhs = DataArray(rhs)

        reslike = (lhs.variables.chunk() + rhs).sum('term_')

        start = self._cCounter
        con = np.arange(start, start + reslike.size).reshape(reslike.shape)
        self._cCounter += reslike.size
        con = DataArray(con, coords=reslike.coords)
        con = con.assign_attrs(name=name)

        if self.chunk:
            lhs = lhs.chunk(self.chunk)
            sign = sign.chunk(self.chunk)
            rhs = rhs.chunk(self.chunk)
            con = con.chunk(self.chunk)

        self.constraints[name] = con
        self.constraints_lhs_coeffs[name] = lhs.coefficients
        self.constraints_lhs_vars[name] = lhs.variables
        self.constraints_sign[name] = sign
        self.constraints_rhs[name] = rhs
        return con


    def add_objective(self, expr):
        assert isinstance(expr, LinearExpression)
        if expr.ndim > 1:
            expr = expr.sum()
        self.objective = expr
        return self.objective


    def linexpr(self, *tuples):
        # TODO: allow setting index from variables name
        tuples = [(coef, self.variables[var]) if isinstance(var, str)
                  else (coef, var) for (coef, var) in tuples]
        return LinearExpression.from_tuples(*tuples, chunk=self.chunk)


    def __repr__(self):
        return (f"<Linopy model>\n"
                f"Variables: {', '.join(self.variables)}\n"
                f"Constraints: {', '.join(self.constraints)}\n"
                f"Dimensions: {', '.join(self.variables.indexes)}")


    def to_file(self, keep_files=False):
        tmpkwargs = dict(text=True, dir=self.solver_dir)

        fdo, objective_fn = mkstemp('.txt', 'linopy-objectve-', **tmpkwargs)
        fdc, constraints_fn = mkstemp('.txt', 'linopy-constraints-', **tmpkwargs)
        fdb, bounds_fn = mkstemp('.txt', 'linopy-bounds-', **tmpkwargs)
        fdi, binaries_fn = mkstemp('.txt', 'linopy-binaries-', **tmpkwargs)
        fdp, problem_fn = mkstemp('.lp', 'linopy-problem-', **tmpkwargs)

        self.objective_f = open(objective_fn, mode='w')
        self.constraints_f = open(constraints_fn, mode='w')
        self.bounds_f = open(bounds_fn, mode='w')
        self.binaries_f = open(binaries_fn, mode='w')

        self.objective_f.write('\* LOPF *\n\nmin\nobj:\n')
        self.constraints_f.write("\n\ns.t.\n\n")
        self.bounds_f.write("\nbounds\n")
        self.binaries_f.write("\nbinary\n")

        # Bounds
        bounds_str = join_str_arrays(
            to_float_str(self.variables_lower_bounds),
            ' <= x',to_int_str(self.variables),
            ' <= ', to_float_str(self.variables_upper_bounds), '\n')
        str_array_to_file(bounds_str, self.bounds_f).compute()


        # Contraints
        lhs_str = join_str_arrays(
            to_float_str(self.constraints_lhs_coeffs),
            ' x', to_int_str(self.constraints_lhs_vars), '\n'
            ).reduce(np.sum, 'term_') # .sum() does not work

        constraints_str = join_str_arrays(
            'c', to_int_str(self.constraints), ': \n',
            lhs_str,
            self.constraints_sign, '\n',
            to_float_str(self.constraints_rhs), '\n\n')
        str_array_to_file(constraints_str, self.constraints_f).compute()


        # Objective
        objective_str = join_str_arrays(
            to_float_str(self.objective.coefficients),
            ' x', to_int_str(self.objective.variables), '\n'
            ).expand_dims('objective').sum('term_') # .sum() does not work
        str_array_to_file(objective_str, self.objective_f).compute()


        # Binaries
        # binaries_str = join_str_arrays(to_int_str(self.binaries))
        # str_array_to_file(binaries_str, self.binaries_f).compute()



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

        # logger.info(f'Total preparation time: {round(time.time()-start, 2)}s')
        return fdp, problem_fn


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
    # TODO: sometimes line are written out two times
    func = np.vectorize(lambda x: fn.write(x))
    return xr.apply_ufunc(func, array, dask='parallelized', output_dtypes=[int])



class LinearExpression:

    def __init__(self, coefficients, variables):

        assert isinstance(coefficients, DataArray)
        assert isinstance(variables, DataArray)

        assert 'term_' in coefficients.dims
        assert 'term_' in variables.dims

        coefficients, variables  = xr.broadcast(coefficients, variables)
        # TODO: perhaps use a datset here in self.data?
        self.coefficients = coefficients
        self.variables = variables


    def __add__(self, other):
        assert isinstance(other, LinearExpression)
        n_terms = len(self.variables.term_) + len(other.variables.term_)
        dim = pd.Index(range(n_terms), name='term_')
        variables = xr.concat([self.variables, other.variables], dim=dim)

        n_terms = len(self.coefficients.term_) + len(other.coefficients.term_)
        dim = pd.Index(range(n_terms), name='term_')
        coefficients = xr.concat([self.coefficients, other.coefficients], dim=dim)

        return LinearExpression(coefficients, variables)

    def __repr__(self):
        coeff_string = self.coefficients.__repr__()
        var_string = self.variables.__repr__()
        return (f"Linear Expression with {self.nterm} terms: \n\n"
                f"Coefficients:\n-------------\n {coeff_string}\n\n"
                f"Variables:\n----------\n{var_string}")


    def sum(self, dims=None):
        # TODO: add a flag with keep_coords=True?
        if dims:
            dims = list(np.atleast_1d(dims))
            if 'term_' not in dims:
                dims = ['term_'] + dims
        else:
            dims = [...]

        variables = self.variables.stack(new_ = dims)
        coefficients = self.coefficients.stack(new_ = dims)

        # reset index (there might be a more clever way)
        term_i = pd.Index(range(len(variables.new_)))
        variables = variables.assign_coords(new_ = term_i).rename(new_='term_')
        term_i = pd.Index(range(len(coefficients.new_)))
        coefficients = coefficients.assign_coords(new_ = term_i).rename(new_='term_')

        return LinearExpression(coefficients, variables)


    def from_tuples(*tuples, chunk=100):

        coeffs, varrs = zip(*tuples)
        coeffs = [c if isinstance(c, DataArray) else DataArray(c) for c in coeffs]
        dim = pd.Index(range(len(coeffs)), name='term_')
        coefficients = xr.concat(coeffs, dim=dim)


        varrs = [DataArray(v) if isinstance(v, int) else v for v in varrs]
        dim = pd.Index(range(len(coeffs)), name='term_')
        variables = xr.concat(varrs, dim=dim)

        if chunk:
            coefficients = coefficients.chunk(chunk)
            variables = variables.chunk(chunk)

        return LinearExpression(coefficients, variables)


    def chunk(self, chunk=None):
        return LinearExpression(self.coefficients.chunk(chunk),
                                self.variables.chunk(chunk))


    def compute(self):
        return LinearExpression(self.coefficients.compute(),
                                self.variables.compute())


    def load(self):
        self.coefficients.load()
        self.variables.load()
        return self


    @property
    def shape(self):
        return self.coefficients.shape

    @property
    def ndim(self):
        return self.coefficients.ndim

    @property
    def nterm(self):
        return len(self.coefficients.term_)

    @property
    def size(self):
        return self.coefficients.size

    @property
    def coords(self):
        return self.coefficients.coords

