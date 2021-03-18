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
import logging
import re

from tempfile import mkstemp, gettempdir
from xarray import DataArray, Dataset, merge
from numpy import inf
from functools import reduce


from .io import to_file
from .solvers import (run_cbc, run_gurobi, run_glpk, run_cplex, run_xpress,
                      available_solvers)

logger = logging.getLogger(__name__)

class Model:


    def __init__(self, solver_dir=None, chunk=None):
        self._xCounter = 1
        self._cCounter = 1
        self.chunk = chunk
        self.status = 'initialized'

        self.variables = Dataset()
        self.variables_lower_bounds = Dataset()
        self.variables_upper_bounds = Dataset()

        self.binaries = Dataset()

        self.constraints = Dataset()
        self.constraints_lhs_coeffs = Dataset()
        self.constraints_lhs_vars = Dataset()
        self.constraints_sign = Dataset()
        self.constraints_rhs = Dataset()

        self.objective = None

        self.solution = Dataset()
        self.dual = Dataset()

        if solver_dir is None:
            self.solver_dir = gettempdir()

    def __repr__(self):
        return (f"<Linopy model>\n"
                f"Variables: {', '.join(self.variables)}\n"
                f"Constraints: {', '.join(self.constraints)}\n"
                f"Dimensions: {', '.join(self.variables.indexes)}\n"
                f"Status: {self.status}")

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.variables[key]
        if isinstance(key, tuple):
            if isinstance(key[0], tuple):
                return self.linexpr(*key)
            return self.linexpr(key)


    def _merge_inplace(self, attr, da, name):
        """
        Assign a new variable to the dataset `attr` by merging.

        This takes care of all coordinate alignments, instead of a direct
        assignment like self.variables[name] = var
        """
        ds = merge([getattr(self, attr), da.to_dataset(name=name)])
        setattr(self, attr, ds)

    # TODO should be named add_variable?
    def add_variables(self, name, lower=-inf, upper=inf, coords=None):

        assert name not in self.variables

        lower = DataArray(lower)
        upper = DataArray(upper)

        if coords is None:
            # only a lazy calculation for extracting coords, shape and size
            broadcasted = (lower.chunk() + upper.chunk())
            coords = broadcasted.coords
            if not coords and broadcasted.size > 1:
                raise ValueError('Both `lower` and `upper` have missing coordinates'
                                 ' while the broadcasted array is of size > 1.')

        broadcasted = DataArray(coords=coords)

        start = self._xCounter
        var = np.arange(start, start + broadcasted.size).reshape(broadcasted.shape)
        self._xCounter += broadcasted.size
        var = xr.DataArray(var, coords=broadcasted.coords)
        var = var.assign_attrs(name=name)

        if self.chunk:
            lower = lower.chunk(self.chunk)
            upper = upper.chunk(self.chunk)
            var = var.chunk(self.chunk)


        self._merge_inplace('variables', var, name)
        self._merge_inplace('variables_lower_bounds', lower, name)
        self._merge_inplace('variables_upper_bounds', upper, name)

        return var


    # TODO should be named add_constraint
    def add_constraints(self, name, lhs, sign, rhs):

        assert name not in self.constraints

        if isinstance(lhs, (list, tuple)):
            lhs = self.linexpr(*lhs)
        assert isinstance(lhs, LinearExpression)

        sign = DataArray(sign)
        rhs = DataArray(rhs)

        if (sign == '==').any():
            raise ValueError('Sign "==" not supported, use "=" instead.')

        broadcasted = (lhs.variables.chunk() + rhs).sum('term_')

        start = self._cCounter
        con = np.arange(start, start + broadcasted.size).reshape(broadcasted.shape)
        self._cCounter += broadcasted.size
        con = DataArray(con, coords=broadcasted.coords)
        con = con.assign_attrs(name=name)

        if self.chunk:
            lhs = lhs.chunk(self.chunk)
            sign = sign.chunk(self.chunk)
            rhs = rhs.chunk(self.chunk)
            con = con.chunk(self.chunk)

        # assign everything
        self._merge_inplace('constraints', con, name)
        self._merge_inplace('constraints_lhs_coeffs', lhs.coefficients, name)
        self._merge_inplace('constraints_lhs_vars', lhs.variables, name)
        self._merge_inplace('constraints_sign', sign, name)
        self._merge_inplace('constraints_rhs', rhs, name)

        return con


    def add_objective(self, expr):
        if isinstance(expr, (list, tuple)):
            expr = self.linexpr(*expr)
        assert isinstance(expr, LinearExpression)
        if expr.ndim > 1:
            expr = expr.sum()
        self.objective = expr
        return self.objective


    def linexpr(self, *tuples):
        tuples = [(coef, self.variables[var]) if isinstance(var, str)
                  else (coef, var) for (coef, var) in tuples]
        return LinearExpression.from_tuples(*tuples, chunk=self.chunk)


    @property
    def coefficientrange(self):
        return xr.concat([self.constraints_lhs_coeffs.min(),
                          self.constraints_lhs_coeffs.max()],
                         dim = pd.Index(['min', 'max'])).to_dataframe().T

    @property
    def objectiverange(self):
        return pd.Series([self.objective.coefficients.min().item(),
                          self.objective.coefficients.max().item()],
                         index = ['min', 'max'])


    def solve(self, solver_name='gurobi', problem_fn=None, solution_fn=None,
              log_fn=None, basis_fn=None, warmstart_fn=None, keep_files=False,
              **solver_options):

        logger.info(f" Solve linear problem using {solver_name.title()} solver")
        assert solver_name in available_solvers, (
            f'Solver {solver_name} not installed')

        tmp_kwargs = dict(text=True, dir=self.solver_dir)
        if problem_fn is None:
            fds, problem_fn = mkstemp('.lp', 'linopy-problem-', **tmp_kwargs)
        if solution_fn is None:
            fds, solution_fn = mkstemp('.sol', 'linopy-solve-', **tmp_kwargs)
        if log_fn is not None:
            logger.info(f'Solver logs written to `{log_fn}`.')

        try:
            self.to_file(problem_fn)
            solve = eval(f'run_{solver_name}')
            res = solve(problem_fn, log_fn, solution_fn, warmstart_fn,
                        basis_fn, **solver_options)

        finally:
            if not keep_files:
                if os.path.exists(problem_fn):
                    os.remove(problem_fn)
                if os.path.exists(solution_fn):
                    os.remove(solution_fn)

        status = res.pop('status')
        termination_condition = res.pop('termination_condition')
        obj = res.pop('objective', None)

        if status == "ok" and termination_condition == "optimal":
            logger.info(f' Optimization successful. Objective value: {obj:.2e}')
        elif status == "warning" and termination_condition == "suboptimal":
            logger.warning(' Optimization solution is sub-optimal. '
                           'Objective value: {obj:.2e}')
        else:
            logger.warning(f' Optimization failed with status {status} and '
                           f'termination condition {termination_condition}')
            return status, termination_condition

        self.objective_value = obj
        self.solver_model = res.pop('model', None)
        self.status = termination_condition

        for v in self.variables:
            idx = self.variables[v].data.ravel()
            sol = res['solution'][idx].values.reshape(self.variables[v].shape)
            self.solution[v] = xr.DataArray(sol, self.variables[v].coords)

        for c in self.constraints:
            idx = self.constraints[c].data.ravel()
            du = res['dual'][idx].values.reshape(self.constraints[c].shape)
            self.dual[c] = xr.DataArray(du, self.constraints[c].coords)

        return self

    # TODO
    # to_netcdf = to_netcdf

    to_file = to_file




class LinearExpression:

    def __init__(self, coefficients, variables):

        assert isinstance(coefficients, DataArray)
        assert isinstance(variables, DataArray)

        assert 'term_' in coefficients.dims
        assert 'term_' in variables.dims

        assert (coefficients.notnull() == variables.notnull()).all()

        coefficients, variables  = xr.broadcast(coefficients, variables)
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

        varrs = [DataArray(v) if isinstance(v, int) else v for v in varrs]
        maybe_names = [v.attrs.get('name', i) for i,v in enumerate(varrs)]
        dim = pd.Index(maybe_names, name='term_')
        variables = xr.concat(varrs, dim=dim, combine_attrs='drop')

        coeffs = [c if isinstance(c, DataArray) else DataArray(c) for c in coeffs]
        coefficients = xr.concat(coeffs, dim=dim, combine_attrs='drop')

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

