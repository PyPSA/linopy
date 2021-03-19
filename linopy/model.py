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
        return Variable(self.variables[key])


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

        return Variable(var)


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

        broadcasted = (lhs.vars.chunk() + rhs).sum('term_')

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
        self._merge_inplace('constraints_lhs_coeffs', lhs.coeffs, name)
        self._merge_inplace('constraints_lhs_vars', lhs.vars, name)
        self._merge_inplace('constraints_sign', sign, name)
        self._merge_inplace('constraints_rhs', rhs, name)

        return con


    def add_objective(self, expr):
        if isinstance(expr, (list, tuple)):
            expr = self.linexpr(*expr)
        assert isinstance(expr, LinearExpression)
        if expr.vars.ndim > 1:
            expr = expr.sum()
        self.objective = expr
        return self.objective


    def linexpr(self, *tuples):
        tuples = [(c, self.variables[v]) if isinstance(v, str)
                  else (c, v) for (c, v) in tuples]
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


class Variable():

    def __init__(self, data):
        self.data = data

    def __repr__(self):
        data_string = self.data.__repr__()
        return (f"Variable container with variables:\n"
                f"----------------------------------\n\n{data_string}")

    def __mul__(self, coefficient):
        return LinearExpression.from_tuples((coefficient, self.data))


    def __rmul__(self, coefficient):
        return LinearExpression.from_tuples((coefficient, self.data))

    def __add__(self, other):
        if isinstance(other, Variable):
            return LinearExpression.from_tuples((1, self.data), (1, other.data))
        elif isinstance(other, LinearExpression):
            return LinearExpression.from_tuples((1, self.data)) + other
        else:
            raise TypeError("unsupported operand type(s) for +: "
                            f"{type(self)} and {type(other)}")



class LinearExpression(Dataset):
    __slots__ = ('_cache', '_coords', '_indexes', '_name', '_variable')

    def __init__(self, dataset):
        assert set(dataset) == {'coeffs', 'vars'}
        (dataset,) = xr.broadcast(dataset)
        super().__init__(dataset)


    def __add__(self, other):
        if isinstance(other, Variable):
            other = LinearExpression.from_tuples((1, other.data))
        if not isinstance(other, LinearExpression):
            raise TypeError("unsupported operand type(s) for +: "
                            f"{type(self)} and {type(other)}")
        return LinearExpression(xr.concat([self, other], dim='term_'))



    def __repr__(self):
        coeff_string = self.coeffs.__repr__()
        var_string = self.vars.__repr__()
        return (f"Linear Expression with {self.nterm} term(s): \n\n"
                f"Coefficients:\n-------------\n {coeff_string}\n\n"
                f"Variables:\n----------\n{var_string}")


    def to_dataset(self):
        return Dataset(self)


    def sum(self, dims=None, keep_coords=False):

        if dims:
            dims = list(np.atleast_1d(dims))
        else:
            dims = [...]
        if 'term_' in dims:
            dims.remove('term_')

        stacked_term_dim = 'term_dim_'
        num = 0
        while stacked_term_dim + str(num) in self.indexes['term_'].names:
            num += 1
        stacked_term_dim += str(num)
        dims.append(stacked_term_dim)

        ds = self.rename(term_ = stacked_term_dim).stack(term_ = dims)
        if not keep_coords:
            ds = ds.assign_coords(term_ = pd.RangeIndex(len(ds.term_)))
        return LinearExpression(ds)


    def from_tuples(*tuples, chunk=None):

        idx = []
        for i, (c, v) in enumerate(tuples):
            if isinstance(v, (Variable, DataArray)):
                idx.append(v.attrs['name'])
            else:
                idx.append(i)

        ds_list = [Dataset({'coeffs': c, 'vars': v}) for c, v in tuples]
        if len(ds_list) > 1:
            ds = xr.concat(ds_list, dim=pd.Index(idx, name='term_'))
        else:
            ds = ds_list[0].expand_dims(term_=idx)
        return LinearExpression(ds)


    @property
    def nterm(self):
        return len(self.term_)

    @property
    def shape(self):
        assert self.vars.shape == self.coeffs.shape
        return self.vars.shape

    @property
    def size(self):
        assert self.vars.size == self.coeffs.size
        return self.vars.size

