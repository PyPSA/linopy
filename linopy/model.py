# -*- coding: utf-8 -*-
"""
Linopy model module.
This module contains frontend implementations of the package.
"""

import logging
import os
from tempfile import gettempdir, mkstemp

import numpy as np
import pandas as pd
import xarray as xr
from numpy import inf
from xarray import DataArray, Dataset, merge

from .io import to_file, to_netcdf
from .solvers import (
    available_solvers,
    run_cbc,
    run_cplex,
    run_glpk,
    run_gurobi,
    run_xpress,
)

logger = logging.getLogger(__name__)


class Model:
    """
    Linear optimization model.

    The Model contains all relevant data of a linear program, including

    * variables with lower and upper bounds
    * constraints with left hand side (lhs) being a linear expression of variables
      and a right hand side (rhs) being a constant. Lhs and rhs
      are set in relation by the sign
    * objective being a linear expression of variables

    The model supports different solvers (see `linopy.available_solvers`) for
    the optimization process.
    """

    def __init__(self, solver_dir=None, chunk=None):
        """
        Initialize the linopy model.

        Parameters
        ----------
        solver_dir : pathlike, optional
            Path where temporary files like the lp file or solution file should
            be stored. The default None results in taking the default temporary
            directory.
        chunk : int, optional
            Chunksize used when assigning data, this can speed up large
            programs while keeping memory-usage low. The default is None.

        Returns
        -------
        linopy.Model

        """
        self._xCounter = 0
        self._cCounter = 0
        self._varnameCounter = 0
        self._connameCounter = 0

        self.chunk = chunk
        self.status = "initialized"
        self.objective_value = None

        self.variables = Dataset()
        self.variables_lower_bound = Dataset()
        self.variables_upper_bound = Dataset()

        self.binaries = Dataset()

        self.constraints = Dataset()
        self.constraints_lhs_coeffs = Dataset()
        self.constraints_lhs_vars = Dataset()
        self.constraints_sign = Dataset()
        self.constraints_rhs = Dataset()

        self.objective = LinearExpression()

        self.solution = Dataset()
        self.dual = Dataset()

        if solver_dir is None:
            self.solver_dir = gettempdir()

    def __repr__(self):
        """Return a string representation of the linopy model."""
        var_string = self.variables.__repr__().split("\n", 1)[1]
        var_string = var_string.replace("Data variables:\n", "Data:\n")
        con_string = self.constraints.__repr__().split("\n", 1)[1]
        con_string = con_string.replace("Data variables:\n", "Data:\n")
        return (
            f"Linopy model\n============\n\n"
            f"Variables:\n----------\n{var_string}\n\n"
            f"Constraints:\n------------\n{con_string}\n\n"
            f"Status:\n-------\n{self.status}"
        )

    def __getitem__(self, key):
        """Get a model variable by the name."""
        return Variable(self.variables[key])

    def _merge_inplace(self, attr, da, name):
        """
        Assign a new variable to the dataset `attr` by merging.

        This takes care of all coordinate alignments, instead of a direct
        assignment like self.variables[name] = var
        """
        ds = merge([getattr(self, attr), da.to_dataset(name=name)])
        setattr(self, attr, ds)

    def add_variables(self, lower=-inf, upper=inf, coords=None, name=None):
        """
        Assign a new, possibly multi-dimensional array of variables to the model.

        Variables may be added with lower and/or upper bounds. Unless a
        `coords` argument is provided, the shape of the lower and upper bounds
        define the number of variables which will be added to the model
        under the name `name`.

        Parameters
        ----------
        lower : float/array_like, optional
            Lower bound of the variable(s). The default is -inf.
        upper : TYPE, optional
            Upper bound of the variable(s). The default is inf.
        coords : list/xarray.Coordinates, optional
            The coords of the variable array. For every single combination of
            coordinates a optimization variable is added to the model.
            The default is None. Is ignored when lower and upper bound provide
            coordinates.
        name : str, optional
            Reference name of the added variables. The default None results in
            a name like "var1", "var2" etc.

        Raises
        ------
        ValueError
            If neither lower bound and upper bound have coordinates, nor
            `coords` are directly given.

        Returns
        -------
        linopy.Variable
            Variable which was added to the model.


        Examples
        --------
        >>> m = linopy.Model()
        >>> time = pd.RangeIndex(10, name="Time")
        >>> m.add_variables(lower=0, coords=[time], name="x")

        ::

            Variable container:
            -------------------

            Variables:
            array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
            Coordinates:
                * Time     (Time) int64 0 1 2 3 4 5 6 7 8 9
            Attributes:
                name:     x

        """
        if name is None:
            while "var" + str(self._varnameCounter) in self.variables:
                self._varnameCounter += 1
            name = "var" + str(self._varnameCounter)

        assert name not in self.variables

        lower = DataArray(lower)
        upper = DataArray(upper)

        if coords is None:
            # only a lazy calculation for extracting coords, shape and size
            broadcasted = lower.chunk() + upper.chunk()
            coords = broadcasted.coords
            if not coords and broadcasted.size > 1:
                raise ValueError(
                    "Both `lower` and `upper` have missing coordinates"
                    " while the broadcasted array is of size > 1."
                )

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

        self._merge_inplace("variables", var, name)
        self._merge_inplace("variables_lower_bound", lower, name)
        self._merge_inplace("variables_upper_bound", upper, name)

        return Variable(var)

    def add_constraints(self, lhs, sign, rhs, name=None):
        """
        Assign a new, possibly multi-dimensional array of constraints to the model.

        Constraints are added by defining a left hand side (lhs), the sign and
        the right hand side (rhs). The lhs has to be a linopy.LinearExpression
        and the rhs a constant (array of constants). The function return the
        an array with the constraint labels (integers).

        Parameters
        ----------
        lhs : linopy.LinearExpression
            Left hand side of the constraint(s).
        sign : str/array_like
            Relation between the lhs and rhs, valid values are {'=', '>=', '<='}.
        rhs : int/float/array_like
            Right hand side of the constraint(s).
        name : str, optional
            Reference name of the added constraints. The default None results
            results a name like "con1", "con2" etc.


        Returns
        -------
        con : xarray.DataArray
            Array containing the labels of the added constraints.

        """
        if name is None:
            while "con" + str(self._connameCounter) in self.constraints:
                self._connameCounter += 1
            name = "con" + str(self._connameCounter)

        assert name not in self.constraints

        if isinstance(lhs, (list, tuple)):
            lhs = self.linexpr(*lhs)
        assert isinstance(lhs, LinearExpression)

        sign = DataArray(sign)
        rhs = DataArray(rhs)

        if (sign == "==").any():
            raise ValueError('Sign "==" not supported, use "=" instead.')

        broadcasted = (lhs.vars.chunk() + rhs).sum("_term")

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
        self._merge_inplace("constraints", con, name)
        self._merge_inplace("constraints_lhs_coeffs", lhs.coeffs, name)
        self._merge_inplace("constraints_lhs_vars", lhs.vars, name)
        self._merge_inplace("constraints_sign", sign, name)
        self._merge_inplace("constraints_rhs", rhs, name)

        return con

    def add_objective(self, expr, overwrite=False):
        """
        Add a linear objective function to the model.

        Parameters
        ----------
        expr : linopy.LinearExpression
            Linear Expressions describing the objective function.
        overwrite : False, optional
            Whether to overwrite the existing objective. The default is False.

        Returns
        -------
        linopy.LinearExpression
            The objective function assigned to the model.

        """
        if not overwrite:
            assert self.objective.empty(), (
                "Objective already defined."
                " Set `overwrite` to True to force overwriting."
            )

        if isinstance(expr, (list, tuple)):
            expr = self.linexpr(*expr)
        assert isinstance(expr, LinearExpression)
        if expr.vars.ndim > 1:
            expr = expr.sum()
        self.objective = expr
        return self.objective

    def remove_variables(self, name):
        """
        Remove all variables stored under reference name `name` from the model.

        This function also removes the variables from constraints and the
        objective function. Note that this will leave blank spaces where the
        variables were stored which ensures dimensional compatibility.

        Parameters
        ----------
        name : str
            Reference name of the variables which to remove, same as used in
            `model.add_variables`.

        Returns
        -------
        None.

        """
        labels = self.variables[name]
        self.variables = self.variables.drop_vars(name)
        self.variables_lower_bound = self.variables_lower_bound.drop_vars(name)
        self.variables_upper_bound = self.variables_upper_bound.drop_vars(name)

        keep_b = ~self.constraints_lhs_vars.isin(labels)
        self.constraints_lhs_coeffs = self.constraints_lhs_coeffs.where(keep_b)
        self.constraints_lhs_vars = self.constraints_lhs_vars.where(keep_b)

        keep_b_con = keep_b.any(dim="_term")
        self.constraints = self.constraints.where(keep_b_con)
        self.constraints_sign = self.constraints_sign.where(keep_b_con)
        self.constraints_rhs = self.constraints_rhs.where(keep_b_con)

        self.objective = self.objective.where(~self.objective.isin(labels))

    def remove_constraints(self, name):
        """
        Remove all constraints stored under reference name `name` from the model.

        Parameters
        ----------
        name : str
            Reference name of the constraints which to remove, same as used in
            `model.add_constraints`.

        Returns
        -------
        None.

        """
        self.constraints = self.constraints.drop_vars(name)
        self.constraints_lhs_coeffs = self.constraints_lhs_coeffs.drop_vars(name)
        self.constraints_lhs_vars = self.constraints_lhs_vars.drop_vars(name)
        self.constraints_sign = self.constraints_sign.drop_vars(name)
        self.constraints_rhs = self.constraints_rhs.drop_vars(name)

    def linexpr(self, *tuples):
        """
        Create a linopy.LinearExpression by using variable names.

        Calls the function LinearExpression.from_tuples but loads variables from
        the model if a variable name is used.

        Parameters
        ----------
        tuples : tuples of (coefficients, variables)
            Each tuple represents on term in the linear expression, which can
            span over multiple dimensions:

            * coefficients : int/float/array_like
                The coefficient(s) in the term, if the coefficients array
                contains dimensions which do not appear in
                the variables, the variables are broadcasted.
            * variables : str/array_like/linopy.Variable
                The variable(s) going into the term. These may be referenced
                by name.

        Returns
        -------
        linopy.LinearExpression

        Examples
        --------
        >>> m = Model()
        >>> m.add_variables(pd.Series([0, 0]), 1, name="x")
        >>> m.add_variables(4, pd.Series([8, 10]), name="y")
        >>> expr = m.linexpr((10, "x"), (1, "y"))

        """
        tuples = [
            (c, self.variables[v]) if isinstance(v, str) else (c, v)
            for (c, v) in tuples
        ]
        return LinearExpression.from_tuples(*tuples, chunk=self.chunk)

    @property
    def coefficientrange(self):
        """Coefficient range of the constraints in the model."""
        return (
            xr.concat(
                [self.constraints_lhs_coeffs.min(), self.constraints_lhs_coeffs.max()],
                dim=pd.Index(["min", "max"]),
            )
            .to_dataframe()
            .T
        )

    @property
    def objectiverange(self):
        """Objective range of the objective in the model."""
        return pd.Series(
            [
                self.objective.coefficients.min().item(),
                self.objective.coefficients.max().item(),
            ],
            index=["min", "max"],
        )

    def solve(
        self,
        solver_name="gurobi",
        problem_fn=None,
        solution_fn=None,
        log_fn=None,
        basis_fn=None,
        warmstart_fn=None,
        keep_files=False,
        **solver_options,
    ):
        """
        Solve the model with possibly different solvers.

        The optimal values of the variables are stored in `model.solution`.
        The optimal dual variables are stored in `model.dual`.

        Parameters
        ----------
        solver_name : str, optional
            Name of the solver to use, this must be in `linopy.available_solvers`.
            The default is 'gurobi'.
        problem_fn : path_like, optional
            Path of the lp file which is written out during the process. The
            default None results in a temporary file.
        solution_fn : path_like, optional
            Path of the solution file which is written out during the process.
            The default None results in a temporary file.
        log_fn : path_like, optional
            Path of the logging file which is written out during the process.
            The default None results in the no log file, hence all solver
            outputs are piped to the python repl.
        basis_fn : path_like, optional
            Path of the basis file of the solution which is written after
            solving. The default None results in a temporary file, if the solver/method
            supports writing out a basis file.
        warmstart_fn : path_like, optional
            Path of the basis file which should be used to warmstart the
            solving. The default is None.
        keep_files : bool, optional
            Whether to keep all temporary files like lp file, solution file.
            This argument is ignored for the logger file `log_fn`. The default
            is False.
        **solver_options : kwargs
            Options passed to the solver.

        Returns
        -------
        linopy.Model
            Optimized model.

        """
        logger.info(f" Solve linear problem using {solver_name.title()} solver")
        assert solver_name in available_solvers, f"Solver {solver_name} not installed"

        tmp_kwargs = dict(text=True, dir=self.solver_dir)
        if problem_fn is None:
            fds, problem_fn = mkstemp(".lp", "linopy-problem-", **tmp_kwargs)
        if solution_fn is None:
            fds, solution_fn = mkstemp(".sol", "linopy-solve-", **tmp_kwargs)
        if log_fn is not None:
            logger.info(f"Solver logs written to `{log_fn}`.")

        try:
            self.to_file(problem_fn)
            solve = eval(f"run_{solver_name}")
            res = solve(
                problem_fn,
                log_fn,
                solution_fn,
                warmstart_fn,
                basis_fn,
                **solver_options,
            )

        finally:
            if not keep_files:
                if os.path.exists(problem_fn):
                    os.remove(problem_fn)
                if os.path.exists(solution_fn):
                    os.remove(solution_fn)

        status = res.pop("status")
        termination_condition = res.pop("termination_condition")
        obj = res.pop("objective", None)

        if status == "ok" and termination_condition == "optimal":
            logger.info(f" Optimization successful. Objective value: {obj:.2e}")
        elif status == "warning" and termination_condition == "suboptimal":
            logger.warning(
                " Optimization solution is sub-optimal. " "Objective value: {obj:.2e}"
            )
        else:
            logger.warning(
                f" Optimization failed with status {status} and "
                f"termination condition {termination_condition}"
            )
            return status, termination_condition

        self.objective_value = obj
        self.solver_model = res.pop("model", None)
        self.status = termination_condition

        res["solution"].loc[np.nan] = np.nan
        for v in self.variables:
            idx = self.variables[v].data.ravel()
            sol = res["solution"][idx].values.reshape(self.variables[v].shape)
            self.solution[v] = xr.DataArray(sol, self.variables[v].coords)

        res["dual"].loc[np.nan] = np.nan
        for c in self.constraints:
            idx = self.constraints[c].data.ravel()
            du = res["dual"][idx].values.reshape(self.constraints[c].shape)
            self.dual[c] = xr.DataArray(du, self.constraints[c].coords)

        return self

    to_netcdf = to_netcdf

    to_file = to_file


class Variable(DataArray):
    """
    Variable Container for storing variable labels.

    The Variable class is a subclass of xr.DataArray hence most xarray functions
    can be applied to it. However most arithmetic operations are overwritten.
    Like this one can easily combine variables into a linear expression.

    Examples
    --------
    >>> m = Model()
    >>> x = m.add_variables(pd.Series([0, 0]), 1, name="x")
    >>> y = m.add_variables(4, pd.Series([8, 10]), name="y")

    Add variable together:

    >>> x + y

    ::

        Linear Expression with 2 term(s):
        ----------------------------------

        Dimensions:  (dim_0: 2, _term: 2)
        Coordinates:
            * dim_0    (dim_0) int64 0 1
            * _term    (_term) int64 0 1
        Data:
            coeffs   (dim_0, _term) int64 1 1 1 1
            vars     (dim_0, _term) int64 1 3 2 4


    Multiply them with a coefficient:

    >>> 3 * x

    ::

        Linear Expression with 1 term(s):
        ----------------------------------

        Dimensions:  (dim_0: 2, _term: 1)
        Coordinates:
            * _term    (_term) int64 0
            * dim_0    (dim_0) int64 0 1
        Data:
            coeffs   (dim_0, _term) int64 3 3
            vars     (dim_0, _term) int64 1 2


    Further operations like taking the negative and subtracting are supported.

    """

    __slots__ = ("_cache", "_coords", "_indexes", "_name", "_variable")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_array(self):
        """Convert the variable array to a xarray.DataArray."""
        return DataArray(self)

    def to_linexpr(self, coefficient=1):
        """Create a linear exprssion from the variables."""
        return LinearExpression.from_tuples((coefficient, self))

    def __repr__(self):
        """Get the string representation of the variables."""
        data_string = "Variables:\n" + self.to_array().__repr__().split("\n", 1)[1]
        return f"Variable container:\n" f"-------------------\n\n{data_string}"

    def _repr_html_(self):
        """Get the html representation of the variables."""
        # return self.__repr__()
        data_string = self.to_array()._repr_html_()
        data_string = data_string.replace("xarray.DataArray", "linopy.Variable")
        return data_string

    def __neg__(self):
        """Calculate the negative of the variables (converts coefficients only)."""
        return self.to_linexpr(-1)

    def __mul__(self, coefficient):
        """Multiply variables with a coefficient."""
        return self.to_linexpr(coefficient)

    def __rmul__(self, coefficient):
        """Right-multiply variables with a coefficient."""
        return self.to_linexpr(coefficient)

    def __add__(self, other):
        """Add variables to linear expressions or other variables."""
        if isinstance(other, Variable):
            return LinearExpression.from_tuples((1, self), (1, other))
        elif isinstance(other, LinearExpression):
            return self.to_linexpr() + other
        else:
            raise TypeError(
                "unsupported operand type(s) for +: " f"{type(self)} and {type(other)}"
            )

    def __sub__(self, other):
        """Subtract linear expressions or other variables from the variables."""
        if isinstance(other, Variable):
            return LinearExpression.from_tuples((1, self), (-1, other))
        elif isinstance(other, LinearExpression):
            return self.to_linexpr() - other
        else:
            raise TypeError(
                "unsupported operand type(s) for -: " f"{type(self)} and {type(other)}"
            )

    def sum(self, dims=None, keep_coords=False):
        """
        Sum the variables over all or a subset of dimensions.

        This stack all terms of the dimensions, that are summed over, together.
        The function works exactly in the same way as ``LinearExpression.sum()``.

        Parameters
        ----------
        dims : str/list, optional
            Dimension(s) to sum over. The default is None which results in all
            dimensions.
        keep_coords : bool, optional
            Whether to keep the coordinates of the stacked dimensions in a
            MultiIndex. The default is False.

        Returns
        -------
        linopy.LinearExpression
            Summed expression.

        """
        return self.to_linexpr().sum(dims, keep_coords)

    def group_terms(self, group):
        """
        Sum variables over groups.

        The function works in the same manner as the xarray.Dataset.groupby
        function, but automatically sums over all terms.

        Parameters
        ----------
        group : DataArray or IndexVariable
            Array whose unique values should be used to group the expressions.

        Returns
        -------
        Grouped linear expression.

        """
        return self.to_linexpr().group_terms(group)


class LinearExpression(Dataset):
    """
    A linear expression consisting of terms of coefficients and variables.

    The LinearExpression class is a subclass of xarray.Dataset which allows to
    apply most xarray functions on it. However most arithmetic operations are
    overwritten. Like this you can easily expand and modify the linear
    expression.

    Examples
    --------
    >>> m = Model()
    >>> x = m.add_variables(pd.Series([0, 0]), 1, name="x")
    >>> y = m.add_variables(4, pd.Series([8, 10]), name="y")

    Combining expressions:

    >>> expr = 3 * x
    >>> type(expr)
    linopy.model.LinearExpression

    >>> other = 4 * y
    >>> type(expr + other)
    linopy.model.LinearExpression

    Multiplying:

    >>> type(3 * expr)
    linopy.model.LinearExpression

    Summation over dimensions

    >>> expr.sum(dim="dim_0")

    ::

        Linear Expression with 2 term(s):
        ----------------------------------

        Dimensions:  (_term: 2)
        Coordinates:
            * _term    (_term) int64 0 1
        Data:
            coeffs   (_term) int64 3 3
            vars     (_term) int64 1 2

    """

    __slots__ = ("_cache", "_coords", "_indexes", "_name", "_variable")

    def __init__(self, dataset=None):
        if dataset is not None:
            assert set(dataset) == {"coeffs", "vars"}
            (dataset,) = xr.broadcast(dataset)
        super().__init__(dataset)

    # We have to set the _reduce_method to None, in order to overwrite basic
    # reduction functions as `sum`. There might be a better solution (?).
    _reduce_method = None

    def __repr__(self):
        """Get the string representation of the expression."""
        ds_string = self.to_dataset().__repr__().split("\n", 1)[1]
        ds_string = ds_string.replace("Data variables:\n", "Data:\n")
        nterm = getattr(self, "nterm", 0)
        return (
            f"Linear Expression with {nterm} term(s):\n"
            f"----------------------------------\n\n{ds_string}"
        )

    def _repr_html_(self):
        """Get the html representation of the expression."""
        # return self.__repr__()
        ds_string = self.to_dataset()._repr_html_()
        ds_string = ds_string.replace("Data variables:\n", "Data:\n")
        ds_string = ds_string.replace("xarray.Dataset", "linopy.LinearExpression")
        return ds_string

    def __add__(self, other):
        """Add a expression to others."""
        if isinstance(other, Variable):
            other = LinearExpression.from_tuples((1, other))
        if not isinstance(other, LinearExpression):
            raise TypeError(
                "unsupported operand type(s) for +: " f"{type(self)} and {type(other)}"
            )
        res = LinearExpression(xr.concat([self, other], dim="_term"))
        if res.indexes["_term"].duplicated().any():
            return res.assign_coords(_term=pd.RangeIndex(len(res._term)))
        return res

    def __sub__(self, other):
        """Subtract others form expression."""
        if isinstance(other, Variable):
            other = LinearExpression.from_tuples((-1, other))
        elif isinstance(other, LinearExpression):
            other = -other
        else:
            raise TypeError(
                "unsupported operand type(s) for -: " f"{type(self)} and {type(other)}"
            )
        res = LinearExpression(xr.concat([self, other], dim="_term"))
        if res.indexes["_term"].duplicated().any():
            return res.assign_coords(_term=pd.RangeIndex(len(res._term)))
        return res

    def __neg__(self):
        """Get the negative of the expression."""
        return LinearExpression(self.assign(coeffs=-self.coeffs))

    def __mul__(self, other):
        """Multiply the expr by a factor."""
        coeffs = other * self.coeffs
        assert coeffs.shape == self.coeffs.shape
        return LinearExpression(self.assign(coeffs=coeffs))

    def __rmul__(self, other):
        """Right-multiply the expr by a factor."""
        return self.__mul__(other)

    def to_dataset(self):
        """Convert the expression to a xarray.Dataset."""
        return Dataset(self)

    def sum(self, dims=None, keep_coords=False):
        """
        Sum the expression over all or a subset of dimensions.

        This stack all terms of the dimensions, that are summed over, together.

        Parameters
        ----------
        dims : str/list, optional
            Dimension(s) to sum over. The default is None which results in all
            dimensions.
        keep_coords : bool, optional
            Whether to keep the coordinates of the stacked dimensions in a
            MultiIndex. The default is False.

        Returns
        -------
        linopy.LinearExpression
            Summed expression.

        """
        if dims:
            dims = list(np.atleast_1d(dims))
        else:
            dims = [...]
        if "_term" in dims:
            dims.remove("_term")

        stacked_term_dim = "term_dim_"
        num = 0
        while stacked_term_dim + str(num) in self.indexes["_term"].names:
            num += 1
        stacked_term_dim += str(num)
        dims.append(stacked_term_dim)

        ds = self.rename(_term=stacked_term_dim).stack(_term=dims)
        if not keep_coords:
            ds = ds.assign_coords(_term=pd.RangeIndex(len(ds._term)))
        return LinearExpression(ds)

    def from_tuples(*tuples, chunk=None):
        """
        Create a linear expression by using tuples of coefficients and variables.

        Parameters
        ----------
        tuples : tuples of (coefficients, variables)
            Each tuple represents on term in the resulting linear expression,
            which can possibly span over multiple dimensions:

            * coefficients : int/float/array_like
                The coefficient(s) in the term, if the coefficients array
                contains dimensions which do not appear in
                the variables, the variables are broadcasted.
            * variables : str/array_like/linopy.Variable
                The variable(s) going into the term. These may be referenced
                by name.

        Returns
        -------
        linopy.LinearExpression

        Examples
        --------
        >>> m = Model()
        >>> x = m.add_variables(pd.Series([0, 0]), 1)
        >>> m.add_variables(4, pd.Series([8, 10]))
        >>> expr = LinearExpression.from_tuples((10, x), (1, y))

        This is the same as calling ``10*x + y`` but a bit more performant.
        """
        idx = pd.RangeIndex(len(tuples))
        ds_list = [Dataset({"coeffs": c, "vars": v}) for c, v in tuples]
        if len(ds_list) > 1:
            ds = xr.concat(ds_list, dim=pd.Index(idx, name="_term"))
        else:
            ds = ds_list[0].expand_dims(_term=idx)
        return LinearExpression(ds)

    def group_terms(self, group):
        """
        Sum expression over groups.

        The function works in the same manner as the xarray.Dataset.groupby
        function, but automatically sums over all terms.

        Parameters
        ----------
        group : DataArray or IndexVariable
            Array whose unique values should be used to group the expressions.

        Returns
        -------
        Grouped linear expression.

        """
        groups = self.groupby(group)
        return groups.map(lambda ds: ds.sum(groups._group_dim))

    @property
    def nterm(self):
        """Get the number of terms in the linear expression."""
        return len(self._term)

    @property
    def shape(self):
        """Get the total shape of the linear expression."""
        assert self.vars.shape == self.coeffs.shape
        return self.vars.shape

    @property
    def size(self):
        """Get the total size of the linear expression."""
        assert self.vars.size == self.coeffs.size
        return self.vars.size

    def empty(self):
        """Get whether the linear expression is empty."""
        return not bool(self)
