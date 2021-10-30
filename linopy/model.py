# -*- coding: utf-8 -*-
"""
Linopy model module.
This module contains frontend implementations of the package.
"""

import logging
import os
import re
from dataclasses import dataclass
from tempfile import NamedTemporaryFile, gettempdir
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr
from numpy import inf
from xarray import DataArray, Dataset, merge

from . import solvers
from .eval import Expr
from .io import to_file, to_netcdf
from .solvers import available_solvers

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

    array_attrs = ["parameters", "solution", "dual"]
    obj_attrs = ["objective_value", "status", "_xCounter", "_cCounter"]

    def __init__(self, solver_dir=None, chunk=None, force_dim_names=False):
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
        force_dim_names : bool
            Whether assigned variables, constraints and data should always have
            custom dimension names, i.e. not matching dimension names "dim_0",
            "dim_1" and so on. These helps to avoid unintended broadcasting
            over dimension. Especially the use of pandas DataFrames and Series
            may become safer.

        Returns
        -------
        linopy.Model

        """
        self._xCounter = 0
        self._cCounter = 0
        self._varnameCounter = counter()
        self._connameCounter = counter()

        self.chunk = chunk
        self.status = "initialized"
        self.objective_value = np.nan
        self.force_dim_names = force_dim_names

        self.variables = Variables(model=self)
        self.constraints = Constraints(model=self)

        for attr in self.array_attrs:
            setattr(self, attr, Dataset())

        self.objective = LinearExpression()

        if solver_dir is None:
            self.solver_dir = gettempdir()

    def __repr__(self):
        """Return a string representation of the linopy model."""
        var_string = self.variables.defs.__repr__().split("\n", 1)[1]
        var_string = var_string.replace("Data variables:\n", "Data:\n")
        con_string = self.constraints.defs.__repr__().split("\n", 1)[1]
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

    def add_variables(
        self, lower=-inf, upper=inf, coords=None, name=None, mask=None, binary=False
    ):
        """
        Assign a new, possibly multi-dimensional array of variables to the model.

        Variables may be added with lower and/or upper bounds. Unless a
        `coords` argument is provided, the shape of the lower and upper bounds
        define the number of variables which will be added to the model
        under the name `name`.

        Parameters
        ----------
        lower : float/array_like, optional
            Lower bound of the variable(s). Ignored if `binary` is True.
            The default is -inf.
        upper : TYPE, optional
            Upper bound of the variable(s). Ignored if `binary` is True.
            The default is inf.
        coords : list/xarray.Coordinates, optional
            The coords of the variable array. For every single combination of
            coordinates a optimization variable is added to the model.
            The default is None. Is ignored when lower and upper bound provide
            coordinates.
        name : str, optional
            Reference name of the added variables. The default None results in
            a name like "var1", "var2" etc.
        mask : array_like, optional
            Boolean mask with False values for variables which are skipped.
            The shape of the mask has to match the shape the added variables.
            Default is None.
        binary : bool
            Whether the new variable is a binary variable which are used for
            Mixed-Integer problems.

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
            name = "var" + str(next(self._varnameCounter))

        assert (
            name not in self.variables.defs
        ), f"Variable '{name}' already assigned to model"

        if not binary:
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
        else:
            lower = DataArray()
            upper = DataArray()

        defs = DataArray(coords=coords).assign_attrs(binary=binary)

        check_force_dim_names(self, defs)

        start = self._xCounter
        defs.data = np.arange(start, start + defs.size).reshape(defs.shape)
        self._xCounter += defs.size

        if mask is not None:
            # assert defs.broadcast_equals(mask), (
            #     "The variable and the mask do not have the same coordinates.")
            defs = defs.where(mask, -1)

        if self.chunk:
            defs = defs.chunk(self.chunk)
            lower = lower.chunk(self.chunk)
            upper = upper.chunk(self.chunk)

        self.variables.add(name, defs, lower, upper)

        return Variable(defs, name=name, model=self)

    def add_constraints(self, lhs, sign, rhs, name=None, mask=None):
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
        mask : array_like, optional
            Boolean mask with False values for variables which are skipped.
            The shape of the mask has to match the shape the added variables.
            Default is None.


        Returns
        -------
        defs : linopy.model.Constraint
            Array containing the labels of the added constraints.

        """
        if name is None:
            name = "con" + str(next(self._connameCounter))

        assert name not in self.constraints.defs

        if isinstance(lhs, (list, tuple)):
            lhs = self.linexpr(*lhs)
        elif isinstance(lhs, Variable):
            lhs = lhs.to_linexpr()
        assert isinstance(lhs, LinearExpression)

        sign = DataArray(sign)
        rhs = DataArray(rhs)

        if (sign == "==").any():
            raise ValueError('Sign "==" not supported, use "=" instead.')

        defs = (lhs.vars.chunk() + rhs).sum("_term")

        check_force_dim_names(self, defs)

        start = self._cCounter
        defs.data = np.arange(start, start + defs.size).reshape(defs.shape)
        self._cCounter += defs.size

        if mask is not None:
            # assert defs.broadcast_equals(mask), (
            #     "The constraint and the mask do not have the same coordinates.")
            defs = defs.where(mask, -1)

        lhs = lhs.rename({"_term": f"{name}_term"})

        if self.chunk:
            lhs = lhs.chunk(self.chunk)
            sign = sign.chunk(self.chunk)
            rhs = rhs.chunk(self.chunk)
            defs = defs.chunk(self.chunk)

        self.constraints.add(name, defs, lhs.coeffs, lhs.vars, sign, rhs)

        return Constraint(defs, name=name, model=self)

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

        This function removes all constraints where the variable was used.

        Parameters
        ----------
        name : str
            Reference name of the variables which to remove, same as used in
            `model.add_variables`.

        Returns
        -------
        None.

        """
        labels = self.variables.defs[name]
        self.variables.remove(name)

        remove_b = self.constraints.vars.isin(labels).any()
        names = [name for name, remove in remove_b.items() if remove.item()]
        self.constraints.remove(names)

        self.objective = self.objective.sel(_term=~self.objective.vars.isin(labels))

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
        self.constraints.remove(name)

    @property
    def _binary_variables(self):
        return [v for v in self.variables if self.variables[v].attrs["binary"]]

    @property
    def _non_binary_variables(self):
        return [v for v in self.variables if not self.variables[v].attrs["binary"]]

    @property
    def binaries(self):
        return self.variables[self._binary_variables]

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

    def _eval(self, expr: str, **kwargs):
        from pandas.core.computation.eval import eval as pd_eval

        kwargs.setdefault("engine", "python")
        resolvers = kwargs.pop("resolvers", None)
        kwargs["level"] = kwargs.pop("level", 0) + 1
        resolvers = [self.variables.defs, self.parameters]
        kwargs["resolvers"] = kwargs.get("resolvers", ()) + tuple(resolvers)
        return pd_eval(expr, inplace=False, **kwargs)

    def vareval(self, expr: str, eval_kw={}, **kwargs):
        """
        Define a variable based a string expression (experimental).

        The function mirrors the behavior of `pandas.DataFrame.eval()`, e.g.
        global variables can be referenced with a @-suffix, model attributes
        such as parameters and variables can be referenced by the key.

        Parameters
        ----------
        expr : str
            Valid string to be compiled as a variable definition (lower and upper bounds!).
        eval_kw : dict
            Keyword arguments to be passed to `pandas.eval`.
        **kwargs :
            Keyword arguments to be passed to `model.add_constraints`.


        Returns
        -------
        linopy.Variable
            Variable which was added to the model.


        Examples
        --------

        >>> import linopy, xarray as xr
        >>> m = linopy.Model()
        >>> lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
        >>> upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
        >>> m.vareval("@lower <= x <= @upper")

        This is the same as
        >>> m.add_variables(lower, upper, name="x")

        """

    def lineval(self, expr: str, eval_kw={}, **kwargs):
        """
        Evaluate linear expressions given as a string (experimental).

        The function mirrors the behavior of `pandas.DataFrame.eval()`, e.g.
        global variables can be referenced with a @-suffix, model attributes
        such as parameters and variables can be referenced by the key.

        Parameters
        ----------
        expr : str
            Valid string to be compiled as a linear expression.
        eval_kw : dict
            Keyword arguments to be passed to `pandas.eval`.
        **kwargs :
            Keyword arguments to be passed to `LinearExpression.from_tuples`.

        Returns
        -------
        A linear expression based on the input string.


        Examples
        --------

        >>> import linopy, xarray as xr
        >>> m = linopy.Model()
        >>> lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
        >>> upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
        >>> m.add_variables(lower, upper, name="x")
        >>> m.add_variables(lower, upper, name="y")
        >>> c = xr.DataArray(np.random.rand(10, 10), coords=[range(10), range(10)])

        Now create the linear expression
        >>> m.lineval("@c * x - y")

        This is the same as
        >>> m.linexpr((c, "x"), (-1, "y"))

        """
        eval_kw["level"] = eval_kw.pop("level", 1) + 1

        tuples = Expr(expr).to_string_tuples()
        tuples = [
            (self._eval(c, **eval_kw), self._eval(v, **eval_kw)) for (c, v) in tuples
        ]
        return self.linexpr(*tuples, **kwargs)

    def coneval(self, expr: str, eval_kw={}, **kwargs):
        """
        Define a constraint determined by a string expression (experimental).

        The function mirrors the behavior of `pandas.DataFrame.eval()`, e.g.
        global variables can be referenced with a @-suffix, model attributes
        such as parameters and variables can be referenced by the key.

        Parameters
        ----------
        expr : str
            Valid string to be compiled as a linear expression.
        eval_kw : dict
            Keyword arguments to be passed to `pandas.eval`.
        **kwargs :
            Keyword arguments to be passed to `model.add_constraints`.


        Returns
        -------
        con : xarray.DataArray
            Array containing the labels of the added constraints.


        Examples
        --------

        >>> import linopy, xarray as xr
        >>> m = linopy.Model()
        >>> lower = xr.DataArray(np.zeros((10, 10)), coords=[range(10), range(10)])
        >>> upper = xr.DataArray(np.ones((10, 10)), coords=[range(10), range(10)])
        >>> m.add_variables(lower, upper, name="x")
        >>> m.add_variables(lower, upper, name="y")
        >>> c = xr.DataArray(np.random.rand(10, 10), coords=[range(10), range(10)])

        Now create the constraint:
        >>> m.coneval("@c * x - y <= 5 ")

        This is the same as
        >>> lhs = m.linexpr((c, "x"), (-1, "y"))
        >>> m.add_constraints(lhs, "<=", 5)

        """
        eval_kw["level"] = eval_kw.pop("level", 1) + 1

        (lhs, sign, rhs), kw = Expr(expr).to_constraint_args_kwargs()
        lhs = [(self._eval(c, **eval_kw), self._eval(v, **eval_kw)) for (c, v) in lhs]
        lhs = self.linexpr(*lhs)
        rhs = self._eval(rhs)
        return self.add_constraints(lhs, sign, rhs, **kw, **kwargs)

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

        tmp_kwargs = dict(mode="w", delete=False, dir=self.solver_dir)
        if problem_fn is None:
            with NamedTemporaryFile(
                suffix=".lp", prefix="linopy-problem-", **tmp_kwargs
            ) as f:
                problem_fn = f.name
        if solution_fn is None:
            with NamedTemporaryFile(
                suffix=".sol", prefix="linopy-solve-", **tmp_kwargs
            ) as f:
                solution_fn = f.name
        if log_fn is not None:
            logger.info(f"Solver logs written to `{log_fn}`.")

        try:
            self.to_file(problem_fn)
            solve = getattr(solvers, f"run_{solver_name}")
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
                f"Optimization solution is sub-optimal. Objective value: {obj:.2e}"
            )
        else:
            logger.warning(
                f"Optimization failed with status {status} and "
                f"termination condition {termination_condition}"
            )
            return status, termination_condition

        self.objective_value = obj
        self.solver_model = res.pop("model", None)
        self.status = termination_condition

        res["solution"].loc[-1] = np.nan
        for v in self.variables:
            idx = np.ravel(self.variables[v])
            sol = res["solution"][idx].values.reshape(self.variables[v].shape)
            self.solution[v] = xr.DataArray(sol, self.variables[v].coords)

        res["dual"].loc[-1] = np.nan
        for c in self.constraints:
            idx = np.ravel(self.constraints[c])
            du = res["dual"][idx].values.reshape(self.constraints[c].shape)
            self.dual[c] = xr.DataArray(du, self.constraints[c].coords)

        return status, termination_condition

    to_netcdf = to_netcdf

    to_file = to_file


def counter():
    """
    Create a counter generator that counts from 0 on upwards.

    Yields
    ------
    num : int
        Current counting.

    """
    num = 0
    while True:
        yield num
        num += 1


def check_force_dim_names(model, ds):
    """
    Ensure that the added data does not lead to unintended broadcasting.

    Parameters
    ----------
    model : linopy.Model
    ds : xr.DataArray/Variable/LinearExpression
        Data that should be added to the model.

    Raises
    ------
    ValueError
        If broadcasted data leads to unspecified dimension names.

    Returns
    -------
    None.

    """
    if model.force_dim_names:
        if any(bool(re.match(r"dim_[0-9]+", dim)) for dim in ds.dims):
            raise ValueError(
                "Added data contains non-customized dimension names. This is not "
                "allowed when setting `force_dim_names` to True."
            )


def _merge_inplace(self, attr, da, name, **kwargs):
    """
    Assign a new dataarray to the dataset `attr` by merging.

    This takes care of all coordinate alignments, instead of a direct
    assignment like self.variables[name] = var
    """
    ds = merge([getattr(self, attr), da.rename(name)], **kwargs)
    setattr(self, attr, ds)


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

    __slots__ = ("_cache", "_coords", "_indexes", "_name", "_variable", "model")

    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop("model", None)
        super().__init__(*args, **kwargs)

    # We have to set the _reduce_method to None, in order to overwrite basic
    # reduction functions as `sum`. There might be a better solution (?).
    _reduce_method = None

    def to_array(self):
        """Convert the variable array to a xarray.DataArray."""
        return DataArray(self)

    def to_linexpr(self, coefficient=1):
        """Create a linear exprssion from the variables."""
        return LinearExpression.from_tuples((coefficient, self))

    def __repr__(self):
        """Get the string representation of the variables."""
        data_string = "Variables:\n" + self.to_array().__repr__().split("\n", 1)[1]
        extend_line = "-" * len(self.name)
        return (
            f"Variable container '{self.name}':\n"
            f"----------------------{extend_line}\n\n"
            f"{data_string}"
        )

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

    def group_terms(self, group):
        """
        Sum variable over groups.

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

    # would like to have this as a property, but this does not work apparently
    def upper_bound(self):
        if self.model is None:
            raise AttributeError("No reference model is assigned to the variable.")
        return self.model.variables_upper_bound[self.name]

    def lower_bound(self):
        if self.model is None:
            raise AttributeError("No reference model is assigned to the variable.")
        return self.model.variables_lower_bound[self.name]

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


@dataclass
class Variables:
    defs: Dataset = Dataset()
    lower: Dataset = Dataset()
    upper: Dataset = Dataset()
    model: Model = None

    data_attrs = ["defs", "lower", "upper"]
    data_attr_names = ["Variables References", "Lower bounds", "Upper bounds"]

    def __getitem__(
        self, names: Union[str, Sequence[str]]
    ) -> Union[Variable, "Variables"]:
        if isinstance(names, str):
            return Variable(self.defs[names], model=self.model)

        return self.__class__(
            self.defs[names], self.lower[names], self.upper[names], self.model
        )

    def __repr__(self):
        """Return a string representation of the linopy model."""
        r = "linopy.model.Variables"
        line = "=" * len(r)
        r += f"\n{line}\n\n"
        for (k, K) in zip(self.data_attrs, self.data_attr_names):
            s = getattr(self, k).__repr__().split("\n", 1)[1]
            s = s.replace("Data variables:\n", "Data:\n")
            line = "-" * (len(K) + 1)
            r += f"{K}:\n{line}\n{s}\n\n"
        return r

    _merge_inplace = _merge_inplace

    def add(self, name, defs: DataArray, lower: DataArray, upper: DataArray):
        self._merge_inplace("defs", defs, name, fill_value=-1)
        self._merge_inplace("lower", lower, name)
        self._merge_inplace("upper", upper, name)

    def remove(self, name):
        for attr in self.data_attrs:
            ds = getattr(self, attr)
            if name in ds:
                setattr(self, attr, ds.drop(name))


class Constraint(DataArray):
    """
    Constraint Container for storing constraint labels.
    The Constraint class is a subclass of xr.DataArray hence most xarray functions
    can be applied to it.
    """

    __slots__ = ("_cache", "_coords", "_indexes", "_name", "_variable", "model")

    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop("model", None)
        super().__init__(*args, **kwargs)

    # We have to set the _reduce_method to None, in order to overwrite basic
    # reduction functions as `sum`. There might be a better solution (?).
    _reduce_method = None

    def __repr__(self):
        """Get the string representation of the constraints."""
        data_string = "Constraints:\n" + self.to_array().__repr__().split("\n", 1)[1]
        extend_line = "-" * len(self.name)
        return (
            f"Variable container '{self.name}':\n"
            f"----------------------{extend_line}\n\n"
            f"{data_string}"
        )

    def _repr_html_(self):
        """Get the html representation of the variables."""
        # return self.__repr__()
        data_string = self.to_array()._repr_html_()
        data_string = data_string.replace("xarray.DataArray", "linopy.Constraint")
        return data_string

    def to_array(self):
        """Convert the variable array to a xarray.DataArray."""
        return DataArray(self)


@dataclass
class Constraints:
    """
    A slightly more helpful representation of all constraints in a model
    which aims at providing easy block writing methods for the constraints.
    """

    defs: Dataset = Dataset()
    coeffs: Dataset = Dataset()
    vars: Dataset = Dataset()
    sign: Dataset = Dataset()
    rhs: Dataset = Dataset()
    model: Model = None

    data_attrs = ["defs", "coeffs", "vars", "sign", "rhs"]
    data_attr_names = [
        "Constraints References",
        "Left-hand-side Coefficients",
        "Left-hand-side Variables",
        "Signs",
        "Right-hand-side Constants",
    ]

    def __repr__(self):
        """Return a string representation of the linopy model."""
        r = "linopy.model.Constraints"
        line = "=" * len(r)
        r += f"\n{line}\n\n"
        for (k, K) in zip(self.data_attrs, self.data_attr_names):
            s = getattr(self, k).__repr__().split("\n", 1)[1]
            s = s.replace("Data variables:\n", "Data:\n")
            line = "-" * (len(K) + 1)
            r += f"{K}:\n{line}\n{s}\n\n"
        return r

    def __getitem__(
        self, names: Union[str, Sequence[str]]
    ) -> Union[Constraint, "Constraints"]:
        if isinstance(names, str):
            return Constraint(self.defs[names], self.model)

        return self.__class__(
            self.defs[names],
            self.coeffs[names],
            self.vars[names],
            self.sign[names],
            self.rhs[names],
            self.model,
        )

    _merge_inplace = _merge_inplace

    def add(
        self,
        name,
        defs: DataArray,
        coeffs: DataArray,
        vars: DataArray,
        sign: DataArray,
        rhs: DataArray,
    ):
        self._merge_inplace("defs", defs, name, fill_value=-1)
        self._merge_inplace("coeffs", coeffs, name)
        self._merge_inplace("vars", vars, name)
        self._merge_inplace("sign", sign, name)
        self._merge_inplace("rhs", rhs, name)

    def remove(self, name):
        for attr in self.data_attrs:
            setattr(self, attr, getattr(self, attr).drop(name))

    @property
    def inequalities(self):
        return self[[n for n, s in self.sign.items() if s in ("<=", ">=")]]

    @property
    def equalities(self):
        return self[[n for n, s in self.sign.items() if s in ("=", "==")]]

    def block_sizes(self, num_blocks, block_map) -> np.ndarray:
        sizes = np.zeros(num_blocks + 1, dtype=int)
        for name in self.defs:
            sizes += self[name].block_sizes(num_blocks, block_map)
        return sizes


def _merge_inplace(self, attr, da, **kwargs):
    """
    Assign a new dataarray to the dataset `attr` by merging.

    This takes care of all coordinate alignments, instead of a direct
    assignment like self.variables[name] = var
    """
    ds = merge([getattr(self, attr), da], **kwargs)
    setattr(self, attr, ds)


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
            dataset = dataset.transpose(..., "_term")
        else:
            dataset = xr.Dataset({"coeffs": DataArray([]), "vars": DataArray([])})
            dataset = dataset.assign_coords(_term=[])
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

        ds = (
            self.rename(_term="_stacked_term")
            .stack(_term=["_stacked_term"] + dims)
            .reset_index("_term", drop=True)
        )
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
        ds_list = [Dataset({"coeffs": c, "vars": v}) for c, v in tuples]
        if len(ds_list) > 1:
            ds = xr.concat(ds_list, dim="_term", coords="minimal")
        else:
            ds = ds_list[0].expand_dims("_term")
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
        return self.shape == (0,)
