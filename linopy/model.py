# -*- coding: utf-8 -*-
"""
Linopy model module.
This module contains frontend implementations of the package.
"""

import logging
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir

import numpy as np
import pandas as pd
import xarray as xr
from numpy import inf, nan
from xarray import DataArray, Dataset

from linopy import solvers
from linopy.common import best_int, replace_by_map
from linopy.constraints import Constraints
from linopy.eval import Expr
from linopy.expressions import AnonymousConstraint, LinearExpression
from linopy.io import to_block_files, to_file, to_netcdf
from linopy.solvers import available_solvers
from linopy.variables import Variable, Variables

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

    __slots__ = (
        # containers
        "_variables",
        "_constraints",
        "_objective",
        "_parameters",
        "_solution",
        "_dual",
        # hidden attributes
        "_status",
        "_termination_condition",
        "_xCounter",
        "_cCounter",
        "_varnameCounter",
        "_connameCounter",
        "_blocks",
        "_objective_value",
        # TODO: check if these should not be mutable
        "_chunk",
        "_force_dim_names",
        "_solver_dir",
        "solver_model",
    )

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
        self._variables = Variables(model=self)
        self._constraints = Constraints(model=self)
        self._objective = LinearExpression()
        self._parameters = Dataset()

        self._solution = Dataset()
        self._dual = Dataset()
        self._objective_value = nan

        self._status = "initialized"
        self._xCounter = 0
        self._cCounter = 0
        self._varnameCounter = 0
        self._connameCounter = 0
        self._blocks = None

        self._chunk = chunk
        self._force_dim_names = bool(force_dim_names)
        self._solver_dir = Path(gettempdir() if solver_dir is None else solver_dir)

    @property
    def variables(self):
        "Variables assigned to the model."
        return self._variables

    @property
    def constraints(self):
        "Constraints assigned to the model."
        return self._constraints

    @property
    def objective(self):
        "Objective assigned to the model."
        return self._objective

    @objective.setter
    def objective(self, value) -> LinearExpression:
        self.add_objective(value, overwrite=True)

    @property
    def parameters(self):
        """
        Parameters assigned to the model.

        The parameters serve as an expta field where additional data may be
        stored.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        self._parameters = Dataset(value)

    @property
    def solution(self):
        "Solution calculated by the optimization."
        return self._solution

    @solution.setter
    def solution(self, value):
        self._solution = Dataset(value)

    @property
    def dual(self):
        "Dual values calculated by the optimization."
        return self._dual

    @dual.setter
    def dual(self, value):
        self._dual = Dataset(value)

    @property
    def status(self):
        "Status of the model."
        return self._status

    @status.setter
    def status(self, value):
        assert value in ["initialized", "ok", "warning"]
        self._status = value

    @property
    def termination_condition(self):
        "Termination condition of the model."
        return self._termination_condition

    @termination_condition.setter
    def termination_condition(self, value):
        self._termination_condition = str(value)

    @property
    def objective_value(self):
        "Objective value of the model."
        return self._objective_value

    @objective_value.setter
    def objective_value(self, value):
        self._objective_value = float(value)

    @property
    def chunk(self):
        "Chunk sizes of the model."
        return self._chunk

    @chunk.setter
    def chunk(self, value):
        if not isinstance(value, [int, dict]) and (value != "auto"):
            raise TypeError("Chunks must int, dict, or 'auto'.")
        self._chunk = value

    @property
    def blocks(self):
        "Blocks of the model."
        return self._blocks

    @blocks.setter
    def blocks(self, value):
        self._blocks = DataArray(value)

    @property
    def force_dim_names(self):
        """
        Whether assigned variables, constraints and data should always have
        custom dimension names, i.e. not matching dimension names "dim_0",
        "dim_1" and so on. These helps to avoid unintended broadcasting
        over dimension. Especially the use of pandas DataFrames and Series
        may become safer.
        """
        return self._force_dim_names

    @force_dim_names.setter
    def force_dim_names(self, value):
        self._force_dim_names = bool(value)

    @property
    def solver_dir(self):
        "Solver directory of the model."
        return self._solver_dir

    @solver_dir.setter
    def solver_dir(self, value):
        if not isinstance(value, [str, Path]):
            raise TypeError("'solver_dir' must path-like.")
        self._solver_dir = Path(value)

    @property
    def dataset_attrs(self):
        return ["parameters", "solution", "dual"]

    @property
    def scalar_attrs(self):
        return [
            "objective_value",
            "status",
            "_xCounter",
            "_cCounter",
            "_varnameCounter",
            "_connameCounter",
            "force_dim_names",
        ]

    def __repr__(self):
        """Return a string representation of the linopy model."""
        var_string = self.variables.labels.__repr__().split("\n", 1)[1]
        var_string = var_string.replace("Data variables:\n", "Data:\n")
        con_string = self.constraints.labels.__repr__().split("\n", 1)[1]
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

    def check_force_dim_names(self, ds):
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
        contains_default_dims = any(
            bool(re.match(r"dim_[0-9]+", dim)) for dim in ds.dims
        )
        if self.force_dim_names and contains_default_dims:
            raise ValueError(
                "Added data contains non-customized dimension names. This is not "
                "allowed when setting `force_dim_names` to True."
            )
        else:
            return

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
            name = "var" + str(self._varnameCounter)
            self._varnameCounter += 1

        if name in self.variables:
            raise ValueError(f"Variable '{name}' already assigned to model")

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
            # for general compatibility when ravelling all values set non-nan
            lower = DataArray(-inf)
            upper = DataArray(inf)

        labels = DataArray(coords=coords).assign_attrs(binary=binary)

        self.check_force_dim_names(labels)

        start = self._xCounter
        labels.data = np.arange(start, start + labels.size).reshape(labels.shape)
        self._xCounter += labels.size

        if mask is not None:
            mask = DataArray(mask)
            assert set(mask.dims).issubset(
                labels.dims
            ), "Dimensions of mask not a subset of resulting labels dimensions."
            labels = labels.where(mask, -1)

        if self.chunk:
            labels = labels.chunk(self.chunk)
            lower = lower.chunk(self.chunk)
            upper = upper.chunk(self.chunk)

        self._variables.add(name, labels, lower, upper)

        return self.variables[name]

    def add_constraints(self, lhs, sign=None, rhs=None, name=None, mask=None):
        """
        Assign a new, possibly multi-dimensional array of constraints to the model.

        Constraints are added by defining a left hand side (lhs), the sign and
        the right hand side (rhs). The lhs has to be a linopy.LinearExpression
        and the rhs a constant (array of constants). The function return the
        an array with the constraint labels (integers).

        Parameters
        ----------
        lhs : linopy.LinearExpression/linopy.AnonynousConstraint
            Left hand side of the constraint(s) or optionally full constraint.
            In case a linear expression is passed, `sign` and `rhs` must not be
            None.
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
        labels : linopy.model.Constraint
            Array containing the labels of the added constraints.

        """
        if name is None:
            name = "con" + str(self._connameCounter)
            self._connameCounter += 1

        if name in self.constraints:
            raise ValueError(f"Constraint '{name}' already assigned to model")

        if isinstance(lhs, AnonymousConstraint):
            if sign is not None or rhs is not None:
                raise ValueError(
                    "Passing arguments `sign` and `rhs` together with a constraint"
                    " is ambiguous."
                )
            sign = lhs.sign
            rhs = lhs.rhs
            lhs = lhs.lhs
        else:
            if sign is None or rhs is None:
                raise ValueError(
                    "Argument `sign` and `rhs` must not be None if first argument "
                    " is an expression."
                )

        if isinstance(lhs, (list, tuple)):
            lhs = self.linexpr(*lhs)
        elif isinstance(lhs, Variable):
            lhs = lhs.to_linexpr()
        assert isinstance(lhs, LinearExpression)

        lhs = lhs.sanitize()
        sign = DataArray(sign)
        rhs = DataArray(rhs)

        if (sign == "==").any():
            raise ValueError('Sign "==" not supported, use "=" instead.')

        labels = (lhs.vars.chunk() + rhs).sum("_term")

        self.check_force_dim_names(labels)

        start = self._cCounter
        labels.data = np.arange(start, start + labels.size).reshape(labels.shape)
        self._cCounter += labels.size

        if mask is not None:
            mask = DataArray(mask)
            assert set(mask.dims).issubset(
                labels.dims
            ), "Dimensions of mask not a subset of resulting labels dimensions."
            labels = labels.where(mask, -1)

        lhs = lhs.rename({"_term": f"{name}_term"})

        if self.chunk:
            lhs = lhs.chunk(self.chunk)
            sign = sign.chunk(self.chunk)
            rhs = rhs.chunk(self.chunk)
            labels = labels.chunk(self.chunk)

        self._constraints.add(name, labels, lhs.coeffs, lhs.vars, sign, rhs)

        return self.constraints[name]

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
        elif isinstance(expr, DataArray):
            expr = LinearExpression(expr)
        assert isinstance(expr, LinearExpression)

        if self.chunk is not None:
            expr = expr.chunk(self.chunk)

        if expr.vars.ndim > 1:
            expr = expr.sum()
        self._objective = expr
        return self._objective

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
        labels = self.variables.labels[name]
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
    def binaries(self):
        "Get all binary variables."
        return self.variables.binaries

    @property
    def non_binaries(self):
        "Get all non-binary variables."
        return self.variables.non_binaries

    @property
    def nvars(self):
        "Get the total number of variables."
        return self.variables.nvars

    @property
    def ncons(self):
        "Get the total number of constraints."
        return self.constraints.ncons

    @property
    def blocks(self):
        """
        Blocks used as a basis to split the variables and constraint matrix.
        """
        return self._blocks

    @blocks.setter
    def blocks(self, blocks):
        if not isinstance(blocks, DataArray):
            raise TypeError("Blocks must be of type DataArray")
        assert len(blocks.dims) == 1

        dtype = best_int(int(blocks.max()) + 1)
        blocks = blocks.astype(dtype)

        if self.chunk is not None:
            blocks = blocks.chunk(self.chunk)

        self._blocks = blocks

    def calculate_block_maps(self):
        """
        Calculate the matrix block mappings based on dimensional blocks.
        """
        assert self.blocks is not None, "Blocks are not defined."

        dtype = self.blocks.dtype
        self.variables.blocks = self.variables.get_blocks(self.blocks)
        block_map = self.variables.blocks_to_blockmap(self.variables.blocks, dtype)

        self.constraints.blocks = self.constraints.get_blocks(block_map)

        blocks = replace_by_map(self.objective.vars, block_map)
        self.objective = self.objective.assign(blocks=blocks)

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
        resolvers = [self.variables.labels, self.parameters]
        kwargs["resolvers"] = kwargs.get("resolvers", ()) + tuple(resolvers)
        return pd_eval(expr, inplace=False, **kwargs)

    def vareval(self, expr: str, eval_kw=None, **kwargs):
        """
        Define a variable based a string expression (experimental).

        The function mirrors the behavior of `pandas.DataFrame.eval()`, e.g.
        global variables can be referenced with a @-suffix, model attributes
        such as parameters and variables can be referenced by the key.

        Parameters
        ----------
        expr : str
            Valid string to be compiled as a variable definition
            (lower and upper bounds!).
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
        if eval_kw is None:
            eval_kw = {}

        eval_kw["level"] = eval_kw.pop("level", 1) + 1

        kw = Expr(expr).to_variable_kwargs()
        for k in ["lower", "upper"]:
            if k in kw:
                kw[k] = self._eval(kw[k], **eval_kw)

        return self.add_variables(**kw, **kwargs)

    def lineval(self, expr: str, eval_kw=None, **kwargs):
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
        if eval_kw is None:
            eval_kw = {}

        eval_kw["level"] = eval_kw.pop("level", 1) + 1

        tuples = Expr(expr).to_string_tuples()
        tuples = [
            (self._eval(c, **eval_kw), self._eval(v, **eval_kw)) for (c, v) in tuples
        ]
        return self.linexpr(*tuples, **kwargs)

    def coneval(self, expr: str, eval_kw=None, **kwargs):
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
        if eval_kw is None:
            eval_kw = {}

        eval_kw["level"] = eval_kw.pop("level", 1) + 1

        (lhs, sign, rhs), kw = Expr(expr).to_constraint_args_kwargs()
        lhs = [(self._eval(c, **eval_kw), self._eval(v, **eval_kw)) for (c, v) in lhs]
        lhs = self.linexpr(*lhs)
        rhs = self._eval(rhs, **eval_kw)
        return self.add_constraints(lhs, sign, rhs, **kw, **kwargs)

    @property
    def coefficientrange(self):
        """Coefficient range of the constraints in the model."""
        return self.constraints.coefficientrange

    @property
    def objectiverange(self):
        """Objective range of the objective in the model."""
        return pd.Series(
            [self.objective.coeffs.min().item(), self.objective.coeffs.max().item()],
            index=["min", "max"],
        )

    def get_solution_file(self, solution_fn=None):
        """Get a fresh created solution file if solution file is None."""
        if solution_fn is None:
            kwargs = dict(
                prefix="linopy-solve-",
                suffix=".sol",
                mode="w",
                dir=self.solver_dir,
                delete=False,
            )
            with NamedTemporaryFile(**kwargs) as f:
                return f.name
        else:
            return solution_fn

    def get_problem_file(self, problem_fn=None):
        """Get a fresh created problem file if problem file is None."""
        if problem_fn is None:
            kwargs = dict(
                prefix="linopy-problem-",
                suffix=".lp",
                mode="w",
                dir=self.solver_dir,
                delete=False,
            )
            with NamedTemporaryFile(**kwargs) as f:
                return f.name
        else:
            return problem_fn

    def solve(
        self,
        solver_name="gurobi",
        io_api=None,
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
        io_api : str, optional
            Api to use for communicating with the solver, must be one of
            {'lp', 'direct'}. If set to 'lp' the problem is written to an
            LP file which is then read by the solver. If set to
            'direct' the problem is communicated to the solver via the solver
            specific API, e.g. gurobipy. This may lead to faster run times.
            The default is set to 'lp' if available.
        problem_fn : path_like, optional
            Path of the lp file or output file/directory which is written out
            during the process. The default None results in a temporary file.
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

        # reset result
        self.solution = xr.Dataset()
        self.dual = xr.Dataset()

        if log_fn is not None:
            logger.info(f"Solver logs written to `{log_fn}`.")

        problem_fn = self.get_problem_file(problem_fn)
        solution_fn = self.get_solution_file(solution_fn)

        try:
            func = getattr(solvers, f"run_{solver_name}")
            res = func(
                self,
                io_api,
                problem_fn,
                solution_fn,
                log_fn,
                warmstart_fn,
                basis_fn,
                keep_files,
                **solver_options,
            )
        finally:
            for fn in (problem_fn, solution_fn):
                if fn is not None:
                    if os.path.exists(fn) and not keep_files:
                        os.remove(fn)

        status = res.pop("status")
        termination_condition = res.pop("termination_condition")
        obj = res.pop("objective", None)
        self.solver_model = res.pop("model", None)

        if status == "ok" and termination_condition == "optimal":
            logger.info(f" Optimization successful. Objective value: {obj:.2e}")
        elif status == "warning" and termination_condition == "suboptimal":
            logger.warning(
                f"Optimization solution is sub-optimal. Objective value: {obj:.2e}"
            )
        else:
            logger.warning(
                f"Optimization failed with status `{status}` and "
                f"termination condition `{termination_condition}`."
            )
            return status, termination_condition

        self.objective_value = obj
        self.status = status
        self.termination_condition = termination_condition

        # map solution and dual to original shape which includes missing values
        sol = res["solution"]
        sol.loc[-1] = nan

        for name, labels in self.variables.labels.items():
            idx = np.ravel(labels)
            vals = sol[idx].values.reshape(labels.shape)
            self.solution[name] = xr.DataArray(vals, labels.coords)

        if res["dual"] is not None:
            dual = res["dual"]
            dual.loc[-1] = nan

            for name, labels in self.constraints.labels.items():
                idx = np.ravel(labels)
                vals = dual[idx].values.reshape(labels.shape)
                self.dual[name] = xr.DataArray(vals, labels.coords)

        return status, termination_condition

    def compute_set_of_infeasible_constraints(self):
        """
        Print out the infeasible subset of constraints.

        This is a prelimary function and is only implemented for gurobi so far.
        """
        import gurobipy

        solver_model = getattr(self, "solver_model")

        if not isinstance(solver_model, gurobipy.Model):
            raise NotImplementedError("Solver model must be a Gurobi Model.")

        solver_model.computeIIS()
        with NamedTemporaryFile(suffix=".ilp", prefix="linopy-iis-") as f:
            solver_model.write(f.name)
            print(f.read().decode())

    to_netcdf = to_netcdf

    to_file = to_file

    to_block_files = to_block_files
