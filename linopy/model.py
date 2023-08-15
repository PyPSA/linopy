# -*- coding: utf-8 -*-
"""
Linopy model module.

This module contains frontend implementations of the package.
"""

import logging
import os
import re
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir

import numpy as np
import pandas as pd
import xarray as xr
from deprecation import deprecated
from numpy import inf, nan
from xarray import DataArray, Dataset

from linopy import solvers
from linopy.common import (
    as_dataarray,
    best_int,
    maybe_replace_signs,
    replace_by_map,
    save_join,
)
from linopy.constants import TERM_DIM, ModelStatus, TerminationCondition
from linopy.constraints import AnonymousScalarConstraint, Constraint, Constraints
from linopy.expressions import (
    LinearExpression,
    QuadraticExpression,
    ScalarLinearExpression,
)
from linopy.io import to_block_files, to_file, to_gurobipy, to_highspy, to_netcdf
from linopy.matrices import MatrixAccessor
from linopy.solvers import available_solvers, quadratic_solvers
from linopy.variables import ScalarVariable, Variable, Variables

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
        # TODO: move counters to Variables and Constraints class
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
        "matrices",
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
        self._objective = LinearExpression(None, self)
        self._parameters = Dataset()

        self._objective_value = nan

        self._status = "initialized"
        self._termination_condition = ""
        self._xCounter = 0
        self._cCounter = 0
        self._varnameCounter = 0
        self._connameCounter = 0
        self._blocks = None

        self._chunk = chunk
        self._force_dim_names = bool(force_dim_names)
        self._solver_dir = Path(gettempdir() if solver_dir is None else solver_dir)

        self.matrices = MatrixAccessor(self)

    @property
    def variables(self):
        """
        Variables assigned to the model.
        """
        return self._variables

    @property
    def constraints(self):
        """
        Constraints assigned to the model.
        """
        return self._constraints

    @property
    def objective(self):
        """
        Objective assigned to the model.
        """
        return self._objective

    @objective.setter
    def objective(self, value) -> LinearExpression:
        self.add_objective(value, overwrite=True)

    @property
    def parameters(self):
        """
        Parameters assigned to the model.

        The parameters serve as an extra field where additional data may
        be stored.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, value):
        self._parameters = Dataset(value)

    @property
    def solution(self):
        """
        Solution calculated by the optimization.
        """
        return self.variables.solution

    @property
    def dual(self):
        """
        Dual values calculated by the optimization.
        """
        return self.constraints.dual

    @property
    def status(self):
        """
        Status of the model.
        """
        return self._status

    @status.setter
    def status(self, value):
        self._status = ModelStatus[value].value

    @property
    def termination_condition(self):
        """
        Termination condition of the model.
        """
        return self._termination_condition

    @termination_condition.setter
    def termination_condition(self, value):
        self._termination_condition = TerminationCondition[value].value

    @property
    def objective_value(self):
        """
        Objective value of the model.
        """
        return self._objective_value

    @objective_value.setter
    def objective_value(self, value):
        self._objective_value = float(value)

    @property
    def chunk(self):
        """
        Chunk sizes of the model.
        """
        return self._chunk

    @chunk.setter
    def chunk(self, value):
        if not isinstance(value, [int, dict]) and (value != "auto"):
            raise TypeError("Chunks must int, dict, or 'auto'.")
        self._chunk = value

    @property
    def blocks(self):
        """
        Blocks of the model.
        """
        return self._blocks

    @blocks.setter
    def blocks(self, value):
        self._blocks = DataArray(value)

    @property
    def force_dim_names(self):
        """
        Whether assigned variables, constraints and data should always have
        custom dimension names, i.e. not matching dimension names "dim_0",
        "dim_1" and so on.

        These helps to avoid unintended broadcasting over dimension.
        Especially the use of pandas DataFrames and Series may become
        safer.
        """
        return self._force_dim_names

    @force_dim_names.setter
    def force_dim_names(self, value):
        self._force_dim_names = bool(value)

    @property
    def solver_dir(self):
        """
        Solver directory of the model.
        """
        return self._solver_dir

    @solver_dir.setter
    def solver_dir(self, value):
        if not isinstance(value, [str, Path]):
            raise TypeError("'solver_dir' must path-like.")
        self._solver_dir = Path(value)

    @property
    def dataset_attrs(self):
        return ["parameters"]

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
        """
        Return a string representation of the linopy model.
        """
        var_string = self.variables.__repr__().split("\n", 2)[2]
        con_string = self.constraints.__repr__().split("\n", 2)[2]
        model_string = f"Linopy {self.type} model"

        return (
            f"{model_string}\n{'=' * len(model_string)}\n\n"
            f"Variables:\n----------\n{var_string}\n"
            f"Constraints:\n------------\n{con_string}\n"
            f"Status:\n-------\n{self.status}"
        )

    def __getitem__(self, key):
        """
        Get a model variable by the name.
        """
        return self.variables[key]

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
        self,
        lower=-inf,
        upper=inf,
        coords=None,
        name=None,
        mask=None,
        binary=False,
        integer=False,
        **kwargs,
    ):
        """
        Assign a new, possibly multi-dimensional array of variables to the
        model.

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
            The coords of the variable array.
            These are directly passed to the DataArray creation of
            `lower` and `upper`. For every single combination of
            coordinates a optimization variable is added to the model.
            The default is None.
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
        integer : bool
            Whether the new variable is a integer variable which are used for
            Mixed-Integer problems.
        **kwargs :
            Additional keyword arguments are passed to the DataArray creation.

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
        >>> from linopy import Model
        >>> import pandas as pd
        >>> m = Model()
        >>> time = pd.RangeIndex(10, name="Time")
        >>> m.add_variables(lower=0, coords=[time], name="x")
        Variable (Time: 10)
        -------------------
        [0]: x[0] ∈ [0, inf]
        [1]: x[1] ∈ [0, inf]
        [2]: x[2] ∈ [0, inf]
        [3]: x[3] ∈ [0, inf]
        [4]: x[4] ∈ [0, inf]
        [5]: x[5] ∈ [0, inf]
        [6]: x[6] ∈ [0, inf]
        [7]: x[7] ∈ [0, inf]
        [8]: x[8] ∈ [0, inf]
        [9]: x[9] ∈ [0, inf]
        """
        if name is None:
            name = f"var{self._varnameCounter}"
            self._varnameCounter += 1

        if name in self.variables:
            raise ValueError(f"Variable '{name}' already assigned to model")

        if binary and integer:
            raise ValueError("Variable cannot be both binary and integer.")

        if binary:
            if (lower != -inf) or (upper != inf):
                raise ValueError("Binary variables cannot have lower or upper bounds.")
            else:
                lower, upper = 0, 1

        data = Dataset(
            {
                "lower": as_dataarray(lower, coords, **kwargs),
                "upper": as_dataarray(upper, coords, **kwargs),
            }
        )

        if mask is not None:
            mask = as_dataarray(mask, coords=data.coords, dims=data.dims).astype(bool)

        labels = DataArray(-2, coords=data.coords)

        self.check_force_dim_names(labels)

        start = self._xCounter
        end = start + labels.size
        labels.data = np.arange(start, end).reshape(labels.shape)
        self._xCounter += labels.size

        if mask is not None:
            labels = labels.where(mask, -1)

        data = data.assign(labels=labels).assign_attrs(
            label_range=(start, end), name=name, binary=binary, integer=integer
        )

        if self.chunk:
            data = data.chunk(self.chunk)

        variable = Variable(data, name=name, model=self)
        self.variables.add(variable)
        return variable

    def add_constraints(
        self, lhs, sign=None, rhs=None, name=None, coords=None, mask=None
    ):
        """
        Assign a new, possibly multi-dimensional array of constraints to the
        model.

        Constraints are added by defining a left hand side (lhs), the sign and
        the right hand side (rhs). The lhs has to be a linopy.LinearExpression
        and the rhs a constant (array of constants). The function return the
        an array with the constraint labels (integers).

        Parameters
        ----------
        lhs : linopy.LinearExpression/linopy.Constraint/callable
            Left hand side of the constraint(s) or optionally full constraint.
            In case a linear expression is passed, `sign` and `rhs` must not be
            None.
            If a function is passed, it is called for every combination of
            coordinates given in `coords`. It's first argument has to be the
            model, followed by scalar argument for each coordinate given in
            coordinates.
        sign : str/array_like
            Relation between the lhs and rhs, valid values are {'=', '>=', '<='}.
        rhs : int/float/array_like
            Right hand side of the constraint(s).
        name : str, optional
            Reference name of the added constraints. The default None results
            results a name like "con1", "con2" etc.
        coords : list/xarray.Coordinates, optional
            The coords of the constraint array. This is only used when lhs is
            a function. The default is None.
        mask : array_like, optional
            Boolean mask with False values for variables which are skipped.
            The shape of the mask has to match the shape the added variables.
            Default is None.


        Returns
        -------
        labels : linopy.model.Constraint
            Array containing the labels of the added constraints.
        """

        def assert_sign_rhs_are_None(lhs, sign, rhs):
            if sign is not None or rhs is not None:
                msg = f"Passing arguments `sign` and `rhs` together with a {type(lhs)} is ambiguous."
                raise ValueError(msg)

        def assert_sign_rhs_not_None(lhs, sign, rhs):
            if sign is None or rhs is None:
                msg = f"Arguments `sign` and `rhs` cannot be None when passing along with a {type(lhs)}."
                raise ValueError(msg)

        if name in self.constraints:
            raise ValueError(f"Constraint '{name}' already assigned to model")
        elif name is None:
            name = f"con{self._connameCounter}"
            self._connameCounter += 1
        if sign is not None:
            sign = maybe_replace_signs(as_dataarray(sign))
        if rhs is not None:
            rhs = as_dataarray(rhs)

        if isinstance(lhs, LinearExpression):
            assert_sign_rhs_not_None(lhs, sign, rhs)
            data = lhs.data.assign(sign=sign, rhs=rhs)
        elif callable(lhs):
            assert coords is not None, "`coords` must be given when lhs is a function"
            rule = lhs
            assert_sign_rhs_are_None(lhs, sign, rhs)
            data = Constraint.from_rule(self, rule, coords).data
        elif isinstance(lhs, AnonymousScalarConstraint):
            assert_sign_rhs_are_None(lhs, sign, rhs)
            data = lhs.to_constraint().data
        elif isinstance(lhs, Constraint):
            assert_sign_rhs_are_None(lhs, sign, rhs)
            data = lhs.data
        elif isinstance(lhs, (list, tuple)):
            assert_sign_rhs_not_None(lhs, sign, rhs)
            data = self.linexpr(*lhs).to_constraint(sign, rhs).data
        elif isinstance(lhs, (Variable, ScalarVariable, ScalarLinearExpression)):
            assert_sign_rhs_not_None(lhs, sign, rhs)
            data = lhs.to_linexpr().to_constraint(sign, rhs).data
        else:
            raise ValueError(
                f"Invalid type of `lhs` ({type(lhs)}) or invalid combination of `lhs`, `sign` and `rhs`."
            )

        if mask is not None:
            mask = as_dataarray(mask).astype(bool)
            # TODO: simplify
            assert set(mask.dims).issubset(
                data.dims
            ), "Dimensions of mask not a subset of resulting labels dimensions."

        labels = DataArray(-1, coords=data.indexes)

        self.check_force_dim_names(labels)

        start = self._cCounter
        end = start + labels.size
        labels.data = np.arange(start, end).reshape(labels.shape)
        self._cCounter += labels.size

        if mask is not None:
            labels = labels.where(mask, -1)

        data = data.assign(labels=labels).assign_attrs(
            label_range=(start, end), name=name
        )

        if self.chunk:
            data = data.chunk(self.chunk)

        constraint = Constraint(data, name=name, model=self)
        self.constraints.add(constraint)
        return constraint

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

        if not isinstance(expr, (LinearExpression, QuadraticExpression)):
            raise ValueError(
                f"Invalid type of `expr` ({type(expr)})."
                " Must be a LinearExpression or QuadraticExpression."
            )

        if self.chunk is not None:
            expr = expr.chunk(self.chunk)

        if len(expr.coord_dims):
            expr = expr.sum()

        if expr.const != 0:
            raise ValueError("Constant values in objective function not supported.")

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
        labels = self.variables[name].labels
        self.variables.remove(name)

        for k in self.constraints:
            vars = self.constraints[k].data["vars"]
            vars = vars.where(~vars.isin(labels), -1)
            self.constraints[k].data["vars"] = vars

        self.objective = self.objective.sel(
            {TERM_DIM: ~self.objective.vars.isin(labels)}
        )

    def remove_constraints(self, name):
        """
        Remove all constraints stored under reference name `name` from the
        model.

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
    def continuous(self):
        """
        Get all continuous variables.
        """
        return self.variables.continuous

    @property
    def binaries(self):
        """
        Get all binary variables.
        """
        return self.variables.binaries

    @property
    def integers(self):
        """
        Get all integer variables.
        """
        return self.variables.integers

    @property
    def is_linear(self):
        return type(self.objective) is LinearExpression

    @property
    def is_quadratic(self):
        return type(self.objective) is QuadraticExpression

    @property
    def type(self):
        if (len(self.binaries) or len(self.integers)) and len(self.continuous):
            variable_type = "MI"
        elif len(self.binaries) or len(self.integers):
            variable_type = "I"
        else:
            variable_type = ""

        objective_type = "Q" if self.is_quadratic else "L"

        return f"{variable_type}{objective_type}P"

    @property
    def nvars(self):
        """
        Get the total number of variables.

        This excludes all variables which are not active.
        """
        return self.variables.nvars

    @property
    def ncons(self):
        """
        Get the total number of constraints.

        This excludes all constraints which are not active.
        """
        return self.constraints.ncons

    @property
    def shape(self):
        """
        Get the shape of the non-filtered constraint matrix.

        This includes all constraints and variables which are not active.
        """
        return (self._cCounter, self._xCounter)

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
        self.variables.set_blocks(self.blocks)
        block_map = self.variables.get_blockmap(dtype)
        self.constraints.set_blocks(block_map)

        blocks = replace_by_map(self.objective.vars, block_map)
        self.objective = self.objective.assign(blocks=blocks)

    def linexpr(self, *args):
        """
        Create a linopy.LinearExpression from argument list.

        Parameters
        ----------
        args : tuples of (coefficients, variables) or tuples of
               coordinates and a function
            If args is a collection of coefficients-variables-tuples, the resulting
            linear expression is built with the function LinearExpression.from_tuples.
            In this case, each tuple represents on term in the linear expression,
            which can span over multiple dimensions:

            * coefficients : int/float/array_like
                The coefficient(s) in the term, if the coefficients array
                contains dimensions which do not appear in
                the variables, the variables are broadcasted.
            * variables : str/array_like/linopy.Variable
                The variable(s) going into the term. These may be referenced
                by name.

            If args is a collection of coordinates with an appended function at the
            end, the function LinearExpression.from_rule is used to build the linear
            expression. Then, the argument are expected to contain:

            * rule : callable
                Function to be called for each combinations in `coords`.
                The first argument of the function is the underlying `linopy.Model`.
                The following arguments are given by the coordinates for accessing
                the variables. The function has to return a
                `ScalarLinearExpression`. Therefore use the direct getter when
                indexing variables.
            * coords : coordinate-like
                Coordinates to be processed by `xarray.DataArray`. For each
                combination of coordinates, the function `rule` is called.
                The order and size of coords has to be same as the argument list
                followed by `model` in function `rule`.


        Returns
        -------
        linopy.LinearExpression

        Examples
        --------

        For creating an expression from tuples:
        >>> from linopy import Model
        >>> import pandas as pd
        >>> m = Model()
        >>> x = m.add_variables(pd.Series([0, 0]), 1, name="x")
        >>> y = m.add_variables(4, pd.Series([8, 10]), name="y")
        >>> expr = m.linexpr((10, "x"), (1, "y"))

        For creating an expression from a rule:
        >>> m = Model()
        >>> coords = pd.RangeIndex(10), ["a", "b"]
        >>> a = m.add_variables(coords=coords)
        >>> def rule(m, i, j):
        ...     return a[i, j] + a[(i + 1) % 10, j]
        ...
        >>> expr = m.linexpr(rule, coords)

        See also
        --------
        LinearExpression.from_tuples, LinearExpression.from_rule
        """
        if callable(args[0]):
            assert len(args) == 2, (
                "When first argument is a function, only one second argument "
                "containing a tuple or a single set of coords must be given."
            )
            rule, coords = args
            return LinearExpression.from_rule(self, rule, coords)
        if not isinstance(args, tuple):
            raise TypeError(f"Not supported type {args}.")
        tuples = [
            (c, self.variables[v]) if isinstance(v, str) else (c, v) for (c, v) in args
        ]
        return LinearExpression.from_tuples(*tuples, chunk=self.chunk)

    @property
    def coefficientrange(self):
        """
        Coefficient range of the constraints in the model.
        """
        return self.constraints.coefficientrange

    @property
    def objectiverange(self):
        """
        Objective range of the objective in the model.
        """
        return pd.Series(
            [self.objective.coeffs.min().item(), self.objective.coeffs.max().item()],
            index=["min", "max"],
        )

    def get_solution_file(self, solution_fn=None):
        """
        Get a fresh created solution file if solution file is None.
        """
        if solution_fn is not None:
            return solution_fn

        kwargs = dict(
            prefix="linopy-solve-",
            suffix=".sol",
            mode="w",
            dir=self.solver_dir,
            delete=False,
        )
        with NamedTemporaryFile(**kwargs) as f:
            return f.name

    def get_problem_file(self, problem_fn=None, io_api=None):
        """
        Get a fresh created problem file if problem file is None.
        """
        if problem_fn is not None:
            return problem_fn

        suffix = ".mps" if io_api == "mps" else ".lp"
        kwargs = dict(
            prefix="linopy-problem-",
            suffix=suffix,
            mode="w",
            dir=self.solver_dir,
            delete=False,
        )
        with NamedTemporaryFile(**kwargs) as f:
            return f.name

    def solve(
        self,
        solver_name=None,
        io_api=None,
        problem_fn=None,
        solution_fn=None,
        log_fn=None,
        basis_fn=None,
        warmstart_fn=None,
        keep_files=False,
        sanitize_zeros=True,
        remote=None,
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
            Default to the first entry in `linopy.available_solvers`.
        io_api : str, optional
            Api to use for communicating with the solver, must be one of
            {'lp', 'mps', 'direct'}. If set to 'lp'/'mps' the problem is written to an
            LP/MPS file which is then read by the solver. If set to
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
        sanitize_zeros : bool, optional
            Whether to set terms with zero coeffficient as missing.
            This will remove unneeded overhead in the lp file writing.
            The default is True.
        remote : linopy.remote.RemoteHandler
            Remote handler to use for solving model on a server. Note that when
            solving on a rSee
            linopy.remote.RemoteHandler for more details.
        **solver_options : kwargs
            Options passed to the solver.

        Returns
        -------
        linopy.Model
            Optimized model.
        """
        if remote:
            solved = remote.solve_on_remote(
                self,
                solver_name=solver_name,
                io_api=io_api,
                problem_fn=problem_fn,
                solution_fn=solution_fn,
                log_fn=log_fn,
                basis_fn=basis_fn,
                warmstart_fn=warmstart_fn,
                keep_files=keep_files,
                sanitize_zeros=sanitize_zeros,
                **solver_options,
            )

            self.objective_value = solved.objective_value
            self.status = solved.status
            self.termination_condition = solved.termination_condition
            for k, v in self.variables.items():
                v.solution = solved.variables[k].solution
            for k, c in self.constraints.items():
                if "dual" in solved.constraints[k]:
                    c.dual = solved.constraints[k].dual
            return self.status, self.termination_condition

        if len(available_solvers) == 0:
            raise RuntimeError("No solver installed.")

        if solver_name is None:
            solver_name = available_solvers[0]

        logger.info(f" Solve problem using {solver_name.title()} solver")
        assert solver_name in available_solvers, f"Solver {solver_name} not installed"

        # reset result
        self.reset_solution()

        if log_fn is not None:
            logger.info(f"Solver logs written to `{log_fn}`.")

        if solver_options:
            options_string = "\n".join(
                f" - {k}: {v}" for k, v in solver_options.items()
            )
            logger.info(f"Solver options:\n{options_string}")

        problem_fn = self.get_problem_file(problem_fn, io_api=io_api)
        solution_fn = self.get_solution_file(solution_fn)

        if sanitize_zeros:
            self.constraints.sanitize_zeros()

        if self.is_quadratic and solver_name not in quadratic_solvers:
            raise ValueError(
                f"Solver {solver_name} does not support quadratic problems."
            )

        try:
            func = getattr(solvers, f"run_{solver_name}")
            result = func(
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
                if fn is not None and (os.path.exists(fn) and not keep_files):
                    os.remove(fn)

        result.info()

        self.objective_value = result.solution.objective
        self.status = result.status.status.value
        self.termination_condition = result.status.termination_condition.value
        self.solver_model = result.solver_model

        if not result.status.is_ok:
            return result.status.status.value, result.status.termination_condition.value

        # map solution and dual to original shape which includes missing values
        sol = result.solution.primal.copy()
        sol.loc[-1] = nan

        for name, var in self.variables.items():
            idx = np.ravel(var.labels)
            try:
                vals = sol[idx].values.reshape(var.labels.shape)
            except KeyError:
                vals = sol.reindex(idx).values.reshape(var.labels.shape)
            var.solution = xr.DataArray(vals, var.coords)

        if not result.solution.dual.empty:
            dual = result.solution.dual.copy()
            dual.loc[-1] = nan

            for name, con in self.constraints.items():
                idx = np.ravel(con.labels)
                try:
                    vals = dual[idx].values.reshape(con.labels.shape)
                except KeyError:
                    vals = dual.reindex(idx).values.reshape(con.labels.shape)
                con.dual = xr.DataArray(vals, con.labels.coords)

        return result.status.status.value, result.status.termination_condition.value

    def compute_infeasibilities(self):
        """
        Compute a set of infeasible constraints.

        This function requires that the model was solved with `gurobi` and the
        termination condition was infeasible.

        Returns
        -------
        labels : list
            Labels of the infeasible constraints.
        """
        if "gurobi" not in available_solvers:
            raise ImportError("Gurobi is required for this method.")

        import gurobipy

        solver_model = getattr(self, "solver_model")

        if not isinstance(solver_model, gurobipy.Model):
            raise NotImplementedError("Solver model must be a Gurobi Model.")

        solver_model.computeIIS()
        f = NamedTemporaryFile(suffix=".ilp", prefix="linopy-iis-", delete=False)
        solver_model.write(f.name)
        labels = []
        for line in f.readlines():
            line = line.decode()
            if line.startswith(" c"):
                labels.append(int(line.split(":")[0][2:]))
        return labels

    def print_infeasibilities(self, display_max_terms=None):
        """
        Print a list of infeasible constraints.

        This function requires that the model was solved with `gurobi` and the
        termination condition was infeasible.
        """
        labels = self.compute_infeasibilities()
        self.constraints.print_labels(labels, display_max_terms=display_max_terms)

    @deprecated(
        details="Use `compute_infeasibilities`/`print_infeasibilities` instead."
    )
    def compute_set_of_infeasible_constraints(self):
        """
        Compute a set of infeasible constraints.

        This function requires that the model was solved with `gurobi` and the
        termination condition was infeasible.

        Returns
        -------
        labels : xr.DataArray
            Labels of the infeasible constraints. Labels with value -1 are not in the set.
        """
        labels = self.compute_infeasibilities()
        cons = self.constraints.labels.isin(np.array(labels))
        subset = self.constraints.labels.where(cons, -1)
        subset = subset.drop_vars(
            [k for (k, v) in (subset == -1).all().items() if v.item()]
        )
        return subset

    def reset_solution(self):
        """
        Reset the solution and dual values if available of the model.
        """
        self.variables.reset_solution()
        self.constraints.reset_dual()

    to_netcdf = to_netcdf

    to_file = to_file

    to_gurobipy = to_gurobipy

    to_highspy = to_highspy

    to_block_files = to_block_files
