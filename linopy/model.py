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
from numpy import inf, nan
from xarray import DataArray, Dataset

from linopy import solvers
from linopy.common import best_int, maybe_replace_signs, replace_by_map
from linopy.constants import ModelStatus, TerminationCondition
from linopy.constraints import (
    AnonymousConstraint,
    AnonymousScalarConstraint,
    Constraints,
)
from linopy.expressions import LinearExpression, ScalarLinearExpression
from linopy.io import to_block_files, to_file, to_gurobipy, to_highspy, to_netcdf
from linopy.matrices import MatrixAccessor
from linopy.solvers import available_solvers
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

        self._solution = Dataset()
        self._dual = Dataset()
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
        return self._solution

    @solution.setter
    def solution(self, value):
        self._solution = Dataset(value)

    @property
    def dual(self):
        """
        Dual values calculated by the optimization.
        """
        return self._dual

    @dual.setter
    def dual(self, value):
        self._dual = Dataset(value)

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
        """
        Return a string representation of the linopy model.
        """
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
            name = "var" + str(self._varnameCounter)
            self._varnameCounter += 1

        if name in self.variables:
            raise ValueError(f"Variable '{name}' already assigned to model")

        if binary and integer:
            raise ValueError("Variable cannot be both binary and integer.")

        if not binary:
            lower = DataArray(lower, coords=coords, **kwargs)
            upper = DataArray(upper, coords=coords, **kwargs)
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
            # check if lower and upper and non-defaults
            if (lower != -inf) or (upper != inf):
                raise ValueError(
                    "Argument lower and upper not valid for binary variables."
                )
            # for general compatibility when ravelling all values set non-nan
            lower = DataArray(-inf, coords=coords, **kwargs)
            upper = DataArray(inf, coords=coords, **kwargs)

        labels = DataArray(-2, coords=coords).assign_attrs(
            binary=binary, integer=integer
        )

        # ensure order of dims is the same
        lower = lower.transpose(*[d for d in labels.dims if d in lower.dims])
        upper = upper.transpose(*[d for d in labels.dims if d in upper.dims])

        if mask is not None:
            mask = DataArray(mask).astype(bool)
            mask, _ = xr.align(mask, labels, join="right")
            assert set(mask.dims).issubset(
                labels.dims
            ), "Dimensions of mask not a subset of resulting labels dimensions."

        # It is important to end up with monotonically increasing labels in the
        # model's variables container as we use it for indirect indexing.
        labels_reindexed = labels.reindex_like(self.variables.labels, fill_value=-1)
        if not labels.equals(labels_reindexed):
            warnings.warn(
                f"Reindexing variable {name} to match existing coordinates.",
                UserWarning,
            )
            labels = labels_reindexed
            lower = lower.reindex_like(labels)
            upper = upper.reindex_like(labels)
            if mask is None:
                mask = labels != -1
            else:
                mask = mask.reindex_like(labels_reindexed, fill_value=False)

        self.check_force_dim_names(labels)

        start = self._xCounter
        end = start + labels.size
        labels.data = np.arange(start, end).reshape(labels.shape)
        self._xCounter += labels.size

        if mask is not None:
            labels = labels.where(mask, -1)

        if self.chunk:
            labels = labels.chunk(self.chunk)
            lower = lower.chunk(self.chunk)
            upper = upper.chunk(self.chunk)

        self.variables.add(name, labels, lower, upper, start=start, end=end)
        return self.variables[name]

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
        lhs : linopy.LinearExpression/linopy.AnonymousConstraint/callable
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
        labels = None

        if name is None:
            name = "con" + str(self._connameCounter)
            self._connameCounter += 1

        if name in self.constraints:
            raise ValueError(f"Constraint '{name}' already assigned to model")

        if callable(lhs):
            assert coords is not None, "`coords` must be given when lhs is a function"
            rule = lhs
            lhs = AnonymousConstraint.from_rule(self, rule, coords)

        if isinstance(lhs, AnonymousScalarConstraint):
            lhs = lhs.to_anonymous_constraint()

        if isinstance(lhs, AnonymousConstraint):
            if sign is not None or rhs is not None:
                raise ValueError(
                    "Passing arguments `sign` and `rhs` together with a constraint"
                    " is ambiguous."
                )
            labels = lhs.labels  # returns an empty data array of correct shape
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
        elif isinstance(lhs, (Variable, ScalarVariable, ScalarLinearExpression)):
            lhs = lhs.to_linexpr()
        assert isinstance(lhs, LinearExpression)

        if isinstance(rhs, (Variable, LinearExpression)):
            raise TypeError(f"Assigned rhs must be a constant, got {type(rhs)}).")

        lhs = lhs.sanitize()
        sign = maybe_replace_signs(DataArray(sign))
        rhs = DataArray(rhs)

        if labels is None:
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

        lhs = lhs.data.rename({"_term": f"{name}_term"})

        if self.chunk:
            lhs = lhs.chunk(self.chunk)
            sign = sign.chunk(self.chunk)
            rhs = rhs.chunk(self.chunk)
            labels = labels.chunk(self.chunk)

        self.constraints.add(name, labels, lhs.coeffs, lhs.vars, sign, rhs)

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
    def non_binaries(self):
        """
        Get all non-binary variables.
        """
        return self.variables.non_binaries

    @property
    def nvars(self):
        """
        Get the total number of variables.
        """
        return self.variables.nvars

    @property
    def ncons(self):
        """
        Get the total number of constraints.
        """
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
        if isinstance(args, tuple):
            tuples = [
                (c, self.variables[v]) if isinstance(v, str) else (c, v)
                for (c, v) in args
            ]
            return LinearExpression.from_tuples(*tuples, chunk=self.chunk)
        else:
            raise TypeError(f"Not supported type {args}.")

    def vareval(self, expr: str, eval_kw=None, **kwargs):
        raise NotImplementedError(
            "vareval was removed in version `v0.1.0`. Please use m.add_variables() instead."
        )

    def lineval(self, expr: str, eval_kw=None, **kwargs):
        raise NotImplementedError(
            "lineval was removed in version `v0.1.0`. Please use arithmetic expressions instead."
        )

    def coneval(self, expr: str, eval_kw=None, **kwargs):
        raise NotImplementedError(
            "coneval was removed in version `v0.1.0`. Please use arithmetic "
            "expressions and m.add_constraints() instead."
        )

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

    def get_problem_file(self, problem_fn=None, io_api=None):
        """
        Get a fresh created problem file if problem file is None.
        """
        if problem_fn is None:

            if io_api == "mps":
                suffix = ".mps"
            else:
                suffix = ".lp"

            kwargs = dict(
                prefix="linopy-problem-",
                suffix=suffix,
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
            The default is 'gurobi'.
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
            self.solution = solved.solution
            self.dual = solved.dual
            return self.status, self.termination_condition

        logger.info(f" Solve linear problem using {solver_name.title()} solver")
        assert solver_name in available_solvers, f"Solver {solver_name} not installed"

        # reset result
        self.solution = xr.Dataset()
        self.dual = xr.Dataset()

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
                if fn is not None:
                    if os.path.exists(fn) and not keep_files:
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

        for name, labels in self.variables.labels.items():
            idx = np.ravel(labels)
            vals = sol[idx].values.reshape(labels.shape)
            self.solution[name] = xr.DataArray(vals, labels.coords)

        if not result.solution.dual.empty:
            dual = result.solution.dual.copy()
            dual.loc[-1] = nan

            for name, labels in self.constraints.labels.items():
                idx = np.ravel(labels)
                try:
                    vals = dual[idx].values.reshape(labels.shape)
                except KeyError:
                    vals = dual.reindex(idx).values.reshape(labels.shape)
                self.dual[name] = xr.DataArray(vals, labels.coords)

        return result.status.status.value, result.status.termination_condition.value

    def compute_set_of_infeasible_constraints(self):
        """
        Compute a set of infeasible constraints.

        This method requires gurobipy to be running.

        Returns
        -------
        labels : xr.DataArray
            Labels of the infeasible constraints. Labels with value -1 are not in the set.
        """
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
        cons = self.constraints.labels.isin(np.array(labels))
        subset = self.constraints.labels.where(cons, -1)
        subset = subset.drop_vars(
            [k for (k, v) in (subset == -1).all().items() if v.item()]
        )
        return subset

    to_netcdf = to_netcdf

    to_file = to_file

    to_gurobipy = to_gurobipy

    to_highspy = to_highspy

    to_block_files = to_block_files
