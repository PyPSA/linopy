"""
Linopy model module.

This module contains frontend implementations of the package.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir
from typing import Any, overload

import numpy as np
import pandas as pd
import xarray as xr
from deprecation import deprecated
from numpy import inf, nan, ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from xarray import DataArray, Dataset
from xarray.core.types import T_Chunks

from linopy import solvers
from linopy.common import (
    as_dataarray,
    assign_multiindex_safe,
    best_int,
    maybe_replace_signs,
    replace_by_map,
    set_int_index,
    to_path,
)
from linopy.constants import (
    GREATER_EQUAL,
    HELPER_DIMS,
    LESS_EQUAL,
    TERM_DIM,
    ModelStatus,
    TerminationCondition,
)
from linopy.constraints import AnonymousScalarConstraint, Constraint, Constraints
from linopy.expressions import (
    LinearExpression,
    QuadraticExpression,
    ScalarLinearExpression,
)
from linopy.io import (
    to_block_files,
    to_file,
    to_gurobipy,
    to_highspy,
    to_mosek,
    to_netcdf,
)
from linopy.matrices import MatrixAccessor
from linopy.objective import Objective
from linopy.remote import OetcHandler, RemoteHandler
from linopy.solvers import (
    IO_APIS,
    NO_SOLUTION_FILE_SOLVERS,
    available_solvers,
    quadratic_solvers,
)
from linopy.types import (
    ConstantLike,
    ConstraintLike,
    ExpressionLike,
    MaskLike,
    SignLike,
    VariableLike,
)
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

    solver_model: Any
    solver_name: str
    _variables: Variables
    _constraints: Constraints
    _objective: Objective
    _parameters: Dataset
    _solution: Dataset
    _dual: Dataset
    _status: str
    _termination_condition: str
    _xCounter: int
    _cCounter: int
    _varnameCounter: int
    _connameCounter: int
    _blocks: DataArray | None
    _chunk: T_Chunks
    _force_dim_names: bool
    _solver_dir: Path
    matrices: MatrixAccessor

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
        # TODO: check if these should not be mutable
        "_chunk",
        "_force_dim_names",
        "_solver_dir",
        "solver_model",
        "solver_name",
        "matrices",
    )

    def __init__(
        self,
        solver_dir: str | None = None,
        chunk: T_Chunks = None,
        force_dim_names: bool = False,
    ) -> None:
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
        self._variables: Variables = Variables({}, model=self)
        self._constraints: Constraints = Constraints({}, model=self)
        self._objective: Objective = Objective(LinearExpression(None, self), self)
        self._parameters: Dataset = Dataset()

        self._status: str = "initialized"
        self._termination_condition: str = ""
        self._xCounter: int = 0
        self._cCounter: int = 0
        self._varnameCounter: int = 0
        self._connameCounter: int = 0
        self._blocks: DataArray | None = None

        self._chunk: T_Chunks = chunk
        self._force_dim_names: bool = bool(force_dim_names)
        self._solver_dir: Path = Path(
            gettempdir() if solver_dir is None else solver_dir
        )

        self.matrices: MatrixAccessor = MatrixAccessor(self)

    @property
    def variables(self) -> Variables:
        """
        Variables assigned to the model.
        """
        return self._variables

    @property
    def constraints(self) -> Constraints:
        """
        Constraints assigned to the model.
        """
        return self._constraints

    @property
    def objective(self) -> Objective:
        """
        Objective assigned to the model.
        """
        return self._objective

    @objective.setter
    def objective(
        self, obj: Objective | LinearExpression | QuadraticExpression
    ) -> Objective:
        if not isinstance(obj, Objective):
            obj = Objective(obj, self)

        self._objective = obj
        return self._objective

    @property
    def sense(self) -> str:
        """
        Sense of the objective function.
        """
        return self.objective.sense

    @sense.setter
    def sense(self, value: str) -> None:
        self.objective.sense = value

    @property
    def parameters(self) -> Dataset:
        """
        Parameters assigned to the model.

        The parameters serve as an extra field where additional data may
        be stored.
        """
        return self._parameters

    @parameters.setter
    def parameters(self, value: Dataset | Mapping) -> None:
        self._parameters = Dataset(value)

    @property
    def solution(self) -> Dataset:
        """
        Solution calculated by the optimization.
        """
        return self.variables.solution

    @property
    def dual(self) -> Dataset:
        """
        Dual values calculated by the optimization.
        """
        return self.constraints.dual

    @property
    def status(self) -> str:
        """
        Status of the model.
        """
        return self._status

    @status.setter
    def status(self, value: str) -> None:
        self._status = ModelStatus[value].value

    @property
    def termination_condition(self) -> str:
        """
        Termination condition of the model.
        """
        return self._termination_condition

    @termination_condition.setter
    def termination_condition(self, value: str) -> None:
        # TODO: remove if-clause, only kept for backward compatibility
        if value:
            self._termination_condition = TerminationCondition[value].value
        else:
            self._termination_condition = value

    @property
    def chunk(self) -> T_Chunks:
        """
        Chunk sizes of the model.
        """
        return self._chunk

    @chunk.setter
    def chunk(self, value: T_Chunks) -> None:
        self._chunk = value

    @property
    def force_dim_names(self) -> bool:
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
    def force_dim_names(self, value: bool) -> None:
        self._force_dim_names = bool(value)

    @property
    def solver_dir(self) -> Path:
        """
        Solver directory of the model.
        """
        return self._solver_dir

    @solver_dir.setter
    def solver_dir(self, value: str | Path) -> None:
        if not isinstance(value, str | Path):
            raise TypeError("'solver_dir' must path-like.")
        self._solver_dir = Path(value)

    @property
    def dataset_attrs(self) -> list[str]:
        return ["parameters"]

    @property
    def scalar_attrs(self) -> list[str]:
        return [
            "status",
            "termination_condition",
            "_xCounter",
            "_cCounter",
            "_varnameCounter",
            "_connameCounter",
            "force_dim_names",
        ]

    def __repr__(self) -> str:
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

    def __getitem__(self, key: str) -> Variable:
        """
        Get a model variable by the name.
        """
        return self.variables[key]

    def check_force_dim_names(self, ds: DataArray | Dataset) -> None:
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
            bool(re.match(r"dim_[0-9]+", str(dim))) for dim in list(ds.dims)
        )
        if self.force_dim_names and contains_default_dims:
            raise ValueError(
                "Added data contains non-customized dimension names. This is not "
                "allowed when setting `force_dim_names` to True."
            )
        else:
            return

    def _check_valid_dim_names(self, ds: DataArray | Dataset) -> None:
        """
        Ensure that the added data does not lead to a naming conflict.

        Parameters
        ----------
        model : linopy.Model
        ds : xr.DataArray/Variable/LinearExpression
            Data that should be added to the model.

        Raises
        ------
        ValueError
            If broadcasted data leads to unsupported dimension names.

        Returns
        -------
        None.
        """
        unsupported_dim_names = ["labels", "coeffs", "vars", "sign", "rhs"]
        if any(dim in unsupported_dim_names for dim in ds.dims):
            raise ValueError(
                "Added data contains unsupported dimension names. "
                "Dimensions cannot be named 'labels', 'coeffs', 'vars', 'sign' or 'rhs'."
            )

    def add_variables(
        self,
        lower: Any = -inf,
        upper: Any = inf,
        coords: Sequence[Sequence | pd.Index | DataArray] | Mapping | None = None,
        name: str | None = None,
        mask: DataArray | ndarray | Series | None = None,
        binary: bool = False,
        integer: bool = False,
        **kwargs: Any,
    ) -> Variable:
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
                "labels": -1,
            }
        )
        (data,) = xr.broadcast(data)
        self.check_force_dim_names(data)
        self._check_valid_dim_names(data)

        if mask is not None:
            mask = as_dataarray(mask, coords=data.coords, dims=data.dims).astype(bool)

        start = self._xCounter
        end = start + data.labels.size
        data.labels.values = np.arange(start, end).reshape(data.labels.shape)
        self._xCounter += data.labels.size

        if mask is not None:
            data.labels.values = data.labels.where(mask, -1).values

        data = data.assign_attrs(
            label_range=(start, end), name=name, binary=binary, integer=integer
        )

        if self.chunk:
            data = data.chunk(self.chunk)

        variable = Variable(data, name=name, model=self, skip_broadcast=True)
        self.variables.add(variable)
        return variable

    def add_constraints(
        self,
        lhs: VariableLike
        | ExpressionLike
        | ConstraintLike
        | Sequence[tuple[ConstantLike, VariableLike | str]]
        | Callable,
        sign: SignLike | None = None,
        rhs: ConstantLike | VariableLike | ExpressionLike | None = None,
        name: str | None = None,
        coords: Sequence[Sequence | pd.Index | DataArray] | Mapping | None = None,
        mask: MaskLike | None = None,
    ) -> Constraint:
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
            Boolean mask with False values for constraints which are skipped.
            The shape of the mask has to match the shape the added constraints.
            Default is None.


        Returns
        -------
        labels : linopy.model.Constraint
            Array containing the labels of the added constraints.
        """

        msg_sign_rhs_none = f"Arguments `sign` and `rhs` cannot be None when passing along with a {type(lhs)}."
        msg_sign_rhs_not_none = f"Arguments `sign` and `rhs` cannot be None when passing along with a {type(lhs)}."

        if name in list(self.constraints):
            raise ValueError(f"Constraint '{name}' already assigned to model")
        elif name is None:
            name = f"con{self._connameCounter}"
            self._connameCounter += 1
        if sign is not None:
            sign = maybe_replace_signs(as_dataarray(sign))

        if isinstance(lhs, LinearExpression):
            if sign is None or rhs is None:
                raise ValueError(msg_sign_rhs_not_none)
            data = lhs.to_constraint(sign, rhs).data
        elif isinstance(lhs, list | tuple):
            if sign is None or rhs is None:
                raise ValueError(msg_sign_rhs_none)
            data = self.linexpr(*lhs).to_constraint(sign, rhs).data
        # directly convert first argument to a constraint
        elif callable(lhs):
            assert coords is not None, "`coords` must be given when lhs is a function"
            rule = lhs
            if sign is not None or rhs is not None:
                raise ValueError(msg_sign_rhs_none)
            data = Constraint.from_rule(self, rule, coords).data
        elif isinstance(lhs, AnonymousScalarConstraint):
            if sign is not None or rhs is not None:
                raise ValueError(msg_sign_rhs_none)
            data = lhs.to_constraint().data
        elif isinstance(lhs, Constraint):
            if sign is not None or rhs is not None:
                raise ValueError(msg_sign_rhs_none)
            data = lhs.data
        elif isinstance(lhs, Variable | ScalarVariable | ScalarLinearExpression):
            if sign is None or rhs is None:
                raise ValueError(msg_sign_rhs_not_none)
            data = lhs.to_linexpr().to_constraint(sign, rhs).data
        else:
            raise ValueError(
                f"Invalid type of `lhs` ({type(lhs)}) or invalid combination of `lhs`, `sign` and `rhs`."
            )

        invalid_infinity_values = (
            (data.sign == LESS_EQUAL) & (data.rhs == -np.inf)
        ) | ((data.sign == GREATER_EQUAL) & (data.rhs == np.inf))  # noqa: F821
        if invalid_infinity_values.any():
            raise ValueError(f"Constraint {name} contains incorrect infinite values.")

        # ensure helper dimensions are not set as coordinates
        if drop_dims := set(HELPER_DIMS).intersection(data.coords):
            # TODO: add a warning here, routines should be safe against this
            data = data.drop_vars(drop_dims)

        data["labels"] = -1
        (data,) = xr.broadcast(data, exclude=[TERM_DIM])

        if mask is not None:
            mask = as_dataarray(mask).astype(bool)
            # TODO: simplify
            assert set(mask.dims).issubset(data.dims), (
                "Dimensions of mask not a subset of resulting labels dimensions."
            )

        self.check_force_dim_names(data)

        start = self._cCounter
        end = start + data.labels.size
        data.labels.values = np.arange(start, end).reshape(data.labels.shape)
        self._cCounter += data.labels.size

        if mask is not None:
            data.labels.values = data.labels.where(mask, -1).values

        data = data.assign_attrs(label_range=(start, end), name=name)

        if self.chunk:
            data = data.chunk(self.chunk)

        constraint = Constraint(data, name=name, model=self, skip_broadcast=True)
        self.constraints.add(constraint)
        return constraint

    def add_objective(
        self,
        expr: Variable
        | LinearExpression
        | QuadraticExpression
        | Sequence[tuple[ConstantLike, VariableLike]],
        overwrite: bool = False,
        sense: str = "min",
    ) -> None:
        """
        Add an objective function to the model.

        Parameters
        ----------
        expr : linopy.LinearExpression, linopy.QuadraticExpression
            Expression describing the objective function.
        overwrite : False, optional
            Whether to overwrite the existing objective. The default is False.

        Returns
        -------
        linopy.LinearExpression
            The objective function assigned to the model.
        """
        if not overwrite:
            assert self.objective.expression.empty, (
                "Objective already defined."
                " Set `overwrite` to True to force overwriting."
            )
        if isinstance(expr, Variable):
            expr = 1 * expr
        self.objective.expression = expr
        self.objective.sense = sense

    def remove_variables(self, name: str) -> None:
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

        for k in list(self.constraints):
            vars = self.constraints[k].data["vars"]
            vars = vars.where(~vars.isin(labels), -1)
            self.constraints[k]._data = assign_multiindex_safe(
                self.constraints[k].data, vars=vars
            )

        self.objective = self.objective.sel(
            {TERM_DIM: ~self.objective.vars.isin(labels)}
        )

    def remove_constraints(self, name: str | list[str]) -> None:
        """
        Remove all constraints stored under reference name 'name' from the
        model.

        Parameters
        ----------
        name : str or list of str
            Reference name(s) of the constraints to remove. If a single name is
            provided, only that constraint will be removed. If a list of names
            is provided, all constraints with those names will be removed.

        Returns
        -------
        None.
        """
        if isinstance(name, list):
            for n in name:
                logger.debug(f"Removed constraint: {n}")
                self.constraints.remove(n)
        else:
            logger.debug(f"Removed constraint: {name}")
            self.constraints.remove(name)

    def remove_objective(self) -> None:
        """
        Remove the objective's linear expression from the model.

        Returns
        -------
        None.
        """
        self.objective = Objective(LinearExpression(None, self), self)

    @property
    def continuous(self) -> Variables:
        """
        Get all continuous variables.
        """
        return self.variables.continuous

    @property
    def binaries(self) -> Variables:
        """
        Get all binary variables.
        """
        return self.variables.binaries

    @property
    def integers(self) -> Variables:
        """
        Get all integer variables.
        """
        return self.variables.integers

    @property
    def is_linear(self) -> bool:
        return self.objective.is_linear

    @property
    def is_quadratic(self) -> bool:
        return self.objective.is_quadratic

    @property
    def type(self) -> str:
        if (len(self.binaries) or len(self.integers)) and len(self.continuous):
            variable_type = "MI"
        elif len(self.binaries) or len(self.integers):
            variable_type = "I"
        else:
            variable_type = ""

        objective_type = "Q" if self.is_quadratic else "L"

        return f"{variable_type}{objective_type}P"

    @property
    def nvars(self) -> int:
        """
        Get the total number of variables.

        This excludes all variables which are not active.
        """
        return self.variables.nvars

    @property
    def ncons(self) -> int:
        """
        Get the total number of constraints.

        This excludes all constraints which are not active.
        """
        return self.constraints.ncons

    @property
    def shape(self) -> tuple[int, int]:
        """
        Get the shape of the non-filtered constraint matrix.

        This includes all constraints and variables which are not active.
        """
        return (self._cCounter, self._xCounter)

    @property
    def blocks(self) -> DataArray | None:
        """
        Blocks used as a basis to split the variables and constraint matrix.
        """
        return self._blocks

    @blocks.setter
    def blocks(self, blocks: DataArray) -> None:
        if not isinstance(blocks, DataArray):
            raise TypeError("Blocks must be of type DataArray")
        assert len(blocks.dims) == 1

        dtype = best_int(int(blocks.max()) + 1)
        blocks = blocks.astype(dtype)

        if self.chunk is not None:
            blocks = blocks.chunk(self.chunk)

        self._blocks = blocks

    def calculate_block_maps(self) -> None:
        """
        Calculate the matrix block mappings based on dimensional blocks.
        """
        assert self.blocks is not None, "Blocks are not defined."

        dtype = self.blocks.dtype.type
        self.variables.set_blocks(self.blocks)
        block_map = self.variables.get_blockmap(dtype)
        self.constraints.set_blocks(block_map)

        blocks = replace_by_map(self.objective.vars, block_map)
        self.objective = self.objective.assign(blocks=blocks)

    @overload
    def linexpr(
        self, *args: Sequence[Sequence | pd.Index | DataArray] | Mapping
    ) -> LinearExpression: ...

    @overload
    def linexpr(
        self, *args: tuple[ConstantLike, str | Variable | ScalarVariable] | ConstantLike
    ) -> LinearExpression: ...

    def linexpr(
        self,
        *args: tuple[ConstantLike, str | Variable | ScalarVariable]
        | ConstantLike
        | Callable
        | Sequence[Sequence | pd.Index | DataArray]
        | Mapping,
    ) -> LinearExpression:
        """
        Create a linopy.LinearExpression from argument list.

        Parameters
        ----------
        args : A mixture of tuples of (coefficients, variables) and constants
            or a function and tuples of coordinates

            If args is a collection of coefficients-variables-tuples and constants, the resulting
            linear expression is built with the function LinearExpression.from_tuples.

            * coefficients : int/float/array_like
                The coefficient(s) in the term, if the coefficients array
                contains dimensions which do not appear in
                the variables, the variables are broadcasted.
            * variables : str/array_like/linopy.Variable
                The variable(s) going into the term. These may be referenced
                by name.
            * constant: int/float/array_like
                The constant value to add to the expression

            If args is a collection of coordinates with an appended function at the
            end, the function LinearExpression.from_rule is used to build the linear
            expression. Then, the argument are expected to contain:

            * rule : callable
                Function to be called for each combinations in `coords`.
                The first argument of the function is the underlying `linopy.Model`.
                The following arguments are given by the coordinates for accessing
                the variables. The function has to return a
                `ScalarLinearExpression`. Therefore, use the direct getter when
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
        ...     return a.at[i, j] + a.at[(i + 1) % 10, j]
        ...
        >>> expr = m.linexpr(rule, coords)

        See Also
        --------
        LinearExpression.from_tuples, LinearExpression.from_rule
        """
        if callable(args[0]):
            assert len(args) == 2, (
                "When first argument is a function, only one second argument "
                "containing a tuple or a single set of coords must be given."
            )
            rule, coords = args
            return LinearExpression.from_rule(self, rule, coords)  # type: ignore
        if not isinstance(args, tuple):
            raise TypeError(f"Not supported type {args}.")

        tuples: list[tuple[ConstantLike, VariableLike] | ConstantLike] = []
        for arg in args:
            if isinstance(arg, tuple):
                c, v = arg
                tuples.append((c, self.variables[v]) if isinstance(v, str) else (c, v))
            else:
                tuples.append(arg)
        return LinearExpression.from_tuples(*tuples, model=self)

    @property
    def coefficientrange(self) -> DataFrame:
        """
        Coefficient range of the constraints in the model.
        """
        return self.constraints.coefficientrange

    @property
    def objectiverange(self) -> Series:
        """
        Objective range of the objective in the model.
        """
        return pd.Series(
            [self.objective.coeffs.min().item(), self.objective.coeffs.max().item()],
            index=["min", "max"],
        )

    def get_solution_file(self) -> Path:
        """
        Get a fresh created solution file if solution file is None.
        """
        with NamedTemporaryFile(
            prefix="linopy-solve-",
            suffix=".sol",
            mode="w",
            dir=str(self.solver_dir),
            delete=False,
        ) as f:
            return Path(f.name)

    def get_problem_file(
        self,
        io_api: str | None = None,
    ) -> Path:
        """
        Get a fresh created problem file if problem file is None.
        """
        suffix = ".mps" if io_api == "mps" else ".lp"
        with NamedTemporaryFile(
            prefix="linopy-problem-",
            suffix=suffix,
            mode="w",
            dir=self.solver_dir,
            delete=False,
        ) as f:
            return Path(f.name)

    def solve(
        self,
        solver_name: str | None = None,
        io_api: str | None = None,
        explicit_coordinate_names: bool = False,
        problem_fn: str | Path | None = None,
        solution_fn: str | Path | None = None,
        log_fn: str | Path | None = None,
        basis_fn: str | Path | None = None,
        warmstart_fn: str | Path | None = None,
        keep_files: bool = False,
        env: Any = None,
        sanitize_zeros: bool = True,
        sanitize_infinities: bool = True,
        slice_size: int = 2_000_000,
        remote: RemoteHandler | OetcHandler = None,  # type: ignore
        progress: bool | None = None,
        **solver_options: Any,
    ) -> tuple[str, str]:
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
        explicit_coordinate_names : bool, optional
            If the Api to use for communicating with the solver is based on 'lp',
            this option allows to keep the variable and constraint names in the
            lp file. This may lead to slower run times.
            The default is set to False.
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
        env : gurobi.Env, optional
            Existing environment passed to the solver (e.g. `gurobipy.Env`).
            Currently only in use for Gurobi. The default is None.
        sanitize_zeros : bool, optional
            Whether to set terms with zero coefficient as missing.
            This will remove unneeded overhead in the lp file writing.
            The default is True.
        sanitize_infinities : bool, optional
            Whether to filter out constraints that are subject to `<= inf` or `>= -inf`.
        slice_size : int, optional
            Size of the slice to use for writing the lp file. The slice size
            is used to split large variables and constraints into smaller
            chunks to avoid memory issues. The default is 2_000_000.
        remote : linopy.remote.RemoteHandler | linopy.oetc.OetcHandler, optional
            Remote handler to use for solving model on a server. Note that when
            solving on a rSee
            linopy.remote.RemoteHandler for more details.
        progress : bool, optional
            Whether to show a progress bar of writing the lp file. The default is
            None, which means that the progress bar is shown if the model has more
            than 10000 variables and constraints.
        **solver_options : kwargs
            Options passed to the solver.

        Returns
        -------
        status : tuple
            Tuple containing the status and termination condition of the
            optimization process.
        """
        # clear cached matrix properties potentially present from previous solve commands
        self.matrices.clean_cached_properties()

        # check io_api
        if io_api is not None and io_api not in IO_APIS:
            raise ValueError(
                f"Keyword argument `io_api` has to be one of {IO_APIS} or None"
            )

        if remote is not None:
            if isinstance(remote, OetcHandler):
                solved = remote.solve_on_oetc(self)
            else:
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

            self.objective.set_value(solved.objective.value)
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

        if problem_fn is None:
            problem_fn = self.get_problem_file(io_api=io_api)
        if solution_fn is None:
            if solver_name in NO_SOLUTION_FILE_SOLVERS and not keep_files:
                # these (solver, keep_files=False) combos do not need a solution file
                solution_fn = None
            else:
                solution_fn = self.get_solution_file()

        if sanitize_zeros:
            self.constraints.sanitize_zeros()

        if sanitize_infinities:
            self.constraints.sanitize_infinities()

        if self.is_quadratic and solver_name not in quadratic_solvers:
            raise ValueError(
                f"Solver {solver_name} does not support quadratic problems."
            )

        try:
            solver_class = getattr(solvers, f"{solvers.SolverName(solver_name).name}")
            # initialize the solver as object of solver subclass <solver_class>
            solver = solver_class(
                **solver_options,
            )
            if io_api == "direct":
                # no problem file written and direct model is set for solver
                result = solver.solve_problem_from_model(
                    model=self,
                    solution_fn=to_path(solution_fn),
                    log_fn=to_path(log_fn),
                    warmstart_fn=to_path(warmstart_fn),
                    basis_fn=to_path(basis_fn),
                    env=env,
                    explicit_coordinate_names=explicit_coordinate_names,
                )
            else:
                if solver_name in ["glpk", "cbc"] and explicit_coordinate_names:
                    logger.warning(
                        f"{solver_name} does not support writing names to lp files, disabling it."
                    )
                    explicit_coordinate_names = False
                problem_fn = self.to_file(
                    to_path(problem_fn),
                    io_api=io_api,
                    explicit_coordinate_names=explicit_coordinate_names,
                    slice_size=slice_size,
                    progress=progress,
                )
                result = solver.solve_problem_from_file(
                    problem_fn=to_path(problem_fn),
                    solution_fn=to_path(solution_fn),
                    log_fn=to_path(log_fn),
                    warmstart_fn=to_path(warmstart_fn),
                    basis_fn=to_path(basis_fn),
                    env=env,
                )

        finally:
            for fn in (problem_fn, solution_fn):
                if fn is not None and (os.path.exists(fn) and not keep_files):
                    os.remove(fn)

        result.info()

        self.objective._value = result.solution.objective
        self.status = result.status.status.value
        self.termination_condition = result.status.termination_condition.value
        self.solver_model = result.solver_model
        self.solver_name = solver_name

        if not result.status.is_ok:
            return result.status.status.value, result.status.termination_condition.value

        # map solution and dual to original shape which includes missing values
        sol = result.solution.primal.copy()
        sol = set_int_index(sol)
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
            dual = set_int_index(dual)
            dual.loc[-1] = nan

            for name, con in self.constraints.items():
                idx = np.ravel(con.labels)
                try:
                    vals = dual[idx].values.reshape(con.labels.shape)
                except KeyError:
                    vals = dual.reindex(idx).values.reshape(con.labels.shape)
                con.dual = xr.DataArray(vals, con.labels.coords)

        return result.status.status.value, result.status.termination_condition.value

    def compute_infeasibilities(self) -> list[int]:
        """
        Compute a set of infeasible constraints.

        This function requires that the model was solved with `gurobi` or `xpress`
        and the termination condition was infeasible. The solver must have detected
        the infeasibility during the solve process.

        Returns
        -------
        labels : list[int]
            Labels of the infeasible constraints.
        """
        solver_model = getattr(self, "solver_model", None)

        # Check for Gurobi
        if "gurobi" in available_solvers:
            try:
                import gurobipy

                if solver_model is not None and isinstance(
                    solver_model, gurobipy.Model
                ):
                    return self._compute_infeasibilities_gurobi(solver_model)
            except ImportError:
                pass

        # Check for Xpress
        if "xpress" in available_solvers:
            try:
                import xpress

                if solver_model is not None and isinstance(
                    solver_model, xpress.problem
                ):
                    return self._compute_infeasibilities_xpress(solver_model)
            except ImportError:
                pass

        # If we get here, either the solver doesn't support IIS or no solver model is available
        if solver_model is None:
            # Check if this is a supported solver without a stored model
            solver_name = getattr(self, "solver_name", "unknown")
            if solver_name in ["gurobi", "xpress"]:
                raise ValueError(
                    "No solver model available. The model must be solved first with "
                    "'gurobi' or 'xpress' solver and the result must be infeasible."
                )
            else:
                # This is an unsupported solver
                raise NotImplementedError(
                    f"Computing infeasibilities is not supported for '{solver_name}' solver. "
                    "Only Gurobi and Xpress solvers support IIS computation."
                )
        else:
            # We have a solver model but it's not a supported type
            raise NotImplementedError(
                "Computing infeasibilities is only supported for Gurobi and Xpress solvers. "
                f"Current solver model type: {type(solver_model).__name__}"
            )

    def _compute_infeasibilities_gurobi(self, solver_model: Any) -> list[int]:
        """Compute infeasibilities for Gurobi solver."""
        solver_model.computeIIS()
        f = NamedTemporaryFile(suffix=".ilp", prefix="linopy-iis-", delete=False)
        solver_model.write(f.name)
        labels = []
        pattern = re.compile(r"^ [^:]+#([0-9]+):")
        for line in f.readlines():
            line_decoded = line.decode()
            try:
                if line_decoded.startswith(" c"):
                    labels.append(int(line_decoded.split(":")[0][2:]))
            except ValueError as _:
                match = pattern.match(line_decoded)
                if match:
                    labels.append(int(match.group(1)))
        f.close()
        return labels

    def _compute_infeasibilities_xpress(self, solver_model: Any) -> list[int]:
        """Compute infeasibilities for Xpress solver."""
        # Compute all IIS
        solver_model.iisall()

        # Get the number of IIS found
        num_iis = solver_model.attributes.numiis
        if num_iis == 0:
            return []

        labels = set()

        # Create constraint mapping for efficient lookups
        constraint_to_index = {
            constraint: idx
            for idx, constraint in enumerate(solver_model.getConstraint())
        }

        # Retrieve each IIS
        for iis_num in range(1, num_iis + 1):
            iis_constraints = self._extract_iis_constraints(solver_model, iis_num)

            # Convert constraint objects to indices
            for constraint_obj in iis_constraints:
                if constraint_obj in constraint_to_index:
                    labels.add(constraint_to_index[constraint_obj])
                # Note: Silently skip constraints not found in mapping
                # This can happen if the model structure changed after solving

        return sorted(list(labels))

    def _extract_iis_constraints(self, solver_model: Any, iis_num: int) -> list[Any]:
        """
        Extract constraint objects from a specific IIS.

        Parameters
        ----------
        solver_model : xpress.problem
            The Xpress solver model
        iis_num : int
            IIS number (1-indexed)

        Returns
        -------
        list[Any]
            List of xpress.constraint objects in the IIS
        """
        # Prepare lists to receive IIS data
        miisrow: list[Any] = []  # xpress.constraint objects in the IIS
        miiscol: list[Any] = []  # xpress.variable objects in the IIS
        constrainttype: list[str] = []  # Constraint types ('L', 'G', 'E')
        colbndtype: list[str] = []  # Column bound types
        duals: list[float] = []  # Dual values
        rdcs: list[float] = []  # Reduced costs
        isolationrows: list[str] = []  # Row isolation info
        isolationcols: list[str] = []  # Column isolation info

        # Get IIS data from Xpress
        solver_model.getiisdata(
            iis_num,
            miisrow,
            miiscol,
            constrainttype,
            colbndtype,
            duals,
            rdcs,
            isolationrows,
            isolationcols,
        )

        return miisrow

    def print_infeasibilities(self, display_max_terms: int | None = None) -> None:
        """
        Print a list of infeasible constraints.

        This function requires that the model was solved using `gurobi` or `xpress`
        and the termination condition was infeasible.

        Parameters
        ----------
        display_max_terms : int, optional
            The maximum number of infeasible terms to display. If `None`,
            all infeasible terms will be displayed.

        Returns
        -------
        None
            This function does not return anything. It simply prints the
            infeasible constraints.
        """
        labels = self.compute_infeasibilities()
        self.constraints.print_labels(labels, display_max_terms=display_max_terms)

    @deprecated(
        details="Use `compute_infeasibilities`/`print_infeasibilities` instead."
    )
    def compute_set_of_infeasible_constraints(self) -> Dataset:
        """
        Compute a set of infeasible constraints.

        This function requires that the model was solved with `gurobi` or `xpress` and the
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

    def reset_solution(self) -> None:
        """
        Reset the solution and dual values if available of the model.
        """
        self.variables.reset_solution()
        self.constraints.reset_dual()

    to_netcdf = to_netcdf

    to_file = to_file

    to_gurobipy = to_gurobipy

    to_mosek = to_mosek

    to_highspy = to_highspy

    to_block_files = to_block_files
