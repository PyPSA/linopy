"""
Linopy model module.

This module contains frontend implementations of the package.
"""

from __future__ import annotations

import logging
import os
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile, gettempdir
from typing import TYPE_CHECKING, Any, Literal, overload
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr
from deprecation import deprecated
from numpy import inf
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from xarray import DataArray, Dataset
from xarray.core.types import T_Chunks

from linopy import solvers
from linopy.alignment import as_dataarray, broadcast_to_coords
from linopy.common import (
    assign_multiindex_safe,
    best_int,
    maybe_replace_signs,
    replace_by_map,
    to_path,
)
from linopy.config import options
from linopy.constants import (
    GREATER_EQUAL,
    HELPER_DIMS,
    LESS_EQUAL,
    SOS_BIG_M_ATTR,
    SOS_DIM_ATTR,
    SOS_TYPE_ATTR,
    TERM_DIM,
    ModelStatus,
    Result,
    TerminationCondition,
)
from linopy.constraints import (
    AnonymousScalarConstraint,
    Constraint,
    ConstraintBase,
    Constraints,
    CSRConstraint,
)
from linopy.dualization import dualize
from linopy.expressions import (
    LinearExpression,
    QuadraticExpression,
    ScalarLinearExpression,
)
from linopy.io import (
    copy,
    deepcopy,
    shallowcopy,
    to_block_files,
    to_cupdlpx,
    to_file,
    to_gurobipy,
    to_highspy,
    to_mosek,
    to_netcdf,
    to_xpress,
)
from linopy.matrices import MatrixAccessor
from linopy.objective import Objective
from linopy.piecewise import (
    add_piecewise_formulation,
)
from linopy.remote import RemoteHandler

try:
    from linopy.remote import OetcHandler
except ImportError:
    OetcHandler = None  # type: ignore
from linopy.solver_capabilities import solver_supports
from linopy.solvers import (
    IO_APIS,
    SolverFeature,
    available_solvers,
)
from linopy.sos_reformulation import (
    SOSReformulationResult,
    reformulate_sos_constraints,
    sos_reformulation_context,
    undo_sos_reformulation,
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

if TYPE_CHECKING:
    from linopy.piecewise import PiecewiseFormulation

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

    _solver: solvers.Solver | None
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
    _pwlCounter: int
    _blocks: DataArray | None
    _chunk: T_Chunks
    _force_dim_names: bool
    _freeze_constraints: bool
    _set_names_in_solver_io: bool
    _solver_dir: Path
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
        "_pwlCounter",
        "_blocks",
        # TODO: check if these should not be mutable
        "_chunk",
        "_force_dim_names",
        "_auto_mask",
        "_freeze_constraints",
        "_set_names_in_solver_io",
        "_solver_dir",
        "_relaxed_registry",
        "_piecewise_formulations",
        "_solver",
        "_sos_reformulation_state",
        "__weakref__",
    )

    def __init__(
        self,
        solver_dir: str | None = None,
        chunk: T_Chunks = None,
        force_dim_names: bool = False,
        auto_mask: bool = False,
        freeze_constraints: bool = False,
        set_names_in_solver_io: bool = True,
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
        auto_mask : bool
            Whether to automatically mask variables and constraints where
            bounds, coefficients, or RHS values contain NaN. The default is
            False.
        freeze_constraints : bool
            Whether constraints added to the model should be frozen to the
            CSR-backed representation by default. The default is False.
        set_names_in_solver_io : bool
            Whether direct solver exports should include variable and
            constraint names by default. The default is True.

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
        self._pwlCounter: int = 0
        self._blocks: DataArray | None = None

        self._chunk: T_Chunks = chunk
        self._force_dim_names: bool = bool(force_dim_names)
        self._auto_mask: bool = bool(auto_mask)
        self._freeze_constraints: bool = bool(freeze_constraints)
        self._set_names_in_solver_io: bool = bool(set_names_in_solver_io)
        self._piecewise_formulations: dict[str, PiecewiseFormulation] = {}
        self._relaxed_registry: dict[str, str] = {}
        self._solver_dir: Path = Path(
            gettempdir() if solver_dir is None else solver_dir
        )
        self._solver: solvers.Solver | None = None
        self._sos_reformulation_state: SOSReformulationResult | None = None

    @property
    def solver(self) -> solvers.Solver | None:
        return self._solver

    @solver.setter
    def solver(self, value: solvers.Solver | None) -> None:
        if self._solver is not None and self._solver is not value:
            self._solver.close()
        self._solver = value

    @property
    def solver_model(self) -> Any:
        return self.solver.solver_model if self.solver is not None else None

    @solver_model.setter
    def solver_model(self, value: Any) -> None:
        if value is not None:
            raise AttributeError("solver state is managed via model.solver")
        self.solver = None

    @property
    def solver_name(self) -> str | None:
        return self.solver.solver_name.value if self.solver is not None else None

    @solver_name.setter
    def solver_name(self, value: str | None) -> None:
        if value is not None:
            raise AttributeError("solver state is managed via model.solver")
        self.solver = None

    @property
    def matrices(self) -> MatrixAccessor:
        """Matrix representation of the model, computed fresh on each access."""
        return MatrixAccessor(self)

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
    def indicator_constraints(self) -> Constraints:
        """
        Indicator constraints assigned to the model.

        Returns the subset of ``model.constraints`` for which
        ``is_indicator`` is True.
        """
        return self.constraints.indicator

    @property
    def objective(self) -> Objective:
        """
        Objective assigned to the model.
        """
        return self._objective

    @objective.setter
    def objective(
        self, obj: Objective | LinearExpression | QuadraticExpression
    ) -> None:
        """
        Set the objective function.

        Parameters
        ----------
        obj : Objective, LinearExpression, or QuadraticExpression
            The objective to assign to the model. If not an Objective instance,
            it will be wrapped in an Objective.
        """
        if not isinstance(obj, Objective):
            obj = Objective(obj, self)

        self._objective = obj

    @property
    def sense(self) -> str:
        """
        Sense of the objective function.
        """
        return self.objective.sense

    @sense.setter
    def sense(self, value: str) -> None:
        """
        Set the sense of the objective function.
        """
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
        """
        Set the parameters of the model.
        """
        self._parameters = (
            value.copy() if isinstance(value, Dataset) else Dataset(value)
        )

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
        """
        Set the status of the model.
        """
        self._status = ModelStatus[value].value

    @property
    def termination_condition(self) -> str:
        """
        Termination condition of the model.
        """
        return self._termination_condition

    @termination_condition.setter
    def termination_condition(self, value: str) -> None:
        """
        Set the termination condition of the model.
        """
        if value == "":
            self._termination_condition = value
        else:
            self._termination_condition = TerminationCondition[value].value

    @property
    def chunk(self) -> T_Chunks:
        """
        Chunk sizes of the model.
        """
        return self._chunk

    @chunk.setter
    def chunk(self, value: T_Chunks) -> None:
        """
        Set the chunk sizes of the model.
        """
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
        """
        Set whether to force custom dimension names for variables and constraints.
        """
        self._force_dim_names = bool(value)

    @property
    def auto_mask(self) -> bool:
        """
        If True, automatically mask variables and constraints where bounds,
        coefficients, or RHS values contain NaN.
        """
        return self._auto_mask

    @auto_mask.setter
    def auto_mask(self, value: bool) -> None:
        """
        Set whether to automatically mask variables and constraints with NaN values.
        """
        self._auto_mask = bool(value)

    @property
    def freeze_constraints(self) -> bool:
        """Whether constraints are frozen to CSR by default when added."""
        return self._freeze_constraints

    @freeze_constraints.setter
    def freeze_constraints(self, value: bool) -> None:
        self._freeze_constraints = bool(value)

    @property
    def set_names_in_solver_io(self) -> bool:
        """Whether direct solver exports include names by default."""
        return self._set_names_in_solver_io

    @set_names_in_solver_io.setter
    def set_names_in_solver_io(self, value: bool) -> None:
        self._set_names_in_solver_io = bool(value)

    @property
    def solver_dir(self) -> Path:
        """
        Solver directory of the model.
        """
        return self._solver_dir

    @solver_dir.setter
    def solver_dir(self, value: str | Path) -> None:
        """
        Set the solver directory of the model.
        """
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
            "_pwlCounter",
            "force_dim_names",
            "auto_mask",
            "freeze_constraints",
            "set_names_in_solver_io",
        ]

    def __repr__(self) -> str:
        """
        Return a string representation of the linopy model.
        """
        from linopy.piecewise import _get_piecewise_groups
        from linopy.piecewise import _repr_summary as pwl_repr_summary

        var_names, con_names = _get_piecewise_groups(self)
        var_string = self.variables._format_items(exclude=var_names)
        con_string = self.constraints._format_items(exclude=con_names)
        model_string = f"Linopy {self.type} model"

        return (
            f"{model_string}\n{'=' * len(model_string)}\n\n"
            f"Variables:\n----------\n{var_string}\n"
            f"Constraints:\n------------\n{con_string}"
            f"{pwl_repr_summary(self)}"
            f"\nStatus:\n-------\n{self.status}"
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
        coords: Sequence[Sequence | pd.Index] | Mapping | None = None,
        name: str | None = None,
        mask: MaskLike | None = None,
        binary: bool = False,
        integer: bool = False,
        semi_continuous: bool = False,
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
            Lower bound of the variable(s). For binary variables it
            defaults to 0 and, if given, must be 0 or 1. The default is -inf.
        upper : TYPE, optional
            Upper bound of the variable(s). For binary variables it
            defaults to 1 and, if given, must be 0 or 1. The default is inf.
        coords : list/dict/xarray.Coordinates, optional
            The coords of the variable array. When provided with **named
            dimensions** (a ``Mapping``, ``xarray.Coordinates``, a
            sequence of named ``pd.Index`` objects, or an unnamed
            sequence paired with ``dims=`` in ``**kwargs``), ``coords``
            is the source of truth for the variable's dimensions,
            order, and values. ``lower``, ``upper`` and ``mask`` are
            aligned to this contract:

            - dims of every bound must be a subset of ``coords.dims``;
              extra dims raise ``ValueError``;
            - dim order in the variable always follows ``coords``;
            - shared-dim coordinate values must equal ``coords``; same
              values in a different order are auto-reindexed, different
              value sets raise ``ValueError``;
            - dims listed in ``coords`` but missing from a bound are
              broadcast to ``coords`` shape.

            One optimization variable is added per combination of
            coordinates. The default is ``None``, in which case the
            shape is inferred from the bounds.
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
        semi_continuous : bool
            Whether the new variable is a semi-continuous variable. A
            semi-continuous variable can take the value 0 or any value
            between its lower and upper bounds. Requires a positive lower
            bound.
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

        Strict coords-as-truth: a bound with an extra dim raises.

        >>> import xarray as xr
        >>> m = Model()
        >>> bad = xr.DataArray(
        ...     [[1.0, 2.0, 3.0]] * 2,
        ...     dims=["extra", "x"],
        ...     coords={"x": [0, 1, 2]},
        ... )
        >>> m.add_variables(lower=bad, coords=[pd.Index([0, 1, 2], name="x")], name="v")
        Traceback (most recent call last):
        ...
        ValueError: lower bound has dimension(s) ['extra'] not declared in coords ...

        Strict coords-as-truth: a bound whose shared-dim values don't
        match raises.

        >>> m = Model()
        >>> wrong = xr.DataArray(
        ...     [1.0, 2.0, 3.0], dims=["x"], coords={"x": [10, 20, 30]}
        ... )
        >>> m.add_variables(
        ...     lower=wrong, coords=[pd.Index([0, 1, 2], name="x")], name="v"
        ... )
        Traceback (most recent call last):
        ...
        ValueError: lower bound: coordinate values for dimension 'x' do not match coords ...

        Strict coords-as-truth, helpful side: a bound whose coord values
        match ``coords`` only in a different order is auto-reindexed.

        >>> m = Model()
        >>> reordered = xr.DataArray(
        ...     [3.0, 1.0, 2.0], dims=["x"], coords={"x": ["c", "a", "b"]}
        ... )
        >>> v = m.add_variables(
        ...     lower=reordered,
        ...     coords=[pd.Index(["a", "b", "c"], name="x")],
        ...     name="r",
        ... )
        >>> v.lower.values.tolist()
        [1.0, 2.0, 3.0]

        Unnamed-coords sequence + ``dims=`` opts into the same strict
        enforcement as a named index — extra dims still raise.

        >>> m = Model()
        >>> m.add_variables(lower=bad, coords=[[0, 1, 2]], dims=["x"], name="w")
        Traceback (most recent call last):
        ...
        ValueError: lower bound has dimension(s) ['extra'] not declared in coords ...

        The same strict contract applies to ``mask`` (including with
        ``coords=[[...]], dims=[...]``).

        >>> m = Model()
        >>> m.add_variables(mask=bad, coords=[[0, 1, 2]], dims=["x"], name="wm")
        Traceback (most recent call last):
        ...
        ValueError: mask has dimension(s) ['extra'] not declared in coords ...
        """
        if name is None:
            name = f"var{self._varnameCounter}"
            self._varnameCounter += 1

        if name in self.variables:
            raise ValueError(f"Variable '{name}' already assigned to model")

        if sum([binary, integer, semi_continuous]) > 1:
            raise ValueError(
                "Variable can only be one of binary, integer, or semi-continuous."
            )

        if binary:
            if np.isscalar(lower) and lower == -inf:
                lower = 0
            elif not (np.isin(lower, (0, 1)) | pd.isna(lower)).all():
                raise ValueError("Binary variable lower bounds must be 0 or 1.")
            if np.isscalar(upper) and upper == inf:
                upper = 1
            elif not (np.isin(upper, (0, 1)) | pd.isna(upper)).all():
                raise ValueError("Binary variable upper bounds must be 0 or 1.")

        if semi_continuous:
            if not np.isscalar(lower) or float(lower) <= 0:  # type: ignore[arg-type]
                raise ValueError(
                    "Semi-continuous variables require a positive scalar lower bound."
                )

        lower_da = broadcast_to_coords(lower, coords, label="lower bound", **kwargs)
        upper_da = broadcast_to_coords(upper, coords, label="upper bound", **kwargs)
        data = Dataset(
            {
                "lower": lower_da,
                "upper": upper_da,
                "labels": -1,
            }
        )
        (data,) = xr.broadcast(data)
        self.check_force_dim_names(data)
        self._check_valid_dim_names(data)

        if mask is not None:
            mask = broadcast_to_coords(
                mask,
                coords if coords is not None else data.coords,
                label="mask",
                **kwargs,
            ).astype(bool)

        # Auto-mask based on NaN in bounds (use numpy for speed)
        if self.auto_mask:
            auto_mask_values = ~np.isnan(data.lower.values) & ~np.isnan(
                data.upper.values
            )
            auto_mask_arr = DataArray(
                auto_mask_values, coords=data.coords, dims=data.dims
            )
            if mask is not None:
                mask = mask & auto_mask_arr
            else:
                mask = auto_mask_arr

        start = self._xCounter
        end = start + data.labels.size
        label_dtype = options["label_dtype"]
        if end > np.iinfo(label_dtype).max:
            raise ValueError(
                f"Number of labels ({end}) exceeds the maximum value for "
                f"{label_dtype.__name__} ({np.iinfo(label_dtype).max})."
            )
        data.labels.values = np.arange(
            start, end, dtype=options["label_dtype"]
        ).reshape(data.labels.shape)
        self._xCounter += data.labels.size

        if mask is not None:
            data.labels.values = np.where(mask.values, data.labels.values, -1)

        data = data.assign_attrs(
            label_range=(start, end),
            name=name,
            binary=binary,
            integer=integer,
            semi_continuous=semi_continuous,
        )

        if self.chunk:
            data = data.chunk(self.chunk)

        variable = Variable(data, name=name, model=self, skip_broadcast=True)
        self.variables.add(variable)
        return variable

    def add_sos_constraints(
        self,
        variable: Variable,
        sos_type: Literal[1, 2],
        sos_dim: str,
        big_m: float | None = None,
    ) -> None:
        """
        Add an sos1 or sos2 constraint for one dimension of a variable

        The dimension values are used as SOS.

        Parameters
        ----------
        variable : Variable
        sos_type : {1, 2}
            Type of SOS
        sos_dim : str
            Which dimension of variable to add SOS constraint to
        big_m : float | None, optional
            Big-M value for SOS reformulation. Only used when reformulating
            SOS constraints for solvers that don't support them natively.

            - None (default): Use variable upper bounds as Big-M
            - float: Custom Big-M value

            The reformulation uses the tighter of big_m and variable upper bound:
            M = min(big_m, var.upper).

            Tighter Big-M values improve LP relaxation quality and solve time.
        """
        if sos_type not in (1, 2):
            raise ValueError(f"sos_type must be 1 or 2, got {sos_type}")
        if sos_dim not in variable.dims:
            raise ValueError(f"sos_dim must name a variable dimension, got {sos_dim}")

        if SOS_TYPE_ATTR in variable.attrs or SOS_DIM_ATTR in variable.attrs:
            existing_sos_type = variable.attrs.get(SOS_TYPE_ATTR)
            existing_sos_dim = variable.attrs.get(SOS_DIM_ATTR)
            raise ValueError(
                f"variable already has an sos{existing_sos_type} constraint on {existing_sos_dim}"
            )

        # Validate that sos_dim coordinates are numeric (needed for weights)
        if not pd.api.types.is_numeric_dtype(variable.coords[sos_dim]):
            raise ValueError(
                f"SOS constraint requires numeric coordinates for dimension '{sos_dim}', "
                f"but got {variable.coords[sos_dim].dtype}"
            )

        attrs_update: dict[str, Any] = {SOS_TYPE_ATTR: sos_type, SOS_DIM_ATTR: sos_dim}
        if big_m is not None:
            if big_m <= 0:
                raise ValueError(f"big_m must be positive, got {big_m}")
            attrs_update[SOS_BIG_M_ATTR] = float(big_m)

        variable.attrs.update(attrs_update)

    add_piecewise_formulation = add_piecewise_formulation

    def _resolve_constraint_name(self, name: str | None, prefix: str = "con") -> str:
        """Validate a constraint name or generate one from ``prefix``."""
        if name in list(self.constraints):
            raise ValueError(f"Constraint '{name}' already assigned to model")
        if name is None:
            name = f"{prefix}{self._connameCounter}"
            self._connameCounter += 1
        return name

    def _constraint_data_from_lhs(
        self,
        lhs: VariableLike
        | ExpressionLike
        | ConstraintLike
        | Sequence[tuple[ConstantLike, VariableLike | str]]
        | Callable,
        sign: SignLike | None,
        rhs: ConstantLike | VariableLike | ExpressionLike | None,
        coords: Sequence[Sequence | pd.Index] | Mapping | None = None,
    ) -> Dataset:
        """Build the constraint Dataset from an ``lhs`` and optional ``sign``/``rhs``."""
        msg_required = (
            f"`sign` and `rhs` are required when `lhs` is a {type(lhs).__name__}."
        )
        msg_must_be_none = (
            f"`sign` and `rhs` must be None when `lhs` is a {type(lhs).__name__}."
        )
        if isinstance(lhs, LinearExpression):
            if sign is None or rhs is None:
                raise ValueError(msg_required)
            return lhs.to_constraint(sign, rhs).data
        elif isinstance(lhs, list | tuple):
            if sign is None or rhs is None:
                raise ValueError(msg_required)
            return self.linexpr(*lhs).to_constraint(sign, rhs).data
        elif callable(lhs):
            assert coords is not None, "`coords` must be given when lhs is a function"
            if sign is not None or rhs is not None:
                raise ValueError(msg_must_be_none)
            return Constraint.from_rule(self, lhs, coords).data
        elif isinstance(lhs, AnonymousScalarConstraint):
            if sign is not None or rhs is not None:
                raise ValueError(msg_must_be_none)
            return lhs.to_constraint().data
        elif isinstance(lhs, ConstraintBase):
            if sign is not None or rhs is not None:
                raise ValueError(msg_must_be_none)
            return lhs.data
        elif isinstance(lhs, Variable | ScalarVariable | ScalarLinearExpression):
            if sign is None or rhs is None:
                raise ValueError(msg_required)
            return lhs.to_linexpr().to_constraint(sign, rhs).data
        else:
            raise TypeError(
                f"`lhs` must be a LinearExpression, Variable, Constraint, tuple, or "
                f"callable, got {type(lhs).__name__}."
            )

    def _allocate_constraint_labels(
        self, data: Dataset, name: str, mask: DataArray | None = None
    ) -> Dataset:
        """Assign label ranges from the constraint counter and apply an optional mask."""
        start = self._cCounter
        end = start + data.labels.size
        label_dtype = options["label_dtype"]
        if end > np.iinfo(label_dtype).max:
            raise ValueError(
                f"Number of labels ({end}) exceeds the maximum value for "
                f"{label_dtype.__name__} ({np.iinfo(label_dtype).max})."
            )
        data.labels.values = np.arange(start, end, dtype=label_dtype).reshape(
            data.labels.shape
        )
        self._cCounter += data.labels.size
        if mask is not None:
            data.labels.values = np.where(mask.values, data.labels.values, -1)
        return data.assign_attrs(label_range=(start, end), name=name)

    @overload
    def add_constraints(
        self,
        lhs: VariableLike
        | ExpressionLike
        | ConstraintLike
        | Sequence[tuple[ConstantLike, VariableLike | str]]
        | Callable,
        sign: SignLike | None = ...,
        rhs: ConstantLike | VariableLike | ExpressionLike | None = ...,
        name: str | None = ...,
        coords: Sequence[Sequence | pd.Index] | Mapping | None = ...,
        mask: MaskLike | None = ...,
        freeze: Literal[False] = ...,
    ) -> Constraint: ...

    @overload
    def add_constraints(
        self,
        lhs: VariableLike
        | ExpressionLike
        | ConstraintLike
        | Sequence[tuple[ConstantLike, VariableLike | str]]
        | Callable,
        sign: SignLike | None = ...,
        rhs: ConstantLike | VariableLike | ExpressionLike | None = ...,
        name: str | None = ...,
        coords: Sequence[Sequence | pd.Index] | Mapping | None = ...,
        mask: MaskLike | None = ...,
        freeze: Literal[True] = ...,
    ) -> CSRConstraint: ...

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
        coords: Sequence[Sequence | pd.Index] | Mapping | None = None,
        mask: MaskLike | None = None,
        freeze: bool | None = None,
    ) -> ConstraintBase:
        """
        Assign a new, possibly multi-dimensional array of constraints to the
        model.

        Constraints are added by defining a left hand side (lhs), the sign and
        the right hand side (rhs). The lhs has to be a linopy.LinearExpression
        and the rhs a constant (array of constants). The function return the
        an array with the constraint labels (integers).

        Parameters
        ----------
        lhs : linopy.LinearExpression/linopy.ConstraintBase/callable
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
        freeze : bool, optional
            If True, convert the constraint to an immutable CSR-backed CSRConstraint
            for better memory efficiency. If None, uses the model default
            ``Model.freeze_constraints`` setting (default False).

        Returns
        -------
        constraint : linopy.ConstraintBase
            The added constraint (Constraint by default, or CSRConstraint if freeze=True).
        """

        name = self._resolve_constraint_name(name)
        if sign is not None:
            sign = maybe_replace_signs(as_dataarray(sign))

        # Capture original RHS for auto-masking before constraint creation
        # (NaN values in RHS are lost during constraint creation)
        # Use numpy for speed instead of xarray's notnull()
        original_rhs_mask = None
        if self.auto_mask and rhs is not None:
            rhs_da = as_dataarray(rhs)
            original_rhs_mask = (rhs_da.coords, rhs_da.dims, ~np.isnan(rhs_da.values))

        data = self._constraint_data_from_lhs(lhs, sign, rhs, coords)

        invalid_infinity_values = (
            (data.sign == LESS_EQUAL) & (data.rhs == -np.inf)
        ) | ((data.sign == GREATER_EQUAL) & (data.rhs == np.inf))  # noqa: F821
        if invalid_infinity_values.any():
            raise ValueError(f"Constraint {name} contains incorrect infinite values.")

        # ensure helper dimensions are not set as coordinates
        if drop_dims := set(HELPER_DIMS).intersection(data.coords):
            # TODO: add a warning here, routines should be safe against this
            data = data.drop_vars(drop_dims)

        rhs_nan = data.rhs.isnull()
        if rhs_nan.any():
            data = assign_multiindex_safe(data, rhs=data.rhs.fillna(0))
            rhs_mask = ~rhs_nan
            mask = (
                rhs_mask
                if mask is None
                else (as_dataarray(mask).astype(bool) & rhs_mask)
            )

        data["labels"] = -1
        (data,) = xr.broadcast(data, exclude=[TERM_DIM])

        if mask is not None:
            mask = broadcast_to_coords(mask, data.coords, label="mask").astype(bool)

        # Auto-mask based on null expressions or NaN RHS (use numpy for speed)
        if self.auto_mask:
            # Check if expression is null: all vars == -1
            # Use max() instead of all() - if max == -1, all are -1 (since valid vars >= 0)
            # This is ~30% faster for large term dimensions
            vars_all_invalid = data.vars.values.max(axis=-1) == -1
            auto_mask_values = ~vars_all_invalid
            if original_rhs_mask is not None:
                coords, dims, rhs_notnull = original_rhs_mask
                if rhs_notnull.shape != auto_mask_values.shape:
                    rhs_da = DataArray(rhs_notnull, coords=coords, dims=dims)
                    rhs_notnull = rhs_da.broadcast_like(data.labels).values
                auto_mask_values = auto_mask_values & rhs_notnull
            auto_mask_arr = DataArray(
                auto_mask_values, coords=data.labels.coords, dims=data.labels.dims
            )
            if mask is not None:
                mask = mask & auto_mask_arr
            else:
                mask = auto_mask_arr

        self.check_force_dim_names(data)

        data = self._allocate_constraint_labels(data, name, mask)

        if self.chunk:
            data = data.chunk(self.chunk)

        constraint = Constraint(data, name=name, model=self, skip_broadcast=True)
        if freeze is None:
            freeze = self.freeze_constraints
        return self.constraints.add(constraint, freeze=freeze and not self.chunk)

    def add_indicator_constraints(
        self,
        binary_var: Variable,
        binary_val: int,
        lhs: ConstraintLike | ExpressionLike | VariableLike,
        sign: SignLike | None = None,
        rhs: ConstantLike | None = None,
        name: str | None = None,
    ) -> ConstraintBase:
        """
        Add indicator constraints to the model.

        An indicator constraint has the form:
            (binary_var == binary_val) => (linear_constraint)

        The linear constraint is only enforced when binary_var equals
        binary_val. These constraints are handled natively by solvers
        like Gurobi and CPLEX via general constraints.

        Parameters
        ----------
        binary_var : linopy.Variable
            Binary variable serving as the indicator. Must have binary=True.
        binary_val : int
            Triggering value, must be 0 or 1.
        lhs : linopy.Constraint, linopy.LinearExpression, or linopy.Variable
            The conditionally enforced constraint. If a LinearExpression or
            Variable is passed, ``sign`` and ``rhs`` must also be provided.
        sign : str, optional
            Constraint sign ('<=', '>=', '='). Required when ``lhs`` is an
            expression.
        rhs : numeric, optional
            Right-hand side. Required when ``lhs`` is an expression.
        name : str, optional
            Name for the indicator constraint group.

        Returns
        -------
        linopy.constraints.ConstraintBase
            The added indicator constraint.
        """
        if not binary_var.attrs.get("binary", False):
            raise ValueError(
                "Indicator variable must be binary. "
                f"Variable '{binary_var.name}' is not binary."
            )

        if binary_val not in (0, 1):
            raise ValueError(f"binary_val must be 0 or 1, got {binary_val}.")

        name = self._resolve_constraint_name(name, prefix="indcon")
        if sign is not None:
            sign = maybe_replace_signs(as_dataarray(sign))

        data = self._constraint_data_from_lhs(lhs, sign, rhs)

        data["binary_var"] = binary_var.labels
        data["binary_val"] = binary_val

        data["labels"] = -1
        (data,) = xr.broadcast(data, exclude=[TERM_DIM])

        data = self._allocate_constraint_labels(data, name)

        con = Constraint(data, name=name, model=self, skip_broadcast=True)
        freeze = self.freeze_constraints
        return self.constraints.add(con, freeze=freeze and not self.chunk)

    def remove_indicator_constraints(self, name: str) -> None:
        """
        Remove indicator constraint by name.
        """
        self.constraints.remove(name)

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
        variable = self.variables[name]

        self._relaxed_registry.pop(name, None)

        to_remove = [
            k for k, con in self.constraints.items() if con.has_variable(variable)
        ]

        if to_remove:
            warnings.warn(
                f"Removing variable '{name}' also removes constraints {to_remove} "
                "because they reference this variable.",
                UserWarning,
                stacklevel=2,
            )
            for k in to_remove:
                self.constraints.remove(k)

        self.variables.remove(name)

        self.objective = self.objective.sel(
            {TERM_DIM: ~self.objective.vars.isin(variable.labels)}
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

    def remove_sos_constraints(self, variable: Variable) -> None:
        """
        Remove all sos constraints from a given variable.

        Parameters
        ----------
        variable : Variable
            Variable instance from which to remove all sos constraints.
            Can be retrieved from `m.variables.sos`.

        Returns
        -------
        None.
        """
        if SOS_TYPE_ATTR not in variable.attrs or SOS_DIM_ATTR not in variable.attrs:
            raise ValueError(f"Variable '{variable.name}' has no SOS constraints")

        sos_type = variable.attrs[SOS_TYPE_ATTR]
        sos_dim = variable.attrs[SOS_DIM_ATTR]

        del variable.attrs[SOS_TYPE_ATTR], variable.attrs[SOS_DIM_ATTR]

        variable.attrs.pop(SOS_BIG_M_ATTR, None)

        logger.debug(
            f"Removed sos{sos_type} constraint on {sos_dim} from {variable.name}"
        )

    reformulate_sos_constraints = reformulate_sos_constraints

    def apply_sos_reformulation(self) -> None:
        """
        Reformulate SOS constraints into binary + linear form, in place.

        The reformulation token is stored on the model so it can be reverted
        with :meth:`undo_sos_reformulation`. This is the stateful counterpart
        to :func:`linopy.sos_reformulation.reformulate_sos_constraints`, where
        the caller owns the token.

        Raises
        ------
        RuntimeError
            If a reformulation has already been applied and not undone.
        """
        if self._sos_reformulation_state is not None:
            raise RuntimeError(
                "SOS reformulation has already been applied to this model. "
                "Call `undo_sos_reformulation()` before applying again."
            )
        self._sos_reformulation_state = reformulate_sos_constraints(self)

    def undo_sos_reformulation(self) -> None:
        """
        Revert a previously applied SOS reformulation.

        Raises
        ------
        RuntimeError
            If no reformulation is currently applied.
        """
        if self._sos_reformulation_state is None:
            raise RuntimeError(
                "No SOS reformulation is currently applied to this model."
            )
        state = self._sos_reformulation_state
        self._sos_reformulation_state = None
        undo_sos_reformulation(self, state)

    def _resolve_sos_reformulation(
        self,
        solver_name: str | None,
        reformulate_sos: bool | Literal["auto"],
    ) -> bool:
        """
        Decide whether ``apply_sos_reformulation`` should run.

        Validates ``reformulate_sos`` and returns ``True`` iff the SOS
        constraints on this model should be reformulated for the chosen
        solver.  ``solver_name`` is only consulted when
        ``reformulate_sos == "auto"`` (to look up SOS support); for
        ``True`` / ``False`` the decision is independent of the solver.
        """
        if reformulate_sos not in (True, False, "auto"):
            raise ValueError(
                f"Invalid value for reformulate_sos: {reformulate_sos!r}. "
                "Must be True, False, or 'auto'."
            )
        if not self.variables.sos:
            return False

        if reformulate_sos is False:
            return False
        elif reformulate_sos is True:
            return True
        elif solver_name is None:
            raise ValueError(
                "`reformulate_sos='auto'` on a model with SOS constraints "
                "requires an explicit `solver_name` so we can check "
                "whether the chosen solver supports SOS. Pass "
                "`solver_name=...` or use `reformulate_sos=True`/`False` "
                "to skip the lookup."
            )
        return not solver_supports(solver_name, SolverFeature.SOS_CONSTRAINTS)

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
    def semi_continuous(self) -> Variables:
        """
        Get all semi-continuous variables.
        """
        return self.variables.semi_continuous

    @property
    def is_linear(self) -> bool:
        """Whether the objective is linear."""
        return self.objective.is_linear

    @property
    def is_quadratic(self) -> bool:
        """Whether the objective is quadratic."""
        return self.objective.is_quadratic

    @property
    def type(self) -> str:
        """Short string identifying the problem type."""
        if (
            len(self.binaries) or len(self.integers) or len(self.semi_continuous)
        ) and len(self.continuous):
            variable_type = "MI"
        elif len(self.binaries) or len(self.integers) or len(self.semi_continuous):
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
        self, *args: Sequence[Sequence | pd.Index] | Mapping
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
        | Sequence[Sequence | pd.Index]
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
        set_names: bool | None = None,
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
        remote: RemoteHandler | OetcHandler | None = None,
        progress: bool | None = None,
        mock_solve: bool = False,
        reformulate_sos: bool | Literal["auto"] = False,
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
        set_names : bool, optional
            Whether to set variable and constraint names when using the direct
            solver API (io_api='direct'). Setting to False can significantly
            speed up model export. If None, uses the model default
            ``Model.set_names_in_solver_io`` setting (default True).
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
        mock_solve : bool, optional
            Whether to run a mock solve. This will skip the actual solving. Variables will be set to have dummy values
        reformulate_sos : bool | Literal["auto"], optional
            Whether to reformulate SOS constraints as binary + linear constraints.
            If True, always reformulates, even when the solver supports SOS natively.
            If "auto", reformulates only when the solver lacks SOS support.
            If False, raises if the solver doesn't support SOS.
            Reformulation uses the Big-M method and requires all SOS variables
            to have finite bounds. Default is False.
        **solver_options : kwargs
            Options passed to the solver.

        Returns
        -------
        status : tuple
            Tuple containing the status and termination condition of the
            optimization process.
        """
        if mock_solve:
            return self._mock_solve(
                sanitize_zeros=sanitize_zeros, sanitize_infinities=sanitize_infinities
            )

        # check io_api
        if io_api is not None and io_api not in IO_APIS:
            raise ValueError(
                f"Keyword argument `io_api` has to be one of {IO_APIS} or None"
            )

        if remote is not None:
            # The remote branch short-circuits before reaching Solver.solve(),
            # which is where the empty-objective check normally fires. Replicate
            # it here. This duplication becomes obsolete once OETC is folded
            # into the Solver pipeline (see PyPSA/linopy#683).
            if self.objective.expression.empty:
                raise ValueError(
                    "No objective has been set on the model. Use "
                    "`m.add_objective(...)` first (e.g. `m.add_objective(0 * x)` "
                    "for a pure feasibility problem)."
                )
            if isinstance(remote, OetcHandler):
                solved = remote.solve_on_oetc(
                    self,
                    solver_name=solver_name,
                    reformulate_sos=reformulate_sos,
                    **solver_options,
                )
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
                    reformulate_sos=reformulate_sos,
                    **solver_options,
                )

            if solved.objective.value is not None:
                self.objective.set_value(float(solved.objective.value))
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

        solver_class = getattr(solvers, solvers.SolverName(solver_name).name)

        if problem_fn is None:
            problem_fn = self.get_problem_file(io_api=io_api)
        if solution_fn is None:
            if (
                solver_class.supports(SolverFeature.SOLUTION_FILE_NOT_NEEDED)
                and not keep_files
            ):
                # these (solver, keep_files=False) combos do not need a solution file
                solution_fn = None
            else:
                solution_fn = self.get_solution_file()

        with sos_reformulation_context(self, solver_name, reformulate_sos):
            if sanitize_zeros:
                self.constraints.sanitize_zeros()
            if sanitize_infinities:
                self.constraints.sanitize_infinities()

            try:
                self.solver = None  # closes any previous solver
                if io_api == "direct":
                    if set_names is None:
                        set_names = self.set_names_in_solver_io
                    build_kwargs: dict[str, Any] = {
                        "explicit_coordinate_names": explicit_coordinate_names,
                        "set_names": set_names,
                        "log_fn": to_path(log_fn),
                    }
                    if env is not None:
                        build_kwargs["env"] = env
                else:
                    build_kwargs = {
                        "explicit_coordinate_names": explicit_coordinate_names,
                        "slice_size": slice_size,
                        "progress": progress,
                        "problem_fn": to_path(problem_fn),
                    }
                self.solver = solver = solvers.Solver.from_name(
                    solver_name,
                    model=self,
                    io_api=io_api,
                    options=solver_options,
                    **build_kwargs,
                )
                if io_api != "direct":
                    problem_fn = solver._problem_fn
                result = solver.solve(
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

            return self.assign_result(result)

    def assign_result(
        self,
        result: Result,
        solver: solvers.Solver | None = None,
    ) -> tuple[str, str]:
        """
        Write a solver Result back onto the model.

        Copies primal / dual values onto variables / constraints, sets
        :attr:`status`, :attr:`termination_condition`, and
        :attr:`objective.value`. When ``solver`` is provided, also stores it on
        ``self.solver`` so post-solve introspection (``model.solver_model``,
        ``compute_infeasibilities()``) works.

        Parameters
        ----------
        result : Result
            The :class:`linopy.constants.Result` returned by
            :meth:`linopy.solvers.Solver.solve`.
        solver : Solver, optional
            The solver instance that produced the result. Pass it on the
            low-level ``Solver.from_name(...).solve()`` path to attach it as
            ``self.solver`` for post-solve introspection. ``Model.solve()``
            attaches the solver itself and does not pass this argument.
        """
        if solver is not None:
            self.solver = solver

        result.info()

        if result.solution is not None:
            self.objective._value = result.solution.objective

        status_value = result.status.status.value
        termination_condition = result.status.termination_condition.value
        self.status = status_value
        self.termination_condition = termination_condition

        if not result.status.is_ok:
            return status_value, termination_condition

        if result.solution is None or len(result.solution.primal) == 0:
            return status_value, termination_condition

        primal = result.solution.primal
        for _, var in self.variables.items():
            start, end = var.range
            var.solution = xr.DataArray(
                primal[start:end].reshape(var.shape), var.coords, dims=var.dims
            )

        if len(result.solution.dual):
            dual = result.solution.dual
            for _, con in self.constraints.items():
                if con.is_indicator:
                    continue
                start, end = con.range
                coords = {dim: con.coords[dim] for dim in con.coord_dims}
                con.dual = xr.DataArray(
                    dual[start:end].reshape(con.shape), coords, dims=con.coord_dims
                )

        return status_value, termination_condition

    def _mock_solve(
        self,
        sanitize_zeros: bool = True,
        sanitize_infinities: bool = True,
    ) -> tuple[str, str]:
        solver_name = "mock"

        logger.info(f" Solve problem using {solver_name.title()} solver")
        self.solver = None
        # reset result
        self.reset_solution()

        if sanitize_zeros:
            self.constraints.sanitize_zeros()

        if sanitize_infinities:
            self.constraints.sanitize_infinities()

        self.objective._value = 0.0
        self.status = "ok"
        self.termination_condition = TerminationCondition.optimal.value

        for name, var in self.variables.items():
            var.solution = xr.DataArray(0.0, var.coords)

        for name, con in self.constraints.items():
            con.dual = xr.DataArray(0.0, con.labels.coords)

        return "ok", "none"

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
        solver_model = self.solver_model

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
            solver_name = self.solver_name or "unknown"
            if self.solver is not None and self.solver.supports(
                SolverFeature.IIS_COMPUTATION
            ):
                raise ValueError(
                    "No solver model available. The model must be solved first with "
                    "a solver that supports IIS computation and the result must be infeasible."
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
        """
        Compute infeasibilities for Xpress solver.

        This function correctly maps solver constraint positions to linopy
        constraint labels, handling masked constraints where some labels may
        be skipped (e.g., labels [0, 2, 4] with gaps instead of sequential
        [0, 1, 2]).
        """
        # Compute a single IIS (matches Gurobi behavior; multiple IIS would
        # otherwise get flattened into an ambiguous union). Mode 2 prioritises
        # a fast IIS search over minimality.
        try:  # Try new API first
            solver_model.firstIIS(2)
        except AttributeError:  # Fallback to old API
            solver_model.iisfirst(2)

        if solver_model.attributes.numiis == 0:
            return []

        clabels = self.constraints.label_index.clabels
        constraint_position_map = {}
        for position, constraint_obj in enumerate(solver_model.getConstraint()):
            if 0 <= position < len(clabels):
                constraint_label = clabels[position]
                if constraint_label >= 0:
                    constraint_position_map[constraint_obj] = constraint_label

        labels = set()
        for constraint_obj in self._extract_iis_constraints(solver_model, 1):
            if constraint_obj in constraint_position_map:
                labels.add(constraint_position_map[constraint_obj])

        return sorted(labels)

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
        # Declare variables before try/except to avoid mypy redefinition errors
        miisrow: list[Any]
        miiscol: list[Any]
        constrainttype: list[str]
        colbndtype: list[str]
        duals: list[float]
        rdcs: list[float]
        isolationrows: list[str]
        isolationcols: list[str]

        try:  # Try new API first
            (
                miisrow,
                miiscol,
                constrainttype,
                colbndtype,
                duals,
                rdcs,
                isolationrows,
                isolationcols,
            ) = solver_model.getIISData(iis_num)

            # Transform list of indices to list of constraint objects
            for i in range(len(miisrow)):
                miisrow[i] = solver_model.getConstraint(miisrow[i])

        except AttributeError:  # Fallback to old API
            # Prepare lists to receive IIS data
            miisrow = []  # xpress.constraint objects in the IIS
            miiscol = []  # xpress.variable objects in the IIS
            constrainttype = []  # Constraint types ('L', 'G', 'E')
            colbndtype = []  # Column bound types
            duals = []  # Dual values
            rdcs = []  # Reduced costs
            isolationrows = []  # Row isolation info
            isolationcols = []  # Column isolation info

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

    def format_infeasibilities(self, display_max_terms: int | None = None) -> str:
        """
        Return a string representation of infeasible constraints.

        This function requires that the model was solved using `gurobi` or `xpress`
        and the termination condition was infeasible.

        Parameters
        ----------
        display_max_terms : int, optional
            The maximum number of infeasible terms to display. If ``None``,
            uses the global ``linopy.options.display_max_terms`` setting.

        Returns
        -------
        str
            String representation of the infeasible constraints.
        """
        labels = self.compute_infeasibilities()
        return self.constraints.format_labels(
            labels, display_max_terms=display_max_terms
        )

    def print_infeasibilities(self, display_max_terms: int | None = None) -> None:
        """
        Print a list of infeasible constraints.

        .. deprecated::
            Use :meth:`format_infeasibilities` instead.
        """
        warn(
            "`Model.print_infeasibilities` is deprecated. Use `Model.format_infeasibilities` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        print(self.format_infeasibilities(display_max_terms=display_max_terms))

    @deprecated(
        details="Use `compute_infeasibilities`/`format_infeasibilities` instead."
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

    copy = copy

    __copy__ = shallowcopy

    __deepcopy__ = deepcopy

    to_netcdf = to_netcdf

    to_file = to_file

    to_gurobipy = to_gurobipy

    to_mosek = to_mosek

    to_highspy = to_highspy

    to_cupdlpx = to_cupdlpx

    to_xpress = to_xpress

    to_block_files = to_block_files

    dualize = dualize
