#!/usr/bin/env python3
"""
Linopy expressions module.

This module contains definition related to affine expressions.
"""

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product, zip_longest
from typing import TYPE_CHECKING, Any, TypeVar, overload
from warnings import warn

import numpy as np
import pandas as pd
import polars as pl
import scipy
import xarray as xr
import xarray.core.groupby
from numpy import array, nan, ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from scipy.sparse import csc_matrix
from xarray import Coordinates, DataArray, Dataset, IndexVariable
from xarray.core.coordinates import DataArrayCoordinates, DatasetCoordinates
from xarray.core.indexes import Indexes
from xarray.core.utils import Frozen

try:
    # resolve breaking change in xarray 2025.03.0
    import xarray.computation.rolling
    from xarray.computation.rolling import DatasetRolling
except ImportError:
    import xarray.core.rolling
    from xarray.core.rolling import DatasetRolling  # type: ignore

from types import EllipsisType, NotImplementedType

from linopy import constraints, variables
from linopy.common import (
    EmptyDeprecationWrapper,
    LocIndexer,
    as_dataarray,
    assign_multiindex_safe,
    check_common_keys_values,
    check_has_nulls,
    check_has_nulls_polars,
    fill_missing_coords,
    filter_nulls_polars,
    forward_as_properties,
    generate_indices_for_printout,
    get_dims_with_index_levels,
    get_index_map,
    group_terms_polars,
    has_optimized_model,
    iterate_slices,
    print_coord,
    print_single_expression,
    to_dataframe,
    to_polars,
)
from linopy.config import options
from linopy.constants import (
    EQUAL,
    FACTOR_DIM,
    GREATER_EQUAL,
    GROUP_DIM,
    GROUPED_TERM_DIM,
    HELPER_DIMS,
    LESS_EQUAL,
    STACKED_TERM_DIM,
    TERM_DIM,
)
from linopy.types import (
    ConstantLike,
    DimsLike,
    ExpressionLike,
    SideLike,
    SignLike,
    VariableLike,
)

if TYPE_CHECKING:
    from linopy.constraints import AnonymousScalarConstraint, Constraint
    from linopy.model import Model
    from linopy.variables import ScalarVariable, Variable

SUPPORTED_CONSTANT_TYPES = (
    np.number,
    int,
    float,
    DataArray,
    pd.Series,
    pd.DataFrame,
    np.ndarray,
    pl.Series,
)


FILL_VALUE = {"vars": -1, "coeffs": np.nan, "const": np.nan}


def exprwrap(
    method: Callable, *default_args: Any, **new_default_kwargs: Any
) -> Callable:
    @functools.wraps(method)
    def _exprwrap(
        expr: LinearExpression | QuadraticExpression, *args: Any, **kwargs: Any
    ) -> LinearExpression | QuadraticExpression:
        for k, v in new_default_kwargs.items():
            kwargs.setdefault(k, v)
        return expr.__class__(
            method(_expr_unwrap(expr), *default_args, *args, **kwargs), expr.model
        )

    _exprwrap.__doc__ = (
        f"Wrapper for the xarray {method.__qualname__} function for linopy.Variable"
    )
    if new_default_kwargs:
        _exprwrap.__doc__ += f" with default arguments: {new_default_kwargs}"

    return _exprwrap


def _expr_unwrap(
    maybe_expr: Any | LinearExpression | QuadraticExpression,
) -> Any:
    if isinstance(maybe_expr, LinearExpression | QuadraticExpression):
        return maybe_expr.data

    return maybe_expr


logger = logging.getLogger(__name__)


@dataclass
@forward_as_properties(groupby=["dims", "groups"])
class LinearExpressionGroupby:
    """
    GroupBy object specialized to grouping LinearExpression objects.
    """

    data: xr.Dataset
    group: Hashable | DataArray | IndexVariable | pd.Series | pd.DataFrame
    model: Any
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def groupby(self) -> xarray.core.groupby.DatasetGroupBy:
        """
        Groups the data using the specified group and kwargs.

        Returns
        -------
        xarray.core.groupby.DataArrayGroupBy
            The groupby object.
        """
        if isinstance(self.group, pd.DataFrame):
            raise ValueError(
                "Grouping by a DataFrame only supported for `sum` operation with `use_fallback=False`."
            )
        if isinstance(self.group, pd.Series):
            group_name = self.group.name or "group"
            group = DataArray(self.group, name=group_name)
        else:
            group = self.group  # type: ignore

        return self.data.groupby(group=group, **self.kwargs)

    def map(
        self,
        func: Callable[..., Dataset],
        shortcut: bool = False,
        args: tuple[()] = (),
        **kwargs: Any,
    ) -> LinearExpression:
        """
        Apply a specified function to the groupby object.

        Parameters
        ----------
        func : callable
            The function to apply.
        shortcut : bool, optional
            Whether to use shortcut or not.
        args : tuple, optional
            The arguments to pass to the function.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        LinearExpression
            The result of applying the function to the groupby object.
        """
        return LinearExpression(
            self.groupby.map(func, shortcut=shortcut, args=args, **kwargs), self.model
        )

    def sum(self, use_fallback: bool = False, **kwargs: Any) -> LinearExpression:
        """
        Sum the groupby object.

        There are two options to perform the summation over groups.
        The first and faster option uses an internal reindexing mechanism, which
        however ignores keyword arguments. This will be used when passing a
        pandas object or a DataArray as group, and setting `use_fallack`
        to False (default).
        The second uses a mapping of xarray groups which performs slower but
        also takes into account the keyword arguments.

        Parameters
        ----------
        use_fallback : bool
            Whether to use the fallback implementation, which is a sort of default
            xarray implementation. If set to False, the operation will be much
            faster but keyword arguments are ignored. Defaults to False.
        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        LinearExpression
            The sum of the groupby object.
        """
        non_fallback_types = (pd.Series, pd.DataFrame, xr.DataArray)
        if isinstance(self.group, non_fallback_types) and not use_fallback:
            group: pd.Series | pd.DataFrame | xr.DataArray = self.group
            if isinstance(group, pd.DataFrame):
                # dataframes do not have a name, so we need to set it
                final_group_name = "group"
            else:
                final_group_name = getattr(group, "name", "group") or "group"

            if isinstance(group, DataArray):
                group = group.to_pandas()

            int_map = None
            if isinstance(group, pd.DataFrame):
                index_name = group.index.name
                group = group.reindex(self.data.indexes[group.index.name])
                group.index.name = index_name  # ensure name for multiindex
                int_map = get_index_map(*group.values.T)
                orig_group = group
                group = group.apply(tuple, axis=1).map(int_map)

            # At this point, group is always a pandas Series
            assert isinstance(group, pd.Series)
            group_dim = group.index.name

            arrays = [group, group.groupby(group).cumcount()]
            idx = pd.MultiIndex.from_arrays(arrays, names=[GROUP_DIM, GROUPED_TERM_DIM])
            new_coords = Coordinates.from_pandas_multiindex(idx, group_dim)
            coords = self.data.indexes[group_dim]
            names_to_drop = [coords.name]
            if isinstance(coords, pd.MultiIndex):
                names_to_drop += list(coords.names)
            ds = self.data.drop_vars(names_to_drop).assign_coords(new_coords)
            ds = ds.unstack(group_dim, fill_value=LinearExpression._fill_value)
            ds = LinearExpression._sum(ds, dim=GROUPED_TERM_DIM)

            if int_map is not None:
                index = ds.indexes[GROUP_DIM].map({v: k for k, v in int_map.items()})
                index.names = [str(col) for col in orig_group.columns]
                index.name = GROUP_DIM
                new_coords = Coordinates.from_pandas_multiindex(index, GROUP_DIM)
                ds = xr.Dataset(ds.assign_coords(new_coords))

            ds = ds.rename({GROUP_DIM: final_group_name})
            return LinearExpression(ds, self.model)

        def func(ds: Dataset) -> Dataset:
            ds = LinearExpression._sum(ds, str(self.groupby._group_dim))
            ds = ds.assign_coords({TERM_DIM: np.arange(len(ds._term))})
            return ds

        return self.map(func, **kwargs, shortcut=True)

    def roll(self, **kwargs: Any) -> LinearExpression:
        """
        Roll the groupby object.

        **kwargs
            Arbitrary keyword arguments.

        Returns
        -------
        LinearExpression
            The result of rolling over the groups.
        """
        return self.map(Dataset.roll, **kwargs)


@dataclass
@forward_as_properties(rolling=["center", "dim", "obj", "rollings", "window"])
class LinearExpressionRolling:
    """
    GroupBy object specialized to grouping LinearExpression objects.
    """

    rolling: DatasetRolling
    model: Any

    def sum(self, **kwargs: Any) -> LinearExpression:
        data = self.rolling.construct("_rolling_term", keep_attrs=True)
        ds = (
            data[["coeffs", "vars"]]
            .rename({TERM_DIM: STACKED_TERM_DIM})
            .stack({TERM_DIM: [STACKED_TERM_DIM, "_rolling_term"]}, create_index=False)
        )
        ds = assign_multiindex_safe(ds, const=data.const.sum("_rolling_term"))
        return LinearExpression(ds, self.model)


class BaseExpression(ABC):
    __slots__ = ("_data", "_model")
    __array_ufunc__ = None
    __array_priority__ = 10000
    __pandas_priority__ = 10000

    _fill_value = FILL_VALUE
    _data: Dataset

    @property
    @abstractmethod
    def flat(self) -> pd.DataFrame: ...

    @abstractmethod
    def to_polars(self) -> pl.DataFrame: ...

    def __init__(self, data: Dataset | Any | None, model: Model) -> None:
        from linopy.model import Model

        if data is None:
            da = xr.DataArray([], dims=[TERM_DIM])
            data = Dataset({"coeffs": da, "vars": da, "const": 0.0})
        elif isinstance(data, SUPPORTED_CONSTANT_TYPES):
            const = as_dataarray(data)
            da = xr.DataArray([], dims=[TERM_DIM])
            data = Dataset({"coeffs": da, "vars": da, "const": const})
        elif not isinstance(data, Dataset):
            supported_types = ", ".join(
                map(lambda s: s.__qualname__, (*SUPPORTED_CONSTANT_TYPES, Dataset))
            )
            raise ValueError(
                f"data must be an instance of {supported_types}, got {type(data)}"
            )

        if not set(data).issuperset({"coeffs", "vars"}):
            raise ValueError(
                "data must contain the fields 'coeffs' and 'vars' or 'const'"
            )

        if np.issubdtype(data.vars, np.floating):
            data = assign_multiindex_safe(data, vars=data.vars.fillna(-1).astype(int))
        if not np.issubdtype(data.coeffs, np.floating):
            data["coeffs"].values = data.coeffs.values.astype(float)

        data = fill_missing_coords(data)

        if TERM_DIM not in data.dims:
            raise ValueError("data must contain one dimension ending with '_term'")

        if "const" not in data:
            data = data.assign(const=0.0)
        elif not np.issubdtype(data.const, np.floating):
            data = assign_multiindex_safe(data, const=data.const.astype(float))

        (data,) = xr.broadcast(data, exclude=HELPER_DIMS)
        (coeffs_vars,) = xr.broadcast(data[["coeffs", "vars"]], exclude=[FACTOR_DIM])
        coeffs_vars_dict = {str(k): v for k, v in coeffs_vars.items()}
        data = assign_multiindex_safe(data, **coeffs_vars_dict)

        # transpose with new Dataset to really ensure correct order
        data = Dataset(data.transpose(..., TERM_DIM))

        # ensure helper dimensions are not set as coordinates
        if drop_dims := set(HELPER_DIMS).intersection(data.coords):
            # TODO: add a warning here, routines should be safe against this
            data = data.drop_vars(drop_dims)

        if not isinstance(model, Model):
            raise ValueError("model must be an instance of linopy.Model")

        self._model = model
        self._data = data

    def __repr__(self) -> str:
        """
        Print the expression arrays.
        """
        max_lines = options["display_max_rows"]
        dims = list(self.coord_sizes)
        dim_names = self.coord_names
        ndim = len(dims)
        dim_sizes = list(self.coord_sizes.values())
        size = np.prod(dim_sizes)  # that the number of theoretical printouts
        masked_entries = 0  # (~self.mask).sum().values.item() if self.mask
        lines = []

        header_string = self.type

        if size > 1 or ndim > 0:
            for indices in generate_indices_for_printout(dim_sizes, max_lines):
                if indices is None:
                    lines.append("\t\t...")
                else:
                    coord = [
                        self.data.indexes[dims[i]][ind] for i, ind in enumerate(indices)
                    ]
                    if self.mask is None or self.mask.values[indices]:
                        expr = print_single_expression(
                            self.coeffs.values[indices],
                            self.vars.values[indices],
                            self.const.values[indices],
                            self.model,
                        )

                        line = print_coord(coord) + f": {expr}"
                    else:
                        line = print_coord(coord) + ": None"
                    lines.append(line)

            shape_str = ", ".join(f"{d}: {s}" for d, s in zip(dim_names, dim_sizes))
            mask_str = f" - {masked_entries} masked entries" if masked_entries else ""
            underscore = "-" * (len(shape_str) + len(mask_str) + len(header_string) + 4)
            lines.insert(0, f"{header_string} [{shape_str}]{mask_str}:\n{underscore}")
        elif size == 1:
            expr = print_single_expression(
                self.coeffs.values, self.vars.values, self.const.item(), self.model
            )
            lines.append(f"{header_string}\n{'-' * len(header_string)}\n{expr}")
        else:
            lines.append(f"{header_string}\n{'-' * len(header_string)}\n<empty>")

        return "\n".join(lines)

    def print(self, display_max_rows: int = 20, display_max_terms: int = 20) -> None:
        """
        Print the linear expression.

        Parameters
        ----------
        display_max_rows : int
            Maximum number of rows to be displayed.
        display_max_terms : int
            Maximum number of terms to be displayed.
        """
        with options as opts:
            opts.set_value(
                display_max_rows=display_max_rows, display_max_terms=display_max_terms
            )
            print(self)

    @abstractmethod
    def __add__(
        self: GenericExpression, other: SideLike
    ) -> GenericExpression | QuadraticExpression: ...

    @abstractmethod
    def __radd__(self: GenericExpression, other: SideLike) -> GenericExpression: ...

    @abstractmethod
    def __sub__(
        self: GenericExpression, other: SideLike
    ) -> GenericExpression | QuadraticExpression: ...

    @abstractmethod
    def __rsub__(self: GenericExpression, other: SideLike) -> GenericExpression: ...

    @abstractmethod
    def __mul__(
        self: GenericExpression, other: SideLike
    ) -> GenericExpression | QuadraticExpression: ...

    @abstractmethod
    def __rmul__(
        self: GenericExpression, other: SideLike
    ) -> GenericExpression | QuadraticExpression: ...

    @abstractmethod
    def __matmul__(
        self: GenericExpression, other: SideLike
    ) -> GenericExpression | QuadraticExpression: ...

    @abstractmethod
    def __pow__(self, other: int) -> QuadraticExpression: ...

    def __neg__(self: GenericExpression) -> GenericExpression:
        """
        Get the negative of the expression.
        """
        return self.assign_multiindex_safe(coeffs=-self.coeffs, const=-self.const)

    def _multiply_by_linear_expression(
        self, other: LinearExpression | ScalarLinearExpression
    ) -> QuadraticExpression:
        if isinstance(other, ScalarLinearExpression):
            other = other.to_linexpr()

        if other.nterm > 1:
            raise TypeError("Multiplication of multiple terms is not supported.")
        # multiplication: (v1 + c1) * (v2 + c2) = v1 * v2 + c1 * v2 + c2 * v1 + c1 * c2
        # with v being the variables and c the constants
        # merge on factor dimension only returns v1 * v2 + c1 * c2
        ds = other.data[["coeffs", "vars"]].sel(_term=0).broadcast_like(self.data)
        ds = assign_multiindex_safe(ds, const=other.const)
        res = merge([self, ds], dim=FACTOR_DIM, cls=QuadraticExpression)  # type: ignore
        # deal with cross terms c1 * v2 + c2 * v1
        if self.has_constant:
            res = res + self.const * other.reset_const()
        if other.has_constant:
            res = res + self.reset_const() * other.const
        return res

    def _multiply_by_constant(
        self: GenericExpression, other: ConstantLike
    ) -> GenericExpression:
        multiplier = as_dataarray(other, coords=self.coords, dims=self.coord_dims)
        coeffs = self.coeffs * multiplier
        assert all(coeffs.sizes[d] == s for d, s in self.coeffs.sizes.items())
        const = self.const * multiplier
        return self.assign(coeffs=coeffs, const=const)

    def __div__(self: GenericExpression, other: SideLike) -> GenericExpression:
        try:
            if isinstance(
                other,
                variables.Variable
                | variables.ScalarVariable
                | LinearExpression
                | ScalarLinearExpression
                | QuadraticExpression,
            ):
                raise TypeError(
                    "unsupported operand type(s) for /: "
                    f"{type(self)} and {type(other)}"
                    "Non-linear expressions are not yet supported."
                )
            return self._multiply_by_constant(other=1 / other)
        except TypeError:
            return NotImplemented

    def __truediv__(self: GenericExpression, other: SideLike) -> GenericExpression:
        return self.__div__(other)

    def __le__(self, rhs: SideLike) -> Constraint:
        return self.to_constraint(LESS_EQUAL, rhs)

    def __ge__(self, rhs: SideLike) -> Constraint:
        return self.to_constraint(GREATER_EQUAL, rhs)

    def __eq__(self, rhs: SideLike) -> Constraint:  # type: ignore
        return self.to_constraint(EQUAL, rhs)

    def __gt__(self, other: Any) -> NotImplementedType:
        raise NotImplementedError(
            "Inequalities only ever defined for >= rather than >."
        )

    def __lt__(self, other: Any) -> NotImplementedType:
        raise NotImplementedError(
            "Inequalities only ever defined for >= rather than >."
        )

    def add(
        self: GenericExpression, other: SideLike
    ) -> GenericExpression | QuadraticExpression:
        """
        Add an expression to others.
        """
        return self.__add__(other)

    def sub(
        self: GenericExpression, other: SideLike
    ) -> GenericExpression | QuadraticExpression:
        """
        Subtract others from expression.
        """
        return self.__sub__(other)

    def mul(
        self: GenericExpression, other: SideLike
    ) -> GenericExpression | QuadraticExpression:
        """
        Multiply the expr by a factor.
        """
        return self.__mul__(other)

    def div(
        self: GenericExpression, other: VariableLike | ConstantLike
    ) -> GenericExpression | QuadraticExpression:
        """
        Divide the expr by a factor.
        """
        return self.__div__(other)

    def pow(self, other: int) -> QuadraticExpression:
        """
        Power of the expression with a coefficient.
        """
        return self.__pow__(other)

    def dot(
        self: GenericExpression, other: ndarray
    ) -> GenericExpression | QuadraticExpression:
        """
        Matrix multiplication with other, similar to xarray dot.
        """
        return self.__matmul__(other)

    def __getitem__(
        self: GenericExpression, selector: int | tuple[slice, list[int]] | slice
    ) -> GenericExpression:
        """
        Get selection from the expression.
        This is a wrapper around the xarray __getitem__ method. It returns a
        new LinearExpression object with the selected data.
        """
        data = Dataset({k: self.data[k][selector] for k in self.data}, attrs=self.attrs)
        return self.__class__(data, self.model)

    @property
    def attrs(self) -> dict[Any, Any]:
        """
        Get the attributes of the expression
        """
        return self.data.attrs

    @property
    def coords(self) -> DatasetCoordinates:
        """
        Get the coordinates of the expression
        """
        return self.data.coords

    @property
    def indexes(self) -> Indexes:
        """
        Get the indexes of the expression
        """
        return self.data.indexes

    @property
    def sizes(self) -> Frozen:
        """
        Get the sizes of the expression
        """
        return self.data.sizes

    @property
    def ndim(self) -> int:
        """
        Get the number of dimensions.
        """
        return self.const.ndim

    @property
    def loc(self) -> LocIndexer:
        return LocIndexer(self)

    @property
    def type(self) -> str:
        return "LinearExpression"

    @property
    def data(self) -> Dataset:
        return self._data

    @property
    def model(self) -> Model:
        return self._model

    @property
    def dims(self) -> tuple[Hashable, ...]:
        # do explicitly sort as in vars (same as in coeffs)
        return self.vars.dims

    @property
    def coord_dims(self) -> tuple[Hashable, ...]:
        return tuple(k for k in self.dims if k not in HELPER_DIMS)

    @property
    def coord_sizes(self) -> dict[Hashable, int]:
        return {k: v for k, v in self.sizes.items() if k not in HELPER_DIMS}

    @property
    def coord_names(self) -> list[str]:
        return get_dims_with_index_levels(self.data, self.coord_dims)

    @property
    def vars(self) -> DataArray:
        return self.data.vars

    @vars.setter
    def vars(self, value: DataArray) -> None:
        self._data = assign_multiindex_safe(self.data, vars=value)

    @property
    def coeffs(self) -> DataArray:
        return self.data.coeffs

    @coeffs.setter
    def coeffs(self, value: DataArray) -> None:
        self._data = assign_multiindex_safe(self.data, coeffs=value)

    @property
    def const(self) -> DataArray:
        return self.data.const

    @const.setter
    def const(self, value: DataArray) -> None:
        self._data = assign_multiindex_safe(self.data, const=value)

    @property
    def has_constant(self) -> DataArray:
        return self.const.any()

    # create a dummy for a mask, which can be implemented later
    @property
    def mask(self) -> None:
        return None

    @has_optimized_model
    def _map_solution(self) -> DataArray:
        """
        Replace variable labels by solution values.
        """
        m = self.model
        sol = pd.Series(m.matrices.sol, m.matrices.vlabels)
        sol[-1] = np.nan
        idx = np.ravel(self.vars)
        values = sol[idx].to_numpy().reshape(self.vars.shape)
        return xr.DataArray(values, dims=self.vars.dims, coords=self.vars.coords)

    @property
    def solution(self) -> DataArray:
        """
        Get the optimal values of the expression.

        The function raises an error in case no model is set as a
        reference or the model is not optimized.
        """
        vals = self._map_solution()
        sol = (self.coeffs * vals).sum(TERM_DIM) + self.const
        return sol.rename("solution")

    def sum(
        self: GenericExpression,
        dim: DimsLike | None = None,
        drop_zeros: bool = False,
        **kwargs: Any,
    ) -> GenericExpression:
        """
        Sum the expression over all or a subset of dimensions.

        This stack all terms of the dimensions, that are summed over, together.

        Parameters
        ----------
        dim : str/list, optional
            Dimension(s) to sum over. The default is None which results in all
            dimensions.
        dims : str/list, optional
            Deprecated. Use ``dim`` instead.

        Returns
        -------
        linopy.LinearExpression
            Summed expression.
        """
        if dim is None and "dims" in kwargs:
            dim = kwargs.pop("dims")
            warn(
                "The `dims` argument in `.sum` is deprecated. Use `dim` instead.",
                DeprecationWarning,
            )
        if kwargs:
            raise ValueError(f"Unknown keyword argument(s): {kwargs}")

        res = self.__class__(self._sum(self, dim=dim), self.model)

        if drop_zeros:
            res = res.densify_terms()

        return res

    def cumsum(
        self,
        dim: DimsLike | None = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> LinearExpression:
        """
        Cumulated sum along a given axis.

        Docstring and arguments are borrowed from `xarray.Dataset.cumsum`

        Parameters
        ----------
        dim : str, Iterable of Hashable, "..." or None, default: None
            Name of dimension[s] along which to apply ``cumsum``. For e.g. ``dim="x"``
            or ``dim=["x", "y"]``. If "..." or None, will reduce over all dimensions.
        skipna : bool or None, optional
            If True, skip missing values (as marked by NaN). By default, only
            skips missing values for float dtypes; other dtypes either do not
            have a sentinel missing value (int) or ``skipna=True`` has not been
            implemented (object, datetime64 or timedelta64).
        keep_attrs : bool or None, optional
            If True, ``attrs`` will be copied from the original
            object to the new one.  If False, the new object will be
            returned without attributes.
        **kwargs : Any
            Additional keyword arguments passed on to the appropriate array
            function for calculating ``cumsum`` on this object's data.
            These could include dask-specific kwargs like ``split_every``.

        Returns
        -------
        linopy.expression.LinearExpression
        """
        # Along every dimensions, we want to perform cumsum along, get the size of the
        # dimension to pass that to self.rolling.
        if not dim:
            # If user did not specify a dimension to sum over, use all relevant
            # dimensions
            dim = self.coord_dims
        if isinstance(dim, str):
            dim = [dim]
        elif isinstance(dim, EllipsisType) or dim is None:
            dim = self.coord_dims
        dim_dict = {dim_name: self.data.sizes[dim_name] for dim_name in dim}
        return self.rolling(dim=dim_dict).sum(keep_attrs=keep_attrs, skipna=skipna)

    def to_constraint(
        self, sign: SignLike, rhs: ConstantLike | VariableLike | ExpressionLike
    ) -> Constraint:
        """
        Convert a linear expression to a constraint.

        Parameters
        ----------
        sign : str, array-like
            Sign(s) of the constraints.
        rhs : constant, Variable, LinearExpression
            Right-hand side of the constraint.

        Returns
        -------
        Constraint with strict separation of the linear expressions of variables
        which are moved to the left-hand-side and constant values which are moved
        to the right-hand side.
        """
        all_to_lhs = (self - rhs).data
        data = assign_multiindex_safe(
            all_to_lhs[["coeffs", "vars"]], sign=sign, rhs=-all_to_lhs.const
        )
        return constraints.Constraint(data, model=self.model)

    def reset_const(self: GenericExpression) -> GenericExpression:
        """
        Reset the constant of the linear expression to zero.
        """
        return self.__class__(self.data[["coeffs", "vars"]], self.model)

    def isnull(self) -> DataArray:
        """
        Get a boolean mask with true values where there is only missing values in an expression.

        Returns
        -------
        xr.DataArray
        """
        helper_dims = set(self.vars.dims).intersection(HELPER_DIMS)
        return (self.vars == -1).all(helper_dims) & self.const.isnull()

    def where(
        self: GenericExpression,
        cond: DataArray,
        other: LinearExpression
        | int
        | DataArray
        | dict[str, float | int | DataArray]
        | None = None,
        **kwargs: Any,
    ) -> GenericExpression:
        """
        Filter variables based on a condition.

        This operation call ``xarray.Dataset.where`` but sets the default
        fill value to -1 for variables and ensures preserving the linopy.LinearExpression type.

        Parameters
        ----------
        cond : DataArray or callable
            Locations at which to preserve this object's values. dtype must be `bool`.
            If a callable, it must expect this object as its only parameter.
        other : expression-like, DataArray or scalar, optional
            Data to use in place of values where cond is False.
            If a DataArray or a scalar is provided, it is only used to fill
            the missing values of constant values (`const`).
            If a DataArray, its coordinates must match this object's.
        **kwargs :
            Keyword arguments passed to ``xarray.Dataset.where``

        Returns
        -------
        linopy.LinearExpression or linopy.QuadraticExpression
        """
        # Cannot set `other` if drop=True
        _other: dict[str, float] | dict[str, int | float | DataArray] | DataArray | None
        if other is None or other is np.nan:
            if not kwargs.get("drop", False):
                _other = FILL_VALUE
            else:
                _other = None
        elif isinstance(other, int | float | DataArray):
            _other = {**self._fill_value, "const": other}
        else:
            _other = _expr_unwrap(other)
        cond = _expr_unwrap(cond)
        if isinstance(cond, DataArray):
            if helper_dims := set(HELPER_DIMS).intersection(cond.dims):
                raise ValueError(
                    f"Filtering by a DataArray with a helper dimension(s) ({helper_dims!r}) is not supported."
                )
        return self.__class__(self.data.where(cond, other=_other, **kwargs), self.model)

    def fillna(
        self: GenericExpression,
        value: int
        | float
        | DataArray
        | Dataset
        | LinearExpression
        | dict[str, float | int | DataArray],
    ) -> GenericExpression:
        """
        Fill missing values with a given value.

        This method fills missing values in the data with a given value. It calls the `fillna` method of the underlying
        `xarray.Dataset` object, but sets the default fill value to -1 for variables and ensures that the output is of
        type `linopy.LinearExpression`.

        Parameters
        ----------
        value : scalar or array_like
            Value(s) to use to fill missing values. If a scalar is provided, it will be used to fill all missing values as a constant.
            If an array-like object is provided, it should have the same shape as the data and will be used to fill missing values element-wise as a constant.

        Returns
        -------
        linopy.LinearExpression or linopy.QuadraticExpression
            A new object with missing values filled with the given value.
        """
        value = _expr_unwrap(value)
        if isinstance(value, DataArray | np.floating | np.integer | int | float):
            value = {"const": value}
        return self.__class__(self.data.fillna(value), self.model)

    def diff(self: GenericExpression, dim: str, n: int = 1) -> GenericExpression:
        """
        Calculate the n-th order discrete difference along given axis.

        This operation call ``xarray.Dataset.diff`` but ensures preserving the
        linopy.LinearExpression type.

        Parameters
        ----------
        dim : str
            Dimension over which to calculate the finite difference.
        n : int, optional
            The number of times values are differenced.

        Returns
        -------
        linopy.LinearExpression or linopy.QuadraticExpression
        """
        return self - self.shift({dim: n})

    def groupby(
        self,
        group: DataFrame | Series | DataArray,
        restore_coord_dims: bool | None = None,
        **kwargs: Any,
    ) -> LinearExpressionGroupby:
        """
        Returns a LinearExpressionGroupBy object for performing grouped
        operations.

        Docstring and arguments are borrowed from `xarray.Dataset.groupby`

        Parameters
        ----------
        group : str, DataArray or IndexVariable
            Array whose unique values should be used to group this array. If a
            string, must be the name of a variable contained in this dataset.
        restore_coord_dims : bool, optional
            If True, also restore the dimension order of multi-dimensional
            coordinates.

        Returns
        -------
        grouped
            A `LinearExpressionGroupBy` containing the xarray groups and ensuring
            the correct return type.
        """
        ds = self.data
        kwargs = dict(restore_coord_dims=restore_coord_dims, **kwargs)
        return LinearExpressionGroupby(ds, group, model=self.model, kwargs=kwargs)

    def rolling(
        self,
        dim: Mapping[Any, int] | None = None,
        min_periods: int | None = None,
        center: bool | Mapping[Any, bool] = False,
        **window_kwargs: int,
    ) -> LinearExpressionRolling:
        """
        Rolling window object.

        Docstring and arguments are borrowed from `xarray.Dataset.rolling`

        Parameters
        ----------
        dim : dict, optional
            Mapping from the dimension name to create the rolling iterator
            along (e.g. `time`) to its moving window size.
        min_periods : int, default: None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). The default, None, is equivalent to
            setting min_periods equal to the size of the window.
        center : bool or mapping, default: False
            Set the labels at the center of the window.
        **window_kwargs : optional
            The keyword arguments form of ``dim``.
            One of dim or window_kwargs must be provided.

        Returns
        -------
        linopy.expression.LinearExpressionRolling
        """
        ds = self.data
        rolling = ds.rolling(
            dim=dim, min_periods=min_periods, center=center, **window_kwargs
        )
        return LinearExpressionRolling(rolling, model=self.model)

    @property
    def nterm(self) -> int:
        """
        Get the number of terms in the linear expression.
        """
        return len(self.data._term)

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Get the total shape of the linear expression.
        """
        assert self.vars.shape == self.coeffs.shape
        return self.vars.shape

    @property
    def size(self) -> int:
        """
        Get the total size of the linear expression.
        """
        return self.vars.size

    @property
    def empty(self) -> EmptyDeprecationWrapper:
        """
        Get whether the linear expression is empty.

        .. versionchanged:: 0.5.1
           Returns a EmptyDeprecationWrapper which behaves like a bool and emits
           a DeprecationWarning when called.
        """
        return EmptyDeprecationWrapper(not self.size)

    def densify_terms(self: GenericExpression) -> GenericExpression:
        """
        Move all non-zero term entries to the front and cut off all-zero
        entries in the term-axis.
        """
        data = self.data.transpose(..., TERM_DIM)

        cdata = data.coeffs.data
        axis = cdata.ndim - 1
        nnz = np.nonzero(cdata)
        nterm = (cdata != 0).sum(axis).max()

        mod_nnz = list(nnz)
        mod_nnz.pop(axis)

        remaining_axes = np.vstack(mod_nnz).T
        _, idx_ = np.unique(remaining_axes, axis=0, return_inverse=True)
        idx = list(idx_)
        new_index = np.array([idx[:i].count(j) for i, j in enumerate(idx)])
        mod_nnz.insert(axis, new_index)

        vdata = np.full_like(cdata, -1)
        vdata[tuple(mod_nnz)] = data.vars.data[nnz]
        data.vars.data = vdata

        cdata = np.zeros_like(cdata)
        cdata[tuple(mod_nnz)] = data.coeffs.data[nnz]
        data.coeffs.data = cdata

        return self.__class__(data.sel({TERM_DIM: slice(0, nterm)}), self.model)

    def sanitize(self: GenericExpression) -> GenericExpression:
        """
        Sanitize LinearExpression by ensuring int dtype for variables.

        Returns
        -------
        linopy.LinearExpression
        """
        if not np.issubdtype(self.vars.dtype, np.integer):
            return self.assign(vars=self.vars.fillna(-1).astype(int))

        return self

    def equals(self, other: BaseExpression) -> bool:
        return self.data.equals(_expr_unwrap(other))

    def __iter__(self) -> Iterator[Hashable]:
        return self.data.__iter__()

    @classmethod
    def _sum(
        cls,
        expr: BaseExpression | Dataset,
        dim: DimsLike | None = None,
    ) -> Dataset:
        data = _expr_unwrap(expr)
        if cls is QuadraticExpression:
            dim = dim or list(set(data.dims) - set(HELPER_DIMS))

        if isinstance(dim, str):
            dim = [dim]
        elif isinstance(dim, EllipsisType):
            dim = None

        if dim is None:
            vars = DataArray(data.vars.data.ravel(), dims=TERM_DIM)
            coeffs = DataArray(data.coeffs.data.ravel(), dims=TERM_DIM)
            const = data.const.sum()
            ds = xr.Dataset({"vars": vars, "coeffs": coeffs, "const": const})
        else:
            dim = [d for d in dim if d != TERM_DIM]
            ds = (
                data[["coeffs", "vars"]]
                .reset_index(dim, drop=True)
                .rename({TERM_DIM: STACKED_TERM_DIM})
                .stack({TERM_DIM: [STACKED_TERM_DIM] + dim}, create_index=False)
            )
            ds = assign_multiindex_safe(ds, const=data.const.sum(dim))

        return ds

    # Wrapped function which would convert variable to dataarray
    assign = exprwrap(assign_multiindex_safe)

    assign_multiindex_safe = exprwrap(assign_multiindex_safe)

    assign_attrs = exprwrap(Dataset.assign_attrs)

    assign_coords = exprwrap(Dataset.assign_coords)

    astype = exprwrap(Dataset.astype)

    bfill = exprwrap(Dataset.bfill)

    broadcast_like = exprwrap(Dataset.broadcast_like)

    chunk = exprwrap(Dataset.chunk)

    drop = exprwrap(Dataset.drop)

    drop_vars = exprwrap(Dataset.drop_vars)

    drop_sel = exprwrap(Dataset.drop_sel)

    drop_isel = exprwrap(Dataset.drop_isel)

    expand_dims = exprwrap(Dataset.expand_dims)

    ffill = exprwrap(Dataset.ffill)

    sel = exprwrap(Dataset.sel)

    isel = exprwrap(Dataset.isel)

    shift = exprwrap(Dataset.shift)

    swap_dims = exprwrap(Dataset.swap_dims)

    set_index = exprwrap(Dataset.set_index)

    reindex = exprwrap(Dataset.reindex, fill_value=_fill_value)

    reindex_like = exprwrap(Dataset.reindex_like, fill_value=_fill_value)

    rename = exprwrap(Dataset.rename)

    reset_index = exprwrap(Dataset.reset_index)

    rename_dims = exprwrap(Dataset.rename_dims)

    roll = exprwrap(Dataset.roll)

    stack = exprwrap(Dataset.stack)

    unstack = exprwrap(Dataset.unstack)

    iterate_slices = iterate_slices


GenericExpression = TypeVar("GenericExpression", bound=BaseExpression)


class LinearExpression(BaseExpression):
    """
    A linear expression consisting of terms of coefficients and variables.

    The LinearExpression class is a subclass of xarray.Dataset which allows to
    apply most xarray functions on it. However most arithmetic operations are
    overwritten. Like this you can easily expand and modify the linear
    expression.

    Examples
    --------
    >>> from linopy import Model
    >>> import pandas as pd
    >>> m = Model()
    >>> x = m.add_variables(pd.Series([0, 0]), 1, name="x")
    >>> y = m.add_variables(4, pd.Series([8, 10]), name="y")

    Combining expressions:

    >>> expr = 3 * x
    >>> type(expr)
    <class 'linopy.expressions.LinearExpression'>

    >>> other = 4 * y
    >>> type(expr + other)
    <class 'linopy.expressions.LinearExpression'>

    Multiplying:

    >>> type(3 * expr)
    <class 'linopy.expressions.LinearExpression'>

    Summation over dimensions

    >>> type(expr.sum(dim="dim_0"))
    <class 'linopy.expressions.LinearExpression'>
    """

    @overload
    def __add__(
        self,
        other: ConstantLike | VariableLike | ScalarLinearExpression | LinearExpression,
    ) -> LinearExpression: ...

    @overload
    def __add__(self, other: QuadraticExpression) -> QuadraticExpression: ...

    def __add__(
        self,
        other: ConstantLike
        | VariableLike
        | ScalarLinearExpression
        | LinearExpression
        | QuadraticExpression,
    ) -> LinearExpression | QuadraticExpression:
        """
        Add an expression to others.

        Note: If other is a numpy array or pandas object without axes names,
        dimension names of self will be filled in other
        """
        if isinstance(other, QuadraticExpression):
            return other.__add__(self)

        try:
            if np.isscalar(other):
                return self.assign(const=self.const + other)

            other = as_expression(other, model=self.model, dims=self.coord_dims)
            return merge([self, other], cls=self.__class__)
        except TypeError:
            return NotImplemented

    def __radd__(self, other: ConstantLike) -> LinearExpression:
        try:
            return self + other
        except TypeError:
            return NotImplemented

    @overload
    def __sub__(
        self,
        other: ConstantLike | VariableLike | ScalarLinearExpression | LinearExpression,
    ) -> LinearExpression: ...

    @overload
    def __sub__(self, other: QuadraticExpression) -> QuadraticExpression: ...

    def __sub__(
        self,
        other: ConstantLike
        | VariableLike
        | ScalarLinearExpression
        | LinearExpression
        | QuadraticExpression,
    ) -> LinearExpression | QuadraticExpression:
        try:
            return self.__add__(-other)
        except TypeError:
            return NotImplemented

    def __rsub__(self, other: ConstantLike | Variable) -> LinearExpression:
        try:
            return (self * -1) + other
        except TypeError:
            return NotImplemented

    @overload
    def __mul__(self, other: ConstantLike) -> LinearExpression: ...

    @overload
    def __mul__(self, other: VariableLike | ExpressionLike) -> QuadraticExpression: ...

    def __mul__(
        self,
        other: SideLike,
    ) -> LinearExpression | QuadraticExpression:
        """
        Multiply the expr by a factor.
        """
        if isinstance(other, QuadraticExpression):
            return other.__rmul__(self)

        try:
            if isinstance(other, variables.Variable | variables.ScalarVariable):
                other = other.to_linexpr()

            if isinstance(other, LinearExpression | ScalarLinearExpression):
                return self._multiply_by_linear_expression(other)
            else:
                return self._multiply_by_constant(other)
        except TypeError:
            return NotImplemented

    def __pow__(self, other: int) -> QuadraticExpression:
        """
        Power of the expression with a coefficient. The only coefficient allowed is 2.
        """
        if not other == 2:
            raise ValueError("Power must be 2.")
        return self * self  # type: ignore

    def __rmul__(self, other: ConstantLike) -> LinearExpression:
        """
        Right-multiply the expr by a factor.
        """
        try:
            return self * other
        except TypeError:
            return NotImplemented

    @overload
    def __matmul__(self, other: ConstantLike) -> LinearExpression: ...

    @overload
    def __matmul__(
        self, other: VariableLike | ExpressionLike
    ) -> QuadraticExpression: ...

    def __matmul__(
        self, other: ConstantLike | VariableLike | ExpressionLike
    ) -> LinearExpression | QuadraticExpression:
        """
        Matrix multiplication with other, similar to xarray dot.
        """
        if not isinstance(other, LinearExpression | variables.Variable):
            other = as_dataarray(other, coords=self.coords, dims=self.coord_dims)

        common_dims = list(set(self.coord_dims).intersection(other.dims))
        return (self * other).sum(dim=common_dims)

    @property
    def flat(self) -> pd.DataFrame:
        """
        Convert the expression to a pandas DataFrame.

        The resulting DataFrame represents a long table format of the all
        expressions with non-zero coefficients. It contains the
        columns `coeffs` and `vars`.

        Returns
        -------
        df : pandas.DataFrame
        """
        ds = self.data

        def mask_func(data: pd.DataFrame) -> pd.Series:
            mask = (data["vars"] != -1) & (data["coeffs"] != 0)
            return mask

        df = to_dataframe(ds, mask_func=mask_func)
        df = df.groupby("vars", as_index=False).sum()
        check_has_nulls(df, name=self.type)
        return df

    def to_quadexpr(self) -> QuadraticExpression:
        """Convert LinearExpression to QuadraticExpression."""
        vars = self.data.vars.expand_dims(FACTOR_DIM)
        fill_value = self._fill_value["vars"]
        vars = xr.concat([vars, xr.full_like(vars, fill_value)], dim=FACTOR_DIM)
        data = self.data.assign(vars=vars)
        return QuadraticExpression(data, self.model)

    def to_polars(self) -> pl.DataFrame:
        """
        Convert the expression to a polars DataFrame.

        The resulting DataFrame represents a long table format of the all
        non-masked expressions with non-zero coefficients. It contains the
        columns `coeffs`, `vars`.

        Returns
        -------
        df : polars.DataFrame
        """
        df = to_polars(self.data)
        df = filter_nulls_polars(df)
        df = group_terms_polars(df)
        check_has_nulls_polars(df, name=self.type)
        return df

    @classmethod
    def _from_scalarexpression_list(
        cls,
        exprs: list[ScalarLinearExpression],
        coords: Mapping,
        model: Model,
    ) -> LinearExpression:
        """
        Create a LinearExpression from a list of lists with different lengths.
        """
        shape = list(map(len, coords.values()))

        coeffs = array(tuple(zip_longest(*(e.coeffs for e in exprs), fillvalue=nan)))
        vars = array(tuple(zip_longest(*(e.vars for e in exprs), fillvalue=-1)))

        nterm = vars.shape[0]
        coeffs = coeffs.reshape((nterm, *shape))
        vars = vars.reshape((nterm, *shape))

        coeffdata = DataArray(coeffs, coords, dims=(TERM_DIM, *coords))
        vardata = DataArray(vars, coords, dims=(TERM_DIM, *coords))
        ds = Dataset({"coeffs": coeffdata, "vars": vardata}).transpose(..., TERM_DIM)

        return cls(ds, model)

    @classmethod
    def from_rule(
        cls,
        model: Model,
        rule: Callable,
        coords: Sequence[Sequence | pd.Index | DataArray] | Mapping | None = None,
    ) -> LinearExpression:
        """
        Create a linear expression from a rule and a set of coordinates.

        This functionality mirrors the assignment of linear expression as done by
        Pyomo.


        Parameters
        ----------
        model : linopy.Model
            Passed to function `rule` as a first argument.
        rule : callable
            Function to be called for each combinations in `coords`.
            The first argument of the function is the underlying `linopy.Model`.
            The following arguments are given by the coordinates for accessing
            the variables. The function has to return a
            `ScalarLinearExpression`. Therefore use the `.at` accessor when
            indexing variables.
        coords : coordinate-like
            Coordinates to processed by `xarray.DataArray`.
            For each combination of coordinates, the function
            given by `rule` is called. The order and size of coords has
            to be same as the argument list followed by `model` in
            function `rule`.


        Returns
        -------
        linopy.LinearExpression

        Examples
        --------
        >>> from linopy import Model, LinearExpression
        >>> m = Model()
        >>> coords = pd.RangeIndex(10), ["a", "b"]
        >>> x = m.add_variables(0, 100, coords)
        >>> def bound(m, i, j):
        ...     if i % 2:
        ...         return (i - 1) * x.at[i - 1, j]
        ...     else:
        ...         return i * x.at[i, j]
        ...
        >>> expr = LinearExpression.from_rule(m, bound, coords)
        >>> con = m.add_constraints(expr <= 10)
        """
        if not isinstance(coords, DataArrayCoordinates):
            coords = DataArray(coords=coords).coords

        # test output type
        output = rule(model, *[c.values[0] for c in coords.values()])
        if not isinstance(output, ScalarLinearExpression) and output is not None:
            msg = f"`rule` has to return ScalarLinearExpression not {type(output)}."
            raise TypeError(msg)

        combinations = product(*[c.values for c in coords.values()])

        placeholder = ScalarLinearExpression((np.nan,), (-1,), model)
        exprs = [rule(model, *coord) or placeholder for coord in combinations]
        return cls._from_scalarexpression_list(exprs, coords, model)

    @classmethod
    def from_tuples(
        cls,
        *tuples: tuple[ConstantLike, str | Variable | ScalarVariable] | ConstantLike,
        model: Model | None = None,
    ) -> LinearExpression:
        """
        Create a linear expression by using tuples of coefficients and
        variables.

        The function internally checks that all variables in the tuples belong to the same
        reference model.

        Parameters
        ----------
        tuples : A list of elements. Each element is either:
            * (coefficients, variables)
            * constant

            Each (coefficients, variables) tuple represents one term in the resulting linear expression,
            which can possibly span over multiple dimensions:

            * coefficients : int/float/array_like
                The coefficient(s) in the term, if the coefficients array
                contains dimensions which do not appear in
                the variables, the variables are broadcasted.
            * variables : str/array_like/linopy.Variable
                The variable(s) going into the term. These may be referenced
                by name.
            * constant: int/float/array_like
                The constant value to add to the expression
        model : The linopy.Model. If None this can be inferred from the provided variables

        Returns
        -------
        linopy.LinearExpression

        Examples
        --------
        >>> from linopy import Model
        >>> import pandas as pd
        >>> m = Model()
        >>> x = m.add_variables(pd.Series([0, 0]), 1)
        >>> y = m.add_variables(4, pd.Series([8, 10]))
        >>> expr = LinearExpression.from_tuples((10, x), (1, y), 1)

        This is the same as calling ``10*x + y`` + 1 but a bit more performant.
        """

        def process_one(
            t: tuple[ConstantLike, str | Variable | ScalarVariable]
            | tuple[ConstantLike]
            | ConstantLike,
        ) -> LinearExpression:
            nonlocal model

            if isinstance(t, SUPPORTED_CONSTANT_TYPES):
                if model is None:
                    raise ValueError("Model must be provided when using constants.")
                expr = LinearExpression(t, model)
                return expr

            if not isinstance(t, tuple):
                raise ValueError("Expected tuple or constant.")

            if len(t) == 2:
                # assume first element is coefficient and second is variable
                c, v = t
                if isinstance(v, variables.ScalarVariable):
                    if not isinstance(c, int | float):
                        raise TypeError(
                            "Expected int or float as coefficient of scalar variable (first element of tuple)."
                        )
                    expr = v.to_linexpr(c)
                elif isinstance(v, variables.Variable):
                    if not isinstance(c, SUPPORTED_CONSTANT_TYPES):
                        raise TypeError(
                            "Expected constant as coefficient of variable (first element of tuple)."
                        )
                    expr = v.to_linexpr(c)
                else:
                    raise TypeError("Expected variable as second element of tuple.")

                if model is None:
                    model = expr.model  # TODO: Ensure equality of models
                return expr

            if len(t) == 1:
                warn(
                    "Passing a single value tuple to LinearExpression.from_tuples is deprecated",
                    DeprecationWarning,
                )
                # assume that the element is a constant
                const = as_dataarray(t[0])
                if model is None:
                    raise ValueError("Model must be provided when using constants.")
                return LinearExpression(const, model)
            else:
                raise ValueError("Expected tuples of length 2")

        exprs = [process_one(t) for t in tuples]

        return merge(exprs, cls=cls) if len(exprs) > 1 else exprs[0]


class QuadraticExpression(BaseExpression):
    """
    A quadratic expression consisting of terms of coefficients and variables.

    The QuadraticExpression class is a subclass of LinearExpression which allows to
    apply most xarray functions on it.
    """

    __slots__ = ("_data", "_model")
    __array_ufunc__ = None
    __array_priority__ = 10000
    __pandas_priority__ = 10000

    _fill_value = {"vars": -1, "coeffs": np.nan, "const": np.nan}

    def __init__(self, data: Dataset | None, model: Model) -> None:
        super().__init__(data, model)

        if data is None:
            da = xr.DataArray([[], []], dims=[FACTOR_DIM, TERM_DIM])
            data = Dataset({"coeffs": da, "vars": da, "const": 0})
        if FACTOR_DIM not in data.vars.dims:
            raise ValueError(f"Data does not include dimension {FACTOR_DIM}")
        elif data.sizes[FACTOR_DIM] != 2:
            raise ValueError(f"Size of dimension {FACTOR_DIM} must be 2.")

        # transpose data to have _term as last dimension and _factor as second last
        data = xr.Dataset(data.transpose(..., FACTOR_DIM, TERM_DIM))
        self._data = data

    @property
    def type(self) -> str:
        return "QuadraticExpression"

    @property
    def shape(self) -> tuple[int, ...]:
        # TODO Implement this
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support shape property"
        )

    def __mul__(self, other: SideLike) -> QuadraticExpression:
        """
        Multiply the expr by a factor.
        """
        if isinstance(
            other,
            BaseExpression
            | ScalarLinearExpression
            | variables.Variable
            | variables.ScalarVariable,
        ):
            raise TypeError(
                "unsupported operand type(s) for *: "
                f"{type(self)} and {type(other)}. "
                "Higher order non-linear expressions are not yet supported."
            )
        try:
            return self._multiply_by_constant(other)
        except TypeError:
            return NotImplemented

    def __rmul__(self, other: SideLike) -> QuadraticExpression:
        return self * other

    def __add__(self, other: SideLike) -> QuadraticExpression:
        """
        Add an expression to others.

        Note: If other is a numpy array or pandas object without axes names,
        dimension names of self will be filled in other
        """
        try:
            if np.isscalar(other):
                return self.assign(const=self.const + other)

            other = as_expression(other, model=self.model, dims=self.coord_dims)

            if isinstance(other, LinearExpression):
                other = other.to_quadexpr()

            return merge([self, other], cls=self.__class__)
        except TypeError:
            return NotImplemented

    def __radd__(self, other: ConstantLike) -> QuadraticExpression:
        """
        Add others to expression.
        """
        return self.__add__(other)

    def __sub__(self, other: SideLike) -> QuadraticExpression:
        """
        Subtract others from expression.

        Note: If other is a numpy array or pandas object without axes names,
        dimension names of self will be filled in other
        """
        try:
            if np.isscalar(other):
                return self.assign(const=self.const - other)

            other = as_expression(other, model=self.model, dims=self.coord_dims)
            if type(other) is LinearExpression:
                other = other.to_quadexpr()
            return merge([self, -other], cls=self.__class__)
        except TypeError:
            return NotImplemented

    def __rsub__(self, other: SideLike) -> QuadraticExpression:
        """
        Subtract expression from others.
        """
        try:
            return (self * -1) + other
        except TypeError:
            return NotImplemented

    def __pow__(self, other: SideLike) -> QuadraticExpression:
        raise TypeError("Higher order non-linear expressions are not yet supported.")

    def __matmul__(
        self, other: ConstantLike | VariableLike | ExpressionLike
    ) -> QuadraticExpression:
        """
        Matrix multiplication with other, similar to xarray dot.
        """
        if isinstance(
            other,
            BaseExpression
            | ScalarLinearExpression
            | variables.Variable
            | variables.ScalarVariable,
        ):
            raise TypeError(
                "Higher order non-linear expressions are not yet supported."
            )

        other = as_dataarray(other, coords=self.coords, dims=self.coord_dims)
        common_dims = list(set(self.coord_dims).intersection(other.dims))
        return (self * other).sum(dim=common_dims)

    @property
    def solution(self) -> DataArray:
        """
        Get the optimal values of the expression.

        The function raises an error in case no model is set as a
        reference or the model is not optimized.
        """
        vals = self._map_solution()
        sol = (self.coeffs * vals.prod(FACTOR_DIM)).sum(TERM_DIM) + self.const
        return sol.rename("solution")

    def to_constraint(self, sign: SignLike, rhs: SideLike) -> NotImplementedType:
        raise NotImplementedError(
            "Quadratic expressions cannot be used in constraints."
        )

    @property
    def flat(self) -> DataFrame:
        """
        Return a flattened expression.
        """
        vars = self.data.vars.assign_coords(
            {FACTOR_DIM: ["vars1", "vars2"]}
        ).to_dataset(FACTOR_DIM)
        ds = self.data.drop_vars("vars").assign(vars)

        def mask_func(data: pd.DataFrame) -> pd.Series:
            mask = ((data["vars1"] != -1) | (data["vars2"] != -1)) & (
                data["coeffs"] != 0
            )
            return mask

        df = to_dataframe(ds, mask_func=mask_func)
        # Group repeated variables in the same constraint
        df = df.groupby(["vars1", "vars2"], as_index=False).sum()
        check_has_nulls(df, name=self.type)
        return df

    def to_polars(self, **kwargs: Any) -> pl.DataFrame:
        """
        Convert the expression to a polars DataFrame.

        The resulting DataFrame represents a long table format of the all
        non-masked expressions with non-zero coefficients. It contains the
        columns `coeffs`, `vars`.

        Returns
        -------
        df : polars.DataFrame
        """
        vars = self.data.vars.assign_coords(
            {FACTOR_DIM: ["vars1", "vars2"]}
        ).to_dataset(FACTOR_DIM)
        ds = self.data.drop_vars("vars").assign(vars)
        df = to_polars(ds, **kwargs)
        df = filter_nulls_polars(df)
        df = group_terms_polars(df)
        check_has_nulls_polars(df, name=self.type)
        return df

    def to_matrix(self) -> scipy.sparse._csc.csc_matrix:
        """
        Return a sparse matrix representation of the expression only including
        quadratic terms.

        Note that the matrix is formulated following the convention of the
        optimization problem, i.e. the quadratic term is 0.5 x^T Q x.
        The matrix Q is therefore symmetric and the diagonal terms are doubled.

        """
        df = self.flat
        # drop linear terms
        df = df[(df.vars1 != -1) & (df.vars2 != -1)]

        # symmetrize cross terms and double diagonal terms
        cross_terms = df.vars1 != df.vars2
        df.loc[~cross_terms, "coeffs"] *= 2
        vals = dict(vars1=df.vars2[cross_terms], vars2=df.vars1[cross_terms])
        df = pd.concat([df, df[cross_terms].assign(**vals)])

        # assign matrix
        data = df.coeffs
        row = df.vars1
        col = df.vars2
        nvars = self.model.shape[1]
        return csc_matrix((data, (row, col)), shape=(nvars, nvars))


def as_expression(
    obj: Any, model: Model, **kwargs: Any
) -> LinearExpression | QuadraticExpression:
    """
    Convert an object to a LinearExpression or QuadraticExpression.

    Parameters
    ----------
    obj : Variable, ScalarVariable, LinearExpression, array_like
        Object to convert to LinearExpression.
    model : linopy.Model, optional
        Assigned model, by default None
    **kwargs :
        Keyword arguments passed to `linopy.as_dataarray`.

    Returns
    -------
    expr : LinearExpression

    Raises
    ------
    ValueError
        If object cannot be converted to LinearExpression.
    """
    if isinstance(obj, LinearExpression | QuadraticExpression):
        return obj
    elif isinstance(obj, variables.Variable | variables.ScalarVariable):
        return obj.to_linexpr()
    else:
        try:
            obj = as_dataarray(obj, **kwargs)
        except ValueError as e:
            raise ValueError("Cannot convert to LinearExpression") from e
        return LinearExpression(obj, model)


def merge(
    exprs: Sequence[
        LinearExpression | QuadraticExpression | variables.Variable | Dataset
    ],
    *add_exprs: tuple[
        LinearExpression | QuadraticExpression | variables.Variable | Dataset
    ],
    dim: str = TERM_DIM,
    cls: type[GenericExpression] = None,  # type: ignore
    **kwargs: Any,
) -> GenericExpression:
    """
    Merge multiple expression together.

    This function is a bit faster than summing over multiple linear expressions.
    In case a list of LinearExpression with exactly the same shape is passed
    and the dimension to concatenate on is TERM_DIM, the concatenation uses
    the coordinates of the first object as a basis which overrides the
    coordinates of the consecutive objects.

    Parameters
    ----------
    *exprs : tuple/list
        List of linear expressions to merge.
    dim : str
        Dimension along which the expressions should be concatenated.
    cls : type
        Explicitly set the type of the resulting expression (So that the type checker will know the return type)
    **kwargs
        Additional keyword arguments passed to xarray.concat. Defaults to
        {coords: "minimal", compat: "override"} or, in the special case described
        above, to {coords: "minimal", compat: "override", "join": "override"}.

    Returns
    -------
    res : linopy.LinearExpression or linopy.QuadraticExpression
    """
    if not isinstance(exprs, list) and len(add_exprs):
        warn(
            "Passing a tuple to the merge function is deprecated. Please pass a list of objects to be merged",
            DeprecationWarning,
        )
        exprs = [exprs] + list(add_exprs)  # type: ignore

    has_quad_expression = any(type(e) is QuadraticExpression for e in exprs)
    has_linear_expression = any(type(e) is LinearExpression for e in exprs)
    if cls is None:
        cls = QuadraticExpression if has_quad_expression else LinearExpression

    if cls is QuadraticExpression and dim == TERM_DIM and has_linear_expression:
        raise ValueError(
            "Cannot merge linear and quadratic expressions along term dimension."
            "Convert to QuadraticExpression first."
        )

    if has_quad_expression and cls is not QuadraticExpression:
        raise ValueError("Cannot merge linear expressions to QuadraticExpression")

    linopy_types = (variables.Variable, LinearExpression, QuadraticExpression)

    model = exprs[0].model

    if cls in linopy_types and dim in HELPER_DIMS:
        coord_dims = [
            {k: v for k, v in e.sizes.items() if k not in HELPER_DIMS} for e in exprs
        ]
        override = check_common_keys_values(coord_dims)  # type: ignore
    else:
        override = False

    data = [e.data if isinstance(e, linopy_types) else e for e in exprs]
    data = [fill_missing_coords(ds, fill_helper_dims=True) for ds in data]

    if not kwargs:
        kwargs = {
            "coords": "minimal",
            "compat": "override",
        }
        if cls == LinearExpression:
            kwargs["fill_value"] = FILL_VALUE
        elif cls == variables.Variable:
            kwargs["fill_value"] = variables.FILL_VALUE

        if override:
            kwargs["join"] = "override"
        else:
            kwargs.setdefault("join", "outer")

    if dim == TERM_DIM:
        ds = xr.concat([d[["coeffs", "vars"]] for d in data], dim, **kwargs)
        subkwargs = {**kwargs, "fill_value": 0}
        const = xr.concat([d["const"] for d in data], dim, **subkwargs).sum(TERM_DIM)
        ds = assign_multiindex_safe(ds, const=const)
    elif dim == FACTOR_DIM:
        ds = xr.concat([d[["vars"]] for d in data], dim, **kwargs)
        coeffs = xr.concat([d["coeffs"] for d in data], dim, **kwargs).prod(FACTOR_DIM)
        const = xr.concat([d["const"] for d in data], dim, **kwargs).prod(FACTOR_DIM)
        ds = assign_multiindex_safe(ds, coeffs=coeffs, const=const)
    else:
        ds = xr.concat(data, dim, **kwargs)

    for d in set(HELPER_DIMS) & set(ds.coords):
        ds = ds.reset_index(d, drop=True)

    return cls(ds, model)


class ScalarLinearExpression:
    """
    A scalar linear expression container.

    In contrast to the LinearExpression class, a ScalarLinearExpression
    only contains one label. Use this class to create a constraint
    in a rule.
    """

    __slots__ = ("_coeffs", "_vars", "_model")

    def __init__(
        self,
        coeffs: tuple[int | float, ...],
        vars: tuple[int, ...],
        model: Model,
    ) -> None:
        self._coeffs = coeffs
        self._vars = vars
        self._model = model

    def __repr__(self) -> str:
        expr_string = print_single_expression(
            np.array(self.coeffs), np.array(self.vars), 0, self.model
        )
        return f"ScalarLinearExpression: {expr_string}"

    @property
    def coeffs(
        self,
    ) -> tuple[int | float, ...]:
        return self._coeffs

    @property
    def vars(
        self,
    ) -> tuple[int, ...] | tuple[int, ...]:
        return self._vars

    @property
    def model(self) -> Model:
        return self._model

    def __add__(
        self, other: ScalarLinearExpression | ScalarVariable
    ) -> ScalarLinearExpression:
        if isinstance(other, variables.ScalarVariable):
            coeffs = self.coeffs + (1,)
            vars = self.vars + (other.label,)
            return ScalarLinearExpression(coeffs, vars, self.model)
        elif not isinstance(other, ScalarLinearExpression):
            raise TypeError(
                f"unsupported operand type(s) for +: {type(self)} and {type(other)}"
            )

        coeffs = self.coeffs + other.coeffs
        vars = self.vars + other.vars
        return ScalarLinearExpression(coeffs, vars, self.model)

    def __radd__(
        self, other: int | float
    ) -> ScalarLinearExpression | NotImplementedType:
        # This is needed for using python's sum function
        if other == 0:
            return self
        return NotImplemented

    @property
    def nterm(self) -> int:
        return len(self.vars)

    def __sub__(
        self, other: ScalarVariable | ScalarLinearExpression
    ) -> ScalarLinearExpression:
        if isinstance(other, variables.ScalarVariable):
            other = other.to_scalar_linexpr(1)
        elif not isinstance(other, ScalarLinearExpression):
            raise TypeError(
                f"unsupported operand type(s) for -: {type(self)} and {type(other)}"
            )

        return ScalarLinearExpression(
            self.coeffs + tuple(-c for c in other.coeffs),
            self.vars + other.vars,
            self.model,
        )

    def __neg__(self) -> ScalarLinearExpression:
        return ScalarLinearExpression(
            tuple(-c for c in self.coeffs), self.vars, self.model
        )

    def __mul__(self, other: float | int) -> ScalarLinearExpression:
        if not isinstance(other, int | float | np.number):
            raise TypeError(
                f"unsupported operand type(s) for *: {type(self)} and {type(other)}"
            )

        return ScalarLinearExpression(
            tuple(other * c for c in self.coeffs), self.vars, self.model
        )

    def __rmul__(self, other: int) -> ScalarLinearExpression:
        return self.__mul__(other)

    def __div__(self, other: float | int) -> ScalarLinearExpression:
        if not isinstance(other, int | float | np.number):
            raise TypeError(
                f"unsupported operand type(s) for /: {type(self)} and {type(other)}"
            )
        return self.__mul__(1 / other)

    def __truediv__(self, other: float | int) -> ScalarLinearExpression:
        return self.__div__(other)

    def __le__(self, other: int | float) -> AnonymousScalarConstraint:
        if not isinstance(other, int | float | np.number):
            raise TypeError(
                f"unsupported operand type(s) for <=: {type(self)} and {type(other)}"
            )

        return constraints.AnonymousScalarConstraint(self, LESS_EQUAL, other)

    def __ge__(self, other: int | float) -> AnonymousScalarConstraint:
        if not isinstance(other, int | float | np.number):
            raise TypeError(
                f"unsupported operand type(s) for >=: {type(self)} and {type(other)}"
            )

        return constraints.AnonymousScalarConstraint(self, GREATER_EQUAL, other)

    def __eq__(self, other: int | float) -> AnonymousScalarConstraint:  # type: ignore
        if not isinstance(other, int | float | np.number):
            raise TypeError(
                f"unsupported operand type(s) for ==: {type(self)} and {type(other)}"
            )

        return constraints.AnonymousScalarConstraint(self, EQUAL, other)

    def __gt__(self, other: int | float) -> AnonymousScalarConstraint:
        raise NotImplementedError(
            "Inequalities only ever defined for >= rather than >."
        )

    def __lt__(self, other: int | float) -> AnonymousScalarConstraint:
        raise NotImplementedError(
            "Inequalities only ever defined for >= rather than >."
        )

    def to_linexpr(self) -> LinearExpression:
        coeffs = xr.DataArray(list(self.coeffs), dims=TERM_DIM)
        vars = xr.DataArray(list(self.vars), dims=TERM_DIM)
        ds = xr.Dataset({"coeffs": coeffs, "vars": vars})
        return LinearExpression(ds, self.model)
