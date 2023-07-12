#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linopy expressions module.

This module contains definition related to affine expressions.
"""

import functools
import logging
import warnings
from dataclasses import dataclass, field
from itertools import product, zip_longest
from typing import Any, Mapping, Union

import numpy as np
import pandas as pd
import xarray as xr
import xarray.core.groupby
import xarray.core.rolling
from deprecation import deprecated
from numpy import arange, array, nan
from scipy.sparse import csc_matrix
from xarray import DataArray, Dataset
from xarray.core.dataarray import DataArrayCoordinates
from xarray.core.types import Dims

from linopy import constraints, expressions, variables
from linopy.common import (
    LocIndexer,
    as_dataarray,
    check_common_keys_values,
    fill_missing_coords,
    forward_as_properties,
    generate_indices_for_printout,
    get_index_map,
    print_single_expression,
)
from linopy.config import options
from linopy.constants import (
    EQUAL,
    FACTOR_DIM,
    GREATER_EQUAL,
    GROUPED_TERM_DIM,
    HELPER_DIMS,
    LESS_EQUAL,
    STACKED_TERM_DIM,
    TERM_DIM,
)


def exprwrap(method, *default_args, **new_default_kwargs):
    @functools.wraps(method)
    def _exprwrap(expr, *args, **kwargs):
        for k, v in new_default_kwargs.items():
            kwargs.setdefault(k, v)
        return expr.__class__(
            method(_expr_unwrap(expr), *default_args, *args, **kwargs), expr.model
        )

    _exprwrap.__doc__ = f"Wrapper for the xarray {method} function for linopy.Variable"
    if new_default_kwargs:
        _exprwrap.__doc__ += f" with default arguments: {new_default_kwargs}"

    return _exprwrap


def _expr_unwrap(maybe_expr):
    if isinstance(maybe_expr, LinearExpression):
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
    group: xr.DataArray
    model: Any
    kwargs: Mapping[str, Any] = field(default_factory=dict)

    @property
    def groupby(self):
        """
        Groups the data using the specified group and kwargs.

        Returns
        -------
        xarray.core.groupby.DataArrayGroupBy
            The groupby object.
        """
        return self.data.groupby(group=self.group, **self.kwargs)

    def map(self, func, shortcut=False, args=(), **kwargs):
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

    def sum(self, use_fallback=False, **kwargs):
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
            group = self.group
            group_name = getattr(group, "name", "group") or "group"

            if isinstance(group, DataArray):
                group = group.to_pandas()

            int_map = None
            if isinstance(group, pd.DataFrame):
                int_map = get_index_map(*group.values.T)
                orig_group = group
                group = group.apply(tuple, axis=1).map(int_map)

            group_dim = group.index.name
            if group_name == group_dim:
                raise ValueError("Group name cannot be the same as group dimension")

            arrays = [group, group.groupby(group).cumcount()]
            idx = pd.MultiIndex.from_arrays(
                arrays, names=[group_name, GROUPED_TERM_DIM]
            )
            ds = self.data.assign_coords({group_dim: idx})
            ds = ds.unstack(group_dim, fill_value=LinearExpression._fill_value)
            ds = LinearExpression._sum(ds, dims=GROUPED_TERM_DIM)

            if int_map is not None:
                index = ds.indexes["group"].map({v: k for k, v in int_map.items()})
                index.names = orig_group.columns
                index.name = group_name
                ds = xr.Dataset(ds.assign_coords({group_name: index}))

            return LinearExpression(ds, self.model)

        def func(ds):
            ds = LinearExpression._sum(ds, self.groupby._group_dim)
            ds = ds.assign_coords({TERM_DIM: np.arange(len(ds._term))})
            return ds

        return self.map(func, **kwargs)

    def roll(self, **kwargs):
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

    rolling: xr.core.rolling.DataArrayRolling
    model: Any

    def sum(self, **kwargs):
        data = self.rolling.construct("_rolling_term", keep_attrs=True)
        ds = (
            data[["coeffs", "vars"]]
            .rename({TERM_DIM: STACKED_TERM_DIM})
            .stack({TERM_DIM: [STACKED_TERM_DIM, "_rolling_term"]}, create_index=False)
        )
        ds["const"] = data.const.sum("_rolling_term")
        return LinearExpression(ds, self.model)


@forward_as_properties(data=["attrs", "coords", "indexes", "sizes"], const=["ndim"])
class LinearExpression:
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

    >>> type(expr.sum(dims="dim_0"))
    <class 'linopy.expressions.LinearExpression'>
    """

    __slots__ = ("_data", "_model")
    __array_ufunc__ = None
    __array_priority__ = 10000

    _fill_value = {"vars": -1, "coeffs": np.nan, "const": 0}

    def __init__(self, data, model):
        from linopy.model import Model

        if data is None:
            da = xr.DataArray([], dims=[TERM_DIM])
            data = Dataset({"coeffs": da, "vars": da, "const": 0})
        elif isinstance(data, DataArray):
            # assume only constant are passed
            const = fill_missing_coords(data)
            da = xr.DataArray([], dims=[TERM_DIM])
            data = Dataset({"coeffs": da, "vars": da, "const": const})
        elif not isinstance(data, Dataset):
            raise ValueError(
                f"data must be an instance of xarray.Dataset or xarray.DataArray, got {type(data)}"
            )

        if not set(data).issuperset({"coeffs", "vars"}):
            raise ValueError(
                "data must contain the fields 'coeffs' and 'vars' or 'const'"
            )

        if np.issubdtype(data.vars, np.floating):
            data["vars"] = data.vars.fillna(-1).astype(int)
        if not np.issubdtype(data.coeffs, np.floating):
            data["coeffs"] = data.coeffs.astype(float)

        data = fill_missing_coords(data)

        if TERM_DIM not in data.dims:
            raise ValueError("data must contain one dimension ending with '_term'")

        if "const" not in data:
            data = data.assign(const=0)

        data = xr.broadcast(data, exclude=HELPER_DIMS)[0]
        data[["coeffs", "vars"]] = xr.broadcast(
            data[["coeffs", "vars"]], exclude=[FACTOR_DIM]
        )[0]

        # transpose with new Dataset to really ensure correct order
        data = Dataset(data.transpose(..., TERM_DIM))

        if not isinstance(model, Model):
            raise ValueError("model must be an instance of linopy.Model")

        self._model = model
        self._data = data

    def __repr__(self):
        """
        Print the expression arrays.
        """
        max_lines = options["display_max_rows"]
        dims = list(self.coord_dims)
        dim_sizes = list(self.coord_dims.values())
        size = np.prod(dim_sizes)  # that the number of theoretical printouts
        masked_entries = self.mask.sum().values if self.mask is not None else 0
        lines = []

        header_string = self.type

        if size > 1:
            for indices in generate_indices_for_printout(dim_sizes, max_lines):
                if indices is None:
                    lines.append("\t\t...")
                else:
                    coord_values = ", ".join(
                        str(self.data[dims[i]].values[ind])
                        for i, ind in enumerate(indices)
                    )
                    if self.mask is None or self.mask.values[indices]:
                        expr = print_single_expression(
                            self.coeffs.values[indices],
                            self.vars.values[indices],
                            self.const.values[indices],
                            self.model,
                        )
                        line = f"[{coord_values}]: {expr}"
                    else:
                        line = f"[{coord_values}]: None"
                    lines.append(line)

            shape_str = ", ".join(f"{d}: {s}" for d, s in zip(dims, dim_sizes))
            mask_str = f" - {masked_entries} masked entries" if masked_entries else ""
            underscore = "-" * (len(shape_str) + len(mask_str) + len(header_string) + 4)
            lines.insert(0, f"{header_string} ({shape_str}){mask_str}:\n{underscore}")
        elif size == 1:
            expr = print_single_expression(
                self.coeffs, self.vars, self.const, self.model
            )
            lines.append(f"{header_string}\n{'-'*len(header_string)}\n{expr}")
        else:
            lines.append(f"{header_string}\n{'-'*len(header_string)}\n<empty>")

        return "\n".join(lines)

    def print(self, display_max_rows=20, display_max_terms=20):
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

    def __add__(self, other):
        """
        Add an expression to others.
        """
        other = as_expression(
            other, model=self.model, coords=self.coords, dims=self.coord_dims
        )
        return merge(self, other, cls=self.__class__)

    def __radd__(self, other):
        # This is needed for using python's sum function
        return self if other == 0 else NotImplemented

    def __sub__(self, other):
        """
        Subtract others from expression.
        """
        other = as_expression(
            other, model=self.model, coords=self.coords, dims=self.coord_dims
        )
        return merge(self, -other, cls=self.__class__)

    def __neg__(self):
        """
        Get the negative of the expression.
        """
        return self.assign(coeffs=-self.coeffs, const=-self.const)

    def __mul__(self, other):
        """
        Multiply the expr by a factor.
        """
        if isinstance(other, QuadraticExpression):
            raise TypeError(
                "unsupported operand type(s) for *: "
                f"{type(self)} and {type(other)}. "
                "Higher order non-linear expressions are not yet supported."
            )
        elif isinstance(other, (variables.Variable, variables.ScalarVariable)):
            other = other.to_linexpr()

        if type(other) is LinearExpression:
            if other.nterm > 1:
                raise TypeError("Multiplication of multiple terms is not supported.")
            ds = other.data[["coeffs", "vars"]].sel(_term=0).broadcast_like(self.data)
            ds = ds.assign(const=other.const)
            return merge(self, ds, dim=FACTOR_DIM, cls=QuadraticExpression)
        else:
            coeffs = self.coeffs * as_dataarray(
                other, coords=self.coords, dims=self.coord_dims
            )
            assert set(coeffs.shape) == set(self.coeffs.shape)
            return self.assign(coeffs=coeffs)

    def __rmul__(self, other):
        """
        Right-multiply the expr by a factor.
        """
        return self.__mul__(other)

    def __div__(self, other):
        if isinstance(
            other, (LinearExpression, variables.Variable, variables.ScalarVariable)
        ):
            raise TypeError(
                "unsupported operand type(s) for /: "
                f"{type(self)} and {type(other)}"
                "Non-linear expressions are not yet supported."
            )
        return self.__mul__(1 / other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __le__(self, rhs):
        return self.to_constraint(LESS_EQUAL, rhs)

    def __ge__(self, rhs):
        return self.to_constraint(GREATER_EQUAL, rhs)

    def __eq__(self, rhs):
        return self.to_constraint(EQUAL, rhs)

    def __gt__(self, other):
        raise NotImplementedError(
            "Inequalities only ever defined for >= rather than >."
        )

    def __lt__(self, other):
        raise NotImplementedError(
            "Inequalities only ever defined for >= rather than >."
        )

    @property
    def loc(self):
        return LocIndexer(self)

    @classmethod
    @property
    def fill_value(cls):
        return cls._fill_value

    @property
    def type(self):
        return "LinearExpression"

    @property
    def data(self):
        return self._data

    @property
    def model(self):
        return self._model

    @property
    def dims(self):
        # do explicitly sort as in vars (same as in coeffs)
        return {k: self.data.dims[k] for k in self.vars.dims}

    @deprecated(details="Use `coord_dims` instead")
    @property
    def non_helper_dims(self):
        return {k: self.data.dims[k] for k in self.dims if not k.startswith("_")}

    @property
    def coord_dims(self):
        return {k: self.data.dims[k] for k in self.dims if k not in HELPER_DIMS}

    @property
    def vars(self):
        return self.data.vars

    @vars.setter
    def vars(self, value):
        self.data["vars"] = value

    @property
    def coeffs(self):
        return self.data.coeffs

    @coeffs.setter
    def coeffs(self, value):
        self.data["coeffs"] = value

    @property
    def const(self):
        return self.data.const

    @const.setter
    def const(self, value):
        self.data["const"] = value

    # create a dummy for a mask, which can be implemented later
    @property
    def mask(self):
        return None

    @classmethod
    def _sum(cls, expr: Union["LinearExpression", Dataset], dims=None) -> Dataset:
        data = _expr_unwrap(expr)

        if dims is None:
            vars = DataArray(data.vars.data.ravel(), dims=TERM_DIM)
            coeffs = DataArray(data.coeffs.data.ravel(), dims=TERM_DIM)
            const = data.const.sum()
            ds = xr.Dataset({"vars": vars, "coeffs": coeffs, "const": const})
        else:
            dims = [d for d in np.atleast_1d(dims) if d != TERM_DIM]
            ds = (
                data[["coeffs", "vars"]]
                .reset_index(dims, drop=True)
                .rename({TERM_DIM: STACKED_TERM_DIM})
                .stack({TERM_DIM: [STACKED_TERM_DIM] + dims}, create_index=False)
            )
            ds["const"] = data.const.sum(dims)

        return ds

    def sum(self, dims=None, drop_zeros=False) -> "LinearExpression":
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
        res = self.__class__(self._sum(self, dims=dims), self.model)

        if drop_zeros:
            res = res.densify_terms()

        return res

    
    def cumsum(
        self,
        dim: Dims = None,
        *,
        skipna: bool | None = None,
        keep_attrs: bool | None = None,
        **kwargs: Any,
    ) -> "LinearExpression":
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
            dim = [d for d in self.data.dims.keys() if d != "_term"]
        if isinstance(dim, str):
            # Make sure, single mentioned dimensions is handled correctly.
            dim = [dim]
        dim_dict = {
            dim_name: self.data.dims[dim_name]
            for dim_name in dim
        }
        return self.rolling(dim=dim_dict).sum(keep_attrs=keep_attrs, skipna=skipna)

    @classmethod
    def from_tuples(cls, *tuples, model=None, chunk=None):
        """
        Create a linear expression by using tuples of coefficients and
        variables.

        The function internally checks that all variables in the tuples belong to the same
        reference model.

        Parameters
        ----------
        tuples : tuples of (coefficients, variables)
            Each tuple represents one term in the resulting linear expression,
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
        >>> from linopy import Model
        >>> import pandas as pd
        >>> m = Model()
        >>> x = m.add_variables(pd.Series([0, 0]), 1)
        >>> y = m.add_variables(4, pd.Series([8, 10]))
        >>> expr = LinearExpression.from_tuples((10, x), (1, y))

        This is the same as calling ``10*x + y`` but a bit more performant.
        """
        exprs = []
        for t in tuples:
            if len(t) == 2:
                # assume first element is coefficient and second is variable
                c, v = t
                if not isinstance(v, (variables.Variable, variables.ScalarVariable)):
                    raise TypeError("Expected variable as second element of tuple.")
                expr = v.to_linexpr(c)
                const = None
                if model is None:
                    model = expr.model  # TODO: Ensure equality of models
            elif len(t) == 1:
                # assume that the element is a constant
                c, v = None, None
                (const,) = as_dataarray(t)
                expr = LinearExpression(const, model)
            else:
                raise ValueError("Expected tuples of length 1 or 2.")

            exprs.append(expr)

        return merge(exprs, cls=cls) if len(exprs) > 1 else exprs[0]

    @classmethod
    def from_rule(cls, model, rule, coords):
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
        ...         return (i - 1) * x[i - 1, j]
        ...     else:
        ...         return i * x[i, j]
        ...
        >>> expr = LinearExpression.from_rule(m, bound, coords)
        >>> con = m.add_constraints(expr <= 10)
        """
        if not isinstance(coords, xr.core.dataarray.DataArrayCoordinates):
            coords = DataArray(coords=coords).coords

        # test output type
        output = rule(model, *[c.values[0] for c in coords.values()])
        if not isinstance(output, ScalarLinearExpression) and output is not None:
            msg = f"`rule` has to return ScalarLinearExpression not {type(output)}."
            raise TypeError(msg)

        combinations = product(*[c.values for c in coords.values()])
        exprs = []
        placeholder = ScalarLinearExpression((np.nan,), (-1,), model)
        exprs = [rule(model, *coord) or placeholder for coord in combinations]
        return cls._from_scalarexpression_list(exprs, coords, model)

    @classmethod
    def _from_scalarexpression_list(cls, exprs, coords: DataArrayCoordinates, model):
        """
        Create a LinearExpression from a list of lists with different lengths.
        """
        shape = list(map(len, coords.values()))

        coeffs = array(tuple(zip_longest(*(e.coeffs for e in exprs), fillvalue=nan)))
        vars = array(tuple(zip_longest(*(e.vars for e in exprs), fillvalue=-1)))

        nterm = vars.shape[0]
        coeffs = coeffs.reshape((nterm, *shape))
        vars = vars.reshape((nterm, *shape))

        coeffs = DataArray(coeffs, coords, dims=(TERM_DIM, *coords))
        vars = DataArray(vars, coords, dims=(TERM_DIM, *coords))
        ds = Dataset({"coeffs": coeffs, "vars": vars}).transpose(..., TERM_DIM)

        return cls(ds, model)

    def to_quadexpr(self):
        """Convert LinearExpression to QuadraticExpression."""
        vars = self.data.vars.expand_dims(FACTOR_DIM)
        fill_value = self._fill_value["vars"]
        vars = xr.concat([vars, xr.full_like(vars, fill_value)], dim=FACTOR_DIM)
        data = self.data.assign(vars=vars)
        return QuadraticExpression(data, self.model)

    def to_constraint(self, sign, rhs):
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
        data = all_to_lhs[["coeffs", "vars"]].assign(sign=sign, rhs=-all_to_lhs.const)
        return constraints.Constraint(data, model=self.model)

    def reset_const(self):
        """
        Reset the constant of the linear expression to zero.
        """
        return self.__class__(self.data[["coeffs", "vars"]], self.model)

    def where(self, cond, other=xr.core.dtypes.NA, **kwargs):
        """
        Filter variables based on a condition.

        This operation call ``xarray.Dataset.where`` but sets the default
        fill value to -1 for variables and ensures preserving the linopy.LinearExpression type.

        Parameters
        ----------
        cond : DataArray or callable
            Locations at which to preserve this object's values. dtype must be `bool`.
            If a callable, it must expect this object as its only parameter.
        **kwargs :
            Keyword arguments passed to ``xarray.Dataset.where``

        Returns
        -------
        linopy.LinearExpression
        """
        # Cannot set `other` if drop=True
        if other is xr.core.dtypes.NA:
            if not kwargs.get("drop", False):
                other = self._fill_value
        else:
            other = _expr_unwrap(other)
        cond = _expr_unwrap(cond)
        return self.__class__(self.data.where(cond, other=other, **kwargs), self.model)

    def diff(self, dim, n=1):
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
        linopy.LinearExpression
        """
        return self - self.shift({dim: n})

    def groupby(
        self, group, squeeze: "bool" = True, restore_coord_dims: "bool" = None, **kwargs
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
        squeeze : bool, optional
            If "group" is a dimension of any arrays in this dataset, `squeeze`
            controls whether the subarrays have a dimension of length 1 along
            that dimension or if the dimension is squeezed out.
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
        kwargs = dict(squeeze=squeeze, restore_coord_dims=restore_coord_dims, **kwargs)
        return LinearExpressionGroupby(ds, group, model=self.model, kwargs=kwargs)

    def rolling(
        self,
        dim: "Mapping[Any, int]" = None,
        min_periods: "int" = None,
        center: "bool | Mapping[Any, bool]" = False,
        **window_kwargs: "int",
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
    def nterm(self):
        """
        Get the number of terms in the linear expression.
        """
        return len(self.data._term)

    @property
    def shape(self):
        """
        Get the total shape of the linear expression.
        """
        assert self.vars.shape == self.coeffs.shape
        return self.vars.shape

    @property
    def size(self):
        """
        Get the total size of the linear expression.
        """
        return self.vars.size

    def empty(self):
        """
        Get whether the linear expression is empty.
        """
        return self.shape == (0,)

    def densify_terms(self):
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
        _, idx = np.unique(remaining_axes, axis=0, return_inverse=True)
        idx = list(idx)
        new_index = np.array([idx[:i].count(j) for i, j in enumerate(idx)])
        mod_nnz.insert(axis, new_index)

        vdata = np.full_like(cdata, -1)
        vdata[tuple(mod_nnz)] = data.vars.data[nnz]
        data.vars.data = vdata

        cdata = np.zeros_like(cdata)
        cdata[tuple(mod_nnz)] = data.coeffs.data[nnz]
        data.coeffs.data = cdata

        return self.__class__(data.sel({TERM_DIM: slice(0, nterm)}), self.model)

    def sanitize(self):
        """
        Sanitize LinearExpression by ensuring int dtype for variables.

        Returns
        -------
        linopy.LinearExpression
        """
        if not np.issubdtype(self.vars.dtype, np.integer):
            return self.assign(vars=self.vars.fillna(-1).astype(int))

        return self

    def equals(self, other: "LinearExpression"):
        return self.data.equals(_expr_unwrap(other))

    def __iter__(self):
        return self.data.__iter__()

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
        if not ds.sizes:
            # fallback for weird error raised due to missing index
            df = pd.DataFrame({k: ds[k].item() for k in ds}, index=[0])
        else:
            df = ds.to_dataframe()
        if "mask" in df:
            mask = df.pop("mask")
            df = df[mask]
        df = df[(df.vars != -1) & (df.coeffs != 0)]
        # Group repeated variables in the same constraint
        df = df.groupby("vars", as_index=False).sum()

        any_nan = df.isna().any()
        if any_nan.any():
            fields = ", ".join("`" + df.columns[any_nan] + "`")
            raise ValueError(
                f"Expression `{self.name}` contains nan's in field(s) {fields}"
            )

        return df

    # Wrapped function which would convert variable to dataarray
    assign = exprwrap(Dataset.assign)

    assign_attrs = exprwrap(Dataset.assign_attrs)

    assign_coords = exprwrap(Dataset.assign_coords)

    astype = exprwrap(Dataset.astype)

    bfill = exprwrap(Dataset.bfill)

    broadcast_like = exprwrap(Dataset.broadcast_like)

    chunk = exprwrap(Dataset.chunk)

    drop = exprwrap(Dataset.drop)

    drop_sel = exprwrap(Dataset.drop_sel)

    drop_isel = exprwrap(Dataset.drop_isel)

    ffill = exprwrap(Dataset.ffill)

    fillna = exprwrap(Dataset.fillna, value=_fill_value)

    sel = exprwrap(Dataset.sel)

    isel = exprwrap(Dataset.isel)

    shift = exprwrap(Dataset.shift)

    reindex = exprwrap(Dataset.reindex, fill_value=_fill_value)

    reindex_like = exprwrap(Dataset.reindex_like, fill_value=_fill_value)

    rename = exprwrap(Dataset.rename)

    rename_dims = exprwrap(Dataset.rename_dims)

    roll = exprwrap(Dataset.roll)


@forward_as_properties(data=["attrs", "coords", "indexes", "sizes"])
class QuadraticExpression(LinearExpression):
    """
    A quadratic expression consisting of terms of coefficients and variables.

    The QuadraticExpression class is a subclass of LinearExpression which allows to
    apply most xarray functions on it.
    """

    __slots__ = ("_data", "_model")
    __array_ufunc__ = None
    __array_priority__ = 10000

    _fill_value = {"vars": -1, "coeffs": np.nan, "const": 0}

    def __init__(self, data, model):
        super().__init__(data, model)

        if FACTOR_DIM not in data.vars.dims:
            raise ValueError(f"Data does not include dimension {FACTOR_DIM}")
        elif data.sizes[FACTOR_DIM] != 2:
            raise ValueError(f"Size of dimension {FACTOR_DIM} must be 2.")

        # transpose data to have _term as last dimension and _factor as second last
        data = xr.Dataset(data.transpose(..., FACTOR_DIM, TERM_DIM))
        self._data = data

    def __mul__(self, other):
        """
        Multiply the expr by a factor.
        """
        if isinstance(
            other,
            (
                LinearExpression,
                QuadraticExpression,
                variables.Variable,
                variables.ScalarVariable,
            ),
        ):
            raise TypeError(
                "unsupported operand type(s) for *: "
                f"{type(self)} and {type(other)}. "
                "Higher order non-linear expressions are not yet supported."
            )
        return super().__mul__(other)

    @property
    def type(self):
        return "QuadraticExpression"

    def __add__(self, other):
        """
        Add an expression to others.
        """
        other = as_expression(
            other, model=self.model, coords=self.coords, dims=self.coord_dims
        )
        if type(other) is LinearExpression:
            other = other.to_quadexpr()
        return merge(self, other, cls=self.__class__)

    def __radd__(self, other):
        """
        Add others to expression.
        """
        if type(other) is LinearExpression:
            other = other.to_quadexpr()
            return other.__add__(self)
        elif other == 0:
            return self
        else:
            return NotImplemented

    def __sub__(self, other):
        """
        Subtract others from expression.
        """
        other = as_expression(
            other, model=self.model, coords=self.coords, dims=self.coord_dims
        )
        if type(other) is LinearExpression:
            other = other.to_quadexpr()
        return merge(self, -other, cls=self.__class__)

    def __rsub__(self, other):
        """
        Subtract expression from others.
        """
        if type(other) is LinearExpression:
            other = other.to_quadexpr()
            return other.__sub__(self)
        else:
            NotImplemented

    @classmethod
    def _sum(cls, expr: "QuadraticExpression", dims=None) -> Dataset:
        data = _expr_unwrap(expr)
        dims = dims or list(set(data.dims) - set(HELPER_DIMS))
        return LinearExpression._sum(expr, dims)

    def to_constraint(self, sign, rhs):
        raise NotImplementedError(
            "Quadratic expressions cannot be used in constraints."
        )

    @property
    def flat(self):
        """
        Return a flattened expression.
        """
        vars = self.data.vars.assign_coords(
            {FACTOR_DIM: ["vars1", "vars2"]}
        ).to_dataset(FACTOR_DIM)
        ds = self.data.drop_vars("vars").assign(vars)
        if not ds.sizes:
            # fallback for weird error raised due to missing index
            df = pd.DataFrame({k: ds[k].item() for k in ds}, index=[0])
        else:
            df = ds.to_dataframe()
        if "mask" in df:
            mask = df.pop("mask")
            df = df[mask]
        df = df[((df.vars1 != -1) | (df.vars2 != -1)) & (df.coeffs != 0)]
        # Group repeated variables in the same constraint
        df = df.groupby(["vars1", "vars2"], as_index=False).sum()

        any_nan = df.isna().any()
        if any_nan.any():
            fields = ", ".join("`" + df.columns[any_nan] + "`")
            raise ValueError(
                f"Expression `{self.name}` contains nan's in field(s) {fields}"
            )

        return df

    def to_matrix(self):
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


def as_expression(obj, model=None, **kwargs):
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
    if isinstance(obj, LinearExpression):
        return obj
    elif isinstance(obj, (variables.Variable, variables.ScalarVariable)):
        return obj.to_linexpr()
    else:
        try:
            obj = as_dataarray(obj, **kwargs)
        except ValueError as e:
            raise ValueError("Cannot convert to LinearExpression") from e
        return LinearExpression(obj, model)


def merge(*exprs, dim=TERM_DIM, cls=LinearExpression, **kwargs):
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
        Type of the resulting expression.
    **kwargs
        Additional keyword arguments passed to xarray.concat. Defaults to
        {coords: "minimal", compat: "override"} or, in the special case described
        above, to {coords: "minimal", compat: "override", "join": "override"}.

    Returns
    -------
    res : linopy.LinearExpression
    """
    linopy_types = (variables.Variable, LinearExpression, QuadraticExpression)

    if (
        cls is QuadraticExpression
        and dim == TERM_DIM
        and any(type(e) is LinearExpression for e in exprs)
    ):
        raise ValueError(
            "Cannot merge linear and quadratic expressions along term dimension."
            "Convert to QuadraticExpression first."
        )

    exprs = exprs[0] if len(exprs) == 1 else list(exprs)  # allow passing a list
    model = exprs[0].model

    if cls in linopy_types and dim in HELPER_DIMS:
        coord_dims = [
            {k: v for k, v in e.dims.items() if k not in HELPER_DIMS} for e in exprs
        ]
        override = check_common_keys_values(coord_dims)
    else:
        override = False

    data = [e.data if isinstance(e, linopy_types) else e for e in exprs]
    data = [fill_missing_coords(ds, fill_helper_dims=True) for ds in data]

    if not kwargs:
        kwargs = {
            "fill_value": cls._fill_value,
            "coords": "minimal",
            "compat": "override",
        }
        if override:
            kwargs["join"] = "override"

    if dim == TERM_DIM:
        ds = xr.concat([d[["coeffs", "vars"]] for d in data], dim, **kwargs)
        const = xr.concat([d["const"] for d in data], dim, **kwargs).sum(TERM_DIM)
        ds["const"] = const
    elif dim == FACTOR_DIM:
        ds = xr.concat([d[["vars"]] for d in data], dim, **kwargs)
        coeffs = xr.concat([d["coeffs"] for d in data], dim, **kwargs).prod(FACTOR_DIM)
        ds["coeffs"] = coeffs
        const = xr.concat([d["const"] for d in data], dim, **kwargs).prod(FACTOR_DIM)
        ds["const"] = const
    else:
        ds = xr.concat(data, dim, **kwargs)

    for d in set(HELPER_DIMS) & set(ds.coords):
        ds = ds.reset_index(d, drop=True)

    return cls(ds, model)


class ScalarLinearExpression:
    """
    A scalar linear expression container.

    In contrast to the LinearExpression class, a ScalarLinearExpression
    only contains only one label. Use this class to create a constraint
    in a rule.
    """

    __slots__ = ("_coeffs", "_vars", "_model")

    def __init__(self, coeffs, vars, model):
        self._coeffs = coeffs
        self._vars = vars
        self._model = model

    def __repr__(self) -> str:
        expr_string = print_single_expression(self.coeffs, self.vars, 0, self.model)
        return f"ScalarLinearExpression: {expr_string}"

    @property
    def coeffs(self):
        return self._coeffs

    @property
    def vars(self):
        return self._vars

    @property
    def model(self):
        return self._model

    def __add__(self, other):
        if isinstance(other, variables.ScalarVariable):
            coeffs = self.coeffs + (1,)
            vars = self.vars + (other.label,)
            return ScalarLinearExpression(coeffs, vars, self.model)
        elif not isinstance(other, ScalarLinearExpression):
            raise TypeError(
                "unsupported operand type(s) for +: " f"{type(self)} and {type(other)}"
            )

        coeffs = self.coeffs + other.coeffs
        vars = self.vars + other.vars
        return ScalarLinearExpression(coeffs, vars, self.model)

    def __radd__(self, other):
        # This is needed for using python's sum function
        if other == 0:
            return self

    @property
    def nterm(self):
        return len(self.vars)

    def __sub__(self, other):
        if isinstance(other, variables.ScalarVariable):
            other = other.to_scalar_linexpr(1)
        elif not isinstance(other, ScalarLinearExpression):
            raise TypeError(
                "unsupported operand type(s) for -: " f"{type(self)} and {type(other)}"
            )

        return ScalarLinearExpression(
            self.coeffs + tuple(-c for c in other.coeffs),
            self.vars + other.vars,
            self.model,
        )

    def __neg__(self):
        return ScalarLinearExpression(
            tuple(-c for c in self.coeffs), self.vars, self.model
        )

    def __mul__(self, other):
        if not isinstance(other, (int, float, np.number)):
            raise TypeError(
                "unsupported operand type(s) for *: " f"{type(self)} and {type(other)}"
            )

        return ScalarLinearExpression(
            tuple(other * c for c in self.coeffs), self.vars, self.model
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if not isinstance(other, (int, float, np.number)):
            raise TypeError(
                "unsupported operand type(s) for /: " f"{type(self)} and {type(other)}"
            )
        return self.__mul__(1 / other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __le__(self, other):
        if not isinstance(other, (int, float, np.number)):
            raise TypeError(
                "unsupported operand type(s) for <=: " f"{type(self)} and {type(other)}"
            )

        return constraints.AnonymousScalarConstraint(self, LESS_EQUAL, other)

    def __ge__(self, other):
        if not isinstance(other, (int, float, np.number)):
            raise TypeError(
                "unsupported operand type(s) for >=: " f"{type(self)} and {type(other)}"
            )

        return constraints.AnonymousScalarConstraint(self, GREATER_EQUAL, other)

    def __eq__(self, other):
        if not isinstance(other, (int, float, np.number)):
            raise TypeError(
                "unsupported operand type(s) for ==: " f"{type(self)} and {type(other)}"
            )

        return constraints.AnonymousScalarConstraint(self, EQUAL, other)

    def __gt__(self, other):
        raise NotImplementedError(
            "Inequalities only ever defined for >= rather than >."
        )

    def __lt__(self, other):
        raise NotImplementedError(
            "Inequalities only ever defined for >= rather than >."
        )

    def to_linexpr(self):
        coeffs = xr.DataArray(list(self.coeffs), dims=TERM_DIM)
        vars = xr.DataArray(list(self.vars), dims=TERM_DIM)
        ds = xr.Dataset({"coeffs": coeffs, "vars": vars})
        return LinearExpression(ds, self.model)
