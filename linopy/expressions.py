#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linopy expressions module.

This module contains definition related to affine expressions.
"""

import functools
import logging
from dataclasses import dataclass
from itertools import product, zip_longest
from typing import Any, Mapping, Union

import numpy as np
import pandas as pd
import xarray as xr
import xarray.core.groupby
import xarray.core.rolling
from deprecation import deprecated
from numpy import array, nan
from xarray import DataArray, Dataset
from xarray.core.dataarray import DataArrayCoordinates

from linopy import constraints, variables
from linopy.common import (
    as_dataarray,
    forward_as_properties,
    head_tail_range,
    print_coord,
    print_single_expression,
)
from linopy.constants import EQUAL, GREATER_EQUAL, LESS_EQUAL


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

    groupby: xr.core.groupby.DatasetGroupBy
    model: Any

    def map(self, func, shortcut=False, args=(), **kwargs):
        return LinearExpression(
            self.groupby.map(func, shortcut=shortcut, args=args, **kwargs), self.model
        )

    def sum(self, **kwargs):
        def func(ds):
            ds = LinearExpression._sum(ds, self.groupby._group_dim)
            ds = ds.assign_coords(_term=np.arange(len(ds._term)))
            return ds

        return self.map(func, **kwargs)

    def roll(self, **kwargs):
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
        ds = (
            self.rolling.construct("_rolling_term", keep_attrs=True)
            .rename(_term="_stacked_term")
            .stack(_term=["_stacked_term", "_rolling_term"])
            .reset_index("_term", drop=True)
        )
        return LinearExpression(ds, self.model)


@forward_as_properties(data=["attrs", "coords", "indexes"])
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

    _fill_value = {"vars": -1, "coeffs": np.nan}

    def __init__(self, data, model):
        from linopy.model import Model

        if data is None:
            da = xr.DataArray([], dims=["_term"])
            data = Dataset({"coeffs": da, "vars": da})

        if not isinstance(data, Dataset):
            raise ValueError(
                f"data must be an instance of xarray.Dataset, got {type(data)}"
            )

        if not set(data).issuperset({"coeffs", "vars"}):
            raise ValueError("data must contain the variables 'coeffs' and 'vars'")

        # make sure that all non-helper dims have coordinates
        if not all(dim in data.coords for dim in data.dims if not dim.startswith("_")):
            raise ValueError("data must have coordinates for all non-helper dimensions")

        if np.issubdtype(data.vars, np.floating):
            data["vars"] = data.vars.fillna(-1).astype(int)

        (data,) = xr.broadcast(data)
        data = data.transpose(..., "_term")

        if not isinstance(model, Model):
            raise ValueError("model must be an instance of linopy.Model")

        self._model = model
        self._data = data

    def unravel_coords(self, index):
        idx = np.unravel_index(index, self.shape[:-1])
        coords = [
            self.indexes[dim][idx[i]]
            for i, dim in enumerate(self.vars.dims)
            if not dim.startswith("_")
        ]
        return list(zip(*coords))

    def __repr__(self):
        """
        Get the string representation of the expression.
        """
        nexprs = self.size // self.nterm

        # don't loop over all values if not necessary
        if self.size == self.nterm:
            expr_string = print_single_expression(
                self.coeffs.values, self.vars.values, self.model
            )
            return f"LinearExpression:\n-----------------\n{expr_string}"

        # print only a few values
        max_prints = 14
        split_at = max_prints // 2
        to_print = head_tail_range(nexprs, max_prints)
        coords = self.unravel_coords(to_print)

        # loop over all values to print
        data_string = ""
        for i, coord in enumerate(coords):

            coord_string = print_coord(coord)
            expr_string = print_single_expression(
                self.coeffs.loc[coord].values, self.vars.loc[coord].values, self.model
            )

            data_string += f"\n{coord_string}:  {expr_string}"

            if i == split_at - 1 and nexprs > max_prints:
                data_string += "\n\t\t..."

        # create shape string
        nonterm_dims = [(k, v) for k, v in self.dims.items() if not k.startswith("_")]
        shape_string = "(" + ", ".join([f"{k}: {v}" for k, v in nonterm_dims]) + ")"
        header = f"LinearExpression {shape_string}:\n{'-' * (18 + len(shape_string))}"
        return f"{header}{data_string}"

    def __add__(self, other):
        """
        Add an expression to others.
        """
        if not isinstance(other, (LinearExpression, variables.Variable)):
            raise TypeError(
                "unsupported operand type(s) for +: " f"{type(self)} and {type(other)}"
            )
        if isinstance(other, variables.Variable):
            other = LinearExpression.from_tuples((1, other))
        return merge(self, other)

    def __sub__(self, other):
        """
        Subtract others from expression.
        """
        if not isinstance(other, (LinearExpression, variables.Variable)):
            raise TypeError(
                "unsupported operand type(s) for +: " f"{type(self)} and {type(other)}"
            )
        return merge(self, -other)

    def __neg__(self):
        """
        Get the negative of the expression.
        """
        return self.assign(coeffs=-self.coeffs)

    def __mul__(self, other):
        """
        Multiply the expr by a factor.
        """
        if isinstance(other, (LinearExpression, variables.Variable)):
            raise TypeError(
                "unsupported operand type(s) for *: "
                f"{type(self)} and {type(other)}. "
                "Non-linear expressions are not yet supported."
            )
        coeffs = other * self.coeffs
        assert coeffs.shape == self.coeffs.shape
        return self.assign(coeffs=coeffs)

    def __rmul__(self, other):
        """
        Right-multiply the expr by a factor.
        """
        return self.__mul__(other)

    def __div__(self, other):
        if isinstance(other, (LinearExpression, variables.Variable)):
            raise TypeError(
                "unsupported operand type(s) for /: "
                f"{type(self)} and {type(other)}"
                "Non-linear expressions are not yet supported."
            )
        return self.__mul__(1 / other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __le__(self, rhs):
        return constraints.AnonymousConstraint(self, LESS_EQUAL, rhs)

    def __ge__(self, rhs):
        return constraints.AnonymousConstraint(self, GREATER_EQUAL, rhs)

    def __eq__(self, rhs):
        return constraints.AnonymousConstraint(self, EQUAL, rhs)

    @deprecated(details="Use the `data` property instead of `to_dataset`")
    def to_dataset(self):
        """
        Convert the expression to a xarray.Dataset.
        """
        return self.data

    @classmethod
    @property
    def fill_value(cls):
        return cls._fill_value

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

    @classmethod
    def _sum(cls, expr: Union["LinearExpression", Dataset], dims=None) -> Dataset:
        data = _expr_unwrap(expr)

        if dims is None:
            vars = DataArray(data.vars.data.ravel(), dims="_term")
            coeffs = DataArray(data.coeffs.data.ravel(), dims="_term")
            ds = xr.Dataset({"vars": vars, "coeffs": coeffs})

        else:
            dims = [d for d in np.atleast_1d(dims) if d != "_term"]
            ds = (
                data.reset_index(dims, drop=True)
                .rename(_term="_stacked_term")
                .stack(_term=["_stacked_term"] + dims)
                .reset_index("_term", drop=True)
            )

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

    @classmethod
    def from_tuples(cls, *tuples, chunk=None):
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
        # when assigning arrays to Datasets it is taken as coordinates
        # support numpy arrays and convert them to dataarrays
        exprs = []
        model = None
        for (c, v) in tuples:
            if not isinstance(v, variables.Variable):
                raise TypeError(f"Expected type `linopy.Variable`, got {type(v)}")
            # check that reference models are consistent
            if model is None:
                model = v.model
            # TODO: Ensure equality of models
            # else:
            # assert model == model

            if isinstance(v, variables.ScalarVariable):
                v = v.label
            elif isinstance(v, variables.Variable):
                v = v.labels
            v = as_dataarray(v)

            if isinstance(c, np.ndarray) or _pd_series_wo_index_name(c):
                c = DataArray(c, v.coords)
            elif _pd_dataframe_wo_axes_names(c):
                if v.ndim == 1:
                    c = DataArray(c, v.coords, dims=("dim_0", v.dims[0]))
                else:
                    c = DataArray(c, v.coords)
            else:
                c = as_dataarray(c)
            ds = Dataset({"coeffs": c, "vars": v}).expand_dims("_term")
            expr = cls(ds, model)
            exprs.append(expr)

        if len(exprs) > 1:
            return merge(exprs, cls=cls)
        else:
            return exprs[0]

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

    def _from_scalarexpression_list(exprs, coords: DataArrayCoordinates, model):
        """
        Create a LinearExpression from a list of lists with different lengths.
        """
        shape = list(map(len, coords.values()))

        coeffs = array(tuple(zip_longest(*(e.coeffs for e in exprs), fillvalue=nan)))
        vars = array(tuple(zip_longest(*(e.vars for e in exprs), fillvalue=-1)))

        nterm = vars.shape[0]
        coeffs = coeffs.reshape((nterm, *shape))
        vars = vars.reshape((nterm, *shape))

        coeffs = DataArray(coeffs, coords, dims=("_term", *coords))
        vars = DataArray(vars, coords, dims=("_term", *coords))
        ds = Dataset({"coeffs": coeffs, "vars": vars}).transpose(..., "_term")

        return LinearExpression(ds, model)

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
        self,
        group,
        squeeze: "bool" = True,
        restore_coord_dims: "bool" = None,
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
        groups = ds.groupby(
            group=group, squeeze=squeeze, restore_coord_dims=restore_coord_dims
        )
        return LinearExpressionGroupby(groups, model=self.model)

    @deprecated("0.1.0", "0.1.2", details="Use groupby (followed by sum) instead.")
    def groupby_sum(self, group):
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
        if isinstance(group, pd.Series):
            logger.info("Converting group pandas.Series to xarray.DataArray")
            group = group.to_xarray()
        groups = xr.Dataset.groupby(self.data, group)

        def func(ds):
            ds = self._sum(ds, groups._group_dim)
            ds = ds.assign_coords(_term=np.arange(len(ds._term)))
            return ds

        return self.__class__(groups.map(func), self.model)  # .reset_index('_term')

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

    @deprecated("0.1.0", "0.1.2", details="Use rolling (followed by sum) instead.")
    def rolling_sum(self, **kwargs):
        """
        Rolling sum of the linear expression.

        Parameters
        ----------
        **kwargs :
            Keyword arguments passed to ``xarray.Dataset.rolling``.

        Returns
        -------
        linopy.LinearExpression
        """

        coeffs = xr.DataArray.rolling(self.coeffs, **kwargs).construct(
            "_rolling_term", keep_attrs=True
        )

        vars = xr.DataArray.rolling(self.vars, **kwargs).construct(
            "_rolling_term",
            fill_value=self._fill_value["vars"],
            keep_attrs=True,
        )

        ds = xr.Dataset({"coeffs": coeffs, "vars": vars})
        ds = (
            ds.rename(_term="_stacked_term")
            .stack(_term=["_stacked_term", "_rolling_term"])
            .reset_index("_term", drop=True)
        )
        return self.__class__(ds, self.model).assign_attrs(self.attrs)

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
        assert self.vars.size == self.coeffs.size
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
        data = self.data.transpose(..., "_term")

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

        return self.__class__(data.sel(_term=slice(0, nterm)), self.model)

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

    # TODO: make this return a LinearExpression (needs refactoring of __init__)
    def rename(self, name_dict=None, **names) -> Dataset:
        return self.data.rename(name_dict, **names)

    def __iter__(self):
        return self.data.__iter__()

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

    rename_dims = exprwrap(Dataset.rename_dims)

    roll = exprwrap(Dataset.roll)


def _pd_series_wo_index_name(ds):
    if isinstance(ds, pd.Series):
        if ds.index.name is None:
            return True
    return False


def _pd_dataframe_wo_axes_names(df):
    if isinstance(df, pd.DataFrame):
        if df.index.name is None and df.columns.name is None:
            return True
        elif df.index.name is None or df.columns.name is None:
            logger.warning(
                "Pandas DataFrame has only one labeled axis. "
                "This might lead to unexpected dimension alignment "
                "of the resulting expression."
            )
            return False
    return False


def merge(*exprs, dim="_term", cls=LinearExpression):
    """
    Merge multiple linear expression together.

    This function is a bit faster than summing over multiple linear expressions.

    Parameters
    ----------
    *exprs : tuple/list
        List of linear expressions to merge.
    dim : str
        Dimension along which the expressions should be concatenated.

    Returns
    -------
    res : linopy.LinearExpression
    """
    if len(exprs) == 1:
        exprs = exprs[0]  # assume one list of mergeable objects is given
    else:
        exprs = list(exprs)

    model = exprs[0].model
    exprs = [e.data if isinstance(e, cls) else e for e in exprs]

    if not all(len(expr._term) == len(exprs[0]._term) for expr in exprs[1:]):
        exprs = [expr.assign_coords(_term=np.arange(len(expr._term))) for expr in exprs]

    kwargs = dict(fill_value=cls._fill_value, coords="minimal", compat="override")
    ds = xr.concat(exprs, dim, **kwargs)
    if "_term" in ds.coords:
        ds = ds.reset_index("_term", drop=True)

    # ensure that coordinates for non-helper dims are explicit
    if isinstance(dim, str):
        if dim not in ds.coords and not dim.startswith("_"):
            ds = ds.assign_coords({dim: pd.RangeIndex(ds.dims[dim])})

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
        expr_string = print_single_expression(self.coeffs, self.vars, self.model)
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
        if not isinstance(other, (int, np.integer, float)):
            raise TypeError(
                "unsupported operand type(s) for *: " f"{type(self)} and {type(other)}"
            )

        return ScalarLinearExpression(
            tuple(other * c for c in self.coeffs), self.vars, self.model
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if not isinstance(other, (int, np.integer, float)):
            raise TypeError(
                "unsupported operand type(s) for /: " f"{type(self)} and {type(other)}"
            )
        return self.__mul__(1 / other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __le__(self, other):
        if not isinstance(other, (int, np.integer, float)):
            raise TypeError(
                "unsupported operand type(s) for <=: " f"{type(self)} and {type(other)}"
            )

        return constraints.AnonymousScalarConstraint(self, LESS_EQUAL, other)

    def __ge__(self, other):
        if not isinstance(other, (int, np.integer, float)):
            raise TypeError(
                "unsupported operand type(s) for >=: " f"{type(self)} and {type(other)}"
            )

        return constraints.AnonymousScalarConstraint(self, GREATER_EQUAL, other)

    def __eq__(self, other):
        if not isinstance(other, (int, np.integer, float)):
            raise TypeError(
                "unsupported operand type(s) for ==: " f"{type(self)} and {type(other)}"
            )

        return constraints.AnonymousScalarConstraint(self, EQUAL, other)

    def to_linexpr(self):
        coeffs = xr.DataArray(list(self.coeffs), dims="_term")
        vars = xr.DataArray(list(self.vars), dims="_term")
        ds = xr.Dataset({"coeffs": coeffs, "vars": vars})
        return LinearExpression(ds, self.model)
