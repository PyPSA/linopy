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
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr
from numpy import array, nan
from xarray import DataArray, Dataset
from xarray.core.dataarray import DataArrayCoordinates
from xarray.core.groupby import _maybe_reorder, peek_at

from linopy import constraints, variables
from linopy.common import as_dataarray


def exprwrap(method, *default_args, **new_default_kwargs):
    @functools.wraps(method)
    def _exprwrap(obj, *args, **kwargs):
        for k, v in new_default_kwargs.items():
            kwargs.setdefault(k, v)
        obj = Dataset(obj)
        return LinearExpression(method(obj, *default_args, *args, **kwargs))

    _exprwrap.__doc__ = f"Wrapper for the xarray {method} function for linopy.Variable"
    if new_default_kwargs:
        _exprwrap.__doc__ += f" with default arguments: {new_default_kwargs}"

    return _exprwrap


logger = logging.getLogger(__name__)


class LinearExpression(Dataset):
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

    __slots__ = ("_cache", "_coords", "_indexes", "_name", "_variable")

    fill_value = {"vars": -1, "coeffs": np.nan}

    def __init__(self, data_vars=None, coords=None, attrs=None):
        ds = Dataset(data_vars, coords, attrs)

        if not len(ds):
            vars = DataArray(np.array([], dtype=int), dims="_term")
            coeffs = DataArray(np.array([], dtype=float), dims="_term")
            ds = ds.assign(coeffs=coeffs, vars=vars)

        assert set(ds).issuperset({"coeffs", "vars"})
        if np.issubdtype(ds.vars, np.floating):
            ds["vars"] = ds.vars.fillna(-1).astype(int)
        (ds,) = xr.broadcast(ds)
        ds = ds.transpose(..., "_term")
        super().__init__(ds)

    # We have to set the _reduce_method to None, in order to overwrite basic
    # reduction functions as `sum`. There might be a better solution (?).
    _reduce_method = None

    # Disable array function, only function defined below are supported
    # and set priority higher than pandas/xarray/numpy
    __array_ufunc__ = None
    __array_priority__ = 10000

    def __repr__(self):
        """
        Get the string representation of the expression.
        """
        ds_string = self.to_dataset().__repr__().split("\n", 1)[1]
        ds_string = ds_string.replace("Data variables:\n", "Data:\n")
        nterm = getattr(self, "nterm", 0)
        return (
            f"Linear Expression with {nterm} term(s):\n"
            f"----------------------------------\n\n{ds_string}"
        )

    def _repr_html_(self):
        """
        Get the html representation of the expression.
        """
        # return self.__repr__()
        ds_string = self.to_dataset()._repr_html_()
        ds_string = ds_string.replace("Data variables:\n", "Data:\n")
        ds_string = ds_string.replace("xarray.Dataset", "linopy.LinearExpression")
        return ds_string

    def __add__(self, other):
        """
        Add a expression to others.
        """
        if isinstance(other, variables.Variable):
            other = LinearExpression.from_tuples((1, other))
        if not isinstance(other, LinearExpression):
            raise TypeError(
                "unsupported operand type(s) for +: " f"{type(self)} and {type(other)}"
            )
        return merge(self, other)

    def __sub__(self, other):
        """
        Subtract others form expression.
        """
        if isinstance(other, variables.Variable):
            other = LinearExpression.from_tuples((-1, other))
        elif isinstance(other, LinearExpression):
            other = -other
        else:
            raise TypeError(
                "unsupported operand type(s) for -: " f"{type(self)} and {type(other)}"
            )
        return merge(self, other)

    def __neg__(self):
        """
        Get the negative of the expression.
        """
        return LinearExpression(self.assign(coeffs=-self.coeffs))

    def __mul__(self, other):
        """
        Multiply the expr by a factor.
        """
        coeffs = other * self.coeffs
        assert coeffs.shape == self.coeffs.shape
        return LinearExpression(self.assign(coeffs=coeffs))

    def __rmul__(self, other):
        """
        Right-multiply the expr by a factor.
        """
        return self.__mul__(other)

    def __le__(self, rhs):
        return constraints.AnonymousConstraint(self, "<=", rhs)

    def __ge__(self, rhs):
        return constraints.AnonymousConstraint(self, ">=", rhs)

    def __eq__(self, rhs):
        return constraints.AnonymousConstraint(self, "=", rhs)

    def to_dataset(self):
        """
        Convert the expression to a xarray.Dataset.
        """
        return Dataset(self)

    def sum(self, dims=None, drop_zeros=False):
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
        if dims is None:
            vars = DataArray(self.vars.data.ravel(), dims="_term")
            coeffs = DataArray(self.coeffs.data.ravel(), dims="_term")
            ds = xr.Dataset({"vars": vars, "coeffs": coeffs})

        else:
            dims = [d for d in np.atleast_1d(dims) if d != "_term"]
            ds = (
                self.reset_index(dims, drop=True)
                .rename(_term="_stacked_term")
                .stack(_term=["_stacked_term"] + dims)
                .reset_index("_term", drop=True)
            )

        if drop_zeros:
            ds = ds.densify_terms()

        return self.__class__(ds)

    def from_tuples(*tuples, chunk=None):
        """
        Create a linear expression by using tuples of coefficients and
        variables.

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
        ds_list = []
        for (c, v) in tuples:
            if isinstance(v, variables.ScalarVariable):
                v = v.label
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
            ds_list.append(ds)

        if len(ds_list) > 1:
            return merge(ds_list)
        else:
            return LinearExpression(ds_list[0])

    def from_rule(model, rule, coords):
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
        if not isinstance(output, ScalarLinearExpression):
            msg = f"`rule` has to return ScalarLinearExpression not {type(output)}."
            raise TypeError(msg)

        combinations = product(*[c.values for c in coords.values()])
        exprs = [rule(model, *coord) for coord in combinations]
        return LinearExpression._from_scalarexpression_list(exprs, coords)

    def _from_scalarexpression_list(exprs, coords: DataArrayCoordinates):
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

        return LinearExpression(ds)

    def where(self, cond, **kwargs):
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
        if not kwargs.get("drop", False) and "other" not in kwargs:
            kwargs["other"] = self.fill_value
        return self.__class__(DataArray.where(self, cond, **kwargs))

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
        groups = xr.Dataset.groupby(self, group)

        def func(ds):
            ds = LinearExpression.sum(ds, groups._group_dim)
            ds = ds.to_dataset()
            ds = ds.assign_coords(_term=np.arange(len(ds._term)))
            return ds

        return LinearExpression(groups.map(func))  # .reset_index('_term')

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
            fill_value=self.fill_value["vars"],
            keep_attrs=True,
        )

        ds = xr.Dataset({"coeffs": coeffs, "vars": vars})
        ds = (
            ds.rename(_term="_stacked_term")
            .stack(_term=["_stacked_term", "_rolling_term"])
            .reset_index("_term", drop=True)
        )
        return self.__class__(ds).assign_attrs(self.attrs)

    @property
    def nterm(self):
        """
        Get the number of terms in the linear expression.
        """
        return len(self._term)

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
        self = self.transpose(..., "_term")

        data = self.coeffs.data
        axis = data.ndim - 1
        nnz = np.nonzero(data)
        nterm = (data != 0).sum(axis).max()

        mod_nnz = list(nnz)
        mod_nnz.pop(axis)

        remaining_axes = np.vstack(mod_nnz).T
        _, idx = np.unique(remaining_axes, axis=0, return_inverse=True)
        idx = list(idx)
        new_index = np.array([idx[:i].count(j) for i, j in enumerate(idx)])
        mod_nnz.insert(axis, new_index)

        vdata = np.full_like(data, -1)
        vdata[tuple(mod_nnz)] = self.vars.data[nnz]
        self.vars.data = vdata

        cdata = np.zeros_like(data)
        cdata[tuple(mod_nnz)] = self.coeffs.data[nnz]
        self.coeffs.data = cdata

        return self.sel(_term=slice(0, nterm))

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

    # Wrapped function which would convert variable to dataarray
    astype = exprwrap(Dataset.astype)

    bfill = exprwrap(Dataset.bfill)

    broadcast_like = exprwrap(Dataset.broadcast_like)

    coarsen = exprwrap(Dataset.coarsen)

    clip = exprwrap(Dataset.clip)

    ffill = exprwrap(Dataset.ffill)

    fillna = exprwrap(Dataset.fillna, value=fill_value)

    shift = exprwrap(Dataset.shift)

    reindex = exprwrap(Dataset.reindex, fill_value=fill_value)

    roll = exprwrap(Dataset.roll)

    rolling = exprwrap(Dataset.rolling)

    # TODO: explicitly disable `dangerous` functions
    conj = property()
    conjugate = property()
    count = property()
    cumsum = property()
    cumprod = property()
    cumulative_integrate = property()
    curvefit = property()
    diff = property()
    differentiate = property()
    groupby_bins = property()
    integrate = property()
    interp = property()
    polyfit = property()
    prod = property()


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


def merge(*exprs, dim="_term"):
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

    if not all(len(expr._term) == len(exprs[0]._term) for expr in exprs[1:]):
        exprs = [expr.assign_coords(_term=np.arange(len(expr._term))) for expr in exprs]

    exprs = [e.to_dataset() if isinstance(e, LinearExpression) else e for e in exprs]
    fill_value = LinearExpression.fill_value
    kwargs = dict(fill_value=fill_value, coords="minimal", compat="override")
    ds = xr.concat(exprs, dim, **kwargs)
    res = LinearExpression(ds)

    if "_term" in res.coords:
        res = res.reset_index("_term", drop=True)

    return res


@dataclass
class ScalarLinearExpression:
    coeffs: tuple
    vars: tuple
    coords: dict = None

    def __add__(self, other):
        if isinstance(other, variables.ScalarVariable):
            coeffs = self.coeffs + (1,)
            vars = self.vars + (other.label,)
            return ScalarLinearExpression(coeffs, vars)
        elif not isinstance(other, ScalarLinearExpression):
            raise TypeError(
                "unsupported operand type(s) for +: " f"{type(self)} and {type(other)}"
            )

        coeffs = self.coeffs + other.coeffs
        vars = self.vars + other.vars
        return ScalarLinearExpression(coeffs, vars)

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
            self.coeffs + tuple(-c for c in other.coeffs), self.vars + other.vars
        )

    def __neg__(self):
        return ScalarLinearExpression(tuple(-c for c in self.coeffs), self.vars)

    def __mul__(self, other):
        if not isinstance(other, (int, np.integer, float)):
            raise TypeError(
                "unsupported operand type(s) for *: " f"{type(self)} and {type(other)}"
            )

        return ScalarLinearExpression(tuple(other * c for c in self.coeffs), self.vars)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        return self.__mul__(1 / other)

    def __le__(self, other):
        if not isinstance(other, (int, np.integer, float)):
            raise TypeError(
                "unsupported operand type(s) for >=: " f"{type(self)} and {type(other)}"
            )

        return constraints.AnonymousScalarConstraint(self, "<=", other)

    def __ge__(self, other):
        if not isinstance(other, (int, np.integer, float)):
            raise TypeError(
                "unsupported operand type(s) for >=: " f"{type(self)} and {type(other)}"
            )

        return constraints.AnonymousScalarConstraint(self, ">=", other)

    def __eq__(self, other):
        if not isinstance(other, (int, np.integer, float)):
            raise TypeError(
                "unsupported operand type(s) for ==: " f"{type(self)} and {type(other)}"
            )

        return constraints.AnonymousScalarConstraint(self, "=", other)

    def to_linexpr(self):
        coeffs = xr.DataArray(list(self.coeffs), dims="_term")
        vars = xr.DataArray(list(self.vars), dims="_term")
        return LinearExpression({"coeffs": coeffs, "vars": vars})
