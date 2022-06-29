#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linopy expressions module.

This module contains definition related to affine expressions.
"""

import functools
import logging
from warnings import warn

import numpy as np
import pandas as pd
import xarray as xr
from xarray import DataArray, Dataset
from xarray.core.groupby import _maybe_reorder, peek_at

from linopy import variables
from linopy.common import as_dataarray


def exprwrap(method, *default_args, **new_default_kwargs):
    @functools.wraps(method)
    def _exprwrap(*args, **kwargs):
        for k, v in new_default_kwargs.items():
            kwargs.setdefault(k, v)
        return LinearExpression(method(*args, **kwargs))

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

    fill_value = {"vars": -1, "coeffs": np.nan}

    def __init__(self, dataset=None):
        if dataset is not None:
            assert set(dataset) == {"coeffs", "vars"}
            if np.issubdtype(dataset.vars, np.floating):
                dataset["vars"] = dataset.vars.fillna(-1).astype(int)
            (dataset,) = xr.broadcast(dataset)
            dataset = dataset.transpose(..., "_term")
        else:
            vars = DataArray(np.array([], dtype=int), dims="_term")
            coeffs = DataArray(np.array([], dtype=float), dims="_term")
            dataset = xr.Dataset({"coeffs": coeffs, "vars": vars})
        super().__init__(dataset)

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
        fill_value = {"vars": -1, "coeffs": np.nan}
        res = LinearExpression(
            xr.concat([self, other], dim="_term", fill_value=fill_value)
        )
        return res

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
        fill_value = {"vars": -1, "coeffs": np.nan}
        res = LinearExpression(
            xr.concat([self, other], dim="_term", fill_value=fill_value)
        )
        return res

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
        return AnonymousConstraint(self, "<=", rhs)

    def __ge__(self, rhs):
        return AnonymousConstraint(self, ">=", rhs)

    def __eq__(self, rhs):
        return AnonymousConstraint(self, "=", rhs)

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
            dims = list(np.atleast_1d(dims))

            if "_term" in dims:
                dims.remove("_term")

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
        >>> m = Model()
        >>> x = m.add_variables(pd.Series([0, 0]), 1)
        >>> m.add_variables(4, pd.Series([8, 10]))
        >>> expr = LinearExpression.from_tuples((10, x), (1, y))

        This is the same as calling ``10*x + y`` but a bit more performant.
        """
        # when assigning arrays to Datasets it is taken as coordinates
        # support numpy arrays and convert them to dataarrays
        ds_list = []
        for (c, v) in tuples:
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
            ds_list.append(Dataset({"coeffs": c, "vars": v}))

        if len(ds_list) > 1:
            ds = xr.concat(ds_list, dim="_term", coords="minimal", compat="override")
        else:
            ds = ds_list[0].expand_dims("_term")
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
        groups = xr.Dataset.groupby(self, group)

        def func(ds):
            ds = ds.sum(groups._group_dim)
            return ds.assign_coords(_term=np.arange(ds.nterm))

        # mostly taken from xarray/core/groupby.py
        applied = (func(ds) for ds in groups._iter_grouped())
        applied_example, applied = peek_at(applied)
        coord, dim, positions = groups._infer_concat_args(applied_example)
        combined = xr.concat(applied, dim, fill_value=self.fill_value)
        combined = _maybe_reorder(combined, dim, positions)
        # assign coord when the applied function does not return that coord
        if coord is not None and dim not in applied_example.dims:
            combined[coord.name] = coord
        combined = groups._maybe_restore_empty_groups(combined)
        res = groups._maybe_unstack(combined).reset_index("_term", drop=True)
        return self.__class__(res)

    def group_terms(self, group):
        warn(
            'The function "group_terms" was renamed to "groupby_sum" and will be remove in v0.0.10.'
        )
        return self.groupby_sum(group)

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
            "_rolling_term", fill_value=self.fill_value["vars"], keep_attrs=True
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

    fillna = exprwrap(Dataset.fillna)

    shift = exprwrap(Dataset.shift)

    reindex = exprwrap(Dataset.reindex, fill_value=fill_value)

    roll = exprwrap(Dataset.roll)

    # TODO: explicitly disable `dangerous` functions
    rolling = property()
    conj = property()
    conjugate = property()
    count = property()
    cumsum = property()
    cumprod = property()
    cumulative_integrate = property()
    curvefit = property()
    diff = property()
    differentiate = property()
    groupby = property()
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
    None.
    """

    if len(exprs) == 1:
        exprs = exprs[0]  # assume one list of mergeable objects is given
    else:
        exprs = list(exprs)

    if not all(len(expr._term) == len(exprs[0]._term) for expr in exprs[1:]):
        exprs = [expr.assign_coords(_term=np.arange(expr.nterm)) for expr in exprs]

    fill_value = LinearExpression.fill_value
    res = LinearExpression(xr.concat(exprs, dim, fill_value=fill_value))
    if "_term" in res.coords:
        res = res.reset_index("_term", drop=True)

    return res


class AnonymousConstraint:
    """
    A constraint container used for storing multiple constraint arrays.
    """

    __slots__ = ("lhs", "sign", "rhs")

    def __init__(self, lhs, sign, rhs):
        """
        Initialize a anonymous constraint.
        """
        self.lhs, self.rhs = xr.align(lhs, DataArray(rhs))
        self.sign = DataArray(sign)

    def __repr__(self):
        """
        Get the string representation of the expression.
        """
        lhs_string = self.lhs.to_dataset().__repr__()  # .split("\n", 1)[1]
        lhs_string = lhs_string.split("Data variables:\n", 1)[1]
        lhs_string = lhs_string.replace("    coeffs", "coeffs")
        lhs_string = lhs_string.replace("    vars", "vars")
        if self.rhs.size == 1:
            rhs_string = self.rhs.item()
        else:
            rhs_string = self.rhs.__repr__().split("\n", 1)[1]
        return (
            f"Anonymous Constraint:\n"
            f"---------------------\n"
            f"\n{lhs_string}"
            f"\n{self.sign.item()}"
            f"\n{rhs_string}"
        )
