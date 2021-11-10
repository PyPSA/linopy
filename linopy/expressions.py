#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linopy expressions module.
This module contains definition related to affine expressions.
"""

import numpy as np
import xarray as xr
from xarray import DataArray, Dataset

from linopy import variables


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

    def __init__(self, dataset=None):
        if dataset is not None:
            assert set(dataset) == {"coeffs", "vars"}
            (dataset,) = xr.broadcast(dataset)
            dataset = dataset.transpose(..., "_term")
        else:
            dataset = xr.Dataset({"coeffs": DataArray([]), "vars": DataArray([])})
            dataset = dataset.assign_coords(_term=[])
        super().__init__(dataset)

    # We have to set the _reduce_method to None, in order to overwrite basic
    # reduction functions as `sum`. There might be a better solution (?).
    _reduce_method = None

    def __repr__(self):
        """Get the string representation of the expression."""
        ds_string = self.to_dataset().__repr__().split("\n", 1)[1]
        ds_string = ds_string.replace("Data variables:\n", "Data:\n")
        nterm = getattr(self, "nterm", 0)
        return (
            f"Linear Expression with {nterm} term(s):\n"
            f"----------------------------------\n\n{ds_string}"
        )

    def _repr_html_(self):
        """Get the html representation of the expression."""
        # return self.__repr__()
        ds_string = self.to_dataset()._repr_html_()
        ds_string = ds_string.replace("Data variables:\n", "Data:\n")
        ds_string = ds_string.replace("xarray.Dataset", "linopy.LinearExpression")
        return ds_string

    def __add__(self, other):
        """Add a expression to others."""
        if isinstance(other, variables.Variable):
            other = LinearExpression.from_tuples((1, other))
        if not isinstance(other, LinearExpression):
            raise TypeError(
                "unsupported operand type(s) for +: " f"{type(self)} and {type(other)}"
            )
        res = LinearExpression(xr.concat([self, other], dim="_term"))
        return res

    def __sub__(self, other):
        """Subtract others form expression."""
        if isinstance(other, variables.Variable):
            other = LinearExpression.from_tuples((-1, other))
        elif isinstance(other, LinearExpression):
            other = -other
        else:
            raise TypeError(
                "unsupported operand type(s) for -: " f"{type(self)} and {type(other)}"
            )
        res = LinearExpression(xr.concat([self, other], dim="_term"))
        return res

    def __neg__(self):
        """Get the negative of the expression."""
        return LinearExpression(self.assign(coeffs=-self.coeffs))

    def __mul__(self, other):
        """Multiply the expr by a factor."""
        coeffs = other * self.coeffs
        assert coeffs.shape == self.coeffs.shape
        return LinearExpression(self.assign(coeffs=coeffs))

    def __rmul__(self, other):
        """Right-multiply the expr by a factor."""
        return self.__mul__(other)

    def to_dataset(self):
        """Convert the expression to a xarray.Dataset."""
        return Dataset(self)

    def sum(self, dims=None, keep_coords=False):
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
        if dims:
            dims = list(np.atleast_1d(dims))
        else:
            dims = [...]
        if "_term" in dims:
            dims.remove("_term")

        ds = (
            self.rename(_term="_stacked_term")
            .stack(_term=["_stacked_term"] + dims)
            .reset_index("_term", drop=True)
        )
        return LinearExpression(ds)

    def from_tuples(*tuples, chunk=None):
        """
        Create a linear expression by using tuples of coefficients and variables.

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
        ds_list = [Dataset({"coeffs": c, "vars": v}) for c, v in tuples]
        if len(ds_list) > 1:
            ds = xr.concat(ds_list, dim="_term", coords="minimal")
        else:
            ds = ds_list[0].expand_dims("_term")
        return LinearExpression(ds)

    def group_terms(self, group):
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
        groups = self.groupby(group)
        return groups.map(lambda ds: ds.sum(groups._group_dim))

    @property
    def nterm(self):
        """Get the number of terms in the linear expression."""
        return len(self._term)

    @property
    def shape(self):
        """Get the total shape of the linear expression."""
        assert self.vars.shape == self.coeffs.shape
        return self.vars.shape

    @property
    def size(self):
        """Get the total size of the linear expression."""
        assert self.vars.size == self.coeffs.size
        return self.vars.size

    def empty(self):
        """Get whether the linear expression is empty."""
        return self.shape == (0,)
