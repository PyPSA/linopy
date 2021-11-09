# -*- coding: utf-8 -*-
"""
Linopy variables module.
This module contains variable related definitions of the package.
"""

from dataclasses import dataclass
from typing import Any, Sequence, Union

from xarray import DataArray, Dataset

import linopy.expressions as expressions
from linopy.common import _merge_inplace


class Variable(DataArray):
    """
    Variable container for storing variable labels.

    The Variable class is a subclass of xr.DataArray hence most xarray functions
    can be applied to it. However most arithmetic operations are overwritten.
    Like this one can easily combine variables into a linear expression.


    Examples
    --------
    >>> m = Model()
    >>> x = m.add_variables(pd.Series([0, 0]), 1, name="x")
    >>> y = m.add_variables(4, pd.Series([8, 10]), name="y")

    Add variable together:

    >>> x + y

    ::

        Linear Expression with 2 term(s):
        ----------------------------------

        Dimensions:  (dim_0: 2, _term: 2)
        Coordinates:
            * dim_0    (dim_0) int64 0 1
            * _term    (_term) int64 0 1
        Data:
            coeffs   (dim_0, _term) int64 1 1 1 1
            vars     (dim_0, _term) int64 1 3 2 4


    Multiply them with a coefficient:

    >>> 3 * x

    ::

        Linear Expression with 1 term(s):
        ----------------------------------

        Dimensions:  (dim_0: 2, _term: 1)
        Coordinates:
            * _term    (_term) int64 0
            * dim_0    (dim_0) int64 0 1
        Data:
            coeffs   (dim_0, _term) int64 3 3
            vars     (dim_0, _term) int64 1 2


    Further operations like taking the negative and subtracting are supported.

    """

    __slots__ = ("_cache", "_coords", "_indexes", "_name", "_variable", "model")

    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop("model", None)
        super().__init__(*args, **kwargs)
        assert self.name is not None, "Variable data does not have a name."

    # We have to set the _reduce_method to None, in order to overwrite basic
    # reduction functions as `sum`. There might be a better solution (?).
    _reduce_method = None

    def to_array(self):
        """Convert the variable array to a xarray.DataArray."""
        return DataArray(self)

    def to_linexpr(self, coefficient=1):
        """Create a linear exprssion from the variables."""
        return expressions.LinearExpression.from_tuples((coefficient, self))

    def __repr__(self):
        """Get the string representation of the variables."""
        data_string = (
            "Variable labels:\n" + self.to_array().__repr__().split("\n", 1)[1]
        )
        extend_line = "-" * len(self.name)
        return (
            f"Variable '{self.name}':\n"
            f"------------{extend_line}\n\n"
            f"{data_string}"
        )

    def _repr_html_(self):
        """Get the html representation of the variables."""
        # return self.__repr__()
        data_string = self.to_array()._repr_html_()
        data_string = data_string.replace("xarray.DataArray", "linopy.Variable")
        return data_string

    def __neg__(self):
        """Calculate the negative of the variables (converts coefficients only)."""
        return self.to_linexpr(-1)

    def __mul__(self, coefficient):
        """Multiply variables with a coefficient."""
        return self.to_linexpr(coefficient)

    def __rmul__(self, coefficient):
        """Right-multiply variables with a coefficient."""
        return self.to_linexpr(coefficient)

    def __add__(self, other):
        """Add variables to linear expressions or other variables."""
        if isinstance(other, Variable):
            return expressions.LinearExpression.from_tuples((1, self), (1, other))
        elif isinstance(other, expressions.LinearExpression):
            return self.to_linexpr() + other
        else:
            raise TypeError(
                "unsupported operand type(s) for +: " f"{type(self)} and {type(other)}"
            )

    def __sub__(self, other):
        """Subtract linear expressions or other variables from the variables."""
        if isinstance(other, Variable):
            return expressions.LinearExpression.from_tuples((1, self), (-1, other))
        elif isinstance(other, expressions.LinearExpression):
            return self.to_linexpr() - other
        else:
            raise TypeError(
                "unsupported operand type(s) for -: " f"{type(self)} and {type(other)}"
            )

    def group_terms(self, group):
        """
        Sum variable over groups.

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
        return self.to_linexpr().group_terms(group)

    # would like to have this as a property, but this does not work apparently
    def get_upper_bound(self):
        """
        Get the upper bounds of the variables.
        The function raises an error in case no model is set as a reference.
        """
        if self.model is None:
            raise AttributeError("No reference model is assigned to the variable.")
        return self.model.variables.upper[self.name]

    def get_lower_bound(self):
        """
        Get the lower bounds of the variables.
        The function raises an error in case no model is set as a reference.
        """
        if self.model is None:
            raise AttributeError("No reference model is assigned to the variable.")
        return self.model.variables.lower[self.name]

    def sum(self, dims=None, keep_coords=False):
        """
        Sum the variables over all or a subset of dimensions.

        This stack all terms of the dimensions, that are summed over, together.
        The function works exactly in the same way as ``LinearExpression.sum()``.

        Parameters
        ----------
        dims : str/list, optional
            Dimension(s) to sum over. The default is None which results in all
            dimensions.
        keep_coords : bool, optional
            Whether to keep the coordinates of the stacked dimensions in a
            MultiIndex. The default is False.

        Returns
        -------
        linopy.LinearExpression
            Summed expression.
        """
        return self.to_linexpr().sum(dims, keep_coords)


@dataclass(repr=False)
class Variables:
    """
    A variables container used for storing multiple variable arrays.
    """

    labels: Dataset = Dataset()
    lower: Dataset = Dataset()
    upper: Dataset = Dataset()
    model: Any = None  # Model is not defined due to circular imports

    dataset_attrs = ["labels", "lower", "upper"]
    dataset_names = ["Variables labels", "Lower bounds", "Upper bounds"]

    def __getitem__(
        self, names: Union[str, Sequence[str]]
    ) -> Union[Variable, "Variables"]:
        if isinstance(names, str):
            return Variable(self.labels[names], model=self.model)

        return self.__class__(
            self.labels[names], self.lower[names], self.upper[names], self.model
        )

    def __repr__(self):
        """Return a string representation of the linopy model."""
        r = "linopy.model.Variables"
        line = "=" * len(r)
        r += f"\n{line}\n\n"
        for (k, K) in zip(self.dataset_attrs, self.dataset_names):
            s = getattr(self, k).__repr__().split("\n", 1)[1]
            s = s.replace("Data variables:\n", "Data:\n")
            line = "-" * (len(K) + 1)
            r += f"{K}:\n{line}\n{s}\n\n"
        return r

    def __iter__(self):
        return self.labels.__iter__()

    _merge_inplace = _merge_inplace

    def add(self, name, labels: DataArray, lower: DataArray, upper: DataArray):
        self._merge_inplace("labels", labels, name, fill_value=-1)
        self._merge_inplace("lower", lower, name)
        self._merge_inplace("upper", upper, name)

    def remove(self, name):
        for attr in self.dataset_attrs:
            ds = getattr(self, attr)
            if name in ds:
                setattr(self, attr, ds.drop_vars(name))
