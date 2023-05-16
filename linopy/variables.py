# -*- coding: utf-8 -*-
"""
Linopy variables module.

This module contains variable related definitions of the package.
"""

import functools
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Union

import dask
import numpy as np
import pandas as pd
from deprecation import deprecated
from numpy import floating, inf, issubdtype
from xarray import DataArray, Dataset, align, zeros_like
from xarray.core import indexing, utils

import linopy.expressions as expressions
from linopy.common import (
    dictsel,
    forward_as_properties,
    has_optimized_model,
    head_tail_range,
    is_constant,
    print_coord,
    print_single_variable,
)
from linopy.config import options


def varwrap(method, *default_args, **new_default_kwargs):
    @functools.wraps(method)
    def _varwrap(var, *args, **kwargs):
        for k, v in new_default_kwargs.items():
            kwargs.setdefault(k, v)
        return var.__class__(
            method(var.data, *default_args, *args, **kwargs), var.model, var.name
        )

    _varwrap.__doc__ = f"Wrapper for the xarray {method} function for linopy.Variable"
    if new_default_kwargs:
        _varwrap.__doc__ += f" with default arguments: {new_default_kwargs}"

    return _varwrap


def _var_unwrap(var):
    if isinstance(var, Variable):
        return var.data
    return var


@forward_as_properties(
    data=[
        "attrs",
        "coords",
        "indexes",
    ],
    labels=[
        "shape",
        "size",
        "dims",
        "ndim",
    ],
)
class Variable:
    """
    Variable container for storing variable labels.

    The Variable class is a subclass of xr.DataArray hence most xarray functions
    can be applied to it. However most arithmetic operations are overwritten.
    Like this one can easily combine variables into a linear expression.


    Examples
    --------
    >>> from linopy import Model
    >>> import pandas as pd
    >>> m = Model()
    >>> x = m.add_variables(pd.Series([0, 0]), 1, name="x")
    >>> y = m.add_variables(4, pd.Series([8, 10]), name="y")

    Add variable together:

    >>> x + y  # doctest: +SKIP
    Linear Expression with 2 term(s):
    ----------------------------------
    <BLANKLINE>
    Dimensions:  (dim_0: 2, _term: 2)
    Coordinates:
      * dim_0    (dim_0) int64 0 1
    Dimensions without coordinates: _term
    Data:
        coeffs   (dim_0, _term) int64 1 1 1 1
        vars     (dim_0, _term) int64 0 2 1 3

    Multiply them with a coefficient:

    >>> 3 * x  # doctest: +SKIP
    Linear Expression with 1 term(s):
    ----------------------------------
    <BLANKLINE>
    Dimensions:  (dim_0: 2, _term: 1)
    Coordinates:
      * dim_0    (dim_0) int64 0 1
    Dimensions without coordinates: _term
    Data:
        coeffs   (dim_0, _term) int64 3 3
        vars     (dim_0, _term) int64 0 1


    Further operations like taking the negative and subtracting are supported.
    """

    __slots__ = ("_data", "_model")
    __array_ufunc__ = None

    _fill_value = -1

    def __init__(self, data: Dataset, model: Any, name: str):
        """
        Initialize the Variable.

        Parameters
        ----------
        labels : xarray.Dataset
            data of the variable.
        model : linopy.Model
            Underlying model.
        """
        from linopy.model import Model

        if not isinstance(data, Dataset):
            raise ValueError(f"data must be a Dataset, got {type(data)}")

        if not isinstance(model, Model):
            raise ValueError(f"model must be a Model, got {type(model)}")

        # check that `labels`, `lower` and `upper`, `sign` and `mask` are in data
        for attr in ("labels", "lower", "upper"):
            if attr not in data:
                raise ValueError(f"missing '{attr}' in data")

        if "label_range" in data.attrs:
            data.assign_attrs(label_range=(data.labels.min(), data.labels.max()))

        self._data = data
        self._model = model

    def __getitem__(self, keys) -> "ScalarVariable":
        keys = (keys,) if not isinstance(keys, tuple) else keys
        assert all(map(pd.api.types.is_scalar, keys)), (
            "The get function of Variable is different as of xarray.DataArray. "
            "Set single values for each dimension in order to obtain a "
            "ScalarVariable. For all other purposes, use `.sel` and `.isel`."
        )
        if not self.labels.ndim:
            return ScalarVariable(self.labels.item(), self.model)
        assert self.labels.ndim == len(
            keys
        ), f"expected {self.labels.ndim} keys, got {len(keys)}."
        key = dict(zip(self.labels.dims, keys))
        selector = [self.labels.get_index(k).get_loc(v) for k, v in key.items()]
        return ScalarVariable(self.labels.data[tuple(selector)], self.model)

    @property
    def loc(self):
        return _LocIndexer(self)

    def to_pandas(self):
        return self.labels.to_pandas()

    def to_linexpr(self, coefficient=1):
        """
        Create a linear expression from the variables.
        """
        if isinstance(coefficient, (expressions.LinearExpression, Variable)):
            raise TypeError(f"unsupported type of coefficient: {type(coefficient)}")
        return expressions.LinearExpression.from_tuples((coefficient, self))

    def __repr__(self):
        """
        Print the variable arrays.
        """
        max_lines = options["display_max_rows"]
        dims = list(self.dims)
        dim_sizes = list(self.data.sizes.values())
        total_lines = int(np.prod(dim_sizes))
        lines_to_skip = total_lines - max_lines + 1 if total_lines > max_lines else 0
        masked_entries = self.mask.sum().values if self.mask is not None else 0
        lines = []

        def domain(self, lower, upper):
            if self.attrs["binary"]:
                return "∈ {0, 1}"
            elif self.attrs["integer"]:
                return f"∈ Z ⋂ [{lower},...,{upper}]"
            else:
                return f"∈ [{lower}, {upper}]"

        if dims:

            def generate_indices():
                if lines_to_skip > 0:
                    half_lines = max_lines // 2
                    for i in range(half_lines):
                        yield np.unravel_index(i, dim_sizes)
                    yield None
                    for i in range(total_lines - half_lines, total_lines):
                        yield np.unravel_index(i, dim_sizes)
                else:
                    for i in range(total_lines):
                        yield np.unravel_index(i, dim_sizes)

            for indices in generate_indices():
                if indices is None:
                    lines.append("\t\t...")
                else:
                    coord_values = ", ".join(
                        str(self.data[dims[i]].values[ind])
                        for i, ind in enumerate(indices)
                    )
                    if self.mask is None or self.mask.values[tuple(indices)]:
                        lower = self.lower.values[tuple(indices)]
                        upper = self.upper.values[tuple(indices)]
                        line = f"[{coord_values}]: {self.name}[{coord_values}] {domain(self, lower, upper)}"
                    else:
                        line = f"[{coord_values}]: None"
                    lines.append(line)

            shape_str = ", ".join(f"{d}: {s}" for d, s in zip(dims, dim_sizes))
            mask_str = f" - {masked_entries} masked entries" if masked_entries else ""
            lines.insert(
                0,
                f"Variable ({shape_str}){mask_str}\n{'-'* (len(shape_str) + len(mask_str) + 11)}",
            )
        else:
            lines.append(
                f"Variable\n{'-'*8}\n{self.name} {domain(self, self.lower.item(), self.upper.item())}"
            )

        return "\n".join(lines)

    def __neg__(self):
        """
        Calculate the negative of the variables (converts coefficients only).
        """
        return self.to_linexpr(-1)

    def __mul__(self, other):
        """
        Multiply variables with a coefficient.
        """
        if isinstance(other, (expressions.LinearExpression, Variable)):
            raise TypeError(
                "unsupported operand type(s) for *: "
                f"{type(self)} and {type(other)}. "
                "Non-linear expressions are not yet supported."
            )
        return self.to_linexpr(other)

    def __rmul__(self, other):
        """
        Right-multiply variables with a coefficient.
        """
        return self.to_linexpr(other)

    def __div__(self, other):
        """
        Divide variables with a coefficient.
        """
        if isinstance(other, (expressions.LinearExpression, Variable)):
            raise TypeError(
                "unsupported operand type(s) for /: "
                f"{type(self)} and {type(other)}. "
                "Non-linear expressions are not yet supported."
            )
        return self.to_linexpr(1 / other)

    def __truediv__(self, coefficient):
        """
        True divide variables with a coefficient.
        """
        return self.__div__(coefficient)

    def __add__(self, other):
        """
        Add variables to linear expressions or other variables.
        """
        if isinstance(other, Variable):
            return expressions.LinearExpression.from_tuples((1, self), (1, other))
        elif isinstance(other, expressions.LinearExpression):
            return self.to_linexpr() + other
        else:
            raise TypeError(
                "unsupported operand type(s) for +: " f"{type(self)} and {type(other)}"
            )

    def __radd__(self, other):
        # This is needed for using python's sum function
        if other == 0:
            return self
        else:
            return NotImplemented

    def __sub__(self, other):
        """
        Subtract linear expressions or other variables from the variables.
        """
        if isinstance(other, Variable):
            return expressions.LinearExpression.from_tuples((1, self), (-1, other))
        elif isinstance(other, expressions.LinearExpression):
            return self.to_linexpr() - other
        else:
            raise TypeError(
                "unsupported operand type(s) for -: " f"{type(self)} and {type(other)}"
            )

    def __le__(self, other):
        return self.to_linexpr().__le__(other)

    def __ge__(self, other):
        return self.to_linexpr().__ge__(other)

    def __eq__(self, other):
        return self.to_linexpr().__eq__(other)

    def __gt__(self, other):
        raise NotImplementedError(
            "Inequalities only ever defined for >= rather than >."
        )

    def __lt__(self, other):
        raise NotImplementedError(
            "Inequalities only ever defined for >= rather than >."
        )

    def groupby(
        self,
        group,
        squeeze: "bool" = True,
        restore_coord_dims: "bool" = None,
    ):
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
        return self.to_linexpr().groupby(
            group=group, squeeze=squeeze, restore_coord_dims=restore_coord_dims
        )

    def groupby_sum(self, group):
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
        return self.to_linexpr().groupby_sum(group)

    def rolling(
        self,
        dim: "Mapping[Any, int]" = None,
        min_periods: "int" = None,
        center: "bool | Mapping[Any, bool]" = False,
        **window_kwargs: "int",
    ) -> "expressions.LinearExpressionRolling":
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
        return self.to_linexpr().rolling(
            dim=dim, min_periods=min_periods, center=center, **window_kwargs
        )

    def rolling_sum(self, **kwargs):
        """
        Rolling sum of variable.

        Parameters
        ----------
        **kwargs :
            Keyword arguments passed to xarray.DataArray.rolling.

        Returns
        -------
        Rolling sum of variable.
        """
        return self.to_linexpr().rolling_sum(**kwargs)

    @property
    def name(self):
        """
        Return the name of the variable.
        """
        return self.attrs["name"]

    @property
    def labels(self):
        """
        Return the labels of the variable.
        """
        return self.data.labels

    @property
    def data(self):
        """
        Get the data of the variable.
        """
        # Needed for compatibility with linopy.merge
        return self._data

    @property
    def model(self):
        """
        Return the model of the variable.
        """
        return self._model

    @property
    def type(self):
        """
        Type of the variable.

        Returns
        -------
        str
            Type of the variable.
        """
        if self.attrs["integer"]:
            return "Integer Variable"
        elif self.attrs["binary"]:
            return "Binary Variable"
        else:
            return "Continuous Variable"

    @property
    def range(self):
        """
        Return the range of the variable.
        """
        return self.data.attrs["label_range"]

    @classmethod
    @property
    def fill_value(self):
        """
        Return the fill value of the variable.
        """
        return self._fill_value

    @property
    def mask(self):
        """
        Get the mask of the variable.

        The mask indicates on which coordinates the variable array is enabled
        (True) and disabled (False).

        Returns
        -------
        xr.DataArray
        """
        return self.data.get("mask")

    @property
    def upper(self):
        """
        Get the upper bounds of the variables.

        The function raises an error in case no model is set as a
        reference.
        """
        return self.data.upper

    @upper.setter
    @is_constant
    def upper(self, value):
        """
        Set the upper bounds of the variables.

        The function raises an error in case no model is set as a
        reference.
        """
        value = DataArray(value).broadcast_like(self.upper)
        if not set(value.dims).issubset(self.model.variables[self.name].dims):
            raise ValueError("Cannot assign new dimensions to existing variable.")
        self.data = self.data.assign(upper=value)

    @property
    def lower(self):
        """
        Get the lower bounds of the variables.

        The function raises an error in case no model is set as a
        reference.
        """
        return self.data.lower

    @lower.setter
    @is_constant
    def lower(self, value):
        """
        Set the lower bounds of the variables.

        The function raises an error in case no model is set as a
        reference.
        """
        value = DataArray(value).broadcast_like(self.lower)
        if not set(value.dims).issubset(self.model.variables[self.name].dims):
            raise ValueError("Cannot assign new dimensions to existing variable.")
        self._data = self.data.assign(lower=value)

    @property
    @has_optimized_model
    def sol(self):
        """
        Get the optimal values of the variable.

        The function raises an error in case no model is set as a
        reference or the model is not optimized.
        """
        if self.model.status != "ok":
            raise AttributeError("Underlying model not optimized.")
        return self.model.solution[self.name]

    def sum(self, dims=None):
        """
        Sum the variables over all or a subset of dimensions.

        This stack all terms of the dimensions, that are summed over, together.
        The function works exactly in the same way as ``LinearExpression.sum()``.

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
        return self.to_linexpr().sum(dims)

    def diff(self, dim, n=1):
        """
        Calculate the n-th order discrete difference along the given dimension.

        This function works exactly in the same way as ``LinearExpression.diff()``.

        Parameters
        ----------
        dim : str
            Dimension over which to calculate the finite difference.
        n : int, default: 1
            The number of times values are differenced.

        Returns
        -------
        linopy.LinearExpression
            Finite difference expression.
        """
        return self.to_linexpr().diff(dim, n)

    def where(self, cond, other=-1, **kwargs):
        """
        Filter variables based on a condition.

        This operation call ``xarray.DataArray.where`` but sets the default
        fill value to -1 and ensures preserving the linopy.Variable type.

        Parameters
        ----------
        cond : DataArray or callable
            Locations at which to preserve this object's values. dtype must be `bool`.
            If a callable, it must expect this object as its only parameter.
        other : scalar, DataArray, Variable, optional
            Value to use for locations in this object where ``cond`` is False.
            By default, these locations filled with -1.
        **kwargs :
            Keyword arguments passed to ``xarray.DataArray.where``

        Returns
        -------
        linopy.Variable
        """
        if isinstance(other, Variable):
            other = other.labels
        elif isinstance(other, ScalarVariable):
            other = other.label
        return self.__class__(self.labels.where(cond, other, **kwargs), self.model)

    def ffill(self, dim, limit=None):
        """
        Forward fill the variable along a dimension.

        This operation call ``xarray.DataArray.ffill`` but ensures preserving
        the linopy.Variable type.

        Parameters
        ----------
        dim : str
            Dimension over which to forward fill.
        limit : int, optional
            Maximum number of consecutive NaN values to forward fill. Must be greater than or equal to 0.

        Returns
        -------
        linopy.Variable
        """
        labels = (
            self.labels.where(self.labels != -1)
            .ffill(dim, limit=limit)
            .fillna(-1)
            .astype(int)
        )
        return self.__class__(labels, self.model)

    def bfill(self, dim, limit=None):
        """
        Backward fill the variable along a dimension.

        This operation call ``xarray.DataArray.bfill`` but ensures preserving
        the linopy.Variable type.

        Parameters
        ----------
        dim : str
            Dimension over which to backward fill.
        limit : int, optional
            Maximum number of consecutive NaN values to backward fill. Must be greater than or equal to 0.

        Returns
        -------
        linopy.Variable
        """
        labels = (
            self.labels.where(self.labels != -1)
            .bfill(dim, limit=limit)
            .fillna(-1)
            .astype(int)
        )
        return self.__class__(labels, self.model)

    def sanitize(self):
        """
        Sanitize variable by ensuring int dtype with fill value of -1.

        Returns
        -------
        linopy.Variable
        """
        if issubdtype(self.labels.dtype, floating):
            return self.__class__(self.labels.fillna(-1).astype(int), self.model)
        return self

    def equals(self, other):
        return self.labels.equals(_var_unwrap(other))

    # Wrapped function which would convert variable to dataarray
    assign_attrs = varwrap(Dataset.assign_attrs)

    assign_coords = varwrap(Dataset.assign_coords)

    broadcast_like = varwrap(Dataset.broadcast_like)

    compute = varwrap(Dataset.compute)

    drop = varwrap(Dataset.drop)

    drop_sel = varwrap(Dataset.drop_sel)

    drop_isel = varwrap(Dataset.drop_isel)

    fillna = varwrap(Dataset.fillna)

    sel = varwrap(Dataset.sel)

    isel = varwrap(Dataset.isel)

    shift = varwrap(Dataset.shift, fill_value=-1)

    rename = varwrap(Dataset.rename)

    roll = varwrap(Dataset.roll)


class _LocIndexer:
    __slots__ = ("variable",)

    def __init__(self, variable: Variable):
        self.variable = variable

    def __getitem__(self, key) -> Dataset:
        if not utils.is_dict_like(key):
            # expand the indexer so we can handle Ellipsis
            labels = indexing.expanded_indexer(key, self.variable.ndim)
            key = dict(zip(self.variable.dims, labels))
        return self.variable.sel(key)


@dataclass(repr=False)
@forward_as_properties(
    labels=[
        "attrs",
        "coords",
        "indexes",
        "dims",
    ]
)
class Variables:
    """
    A variables container used for storing multiple variable arrays.
    """

    variables: Dict[str, Variable] = field(default_factory=dict)
    model: Any = None  # Model is not defined due to circular imports

    dataset_attrs = ["labels", "lower", "upper"]
    dataset_names = ["Labels", "Lower bounds", "Upper bounds"]

    def __getitem__(
        self, names: Union[str, Sequence[str]]
    ) -> Union[Variable, "Variables"]:
        if isinstance(names, str):
            return self.variables[names]

        return self.__class__(
            {name: self.variables[name] for name in names}, self.model
        )

    def __getattr__(self, name: str) -> Variable:
        try:
            return self.variables[name]
        except KeyError:
            raise AttributeError(f"Variable {name} not found.")

    def __repr__(self):
        """
        Return a string representation of the linopy model.
        """
        r = "linopy.model.Variables"
        line = "-" * len(r)
        r += f"\n{line}\n\n"

        labelprint = self.labels.__repr__()
        coordspattern = r"(?s)(?<=\<xarray\.Dataset\>\n).*?(?=Data variables:)"
        r += re.search(coordspattern, labelprint).group()
        r += "Variables:\n"
        for name in self.labels:
            r += f"  *  {name} ({', '.join(self.labels[name].coords)})\n"
        return r

    def __iter__(self):
        return self.labels.__iter__()

    def _ipython_key_completions_(self):
        """
        Provide method for the key-autocompletions in IPython.

        See
        http://ipython.readthedocs.io/en/stable/config/integrating.html#tab-completion
        For the details.
        """
        return list(self)

    def add(self, variable):
        """
        Add a variable to the variables constrainer.
        """
        self.variables[variable.name] = variable

    def remove(self, name):
        """
        Remove variable `name` from the variables.
        """
        self.variables.pop(name)

    @property
    def labels(self):
        """
        Get the labels of all variables.
        """
        return Dataset({name: var.labels for name, var in self.variables.items()})

    @property
    def lower(self):
        """
        Get the lower bounds of all variables.
        """
        return Dataset({name: var.lower for name, var in self.variables.items()})

    @property
    def upper(self):
        """
        Get the upper bounds of all variables.
        """
        return Dataset({name: var.upper for name, var in self.variables.items()})

    @property
    def sign(self):
        """
        Get the signs of all variables.
        """
        return Dataset({name: var.sign for name, var in self.variables.items()})

    @property
    def nvars(self):
        """
        Get the number all variables which were at some point added to the
        model.

        These also include variables with missing labels.
        """
        return self.ravel("labels", filter_missings=True).shape[0]

    @property
    def _binary_variables(self):
        return [v for v in self if self[v].attrs["binary"]]

    @property
    def _non_binary_variables(self):
        return [v for v in self if not self[v].attrs["binary"]]

    @property
    def binaries(self):
        """
        Get all binary variables.
        """
        return self[self._binary_variables]

    @property
    def non_binaries(self):
        """
        Get all non-binary variables.
        """
        return self[self._non_binary_variables]

    @property
    def _integer_variables(self):
        return [v for v in self if self[v].attrs["integer"]]

    @property
    def integers(self):
        """
        Get all integers variables.
        """
        return self[self._integer_variables]

    def get_name_by_label(self, label):
        """
        Get the variable name of the variable containing the passed label.

        Parameters
        ----------
        label : int
            Integer label within the range [0, MAX_LABEL] where MAX_LABEL is the last assigned
            variable label.

        Raises
        ------
        ValueError
            If label is not contained by any variable.

        Returns
        -------
        name : str
            Name of the containing variable.
        """
        if not isinstance(label, (float, int)) or label < 0:
            raise ValueError("Label must be a positive number.")
        for name, labels in self.labels.items():
            if label in labels:
                return name
        raise ValueError(f"No variable found containing the label {label}.")

    def get_label_range(self, name: str):
        """
        Get starting and ending label for a variable.
        """
        return self[name].range

    def get_label_position(self, values):
        """
        Get tuple of name and coordinate for variable labels.
        """
        coords = []
        return_list = True
        if not isinstance(values, Iterable):
            values = [values]
            return_list = False

        for value in values:
            for name, labels in self.labels.items():
                start, stop = self.get_label_range(name)

                if value >= start and value < stop:
                    index = np.unravel_index(value - start, labels.shape)

                    # Extract the coordinates from the indices
                    coord = {
                        dim: labels.indexes[dim][i]
                        for dim, i in zip(labels.dims, index)
                    }

                    # Add the name of the DataArray and the coordinates to the result list
                    coords.append((name, coord))

        return coords if return_list else coords[0]

    def iter_ravel(self, key, filter_missings=False):
        """
        Create an generator which iterates over all arrays in `key` and
        flattens them.

        Parameters
        ----------
        key : str/Dataset
            Key to be iterated over. Optionally pass a dataset which is
            broadcastable to the variable labels.
        filter_missings : bool, optional
            Filter out values where the variables labels are -1. This will
            raise an error if the filtered data still contains nan's.
            When enabled, the data is loaded into memory. The default is False.


        Yields
        ------
        flat : np.array/dask.array
        """
        if isinstance(key, str):
            ds = getattr(self, key)
        elif isinstance(key, Dataset):
            ds = key
        else:
            raise TypeError("Argument `key` must be of type string or xarray.Dataset")

        for name, labels in self.labels.items():
            broadcasted = ds[name].broadcast_like(labels)
            if labels.chunks is not None:
                broadcasted = broadcasted.chunk(labels.chunks)

            if filter_missings:
                flat = np.ravel(broadcasted)
                flat = flat[np.ravel(labels) != -1]
                if pd.isna(flat).any():
                    ds_name = self.dataset_names[self.dataset_attrs.index(key)]
                    err = f"{ds_name} of variable '{name}' contains nan's."
                    raise ValueError(err)
            else:
                flat = broadcasted.data.ravel()
            yield flat

    def ravel(self, key, filter_missings=False, compute=True):
        """
        Ravel and concate all arrays in `key` while aligning to
        `broadcast_like`.

        Parameters
        ----------
        key : str/Dataset
            Key to be iterated over. Optionally pass a dataset which is
            broadcastable to `broadcast_like`.
        broadcast_like : str, optional
            Name of the dataset to which the input data in `key` is aligned to.
            The default is "labels".
        filter_missings : bool, optional
            Filter out values where `broadcast_like` data is -1.
            The default is False.
        compute : bool, optional
            Whether to compute lazy data. The default is False.

        Returns
        -------
        flat
            One dimensional data with all values in `key`.
        """
        res = np.concatenate(list(self.iter_ravel(key, filter_missings)))
        if compute:
            return dask.compute(res)[0]
        else:
            return res

    def get_blocks(self, blocks: DataArray):
        """
        Get a dataset of same shape as variables.labels indicating the blocks.
        """
        dim = blocks.dims[0]
        assert dim in self.labels.dims, "Block dimension not in variables."

        block_map = zeros_like(self.labels, dtype=blocks.dtype)
        for name, variable in self.labels.items():
            if dim in variable.dims:
                block_map[name] = blocks.broadcast_like(variable)
        return block_map.where(self.labels != -1, -1)

    def blocks_to_blockmap(self, block_map, dtype=np.int8):
        """
        Get a one-dimensional array mapping the variables to blocks.
        """
        # non-assigned variables are assumed to be masked, insert -1
        res = np.full(self.model._xCounter + 1, -1, dtype=dtype)
        for name, labels in self.labels.items():
            res[np.ravel(labels)] = np.ravel(block_map[name])
        res[-1] = -1
        return res


class ScalarVariable:
    """
    A scalar variable container.

    In contrast to the Variable class, a ScalarVariable only contains
    only one label. Use this class to create a expression or constraint
    in a rule.
    """

    __slots__ = ("_label", "_model")

    def __init__(self, label: int, model: Any):
        self._label = label
        self._model = model

    def __repr__(self) -> str:
        if self.label == -1:
            return "ScalarVariable: None"
        name, coord = self.model.variables.get_label_position(self.label)
        coord_string = print_coord(coord)
        return f"ScalarVariable: {name}{coord_string}"

    @property
    def label(self):
        """
        Get the label of the variable.
        """
        return self._label

    @property
    def model(self):
        """
        Get the model to which the variable belongs.
        """
        return self._model

    def to_scalar_linexpr(self, coeff=1):
        if not isinstance(coeff, (int, np.integer, float)):
            raise TypeError(f"Coefficient must be a numeric value, got {type(coeff)}.")
        return expressions.ScalarLinearExpression((coeff,), (self.label,), self.model)

    def to_linexpr(self, coeff=1):
        return self.to_scalar_linexpr(coeff).to_linexpr()

    def __neg__(self):
        return self.to_scalar_linexpr(-1)

    def __add__(self, other):
        return self.to_scalar_linexpr(1) + other

    def __radd__(self, other):
        # This is needed for using python's sum function
        if other == 0:
            return self
        else:
            return NotImplemented

    def __sub__(self, other):
        return self.to_scalar_linexpr(1) - other

    def __mul__(self, coeff):
        return self.to_scalar_linexpr(coeff)

    def __rmul__(self, coeff):
        return self.to_scalar_linexpr(coeff)

    def __div__(self, coeff):
        return self.to_scalar_linexpr(1 / coeff)

    def __truediv__(self, coeff):
        return self.__div__(coeff)

    def __le__(self, other):
        return self.to_scalar_linexpr(1).__le__(other)

    def __ge__(self, other):
        return self.to_scalar_linexpr(1).__ge__(other)

    def __eq__(self, other):
        return self.to_scalar_linexpr(1).__eq__(other)

    def __gt__(self, other):
        raise NotImplementedError(
            "Inequalities only ever defined for >= rather than >."
        )

    def __lt__(self, other):
        raise NotImplementedError(
            "Inequalities only ever defined for >= rather than >."
        )
