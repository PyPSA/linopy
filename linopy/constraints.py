# -*- coding: utf-8 -*-
"""
Linopy constraints module.

This module contains implementations for the Constraint{s} class.
"""

import re
from dataclasses import dataclass
from itertools import product
from typing import Any, Sequence, Union

import dask
import numpy as np
import pandas as pd
import xarray as xr
from numpy import arange, array
from scipy.sparse import coo_matrix
from xarray import DataArray, Dataset

from linopy import expressions, variables
from linopy.common import (
    _merge_inplace,
    has_assigned_model,
    has_optimized_model,
    is_constant,
    replace_by_map,
)


class Constraint(DataArray):
    """
    Constraint container for storing constraint labels.

    The Constraint class is a subclass of xr.DataArray hence most xarray
    functions can be applied to it.
    """

    __slots__ = ("_cache", "_coords", "_indexes", "_name", "_variable", "model")

    def __init__(self, *args, **kwargs):

        # workaround until https://github.com/pydata/xarray/pull/5984 is merged
        if isinstance(args[0], DataArray):
            da = args[0]
            args = (da.data, da.coords)
            kwargs.update({"attrs": da.attrs, "name": da.name})

        self.model = kwargs.pop("model", None)
        super().__init__(*args, **kwargs)
        assert self.name is not None, "Constraint data does not have a name."

    # We have to set the _reduce_method to None, in order to overwrite basic
    # reduction functions as `sum`. There might be a better solution (?).
    _reduce_method = None

    # Disable array function, only function defined below are supported
    # and set priority higher than pandas/xarray/numpy
    __array_ufunc__ = None
    __array_priority__ = 10000

    def __repr__(self):
        """
        Get the string representation of the constraints.
        """
        data_string = (
            "Constraint labels:\n" + self.to_array().__repr__().split("\n", 1)[1]
        )
        extend_line = "-" * len(self.name)
        return (
            f"Constraint '{self.name}':\n"
            f"--------------{extend_line}\n\n"
            f"{data_string}"
        )

    def _repr_html_(self):
        """
        Get the html representation of the variables.
        """
        # return self.__repr__()
        data_string = self.to_array()._repr_html_()
        data_string = data_string.replace("xarray.DataArray", "linopy.Constraint")
        return data_string

    def to_array(self):
        """
        Convert the variable array to a xarray.DataArray.
        """
        return DataArray(self)

    @property
    @has_assigned_model
    def coeffs(self):
        """
        Get the left-hand-side coefficients of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        return self.model.constraints.coeffs[self.name]

    @coeffs.setter
    @has_assigned_model
    def coeffs(self, value):
        term_dim = self.name + "_term"
        value = DataArray(value).broadcast_like(self.vars, exclude=[term_dim])
        try:
            self.model.constraints.coeffs[self.name] = value
        except ValueError:
            coeffs = self.model.constraints.coeffs
            coeffs = coeffs.assign_coords({term_dim: coeffs[term_dim]})
            value = value.assign_coords({term_dim: value[term_dim]})
            coeffs[self.name] = value
            coeffs = coeffs.reset_index(term_dim, drop=True)
            self.model.constraints.coeffs = coeffs

    @property
    @has_assigned_model
    def vars(self):
        """
        Get the left-hand-side variables of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        return self.model.constraints.vars[self.name]

    @vars.setter
    @has_assigned_model
    def vars(self, value):
        term_dim = self.name + "_term"
        value = DataArray(value).broadcast_like(self.coeffs, exclude=[term_dim])
        try:
            self.model.constraints.vars[self.name] = value
        except ValueError:
            vars = self.model.constraints.vars
            vars = vars.assign_coords({term_dim: vars[term_dim]})
            value = value.assign_coords({term_dim: value[term_dim]})
            vars[self.name] = value
            vars = vars.reset_index(term_dim, drop=True)
            self.model.constraints.vars = vars

    @property
    @has_assigned_model
    def lhs(self):
        """
        Get the left-hand-side linear expression of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        term_dim = self.name + "_term"
        coeffs = self.coeffs.rename({term_dim: "_term"})
        vars = self.vars.rename({term_dim: "_term"})
        return expressions.LinearExpression(Dataset({"coeffs": coeffs, "vars": vars}))

    @lhs.setter
    @has_assigned_model
    def lhs(self, value):
        if not isinstance(value, expressions.LinearExpression):
            raise TypeError("Assigned lhs must be a LinearExpression.")

        value = value.rename(_term=self.name + "_term")
        self.coeffs = value.coeffs
        self.vars = value.vars

    @property
    @has_assigned_model
    def sign(self):
        """
        Get the signs of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        return self.model.constraints.sign[self.name]

    @sign.setter
    @has_assigned_model
    @is_constant
    def sign(self, value):
        value = DataArray(value).broadcast_like(self)
        if (value == "==").any():
            raise ValueError('Sign "==" not supported, use "=" instead.')
        self.model.constraints.sign[self.name] = value

    @property
    @has_assigned_model
    def rhs(self):
        """
        Get the right hand side constants of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        return self.model.constraints.rhs[self.name]

    @rhs.setter
    @has_assigned_model
    @is_constant
    def rhs(self, value):
        if isinstance(value, (variables.Variable, expressions.LinearExpression)):
            raise TypeError(f"Assigned rhs must be a constant, got {type(value)}).")
        value = DataArray(value).broadcast_like(self)
        self.model.constraints.rhs[self.name] = value

    @property
    @has_optimized_model
    def dual(self):
        """
        Get the dual values of the constraint.

        The function raises an error in case no model is set as a
        reference or the model status is not okay.
        """
        if not list(self.model.dual):
            raise AttributeError(
                "Underlying is optimized but does not have dual values stored."
            )
        return self.model.dual[self.name]


@dataclass(repr=False)
class Constraints:
    """
    A constraint container used for storing multiple constraint arrays.
    """

    labels: Dataset = Dataset()
    coeffs: Dataset = Dataset()
    vars: Dataset = Dataset()
    sign: Dataset = Dataset()
    rhs: Dataset = Dataset()
    blocks: Dataset = Dataset()
    model: Any = None  # Model is not defined due to circular imports

    dataset_attrs = ["labels", "coeffs", "vars", "sign", "rhs"]
    dataset_names = [
        "Labels",
        "Left-hand-side coefficients",
        "Left-hand-side variables",
        "Signs",
        "Right-hand-side constants",
    ]

    def __repr__(self):
        """
        Return a string representation of the linopy model.
        """
        r = "linopy.model.Constraints"
        line = "-" * len(r)
        r += f"\n{line}\n\n"
        # matches string between "Data variables" and "Attributes"/end of string
        coordspattern = r"(?s)(?<=\<xarray\.Dataset\>\n).*?(?=Data variables:)"
        datapattern = r"(?s)(?<=Data variables:).*?(?=($|\nAttributes))"
        for (k, K) in zip(self.dataset_attrs, self.dataset_names):
            orig = getattr(self, k).__repr__()
            if k == "labels":
                r += re.search(coordspattern, orig).group() + "\n"
            data = re.search(datapattern, orig).group()
            # drop first line which includes counter for long ds
            data = data.split("\n", 1)[1]
            line = "-" * (len(K) + 1)
            r += f"{K}:\n{data}\n\n"
        return r

    def __getitem__(
        self, names: Union[str, Sequence[str]]
    ) -> Union[Constraint, "Constraints"]:
        if isinstance(names, str):
            return Constraint(self.labels[names], model=self.model)

        return self.__class__(
            self.labels[names],
            self.coeffs[names],
            self.vars[names],
            self.sign[names],
            self.rhs[names],
            self.model,
        )

    def __iter__(self):
        return self.labels.__iter__()

    _merge_inplace = _merge_inplace

    def _ipython_key_completions_(self):
        """
        Provide method for the key-autocompletions in IPython.

        See http://ipython.readthedocs.io/en/stable/config/integrating.html#tab-completion
        For the details.
        """
        return list(self)

    def add(
        self,
        name,
        labels: DataArray,
        coeffs: DataArray,
        vars: DataArray,
        sign: DataArray,
        rhs: DataArray,
    ):
        """
        Add constraint `name`.
        """
        self._merge_inplace("labels", labels, name, fill_value=-1)
        self._merge_inplace("coeffs", coeffs, name)
        self._merge_inplace("vars", vars, name, fill_value=-1)
        self._merge_inplace("sign", sign, name)
        self._merge_inplace("rhs", rhs, name)

    def remove(self, name):
        """
        Remove constraint `name` from the constraints.
        """
        for attr in self.dataset_attrs:
            setattr(self, attr, getattr(self, attr).drop_vars(name))

    @property
    def coefficientrange(self):
        """
        Coefficient range of the constraint.
        """
        return (
            xr.concat(
                [self.coeffs.min(), self.coeffs.max()],
                dim=pd.Index(["min", "max"]),
            )
            .to_dataframe()
            .T
        )

    @property
    def ncons(self):
        """
        Get the number all constraints which were at some point added to the
        model.

        These also include constraints with missing labels.
        """
        return self.ravel("labels", filter_missings=True).shape[0]

    @property
    def inequalities(self):
        """
        Get the subset of constraints which are purely inequalities.
        """
        return self[[n for n, s in self.sign.items() if s in ("<=", ">=")]]

    @property
    def equalities(self):
        """
        Get the subset of constraints which are purely equalities.
        """
        return self[[n for n, s in self.sign.items() if s in ("=", "==")]]

    def sanitize_zeros(self):
        """
        Filter out terms with zero coefficient.
        """
        for name in self:
            term_dim = name + "_term"
            not_zero = self.coeffs[name] != 0
            self.vars[name] = self.vars[name].where(not_zero, -1)
            self.coeffs[name] = self.coeffs[name].where(not_zero)

    def sanitize_missings(self):
        """
        Set constraints labels to -1 where all variables in the lhs are
        missing.
        """
        for name in self:
            term_dim = name + "_term"
            contains_non_missing = (self.vars[name] != -1).any(term_dim)
            self.labels[name] = self.labels[name].where(contains_non_missing, -1)

    def get_name_by_label(self, label):
        """
        Get the constraint name of the constraint containing the passed label.

        Parameters
        ----------
        label : int
            Integer label within the range [0, MAX_LABEL] where MAX_LABEL is the last assigned
            constraint label.

        Raises
        ------
        ValueError
            If label is not contained by any constraint.

        Returns
        -------
        name : str
            Name of the containing constraint.
        """
        if not isinstance(label, (float, int)) or label < 0:
            raise ValueError("Label must be a positive number.")
        for name, labels in self.labels.items():
            if label in labels:
                return name
        raise ValueError(f"No constraint found containing the label {label}.")

    def get_blocks(self, block_map):
        """
        Get a dataset of same shape as constraints.labels with block values.

        Let N be the number of blocks.
        The following cases are considered:

            * where are all vars are -1, the block is -1
            * where are all vars are 0, the block is 0
            * where all vars are n, the block is n
            * where vars are n or 0 (both present), the block is n
            * N+1 otherwise
        """
        N = block_map.max()
        var_blocks = replace_by_map(self.vars, block_map)
        res = xr.full_like(self.labels, N + 1, dtype=block_map.dtype)

        for name, entries in var_blocks.items():
            term_dim = f"{name}_term"

            not_zero = entries != 0
            not_missing = entries != -1
            for n in range(N + 1):
                not_n = entries != n
                mask = not_n & not_zero & not_missing
                res[name] = res[name].where(mask.any(term_dim), n)

            res[name] = res[name].where(not_missing.any(term_dim), -1)
            res[name] = res[name].where(not_zero.any(term_dim), 0)

        self.blocks = res
        self.var_blocks = var_blocks
        return self.blocks

    def iter_ravel(self, key, broadcast_like="labels", filter_missings=False):
        """
        Create an generator which iterates over all arrays in `key` and
        flattens them.

        Parameters
        ----------
        key : str/Dataset
            Key to be iterated over. Optionally pass a dataset which is
            broadcastable to `broadcast_like`.
        broadcast_like : str, optional
            Name of the dataset to which the input data in `key` is aligned to.
            Must be one of "labels", "vars". The default is "labels".
        filter_missings : bool, optional
            Filter out values where `labels` data is -1. If broadcast is `vars`
            also values where `vars` is -1 are filtered. This will raise an
            error if the filtered data still contains nan's.
            When enabled, the data is load into memory. The default is False.


        Yields
        ------
        flat : np.array/dask.array
        """
        if isinstance(key, str):
            ds = getattr(self, key)
        elif isinstance(key, xr.Dataset):
            ds = key
        else:
            raise TypeError("Argument `key` must be of type string or xarray.Dataset")

        assert broadcast_like in ["labels", "vars"]

        for name, values in getattr(self, broadcast_like).items():

            broadcasted = ds[name].broadcast_like(values)
            if values.chunks is not None:
                broadcasted = broadcasted.chunk(values.chunks)

            if filter_missings:
                flat = np.ravel(broadcasted)
                mask = np.ravel(values) != -1
                if broadcast_like != "labels":
                    labels = np.ravel(self.labels[name].broadcast_like(values))
                    mask &= labels != -1
                flat = flat[mask]
                if pd.isna(flat).any():
                    names = self.dataset_names
                    ds_name = names[self.dataset_attrs.index(key)]
                    bc_name = names[self.dataset_attrs.index(broadcast_like)]
                    err = (
                        f"{ds_name} of constraint '{name}' are missing (nan) "
                        f"where {bc_name.lower()} are defined (not -1)."
                    )
                    raise ValueError(err)
            else:
                flat = broadcasted.data.ravel()
            yield flat

    def ravel(self, key, broadcast_like="labels", filter_missings=False, compute=True):
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
        res = list(self.iter_ravel(key, broadcast_like, filter_missings))
        res = np.concatenate(res)
        if compute:
            return dask.compute(res)[0]
        else:
            return res

    def to_matrix(self, filter_missings=False):
        """
        Construct a constraint matrix in sparse format.

        Missing values, i.e. -1 in labels and vars, are ignored filtered out.

        If filter_missings is set to True, the index of the rows and columns
        correspond to the constraint and variable labels stored in the model.
        If set to False, the rows correspond to the constraints given by
        `m.constraints.ravel('labels', filter_missings=True)` and columns to
        `m.variables.ravel('labels', filter_missings=True)` where `m` is the
        underlying model. The matrix has then a shape of (`m.ncons`, `m.nvars`).
        """
        self.sanitize_missings()

        keys = ["coeffs", "labels", "vars"]
        data, rows, cols = [
            self.ravel(k, broadcast_like="vars", filter_missings=True) for k in keys
        ]
        shape = (self.model._cCounter, self.model._xCounter)

        if filter_missings:
            # We have to map the variables to the filtered layout
            clabels = self.ravel("labels", filter_missings=True)
            ncons = clabels.shape[0]
            cmap = np.empty(self.model._cCounter)
            cmap[clabels] = arange(clabels.shape[0])
            rows = cmap[rows]

            variables = self.model.variables
            vlabels = variables.ravel("labels", filter_missings=True)
            nvars = vlabels.shape[0]
            vmap = np.empty(self.model._xCounter)
            vmap[vlabels] = arange(nvars)
            cols = vmap[cols]

            shape = (ncons, nvars)  # same as model.nvars/ncons but already there

        return coo_matrix((data, (rows, cols)), shape=shape)


class AnonymousConstraint:
    """
    A constraint container used for storing multiple constraint arrays.
    """

    __slots__ = ("_lhs", "_sign", "_rhs")

    def __init__(self, lhs, sign, rhs):
        """
        Initialize a anonymous constraint.
        """
        if isinstance(rhs, (variables.Variable, expressions.LinearExpression)):
            raise TypeError(f"Assigned rhs must be a constant, got {type(rhs)}).")
        self._lhs, self._rhs = xr.align(lhs, DataArray(rhs))
        self._sign = DataArray(sign)

    @property
    def lhs(self):
        return self._lhs

    @property
    def sign(self):
        return self._sign

    @property
    def rhs(self):
        return self._rhs

    def __repr__(self):
        """
        Get the string representation of the expression.
        """
        lhs_string = self.lhs.to_dataset().__repr__()  # .split("\n", 1)[1]
        lhs_string = lhs_string.split("Data variables:\n", 1)[1]
        lhs_string = lhs_string.replace("    coeffs", "coeffs")
        lhs_string = lhs_string.replace("    vars", "vars")
        if self.sign.size == 1:
            sign_string = self.sign.item()
        else:
            sign_string = self.sign.__repr__().split("\n", 1)[1]
        if self.rhs.size == 1:
            rhs_string = self.rhs.item()
        else:
            rhs_string = self.rhs.__repr__().split("\n", 1)[1]
        return (
            f"Anonymous Constraint:\n"
            f"---------------------\n"
            f"\n{lhs_string}"
            f"\n{sign_string}"
            f"\n{rhs_string}"
        )

    def from_rule(model, rule, coords):
        """
        Create a constraint from a rule and a set of coordinates.

        This functionality mirrors the assignment of constraints as done by
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
            `AnonymousScalarConstraint`. Therefore use the direct getter when
            indexing variables in the linear expression.
        coords : coordinate-like
            Coordinates to processed by `xarray.DataArray`.
            For each combination of coordinates, the function given by `rule` is called.
            The order and size of coords has to be same as the argument list
            followed by `model` in function `rule`.


        Returns
        -------
        linopy.AnonymousConstraint

        Examples
        --------
        >>> from linopy import Model, LinearExpression
        >>> m = Model()
        >>> coords = pd.RangeIndex(10), ["a", "b"]
        >>> x = m.add_variables(0, 100, coords)
        >>> def bound(m, i, j):
        ...     if i % 2:
        ...         return (i - 1) * x[i - 1, j] >= 0
        ...     else:
        ...         return i * x[i, j] >= 0
        ...
        >>> con = AnonymousConstraint.from_rule(m, bound, coords)
        >>> con = m.add_constraints(con)
        """
        if not isinstance(coords, xr.core.dataarray.DataArrayCoordinates):
            coords = DataArray(coords=coords).coords
        shape = list(map(len, coords.values()))

        # test output type
        output = rule(model, *[c.values[0] for c in coords.values()])
        if not isinstance(output, AnonymousScalarConstraint):
            msg = f"`rule` has to return AnonymousScalarConstraint not {type(output)}."
            raise TypeError(msg)

        combinations = product(*[c.values for c in coords.values()])
        cons = [rule(model, *coord) for coord in combinations]
        exprs = [con.lhs for con in cons]

        lhs = expressions.LinearExpression._from_scalarexpression_list(exprs, coords)
        sign = DataArray(array([c.sign for c in cons]).reshape(shape), coords)
        rhs = DataArray(array([c.rhs for c in cons]).reshape(shape), coords)
        return AnonymousConstraint(lhs, sign, rhs)


@dataclass
class AnonymousScalarConstraint:
    """
    Container for anonymous scalar constraint.

    This contains a left-hand-side (lhs), a sign and a right-hand-side
    (rhs) for exactly one constraint.
    """

    lhs: expressions.ScalarLinearExpression
    sign: str
    rhs: float

    def to_anonymous_constraint(self):
        return AnonymousConstraint(self.lhs.to_linexpr(), self.sign, self.rhs)
