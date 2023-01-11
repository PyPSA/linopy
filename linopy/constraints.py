# -*- coding: utf-8 -*-
"""
Linopy constraints module.

This module contains implementations for the Constraint{s} class.
"""

import re
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Sequence, Union

import dask
import numpy as np
import pandas as pd
import xarray as xr
from deprecation import deprecated
from numpy import arange, array
from scipy.sparse import coo_matrix
from xarray import DataArray, Dataset

from linopy import expressions, variables
from linopy.common import (
    _merge_inplace,
    forward_as_properties,
    has_optimized_model,
    is_constant,
    maybe_replace_signs,
    print_coord,
    print_single_expression,
    replace_by_map,
)
from linopy.constants import EQUAL, GREATER_EQUAL, LESS_EQUAL


@forward_as_properties(
    labels=[
        "attrs",
        "coords",
        "indexes",
        "name",
        "shape",
        "size",
        "values",
        "dims",
        "ndim",
    ],
    lhs=["nterm"],
)
class Constraint:
    """
    Projection to a single constraint in a model.

    The Constraint class is a subclass of xr.DataArray hence most xarray
    functions can be applied to it.
    """

    __slots__ = ("_model", "_name", "_labels")

    def __init__(self, labels: DataArray, model: Any, name: str):
        """
        Initialize the Constraint.

        Parameters
        ----------
        labels : xarray.DataArray
            labels of the constraint.
        model : linopy.Model
            Underlying model.
        name : str
            Name of the constraint.
        """
        self._labels = labels
        self._model = model
        self._name = name

    @property
    def labels(self):
        """
        Get the labels of the constraint.
        """
        return self._labels

    @property
    def model(self):
        """
        Get the model of the constraint.
        """
        return self._model

    @property
    def name(self):
        """
        Get the name of the constraint.
        """
        return self._name

    def __repr__(self):
        """
        Get the string representation of the Constraint.
        """
        # return single if only one exist
        if self.size == self.nterm:
            expr_string = print_single_expression(
                self.lhs.coeffs.values, self.lhs.vars.values, self.lhs.model
            )
            header = f"{self.type}\n" + "-" * (len(self.type) + 1)
            return f"{header}\n{expr_string} {self.sign.item()} {self.rhs.item()}"

        # print only a few values
        max_print = 14
        split_at = max_print // 2
        to_print = np.flatnonzero(self.mask)
        truncate = len(to_print) > max_print
        if truncate:
            to_print = np.hstack([to_print[:split_at], to_print[-split_at:]])

        # create string, we use numpy to get the indexes
        data_string = ""
        idx = np.unravel_index(to_print, self.shape)
        indexes = np.stack(idx)
        coords = [self.indexes[self.dims[i]][idx[i]] for i in range(len(self.dims))]

        # loop over all values to LinearExpression(print
        data_string = ""
        for i in range(len(to_print)):
            # this is the index for the labels array
            ix = tuple(indexes[..., i])
            # sign and rhs might only be defined for some dimensions
            six = tuple(
                ix[i] for i in range(self.ndim) if self.dims[i] in self.sign.dims
            )
            rix = tuple(
                ix[i] for i in range(self.ndim) if self.dims[i] in self.rhs.dims
            )

            coord = [c[i] for c in coords]
            coord_string = print_coord(coord)
            expr_string = print_single_expression(
                self.coeffs.values[ix], self.vars.values[ix], self.lhs.model
            )
            sign_string = f"{self.sign.values[six]}"
            rhs_string = f"{self.rhs.values[rix]}"

            data_string += f"\n{self.name}{coord_string}:  {expr_string} {sign_string} {rhs_string}"

            if i == split_at - 1 and truncate:
                data_string += "\n\t\t..."

        # create shape string
        shape_string = ", ".join(
            [f"{self.dims[i]}: {self.shape[i]}" for i in range(self.ndim)]
        )
        shape_string = f"({shape_string})"
        n_masked = (~self.mask).sum().item()
        mask_string = f" - {n_masked} masked entries" if n_masked else ""
        header = f"{self.type} {shape_string}{mask_string}\n" + "-" * (
            len(self.type) + len(shape_string) + len(mask_string) + 1
        )
        return f"{header}{data_string}"

    @deprecated(details="Use the `labels` property instead of `to_array`")
    def to_array(self):
        """
        Convert the variable array to a xarray.DataArray.
        """
        return self.labels

    @property
    def type(self):
        """
        Get the type of the constraint.
        """
        return "Constraint"

    @property
    def mask(self):
        """
        Get the mask of the constraint.

        The mask indicates on which coordinates the constraint array is enabled
        (True) and disabled (False).

        Returns
        -------
        xr.DataArray
        """
        return (self.labels != -1).astype(bool)

    @property
    def coeffs(self):
        """
        Get the left-hand-side coefficients of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        return self.model.constraints.coeffs[self.name]

    @coeffs.setter
    def coeffs(self, value):
        term_dim = self.name + "_term"
        value = DataArray(value).broadcast_like(self.vars, exclude=[term_dim])
        self.model.constraints.coeffs = self.model.constraints.coeffs.drop_vars(
            self.name
        ).assign({self.name: value})

    @property
    def vars(self):
        """
        Get the left-hand-side variables of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        return self.model.constraints.vars[self.name]

    @vars.setter
    def vars(self, value):
        term_dim = self.name + "_term"
        if isinstance(value, variables.Variable):
            value = value.labels
        if not isinstance(value, DataArray):
            raise TypeError("Expected value to be of type DataArray or Variable")
        value = value.broadcast_like(self.coeffs, exclude=[term_dim])
        self.model.constraints.vars = self.model.constraints.vars.drop_vars(
            self.name
        ).assign({self.name: value})

    @property
    def lhs(self):
        """
        Get the left-hand-side linear expression of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        term_dim = self.name + "_term"
        coeffs = self.coeffs.rename({term_dim: "_term"})
        vars = self.vars.rename({term_dim: "_term"})
        ds = Dataset({"coeffs": coeffs, "vars": vars})
        return expressions.LinearExpression(ds, self.model)

    @lhs.setter
    def lhs(self, value):
        if not isinstance(value, expressions.LinearExpression):
            raise TypeError("Assigned lhs must be a LinearExpression.")

        value = value.rename(_term=self.name + "_term")
        self.coeffs = value.coeffs
        self.vars = value.vars

    @property
    def sign(self):
        """
        Get the signs of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        return self.model.constraints.sign[self.name]

    @sign.setter
    @is_constant
    def sign(self, value):
        value = maybe_replace_signs(DataArray(value)).broadcast_like(self.sign)
        self.model.constraints.sign[self.name] = value

    @property
    def rhs(self):
        """
        Get the right hand side constants of the constraint.

        The function raises an error in case no model is set as a
        reference.
        """
        return self.model.constraints.rhs[self.name]

    @rhs.setter
    @is_constant
    def rhs(self, value):
        if isinstance(value, (variables.Variable, expressions.LinearExpression)):
            raise TypeError(f"Assigned rhs must be a constant, got {type(value)}).")
        value = DataArray(value).broadcast_like(self.rhs)
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

    @property
    def shape(self):
        return self.labels.shape


@dataclass(repr=False)
class Constraints:
    """
    A constraint container used for storing multiple constraint arrays.
    """

    labels: Dataset = field(default_factory=Dataset)
    coeffs: Dataset = field(default_factory=Dataset)
    vars: Dataset = field(default_factory=Dataset)
    sign: Dataset = field(default_factory=Dataset)
    rhs: Dataset = field(default_factory=Dataset)
    blocks: Dataset = field(default_factory=Dataset)
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
            return Constraint(self.labels[names], self.model, names)

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

        See
        http://ipython.readthedocs.io/en/stable/config/integrating.html#tab-completion
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
        return self[
            [n for n, s in self.sign.items() if s in (LESS_EQUAL, GREATER_EQUAL)]
        ]

    @property
    def equalities(self):
        """
        Get the subset of constraints which are purely equalities.
        """
        return self[[n for n, s in self.sign.items() if s == EQUAL]]

    def sanitize_zeros(self):
        """
        Filter out terms with zero and close-to-zero coefficient.
        """
        for name in self:
            not_zero = abs(self.coeffs[name]) > 1e-10
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

        # Group repeated variables in the same constraint
        df = pd.DataFrame({"data": data, "rows": rows, "cols": cols})
        df = df.groupby(["rows", "cols"], as_index=False).sum()

        return coo_matrix((df.data, (df.rows, df.cols)), shape=shape)


@dataclass(repr=False)
@forward_as_properties(
    labels=[
        "attrs",
        "coords",
        "indexes",
        "shape",
        "size",
        "values",
        "dims",
        "ndim",
    ],
    lhs=["nterm", "coeffs", "vars"],
)
class AnonymousConstraint:
    """
    A constraint container used for storing multiple constraint arrays.
    """

    _lhs: "expressions.LinearExpression"
    _sign: Union[DataArray, str]
    _rhs: Union[DataArray, float, int]

    def __post_init__(self):
        """
        Initialize a anonymous constraint.
        """
        if isinstance(self.rhs, (variables.Variable, expressions.LinearExpression)):
            raise TypeError(f"Assigned rhs must be a constant, got {type(self.rhs)}).")
        lhs_data, rhs = xr.align(self.lhs.data, DataArray(self.rhs))
        self._labels = (lhs_data.vars.chunk() + rhs).sum("_term")
        self._labels.data = np.full(self.labels.shape, np.nan)
        self._lhs = expressions.LinearExpression(lhs_data, self.lhs.model)
        self._rhs = rhs
        self._sign = DataArray(self.sign)

    def __repr__(self):
        """
        Get the representation of the AnonymousConstraint.
        """
        return Constraint.__repr__(self)

    @property
    def name(self):
        return ""

    @property
    def type(self):
        """
        Get the type of the constraint.
        """
        # needed for __repr__ compatibility.
        return "AnomymousConstraint"

    @property
    def mask(self):
        """
        Get the mask of the constraint.
        """
        # needed for __repr__ compatibility.
        return xr.full_like(self.labels, True, dtype=bool)

    @property
    def labels(self):
        """
        Get the labels of the constraint.
        """
        return self._labels

    @property
    def lhs(self):
        """
        Get the left hand side of the constraint.
        """
        return self._lhs

    @property
    def sign(self):
        """
        Get the sign of the constraint.
        """
        return self._sign

    @property
    def rhs(self):
        """
        Get the right hand side of the constraint.
        """
        return self._rhs

    @classmethod
    def from_rule(cls, model, rule, coords):
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
        if not isinstance(output, AnonymousScalarConstraint) and not output is None:
            msg = f"`rule` has to return AnonymousScalarConstraint not {type(output)}."
            raise TypeError(msg)

        combinations = product(*[c.values for c in coords.values()])
        placeholder_lhs = expressions.ScalarLinearExpression((np.nan,), (-1,), model)
        placeholder = AnonymousScalarConstraint(placeholder_lhs, "=", np.nan)
        cons = [rule(model, *coord) or placeholder for coord in combinations]
        exprs = [con.lhs for con in cons]

        lhs = expressions.LinearExpression._from_scalarexpression_list(
            exprs, coords, model
        )
        sign = DataArray(array([c.sign for c in cons]).reshape(shape), coords)
        rhs = DataArray(array([c.rhs for c in cons]).reshape(shape), coords)
        return cls(lhs, sign, rhs)


class AnonymousScalarConstraint:
    """
    Container for anonymous scalar constraint.

    This contains a left-hand-side (lhs), a sign and a right-hand-side
    (rhs) for exactly one constraint.
    """

    _lhs: "expressions.ScalarLinearExpression"
    _sign: str
    _rhs: float

    def __init__(self, lhs, sign, rhs):
        """
        Initialize a anonymous scalar constraint.
        """
        if not isinstance(rhs, (int, float, np.float32, np.float64, np.integer)):
            raise TypeError(f"Assigned rhs must be a constant, got {type(rhs)}).")
        self._lhs = lhs
        self._sign = sign
        self._rhs = rhs

    def __repr__(self):
        """
        Get the representation of the AnonymousScalarConstraint.
        """
        expr_string = print_single_expression(
            self.lhs.coeffs, self.lhs.vars, self.lhs.model
        )
        return f"AnonymousScalarConstraint: {expr_string} {self.sign} {self.rhs}"

    @property
    def lhs(self):
        """
        Get the left hand side of the constraint.
        """
        return self._lhs

    @property
    def sign(self):
        """
        Get the sign of the constraint.
        """
        return self._sign

    @property
    def rhs(self):
        """
        Get the right hand side of the constraint.
        """
        return self._rhs

    def to_anonymous_constraint(self):
        return AnonymousConstraint(self.lhs.to_linexpr(), self.sign, self.rhs)
