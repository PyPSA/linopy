# -*- coding: utf-8 -*-
"""
Linopy constraints module.
This module contains implementations for the Constraint{s} class.
"""

import re
from dataclasses import dataclass
from typing import Any, Sequence, Union

import dask
import numpy as np
import pandas as pd
import xarray as xr
from deprecation import deprecated
from scipy.sparse import coo_matrix
from xarray import DataArray, Dataset

from linopy.common import _merge_inplace, replace_by_map
from linopy.expressions import LinearExpression


class Constraint(DataArray):
    """
    Constraint container for storing constraint labels.

    The Constraint class is a subclass of xr.DataArray hence most xarray functions
    can be applied to it.
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

    def __repr__(self):
        """Get the string representation of the constraints."""
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
        """Get the html representation of the variables."""
        # return self.__repr__()
        data_string = self.to_array()._repr_html_()
        data_string = data_string.replace("xarray.DataArray", "linopy.Constraint")
        return data_string

    def to_array(self):
        """Convert the variable array to a xarray.DataArray."""
        return DataArray(self)

    @property
    def coeffs(self):
        """
        Get the left-hand-side coefficients of the constraint.
        The function raises an error in case no model is set as a reference.
        """
        if self.model is None:
            raise AttributeError("No reference model is assigned to the constraint.")
        return self.model.constraints.coeffs[self.name]

    @coeffs.setter
    def coeffs(self, value):
        labels = self.model.constraints.labels
        value = DataArray(value)
        term_dim = self.name + "_term"

        if term_dim not in value.dims:
            value = value.expand_dims(term_dim)

        assert (set(value.dims) - {term_dim}).issubset(labels.dims), (
            "Dimensions of new values not a subset of labels dimensions, "
            "therefore the new coefficients cannot be aligned with the existing labels."
        )

        self.model.constraints.coeffs[self.name] = value

    @property
    def vars(self):
        """
        Get the left-hand-side variables of the constraint.
        The function raises an error in case no model is set as a reference.
        """
        if self.model is None:
            raise AttributeError("No reference model is assigned to the constraint.")
        return self.model.constraints.vars[self.name]

    @vars.setter
    def vars(self, value):
        labels = self.model.constraints.labels
        value = DataArray(value)
        term_dim = self.name + "_term"

        if term_dim not in value.dims:
            value = value.expand_dims(term_dim)

        assert (set(value.dims) - {term_dim}).issubset(labels.dims), (
            "Dimensions of new values not a subset of labels dimensions, "
            "therefore the new variables cannot be aligned with the existing labels."
        )

        self.model.constraints.vars[self.name] = value

    @property
    def lhs(self):
        """
        Get the left-hand-side linear expression of the constraint.
        The function raises an error in case no model is set as a reference.
        """
        return LinearExpression(Dataset({"coeffs": self.coeffs, "vars": self.vars}))

    @lhs.setter
    def lhs(self, value):
        if not isinstance(value, LinearExpression):
            raise TypeError("Assigned lhs must be a LinearExpression.")

        value = value.rename(_term=self.name + "_term")
        self.coeffs = value.coeffs
        self.vars = value.vars

    @property
    def sign(self):
        """
        Get the signs of the constraint.
        The function raises an error in case no model is set as a reference.
        """
        if self.model is None:
            raise AttributeError("No reference model is assigned to the constraint.")
        return self.model.constraints.sign[self.name]

    @sign.setter
    def sign(self, value):
        labels = self.model.constraints.labels
        value = DataArray(value)
        assert set(value.dims).issubset(labels.dims), (
            "Dimensions of new values not a subset of labels dimensions, "
            "therefore the new signs cannot be aligned with the existing labels."
        )
        self.model.constraints.sign[self.name] = value

    @property
    def rhs(self):
        """
        Get the right hand side constants of the constraint.
        The function raises an error in case no model is set as a reference.
        """
        if self.model is None:
            raise AttributeError("No reference model is assigned to the constraint.")
        return self.model.constraints.rhs[self.name]

    @rhs.setter
    def rhs(self, value):
        labels = self.model.constraints.labels
        value = DataArray(value)
        assert set(value.dims).issubset(labels.dims), (
            "Dimensions of new values not a subset of labels dimensions, "
            "therefore the new right-hand-side cannot be aligned with the existing labels."
        )

        self.model.constraints.rhs[self.name] = value

    @deprecated("0.0.5", "0.0.6", details="Use the `coeffs` accessor instead.")
    def get_coeffs(self):
        return self.coeffs

    @deprecated("0.0.5", "0.0.6", details="Use the `vars` accessor instead.")
    def get_vars(self):
        return self.vars

    @deprecated("0.0.5", "0.0.6", details="Use the `sign` accessor instead.")
    def get_sign(self):
        return self.sign

    @deprecated("0.0.5", "0.0.6", details="Use the `rhs` accessor instead.")
    def get_rhs(self):
        return self.rhs


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
        """Return a string representation of the linopy model."""
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

    def add(
        self,
        name,
        labels: DataArray,
        coeffs: DataArray,
        vars: DataArray,
        sign: DataArray,
        rhs: DataArray,
    ):
        """Add constraint `name`."""
        self._merge_inplace("labels", labels, name, fill_value=-1)
        self._merge_inplace("coeffs", coeffs, name)
        self._merge_inplace("vars", vars, name, fill_value=-1)
        self._merge_inplace("sign", sign, name)
        self._merge_inplace("rhs", rhs, name)

    def remove(self, name):
        """Remove constraint `name` from the constraints."""
        for attr in self.dataset_attrs:
            setattr(self, attr, getattr(self, attr).drop_vars(name))

    @property
    def coefficientrange(self):
        """Coefficient range of the constraint."""
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
        Get the number all constraints which were at some point added to the model.
        These also include constraints with missing labels.
        """
        return self.ravel("labels", filter_missings=True).shape[0]

    @property
    def inequalities(self):
        "Get the subset of constraints which are purely inequalities."
        return self[[n for n, s in self.sign.items() if s in ("<=", ">=")]]

    @property
    def equalities(self):
        "Get the subset of constraints which are purely equalities."
        return self[[n for n, s in self.sign.items() if s in ("=", "==")]]

    def sanitize_missings(self):
        """
        Set constraints labels to -1 if either rhs, coeffs or vars are missing.
        Also set vars to -1 where labels are -1.
        """
        for name in self:
            term_dim = name + "_term"
            no_missing_vars = (self.vars[name] != -1).any(term_dim)
            no_missing_coeffs = self.coeffs[name].notnull().any(term_dim)
            no_missing_rhs = self.rhs[name].notnull()
            no_missing = no_missing_vars & no_missing_coeffs & no_missing_rhs

            self.labels[name] = self.labels[name].where(no_missing, -1)

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
        Create an generator which iterates over all arrays in `key` and flattens them.

        Parameters
        ----------
        key : str/Dataset
            Key to be iterated over. Optionally pass a dataset which is
            broadcastable to `broadcast_like`. Must be on of 'labels', 'vars'.
        broadcast_like : str, optional
            Name of the dataset to which the input data in `key` is aligned to.
            The default is "labels".
        filter_missings : bool, optional
            Filter out values where `labels` data is -1. If broadcast is `vars`
            also values where `vars` is -1 are filtered. When enabled, the
            data is load into memory. The default is False.


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
            else:
                flat = broadcasted.data.ravel()
            yield flat

    def ravel(self, key, broadcast_like="labels", filter_missings=False, compute=True):
        """
        Ravel and concate all arrays in `key` while aligning to `broadcast_like`.

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
            cmap[clabels] = np.arange(clabels.shape[0])
            rows = cmap[rows]

            variables = self.model.variables
            vlabels = variables.ravel("labels", filter_missings=True)
            nvars = vlabels.shape[0]
            vmap = np.empty(self.model._xCounter)
            vmap[vlabels] = np.arange(nvars)
            cols = vmap[cols]

            shape = (ncons, nvars)  # same as model.nvars/ncons but already there

        return coo_matrix((data, (rows, cols)), shape=shape)
