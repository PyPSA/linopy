# -*- coding: utf-8 -*-
"""
Linopy constraints module.
This module contains implementations for the Constraint{s} class.
"""

from dataclasses import dataclass
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr
from numpy import asarray
from scipy.sparse import coo_matrix
from xarray import DataArray, Dataset

from linopy.common import _merge_inplace, replace_by_map


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

    # would like to have this as a property, but this does not work apparently
    def get_coeffs(self):
        """
        Get the left-hand-side coefficients of the constraint.
        The function raises an error in case no model is set as a reference.
        """
        if self.model is None:
            raise AttributeError("No reference model is assigned to the variable.")
        return self.model.constraints.coeffs[self.name]

    def get_vars(self):
        """
        Get the left-hand-side variables of the constraint.
        The function raises an error in case no model is set as a reference.
        """
        if self.model is None:
            raise AttributeError("No reference model is assigned to the variable.")
        return self.model.constraints.vars[self.name]

    def get_sign(self):
        """
        Get the sign of the constraint.
        The function raises an error in case no model is set as a reference.
        """
        if self.model is None:
            raise AttributeError("No reference model is assigned to the variable.")
        return self.model.constraints.sign[self.name]

    def get_rhs(self):
        """
        Get the right-hand-side constant of the constraint.
        The function raises an error in case no model is set as a reference.
        """
        if self.model is None:
            raise AttributeError("No reference model is assigned to the variable.")
        return self.model.constraints.rhs[self.name]


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
        "Constraint labels",
        "Left-hand-side coefficients",
        "Left-hand-side variables",
        "Signs",
        "Right-hand-side constants",
    ]

    def __repr__(self):
        """Return a string representation of the linopy model."""
        r = "linopy.model.Constraints"
        line = "=" * len(r)
        r += f"\n{line}\n\n"
        for (k, K) in zip(self.dataset_attrs, self.dataset_names):
            s = getattr(self, k).__repr__().split("\n", 1)[1]
            s = s.replace("Data variables:\n", "Data:\n")
            line = "-" * (len(K) + 1)
            r += f"{K}:\n{line}\n{s}\n\n"
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
        self._merge_inplace("labels", labels, name, fill_value=-1)
        self._merge_inplace("coeffs", coeffs, name)
        self._merge_inplace("vars", vars, name, fill_value=-1)
        self._merge_inplace("sign", sign, name)
        self._merge_inplace("rhs", rhs, name)

    def remove(self, name):
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
        return self.model.ncons

    @property
    def inequalities(self):
        return self[[n for n, s in self.sign.items() if s in ("<=", ">=")]]

    @property
    def equalities(self):
        return self[[n for n, s in self.sign.items() if s in ("=", "==")]]

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
        block_entries = replace_by_map(self.vars, block_map)
        res = xr.full_like(self.labels, N + 1, dtype=block_map.dtype)

        for name, entries in block_entries.items():
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
        return self.blocks

    def ravel(self, key, broadcast_like="labels", filter_missings=False):
        res = []
        for name, values in getattr(self, broadcast_like).items():
            flat = getattr(self, key)[name].broadcast_like(values).data.ravel()
            if filter_missings:
                flat = flat[values.data.ravel() != -1]
            res.append(flat)
        return np.concatenate(res)

    def to_matrix(self):
        "Construct a constraint matrix."
        shape = (self.model.ncons, self.model.nvars)
        keys = ["coeffs", "labels", "vars"]
        data, rows, cols = [self.ravel(k, broadcast_like="vars") for k in keys]
        non_missing = (rows != -1) & (cols != -1)
        data = asarray(data[non_missing])
        rows = asarray(rows[non_missing])
        cols = asarray(cols[non_missing])
        return coo_matrix((data, (rows, cols)), shape=shape)

    def to_inequality_rhs(self):
        lo = []
        hi = []
        for name, labels in self.labels.items():
            dims = labels.dims
            data = np.ravel(self.rhs[name].broadcast_like(labels).transpose(*dims))
            s = self.sign[name].item()
            if s == "<=":
                lo.append(np.full_like(data, -np.inf, dtype=float))
                hi.append(data)
            elif s == ">=":
                lo.append(data)
                hi.append(np.full_like(data, np.inf, dtype=float))

        return np.concatenate(lo), np.concatenate(hi)

    def to_equality_rhs(self, vars, rhs):
        r = []
        for name, labels in self.labels.items():
            dims = labels.dims
            data = np.ravel(self.rhs[name].broadcast_like(labels).transpose(*dims))
            s = self.sign[name].item()
            if s == "==":
                r.append(data)

        return np.concatenate(r)

    def to_rhs(self, vars, rhs):
        r = []
        for name, labels in self.labels.items():
            dims = labels.dims
            data = np.ravel(self.rhs[name].broadcast_like(labels).transpose(*dims))
            r.append(data)

        return np.concatenate(r)
