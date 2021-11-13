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
from scipy.sparse import coo_matrix, csr_matrix, vstack
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

    def block_indicator(self, block_map, nblocks=None):
        """
        Constructs the block_indicator for this set of constraints

        The block_indicator is a nblocks x dim1 x dim2 boolean array, where
        indicator[block, d1, d2] indicates that for constraint `name` d1, d2 is part of
        block. The same constraint is normally part of several blocks.

        TODO
        ----
        Unclear whether it fits into dask here, maybe pull out of class. I think it does,
        the best dask way in my mind would be to make use of xarray's high-level map_blocks
        method, which wraps the different dask blocks into DataArrays again, so that indices
        are available. The involved overhead is probably necessary.
        """
        if nblocks is None:
            nblocks = block_map.max().item() + 2
        vars = self.get_vars()
        constr_block_map = block_map[
            vars.transpose(f"{self.name}_term", *self.dims).values
        ]
        indicator = np.zeros((nblocks,) + self.shape, dtype=bool)
        indicator[
            (constr_block_map,)
            + tuple(np.ogrid[tuple(slice(None, s) for s in self.shape)])
        ] = True
        return indicator

    def block_sizes(self, block_map, nblocks=None):
        """ "
        TODO
        ----
        Unclear whether it fits into dask here, maybe pull out of class
        """
        if nblocks is None:
            nblocks = block_map.max().item() + 2
        sizes = np.zeros(nblocks + 1, dtype=int)
        indicator = self.block_indicator(block_map, nblocks)

        num_of_nonzero_blocks = indicator[1:].sum(axis=0)
        sizes[0] += (num_of_nonzero_blocks == 0).sum()

        onlyone_b = num_of_nonzero_blocks == 1
        if onlyone_b.any():
            sizes[1:nblocks] = indicator[1:, onlyone_b].sum(axis=1)

        sizes[nblocks] += (num_of_nonzero_blocks > 1).sum()
        return sizes


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
    def inequalities(self):
        return self[[n for n, s in self.sign.items() if s in ("<=", ">=")]]

    @property
    def equalities(self):
        return self[[n for n, s in self.sign.items() if s in ("=", "==")]]

    def get_xblock_map(self, block_map):
        "Get a dataset of same shape as constraints.labels with block values."
        global_block = block_map.max() + 1
        block_entries = replace_by_map(self.vars, block_map)
        for name, entries in block_entries.items():
            term_dim = f"{name}_term"
            first_block = entries.isel({term_dim: 0})
            linking = (entries != first_block).any(term_dim)
            masked = (entries == -1).any(term_dim)
            block_entries[name] = first_block.where(~linking, global_block).where(
                ~masked, -1
            )
        return block_entries

    def get_block_map(self, var_block_map):
        "Get a one-dimensional numpy array mapping the constraints to blocks."
        block_map = np.empty(self.model.ncons + 1, dtype=int)
        cblocks = self.get_xblock_map(var_block_map)

        for name, blocks in cblocks.items():
            constraint = self.labels[name]
            block_map[np.ravel(constraint)] = np.ravel(blocks)
        block_map[-1] = -1
        return block_map

    def to_matrix(self):
        "Construct a constraint matrix."
        data = []
        rows = []
        cols = []
        for name, labels in self.labels.items():
            dims = labels.dims + (f"{name}_term",)
            coeffs = self.coeffs[name].transpose(*dims)
            vars = self.vars[name].transpose(*dims)

            d = np.ravel(coeffs)
            r = np.ravel(labels.broadcast_like(coeffs))
            c = np.ravel(vars)

            nonzero = (c != -1) & (r != -1)
            data.append(d[nonzero])
            rows.append(r[nonzero])
            cols.append(c[nonzero])

        data = np.concatenate(data)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)

        shape = (self.model.ncons, self.model.nvars)
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
