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
from xarray import DataArray, Dataset

from linopy.common import _merge_inplace


class Constraint(DataArray):
    """
    Constraint container for storing constraint labels.

    The Constraint class is a subclass of xr.DataArray hence most xarray functions
    can be applied to it.
    """

    __slots__ = ("_cache", "_coords", "_indexes", "_name", "_variable", "model")

    def __init__(self, *args, **kwargs):
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
        self._merge_inplace("vars", vars, name)
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

    def block_sizes(self, num_blocks, block_map) -> np.ndarray:
        sizes = np.zeros(num_blocks + 1, dtype=int)
        for name in self.labels:
            sizes += self[name].block_sizes(num_blocks, block_map)
        return sizes
