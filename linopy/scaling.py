#!/usr/bin/env python3
"""
Linopy scaling module.

This module contains helpers to validate scaling factors and to build
solver-side scaling lookups for variables and constraints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from xarray import DataArray, Dataset

from linopy.common import assign_multiindex_safe

if TYPE_CHECKING:
    from linopy.model import Model
    from linopy.variables import Variable


def validate_scaling(scaling: DataArray, label: str = "scaling") -> DataArray:
    """
    Validate and normalize a scaling array.

    Scaling values are positive numeric factors used during solver export.
    Column scaling converts variable units, while row and objective scaling
    multiply exported coefficients. They must be finite and strictly positive.
    """
    scaling = scaling.astype(float)
    values = scaling.values
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{label} must contain only finite positive values.")
    return scaling


def ensure_scaling(data: Dataset, like: DataArray, label: str) -> Dataset:
    """Default-fill and validate the ``scaling`` field of a data set."""
    if "scaling" not in data:
        data = assign_multiindex_safe(data, scaling=DataArray(1.0).broadcast_like(like))
    return assign_multiindex_safe(data, scaling=validate_scaling(data.scaling, label))


def variable_solver_scaling(variable: Variable) -> DataArray:
    """
    Return solver-side scaling for a variable.

    Discrete variables keep ordinary solver columns, so their supplied scaling
    is metadata only.
    """
    if variable.attrs.get("binary", False) or variable.attrs.get("integer", False):
        return DataArray(1.0).broadcast_like(variable.labels)
    return variable.scaling


def _scatter_active(target: np.ndarray, labels: np.ndarray, values: np.ndarray) -> None:
    """Scatter ``values`` into ``target`` at active (non ``-1``) label positions."""
    mask = labels != -1
    target[labels[mask]] = values[mask]


def variable_scaling_lookup(model: Model) -> np.ndarray:
    """Return solver-side variable scaling indexed by raw variable label."""
    scaling = np.ones(model._xCounter, dtype=float)
    for var in model.variables.data.values():
        labels = var.labels.values.ravel()
        _scatter_active(scaling, labels, variable_solver_scaling(var).values.ravel())
    return scaling


def constraint_scaling_lookup(model: Model) -> np.ndarray:
    """Return constraint scaling indexed by raw constraint label."""
    from linopy.constraints import CSRConstraint

    scaling = np.ones(model._cCounter, dtype=float)
    for con in model.constraints.data.values():
        if isinstance(con, CSRConstraint):
            scaling[con._con_labels] = con._scaling
            continue
        labels = con.labels.values.ravel()
        _scatter_active(scaling, labels, con.scaling.values.ravel())
    return scaling
