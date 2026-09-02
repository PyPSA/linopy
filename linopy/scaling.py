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
    """
    Default-fill the ``scaling`` field of a data set when it is missing.

    Present scaling is left untouched: it is validated at the user-facing
    entry points (``add_variables``/``add_constraints`` and the ``scaling``
    setters), and this internal reconstruction path must neither reorder the
    dimensions nor reject the NaN-padded rows produced by ``sel``/``reindex``.
    """
    if "scaling" in data:
        return data
    scaling = validate_scaling(DataArray(1.0).broadcast_like(like), label)
    return assign_multiindex_safe(data, scaling=scaling)


def is_trivial(scaling: np.ndarray | DataArray | float) -> bool:
    """
    Whether every factor equals 1, i.e. the scaling is a no-op.

    Checked with ``min``/``max`` rather than an equality mask so that an
    unscaled model does not pay a temporary array per variable/constraint.
    """
    values = np.asarray(scaling)
    if values.size == 0:
        return True
    return bool(values.min() == 1.0 and values.max() == 1.0)


def _scatter_active(target: np.ndarray, labels: np.ndarray, values: np.ndarray) -> None:
    """Scatter ``values`` into ``target`` at active (non ``-1``) label positions."""
    mask = labels != -1
    target[labels[mask]] = values[mask]


def variable_scaling_lookup(model: Model) -> np.ndarray | None:
    """
    Return solver-side variable scaling indexed by raw variable label.

    ``None`` is returned when no variable is scaled. Export paths then skip
    the rescaling altogether instead of allocating a full lookup of ones and
    multiplying every coefficient by 1.
    """
    variables = list(model.variables.data.values())
    if all(is_trivial(var.solver_scaling.values) for var in variables):
        return None

    scaling = np.ones(model._xCounter, dtype=float)
    for var in variables:
        labels = var.labels.values
        values = np.broadcast_to(var.solver_scaling.values, labels.shape)
        _scatter_active(scaling, labels.ravel(), values.ravel())
    return scaling


def constraint_scaling_lookup(model: Model) -> np.ndarray | None:
    """
    Return constraint scaling indexed by raw constraint label.

    ``None`` is returned when no constraint row is scaled, see
    :func:`variable_scaling_lookup`.
    """
    from linopy.constraints import CSRConstraint

    constraints = list(model.constraints.data.values())
    if all(
        is_trivial(con._scaling if isinstance(con, CSRConstraint) else con.scaling)
        for con in constraints
    ):
        return None

    scaling = np.ones(model._cCounter, dtype=float)
    for con in constraints:
        if isinstance(con, CSRConstraint):
            scaling[con._con_labels] = con._scaling
            continue
        labels = con.labels.values.ravel()
        _scatter_active(scaling, labels, con.scaling.values.ravel())
    return scaling
