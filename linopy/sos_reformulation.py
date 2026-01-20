"""
Linopy SOS constraint reformulation module.

This module provides functions to reformulate SOS1 and SOS2 constraints
as binary + linear constraints for solvers that don't support them natively.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from xarray import DataArray

if TYPE_CHECKING:
    from linopy.model import Model
    from linopy.variables import Variable

logger = logging.getLogger(__name__)


def validate_bounds_for_reformulation(var: Variable) -> None:
    """
    Validate that a variable has finite bounds required for SOS reformulation.

    Parameters
    ----------
    var : Variable
        Variable to validate.

    Raises
    ------
    ValueError
        If any bound is infinite (required for Big-M formulation).
    """
    lower = var.lower
    upper = var.upper

    if np.isinf(lower).any():
        raise ValueError(
            f"Variable '{var.name}' has infinite lower bounds. "
            "Finite bounds are required for SOS reformulation (Big-M method)."
        )
    if np.isinf(upper).any():
        raise ValueError(
            f"Variable '{var.name}' has infinite upper bounds. "
            "Finite bounds are required for SOS reformulation (Big-M method)."
        )


def compute_big_m_values(var: Variable) -> tuple[DataArray, DataArray]:
    """
    Compute Big-M values from variable bounds.

    Parameters
    ----------
    var : Variable
        Variable with finite bounds.

    Returns
    -------
    tuple[DataArray, DataArray]
        (M_upper, M_lower) - Big-M values computed from bounds.
        M_upper = upper bound (for x <= U * y constraints)
        M_lower = lower bound (for x >= L * y constraints)
    """
    return var.upper, var.lower


def reformulate_sos1(model: Model, var: Variable, prefix: str) -> None:
    """
    Reformulate an SOS1 constraint as binary + linear constraints.

    SOS1: At most one variable can be non-zero.

    Reformulation:
        For each x[i] with bounds [L[i], U[i]]:
        - Add binary y[i]
        - x[i] <= U[i] * y[i]   (if U[i] > 0)
        - x[i] >= L[i] * y[i]   (if L[i] < 0)
        - sum(y) <= 1

    Parameters
    ----------
    model : Model
        The model to add reformulation constraints to.
    var : Variable
        The variable with SOS1 constraint.
    prefix : str
        Prefix for naming auxiliary variables and constraints.
    """
    sos_dim = var.attrs["sos_dim"]
    var_name = var.name

    # Get bounds
    M_upper, M_lower = compute_big_m_values(var)

    # Extract coords as list of DataArrays for add_variables
    coords_list = [var.coords[d] for d in var.dims]

    # Create binary indicator variables with same dimensions as original variable
    y_name = f"{prefix}{var_name}_y"
    y = model.add_variables(
        coords=coords_list,
        name=y_name,
        binary=True,
    )

    # Add upper bound constraints: x <= U * y (when U > 0)
    # This ensures x can only be positive when y = 1
    has_positive_upper = (M_upper > 0).any()
    if has_positive_upper:
        model.add_constraints(
            var <= M_upper * y,
            name=f"{prefix}{var_name}_upper",
        )

    # Add lower bound constraints: x >= L * y (when L < 0)
    # This ensures x can only be negative when y = 1
    has_negative_lower = (M_lower < 0).any()
    if has_negative_lower:
        model.add_constraints(
            var >= M_lower * y,
            name=f"{prefix}{var_name}_lower",
        )

    # Add cardinality constraint: sum(y) <= 1 over the SOS dimension
    # This is summed over sos_dim, keeping other dimensions
    model.add_constraints(
        y.sum(dim=sos_dim) <= 1,
        name=f"{prefix}{var_name}_card",
    )

    logger.debug(f"Reformulated SOS1 constraint for variable '{var_name}'")


def reformulate_sos2(model: Model, var: Variable, prefix: str) -> None:
    """
    Reformulate an SOS2 constraint as binary + linear constraints.

    SOS2: At most two adjacent variables can be non-zero.

    Reformulation:
        For ordered x[i], i = 0..n-1:
        - Add binary z[i] for i = 0..n-2 (segment indicators)
        - x[0] <= U[0] * z[0]
        - x[i] <= U[i] * (z[i-1] + z[i]) for i = 1..n-2
        - x[n-1] <= U[n-1] * z[n-2]
        - Similar for lower bounds if L[i] < 0
        - sum(z) <= 1

    Parameters
    ----------
    model : Model
        The model to add reformulation constraints to.
    var : Variable
        The variable with SOS2 constraint.
    prefix : str
        Prefix for naming auxiliary variables and constraints.
    """
    sos_dim = var.attrs["sos_dim"]
    var_name = var.name

    # Get the size of the SOS dimension
    n = var.sizes[sos_dim]

    # Trivial case: single element always satisfies SOS2
    if n <= 1:
        logger.debug(
            f"Skipping SOS2 reformulation for '{var_name}' (trivial case: n={n})"
        )
        return

    # Get bounds
    M_upper, M_lower = compute_big_m_values(var)

    # Create n-1 binary segment indicators
    # z[i] indicates that the "active segment" is between positions i and i+1
    segment_coords_values = var.coords[sos_dim].values[:-1]  # n-1 segment indicators

    # Build coords for z: same as var but with truncated sos_dim
    z_coords_list = []
    for d in var.dims:
        if d == sos_dim:
            z_coords_list.append(pd.Index(segment_coords_values, name=sos_dim))
        else:
            z_coords_list.append(var.coords[d])

    z_name = f"{prefix}{var_name}_z"
    z = model.add_variables(
        coords=z_coords_list,
        name=z_name,
        binary=True,
    )

    # For SOS2, each x[i] can only be non-zero if an adjacent segment is active
    # x[0] needs z[0] active
    # x[i] (1 <= i <= n-2) needs z[i-1] or z[i] active
    # x[n-1] needs z[n-2] active

    # Convert to LinearExpression to avoid SOS attribute validation issues during isel
    # (Variable's isel preserves SOS attrs, which fail validation when dim is removed)
    x_expr = 1 * var
    z_expr = 1 * z

    # Process upper bound constraints (for positive bounds)
    has_positive_upper = (M_upper > 0).any()
    if has_positive_upper:
        # First element: x[0] <= U[0] * z[0]
        x_first = x_expr.isel({sos_dim: 0})
        z_first = z_expr.isel({sos_dim: 0})
        U_first = M_upper.isel({sos_dim: 0})
        model.add_constraints(
            x_first <= U_first * z_first,
            name=f"{prefix}{var_name}_upper_first",
        )

        # Middle elements: x[i] <= U[i] * (z[i-1] + z[i]) for i = 1..n-2
        if n > 2:
            for i in range(1, n - 1):
                x_i = x_expr.isel({sos_dim: i})
                z_prev = z_expr.isel({sos_dim: i - 1})
                z_curr = z_expr.isel({sos_dim: i})
                U_i = M_upper.isel({sos_dim: i})
                model.add_constraints(
                    x_i <= U_i * (z_prev + z_curr),
                    name=f"{prefix}{var_name}_upper_mid_{i}",
                )

        # Last element: x[n-1] <= U[n-1] * z[n-2]
        x_last = x_expr.isel({sos_dim: n - 1})
        z_last = z_expr.isel({sos_dim: n - 2})
        U_last = M_upper.isel({sos_dim: n - 1})
        model.add_constraints(
            x_last <= U_last * z_last,
            name=f"{prefix}{var_name}_upper_last",
        )

    # Process lower bound constraints (for negative bounds)
    has_negative_lower = (M_lower < 0).any()
    if has_negative_lower:
        # First element: x[0] >= L[0] * z[0]
        x_first = x_expr.isel({sos_dim: 0})
        z_first = z_expr.isel({sos_dim: 0})
        L_first = M_lower.isel({sos_dim: 0})
        model.add_constraints(
            x_first >= L_first * z_first,
            name=f"{prefix}{var_name}_lower_first",
        )

        # Middle elements: x[i] >= L[i] * (z[i-1] + z[i]) for i = 1..n-2
        if n > 2:
            for i in range(1, n - 1):
                x_i = x_expr.isel({sos_dim: i})
                z_prev = z_expr.isel({sos_dim: i - 1})
                z_curr = z_expr.isel({sos_dim: i})
                L_i = M_lower.isel({sos_dim: i})
                model.add_constraints(
                    x_i >= L_i * (z_prev + z_curr),
                    name=f"{prefix}{var_name}_lower_mid_{i}",
                )

        # Last element: x[n-1] >= L[n-1] * z[n-2]
        x_last = x_expr.isel({sos_dim: n - 1})
        z_last = z_expr.isel({sos_dim: n - 2})
        L_last = M_lower.isel({sos_dim: n - 1})
        model.add_constraints(
            x_last >= L_last * z_last,
            name=f"{prefix}{var_name}_lower_last",
        )

    # Add cardinality constraint: sum(z) <= 1
    model.add_constraints(
        z.sum(dim=sos_dim) <= 1,
        name=f"{prefix}{var_name}_card",
    )

    logger.debug(f"Reformulated SOS2 constraint for variable '{var_name}'")


def reformulate_all_sos(model: Model, prefix: str = "_sos_reform_") -> list[str]:
    """
    Reformulate all SOS constraints in the model.

    Parameters
    ----------
    model : Model
        The model containing SOS constraints to reformulate.
    prefix : str, optional
        Prefix for naming auxiliary variables and constraints.
        Default is "_sos_reform_".

    Returns
    -------
    list[str]
        List of variable names that were reformulated.
    """
    reformulated_vars = []

    # Get all SOS variables
    sos_vars = list(model.variables.sos)

    for var_name in sos_vars:
        var = model.variables[var_name]
        sos_type = var.attrs.get("sos_type")
        sos_dim = var.attrs.get("sos_dim")

        if sos_type is None or sos_dim is None:
            continue

        # Skip single-element SOS (trivially satisfied)
        if var.sizes[sos_dim] <= 1:
            logger.debug(
                f"Skipping SOS{sos_type} reformulation for '{var_name}' (single element)"
            )
            continue

        # Validate bounds
        validate_bounds_for_reformulation(var)

        # Check if all bounds are zero (variable already fixed to 0)
        M_upper, M_lower = compute_big_m_values(var)
        if (M_upper == 0).all() and (M_lower == 0).all():
            logger.debug(
                f"Skipping SOS{sos_type} reformulation for '{var_name}' (fixed to zero)"
            )
            continue

        # Perform reformulation based on SOS type
        if sos_type == 1:
            reformulate_sos1(model, var, prefix)
        elif sos_type == 2:
            reformulate_sos2(model, var, prefix)

        # Remove the SOS constraint from the variable
        model.remove_sos_constraints(var)
        reformulated_vars.append(var_name)

    logger.info(f"Reformulated {len(reformulated_vars)} SOS constraint(s)")
    return reformulated_vars
