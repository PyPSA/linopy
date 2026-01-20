"""
SOS constraint reformulation using Big-M method.

Converts SOS1/SOS2 constraints to binary + linear constraints for solvers
that don't support them natively.
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


def compute_big_m_values(var: Variable) -> tuple[DataArray, DataArray]:
    """
    Compute Big-M values from variable bounds and custom big_m attributes.

    Uses the tighter of variable bounds and custom big_m values to ensure
    the best possible LP relaxation.

    Parameters
    ----------
    var : Variable
        Variable with bounds (and optionally big_m_upper/big_m_lower attrs).

    Returns
    -------
    tuple[DataArray, DataArray]
        (M_upper, M_lower) for reformulation constraints:
        x <= M_upper * y and x >= M_lower * y

    Raises
    ------
    ValueError
        If resulting Big-M values are infinite.
    """
    big_m_upper = var.attrs.get("big_m_upper")
    big_m_lower = var.attrs.get("big_m_lower")

    M_upper = var.upper
    M_lower = var.lower

    if big_m_upper is not None:
        M_upper = np.minimum(M_upper, big_m_upper)
    if big_m_lower is not None:
        M_lower = np.maximum(M_lower, big_m_lower)

    # Validate finiteness
    if np.isinf(M_upper).any():
        raise ValueError(
            f"Variable '{var.name}' has infinite upper bounds. "
            "Set finite bounds or specify big_m in add_sos_constraints()."
        )
    if np.isinf(M_lower).any():
        raise ValueError(
            f"Variable '{var.name}' has infinite lower bounds. "
            "Set finite bounds or specify big_m in add_sos_constraints()."
        )

    return M_upper, M_lower


def reformulate_sos1(model: Model, var: Variable, prefix: str) -> None:
    """
    Reformulate SOS1 constraint as binary + linear constraints.

    For each x[i] with bounds [L[i], U[i]]:
    - Add binary indicator y[i]
    - x[i] <= U[i] * y[i]  (upper linking, if U > 0)
    - x[i] >= L[i] * y[i]  (lower linking, if L < 0)
    - sum(y) <= 1          (cardinality)

    Parameters
    ----------
    model : Model
        Model to add reformulation constraints to.
    var : Variable
        Variable with SOS1 constraint.
    prefix : str
        Prefix for naming auxiliary variables and constraints.
    """
    sos_dim = var.attrs["sos_dim"]
    name = var.name
    M_upper, M_lower = compute_big_m_values(var)

    coords = [var.coords[d] for d in var.dims]
    y = model.add_variables(coords=coords, name=f"{prefix}{name}_y", binary=True)

    if (M_upper > 0).any():
        model.add_constraints(var <= M_upper * y, name=f"{prefix}{name}_upper")
    if (M_lower < 0).any():
        model.add_constraints(var >= M_lower * y, name=f"{prefix}{name}_lower")

    model.add_constraints(y.sum(dim=sos_dim) <= 1, name=f"{prefix}{name}_card")


def reformulate_sos2(model: Model, var: Variable, prefix: str) -> None:
    """
    Reformulate SOS2 constraint as binary + linear constraints.

    For ordered x[0..n-1]:
    - Add n-1 binary segment indicators z[i]
    - x[0] <= U[0] * z[0]
    - x[i] <= U[i] * (z[i-1] + z[i])  for middle elements
    - x[n-1] <= U[n-1] * z[n-2]
    - Similar for lower bounds if L < 0
    - sum(z) <= 1

    Parameters
    ----------
    model : Model
        Model to add reformulation constraints to.
    var : Variable
        Variable with SOS2 constraint.
    prefix : str
        Prefix for naming auxiliary variables and constraints.
    """
    sos_dim = var.attrs["sos_dim"]
    name = var.name
    n = var.sizes[sos_dim]

    if n <= 1:
        return

    M_upper, M_lower = compute_big_m_values(var)

    # Create n-1 segment indicators
    z_coords = [
        pd.Index(var.coords[sos_dim].values[:-1], name=sos_dim)
        if d == sos_dim
        else var.coords[d]
        for d in var.dims
    ]
    z = model.add_variables(coords=z_coords, name=f"{prefix}{name}_z", binary=True)

    # Convert to expressions to avoid SOS attr validation on isel
    x, z_expr = 1 * var, 1 * z

    def add_linking_constraints(M: DataArray, sign: str, suffix: str) -> None:
        """Add x <= M*z or x >= M*z constraints for first/middle/last elements."""
        if sign == "upper" and not (M > 0).any():
            return
        if sign == "lower" and not (M < 0).any():
            return

        op = (lambda a, b: a <= b) if sign == "upper" else (lambda a, b: a >= b)

        # First: x[0] op M[0] * z[0]
        model.add_constraints(
            op(x.isel({sos_dim: 0}), M.isel({sos_dim: 0}) * z_expr.isel({sos_dim: 0})),
            name=f"{prefix}{name}_{suffix}_first",
        )
        # Middle: x[i] op M[i] * (z[i-1] + z[i])
        for i in range(1, n - 1):
            model.add_constraints(
                op(
                    x.isel({sos_dim: i}),
                    M.isel({sos_dim: i})
                    * (z_expr.isel({sos_dim: i - 1}) + z_expr.isel({sos_dim: i})),
                ),
                name=f"{prefix}{name}_{suffix}_mid_{i}",
            )
        # Last: x[n-1] op M[n-1] * z[n-2]
        model.add_constraints(
            op(
                x.isel({sos_dim: n - 1}),
                M.isel({sos_dim: n - 1}) * z_expr.isel({sos_dim: n - 2}),
            ),
            name=f"{prefix}{name}_{suffix}_last",
        )

    add_linking_constraints(M_upper, "upper", "upper")
    add_linking_constraints(M_lower, "lower", "lower")

    model.add_constraints(z.sum(dim=sos_dim) <= 1, name=f"{prefix}{name}_card")


def reformulate_all_sos(model: Model, prefix: str = "_sos_reform_") -> list[str]:
    """
    Reformulate all SOS constraints in the model.

    Parameters
    ----------
    model : Model
        Model containing SOS constraints to reformulate.
    prefix : str, optional
        Prefix for auxiliary variables and constraints. Default: "_sos_reform_"

    Returns
    -------
    list[str]
        Names of variables that were reformulated.
    """
    reformulated = []

    for var_name in list(model.variables.sos):
        var = model.variables[var_name]
        sos_type = var.attrs.get("sos_type")
        sos_dim = var.attrs.get("sos_dim")

        if sos_type is None or sos_dim is None:
            continue
        if var.sizes[sos_dim] <= 1:
            continue

        # Check if fixed to zero
        M_upper, M_lower = compute_big_m_values(var)
        if (M_upper == 0).all() and (M_lower == 0).all():
            continue

        if sos_type == 1:
            reformulate_sos1(model, var, prefix)
        elif sos_type == 2:
            reformulate_sos2(model, var, prefix)

        model.remove_sos_constraints(var)
        reformulated.append(var_name)

    logger.info(f"Reformulated {len(reformulated)} SOS constraint(s)")
    return reformulated
