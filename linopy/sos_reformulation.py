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


def compute_big_m_values(var: Variable) -> DataArray:
    """
    Compute Big-M values from variable bounds and custom big_m attribute.

    Uses the tighter of variable upper bound and custom big_m to ensure
    the best possible LP relaxation.

    Parameters
    ----------
    var : Variable
        Variable with bounds (and optionally big_m_upper attr).

    Returns
    -------
    DataArray
        M_upper for reformulation constraints: x <= M_upper * y

    Raises
    ------
    ValueError
        If variable has negative lower bounds or infinite upper bounds.
    """
    # SOS reformulation requires non-negative variables
    if (var.lower < 0).any():
        raise ValueError(
            f"Variable '{var.name}' has negative lower bounds. "
            "SOS reformulation requires non-negative variables (lower >= 0)."
        )

    big_m_upper = var.attrs.get("big_m_upper")
    M_upper = var.upper

    if big_m_upper is not None:
        M_upper = M_upper.clip(max=big_m_upper)  # type: ignore[arg-type]

    # Validate finiteness
    if np.isinf(M_upper).any():
        raise ValueError(
            f"Variable '{var.name}' has infinite upper bounds. "
            "Set finite bounds or specify big_m in add_sos_constraints()."
        )

    return M_upper


def reformulate_sos1(model: Model, var: Variable, prefix: str) -> None:
    """
    Reformulate SOS1 constraint as binary + linear constraints.

    For each x[i] with upper bound M[i]:
    - Add binary indicator y[i]
    - x[i] <= M[i] * y[i]
    - sum(y) <= 1

    Parameters
    ----------
    model : Model
        Model to add reformulation constraints to.
    var : Variable
        Variable with SOS1 constraint (must have non-negative lower bounds).
    prefix : str
        Prefix for naming auxiliary variables and constraints.
    """
    sos_dim = str(var.attrs["sos_dim"])
    name = var.name
    M = compute_big_m_values(var)

    coords = [var.coords[d] for d in var.dims]
    y = model.add_variables(coords=coords, name=f"{prefix}{name}_y", binary=True)

    model.add_constraints(var <= M * y, name=f"{prefix}{name}_upper")
    model.add_constraints(y.sum(dim=sos_dim) <= 1, name=f"{prefix}{name}_card")


def reformulate_sos2(model: Model, var: Variable, prefix: str) -> None:
    """
    Reformulate SOS2 constraint as binary + linear constraints.

    For ordered x[0..n-1] with upper bounds M[i]:
    - Add n-1 binary segment indicators z[i]
    - x[0] <= M[0] * z[0]
    - x[i] <= M[i] * (z[i-1] + z[i])  for middle elements
    - x[n-1] <= M[n-1] * z[n-2]
    - sum(z) <= 1

    Parameters
    ----------
    model : Model
        Model to add reformulation constraints to.
    var : Variable
        Variable with SOS2 constraint (must have non-negative lower bounds).
    prefix : str
        Prefix for naming auxiliary variables and constraints.
    """
    sos_dim = str(var.attrs["sos_dim"])
    name = var.name
    n = var.sizes[sos_dim]

    if n <= 1:
        return

    M = compute_big_m_values(var)

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

    # First: x[0] <= M[0] * z[0]
    model.add_constraints(
        x.isel({sos_dim: 0}) <= M.isel({sos_dim: 0}) * z_expr.isel({sos_dim: 0}),
        name=f"{prefix}{name}_upper_first",
    )
    # Middle: x[i] <= M[i] * (z[i-1] + z[i])
    for i in range(1, n - 1):
        model.add_constraints(
            x.isel({sos_dim: i})
            <= M.isel({sos_dim: i})
            * (z_expr.isel({sos_dim: i - 1}) + z_expr.isel({sos_dim: i})),
            name=f"{prefix}{name}_upper_mid_{i}",
        )
    # Last: x[n-1] <= M[n-1] * z[n-2]
    model.add_constraints(
        x.isel({sos_dim: n - 1})
        <= M.isel({sos_dim: n - 1}) * z_expr.isel({sos_dim: n - 2}),
        name=f"{prefix}{name}_upper_last",
    )

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
        M = compute_big_m_values(var)
        if (M == 0).all():
            continue

        if sos_type == 1:
            reformulate_sos1(model, var, prefix)
        elif sos_type == 2:
            reformulate_sos2(model, var, prefix)

        model.remove_sos_constraints(var)
        reformulated.append(var_name)

    logger.info(f"Reformulated {len(reformulated)} SOS constraint(s)")
    return reformulated
