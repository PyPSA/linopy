"""
SOS constraint reformulation using Big-M method.

Converts SOS1/SOS2 constraints to binary + linear constraints for solvers
that don't support them natively.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from linopy.constants import SOS_BIG_M_ATTR, SOS_DIM_ATTR, SOS_TYPE_ATTR

if TYPE_CHECKING:
    from xarray import DataArray

    from linopy.model import Model
    from linopy.variables import Variable

logger = logging.getLogger(__name__)


@dataclass
class SOSReformulationResult:
    """Tracks what was added/changed during SOS reformulation for undo."""

    reformulated: list[str] = field(default_factory=list)
    added_variables: list[str] = field(default_factory=list)
    added_constraints: list[str] = field(default_factory=list)
    saved_attrs: dict[str, dict] = field(default_factory=dict)


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

    big_m_upper = var.attrs.get(SOS_BIG_M_ATTR)
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


def reformulate_sos1(
    model: Model, var: Variable, prefix: str, M: DataArray | None = None
) -> tuple[list[str], list[str]]:
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
    M : DataArray, optional
        Precomputed Big-M values. Computed from variable bounds if not provided.

    Returns
    -------
    tuple[list[str], list[str]]
        Names of added variables and constraints.
    """
    if M is None:
        M = compute_big_m_values(var)
    sos_dim = str(var.attrs[SOS_DIM_ATTR])
    name = var.name

    y_name = f"{prefix}{name}_y"
    upper_name = f"{prefix}{name}_upper"
    card_name = f"{prefix}{name}_card"

    coords = [var.coords[d] for d in var.dims]
    y = model.add_variables(coords=coords, name=y_name, binary=True)

    model.add_constraints(var <= M * y, name=upper_name)
    model.add_constraints(y.sum(dim=sos_dim) <= 1, name=card_name)

    return [y_name], [upper_name, card_name]


def reformulate_sos2(
    model: Model, var: Variable, prefix: str, M: DataArray | None = None
) -> tuple[list[str], list[str]]:
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
    M : DataArray, optional
        Precomputed Big-M values. Computed from variable bounds if not provided.

    Returns
    -------
    tuple[list[str], list[str]]
        Names of added variables and constraints.
    """
    sos_dim = str(var.attrs[SOS_DIM_ATTR])
    name = var.name
    n = var.sizes[sos_dim]

    if n <= 1:
        return [], []

    if M is None:
        M = compute_big_m_values(var)

    z_name = f"{prefix}{name}_z"
    first_name = f"{prefix}{name}_upper_first"
    last_name = f"{prefix}{name}_upper_last"
    card_name = f"{prefix}{name}_card"

    z_coords = [
        pd.Index(var.coords[sos_dim].values[:-1], name=sos_dim)
        if d == sos_dim
        else var.coords[d]
        for d in var.dims
    ]
    z = model.add_variables(coords=z_coords, name=z_name, binary=True)

    x_expr, z_expr = 1 * var, 1 * z

    added_constraints = [first_name]

    model.add_constraints(
        x_expr.isel({sos_dim: 0}) <= M.isel({sos_dim: 0}) * z_expr.isel({sos_dim: 0}),
        name=first_name,
    )

    if n > 2:
        mid_slice = slice(1, n - 1)
        x_mid = x_expr.isel({sos_dim: mid_slice})
        M_mid = M.isel({sos_dim: mid_slice})

        z_left_coords = var.coords[sos_dim].values[: n - 2]
        z_right_coords = var.coords[sos_dim].values[1 : n - 1]

        z_left = z_expr.sel({sos_dim: z_left_coords})
        z_right = z_expr.sel({sos_dim: z_right_coords})

        z_left_aligned = z_left.assign_coords({sos_dim: M_mid.coords[sos_dim].values})
        z_right_aligned = z_right.assign_coords({sos_dim: M_mid.coords[sos_dim].values})

        mid_name = f"{prefix}{name}_upper_mid"
        model.add_constraints(
            x_mid <= M_mid * (z_left_aligned + z_right_aligned),
            name=mid_name,
        )
        added_constraints.append(mid_name)

    model.add_constraints(
        x_expr.isel({sos_dim: n - 1})
        <= M.isel({sos_dim: n - 1}) * z_expr.isel({sos_dim: n - 2}),
        name=last_name,
    )
    added_constraints.extend([last_name, card_name])

    model.add_constraints(z.sum(dim=sos_dim) <= 1, name=card_name)

    return [z_name], added_constraints


def reformulate_all_sos(
    model: Model, prefix: str = "_sos_reform_"
) -> SOSReformulationResult:
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
    SOSReformulationResult
        Tracks what was changed, enabling undo via ``undo_sos_reformulation``.
    """
    result = SOSReformulationResult()

    for var_name in list(model.variables.sos):
        var = model.variables[var_name]
        sos_type = var.attrs[SOS_TYPE_ATTR]
        sos_dim = var.attrs[SOS_DIM_ATTR]

        if var.sizes[sos_dim] <= 1:
            continue

        M = compute_big_m_values(var)
        if (M == 0).all():
            continue

        result.saved_attrs[var_name] = dict(var.attrs)

        if sos_type == 1:
            added_vars, added_cons = reformulate_sos1(model, var, prefix, M)
        elif sos_type == 2:
            added_vars, added_cons = reformulate_sos2(model, var, prefix, M)
        else:
            raise ValueError(f"Unknown sos_type={sos_type} on variable '{var_name}'")

        result.added_variables.extend(added_vars)
        result.added_constraints.extend(added_cons)

        model.remove_sos_constraints(var)
        result.reformulated.append(var_name)

    logger.info(f"Reformulated {len(result.reformulated)} SOS constraint(s)")
    return result


def undo_sos_reformulation(model: Model, result: SOSReformulationResult) -> None:
    """
    Undo a previous SOS reformulation, restoring the model to its original state.

    Parameters
    ----------
    model : Model
        Model that was reformulated.
    result : SOSReformulationResult
        Result from ``reformulate_all_sos`` tracking what was added.
    """
    objective_value = model.objective._value

    for con_name in result.added_constraints:
        if con_name in model.constraints:
            model.remove_constraints(con_name)

    for var_name in result.added_variables:
        if var_name in model.variables:
            model.remove_variables(var_name)

    for var_name, attrs in result.saved_attrs.items():
        if var_name in model.variables:
            model.variables[var_name].attrs.update(attrs)

    model.objective._value = objective_value
