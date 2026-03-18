"""
Linopy dual module.

This module contains implementations for constructing the dual of a linear optimization problem.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from linopy.expressions import LinearExpression

if TYPE_CHECKING:
    from linopy.model import Model

logger = logging.getLogger(__name__)


def _var_lookup(m: Model) -> dict:
    """
    Build a flat label -> (var_name, coord_dict) lookup for all variables in m.

    Used to map entries in m.matrices.vlabels back to their variable name
    and xarray coordinates for use in dual feasibility constraint construction.

    Skips masked entries (label == -1) and empty variables.

    Parameters
    ----------
    m : Model
        Primal linopy model.

    Returns
    -------
    dict
        Mapping from flat integer label to (var_name, coord_dict) tuple.
    """
    var_lookup = {}
    logger.debug("Building variable label lookup.")
    for var_name, var in m.variables.items():
        labels = var.labels
        flat_labels = labels.values.flatten()

        if len(flat_labels) == 0:
            logger.debug(f"Skipping empty variable '{var_name}'.")
            continue
        if not (flat_labels != -1).any():
            logger.debug(f"Variable '{var_name}' is fully masked, skipping.")
            continue

        logger.debug(
            f"Creating label lookup for variable '{var_name}' with shape {labels.shape} and dims {labels.dims}."
        )

        coord_arrays = (
            np.meshgrid(
                *[labels.coords[dim].values for dim in labels.dims], indexing="ij"
            )
            if len(labels.dims) > 0
            else []
        )
        flat_coords = [arr.flatten() for arr in coord_arrays]

        for k, flat in enumerate(flat_labels):
            if flat != -1:
                var_lookup[int(flat)] = (
                    var_name,
                    {dim: flat_coords[i][k] for i, dim in enumerate(labels.dims)},
                )

    return var_lookup


def _con_lookup(m: Model) -> dict:
    """
    Build a flat label -> (con_name, coord_dict) lookup for all constraints in m.

    Used to map entries in m.matrices.clabels back to their constraint name
    and xarray coordinates for use in dual feasibility constraint construction.

    Skips masked entries (label == -1) and empty or fully-masked constraints.

    Parameters
    ----------
    m : Model
        Primal linopy model.

    Returns
    -------
    dict
        Mapping from flat integer label to (con_name, coord_dict) tuple.
    """
    con_lookup = {}
    logger.debug("Building constraint label lookup.")
    for con_name, con in m.constraints.items():
        labels = con.labels
        flat_labels = labels.values.flatten()

        if len(flat_labels) == 0:
            logger.debug(f"Skipping empty constraint '{con_name}'.")
            continue
        if not (flat_labels != -1).any():
            logger.debug(f"Constraint '{con_name}' is fully masked, skipping.")
            continue

        logger.debug(
            f"Creating label lookup for constraint '{con_name}' with shape {labels.shape} and dims {labels.dims}."
        )

        coord_arrays = (
            np.meshgrid(
                *[labels.coords[dim].values for dim in labels.dims], indexing="ij"
            )
            if len(labels.dims) > 0
            else []
        )
        flat_coords = [arr.flatten() for arr in coord_arrays]

        for k, flat in enumerate(flat_labels):
            if flat != -1:
                con_lookup[int(flat)] = (
                    con_name,
                    {dim: flat_coords[i][k] for i, dim in enumerate(labels.dims)},
                )

    return con_lookup


def bounds_to_constraints(self) -> None:
    """
    Add explicit bound constraints for variables with bounds set directly
    in the variable rather than via explicit constraints.

    Adds constraints named '{var_name}-bound-lower' and '{var_name}-bound-upper'
    to distinguish from PyPSA's automatic '-fix-*' constraints.

    Also resets variable bounds to [-inf, inf] after adding constraints,
    to avoid double-counting in the dual.
    """
    logger.debug("Converting variable bounds to explicit constraints.")
    logger.debug("Relaxing variable bounds to [-inf, inf].")
    for var_name, var in self.variables.items():
        mask = var.labels != -1
        lb = var.lower
        ub = var.upper

        # lower bound
        if f"{var_name}-bound-lower" not in self.constraints:
            has_finite_lb = np.isfinite(lb.values[mask.values]).any()
            if has_finite_lb:
                self.add_constraints(
                    var >= lb,
                    name=f"{var_name}-bound-lower",
                    mask=mask,
                )
                logger.debug(f"Added lower bound constraint for '{var_name}'.")
                var.lower.values[mask.values] = -np.inf
                # Remove bounds to avoid double-counting in the dual. Rely on the new constraints instead.
                self.variables[var_name].lower.values[mask.values] = -np.inf
            else:
                logger.debug(
                    f"Variable '{var_name}' has no finite lower bound, skipping."
                )

        # upper bound
        if f"{var_name}-bound-upper" not in self.constraints:
            has_finite_ub = np.isfinite(ub.values[mask.values]).any()
            if has_finite_ub:
                self.add_constraints(
                    var <= ub,
                    name=f"{var_name}-bound-upper",
                    mask=mask,
                )
                logger.debug(f"Added upper bound constraint for '{var_name}'.")
                var.upper.values[mask.values] = np.inf
                # Remove bounds to avoid double-counting in the dual. Rely on the new constraints instead.
                self.variables[var_name].upper.values[mask.values] = np.inf
            else:
                logger.debug(
                    f"Variable '{var_name}' has no finite upper bound, skipping."
                )


def _add_dual_variables(m: Model, m2: Model) -> dict:
    """
    Add dual variables to m2 corresponding to constraints in m.

    For each active constraint in m, adds a dual variable to m2 following
    linopy's sign convention:

    - Equality constraints (=)  -> free dual variable (lower=-inf, upper=inf)
    - <= constraints            -> non-positive dual variable (lower=-inf, upper=0)
    - >= constraints            -> non-negative dual variable (lower=0, upper=inf)

    This convention ensures that m2.variables[con_name].solution has the same
    sign as m.constraints[con_name].dual after solving, allowing direct
    comparison without sign adjustments.

    The sign encodes the direction of impact on the objective per unit RHS change:
    - <= constraint dual (<=0): increasing RHS by 1 unit changes objective by
      dual units (negative = cost decreases, i.e. relaxing the constraint).
    - >= constraint dual (>=0): increasing RHS by 1 unit changes objective by
      dual units (positive = cost increases, i.e. tightening the constraint).

    Skips constraints with no active rows (empty or fully masked).

    Parameters
    ----------
    m : Model
        Primal linopy model containing the constraints to dualize.
    m2 : Model
        Dual linopy model to which dual variables are added.

    Returns
    -------
    dict
        Mapping from constraint name (str) to the corresponding dual
        variable (linopy.Variable) in m2.
    """
    dual_vars = {}
    for name, con in m.constraints.items():
        sign_vals = con.sign.values.flatten()

        if len(sign_vals) == 0:
            logger.warning(f"Constraint '{name}' has no sign values, skipping.")
            continue

        mask = con.labels != -1
        if not mask.any():
            logger.debug(f"Constraint '{name}' is fully masked, skipping.")
            continue

        if sign_vals[0] == "=":
            lower, upper = -np.inf, np.inf
            var_type = "free"
        elif sign_vals[0] == "<=":
            lower, upper = -np.inf, 0
            var_type = "non-positive"
        else:  # >=
            lower, upper = 0, np.inf
            var_type = "non-negative"

        logger.debug(
            f"Adding {var_type} dual variable for constraint '{name}' with shape {con.shape} and dims {con.labels.dims}."
        )
        dual_vars[name] = m2.add_variables(
            lower=lower,
            upper=upper,
            coords=list(con.coords.values()),
            name=name,
            mask=mask,
        )

    return dual_vars


def _build_dual_feas_terms(
    m: Model,
    dual_vars: dict,
    var_lookup: dict,
    con_lookup: dict,
) -> dict:
    """
    Build dual feasibility terms for each primal variable in m.

    For each active primal variable x_j, collects the constraint matrix
    entries A_ji and their corresponding constraint names and coordinates,
    forming the terms of the stationarity condition:
        sum_i (A_ji * lambda_i) = c_j

    Raw constraint matrix coefficients are used directly without sign
    factors, as the sign convention is encoded in the dual variable bounds:
    - <= constraints: lambda_i <= 0
    - >= constraints: lambda_i >= 0
    - =  constraints: lambda_i free

    Parameters
    ----------
    m : Model
        Primal linopy model.
    dual_vars : dict
        Mapping from constraint name to dual variable in m2,
        as returned by _add_dual_variables(). Used to skip constraints
        that were not dualized (e.g. empty or fully masked).
    var_lookup : dict
        Mapping from flat variable label to (var_name, coord_dict),
        as returned by _var_lookup().
    con_lookup : dict
        Mapping from flat constraint label to (con_name, coord_dict),
        as returned by _con_lookup().

    Returns
    -------
    dict
        Nested dict: {var_name: {flat_label: (var_coords, terms, obj_coeff)}}
        where terms is a list of (con_name, con_coords, coeff) tuples.
    """
    A_csc = m.matrices.A.tocsc()
    c = m.matrices.c
    indptr = A_csc.indptr
    indices = A_csc.indices
    data = A_csc.data
    vlabels = m.matrices.vlabels
    clabels = m.matrices.clabels

    dual_feas_terms = {var_name: {} for var_name in m.variables}

    logger.debug("Building dual feasibility terms for each primal variable.")

    for i in range(A_csc.shape[1]):
        flat_var = vlabels[i]
        if flat_var == -1:
            continue
        if flat_var not in var_lookup:
            continue
        var_name, var_coords = var_lookup[flat_var]
        terms = []
        for k in range(indptr[i], indptr[i + 1]):
            j = indices[k]
            flat_con = clabels[j]
            if flat_con == -1:
                continue
            if flat_con not in con_lookup:
                continue
            con_name, con_coords = con_lookup[flat_con]
            if con_name not in dual_vars:
                continue
            coeff = data[k]
            terms.append((con_name, con_coords, coeff))
        dual_feas_terms[var_name][flat_var] = (var_coords, terms, c[i])

    return dual_feas_terms


def _add_dual_feasibility_constraints(
    m: Model,
    m2: Model,
    dual_vars: dict,
    var_lookup: dict,
    con_lookup: dict,
) -> None:
    """
    Add dual feasibility constraints to m2.

    For each primal variable x_j in m, adds the stationarity constraint:
        sum_i (A_ji * lambda_i) = c_j
    where:
    - A is the primal constraint matrix
    - lambda_i are the dual variables in m2
    - c_j is the objective coefficient of x_j

    Raw constraint matrix coefficients are used directly without sign factors,
    because the sign convention is encoded in the dual variable bounds:
    - <= constraints: lambda_i <= 0
    - >= constraints: lambda_i >= 0
    - =  constraints: lambda_i free

    Skips masked variable entries (label == -1) and variables not present
    in var_lookup (e.g. from empty constraints).

    Parameters
    ----------
    m : Model
        Primal linopy model.
    m2 : Model
        Dual linopy model.
    dual_vars : dict
        Mapping from constraint name to dual variable in m2,
        as returned by _add_dual_variables().
    var_lookup : dict
        Mapping from flat variable label to (var_name, coord_dict),
        as returned by _var_lookup().
    con_lookup : dict
        Mapping from flat constraint label to (con_name, coord_dict),
        as returned by _con_lookup().
    """

    dual_feas_terms = _build_dual_feas_terms(m, dual_vars, var_lookup, con_lookup)

    c = m.matrices.c
    vlabels = m.matrices.vlabels

    # build objective coefficient lookup by flat variable label
    c_by_label = {vlabels[i]: c[i] for i in range(len(vlabels))}

    # add dual feasibility constraints to m2
    logger.debug("Adding dual feasibility constraints to model.")
    for var_name, var in m.variables.items():
        coords = [
            pd.Index(var.labels.coords[dim].values, name=dim) for dim in var.labels.dims
        ]
        mask = var.labels != -1

        c_vals = xr.DataArray(
            np.vectorize(lambda flat: c_by_label.get(flat, 0.0))(var.labels.values),
            coords=var.labels.coords,
        )

        def rule(m, *coord_vals, vname=var_name, vdims=var.labels.dims):
            coord_dict = dict(zip(vdims, coord_vals))
            flat = var.labels.sel(**coord_dict).item()
            if flat == -1:
                return None
            if flat not in dual_feas_terms[vname]:
                return None
            _, terms, _ = dual_feas_terms[vname][flat]
            if not terms:
                return None
            return sum(
                coeff * dual_vars[con_name].at[tuple(con_coords.values())]
                for con_name, con_coords, coeff in terms
            )

        lhs = LinearExpression.from_rule(m2, rule, coords)
        m2.add_constraints(lhs == c_vals, name=var_name, mask=mask)


def _add_dual_objective(
    m: Model,
    m2: Model,
    dual_vars: dict,
    add_objective_constant: float = 0.0,
) -> None:
    """
    Construct and add the dual objective to m2.

    The dual objective is sum(rhs * dual) over all constraints, added uniformly
    with a + sign. The sign convention is encoded in the dual variable bounds:
    - <= constraints: dual <= 0, so rhs * dual contributes negatively
    - >= constraints: dual >= 0, so rhs * dual contributes positively
    - =  constraints: dual free

    This matches linopy's and Gurobi's native dual sign convention, allowing
    direct comparison between m2.variables[con_name].solution and
    m.constraints[con_name].dual without sign adjustments.

    The dual objective sense is flipped relative to the primal:
    - min primal -> max dual
    - max primal -> min dual

    Parameters
    ----------
    m : Model
        Primal linopy model.
    m2 : Model
        Dual linopy model.
    dual_vars : dict
        Mapping from constraint name to dual variable in m2,
        as returned by _add_dual_variables().
    add_objective_constant : float, optional
        Constant term to add to the dual objective. Use this to pass through
        a primal objective constant excluded via include_objective_constant=False
        during model creation. Default is 0.0.
    """
    dual_obj = 0
    sense = "max" if m.objective.sense == "min" else "min"

    for name, con in m.constraints.items():
        if name not in dual_vars:
            continue

        mask = con.labels != -1
        rhs_masked = con.rhs.where(mask, 0)
        dual_obj += (rhs_masked * dual_vars[name]).sum()

    if add_objective_constant != 0.0:
        dual_obj += add_objective_constant
        logger.debug(f"Added constant {add_objective_constant} to dual objective.")

    logger.debug(f"Constructed dual objective with {len(dual_obj.coeffs)} terms.")
    logger.debug("Adding dual objective to model.")
    m2.add_objective(dual_obj, sense=sense, overwrite=True)


def dualize(
    self,
    add_objective_constant: float = 0.0,
) -> Model:
    """
    Construct the dual of a linopy LP model.

    Transforms the primal model into its dual equivalent m2 by:
    1. Converting variable bounds to explicit constraints
    2. Adding dual variables to m2 (one per active constraint)
    3. Adding dual feasibility constraints to m2 (one per primal variable)
    4. Adding the dual objective to m2

    The dual is constructed following standard LP duality theory:

    Primal (min):                       Dual (max):
    min  c^T x                          max  b_eq^T λ + b_leq^T μ + b_geq^T ν
    s.t.    A_eq x  =  b_eq   : λ free  s.t.    A_eq^T λ + A_leq^T μ + A_geq^T ν = c
            A_leq x <= b_leq  : μ <= 0          λ free, μ <= 0, ν >= 0
            A_geq x >= b_geq  : ν >= 0

    Variable bounds are converted to explicit constraints before dualization
    via bounds_to_constraints(), so that they appear in the constraint matrix
    A and are correctly reflected in the dual.

    The dual variables in m2 are named identically to their corresponding
    primal constraints and are accessible via m2.variables[con_name].

    Strong duality guarantees that at optimality:
        primal objective = dual objective

    Note: The standalone dual m2 may be unbounded if the primal is degenerate.

    Parameters
    ----------
    add_objective_constant : float, optional
        Constant term to add to the dual objective. Use this to pass through
        a primal objective constant. Default is 0.0.

    Returns
    -------
    Model
        The dual linopy model. Dual variables are named after their
        corresponding primal constraints.

    Examples
    --------
    >>> m2 = m.dualize()
    >>> m2.solve(solver_name="gurobi", Method=2, Crossover=0)
    >>> gap = abs(m.objective.value - m2.objective.value)
    """
    from linopy.model import Model

    m = self.copy()
    m2 = Model()

    if not m.variables or not m.constraints:
        logger.warning(
            "Primal model has no variables or constraints. Returning empty dual model."
        )
        return m2

    m.bounds_to_constraints()
    var_lup = _var_lookup(m)
    con_lup = _con_lookup(m)
    dual_vars = _add_dual_variables(m, m2)
    _add_dual_feasibility_constraints(m, m2, dual_vars, var_lup, con_lup)
    _add_dual_objective(m, m2, dual_vars, add_objective_constant=add_objective_constant)
    return m2
