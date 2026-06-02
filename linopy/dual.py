"""
Linopy dual module.

This module contains implementations for constructing the dual of a linear optimization problem.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import xarray as xr

from linopy.expressions import LinearExpression

if TYPE_CHECKING:
    from linopy.model import Model

logger = logging.getLogger(__name__)


def _skip(
    da: xr.DataArray, component_type: Literal["variable", "constraint"], name: str
) -> bool:
    """Return True if the label array is empty or entirely masked (all -1)."""
    if da.size == 0:
        logger.debug(f"Skipping empty {component_type} '{name}'.")
        return True

    if (da == -1).all():
        logger.debug(f"{component_type} '{name}' is fully masked, skipping.")
        return True
    return False


def _lift_bounds_to_constraints(m: Model) -> None:
    """
    Convert finite variable bounds to explicit ``>=`` / ``<=`` constraints.

    Each finite lower bound becomes a ``'{var_name}-bound-lower'`` constraint and
    each finite upper bound becomes a ``'{var_name}-bound-upper'`` constraint.
    The variable bounds are then relaxed to ``[-inf, inf]`` to avoid
    double-counting when the dual is formed.

    Parameters
    ----------
    m : Model
        Model to mutate in-place.
    """
    logger.debug("Converting variable bounds to explicit constraints.")
    logger.debug("Relaxing variable bounds to [-inf, inf].")
    for var_name, var in m.variables.items():
        mask = var.labels != -1
        lb = var.lower
        ub = var.upper

        # lower bound
        if f"{var_name}-bound-lower" not in m.constraints:
            has_finite_lb = np.isfinite(lb.values[mask.values]).any()
            if has_finite_lb:
                m.add_constraints(
                    var >= lb,
                    name=f"{var_name}-bound-lower",
                    mask=mask,
                )
                logger.debug(f"Added lower bound constraint for '{var_name}'.")
                var.lower.values[mask.values] = -np.inf
                # Remove bounds to avoid double-counting in the dual. Rely on the new constraints instead.
                m.variables[var_name].lower.values[mask.values] = -np.inf
            else:
                logger.debug(
                    f"Variable '{var_name}' has no finite lower bound, skipping."
                )

        # upper bound
        if f"{var_name}-bound-upper" not in m.constraints:
            has_finite_ub = np.isfinite(ub.values[mask.values]).any()
            if has_finite_ub:
                m.add_constraints(
                    var <= ub,
                    name=f"{var_name}-bound-upper",
                    mask=mask,
                )
                logger.debug(f"Added upper bound constraint for '{var_name}'.")
                var.upper.values[mask.values] = np.inf
                # Remove bounds to avoid double-counting in the dual. Rely on the new constraints instead.
                m.variables[var_name].upper.values[mask.values] = np.inf
            else:
                logger.debug(
                    f"Variable '{var_name}' has no finite upper bound, skipping."
                )


def _add_dual_variables(m: Model, m2: Model) -> dict:
    """
    Add one dual variable to m2 for each included constraint in m.

    Dual variable bounds encode the sign convention for each constraint type
    and primal objective sense:

    ============  ===========  ================  ================
    Constraint    Primal sense  lower             upper
    ============  ===========  ================  ================
    =             min / max     -inf              +inf  (free)
    <=            min           -inf              0
    <=            max           0                 +inf
    >=            min           0                 +inf
    >=            max           -inf              0
    ============  ===========  ================  ================

    Fully masked or empty constraints are skipped.

    Parameters
    ----------
    m : Model
        Primal model.
    m2 : Model
        Dual model to populate.

    Returns
    -------
    dict
        ``{constraint_name: dual_variable}`` for every dualized constraint.
    """
    primal_is_min = m.objective.sense == "min"

    dual_vars = {}
    for name, con in m.constraints.items():
        if _skip(con.labels, "constraint", name):
            continue

        mask = con.labels != -1
        sign = con.sign.isel({d: 0 for d in con.sign.dims}).item()

        match sign:
            case "=":
                lower, upper = -np.inf, np.inf
                var_type = "free"
            case "<=":
                lower, upper = (-np.inf, 0) if primal_is_min else (0, np.inf)
                var_type = "non-positive" if primal_is_min else "non-negative"
            case ">=":
                lower, upper = (0, np.inf) if primal_is_min else (-np.inf, 0)
                var_type = "non-negative" if primal_is_min else "non-positive"
            case _:
                logger.warning(
                    f"Constraint '{name}' has unrecognized sign '{sign}', skipping."
                )
                continue

        logger.debug(
            f"Adding {var_type} dual variable for constraint '{name}' with shape {con.shape} and dims {con.labels.dims}."
        )
        coords = (
            [con.labels.coords[dim] for dim in con.labels.dims]
            if con.labels.dims
            else None
        )
        dual_vars[name] = m2.add_variables(
            lower=lower,
            upper=upper,
            coords=coords,
            name=name,
            mask=mask,
        )
    return dual_vars


def _build_flat_con_to_dual_label_lookup(m: Model, dual_vars: dict) -> np.ndarray:
    """
    Build a lookup from flat primal constraint labels to flat dual variable labels.

    The returned array maps each flat constraint label to the corresponding
    flat dual variable label in ``m2``. Entries are -1 for constraints not in
    ``dual_vars`` or with masked labels.

    Parameters
    ----------
    m : Model
        Primal model.
    dual_vars : dict
        ``{constraint_name: dual_variable}`` as returned by _add_dual_variables().

    Returns
    -------
    np.ndarray of int64
        Lookup array of length ``max_flat_con_label + 1``; empty if no valid
        constraint labels exist.
    """
    max_flat_con = -1
    for con_name in dual_vars:
        flat = m.constraints[con_name].labels.values.ravel()
        valid = flat[flat != -1]
        if len(valid):
            max_flat_con = max(max_flat_con, int(valid.max()))

    if max_flat_con < 0:
        return np.array([], dtype=np.int64)

    lookup = np.full(max_flat_con + 1, -1, dtype=np.int64)
    for con_name, dv in dual_vars.items():
        con_flat = m.constraints[con_name].labels.values.ravel().astype(np.int64)
        dv_flat = dv.labels.values.ravel().astype(np.int64)
        valid = (con_flat != -1) & (dv_flat != -1)
        if valid.any():
            lookup[con_flat[valid]] = dv_flat[valid]

    return lookup


def _extract_dual_feas_entries(
    A: Any,
    vlabels: np.ndarray,
    clabels: np.ndarray,
    flat_con_to_dual: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return ``(flat_var_label, flat_dual_label, coeff)`` for each included nonzero entry of ``A``.

    Converts sparse matrix ``A`` to coordinate format, maps rows/columns to
    flat labels via ``clabels`` / ``vlabels``, and drops entries that are masked
    or belong to a constraint without a dual variable.

    Parameters
    ----------
    A : scipy sparse matrix
        Primal constraint matrix in any scipy sparse format.
    vlabels : np.ndarray of int64
        Flat variable label per column of ``A``; -1 for masked columns.
    clabels : np.ndarray of int64
        Flat constraint label per row of ``A``; -1 for masked rows.
    flat_con_to_dual : np.ndarray of int64
        Lookup array as returned by ``_build_flat_con_to_dual_label_lookup()``.

    Returns
    -------
    flat_v : np.ndarray of int64
        Flat primal variable label for each retained nonzero entry.
    flat_d : np.ndarray of int64
        Corresponding flat dual variable label.
    coeffs : np.ndarray of float64
        Corresponding coefficient from ``A``.
    """
    A_coo = A.tocoo()
    flat_v = vlabels[A_coo.col].astype(np.int64)
    flat_c = clabels[A_coo.row].astype(np.int64)
    coeffs = A_coo.data

    # Drop entries where either label is masked.
    has_labels = (flat_v != -1) & (flat_c != -1)
    flat_v, flat_c, coeffs = flat_v[has_labels], flat_c[has_labels], coeffs[has_labels]

    # Map primal constraint labels to flat dual variable labels.
    n = len(flat_con_to_dual)
    in_range = flat_c < n
    flat_d = np.full(len(flat_c), -1, dtype=np.int64)
    flat_d[in_range] = flat_con_to_dual[flat_c[in_range]]

    # Drop entries with no corresponding dual variable.
    has_dual = flat_d != -1
    return flat_v[has_dual], flat_d[has_dual], coeffs[has_dual]


def _build_obj_coeff_lookup(vlabels: np.ndarray, c_vec: np.ndarray) -> np.ndarray:
    """
    Build a lookup from flat variable labels to objective coefficients.

    The returned array maps each valid flat variable label to its objective
    coefficient. Labels that do not occur in ``vlabels`` but fall within the
    lookup range have value 0.0.

    Parameters
    ----------
    vlabels : np.ndarray of int64
        Flat variable label per column of ``A``; -1 for masked columns.
    c_vec : np.ndarray of float64
        Objective coefficient per column, in the same ordering as ``vlabels``.

    Returns
    -------
    np.ndarray of float64
        Lookup array of length ``max_flat_var_label + 1``; empty if no valid
        variable labels exist.
    """
    valid_mask = vlabels != -1
    valid_vlabels = vlabels[valid_mask]
    if not len(valid_vlabels):
        return np.array([], dtype=np.float64)

    lookup = np.zeros(int(valid_vlabels.max()) + 1, dtype=np.float64)
    lookup[valid_vlabels.astype(np.int64)] = c_vec[valid_mask]
    return lookup


def _build_dual_feas_lhs(
    var: Any,
    flat_v: np.ndarray,
    flat_d: np.ndarray,
    nnz_data: np.ndarray,
    m_dual: Model,
) -> LinearExpression:
    """
    Build the dual-feasibility LHS for one primal variable.

    For each coordinate of ``var``, constructs the linear expression
    ``sum_j A[j, var] * lambda[j]`` over the corresponding dual variables.

    Parameters
    ----------
    var : linopy.Variable
        Primal variable whose dual-feasibility LHS is being constructed.
    flat_v : np.ndarray of int64
        Flat primal variable labels for all included nonzero entries, as returned
        by ``_extract_dual_feas_entries()``.
    flat_d : np.ndarray of int64
        Corresponding flat dual variable labels.
    nnz_data : np.ndarray of float64
        Corresponding nonzero coefficients from the primal constraint matrix.
    m_dual : Model
        Dual model owning the dual variables.

    Returns
    -------
    LinearExpression
        Dual-feasibility LHS over ``var``'s coordinate space. Returns an empty
        expression (constant zero) when ``var`` has no included constraint-matrix
        entries.
    """
    var_flat = var.labels.values.ravel().astype(np.int64)
    n_elements = len(var_flat)
    valid_mask = var_flat != -1

    if not valid_mask.any():
        return LinearExpression(None, m_dual)

    # Map flat variable labels to ravelled positions in this variable.
    max_fv = int(var_flat[valid_mask].max())
    var_to_idx = np.full(max_fv + 1, -1, dtype=np.int64)
    var_to_idx[var_flat[valid_mask]] = np.where(valid_mask)[0].astype(np.int64)

    # Locate entries that belong to this variable via bounded lookup.
    lin_idx = np.full(len(flat_v), -1, dtype=np.int64)
    in_range = (flat_v >= 0) & (flat_v <= max_fv)
    if in_range.any():
        lin_idx[in_range] = var_to_idx[flat_v[in_range]]

    keep = lin_idx != -1
    lin_idx_f, flat_d_f, coeffs_f = lin_idx[keep], flat_d[keep], nnz_data[keep]

    if not len(lin_idx_f):
        return LinearExpression(None, m_dual)

    # Sort by ravelled position so entries for the same element are contiguous.
    order = np.argsort(lin_idx_f, kind="stable")
    lin_idx_s = lin_idx_f[order]
    flat_d_s = flat_d_f[order]
    coeffs_s = coeffs_f[order]

    # Compute within-group term-slot positions.
    # group_start[i] is True at the first entry of each new element group.
    group_start = np.empty(len(lin_idx_s), dtype=bool)
    group_start[0] = True
    group_start[1:] = lin_idx_s[1:] != lin_idx_s[:-1]
    group_ids = np.cumsum(group_start) - 1
    group_start_pos = np.where(group_start)[0]
    col_idx = np.arange(len(lin_idx_s), dtype=np.int64) - group_start_pos[group_ids]

    max_terms = int(col_idx.max()) + 1

    # Populate (n_elements, max_terms) arrays via advanced indexing.
    dual_labels_2d = np.full((n_elements, max_terms), -1, dtype=np.int64)
    dual_coeffs_2d = np.zeros((n_elements, max_terms), dtype=np.float64)
    dual_labels_2d[lin_idx_s, col_idx] = flat_d_s
    dual_coeffs_2d[lin_idx_s, col_idx] = coeffs_s

    # Wrap in a LinearExpression, reshaping to (*var_dims, _term).
    target_shape = var.labels.shape + (max_terms,)
    dims = list(var.labels.dims) + ["_term"]
    ds = xr.Dataset(
        {
            "vars": xr.DataArray(dual_labels_2d.reshape(target_shape), dims=dims),
            "coeffs": xr.DataArray(dual_coeffs_2d.reshape(target_shape), dims=dims),
        },
        coords={dim: var.labels.coords[dim] for dim in var.labels.dims},
    )
    return LinearExpression(ds, m_dual)


def _add_dual_feasibility_constraints(
    m: Model,
    m2: Model,
    dual_vars: dict,
) -> None:
    """
    Add the stationarity constraint ``sum_i(A_ji * lambda_i) = c_j`` for each primal variable.

    Sign conventions are already encoded in the dual variable bounds produced by
    _add_dual_variables(), so raw A coefficients are used without adjustment.
    Variables with no constraint connections (e.g. unconstrained free variables)
    are skipped; the dual is infeasible for such problems if ``c_j != 0``.

    Parameters
    ----------
    m : Model
        Primal model.
    m2 : Model
        Dual model to populate.
    dual_vars : dict
        ``{constraint_name: dual_variable}`` as returned by _add_dual_variables().
    """
    A = m.matrices.A
    if A is None:
        raise ValueError("Constraint matrix is None, model has no constraints.")

    vlabels = np.asarray(m.matrices.vlabels, dtype=np.int64)
    clabels = np.asarray(m.matrices.clabels, dtype=np.int64)

    flat_con_to_dual = _build_flat_con_to_dual_label_lookup(m, dual_vars)
    if not len(flat_con_to_dual):
        logger.warning(
            "No valid constraint labels found, skipping dual feasibility constraints."
        )
        return

    flat_v, flat_d, nnz_data = _extract_dual_feas_entries(
        A, vlabels, clabels, flat_con_to_dual
    )
    c_lookup = _build_obj_coeff_lookup(vlabels, m.matrices.c)

    logger.debug("Building dual feasibility constraints for each primal variable.")
    for var_name, var in m.variables.items():
        if _skip(var.labels, "variable", var_name):
            continue

        mask = var.labels != -1
        var_flat = var.labels.values.ravel().astype(np.int64)

        # RHS: objective coefficient for each element of this variable.
        in_c_range = (var_flat != -1) & (var_flat < len(c_lookup))
        safe_flat = np.where(in_c_range, var_flat, 0)
        c_vals = xr.DataArray(
            np.where(in_c_range, c_lookup[safe_flat], 0.0).reshape(var.labels.shape),
            coords=var.labels.coords,
        )

        lhs = _build_dual_feas_lhs(var, flat_v, flat_d, nnz_data, m2)
        if lhs.is_constant:
            # Variable has no constraint connections (free, unconstrained variable).
            # The stationarity condition 0 = c_j cannot be expressed as a linopy
            # constraint; the dual is infeasible if c_j != 0, trivial if c_j == 0.
            logger.debug(
                f"Variable '{var_name}' has no constraint connections; "
                "skipping dual feasibility constraint."
            )
            continue
        m2.add_constraints(lhs == c_vals, name=var_name, mask=mask)


def _add_dual_objective(
    m: Model,
    m2: Model,
    dual_vars: dict,
    add_objective_constant: float = 0.0,
) -> None:
    """
    Construct and add ``sum(rhs_i * lambda_i)`` as the dual objective of m2.

    The uniform ``+`` sign is correct because sign conventions are already
    encoded in the dual variable bounds: a ``<=`` constraint has ``lambda <= 0``,
    so ``rhs * lambda`` contributes negatively without an explicit sign flip.
    This aligns with linopy's native dual convention, so
    ``m2.variables[con_name].solution`` can be compared directly with
    ``m.constraints[con_name].dual`` after solving.

    The objective sense is flipped: ``min`` primal → ``max`` dual, and vice versa.

    Parameters
    ----------
    m : Model
        Primal model.
    m2 : Model
        Dual model to populate.
    dual_vars : dict
        ``{constraint_name: dual_variable}`` as returned by _add_dual_variables().
    add_objective_constant : float, optional
        Constant added to the dual objective, e.g. to pass through a primal
        objective constant that was excluded from the model.
    """
    dual_obj: LinearExpression = LinearExpression(None, m2)
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
    m: Model,
    add_objective_constant: float = 0.0,
) -> Model:
    """
    Construct the dual of a linopy LP model.

    Transforms the primal model into its dual equivalent m2 following
    standard LP duality theory. The dual sense is flipped relative to the
    primal (min -> max, max -> min), and dual variable bounds depend on
    both constraint type and primal objective sense.

    For a minimization primal:

        Primal (min):                       Dual (max):
        min  c^T x                          max  b_eq^T λ + b_leq^T μ + b_geq^T ν
        s.t.    A_eq x  =  b_eq   : λ free  s.t.    A_eq^T λ + A_leq^T μ + A_geq^T ν = c
                A_leq x <= b_leq  : μ <= 0          λ free, μ <= 0, ν >= 0
                A_geq x >= b_geq  : ν >= 0

    For a maximization primal the dual variable bounds are flipped:
    μ >= 0 for <= constraints, ν <= 0 for >= constraints.

    Variable bounds are converted to explicit constraints before dualization
    via _lift_bounds_to_constraints(), so that they appear in the constraint matrix
    A and are correctly reflected in the dual.

    The dual variables in m2 are named identically to their corresponding
    primal constraints and are accessible via m2.variables[con_name].

    Strong duality guarantees that at optimality:
        primal objective = dual objective

    Note: The standalone dual m2 may be unbounded if the primal is degenerate.
    Only linear programs (LP) are supported.

    Parameters
    ----------
    m : Model
        Primal linopy model to dualize. Must have a linear objective and linear constraints.

    add_objective_constant : float, optional
        Constant added to the dual objective, e.g. to pass through a primal
        objective constant that was excluded from the model.

    Returns
    -------
    Model
        Dual model whose variables are named after the primal constraints.

    Examples
    --------
    .. code-block:: python

        m2 = m.dualize()
        m2.solve(solver_name="gurobi", Method=2, Crossover=1)
        gap = abs(m.objective.value - m2.objective.value)
    """
    from linopy.model import Model

    m1 = m.copy()
    m2 = Model()

    if not m.variables or not m.constraints:
        logger.warning(
            "Primal model has no variables or constraints. Returning empty dual model."
        )
        return m2

    _lift_bounds_to_constraints(m1)
    dual_vars = _add_dual_variables(m1, m2)
    _add_dual_feasibility_constraints(m1, m2, dual_vars)
    _add_dual_objective(
        m1, m2, dual_vars, add_objective_constant=add_objective_constant
    )
    return m2
