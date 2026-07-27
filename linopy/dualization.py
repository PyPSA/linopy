"""
Linopy dualization module.

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

    Bounds may vary elementwise within one variable block. Infinite bounds are
    not converted into constraints.

    Parameters
    ----------
    m : Model
        Model to mutate in-place.
    """
    for var_name, var in m.variables.items():
        label_mask = var.labels != -1
        lb = var.lower
        ub = var.upper

        finite_lb = xr.DataArray(np.isfinite(lb.values), coords=lb.coords, dims=lb.dims)
        finite_ub = xr.DataArray(np.isfinite(ub.values), coords=ub.coords, dims=ub.dims)

        bound_specs = [
            ("lower", var >= lb, label_mask & finite_lb, var.lower, -np.inf),
            ("upper", var <= ub, label_mask & finite_ub, var.upper, np.inf),
        ]
        for suffix, con, bound_mask, bound, relaxed in bound_specs:
            con_name = f"{var_name}-bound-{suffix}"
            if con_name in m.constraints or not bool(bound_mask.any()):
                continue
            m.add_constraints(con, name=con_name, mask=bound_mask)
            bound.values[bound_mask.values] = relaxed


def _dual_bounds_from_constraint_signs(
    signs: xr.DataArray,
    labels: xr.DataArray,
    primal_is_min: bool,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Return elementwise dual-variable bounds for constraint signs.

    ``signs`` is broadcast to ``labels``. Valid signs are ``=``, ``<=``, and
    ``>=``. ``valid_sign`` is True where the broadcast sign is valid.
    """
    signs = signs.broadcast_like(labels)

    is_eq = signs == "="
    is_le = signs == "<="
    is_ge = signs == ">="
    valid_sign = is_eq | is_le | is_ge

    lower = xr.zeros_like(labels, dtype=float)
    upper = xr.zeros_like(labels, dtype=float)

    lower = lower.where(~is_eq, -np.inf)
    upper = upper.where(~is_eq, np.inf)

    if primal_is_min:
        lower = lower.where(~is_le, -np.inf)
        upper = upper.where(~is_le, 0.0)

        lower = lower.where(~is_ge, 0.0)
        upper = upper.where(~is_ge, np.inf)
    else:
        lower = lower.where(~is_le, 0.0)
        upper = upper.where(~is_le, np.inf)

        lower = lower.where(~is_ge, -np.inf)
        upper = upper.where(~is_ge, 0.0)

    return lower, upper, valid_sign


def _add_dual_variables(m: Model, m_dual: Model) -> dict:
    """
    Add one dual variable to m_dual for each included constraint in m.

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

    Fully masked or empty constraints are skipped. Constraint arrays may contain
    mixed signs; dual variable bounds are assigned elementwise.

    Parameters
    ----------
    m : Model
        Primal model.
    m_dual : Model
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

        lower, upper, valid_sign = _dual_bounds_from_constraint_signs(
            con.sign,
            con.labels,
            primal_is_min,
        )

        unmasked = con.labels != -1
        invalid_unmasked = unmasked & ~valid_sign
        if bool(invalid_unmasked.any()):
            logger.warning(
                f"Constraint '{name}' has unrecognized signs; invalid entries "
                "will be skipped."
            )

        mask = unmasked & valid_sign
        if not bool(mask.any()):
            logger.warning(
                f"Constraint '{name}' has no entries with valid signs, skipping."
            )
            continue

        logger.debug(
            f"Adding dual variable for constraint '{name}' "
            f"with shape {con.shape} and dims {con.labels.dims}."
        )
        coords = (
            [con.indexes[dim] for dim in con.labels.dims] if con.coord_dims else None
        )
        dual_vars[name] = m_dual.add_variables(
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
    flat dual variable label from ``dual_vars``. Entries are -1 for constraints
    not in ``dual_vars`` or with masked labels.

    Parameters
    ----------
    m : Model
        Primal model.
    dual_vars : dict
        ``{constraint_name: dual_variable}`` as returned by ``_add_dual_variables()``.

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
    Return ``(flat_var_label, flat_dual_label, coeff)`` for each included
    nonzero entry of ``A``.

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

    has_labels = (flat_v != -1) & (flat_c != -1)
    flat_v, flat_c, coeffs = flat_v[has_labels], flat_c[has_labels], coeffs[has_labels]

    flat_d = _gather_with_default(flat_c, flat_con_to_dual, -1, np.int64)

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


def _build_label_to_flat_index_lookup(labels: np.ndarray) -> np.ndarray:
    """
    Build a lookup from flat labels to flat indices.

    ``labels`` is expected to contain nonnegative integer labels, with -1 used
    as the masked-label sentinel.

    ``lookup[label]`` gives the flat index of ``label`` in ``labels``.
    Labels that do not occur in ``labels`` map to -1. If no valid labels exist,
    returns an empty int64 array.
    """
    valid = labels != -1
    if not valid.any():
        return np.array([], dtype=np.int64)

    lookup = np.full(int(labels[valid].max()) + 1, -1, dtype=np.int64)
    lookup[labels[valid].astype(np.int64)] = np.where(valid)[0].astype(np.int64)
    return lookup


def _gather_with_default(
    labels: np.ndarray, lookup: np.ndarray, default: float, dtype: type
) -> np.ndarray:
    """
    Gather ``lookup[labels]`` elementwise, using ``default`` where ``labels`` fall
    outside ``lookup``'s range (including the -1 masked sentinel).
    """
    out: np.ndarray = np.full(len(labels), default, dtype=dtype)
    in_range = (labels >= 0) & (labels < len(lookup))
    if in_range.any():
        out[in_range] = lookup[labels[in_range].astype(np.int64)]
    return out


def _lookup_flat_indices(labels: np.ndarray, lookup: np.ndarray) -> np.ndarray:
    """
    Look up flat indices for flat labels.

    ``labels`` is expected to contain nonnegative integer labels, with -1 used
    as the masked-label sentinel. ``lookup`` is expected to be an array as
    returned by ``_build_label_to_flat_index_lookup()``.

    Labels outside the lookup range, including masked labels, map to -1.
    """
    return _gather_with_default(labels, lookup, -1, np.int64)


def _term_slots_for_sorted_flat_indices(sorted_flat_indices: np.ndarray) -> np.ndarray:
    """
    Return the term slot within each run of equal sorted flat indices.

    ``sorted_flat_indices`` must be non-empty and sorted in nondecreasing order.
    Each run of equal values is treated as one group.

    Example:
    -------
    ``[2, 2, 2, 5, 5, 9]`` becomes ``[0, 1, 2, 0, 1, 0]``.
    """
    group_start = np.empty(len(sorted_flat_indices), dtype=bool)
    group_start[0] = True
    group_start[1:] = sorted_flat_indices[1:] != sorted_flat_indices[:-1]

    group_ids = np.cumsum(group_start) - 1
    group_start_pos = np.where(group_start)[0]

    return (
        np.arange(len(sorted_flat_indices), dtype=np.int64) - group_start_pos[group_ids]
    )


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
        expression, equivalent to constant zero, when ``var`` has no included
        constraint-matrix entries.
    """
    var_flat = var.labels.values.ravel().astype(np.int64)
    n_elements = len(var_flat)

    var_to_idx = _build_label_to_flat_index_lookup(var_flat)
    if not len(var_to_idx):
        return LinearExpression(None, m_dual)

    lin_idx = _lookup_flat_indices(flat_v, var_to_idx)

    keep = lin_idx != -1
    if not keep.any():
        return LinearExpression(None, m_dual)

    lin_idx = lin_idx[keep]
    dual_labels = flat_d[keep]
    coeffs = nnz_data[keep]

    order = np.argsort(lin_idx, kind="stable")
    lin_idx = lin_idx[order]
    dual_labels = dual_labels[order]
    coeffs = coeffs[order]

    col_idx = _term_slots_for_sorted_flat_indices(lin_idx)
    max_terms = int(col_idx.max()) + 1

    dual_labels_2d = np.full((n_elements, max_terms), -1, dtype=np.int64)
    dual_coeffs_2d = np.zeros((n_elements, max_terms), dtype=np.float64)
    dual_labels_2d[lin_idx, col_idx] = dual_labels
    dual_coeffs_2d[lin_idx, col_idx] = coeffs

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
    m_dual: Model,
    dual_vars: dict,
) -> None:
    """
    Add the stationarity constraint ``sum_i(A_ji * lambda_i) = c_j`` for each
    primal variable.

    Sign conventions are already encoded in the dual variable bounds produced by
    ``_add_dual_variables()``, so raw A coefficients are used without adjustment.
    Variables with no constraint connections (e.g. unconstrained free variables)
    are skipped; the dual is infeasible for such problems if ``c_j != 0``.

    Parameters
    ----------
    m : Model
        Primal model.
    m_dual : Model
        Dual model to populate.
    dual_vars : dict
        ``{constraint_name: dual_variable}`` as returned by ``_add_dual_variables()``.
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

        c_arr = _gather_with_default(var_flat, c_lookup, 0.0, np.float64)
        c_vals = xr.DataArray(
            c_arr.reshape(var.labels.shape),
            coords=var.labels.coords,
            dims=var.labels.dims,
        )

        lhs = _build_dual_feas_lhs(var, flat_v, flat_d, nnz_data, m_dual)

        if lhs.is_constant:
            if np.any(c_arr[mask.values.ravel()] != 0):
                logger.warning(
                    f"Variable '{var_name}' has no constraint connections but has "
                    "nonzero objective coefficients; the corresponding dual-feasibility "
                    "condition is infeasible."
                )
            else:
                logger.debug(
                    f"Variable '{var_name}' has no constraint connections and zero "
                    "objective coefficients; skipping redundant dual feasibility constraint."
                )
            continue

        m_dual.add_constraints(lhs == c_vals, name=var_name, mask=mask)


def _add_dual_objective(
    m: Model,
    m_dual: Model,
    dual_vars: dict,
) -> None:
    """
    Construct and add ``sum(rhs_i * lambda_i)`` as the dual objective of m_dual.

    The uniform ``+`` sign is correct because sign conventions are already
    encoded in the dual variable bounds: a ``<=`` constraint has ``lambda <= 0``,
    so ``rhs * lambda`` contributes negatively without an explicit sign flip.
    This aligns with linopy's native dual convention, so
    ``m_dual.variables[con_name].solution`` can be compared directly with
    ``m.constraints[con_name].dual`` after solving.

    The objective sense is flipped: ``min`` primal → ``max`` dual, and vice versa.

    Parameters
    ----------
    m : Model
        Primal model.
    m_dual : Model
        Dual model to populate.
    dual_vars : dict
        ``{constraint_name: dual_variable}`` as returned by ``_add_dual_variables()``.
    """
    dual_obj: LinearExpression = LinearExpression(None, m_dual)
    sense = "max" if m.objective.sense == "min" else "min"

    for name, con in m.constraints.items():
        if name not in dual_vars:
            continue

        mask = con.labels != -1
        rhs_masked = con.rhs.where(mask, 0)
        dual_obj += (rhs_masked * dual_vars[name]).sum()

    logger.debug(f"Constructed dual objective with {len(dual_obj.coeffs)} terms.")
    logger.debug("Adding dual objective to model.")
    m_dual.add_objective(dual_obj, sense=sense, overwrite=True)


def dualize(
    m: Model,
) -> Model:
    """
    Construct the dual of a linopy LP model.

    Transforms the primal model into its dual equivalent m_dual following
    standard LP duality theory. The dual sense is flipped relative to the
    primal (min -> max, max -> min), and dual variable bounds depend on
    both constraint type and primal objective sense.

    For a minimization primal:

        Primal (min):
        min  c^T x
        s.t.    A_eq x  =  b_eq   : λ free
                A_leq x <= b_leq  : μ <= 0
                A_geq x >= b_geq  : ν >= 0

        Dual (max):
        max  b_eq^T λ + b_leq^T μ + b_geq^T ν
        s.t.    A_eq^T λ + A_leq^T μ + A_geq^T ν = c
                λ free, μ <= 0, ν >= 0

    For a maximization primal the dual variable bounds are flipped:
    μ >= 0 for <= constraints, ν <= 0 for >= constraints.

    The corresponding bound conventions are:

    ============  ===========  ================  ================
    Constraint    Primal sense  lower             upper
    ============  ===========  ================  ================
    =             min / max     -inf              +inf  (free)
    <=            min           -inf              0
    <=            max           0                 +inf
    >=            min           0                 +inf
    >=            max           -inf              0
    ============  ===========  ================  ================

    Variable bounds are converted to explicit constraints before dualization
    via _lift_bounds_to_constraints(), so that they appear in the constraint matrix
    A and are correctly reflected in the dual.

    The dual variables in m_dual are named identically to their corresponding
    primal constraints and are accessible via m_dual.variables[con_name].

    Strong duality guarantees that at optimality:
        primal objective = dual objective

    Note: This constructs a standalone dual model. Pathological or unsupported
    primal formulations may lead to infeasible or unbounded dual models or may
    cause this function to raise an error. Only linear primal models with linear
    objectives and linear constraints are supported; finite variable bounds are
    lifted into explicit constraints before dualization.

    Parameters
    ----------
    m : Model
        Primal linopy model to dualize. Must have a linear objective and either
        linear constraints or finite variable bounds.

    Returns
    -------
    Model
        Dual model whose variables are named after the primal constraints and
        constraints are named after the primal variables.

    Example
    -------
    .. code-block:: python

        m_dual = m.dualize()
        m_dual.solve(solver_name="gurobi", Method=2, Crossover=1)
        gap = abs(m.objective.value - m_dual.objective.value)
    """
    from linopy.model import Model

    m1 = m.copy()
    m_dual = Model()

    if not m1.variables:
        logger.warning("Primal model has no variables. Returning empty dual model.")
        return m_dual

    _lift_bounds_to_constraints(m1)

    if not m1.constraints:
        logger.warning(
            "Primal model has no constraints after lifting variable bounds. "
            "Returning empty dual model."
        )
        return m_dual

    dual_vars = _add_dual_variables(m1, m_dual)
    _add_dual_feasibility_constraints(m1, m_dual, dual_vars)
    _add_dual_objective(m1, m_dual, dual_vars)
    return m_dual
