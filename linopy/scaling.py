"""
Utilities for scaling linear problems before handing them to a solver.

The scaling performed here is intentionally conservative: it supports
row (constraint) scaling for all problem types and optional column
(variable) scaling for continuous variables. Scaling is applied to the
data structures used by the solver interfaces (`matrices`), leaving the
original model untouched. Primal and dual values can be mapped back to
the unscaled space via the provided helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.sparse import csc_matrix, diags

from linopy.common import set_int_index
from linopy.constants import Result, Solution
from linopy.matrices import MatrixAccessor

ScaleMethod = Literal["row-max", "row-l2"]


@dataclass
class ScaleOptions:
    """
    Configuration for model scaling.

    Attributes
    ----------
    enabled : bool
        Whether scaling is enabled.
    method : {"row-max", "row-l2"}
        Norm used to calculate row/column magnitudes.
    variable_scaling : bool
        Enable column/variable scaling in addition to row scaling.
    scale_integer_variables : bool
        Whether integer/binary variables are scaled. Default False keeps
        integrality-safe scaling by leaving them untouched.
    scale_bounds : bool
        Whether bounds are scaled along with variables. Required for
        consistent variable scaling; ignored when `variable_scaling` is
        False.
    target : float
        Target magnitude per row/column after scaling.
    zero_floor : float
        Lower bound for norms to avoid division by (near) zero.
    """

    enabled: bool = True
    method: ScaleMethod = "row-max"
    variable_scaling: bool = False
    scale_integer_variables: bool = False
    scale_bounds: bool = True
    target: float = 1.0
    zero_floor: float = 1e-12


@dataclass
class ScaledMatrices:
    """
    Container for scaled matrix data passed to solver frontends.
    """

    A: csc_matrix | None
    b: ndarray
    c: ndarray
    lb: ndarray
    ub: ndarray
    sense: ndarray
    vtypes: ndarray
    vlabels: ndarray
    clabels: ndarray
    Q: csc_matrix | None


def _row_norms(A: csc_matrix, method: ScaleMethod) -> ndarray:
    """
    Compute per-row magnitudes for a sparse matrix.
    """
    A_csr = A.tocsr()
    if method == "row-l2":
        norms = np.sqrt(np.array(A_csr.power(2).mean(axis=1)).ravel(), dtype=float)
    else:
        A_abs = A_csr.copy()
        A_abs.data = np.abs(A_abs.data)
        norms = np.array(A_abs.max(axis=1).toarray()).ravel().astype(float)

    # rows without entries yield 0 or nan; keep them unscaled
    norms = np.where(np.isnan(norms) | (norms == 0), 1.0, norms)
    return norms


def _col_norms(A: csc_matrix, method: ScaleMethod) -> ndarray:
    """
    Compute per-column magnitudes for a sparse matrix.
    """
    A_csc = A.tocsc()
    if method == "row-l2":
        norms = np.sqrt(np.array(A_csc.power(2).mean(axis=0)).ravel(), dtype=float)
    else:
        A_abs = A_csc.copy()
        A_abs.data = np.abs(A_abs.data)
        norms = np.array(A_abs.max(axis=0).toarray()).ravel().astype(float)

    norms = np.where(np.isnan(norms) | (norms == 0), 1.0, norms)
    return norms


def _safe_scale(norms: ndarray, target: float, floor: float) -> ndarray:
    norms = np.maximum(norms, floor)
    return target / norms


def resolve_options(scale: bool | str | ScaleOptions | None) -> ScaleOptions:
    """
    Normalize scale input into a ScaleOptions instance.
    """
    if scale is None or scale is False:
        return ScaleOptions(enabled=False)
    if scale is True:
        return ScaleOptions()
    if isinstance(scale, str):
        if scale not in ("row-max", "row-l2"):
            msg = f"Invalid scale method: {scale!r}. Must be 'row-max' or 'row-l2'."
            raise ValueError(msg)
        return ScaleOptions(method=scale)  # type: ignore[arg-type]
    if isinstance(scale, ScaleOptions):
        return scale

    msg = "scale must be one of None, bool, str {'row-max','row-l2'} or ScaleOptions."
    raise ValueError(msg)


class ScalingContext:
    """
    Compute and apply scaling factors for a model's matrices.
    """

    def __init__(
        self,
        options: ScaleOptions,
        row_scale: ndarray,
        col_scale: ndarray,
        matrices: ScaledMatrices,
    ) -> None:
        self.options = options
        self.row_scale = row_scale
        self.col_scale = col_scale
        self.col_inv = np.reciprocal(col_scale)
        self.matrices = matrices

    @classmethod
    def from_model(
        cls,
        matrices: MatrixAccessor,
        options: ScaleOptions,
    ) -> ScalingContext:
        """
        Build a scaling context from a model's matrix accessor.
        """
        if not options.enabled:
            raise ValueError("ScalingContext.from_model called with scaling disabled.")

        A = matrices.A

        if A is None:
            row_scale = np.ones_like(matrices.clabels, dtype=float)
        else:
            row_norm = _row_norms(A, options.method)
            row_scale = _safe_scale(row_norm, options.target, options.zero_floor)

        b = matrices.b * row_scale if len(matrices.b) else matrices.b

        Q = matrices.Q
        if options.variable_scaling and not options.scale_bounds:
            raise ValueError(
                "Variable scaling requires scaling bounds for consistency. "
                "Set `scale_bounds=True` or disable `variable_scaling`."
            )

        if options.variable_scaling and A is not None:
            A_rows = diags(row_scale) @ A
            col_norm = _col_norms(A_rows, options.method)
            col_scale = _safe_scale(col_norm, options.target, options.zero_floor)
        else:
            col_scale = np.ones_like(matrices.vlabels, dtype=float)

        if not options.scale_integer_variables:
            vtypes = matrices.vtypes
            mask_int = (vtypes == "I") | (vtypes == "B")
            col_scale[mask_int] = 1.0

        col_inv = np.reciprocal(np.maximum(col_scale, options.zero_floor))

        if A is not None:
            A_scaled = diags(row_scale) @ A @ diags(col_inv)
        else:
            A_scaled = None

        c = matrices.c * col_inv
        lb = matrices.lb.copy()
        ub = matrices.ub.copy()
        if options.variable_scaling and options.scale_bounds:
            lb = lb * col_scale
            ub = ub * col_scale

        if Q is not None:
            D_inv = diags(col_inv)
            Q = D_inv @ Q @ D_inv

        scaled = ScaledMatrices(
            A=A_scaled,
            b=b,
            c=c,
            lb=lb,
            ub=ub,
            sense=matrices.sense,
            vtypes=matrices.vtypes,
            vlabels=matrices.vlabels,
            clabels=matrices.clabels,
            Q=Q,
        )
        return cls(
            options=options, row_scale=row_scale, col_scale=col_scale, matrices=scaled
        )

    def unscale_primal(self, primal: pd.Series) -> pd.Series:
        """
        Map primal values from scaled to original space.
        """
        primal = set_int_index(primal)
        factors = pd.Series(self.col_scale, index=self.matrices.vlabels)
        return primal.div(factors, fill_value=np.nan)

    def unscale_dual(self, dual: pd.Series) -> pd.Series:
        """
        Map dual values from scaled to original space.
        """
        dual = set_int_index(dual)
        factors = pd.Series(self.row_scale, index=self.matrices.clabels)
        return dual.mul(factors, fill_value=np.nan)

    def unscale_solution(self, result: Result | None) -> Result | None:
        """
        Apply unscaling to primal and dual values stored in a solver result.
        """
        if result is None or result.solution is None:
            return result

        sol: Solution = result.solution
        if not sol.primal.empty:
            sol.primal = self.unscale_primal(sol.primal)
        if not sol.dual.empty:
            sol.dual = self.unscale_dual(sol.dual)
        result.solution = sol
        return result
