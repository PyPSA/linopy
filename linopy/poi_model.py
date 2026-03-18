"""
PoiModel: A wrapper that holds a pyoptinterface model alongside a linopy model,
with mappings between the two, and supports incremental updates via model diffing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from linopy.model import Model


@dataclass
class PoiModel:
    """
    Wraps a pyoptinterface model built from a linopy model.

    Attributes
    ----------
    poi : Any
        The pyoptinterface model instance.
    linopy_model : Model
        The linopy model used to build the POI model.
    vars_to_poi : np.ndarray
        Array of shape (num_vars+1,) mapping linopy variable labels to POI variable indices.
    cons_to_poi : np.ndarray
        Array of shape (num_constrs+1,) mapping linopy constraint labels to POI constraint indices.
    """

    poi: Any
    linopy_model: Model
    vars_to_poi: np.ndarray = field(repr=False)
    cons_to_poi: np.ndarray = field(repr=False)

    @classmethod
    def from_linopy(
        cls, m: Model, poi_model: Any, slice_size: int = 2_000_000
    ) -> PoiModel:
        """
        Build a PoiModel from a linopy model and an existing pyoptinterface model instance.

        Parameters
        ----------
        m : linopy.Model
            The linopy model to transfer.
        poi_model : Any
            An already-constructed POI model (e.g. ``pyoptinterface.highs.Model()``).
        slice_size : int, optional
            Number of constraint rows to process at once.

        Returns
        -------
        PoiModel
        """
        from linopy.io import to_poi

        vars_to_poi, cons_to_poi = to_poi(m, poi_model, slice_size=slice_size)
        return cls(
            poi=poi_model,
            linopy_model=m,
            vars_to_poi=vars_to_poi,
            cons_to_poi=cons_to_poi,
        )

    def update_linopy(self, new_m: Model) -> None:
        """
        Update the POI model in-place to reflect a new linopy model.

        Uses pyoptinterface's ``set_normalized_rhs`` and ``set_normalized_coeff``
        to apply only the differences between the stored linopy model and ``new_m``.

        Assumptions (asserted):
        - Same set of constraint names.
        - For each constraint, labels, variable references, and signs are identical.
        - Variable labels and names are identical.

        Parameters
        ----------
        new_m : linopy.Model
            The updated linopy model. Must have the same structure as the original.
        """
        import pyoptinterface as poi

        def make_con(idx: int) -> poi.ConstraintIndex:
            return poi.ConstraintIndex(poi.ConstraintType.Linear, idx)

        old_m = self.linopy_model

        # --- Validate variable structure ---
        old_vars = set(old_m.variables)
        new_vars = set(new_m.variables)
        assert old_vars == new_vars, (
            f"Variable sets differ. Added: {new_vars - old_vars}, "
            f"Removed: {old_vars - new_vars}"
        )

        # --- Validate constraint structure ---
        old_cons = set(old_m.constraints)
        new_cons = set(new_m.constraints)
        assert old_cons == new_cons, (
            f"Constraint sets differ. Added: {new_cons - old_cons}, "
            f"Removed: {old_cons - new_cons}"
        )

        vars_to_poi = self.vars_to_poi
        cons_to_poi = self.cons_to_poi

        # --- Update variable bounds ---
        for vn in old_m.variables:
            d1 = old_m.variables[vn].data
            d2 = new_m.variables[vn].data

            assert (d1["labels"].values == d2["labels"].values).all(), (
                f"Variable '{vn}': labels differ"
            )

            lower_diff = np.ravel(d1["lower"].values != d2["lower"].values)
            upper_diff = np.ravel(d1["upper"].values != d2["upper"].values)
            both_diff = lower_diff & upper_diff
            only_lower = lower_diff & ~upper_diff
            only_upper = upper_diff & ~lower_diff

            labels = np.ravel(d2["labels"].values)
            lower = np.ravel(d2["lower"].fillna(-np.inf).values)
            upper = np.ravel(d2["upper"].fillna(np.inf).values)

            print(both_diff.sum(), only_lower.sum(), only_upper.sum())

            for var_idx, lb, ub in zip(
                vars_to_poi[labels[both_diff]],
                lower[both_diff],
                upper[both_diff],
            ):
                self.poi.set_variable_bounds(poi.VariableIndex(var_idx), lb, ub)
            for var_idx, lb in zip(
                vars_to_poi[labels[only_lower]],
                lower[only_lower],
            ):
                self.poi.set_variable_lower_bound(poi.VariableIndex(var_idx), lb)
            for var_idx, ub in zip(
                vars_to_poi[labels[only_upper]],
                upper[only_upper],
            ):
                self.poi.set_variable_upper_bound(poi.VariableIndex(var_idx), ub)

        # --- Update constraints ---
        for cn in old_m.constraints:
            c1 = old_m.constraints[cn].data
            c2 = new_m.constraints[cn].data

            c1_labels = np.ravel(c1["labels"])
            c2_labels = np.ravel(c2["labels"])
            c1_vars = np.ravel(c1["vars"])
            c2_vars = np.ravel(c2["vars"])
            c1_sign = np.ravel(c1["sign"])
            c2_sign = np.ravel(c2["sign"])

            assert (c1_labels == c2_labels).all(), f"Constraint '{cn}': labels differ"
            assert (c1_vars == c2_vars).all(), (
                f"Constraint '{cn}': variable references differ"
            )
            assert (c1_sign == c2_sign).all(), f"Constraint '{cn}': signs differ"

            # Update RHS where changed
            c1_rhs = np.ravel(c1["rhs"])
            c2_rhs = np.ravel(c2["rhs"])
            rhs_diff = c1_rhs != c2_rhs
            if rhs_diff.any():
                poi_cons = cons_to_poi[c2_labels[rhs_diff]]
                new_rhs = c2_rhs[rhs_diff]
                for con_idx, rhs_val in zip(poi_cons, new_rhs):
                    self.poi.set_normalized_rhs(make_con(con_idx), rhs_val)

            # Update coefficients where changed
            c1_coeffs = np.ravel(c1["coeffs"])
            c2_coeffs = np.ravel(c2["coeffs"])
            # Treat NaN == NaN as equal (no change)
            both_nan = np.isnan(c1_coeffs) & np.isnan(c2_coeffs)
            coeffs_diff = (c1_coeffs != c2_coeffs) & ~both_nan
            if coeffs_diff.any():
                broadcast_labels = np.ravel(
                    np.broadcast_to(
                        c1["labels"].values[..., np.newaxis], c1["coeffs"].shape
                    )
                )
                changed_labels = broadcast_labels[coeffs_diff]
                changed_vars = c1_vars[coeffs_diff]
                changed_coeffs = c2_coeffs[coeffs_diff]

                # Group by (con, var) and sum coefficients
                updates = (
                    pl.DataFrame(
                        {
                            "con": cons_to_poi[changed_labels],
                            "var": vars_to_poi[changed_vars],
                            "coeff": changed_coeffs,
                        }
                    )
                    .group_by("con", "var")
                    .agg(pl.col("coeff").sum())
                )

                for con_idx, var_idx, coeff_val in updates.iter_rows():
                    self.poi.set_normalized_coefficient(
                        make_con(con_idx),
                        poi.VariableIndex(var_idx),
                        coeff_val,
                    )

        # Update the stored linopy model reference
        self.linopy_model = new_m
