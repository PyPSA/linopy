"""
Regression coverage for SOS constraints on masked variables (#688).

The bug being pinned here has two related failure modes:

1. **Position-vs-label**: direct-API builds (gurobi, xpress) pass linopy variable
   labels straight to vendor ``addSOS`` as if they were 0-based column positions
   in the active-variable array. They only happen to coincide when no variable
   in the model is masked anywhere.

2. **LP file emits ``x-1``**: the LP writer iterates raw label arrays and emits
   names like ``x-1`` for masked SOS entries, which LP parsers either reject
   outright or (gurobi LP reader) silently corrupt into wrong SOS sets.

The fixture asymmetric-coefficient design plus three-layer oracle (status,
objective, element-wise solution) ensures any wrong indexing surfaces as a
visible failure rather than a permutation-equivalent silent pass.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model, available_solvers
from linopy.solver_capabilities import SolverFeature, solver_supports

# ---------------------------------------------------------------------------
# Capability-derived solver / io_api parametrization
# ---------------------------------------------------------------------------

SOS_DIRECT = sorted(
    s
    for s in available_solvers
    if solver_supports(s, SolverFeature.SOS_CONSTRAINTS)
    and solver_supports(s, SolverFeature.DIRECT_API)
)
SOS_FILE = sorted(
    s for s in available_solvers if solver_supports(s, SolverFeature.SOS_CONSTRAINTS)
)
SOS_PATHS = [
    *[pytest.param(s, "direct", id=f"{s}-direct") for s in SOS_DIRECT],
    *[pytest.param(s, "lp", id=f"{s}-lp") for s in SOS_FILE],
]

# ---------------------------------------------------------------------------
# Analytical optimum (matches solver semantics: list-position adjacency for SOS2)
# ---------------------------------------------------------------------------


def _optimize_sos_set(
    active_i: list[int], coefs: dict[int, float], sos_type: int
) -> tuple[float, dict[int, float]]:
    """
    Closed-form optimum for one SOS set with binary [0,1] members.

    ``active_i`` is the sorted list of active (unmasked) member indices in the
    SOS dimension. ``coefs`` maps each active index to its objective coefficient
    (minimization). For SOS2, adjacency is list-position adjacency, matching the
    semantics of gurobi/xpress ``addSOS``.
    """
    if not active_i:
        return 0.0, {}

    best_obj = 0.0
    best_sol: dict[int, float] = {}

    # singletons
    for i in active_i:
        if coefs[i] < best_obj:
            best_obj = coefs[i]
            best_sol = {i: 1.0}

    if sos_type == 2:
        # adjacent pairs in the (sorted-by-weight) list
        for k in range(len(active_i) - 1):
            i1, i2 = active_i[k], active_i[k + 1]
            pair_obj = coefs[i1] + coefs[i2]
            if pair_obj < best_obj:
                best_obj = pair_obj
                best_sol = {i1: 1.0, i2: 1.0}

    return best_obj, best_sol


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

MaskOnSos = Literal[None, "sos_dim", "non_sos_dim", "both_dims"]


@pytest.fixture
def sos_masked_model() -> Callable[..., tuple[Model, float, np.ndarray]]:  # noqa: E501
    """
    Factory producing SOS{1,2} models with controllable mask placement.

    Objective coefficients along the SOS dim are ``[-1, -2, -3, -4]``,
    asymmetric to break permutation symmetry — wrong indexing then produces an
    observably different objective AND solution.

    Returns ``(model, expected_obj, expected_sol)``. ``expected_sol`` is shaped
    like ``sos_var.solution`` (with ``NaN`` where the mask removes a slot).
    """

    def _build(
        sos_type: Literal[1, 2] = 1,
        sos_var_2d: bool = False,
        mask_on_sos: MaskOnSos = None,
        mask_on_other: bool = False,
    ) -> tuple[Model, float, np.ndarray]:
        if not sos_var_2d and mask_on_sos in ("non_sos_dim", "both_dims"):
            raise ValueError(f"mask_on_sos={mask_on_sos!r} requires sos_var_2d=True")

        m = Model()

        # Optional unrelated masked variable: shifts label->position mapping
        # for all subsequent variables, exposing the position-vs-label bug.
        if mask_on_other:
            ck = pd.Index([0, 1, 2, 3], name="k")
            m.add_variables(
                lower=0,
                upper=1,
                coords=[ck],
                mask=pd.Series([False, True, True, True], index=ck),
                name="other",
            )

        ci = pd.Index([0, 1, 2, 3], name="i")
        cj = pd.Index([0, 1], name="j")

        # Construct sos_var mask
        if mask_on_sos is None:
            sos_mask = None
        elif mask_on_sos == "sos_dim":
            mask_i = np.array([True, True, False, True])
            if sos_var_2d:
                sos_mask = xr.DataArray(
                    np.broadcast_to(mask_i[:, None], (4, 2)).copy(),
                    coords=[ci, cj],
                    dims=["i", "j"],
                )
            else:
                sos_mask = pd.Series(mask_i, index=ci)
        elif mask_on_sos == "non_sos_dim":
            assert sos_var_2d
            mask_j = np.array([False, True])
            sos_mask = xr.DataArray(
                np.broadcast_to(mask_j[None, :], (4, 2)).copy(),
                coords=[ci, cj],
                dims=["i", "j"],
            )
        elif mask_on_sos == "both_dims":
            assert sos_var_2d
            mask_i = np.array([True, True, False, True])
            mask_j = np.array([False, True])
            combined = mask_i[:, None] & mask_j[None, :]
            sos_mask = xr.DataArray(combined, coords=[ci, cj], dims=["i", "j"])
        else:
            raise ValueError(f"unknown mask_on_sos={mask_on_sos!r}")

        sos_coords = [ci, cj] if sos_var_2d else [ci]
        sos_var = m.add_variables(
            lower=0,
            upper=1,
            coords=sos_coords,
            mask=sos_mask,
            name="sos_var",
        )
        m.add_sos_constraints(sos_var, sos_type=sos_type, sos_dim="i")

        # Asymmetric coefficients along the SOS dim; broadcast across j in 2D
        coefs_i = np.array([-1.0, -2.0, -3.0, -4.0])
        if sos_var_2d:
            coefs = xr.DataArray(
                np.broadcast_to(coefs_i[:, None], (4, 2)).copy(),
                coords=[ci, cj],
                dims=["i", "j"],
            )
        else:
            coefs = xr.DataArray(coefs_i, coords=[ci], dims=["i"])
        m.add_objective(sos_var * coefs)

        # ------------------------------------------------------------------
        # Compute expected_obj and expected_sol from the same mask logic
        # ------------------------------------------------------------------
        coefs_dict = {i: float(coefs_i[i]) for i in range(4)}

        # active_per_j[j] = sorted list of active i for SOS set at j (or for
        # the single 1D set we use j=None as a sentinel)
        if sos_var_2d:
            # Reconstruct the 2D mask (default to all True if none)
            if sos_mask is None:
                mask_arr = np.ones((4, 2), dtype=bool)
            else:
                mask_arr = np.asarray(sos_mask.values, dtype=bool)
            active_per_j: dict[int | None, list[int]] = {
                j: [i for i in range(4) if mask_arr[i, j]] for j in range(2)
            }
        else:
            if sos_mask is None:
                active = list(range(4))
            else:
                active = [i for i in range(4) if bool(sos_mask.iloc[i])]
            active_per_j = {None: active}

        expected_obj = 0.0
        # Build expected_sol with the right shape and NaN-fill masked slots
        if sos_var_2d:
            expected_sol: np.ndarray = np.full((4, 2), 0.0)
            if sos_mask is not None:
                mask_arr = np.asarray(sos_mask.values, dtype=bool)
                expected_sol[~mask_arr] = np.nan
        else:
            expected_sol = np.full(4, 0.0)
            if sos_mask is not None:
                for i in range(4):
                    if not bool(sos_mask.iloc[i]):
                        expected_sol[i] = np.nan

        for j_key, active in active_per_j.items():
            obj_j, sol_j = _optimize_sos_set(active, coefs_dict, sos_type)
            expected_obj += obj_j
            for i, value in sol_j.items():
                if sos_var_2d:
                    expected_sol[i, j_key] = value
                else:
                    expected_sol[i] = value

        return m, expected_obj, expected_sol

    return _build


# ---------------------------------------------------------------------------
# Test matrix: 11 fixture configs × 2 SOS types × (solver, io_api)
# ---------------------------------------------------------------------------

# Each entry: (sos_var_2d, mask_on_sos, mask_on_other)
FIXTURE_CONFIGS = [
    pytest.param(False, None, False, id="1d-no_mask"),
    pytest.param(False, "sos_dim", False, id="1d-mask_sos"),
    pytest.param(False, None, True, id="1d-mask_other"),
    pytest.param(False, "sos_dim", True, id="1d-mask_both"),
    pytest.param(True, None, False, id="2d-no_mask"),
    pytest.param(True, "sos_dim", False, id="2d-mask_sos_dim"),
    pytest.param(True, "non_sos_dim", False, id="2d-mask_non_sos_dim"),
    pytest.param(True, "both_dims", False, id="2d-mask_both_dims"),
    pytest.param(True, "sos_dim", True, id="2d-mask_sos_dim+other"),
    pytest.param(True, "non_sos_dim", True, id="2d-mask_non_sos_dim+other"),
    pytest.param(True, "both_dims", True, id="2d-mask_both_dims+other"),
]


@pytest.mark.skipif(not SOS_PATHS, reason="No SOS-capable solver installed")
@pytest.mark.parametrize("sos_type", [1, 2])
@pytest.mark.parametrize(("solver", "io_api"), SOS_PATHS)
@pytest.mark.parametrize(
    ("sos_var_2d", "mask_on_sos", "mask_on_other"), FIXTURE_CONFIGS
)
def test_sos_with_masked_variables(
    sos_masked_model: Callable[..., tuple[Model, float, np.ndarray]],
    solver: str,
    io_api: str,
    sos_type: int,
    sos_var_2d: bool,
    mask_on_sos: MaskOnSos,
    mask_on_other: bool,
) -> None:
    """
    Three-oracle test: status + objective + element-wise solution.

    Asymmetric objective + element-wise solution check ensures we catch:
    - direct-path OOB raises (status != ok)
    - LP parser rejections (status != ok)
    - silent SOS-set corruption (objective and/or solution differ)
    """
    m, expected_obj, expected_sol = sos_masked_model(
        sos_type=sos_type,
        sos_var_2d=sos_var_2d,
        mask_on_sos=mask_on_sos,
        mask_on_other=mask_on_other,
    )
    m.solve(solver_name=solver, io_api=io_api)

    # Oracle 1: did the solve succeed?
    assert m.status == "ok", (
        f"solver={solver} io_api={io_api} status={m.status!r} "
        f"termination={m.termination_condition!r}"
    )

    # Oracle 2: is the objective at the analytical optimum?
    assert m.objective.value is not None
    assert m.objective.value == pytest.approx(expected_obj, abs=1e-5)

    # Oracle 3: are the right slots at the right values?
    actual_sol = m.variables["sos_var"].solution.values
    np.testing.assert_allclose(
        actual_sol,
        expected_sol,
        atol=1e-5,
        equal_nan=True,
        err_msg=(
            f"sos_var.solution mismatch for solver={solver} io_api={io_api} "
            f"sos_type={sos_type} sos_var_2d={sos_var_2d} "
            f"mask_on_sos={mask_on_sos!r} mask_on_other={mask_on_other}"
        ),
    )


def test_sos_to_file_skips_fully_masked_sos_variable(tmp_path: Path) -> None:
    """A fully-masked SOS variable writes no LP ``sos`` set entries."""
    m = Model()
    ci = pd.Index([0, 1, 2, 3], name="i")
    free = m.add_variables(lower=0, upper=1, name="free")
    sos_var = m.add_variables(
        lower=0,
        upper=1,
        coords=[ci],
        mask=pd.Series(False, index=ci),
        name="sos_var",
    )
    m.add_sos_constraints(sos_var, sos_type=1, sos_dim="i")
    m.add_objective(free)

    fn = tmp_path / "model.lp"
    m.to_file(fn)
    lp = fn.read_text()

    assert "x-1" not in lp
    sos_section = lp.partition("\nsos\n")[2]
    assert "S1 ::" not in sos_section
