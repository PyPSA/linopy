# SPDX-FileCopyrightText: Contributors to linopy <https://github.com/PyPSA/linopy>
#
# SPDX-License-Identifier: MIT
"""
Drive linopy.contrib.plasmo on testProblem.nc: partition into a Benders master +
per-year subproblems, solve with PlasmoBenders, and cross-check the objective
against a monolithic HiGHS solve.

Run with: pixi run python run_benders.py
"""

import numpy as np

import linopy
from linopy.contrib.plasmo import Partition, PlasmoModel, benders, group, has

MODEL = "testProblem.nc"


def clean_model(m: linopy.Model) -> None:
    """
    testProblem.nc stores some term cells with a valid ``var`` but a NaN
    ``coeff`` (a padding artifact of the ragged ``_term`` axis). Replace those
    NaNs with 0, then let ``sanitize_zeros`` mask the now-zero terms so the
    exported matrix is clean (no NaN, no empty constraint rows).
    """
    for name in m.constraints:
        con = m.constraints[name]
        # replace NaN coeffs only (leave rhs/bounds, which may legitimately be
        # +/-inf, untouched -- we only touch coefficients here)
        con._update_data(coeffs=con.coeffs.where(~np.isnan(con.coeffs), 0.0))
    m.constraints.sanitize_zeros()


def main() -> None:
    m = linopy.read_netcdf(MODEL)
    clean_model(m)

    partition = Partition(
        top=~has("set_time_steps_yearly") | ~has("set_nodes"),
        sub=group("set_time_steps_yearly") & has("set_nodes"),
    )

    print("Solving with PlasmoBenders...")
    pm = PlasmoModel(m, partition)
    benders(pm)
    obj_benders = pm.result().solution.objective
    print(f"Benders objective:    {obj_benders:.6g}")

    # monolithic reference. NB: on this model linopy's assign_result crashes
    # when writing constraint duals back (a linopy bug unrelated to Benders);
    # the objective value is set before that step, so tolerate it.
    print("Solving monolithically (HiGHS)...")
    m2 = linopy.read_netcdf(MODEL)
    try:
        m2.solve(solver_name="highs")
    except Exception as e:  # noqa: BLE001
        # objective.value is assigned before the failing dual write-back
        print(f"  (tolerated linopy dual-assignment bug: {e})")
    print(f"Monolithic objective: {m2.objective.value:.6g}")

    rel = abs(obj_benders - m2.objective.value) / max(1.0, abs(m2.objective.value))
    print(f"relative difference:  {rel:.2e}")
    assert rel < 1e-4, "Benders objective does not match monolithic solve"
    print("OK: objectives match")


if __name__ == "__main__":
    main()
