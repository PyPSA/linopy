"""
SOS1 benchmark: multi-mode generation with at-most-one-mode-per-generator.

Each generator has ``n_modes`` operating modes (different cap/cost tradeoff).
SOS1 over the ``mode`` dimension enforces that each generator picks at most
one mode.

Decision variables:
    y[gen, mode]  >= 0     continuous output per (generator, mode)

Constraints:
    y[gen, mode]  <= cap[mode]
    sum_{gen,mode} y  >= demand_total
    SOS1 over "mode" for each gen

This benchmark exercises ``Model.add_sos_constraints`` (commits be6d3a3 /
8aa8d0c) and the LP-writer's SOS section. In linopy, native SOS support is
declared by Gurobi / Cplex / Xpress only (see ``SolverFeature.SOS_CONSTRAINTS``).
HiGHS and Mosek would need ``apply_sos_reformulation()`` first.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

import linopy
from benchmarks.registry import (
    BUILD,
    FROM_NETCDF,
    MATRICES,
    TO_GUROBIPY,
    TO_LP,
    TO_NETCDF,
    TO_XPRESS,
    BenchSpec,
    register,
)

SIZES = (10, 1_000)

_N_MODES = 5
_API_AVAILABLE = hasattr(linopy.Model, "add_sos_constraints")


def build_sos(n_gens: int) -> linopy.Model:
    modes = np.arange(_N_MODES)
    cap = xr.DataArray(np.linspace(20.0, 100.0, _N_MODES), coords=[("mode", modes)])
    cost = xr.DataArray(np.linspace(1.0, 8.0, _N_MODES), coords=[("mode", modes)])

    m = linopy.Model()
    y = m.add_variables(
        lower=0,
        upper=float(cap.max()),
        coords=[range(n_gens), modes],
        dims=["gen", "mode"],
        name="y",
    )

    m.add_constraints(y <= cap, name="mode_cap")
    demand_total = 0.4 * n_gens * float(cap.max())
    m.add_constraints(y.sum() >= demand_total, name="demand")

    m.add_sos_constraints(y, sos_type=1, sos_dim="mode")

    m.add_objective((cost * y).sum())
    return m


# ``add_sos_constraints`` is a recent API. On older linopy we silently skip
# registering this model — the rest of the suite stays usable.
SPEC: BenchSpec | None
if _API_AVAILABLE:
    SPEC = register(
        BenchSpec(
            name="sos",
            build=build_sos,
            sweep=SIZES,
            # HiGHS / Mosek lack native SOS in linopy — would need
            # ``reformulate_sos=True``, which mutates the model and defeats
            # the benchmark. Only solvers with native SOS appear here.
            phases=frozenset(
                {
                    BUILD,
                    MATRICES,
                    TO_LP,
                    TO_NETCDF,
                    FROM_NETCDF,
                    TO_GUROBIPY,
                    TO_XPRESS,
                }
            ),
        )
    )
else:
    SPEC = None
