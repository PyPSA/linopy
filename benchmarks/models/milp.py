"""
MILP benchmark: capacitated facility location with general integers.

Decision variables:
    y_f  in {0,1,...,K}      integer "modules" to open at facility f
    x_{f,c} >= 0             continuous flow from facility f to customer c

Constraints:
    sum_c x_{f,c}  <=  cap * y_f       (capacity per facility)
    sum_f x_{f,c}  ==  d_c             (demand at each customer)

Objective:
    minimize  sum_{f,c} t_{f,c} * x_{f,c}  +  sum_f f_f * y_f

The general-integer ``y`` exercises the matrix accessor's MIP integer-section
path and the LP-writer's general-integer block — neither the binary knapsack
nor the continuous LPs hit those paths.
"""

from __future__ import annotations

import numpy as np

import linopy
from benchmarks.registry import (
    DEFAULT_PHASES,
    BenchSpec,
    register,
)

SIZES = (10, 50)


def build_milp(n: int) -> linopy.Model:
    rng = np.random.default_rng(42)
    facilities = np.arange(n)
    customers = np.arange(n)

    cap = 100.0  # capacity per module
    Y_MAX = 5  # max modules per facility
    transport = rng.uniform(1, 20, size=(n, n))  # per-unit shipping cost
    fixed = rng.uniform(50, 200, size=n)  # cost per facility module
    demand = rng.uniform(20, 80, size=n)  # demand at each customer

    m = linopy.Model()
    y = m.add_variables(
        lower=0,
        upper=Y_MAX,
        coords=[facilities],
        dims=["facility"],
        integer=True,
        name="y",
    )
    x = m.add_variables(
        lower=0,
        coords=[facilities, customers],
        dims=["facility", "customer"],
        name="x",
    )

    m.add_constraints(x.sum("customer") - cap * y <= 0, name="capacity")
    m.add_constraints(x.sum("facility") == demand, name="demand")

    m.add_objective((transport * x).sum() + (fixed * y).sum())
    return m


SPEC = register(
    BenchSpec(
        name="milp",
        build=build_milp,
        sweep=SIZES,
        phases=DEFAULT_PHASES,
    )
)
