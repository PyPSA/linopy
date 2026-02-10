#!/usr/bin/env python3
"""
Ad-hoc benchmark to compare solve times with and without pre-solve scaling.

This is intentionally lightweight and meant for local experimentation.
It relies on HiGHS (highspy) being installed. Adjust sizes or iterations
via CLI flags if you want to stress test further.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable

import numpy as np

from linopy import Model
from linopy.scaling import ScaleOptions
from linopy.solvers import available_solvers


def build_model(n_vars: int, n_cons: int, density: float) -> Model:
    rng = np.random.default_rng(123)
    m = Model()
    x = m.add_variables(lower=0, name="x", coords=[range(n_vars)])

    data = rng.lognormal(mean=0.0, sigma=2.0, size=int(n_vars * n_cons * density))
    rows = rng.integers(0, n_cons, size=data.size)
    cols = rng.integers(0, n_vars, size=data.size)

    # accumulate entries per row
    for i in range(n_cons):
        mask = rows == i
        if not mask.any():
            continue
        coeffs = data[mask]
        vars_idx = cols[mask]
        lhs = sum(coeff * x.isel(dim_0=idx) for coeff, idx in zip(coeffs, vars_idx))
        rhs = abs(coeffs).sum() * 0.1
        m.add_constraints(lhs >= rhs, name=f"c{i}")

    obj_coeffs = rng.uniform(0.1, 1.0, size=n_vars)
    m.objective = (obj_coeffs * x).sum()
    return m


def time_solve(m: Model, scale: bool | ScaleOptions, repeats: int) -> Iterable[float]:
    for _ in range(repeats):
        start = time.perf_counter()
        status, _ = m.solve("highs", io_api="direct", scale=scale)
        end = time.perf_counter()
        if status != "ok":
            raise RuntimeError(f"Solve failed with status {status}")
        yield end - start


def run_benchmark(
    n_vars: int, n_cons: int, density: float, repeats: int
) -> tuple[np.ndarray, np.ndarray]:
    base_model = build_model(n_vars, n_cons, density)
    scaled_model = build_model(n_vars, n_cons, density)

    base_times = np.fromiter(time_solve(base_model, False, repeats), dtype=float)
    scaled_times = np.fromiter(time_solve(scaled_model, True, repeats), dtype=float)
    return base_times, scaled_times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vars", type=int, default=400, help="Number of variables.")
    parser.add_argument("--cons", type=int, default=300, help="Number of constraints.")
    parser.add_argument(
        "--density",
        type=float,
        default=0.01,
        help="Constraint density (0-1) for random coefficients.",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Number of solve repetitions."
    )
    args = parser.parse_args()

    if "highs" not in available_solvers:
        raise RuntimeError("HiGHS (highspy) is required for this benchmark.")

    base_times, scaled_times = run_benchmark(
        n_vars=args.vars, n_cons=args.cons, density=args.density, repeats=args.repeats
    )

    print(f"Solve times without scaling: {base_times}")
    print(f"Solve times with scaling   : {scaled_times}")
    print(
        f"Median speedup: {np.median(base_times) / np.median(scaled_times):.2f}x "
        f"(lower is better for scaled)"
    )


if __name__ == "__main__":
    main()
