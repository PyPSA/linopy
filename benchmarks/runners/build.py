"""Build runner: measures model construction speed."""

from __future__ import annotations

import gc
import time

import numpy as np

PHASE = "build"


def run(
    name: str,
    builder,
    builder_args: dict,
    iterations: int = 30,
    **kwargs,
) -> dict | None:
    """
    Time model construction over multiple iterations.

    Returns dict with median, q25, q75 build times and model stats.
    """
    # Warmup
    model = builder(**builder_args)
    if model is None:
        return None
    del model
    gc.collect()

    times = []
    nvars = 0
    ncons = 0

    for _ in range(iterations):
        gc.collect()
        t0 = time.perf_counter()
        model = builder(**builder_args)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        nvars = int(getattr(model, "nvars", 0))
        ncons = int(getattr(model, "ncons", 0))
        del model

    times_arr = np.array(times)
    return {
        "phase": PHASE,
        "name": name,
        "params": builder_args,
        "iterations": iterations,
        "build_time_median_s": float(np.median(times_arr)),
        "build_time_q25_s": float(np.percentile(times_arr, 25)),
        "build_time_q75_s": float(np.percentile(times_arr, 75)),
        "nvars": nvars,
        "ncons": ncons,
    }
