"""Memory runner: measures peak memory during model construction."""

from __future__ import annotations

import gc
import tracemalloc

import numpy as np

PHASE = "memory"


def run(
    name: str,
    builder,
    builder_args: dict,
    iterations: int = 5,
    **kwargs,
) -> dict | None:
    """
    Measure peak memory via tracemalloc over multiple iterations.

    Uses fewer iterations by default since memory measurement is slower.
    Returns dict with median/max peak memory and model stats.
    """
    # Warmup
    model = builder(**builder_args)
    if model is None:
        return None
    del model
    gc.collect()

    peaks = []
    nvars = 0
    ncons = 0

    for _ in range(iterations):
        gc.collect()
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        tracemalloc.start()
        tracemalloc.reset_peak()

        model = builder(**builder_args)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if model is None:
            continue

        nvars = int(getattr(model, "nvars", 0))
        ncons = int(getattr(model, "ncons", 0))
        peaks.append(peak / 1e6)  # bytes to MB
        del model

    if not peaks:
        return None

    peaks_arr = np.array(peaks)
    return {
        "phase": PHASE,
        "name": name,
        "params": builder_args,
        "iterations": iterations,
        "peak_memory_median_mb": float(np.median(peaks_arr)),
        "peak_memory_max_mb": float(np.max(peaks_arr)),
        "nvars": nvars,
        "ncons": ncons,
    }
