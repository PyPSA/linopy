"""LP write runner: measures LP file writing speed."""

from __future__ import annotations

import gc
import tempfile
import time
from pathlib import Path

import numpy as np

PHASE = "lp_write"


def run(
    label: str,
    builder,
    builder_args: dict,
    iterations: int = 10,
    **kwargs,
) -> dict | None:
    """
    Time LP file writing over multiple iterations.

    Builds the model once, then times repeated LP file writes.
    Returns dict with median, q25, q75 write times.
    """
    model = builder(**builder_args)
    if model is None:
        return None

    nvars = int(getattr(model, "nvars", 0))
    ncons = int(getattr(model, "ncons", 0))

    times = []
    with tempfile.TemporaryDirectory() as tmpdir:
        lp_path = Path(tmpdir) / "model.lp"

        # Warmup
        model.to_file(lp_path)

        for _ in range(iterations):
            gc.collect()
            t0 = time.perf_counter()
            model.to_file(lp_path)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

    times_arr = np.array(times)
    return {
        "phase": PHASE,
        "label": label,
        "params": builder_args,
        "iterations": iterations,
        "write_time_median_s": float(np.median(times_arr)),
        "write_time_q25_s": float(np.percentile(times_arr, 25)),
        "write_time_q75_s": float(np.percentile(times_arr, 75)),
        "nvars": nvars,
        "ncons": ncons,
    }
