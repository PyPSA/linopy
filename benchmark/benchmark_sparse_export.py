#!/usr/bin/env python3
"""
Benchmark the build and export of a PyPSA-like dispatch model.

Run as ``python benchmark/benchmark_sparse_export.py {sparse,dense,frozen}``.
``sparse`` uses ``Model(sparse=True)``, ``dense`` keeps mutable constraints and
``frozen`` freezes each constraint of a dense model. Every phase reports its
wall time, its peak traced memory and the operations that densified.
"""

from __future__ import annotations

import argparse
import gc
import tempfile
import time
import tracemalloc
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import linopy
from linopy.constants import PerformanceWarning
from linopy.io import to_highspy

MATRIX_ATTRS = ("A", "b", "c", "lb", "ub", "sense", "vlabels", "clabels")


class Phases:
    def __init__(self) -> None:
        self.rows: list[tuple[str, float, float, list[str]]] = []

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        gc.collect()
        tracemalloc.start()
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
        elapsed = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        densified = [
            str(w.message)[:90]
            for w in caught
            if issubclass(w.category, PerformanceWarning)
        ]
        self.rows.append((name, elapsed, peak / 1e6, densified))

    def report(self) -> None:
        for name, elapsed, peak, densified in self.rows:
            note = "DENSIFY: " + "; ".join(densified) if densified else ""
            print(f"{name:28s} {elapsed * 1e3:9.0f} ms   peak {peak:8.1f} MB   {note}")
        total = sum(row[1] for row in self.rows)
        print(f"{'TOTAL':28s} {total * 1e3:9.0f} ms")


def run(mode: str, n_bus: int, n_t: int, n_gen: int, n_line: int, n_sto: int) -> None:
    linopy.options["semantics"] = "v1"
    linopy.options["warn_on_densify"] = True
    rng = np.random.default_rng(0)

    bus = pd.RangeIndex(n_bus, name="bus")
    t = pd.RangeIndex(n_t, name="t")
    gen = pd.RangeIndex(n_gen, name="gen")
    line = pd.RangeIndex(n_line, name="line")
    sto = pd.RangeIndex(n_sto, name="sto")

    gen_bus = rng.integers(0, n_bus, n_gen)
    gen_bus[rng.random(n_gen) < 0.4] = 0
    gen_bus = pd.Series(gen_bus, index=gen, name="bus")
    sto_bus = pd.Series(np.arange(n_sto) % n_bus, index=sto, name="bus")
    bus0 = np.arange(n_line) % n_bus
    bus1 = (bus0 + 1 + rng.integers(0, 3, n_line)) % n_bus
    incidence = np.zeros((n_line, n_bus))
    incidence[np.arange(n_line), bus0] = -1
    incidence[np.arange(n_line), bus1] = 1
    incidence = xr.DataArray(incidence, coords=[line, bus])
    demand = xr.DataArray(rng.uniform(10, 100, (n_bus, n_t)), coords=[bus, t])
    pmax = xr.DataArray(rng.uniform(0, 1, (n_gen, n_t)), coords=[gen, t])

    freeze = {"sparse": None, "dense": False, "frozen": True}[mode]
    m = linopy.Model(sparse=mode == "sparse")
    phase = Phases()

    with phase("vars"):
        p = m.add_variables(0, coords=[gen, t], name="p")
        flow = m.add_variables(-100, 100, coords=[line, t], name="flow")
        soc = m.add_variables(0, 100, coords=[sto, t], name="soc")
        ch = m.add_variables(0, 50, coords=[sto, t], name="ch")
        dis = m.add_variables(0, 50, coords=[sto, t], name="dis")
    with phase("expr: groupby nodal supply"):
        supply = p.groupby(gen_bus).sum()
    with phase("expr: sto groupby"):
        sto_net = (dis - ch).groupby(sto_bus).sum()
    with phase("expr: flow @ incidence"):
        net_flow = flow @ incidence
    with phase("expr: balance sum"):
        balance = supply + sto_net + net_flow
    with phase("cons: balance"):
        m.add_constraints(balance == demand, name="balance", freeze=freeze)
    with phase("expr: soc shift"):
        soc_expr = soc - 0.99 * soc.shift(t=1) - 0.95 * ch + dis / 0.95
    with phase("cons: soc"):
        m.add_constraints(soc_expr == 0, name="soc", freeze=freeze)
    with phase("expr+cons: p <= pmax"):
        m.add_constraints(p <= pmax * 100, name="p_max", freeze=freeze)
    with phase("objective"):
        m.add_objective((p * 2).sum() + (ch + dis).sum())
    with phase("matrices"):
        matrices = m.matrices
        for attr in MATRIX_ATTRS:
            getattr(matrices, attr)
    with phase("to_highspy"):
        to_highspy(m)
    with phase("to_file(lp)"):
        m.to_file(Path(tempfile.mkdtemp()) / "model.lp", progress=False)

    print(
        f"=== {mode}  buses={n_bus} snapshots={n_t} generators={n_gen} "
        f"lines={n_line} storage={n_sto}  nvars={m._xCounter} ncons={m._cCounter}"
    )
    phase.report()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["sparse", "dense", "frozen"])
    parser.add_argument("--buses", type=int, default=200)
    parser.add_argument("--snapshots", type=int, default=720)
    parser.add_argument("--generators", type=int, default=2000)
    parser.add_argument("--lines", type=int, default=300)
    parser.add_argument("--storage", type=int, default=200)
    args = parser.parse_args()
    run(
        args.mode, args.buses, args.snapshots, args.generators, args.lines, args.storage
    )


if __name__ == "__main__":
    main()
