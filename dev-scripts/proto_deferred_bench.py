"""
Benchmark: dense groupby-sum balance constraint vs deferred CSR realization.

Scenario from issue #745 (hub-skewed nodal balance): 120 buses, one hub with
8000 generators, 100 on each other bus (19,900 total), ring of 120 lines,
24 snapshots. Build-only.

Run under memray, one variant per process:
    memray run -o dense.bin proto_deferred_bench.py dense
    memray run -o deferred.bin proto_deferred_bench.py deferred
"""

from __future__ import annotations

import sys
import time

from proto_deferred_check import build_base_model
from proto_deferred_groupby import DeferredGroupbySum, add_deferred_constraints

N_BUS = 120
HUB = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
GENS_PER_BUS = [HUB] + [100] * (N_BUS - 1)
N_SNAP = 24


def main() -> None:
    variant = sys.argv[1]
    m, gen_p, flow, eff, gbus, bus0, bus1, load, buses = build_base_model(
        N_BUS, GENS_PER_BUS, N_SNAP
    )
    start = time.perf_counter()

    if variant == "setup":
        print(f"setup only: {time.perf_counter() - start:.2f}s")
        return
    if variant == "dense":
        lhs = (
            (eff * gen_p).groupby(gbus).sum()
            + (1.0 * flow).groupby(bus0).sum()
            - (1.0 * flow).groupby(bus1).sum()
        )
        con = m.add_constraints(lhs == load, name="balance")
        nterm = con.data.sizes["_term"]
    elif variant == "deferred":
        parts = [
            DeferredGroupbySum(eff * gen_p, gbus),
            DeferredGroupbySum(1.0 * flow, bus0),
            DeferredGroupbySum(-1.0 * flow, bus1),
        ]
        con = add_deferred_constraints(m, parts, "=", load, "balance", buses)
        nterm = con.nterm
    else:
        raise SystemExit(f"unknown variant {variant!r}")

    build_s = time.perf_counter() - start

    # prove the export seam works: the LP writer consumes this frame
    start = time.perf_counter()
    n_rows = con.to_polars().height
    export_s = time.perf_counter() - start

    print(
        f"{variant}: build {build_s:.2f}s, to_polars {export_s:.2f}s, "
        f"nterm={nterm}, polars term rows={n_rows}"
    )


if __name__ == "__main__":
    main()
