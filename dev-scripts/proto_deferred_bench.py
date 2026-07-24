"""
Benchmark: eager groupby-sum balance constraint vs sparse CSR realization.

Scenario from issue #745 (hub-skewed nodal balance): 120 buses, one hub with
HUB generators, 100 on each other bus, ring of 120 lines, 24 snapshots.
Build-only, using the real API: ``groupby(g).sum(sparse=...)`` +
``add_constraints(lhs == load, freeze=...)``.

Run under memray, one variant per process:
    memray run -o eager.bin proto_deferred_bench.py eager [hub]
    memray run -o sparse.bin  proto_deferred_bench.py sparse  [hub]
"""

from __future__ import annotations

import sys
import time

sys.path.insert(
    0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "test")
)
from test_sparse_groupby import balance_lhs, base_model

N_BUS = 120
HUB = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
GENS_PER_BUS = tuple([HUB] + [100] * (N_BUS - 1))
N_SNAP = 24


def main() -> None:
    import linopy

    linopy.options["semantics"] = "v1"  # the sparse path is gated behind v1
    variant = sys.argv[1]
    m, gen_p, flow, eff, gbus, bus0, bus1, load, buses = base_model(
        gens_per_bus=GENS_PER_BUS, n_snap=N_SNAP
    )
    start = time.perf_counter()

    if variant == "setup":
        print(f"setup only: {time.perf_counter() - start:.2f}s")
        return

    sparse = variant == "sparse"
    lhs = balance_lhs(gen_p, flow, eff, gbus, bus0, bus1, sparse=sparse)
    con = m.add_constraints(lhs == load, name="balance", freeze=sparse)
    build_s = time.perf_counter() - start

    start = time.perf_counter()
    n_rows = con.to_polars().height
    export_s = time.perf_counter() - start

    print(
        f"{variant}: build {build_s:.2f}s, to_polars {export_s:.2f}s, "
        f"type={type(con).__name__}, polars term rows={n_rows}"
    )


if __name__ == "__main__":
    main()
