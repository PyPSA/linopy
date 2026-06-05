# io_api comparison: linopy vs polar-high, same machine

Self-contained build+IO benchmark answering two questions raised in
[#740](https://github.com/PyPSA/linopy/issues/740):

1. Does `io_api="direct"` change linopy's build column versus `io_api="lp"`?
2. How does linopy compare to [polar-high](https://github.com/nodal-tools/polar-high)
   (a Polars-backed LP/MIP eDSL on HiGHS) when both run on the **same machine**?

The original cross-tool numbers in #740 were measured on different hardware, so
only ratios were comparable. This harness runs all four columns through one
process-per-cell harness, one HiGHS build, one machine.

## The model

linopy's own `basic_model` dense LP (see `benchmark/scripts/benchmark_linopy.py`),
and the algebraically identical LP expressed in polar-high:

```
min  sum_{i,j} (2 x[i,j] + y[i,j])
s.t. x[i,j] - y[i,j] >= i        for i,j in {1..N}
     x[i,j] + y[i,j] >= 0
     x, y >= 0
```

`2*N^2` variables, `2*N^2` constraint rows.

## What is measured

Each `(tool, N)` cell runs in a fresh subprocess under `/usr/bin/time -l`:

- **build+IO time** — wall clock for model construction plus the matrix
  serialization / solver-load path, with HiGHS time-limited to ~0 so no real
  solve happens. linopy builds its matrix lazily in xarray, so almost all of its
  cost lands in the IO step inside `solve()`, not in the construction call;
  `total_s` (build + IO, HiGHS short-circuited) is the only fair cross-tool
  metric and matches the "Total time" column in #740.
- **peak_rss_gb** — process maximum resident set size (`ru_maxrss`), the
  high-water mark including transient HiGHS-setup peaks.

Four columns:

| Tool | Path |
|---|---|
| `linopy-lp` | linopy build -> `.lp` text file -> HiGHS reparse |
| `linopy-direct` | linopy build -> in-memory highspy load (no disk) |
| `polar` | polar-high (regular) -> highspy |
| `polar-sm` | polar-high `save_memory=True` -> highspy (MPS roundtrip) |

## Running it

Dependencies (latest released wheels): `linopy`, `highspy`, `polar-high`,
`polars`, `numpy`, `pandas`, `matplotlib`. Then:

```bash
python benchmark_io_api.py                       # full sweep N=500..3000
python benchmark_io_api.py --nrange 200,500 --tools linopy-direct,polar   # subset
python plot_io_api.py                            # writes tool_compare.svg / .png
```

Peak-RSS parsing currently targets macOS `/usr/bin/time -l` (bytes) with a GNU
`time -v` (kbytes) fallback.

## Results (macOS arm64, single thread, HiGHS 1.14.0, linopy 0.7.0, polar-high 2.4.5)

See `tool_compare_results.csv` and `tool_compare.svg`. Headline at N=3000
(~18M variables):

| Tool | build+IO time | peak RSS |
|---|---|---|
| `linopy-direct` | 33.7 s | 15.6 GB |
| polar-high (regular) | 53.9 s | 12.8 GB |
| polar-high (save_memory) | 82.7 s | 11.0 GB |
| `linopy-lp` | 94.4 s | 16.0 GB |

`io_api="direct"` is the fastest of the four on build+IO; polar-high retains a
lower peak-RSS. The modelling-layer gap is a memory gap, not a speed gap.
