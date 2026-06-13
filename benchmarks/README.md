# Internal Performance Benchmarks

End-to-end performance tracking for `linopy` — build → matrix generation →
LP / netCDF (de)serialization → solver handoff → a fixed PyPSA model. Solver
algorithm runtime is out of scope.

The suite is a set of `pytest-benchmark` tests driven by a model registry.
**CodSpeed** measures them in CI (walltime on dedicated runners, memory on every
PR); locally you just run `pytest`.

> `benchmark/` (singular) is the legacy external-framework suite.
> `benchmarks/` (plural) is this internal suite.

## Models vs patterns

Two kinds of benchmark spec, same harness and same phases, distinguished by
their sweep axis:

- **Models** (`models/`, `REGISTRY`) — whole `linopy.Model`s swept over
  `size` (axis `n`): "how does cost scale with the problem?"
- **Patterns** (`patterns/`, `PATTERNS`) — fragments of realistic modelling
  code (a balance constraint, a KVL contraction) swept over `severity`
  (0–100, axis `severity`): "how does cost respond as one data shape goes
  from benign to pathological?" Each `PatternSpec.description` documents what
  its dial means (`"0: …, 100: …"`).

Both kinds build a complete `linopy.Model`, so both run the **same phases** and
share the phase drivers (`test_build.py`, `test_matrices.py`, …) — they're just
more `(spec, value)` rows, tagged by `axis`. There is no separate pattern
driver. Running a pattern through `build` *and* `to_lp` shows whether a
dense-`_term` blow-up propagates to export or collapses.

Patterns target the operations where the dense-`_term` representation forces
materialisation — `groupby().sum()` padding, sparse `@` densification — so a
`severity` sweep draws the cost cliff. Adding either kind is one file: drop it
in `models/` or `patterns/`, call `register(...)` / `register_pattern(...)`.

## Install

```bash
uv sync --extra dev --extra benchmarks
source .venv/bin/activate
```

`pypsa` is optional — `pypsa_scigrid` and `test_pypsa_carbon_management.py`
skip gracefully without it: `uv pip install pypsa`.

The `[benchmarks]` extra in `pyproject.toml` pins every direct dep that affects
measurement (`numpy`, `scipy`, `xarray`, `pandas`, `polars`, `dask`, …) so
run-to-run deltas reflect linopy changes, not dependency bumps.

## Running

```bash
pytest benchmarks/                       # the full suite
pytest benchmarks/ --quick               # the per-PR subset (smaller sizes)
pytest benchmarks/ --quick --benchmark-disable -q   # smoke: every spec builds once
```

Size tiers are explicit per spec (`SIZES` / `QUICK_SIZES` / `LONG_SIZES`):
`--quick` runs the subset, the default is `SIZES − LONG_SIZES`, `--long` runs
everything. A single `skip_reason` gates the suite.

## CI

- **Smoke** (`benchmark-smoke.yml`) — every PR: every spec builds and every
  phase fires once under `--quick --benchmark-disable`. A "did a refactor break
  a spec?" check, not timing.
- **CodSpeed memory** (`codspeed-memory.yml`) — every PR: heap-allocation
  tracking, informational, non-gating.
- **CodSpeed walltime** (`codspeed-macro.yml`) — on `master` or a PR labelled
  `trigger:benchmark`: wall-clock on dedicated bare-metal runners.

Activating CodSpeed upstream needs a maintainer to connect the repo to the
CodSpeed app (OIDC auth, no token secret); the workflows are already wired.
