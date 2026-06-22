# Internal Performance Benchmarks

End-to-end performance tracking for `linopy` — build → matrix generation →
LP / netCDF (de)serialization → solver handoff → a fixed PyPSA model. Solver
algorithm runtime is out of scope.

The suite is a set of `pytest-benchmark` tests driven by a model registry.
**CodSpeed** measures them in CI (walltime on dedicated runners, memory on every
PR); locally you just run `pytest`.

> `benchmark/` (singular) is the legacy external-framework suite.
> `benchmarks/` (plural) is this internal suite.

## Layout

- `registry.py`, `phases.py`, `conftest.py` — the harness (specs, measured
  verbs, pytest wiring).
- `models/`, `patterns/` — the subjects; each file self-registers one `BenchSpec`.
- `drivers/` — one `test_<phase>.py` per measured phase.

## Models vs patterns

Two kinds of benchmark spec, same harness and same phases, distinguished by
their sweep axis:

- **Models** (`models/`, `REGISTRY`) — whole `linopy.Model`s swept over
  `size` (axis `n`): "how does cost scale with the problem?"
- **Patterns** (`patterns/`, `PATTERNS`) — fragments of realistic modelling
  code (a balance constraint, a KVL contraction) swept over `severity`
  (0–100, axis `severity`): "how does cost respond as one data shape goes
  from benign to pathological?"

Both kinds build a complete `linopy.Model`, so both run the **same phases** and
share the phase drivers (`drivers/test_build.py`, `drivers/test_matrices.py`, …)
— they're just more `(spec, value)` rows, tagged by `axis`. There is no separate
pattern driver. Running a pattern through `build` *and* `to_lp` shows whether a
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

`pypsa` is optional — `pypsa_scigrid` and `drivers/test_pypsa_carbon_management.py`
skip gracefully without it: `uv pip install pypsa`.

The `[benchmarks]` extra in `pyproject.toml` pins every direct dep that affects
measurement (`numpy`, `scipy`, `xarray`, `pandas`, `polars`, `dask`, …) so
run-to-run deltas reflect linopy changes, not dependency bumps.

## Running

```bash
pytest benchmarks/                       # the suite
pytest benchmarks/ --benchmark-disable -q   # smoke: every spec builds once
pytest benchmarks/ --pipeline            # + the opt-in end-to-end pipeline test
```

Each spec declares one `sizes` (models) / `severities` (patterns) tuple — a
small representative set, kept tight because CodSpeed measures it on every PR.
Need a scaling curve? That's a local pytest-benchmem job, not this suite.

## CI

- **Smoke** (`benchmark-smoke.yml`) — every PR: every spec builds and every
  phase fires once under `--benchmark-disable`. A "did a refactor break a
  spec?" check, not timing.
- **CodSpeed** (`codspeed.yml`) — two jobs: **memory** (heap-allocation
  tracking, every PR, free GitHub runner) and **walltime** (bare-metal macro
  runner, on `master` or a PR labelled `trigger:benchmark`). Informational,
  non-gating.

Activating CodSpeed upstream needs a maintainer to connect the repo to the
CodSpeed app (OIDC auth, no token secret); the workflows are already wired.
