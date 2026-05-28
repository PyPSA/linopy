# Internal Performance Benchmarks

This suite benchmarks the **linopy part end-to-end** across four phases:

1. **Build** — construct the linopy model.
2. **Solver handoff** — convert a built model into solver-consumable form
   (in-memory matrices, LP file, native solver instance).
3. **netCDF serialization / deserialization** — `to_netcdf` / `read_netcdf`.
4. **End-to-end** — a fixed real-world PyPSA model all the way to a solver
   instance.

Solver algorithm runtime is intentionally out of scope.

| Phase                  | Test file                           | Measures                                                                  |
| ---------------------- | ----------------------------------- | ------------------------------------------------------------------------- |
| Build                  | `test_build.py`                     | constructing variables / expressions / constraints / objective            |
| Solver handoff         | `test_matrices.py`                  | `A`, `b`, `c`, bounds, labels, `Q` for QP                                 |
| Solver handoff         | `test_lp_write.py`                  | `model.to_file(...)` — LP / MPS serialization                             |
| Solver handoff         | `test_solver_handoff.py`            | `lp.io.to_highspy` / `to_gurobipy` / `to_mosek` / `to_xpress`             |
| netCDF (de)serialization | `test_netcdf.py`                  | `to_netcdf` and `read_netcdf` round-trip                                  |
| End-to-end (PyPSA)     | `test_pypsa_carbon_management.py`   | Fixed real-world pypsa network through `network.optimize.create_model` and on to highspy; sweeps `freeze_constraints` and `set_names`. |

The netCDF benchmarks reuse the same file path across pytest-benchmark
iterations, so reads run hot-cache by design — what we want to track is
the (de)serialization code in `linopy` / `xarray`, not disk hardware.

> **Note:** `benchmark/` (singular) is for external framework comparisons.
> `benchmarks/` (plural) is only for internal linopy performance tracking.

## Setup

Two install paths:

```bash
# Development / casual benchmark runs — loose constraints from pyproject
uv sync --extra dev --extra benchmarks
source .venv/bin/activate

# Stable measurement environment — fully resolved lockfile
uv pip install -r benchmarks/requirements.lock
uv pip install -e .            # current linopy
# — or —
uv pip install linopy==0.5.0   # cross-version sweep target
```

`pypsa` is an **optional** benchmark dep — the `pypsa_scigrid` registry
spec and `test_pypsa_carbon_management.py` skip gracefully without it.
Install separately when you want them:

```bash
uv pip install pypsa
```

The lockfile excludes linopy itself so the same lockfile works for both
current-tip regression runs and `sweep` against older releases. Absolute
benchmark numbers are still machine-dependent (CPU, cache, memory
bandwidth) — what the lockfile gives you is consistency over time on the
same machine, so deltas reflect linopy changes, not a numpy upgrade.

Regenerate after bumping the `[benchmarks]` pins in `pyproject.toml`:

```bash
uv pip compile pyproject.toml --extra benchmarks --extra dev --extra solvers \
  --no-emit-package linopy \
  -o benchmarks/requirements.lock
```

## Run

Everything is exposed through one typer CLI. Its `--help` is the source of
truth — no command menu duplicated here:

```bash
python -m benchmarks --help
python -m benchmarks <command> --help
```

Three size tiers, configured per spec via `quick_threshold` / `long_threshold`:

| Mode         | Sizes included            | Typical use                              |
| ------------ | ------------------------- | ---------------------------------------- |
| `smoke`      | `size <= quick_threshold` | CI smoke (~18 s), fast local sanity      |
| `run`        | `size <= long_threshold`  | Default regression timing (~45 s)        |
| `run --long` | all sizes                 | Full sweep — the slow stuff (~2 min)     |

Pytest still works directly for power users (`pytest benchmarks/ ...`).

## Walkthrough

[`notebooks/registry_usage.ipynb`](notebooks/registry_usage.ipynb) is the
canonical walkthrough: import the registry, look up / iterate / filter
specs, build a model, parametrize your own pytest test off the registry,
spot-profile memory. GitHub renders it inline; CI executes it on every PR
(`python -m benchmarks notebook`) so the examples can't silently rot.

Open it locally with JupyterLab launched from the repo root:

```bash
jupyter lab benchmarks/notebooks/registry_usage.ipynb
```

## Metrics

- **Time** — pytest-benchmark median runtime (IQR for stability). Snapshots
  are JSON; pass `--json <path>` to `run` to save one, then diff against a
  baseline.
- **Memory** — peak allocations (MiB) via `memray.Tracker`, measured per
  `(phase, spec, size)` across all phases. The model is built *outside* the
  tracked region so the peak reflects only the phase work, not model
  construction. Use `memory save` (optionally `--phase` to scope) and
  `memory compare`.
