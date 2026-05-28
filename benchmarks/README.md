# Internal Performance Benchmarks

This suite benchmarks the **linopy part end-to-end** across three phases:

1. **Build** — construct the linopy model.
2. **Solver handoff** — convert a built model into solver-consumable form
   (in-memory matrices, LP file, native solver instance, netCDF).
3. **Persistence round-trip** — `to_netcdf` / `read_netcdf`.

> **Note:** `benchmark/` (singular) is for external framework comparisons.
> `benchmarks/` is only for internal linopy performance tracking.

## What is covered

| Phase                 | Test file                       | Notes                                              |
| --------------------- | ------------------------------- | -------------------------------------------------- |
| Build                 | `test_build.py`                 | variables / expressions / constraints / objective  |
| Matrices              | `test_matrices.py`              | `A`, `b`, `c`, bounds, labels, `Q` for QP          |
| LP write              | `test_lp_write.py`              | `model.to_file(...)`                               |
| netCDF write/read     | `test_netcdf.py`                | `to_netcdf` / `read_netcdf`                        |
| Solver handoff        | `test_solver_handoff.py`        | `lp.io.to_highspy / to_gurobipy / to_mosek / to_xpress` — skipped per-solver when not installed |
| PyPSA carbon handoff  | `test_pypsa_carbon_management.py` | `set_names=True/False`, `freeze_constraints=True/False` |

What we *don't* cover: solver algorithm performance (`Solver.solve()`
runtime), cross-solver ranking, nonlinear / general-quadratic constraint
suites.

## Models

The suite is driven by a **reusable model registry**. Each model file under
`benchmarks/models/` exposes a `build_<name>(size) -> linopy.Model` callable
and a module-level `SPEC` describing features, applicable phases, default
sizes, and optional dependencies.

| Name                    | Features            | Typical use                                         |
| ----------------------- | ------------------- | --------------------------------------------------- |
| `basic`                 | continuous          | dense LP scaling                                    |
| `knapsack`              | binary              | MIP binary-section path                             |
| `expression_arithmetic` | continuous          | stresses `+`, `*`, `sum`, broadcasting              |
| `sparse_network`        | continuous          | mismatched-coordinate / sparse coefficient handling |
| `milp`                  | integer             | general-integer (non-binary) MIP path               |
| `qp`                    | quadratic           | continuous QP / `matrices.Q` path                   |
| `sos` *(linopy ≥ recent)* | sos              | `Model.add_sos_constraints` + LP SOS section        |
| `piecewise` *(linopy ≥ recent)* | piecewise  | `Model.add_piecewise_formulation`                    |
| `masked`                | masked              | `mask=` on `add_variables` / `add_constraints`      |
| `pypsa_scigrid` *(optional)* | continuous     | real PyPSA model                                    |

The `sos` and `piecewise` specs are skipped automatically if the underlying
APIs aren't present in the installed linopy.

### Reusing the registry outside the suite

The registry is a plain importable object — use it from any test, script,
or profiling session:

```python
from benchmarks import REGISTRY

# Look up by name
model = REGISTRY["basic"].build(100)

# Iterate (e.g. parametrize your own test)
for spec in REGISTRY.values():
    m = spec.build(spec.sizes[0])
    ...

# Filter by feature or phase
from benchmarks import filter_by, QUADRATIC, TO_GUROBIPY

qp_specs = filter_by(has_feature=QUADRATIC)
gurobi_specs = filter_by(has_phase=TO_GUROBIPY)
```

To add a new model, drop a file under `benchmarks/models/`, expose a
`build_<name>(size)`, and call `register(ModelSpec(...))`. Import it from
`benchmarks/models/__init__.py` so the registration fires.

### Worked walkthrough

[`notebooks/registry_usage.ipynb`](notebooks/registry_usage.ipynb) is the
canonical walkthrough — it runs through every pattern above end-to-end.
GitHub renders it inline. CI executes it on every PR via `jupyter nbconvert
--execute`, so the examples can't silently rot.

Open it locally with JupyterLab launched from the repo root:

```bash
jupyter lab benchmarks/notebooks/registry_usage.ipynb
```

## Setup

Two install paths, depending on what you're doing:

```bash
# Development / casual benchmark runs — loose constraints from pyproject
uv sync --extra dev --extra benchmarks
source .venv/bin/activate

# Stable measurement environment — fully resolved lockfile (linopy itself
# is excluded, so you install whichever linopy version you want on top)
uv pip install -r benchmarks/requirements.lock
uv pip install -e .            # current linopy
# — or —
uv pip install linopy==0.5.0   # for a cross-version sweep
```

The lockfile pins every transitive (numpy / scipy / pandas / xarray / ...)
so the *environment around linopy* stays stable. Absolute numbers are still
machine-dependent (CPU, cache, memory bandwidth) — what the lockfile gives
you is consistency over time on the same machine, so when you run the suite
at two points the delta reflects linopy changes, not a numpy upgrade.

Regenerate after bumping the ``[benchmarks]`` pins in ``pyproject.toml``:

```bash
uv pip compile pyproject.toml --extra benchmarks --extra dev --extra solvers \
  --no-emit-package linopy \
  -o benchmarks/requirements.lock
```

## Run benchmarks

Everything is exposed through a single typer-based CLI. The CLI's
`--help` is the source of truth — run it for the full menu:

```bash
python -m benchmarks --help
python -m benchmarks <command> --help
```

Pytest still works directly for power users (`pytest benchmarks/ ...`).

### Size tiers

Each spec declares its own `quick_threshold` and `long_threshold`:

| Mode              | Sizes included            | Typical use                            |
| ----------------- | ------------------------- | -------------------------------------- |
| `smoke`           | `size <= quick_threshold` | CI smoke, fast local sanity check      |
| `run`             | `size <= long_threshold`  | Default: medium-cost regression timing |
| `run --long`      | all sizes                 | Full sweep (the slow stuff — many min) |

### Quick reference

```bash
# Fastest sanity check (~18s, what CI runs)
python -m benchmarks smoke

# Default timing run
python -m benchmarks run

# Save / compare memory snapshots
python -m benchmarks memory save "$(git rev-parse --short HEAD)"
python -m benchmarks memory compare master my-feature
```

## Metrics

- **Time** — pytest-benchmark median runtime (IQR for stability).
- **Memory** — pytest-memray peak RSS (MiB), tracked for Build only because
  later phases include build allocations and make attribution unreliable.

## Results and history

Raw outputs live in `.benchmarks/` (gitignored). Store comparison snapshots
as JSON and compare to a rolling `master` baseline:

```bash
# Timing snapshot
pytest benchmarks/ \
  --benchmark-json ".benchmarks/timing-$(date +%Y%m%d-%H%M%S).json"

# Memory snapshot (Build by default)
python benchmarks/memory.py save "$(git rev-parse --short HEAD)"

# Compare memory snapshots
python benchmarks/memory.py compare <baseline-label> <candidate-label>
```
