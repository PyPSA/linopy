# Internal Performance Benchmarks

End-to-end timing and memory for the linopy half of an optimization run:
build a model, hand it off to a solver, round-trip via netCDF. Solver
algorithm runtime is intentionally out of scope.

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
- **Memory** — pytest-memray peak RSS (MiB), tracked for the build phase
  only because later phases include build allocations and attribution
  becomes unreliable. Use `memory save` / `memory compare`.
