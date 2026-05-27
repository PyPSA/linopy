# Internal Performance Benchmarks

This suite benchmarks the **linopy part end-to-end** in two phases:

1. **Build**: construct the linopy model.
2. **Solver handoff**: convert a built model into solver-consumable form.

> **Note:** `benchmark/` (singular) is for external framework comparisons. `benchmarks/` is only for internal linopy performance tracking.

## What is covered

- **Build** (`benchmarks/test_build.py`): variable creation, expression construction, constraints, objective.
- **Solver handoff**:
  - canonical in-memory (`benchmarks/test_matrices.py`) via `A`, `b`, `c`, bounds, labels (**required**),
  - file handoff (`benchmarks/test_lp_write.py`) via LP serialization (**optional**),
  - direct API handoff (e.g. `to_highspy`) when enabled (**optional**, solver-specific).

## What is not covered

- Solver algorithm performance (optimize/solve runtime).
- Cross-solver ranking.
- Nonlinear/quadratic benchmark suites.

## Models

Core models:

- `basic`
- `knapsack`
- `expression_arithmetic`
- `sparse_network`

Extended (optional dependency):

- `pypsa_scigrid`

## Setup

```bash
pip install -e ".[benchmarks]"
```

## Run benchmarks

```bash
# Quick smoke run
pytest benchmarks/ --quick

# Full timing run (build + handoff)
pytest benchmarks/test_build.py benchmarks/test_matrices.py benchmarks/test_lp_write.py

# Single model
pytest benchmarks/test_build.py -k basic
```

## Metrics

- **Time**: pytest-benchmark median runtime (IQR for stability).
- **Memory**: pytest-memray peak RSS (MiB), primarily tracked for Build.

## Results and history

- Raw outputs live in `.benchmarks/` (gitignored).
- Store comparison snapshots as JSON and compare to a rolling `master` baseline.

```bash
# Timing snapshot
pytest benchmarks/test_build.py benchmarks/test_matrices.py benchmarks/test_lp_write.py \
  --benchmark-json ".benchmarks/timing-$(date +%Y%m%d-%H%M%S).json"

# Memory snapshot (Build by default)
python benchmarks/memory.py save "$(git rev-parse --short HEAD)"

# Compare memory snapshots
python benchmarks/memory.py compare <baseline-label> <candidate-label>
```
