# Internal Performance Benchmarks

Measures linopy's own performance (build time, LP write speed, memory usage) across problem sizes using [pytest-benchmark](https://pytest-benchmark.readthedocs.io/) and [pytest-memray](https://pytest-memray.readthedocs.io/). Use these to check whether a code change introduces a regression or improvement.

> **Note:** The `benchmark/` directory (singular) contains *external* benchmarks comparing linopy against other modeling frameworks. This directory (`benchmarks/`) is for *internal* performance tracking only.

## Setup

```bash
pip install -e ".[benchmarks]"
```

## Running benchmarks

```bash
# Quick smoke test (small sizes only)
pytest benchmarks/ --quick

# Full timing benchmarks
pytest benchmarks/test_build.py benchmarks/test_lp_write.py benchmarks/test_matrices.py

# Run a specific model
pytest benchmarks/test_build.py -k basic
```

## Comparing timing between branches

```bash
# Save baseline results on master
git checkout master
pytest benchmarks/test_build.py --benchmark-save=master

# Switch to feature branch and compare
git checkout my-feature
pytest benchmarks/test_build.py --benchmark-save=my-feature --benchmark-compare=0001_master

# Compare saved results without re-running
pytest-benchmark compare 0001_master 0002_my-feature --columns=median,iqr
```

Results are stored in `.benchmarks/` (gitignored).

## Memory benchmarks

`memory.py` runs each test in a separate process with pytest-memray to get accurate per-test peak memory (including C/numpy allocations). Results are saved as JSON and can be compared across branches.

By default, only the build phase (`test_build.py`) is measured. Unlike timing benchmarks where `benchmark()` isolates the measured function, memray tracks all allocations within a test — including model construction in setup. This means LP write and matrix tests would report build + phase memory combined, making the phase-specific contribution impossible to isolate. Since model construction dominates memory usage, measuring build alone gives the most actionable numbers.

```bash
# Save baseline on master
git checkout master
python benchmarks/memory.py save master

# Save feature branch
git checkout my-feature
python benchmarks/memory.py save my-feature

# Compare
python benchmarks/memory.py compare master my-feature

# Quick mode (smaller sizes, faster)
python benchmarks/memory.py save master --quick

# Measure a specific phase (includes build overhead)
python benchmarks/memory.py save master --test-path benchmarks/test_lp_write.py
```

Results are stored in `.benchmarks/memory/` (gitignored). Requires Linux or macOS (memray is not available on Windows).

> **Note:** Small tests (~5 MiB) are near the import-overhead floor and may show noise of ~1 MiB between runs. Focus on larger tests for meaningful memory comparisons. Do not combine `--memray` with timing benchmarks — memray adds ~2x overhead that invalidates timing results.

## Models

| Model | Description | Sizes |
|-------|-------------|-------|
| `basic` | Dense N*N model, 2*N^2 vars/cons | 10 — 1600 |
| `knapsack` | N binary variables, 1 constraint | 100 — 1M |
| `expression_arithmetic` | Broadcasting, scaling, summation across dims | 10 — 1000 |
| `sparse_network` | Ring network with mismatched bus/line coords | 10 — 1000 |
| `pypsa_scigrid` | Real power system (requires `pypsa`) | 10 — 200 snapshots |

## Phases

| Phase | File | What it measures |
|-------|------|------------------|
| Build | `test_build.py` | Model construction (add_variables, add_constraints, add_objective) |
| LP write | `test_lp_write.py` | Writing the model to an LP file |
| Matrices | `test_matrices.py` | Generating sparse matrices (A, b, c, bounds) from the model |

## Adding a new model

1. Create `benchmarks/models/my_model.py` with a `build_my_model(n)` function and a `SIZES` list
2. Add parametrized tests in the relevant `test_*.py` files
3. Add a quick threshold in `conftest.py`
