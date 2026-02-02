# Benchmarks

Modular benchmark framework for linopy. All commands use [`just`](https://github.com/casey/just).

## Quick Start

```bash
# Install just (macOS)
brew install just

# List available models and phases
just bench-list

# Quick smoke test
just bench-quick

# Full benchmark suite
just bench label="my-branch"

# Single phase
just bench-build label="my-branch"
just bench-memory label="my-branch"
just bench-write label="my-branch"

# Single model + phase
just bench-model basic build label="my-branch"

# Compare two runs
just bench-compare benchmarks/results/old_basic_build.json benchmarks/results/new_basic_build.json
```

## Models

| Name | Description |
|------|-------------|
| `basic` | 2×N² vars/cons — simple dense model |
| `knapsack` | N binary variables — integer programming |
| `pypsa_scigrid` | Real power system from PyPSA SciGrid-DE |
| `sparse` | Sparse ring network — exercises alignment |
| `large_expr` | Many-term expressions — stress test |

## Phases

| Name | Description |
|------|-------------|
| `build` | Model construction speed (time) |
| `memory` | Peak memory via tracemalloc |
| `lp_write` | LP file writing speed |

## Output

Results are saved as JSON files in `benchmarks/results/` (gitignored).
Pattern: `{label}_{model}_{phase}.json`
